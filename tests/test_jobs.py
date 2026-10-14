"""Tests for durable evolution job records."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from methodos.jobs import (
    EvolutionJob,
    EvolutionJobWorker,
    JobIdempotencyConflictError,
    JobLeaseLostError,
    JobNotFoundError,
    JobQuotaExceededError,
    SQLiteEvolutionJobStore,
    StoreBackedEvolutionRunner,
)


@pytest.fixture
def store(tmp_path: Path) -> SQLiteEvolutionJobStore:
    """Return a fresh SQLite job store."""
    return SQLiteEvolutionJobStore(tmp_path / "jobs.db")


class TestSQLiteEvolutionJobStore:
    async def test_create_and_update_survive_new_store_instance(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        created = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=3,
            idempotency_key="retry-1",
        )
        assert created.status == "queued"
        assert created.lease_token is None
        counts = await store.counts()
        assert counts.queued == 1
        assert counts.running == counts.failed == 0
        running = await store.update(created.job_id, status="running", detail="started")
        assert running.detail == "started"

        reopened = SQLiteEvolutionJobStore(store.database_path)
        loaded = await reopened.get(created.job_id)
        assert loaded == running

    async def test_same_key_returns_same_job(self, store: SQLiteEvolutionJobStore) -> None:
        first = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=3,
            idempotency_key="retry-1",
        )
        second = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=3,
            idempotency_key="retry-1",
        )
        assert second == first

    async def test_tenant_ownership_survives_restart_and_scopes_keys(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        first = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=3,
            idempotency_key="retry-1",
            tenant_id="alpha",
        )
        second = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=3,
            idempotency_key="retry-1",
            tenant_id="beta",
        )
        assert first.tenant_id == "alpha"
        assert second.tenant_id == "beta"
        assert first.job_id != second.job_id

        reopened = SQLiteEvolutionJobStore(store.database_path)
        assert await reopened.get(first.job_id) == first

    async def test_reusing_key_for_different_request_fails(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        await store.create_or_get(graph_id="graph-1", k_rounds=3, idempotency_key="retry-1")
        with pytest.raises(JobIdempotencyConflictError):
            await store.create_or_get(graph_id="graph-2", k_rounds=3, idempotency_key="retry-1")

    async def test_keyless_requests_are_distinct(self, store: SQLiteEvolutionJobStore) -> None:
        first = await store.create_or_get(graph_id="graph-1", k_rounds=3, idempotency_key=None)
        second = await store.create_or_get(graph_id="graph-1", k_rounds=3, idempotency_key=None)
        assert first.job_id != second.job_id

    async def test_active_job_quota_is_tenant_scoped(self, store: SQLiteEvolutionJobStore) -> None:
        first = await store.create_or_get(
            graph_id="graph-1",
            k_rounds=1,
            idempotency_key=None,
            tenant_id="alpha",
            max_active_jobs=2,
            max_active_jobs_per_graph=1,
        )
        with pytest.raises(JobQuotaExceededError):
            await store.create_or_get(
                graph_id="graph-1",
                k_rounds=1,
                idempotency_key=None,
                tenant_id="alpha",
                max_active_jobs=2,
                max_active_jobs_per_graph=1,
            )
        third = await store.create_or_get(
            graph_id="graph-2",
            k_rounds=1,
            idempotency_key=None,
            tenant_id="alpha",
            max_active_jobs=2,
            max_active_jobs_per_graph=1,
        )
        second = await store.create_or_get(
            graph_id="graph-2",
            k_rounds=1,
            idempotency_key=None,
            tenant_id="beta",
            max_active_jobs=1,
            max_active_jobs_per_graph=1,
        )
        assert first.tenant_id == "alpha"
        assert third.tenant_id == "alpha"
        assert second.tenant_id == "beta"

    async def test_invalid_inputs_and_missing_update_fail(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        with pytest.raises(ValueError):
            await store.create_or_get(graph_id="", k_rounds=1, idempotency_key=None)
        with pytest.raises(ValueError):
            await store.create_or_get(
                graph_id="graph-1",
                k_rounds=1,
                idempotency_key=None,
                max_active_jobs=0,
            )
        with pytest.raises(ValueError):
            await store.create_or_get(
                graph_id="graph-1",
                k_rounds=1,
                idempotency_key=None,
                max_active_jobs_per_graph=0,
            )
        with pytest.raises(ValueError):
            await store.update("missing", status="unknown")
        with pytest.raises(JobNotFoundError):
            await store.update("missing", status="failed")
        with pytest.raises(ValueError):
            await store.get("")
        assert await store.get("missing") is None

    async def test_worker_claims_and_completes_a_job(self, store: SQLiteEvolutionJobStore) -> None:
        job = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)
        executed: list[str] = []

        class Executor:
            async def execute(self, claimed: EvolutionJob) -> None:
                executed.append(claimed.job_id)

        worker = EvolutionJobWorker(store, Executor())
        assert await worker.run_once()
        assert not await worker.run_once()
        assert executed == [job.job_id]
        completed = await store.get(job.job_id)
        assert completed is not None
        assert completed.status == "succeeded"

    async def test_worker_renews_long_running_lease(self, store: SQLiteEvolutionJobStore) -> None:
        job = await store.create_or_get(graph_id="graph-lease", k_rounds=2, idempotency_key=None)

        class SlowExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                assert claimed.lease_token is not None
                await asyncio.sleep(0.8)

        worker = EvolutionJobWorker(store, SlowExecutor(), lease_seconds=0.5)
        assert await worker.run_once()
        completed = await store.get(job.job_id)
        assert completed is not None
        assert completed.status == "succeeded"

    async def test_renew_lease_rejects_stale_owner(self, store: SQLiteEvolutionJobStore) -> None:
        job = await store.create_or_get(graph_id="graph-stale", k_rounds=1, idempotency_key=None)
        claimed = await store.claim_next(lease_seconds=0.05)
        assert claimed is not None
        assert claimed.lease_token is not None
        await asyncio.sleep(0.06)
        await store.recover_expired(max_attempts=3)
        with pytest.raises(JobLeaseLostError):
            await store.renew_lease(job.job_id, claimed.lease_token, lease_seconds=1)

    async def test_lease_operations_validate_ownership_inputs(
        self, store: SQLiteEvolutionJobStore
    ) -> None:
        job = await store.create_or_get(graph_id="graph-inputs", k_rounds=1, idempotency_key=None)
        claimed = await store.claim_next(lease_seconds=1)
        assert claimed is not None
        assert claimed.lease_token is not None
        with pytest.raises(ValueError):
            await store.renew_lease(job.job_id, "", lease_seconds=1)
        with pytest.raises(ValueError):
            await store.renew_lease(job.job_id, claimed.lease_token, lease_seconds=0)
        await store.update(job.job_id, status="queued", detail="requeued")
        with pytest.raises(JobLeaseLostError):
            await store.update(
                job.job_id,
                status="succeeded",
                lease_token=claimed.lease_token,
            )

    async def test_worker_executes_without_a_claimed_lease(
        self, store: SQLiteEvolutionJobStore
    ) -> None:
        job = await store.create_or_get(graph_id="graph-direct", k_rounds=1, idempotency_key=None)
        executed: list[str] = []

        class Executor:
            async def execute(self, claimed: EvolutionJob) -> None:
                executed.append(claimed.job_id)

        worker = EvolutionJobWorker(store, Executor())
        await worker._execute_with_lease(job)
        assert executed == [job.job_id]

    async def test_worker_refreshes_heartbeat_during_nonleased_execution(
        self, store: SQLiteEvolutionJobStore
    ) -> None:
        job = await store.create_or_get(
            graph_id="graph-heartbeat", k_rounds=1, idempotency_key=None
        )

        class SlowExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                await asyncio.sleep(0.05)

        worker = EvolutionJobWorker(store, SlowExecutor())
        execution = asyncio.create_task(worker._execute_with_lease(job))
        await asyncio.sleep(0.01)
        heartbeat = worker.health().last_heartbeat
        await execution
        assert heartbeat is not None
        assert time.time() - heartbeat < 1.0

    async def test_worker_stops_when_lease_is_lost(self, store: SQLiteEvolutionJobStore) -> None:
        await store.create_or_get(graph_id="graph-lost", k_rounds=1, idempotency_key=None)

        class LosingStore(SQLiteEvolutionJobStore):
            async def renew_lease(
                self, job_id: str, lease_token: str, *, lease_seconds: float
            ) -> EvolutionJob:
                del job_id, lease_token, lease_seconds
                raise JobLeaseLostError("simulated lease loss")

        losing_store = LosingStore(store.database_path)

        class SlowExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                await asyncio.sleep(1)

        worker = EvolutionJobWorker(losing_store, SlowExecutor(), lease_seconds=0.03)
        with pytest.raises(JobLeaseLostError):
            await worker.run_once()

    async def test_concurrent_claimers_never_claim_the_same_job(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        jobs = [
            await store.create_or_get(graph_id=f"graph-{index}", k_rounds=1, idempotency_key=None)
            for index in range(8)
        ]
        claimed = await asyncio.gather(
            *(store.claim_next(lease_seconds=30) for _ in range(len(jobs)))
        )
        claimed_ids = {job.job_id for job in claimed if job is not None}
        assert claimed_ids == {job.job_id for job in jobs}

    async def test_worker_records_failure_and_cancellation(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        failed = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)

        class FailingExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                raise RuntimeError("boom")

        assert await EvolutionJobWorker(store, FailingExecutor(), max_attempts=1).run_once()
        failed_record = await store.get(failed.job_id)
        assert failed_record is not None
        assert failed_record.status == "failed"

        cancelled = await store.create_or_get(graph_id="graph-2", k_rounds=2, idempotency_key=None)

        class CancelledExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await EvolutionJobWorker(store, CancelledExecutor()).run_once()
        cancelled_record = await store.get(cancelled.job_id)
        assert cancelled_record is not None
        assert cancelled_record.status == "cancelled"

    async def test_worker_retries_before_failing(self, store: SQLiteEvolutionJobStore) -> None:
        job = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)
        attempts = 0

        class RetryExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                nonlocal attempts
                assert claimed.job_id == job.job_id
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient")

        worker = EvolutionJobWorker(store, RetryExecutor(), max_attempts=2)
        assert await worker.run_once()
        queued = await store.get(job.job_id)
        assert queued is not None
        assert queued.status == "queued"
        assert await worker.run_once()
        completed = await store.get(job.job_id)
        assert completed is not None
        assert completed.status == "succeeded"

    async def test_worker_loop_stops_after_signal(self, store: SQLiteEvolutionJobStore) -> None:
        stop_event = asyncio.Event()
        await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)

        class Executor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                stop_event.set()

        worker = EvolutionJobWorker(store, Executor())
        await worker.run_forever(stop_event, poll_interval=0.01)
        health = worker.health()
        assert not health.running
        assert health.last_heartbeat is not None
        with pytest.raises(ValueError):
            await EvolutionJobWorker(store, Executor()).run_forever(
                asyncio.Event(), poll_interval=0
            )

        idle_stop = asyncio.Event()

        async def stop_idle_worker() -> None:
            await asyncio.sleep(0.01)
            idle_stop.set()

        stopper = asyncio.create_task(stop_idle_worker())
        await EvolutionJobWorker(store, Executor()).run_forever(idle_stop, poll_interval=0.001)
        await stopper

    async def test_worker_survives_one_storage_failure(
        self,
        store: SQLiteEvolutionJobStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stop_event = asyncio.Event()
        await store.create_or_get(graph_id="graph-1", k_rounds=1, idempotency_key=None)
        original_claim = store.claim_next
        calls = 0

        async def flaky_claim(*, lease_seconds: float = 300.0) -> EvolutionJob | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("temporary database outage")
            return await original_claim(lease_seconds=lease_seconds)

        monkeypatch.setattr(store, "claim_next", flaky_claim)

        class Executor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                stop_event.set()

        worker = EvolutionJobWorker(store, Executor())
        await worker.run_forever(stop_event, poll_interval=0.001)
        health = worker.health()
        assert calls == 2
        assert health.consecutive_errors == 0
        assert health.last_error is None

    async def test_expired_lease_is_recovered(self, store: SQLiteEvolutionJobStore) -> None:
        job = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)
        claimed = await store.claim_next(lease_seconds=0.01)
        assert claimed is not None
        await asyncio.sleep(0.02)
        assert await store.recover_expired(max_attempts=2) == 1
        recovered = await store.get(job.job_id)
        assert recovered is not None
        assert recovered.status == "queued"

    async def test_expired_lease_reaches_failure_after_attempt_limit(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        job = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)
        assert await store.claim_next(lease_seconds=0.01) is not None
        await asyncio.sleep(0.02)
        assert await store.recover_expired(max_attempts=1) == 1
        failed = await store.get(job.job_id)
        assert failed is not None
        assert failed.status == "failed"

    async def test_store_migrates_legacy_job_schema(self, tmp_path: Path) -> None:
        database_path = tmp_path / "legacy.db"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "CREATE TABLE evolution_jobs (job_id TEXT PRIMARY KEY, graph_id TEXT NOT NULL, "
                "k_rounds INTEGER NOT NULL, status TEXT NOT NULL, detail TEXT, "
                "idempotency_key TEXT UNIQUE, request_fingerprint TEXT NOT NULL, "
                "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
            )
        migrated = SQLiteEvolutionJobStore(database_path)
        job = await migrated.create_or_get(graph_id="graph-1", k_rounds=1, idempotency_key=None)
        assert job.attempt_count == 0

    async def test_invalid_worker_configuration_and_lease_inputs(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        class Executor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed

        with pytest.raises(ValueError):
            EvolutionJobWorker(store, Executor(), max_attempts=0)
        with pytest.raises(ValueError):
            EvolutionJobWorker(store, Executor(), lease_seconds=0)
        with pytest.raises(ValueError):
            await store.claim_next(lease_seconds=0)
        with pytest.raises(ValueError):
            await store.recover_expired(max_attempts=0)
        with pytest.raises(ValueError):
            StoreBackedEvolutionRunner(store, max_active_jobs=0)
        with pytest.raises(ValueError):
            StoreBackedEvolutionRunner(store, max_active_jobs_per_graph=0)

    async def test_store_backed_runner_round_trip(self, store: SQLiteEvolutionJobStore) -> None:
        runner = StoreBackedEvolutionRunner(store)
        job_id = await runner.start(graph_id="graph-1", k_rounds=2, request_id="request-1")
        same_job_id = await runner.start_idempotent(
            graph_id="graph-2",
            k_rounds=2,
            request_id="request-2",
            idempotency_key="retry-2",
        )
        assert job_id != same_job_id
        status = await runner.status(job_id)
        assert status is not None
        assert status.status == "queued"
        assert await runner.status("missing") is None
