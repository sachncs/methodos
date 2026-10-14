"""Tests for durable evolution job records."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from methodos.jobs import (
    EvolutionJob,
    EvolutionJobWorker,
    JobIdempotencyConflictError,
    JobNotFoundError,
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

    async def test_invalid_inputs_and_missing_update_fail(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        with pytest.raises(ValueError):
            await store.create_or_get(graph_id="", k_rounds=1, idempotency_key=None)
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

    async def test_worker_records_failure_and_cancellation(
        self,
        store: SQLiteEvolutionJobStore,
    ) -> None:
        failed = await store.create_or_get(graph_id="graph-1", k_rounds=2, idempotency_key=None)

        class FailingExecutor:
            async def execute(self, claimed: EvolutionJob) -> None:
                del claimed
                raise RuntimeError("boom")

        assert await EvolutionJobWorker(store, FailingExecutor()).run_once()
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
