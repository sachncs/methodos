"""Tests for durable evolution job records."""

from __future__ import annotations

from pathlib import Path

import pytest

from methodos.jobs import (
    JobIdempotencyConflictError,
    JobNotFoundError,
    SQLiteEvolutionJobStore,
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
