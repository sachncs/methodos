"""Integration tests for the optional Postgres repository."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from methodos.jobs import (
    EvolutionJob,
    JobIdempotencyConflictError,
    JobQuotaExceededError,
    PostgresEvolutionJobStore,
)
from methodos.repo import (
    GraphAlreadyExistsError,
    GraphRevisionConflictError,
    PostgresRepository,
    Task,
    Trajectory,
)
from methodos.schema import Node, ProceduralGraph

POSTGRES_DSN = os.environ.get("METHODOS_POSTGRES_TEST_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not POSTGRES_DSN, reason="METHODOS_POSTGRES_TEST_DSN is not set"),
]


def make_graph(graph_id: str) -> ProceduralGraph:
    """Build a minimal graph for an integration round-trip."""
    return ProceduralGraph(
        id=graph_id,
        nodes={"start": Node(id="start"), "answer": Node(id="answer")},
        edges=[],
        terminal_ids={"answer"},
    )


@pytest.mark.asyncio
async def test_postgres_repository_round_trip_and_cas() -> None:
    """Exercise schema setup, CRUD, trajectories, snapshots, and CAS."""
    assert POSTGRES_DSN is not None
    repo = PostgresRepository(dsn=POSTGRES_DSN)
    graph_id = f"integration-{uuid4().hex}"
    graph = make_graph(graph_id)
    try:
        await repo.initialize()
        rate_key = f"rate-{uuid4().hex}"
        allowed, retry_after = await repo.consume_rate_limit(rate_key, 1)
        assert allowed
        assert 1 <= retry_after <= 60
        allowed, retry_after = await repo.consume_rate_limit(rate_key, 1)
        assert not allowed
        assert 1 <= retry_after <= 60
        await repo.create_graph(graph)
        with pytest.raises(GraphAlreadyExistsError):
            await repo.create_graph(graph)

        loaded = await repo.load_graph(graph_id)
        assert loaded.revision == 0

        updated = graph.model_copy(update={"description": "updated"})
        stored = await repo.save_graph_if_revision(updated, expected_revision=0)
        assert stored.revision == 1
        with pytest.raises(GraphRevisionConflictError):
            await repo.save_graph_if_revision(updated, expected_revision=0)

        await repo.snapshot(graph_id, "release-1")
        await repo.append_trajectory(
            graph_id,
            "train",
            Trajectory(task=Task(query="question"), steps=(("answer", "done"),), score=1.0),
        )
        trajectories = [
            trajectory async for trajectory in repo.read_trajectories(graph_id, "train")
        ]
        assert len(trajectories) == 1
        assert trajectories[0].task.query == "question"
    finally:
        if repo._pool is not None:
            async with repo._pool.acquire() as conn:
                await conn.execute("DELETE FROM methodos_graphs WHERE id = $1", graph_id)
                await conn.execute("DELETE FROM methodos_rate_limits WHERE key = $1", rate_key)
        await repo.close()


@pytest.mark.asyncio
async def test_postgres_evolution_job_store_is_durable_and_idempotent() -> None:
    """Exercise shared job state used by horizontally scaled workers."""
    assert POSTGRES_DSN is not None
    store = PostgresEvolutionJobStore(dsn=POSTGRES_DSN)
    key = f"job-key-{uuid4().hex}"
    created = None
    try:
        await store.initialize()
        created = await store.create_or_get(
            graph_id=f"job-graph-{uuid4().hex}",
            k_rounds=2,
            idempotency_key=key,
        )
        repeated = await store.create_or_get(
            graph_id=created.graph_id,
            k_rounds=created.k_rounds,
            idempotency_key=key,
        )
        assert repeated == created
        with pytest.raises(JobIdempotencyConflictError):
            await store.create_or_get(
                graph_id=created.graph_id,
                k_rounds=3,
                idempotency_key=key,
            )
        updated = await store.update(created.job_id, status="running", detail="worker claimed")
        assert (await store.get(created.job_id)) == updated
    finally:
        if created is not None and store._pool is not None:
            async with store._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM methodos_evolution_jobs WHERE job_id = $1",
                    created.job_id,
                )
        await store.close()


@pytest.mark.asyncio
async def test_postgres_evolution_job_quota_is_atomic_under_concurrency() -> None:
    """Concurrent requests cannot bypass a tenant active-job quota."""
    assert POSTGRES_DSN is not None
    store = PostgresEvolutionJobStore(dsn=POSTGRES_DSN)
    graph_id = f"quota-graph-{uuid4().hex}"
    try:
        await store.initialize()
        results = await asyncio.gather(
            *(
                store.create_or_get(
                    graph_id=graph_id,
                    k_rounds=1,
                    idempotency_key=None,
                    tenant_id="quota-tenant",
                    max_active_jobs=1,
                )
                for _ in range(8)
            ),
            return_exceptions=True,
        )
        successful = [result for result in results if isinstance(result, EvolutionJob)]
        rejected = [result for result in results if isinstance(result, JobQuotaExceededError)]
        assert len(successful) == 1
        assert len(rejected) == 7
    finally:
        if store._pool is not None:
            async with store._pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM methodos_evolution_jobs WHERE graph_id = $1", graph_id
                )
        await store.close()
