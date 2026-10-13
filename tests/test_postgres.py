"""Integration tests for the optional Postgres repository."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

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
        trajectories = [trajectory async for trajectory in repo.read_trajectories(graph_id, "train")]
        assert len(trajectories) == 1
        assert trajectories[0].task.query == "question"
    finally:
        if repo._pool is not None:
            async with repo._pool.acquire() as conn:
                await conn.execute("DELETE FROM methodos_graphs WHERE id = $1", graph_id)
        await repo.close()
