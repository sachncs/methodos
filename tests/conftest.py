"""Fakes and shared fixtures for `methodos` tests.

This module exposes the standard fakes used across the test suite:
- `FakeLLM`: scripted `LLMClient` returning canned responses in order.
- `SequenceSolver`: `Solver` that returns a different action each call.
- `StaticSolver`: `Solver` that returns a fixed action.
- `RaisingSolver`: `Solver` that raises on every call.
- `InMemoryRepository`: in-process `Repository` (Phase 4).
- `NoOpVectorIndex`: `VectorIndex` that does nothing (Phase 4).
- Standard fixtures: `fake_llm`, `sequence_solver`, `sample_graph`,
  `in_memory_repository`.

Engineering:
- No `_foo()` markers; every helper here is part of the public test API.
- Fakes satisfy their respective Protocols via duck typing; no inheritance.
- Single shared source so future phases can reuse without duplication.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from methodos.adapter import AgentState, GuidanceCache
from methodos.llm import LLMClient
from methodos.repo import Trajectory
from methodos.schema import (
    Attribute,
    Edge,
    Node,
    ProceduralGraph,
    Relation,
)


class FakeLLM(LLMClient):
    """Canned `LLMClient` that returns scripted responses in order.

    Use the `calls` attribute to assert what was sent. The default response
    (when the queue is empty) is the empty string, which keeps the
    adapter cache key deterministic without raising.
    """

    def __init__(self, responses: Sequence[str] = ()) -> None:
        self.responses: list[str] = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "json_schema": json_schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            return ""
        return self.responses.pop(0)


class SequenceSolver:
    """Solver that returns a different action on each call.

    After the supplied actions are exhausted, returns `"FINISH"`.
    Records every `AgentState` it sees in `states_seen`.
    """

    def __init__(self, actions: Sequence[str] = ()) -> None:
        self.actions: list[str] = list(actions)
        self.states_seen: list[AgentState] = []

    async def step(self, state: AgentState) -> str:
        self.states_seen.append(state)
        if not self.actions:
            return "FINISH"
        return self.actions.pop(0)


class StaticSolver:
    """Solver that returns the same fixed action regardless of state."""

    def __init__(self, action: str = "answer") -> None:
        self.action = action
        self.states_seen: list[AgentState] = []

    async def step(self, state: AgentState) -> str:
        self.states_seen.append(state)
        return self.action


class RaisingSolver:
    """Solver that raises a fixed exception on every call."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("solver failed")
        self.call_count = 0

    async def step(self, state: AgentState) -> str:
        self.call_count += 1
        raise self.exc


class InMemoryRepository:
    """In-process Repository implementation for tests.

    Satisfies the `Repository` Protocol so it can be passed where any
    Repository is expected. State lives in plain dicts / lists.
    """

    def __init__(self) -> None:
        self.graphs: dict[str, ProceduralGraph] = {}
        self.trajectories: list[tuple[str, str, Trajectory]] = []
        self.snapshots: list[tuple[str, str, ProceduralGraph]] = []

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        if graph_id not in self.graphs:
            raise FileNotFoundError(f"graph {graph_id!r} not found")
        return self.graphs[graph_id].model_copy(deep=True)

    async def save_graph(self, graph: ProceduralGraph) -> None:
        self.graphs[graph.id] = graph.model_copy(deep=True)

    async def snapshot(self, graph_id: str, tag: str) -> None:
        graph = await self.load_graph(graph_id)
        self.snapshots.append((graph_id, tag, graph))

    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None:
        self.trajectories.append((graph_id, split, trajectory))

    async def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        for gid, sp, traj in self.trajectories:
            if gid == graph_id and sp == split:
                yield traj


class NoOpVectorIndex:
    """No-op VectorIndex that satisfies the duck-typed protocol."""

    def upsert(self, key: str, vector: list[float]) -> None:
        pass

    def query(self, vector: list[float], k: int) -> list[Any]:
        return []


# ----------------------------------------------------------------------------
# Standard fixtures
# ----------------------------------------------------------------------------


def make_sample_graph() -> ProceduralGraph:
    """Construct a small three-node graph used across many tests."""
    attr_search = Attribute(
        condition="need information",
        guidance="call search with focused query",
        pitfalls="do not search with the full question",
    )
    attr_answer = Attribute(
        condition="have enough information",
        guidance="synthesize a concise answer",
        pitfalls="do not repeat observations",
    )
    return ProceduralGraph(
        id="test",
        nodes={
            "start": Node(id="start", description="begin task"),
            "search": Node(id="search", description="search Wikipedia"),
            "answer": Node(id="answer", description="final answer"),
        },
        edges=[
            Edge(
                src="start",
                dst="search",
                relation=Relation.LEADS_TO,
                attribute=attr_search,
            ),
            Edge(
                src="search",
                dst="answer",
                relation=Relation.LEADS_TO,
                attribute=attr_answer,
            ),
        ],
        terminal_ids={"answer"},
        metadata={"purpose": "test"},
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Default FakeLLM with one canned response."""
    return FakeLLM(responses=["guidance-text"])


@pytest.fixture
def sequence_solver() -> SequenceSolver:
    """Solver that yields a fixed sequence of actions."""
    return SequenceSolver(actions=["search", "answer"])


@pytest.fixture
def sample_graph() -> ProceduralGraph:
    """Standard three-node graph for adapter tests."""
    return make_sample_graph()


@pytest.fixture
def in_memory_repository() -> InMemoryRepository:
    """In-memory Repository instance."""
    return InMemoryRepository()


__all__ = [
    "FakeLLM",
    "InMemoryRepository",
    "NoOpVectorIndex",
    "RaisingSolver",
    "SequenceSolver",
    "StaticSolver",
    "make_sample_graph",
]
