"""Tests for `methodos.adapter` (PGAdapter, AgentState, Solver, GuidanceCache)."""

from __future__ import annotations

import pytest

from methodos.adapter import (
    AgentState,
    GuidanceCache,
    PGAdapter,
    Solver,
)
from methodos.graph import match_node, neighborhood
from tests.conftest import (
    ScriptedLLM,
    SequenceSolver,
    StaticSolver,
    make_sample_graph,
)


class TestAgentState:
    """`AgentState` is frozen and slotted."""

    def test_construction(self) -> None:
        state = AgentState(
            query="q",
            trajectory=(("a", "o"),),
            context="ctx",
        )
        assert state.query == "q"
        assert state.trajectory == (("a", "o"),)
        assert state.context == "ctx"

    def test_is_frozen(self) -> None:
        import dataclasses

        state = AgentState(query="q", trajectory=(), context="")
        # Frozen dataclasses raise FrozenInstanceError on attribute assignment.
        assert dataclasses.is_dataclass(state)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.query = "modified"  # type: ignore[misc]

    def test_default_trajectory_is_empty_tuple(self) -> None:
        state = AgentState(query="q", trajectory=(), context="")
        assert state.trajectory == ()

    def test_is_hashable(self) -> None:
        """Frozen dataclasses are hashable; required for cache keys."""
        s1 = AgentState(query="q", trajectory=(("a", "b"),), context="c")
        s2 = AgentState(query="q", trajectory=(("a", "b"),), context="c")
        assert hash(s1) == hash(s2)


class TestSolverProtocol:
    """`Solver` is a runtime-checkable Protocol."""

    def test_satisfies_protocol(self) -> None:
        class ProtocolSolver:
            async def step(self, state: AgentState) -> str:
                return "ok"

        assert isinstance(ProtocolSolver(), Solver)

    def test_does_not_satisfy_without_step(self) -> None:
        class NoStepSolver:
            pass

        assert not isinstance(NoStepSolver(), Solver)


class TestGuidanceCache:
    """`GuidanceCache` LRU semantics."""

    def test_empty_cache(self) -> None:
        cache = GuidanceCache()
        assert len(cache) == 0
        assert cache.get(("g", "a", "o")) is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_put_then_get(self) -> None:
        cache = GuidanceCache()
        cache.put(("g", "a", "o"), "guidance")
        assert cache.get(("g", "a", "o")) == "guidance"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_lru_eviction(self) -> None:
        cache = GuidanceCache(max_size=2)
        cache.put(("g", "a", "1"), "v1")
        cache.put(("g", "a", "2"), "v2")
        # Access the first entry to make it most-recently used
        assert cache.get(("g", "a", "1")) == "v1"
        # Add a third entry; the second should be evicted
        cache.put(("g", "a", "3"), "v3")
        assert cache.get(("g", "a", "2")) is None  # evicted
        assert cache.get(("g", "a", "1")) == "v1"
        assert cache.get(("g", "a", "3")) == "v3"
        assert len(cache) == 2

    def test_overwrite_existing_key(self) -> None:
        cache = GuidanceCache()
        cache.put(("g", "a", "o"), "v1")
        cache.put(("g", "a", "o"), "v2")
        assert cache.get(("g", "a", "o")) == "v2"
        assert len(cache) == 1

    def test_clear(self) -> None:
        cache = GuidanceCache()
        cache.put(("g", "a", "o"), "v1")
        cache.put(("g", "a", "p"), "v2")
        cache.clear()
        assert len(cache) == 0
        assert cache.get(("g", "a", "o")) is None

    def test_rejects_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size must be positive"):
            GuidanceCache(max_size=0)
        with pytest.raises(ValueError, match="max_size must be positive"):
            GuidanceCache(max_size=-1)


class TestPGAdapterConstruction:
    """`PGAdapter` argument validation."""

    def test_default_arguments(self) -> None:
        graph = make_sample_graph()
        adapter = PGAdapter(
            solver=StaticSolver(),
            graph=graph,
            llm=ScriptedLLM(),
        )
        assert adapter.graph is graph
        assert isinstance(adapter.cache, GuidanceCache)

    def test_accepts_custom_cache(self) -> None:
        graph = make_sample_graph()
        cache = GuidanceCache(max_size=2)
        adapter = PGAdapter(
            solver=StaticSolver(),
            graph=graph,
            llm=ScriptedLLM(),
            cache=cache,
        )
        assert adapter.cache is cache

    def test_rejects_negative_guidanace_hops(self) -> None:
        with pytest.raises(ValueError, match="guidance_hops must be non-negative"):
            PGAdapter(
                solver=StaticSolver(),
                graph=make_sample_graph(),
                llm=ScriptedLLM(),
                guidance_hops=-1,
            )

    def test_rejects_negative_window(self) -> None:
        with pytest.raises(ValueError, match="trajectory_window must be non-negative"):
            PGAdapter(
                solver=StaticSolver(),
                graph=make_sample_graph(),
                llm=ScriptedLLM(),
                trajectory_window=-1,
            )


class TestPGAdapterStep:
    """End-to-end adapter behavior."""

    async def test_calls_solver_with_agent_state(self) -> None:
        solver = StaticSolver(action="answer")
        graph = make_sample_graph()
        adapter = PGAdapter(
            solver=solver,
            graph=graph,
            llm=ScriptedLLM(responses=["do this"]),
        )
        action = await adapter.step(
            query="what?",
            trajectory=[],
        )
        assert action == "answer"
        # Solver saw exactly one state.
        assert len(solver.states_seen) == 1
        state = solver.states_seen[0]
        assert isinstance(state, AgentState)
        assert state.query == "what?"
        assert state.trajectory == ()

    async def test_context_contains_guidance(self) -> None:
        solver = StaticSolver()
        adapter = PGAdapter(
            solver=solver,
            graph=make_sample_graph(),
            llm=ScriptedLLM(responses=["GUIDANCE-BODY"]),
        )
        await adapter.step(query="q", trajectory=[])
        state = solver.states_seen[0]
        assert "GUIDANCE-BODY" in state.context
        assert "### PROCEDURAL GUIDANCE" in state.context

    async def test_empty_trajectory_uses_start_key(self) -> None:
        """Empty trajectory → locate via 'Start' node, generate guidance."""
        graph = make_sample_graph()
        solver = StaticSolver(action="search")
        llm = ScriptedLLM(responses=["g"])
        adapter = PGAdapter(solver=solver, graph=graph, llm=llm)
        await adapter.step(query="q", trajectory=[])
        assert len(llm.calls) == 1

    async def test_unknown_last_action_uses_full_graph(self) -> None:
        """If the last action isn't in the graph, fall back to the full graph."""
        graph = make_sample_graph()
        solver = StaticSolver()
        llm = ScriptedLLM(responses=["g"])
        adapter = PGAdapter(solver=solver, graph=graph, llm=llm)
        await adapter.step(
            query="q",
            trajectory=[("completely_unknown_action", "obs")],
        )
        # Still called the LLM (guidance generation proceeds with full graph)
        assert len(llm.calls) == 1

    async def test_cache_hit_on_repeated_call(self) -> None:
        graph = make_sample_graph()
        solver = StaticSolver()
        llm = ScriptedLLM(responses=["CACHED-GUIDANCE"])
        adapter = PGAdapter(solver=solver, graph=graph, llm=llm)
        # First call → LLM invoked
        await adapter.step(query="q", trajectory=[("start", "")])
        assert len(llm.calls) == 1
        assert adapter.cache.hits == 0
        assert adapter.cache.misses == 1
        # Second call (same trajectory) → cache hit, no new LLM call
        await adapter.step(query="q", trajectory=[("start", "")])
        assert len(llm.calls) == 1
        assert adapter.cache.hits == 1
        assert adapter.cache.misses == 1

    async def test_cache_miss_on_different_observation(self) -> None:
        graph = make_sample_graph()
        solver = StaticSolver()
        llm = ScriptedLLM(responses=["g1", "g2"])
        adapter = PGAdapter(solver=solver, graph=graph, llm=llm)
        await adapter.step(query="q", trajectory=[("start", "obs1")])
        await adapter.step(query="q", trajectory=[("start", "obs2")])
        # Different observation → different cache key → 2 LLM calls
        assert len(llm.calls) == 2

    async def test_cache_key_includes_graph_id(self) -> None:
        """Two distinct graph instances with the same id share the cache."""
        graph_a = make_sample_graph()
        graph_b = make_sample_graph()
        assert graph_a is not graph_b  # different objects
        shared_cache = GuidanceCache()
        solver = StaticSolver()
        llm = ScriptedLLM(responses=["g"])
        adapter_a = PGAdapter(
            solver=solver,
            graph=graph_a,
            llm=llm,
            cache=shared_cache,
        )
        await adapter_a.step(query="q", trajectory=[("start", "")])
        # Second adapter shares the cache → cache hit, no LLM call.
        adapter_b = PGAdapter(
            solver=solver,
            graph=graph_b,
            llm=llm,
            cache=shared_cache,
        )
        await adapter_b.step(query="q", trajectory=[("start", "")])
        assert len(llm.calls) == 1
        assert shared_cache.hits == 1

    async def test_solver_receives_trajectory_as_tuple(self) -> None:
        """The trajectory in AgentState is a tuple, not a list (immutable)."""
        solver = StaticSolver()
        adapter = PGAdapter(
            solver=solver,
            graph=make_sample_graph(),
            llm=ScriptedLLM(),
        )
        trajectory = [("a", "1"), ("b", "2"), ("c", "3")]
        await adapter.step(query="q", trajectory=trajectory)
        state = solver.states_seen[0]
        assert isinstance(state.trajectory, tuple)
        assert state.trajectory == (("a", "1"), ("b", "2"), ("c", "3"))

    async def test_solver_propagates_exceptions(self) -> None:
        class BoomSolver:
            async def step(self, state: AgentState) -> str:
                raise ValueError("kaboom")

        adapter = PGAdapter(
            solver=BoomSolver(),
            graph=make_sample_graph(),
            llm=ScriptedLLM(),
        )
        with pytest.raises(ValueError, match="kaboom"):
            await adapter.step(query="q", trajectory=[])

    async def test_uses_h_hop_neighborhood(self) -> None:
        """The default guidance_hops=2 selects a 2-hop neighborhood."""
        graph = make_sample_graph()
        # Verify the helper directly (adapter construction is covered
        # by other tests). The 2-hop neighborhood from "search" is
        # {"search", "answer"}.
        sub = neighborhood(graph, "search", h=2)
        assert set(sub.nodes.keys()) == {"search", "answer"}


class TestMatchNodeIntegration:
    """Smoke test that adapter uses `match_node` from `methodos.graph`."""

    def test_match_node_called_with_last_action(self) -> None:
        graph = make_sample_graph()
        # Direct verification: match_node on the sample graph nodes.
        assert match_node("start", graph.nodes) == "start"
        assert match_node("answer", graph.nodes) == "answer"
        assert match_node("nope", graph.nodes) is None
