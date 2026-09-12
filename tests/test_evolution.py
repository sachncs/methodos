"""Tests for `methodos.evolution` (Algorithm 1, EvolutionEngine, helpers)."""

from __future__ import annotations

import json

import pytest

import methodos.evolution as ev_mod
from methodos.adapter import Solver
from methodos.evolution import (
    REFINER_SYSTEM_PROMPT,
    TERMINATE_FAILURE,
    TERMINATE_SUCCESS,
    EvolutionEngine,
    RejectionMemory,
    RolloutResult,
    execute_action,
    mean_score,
    propose_edits,
    run_rollout,
    score,
    tail_concat,
    validate_candidate,
)
from methodos.llm import LLMClient
from methodos.repo import Task, Trajectory
from methodos.schema import (
    Attribute,
    Edge,
    EditAddEdge,
    EditAddNode,
    EditDeleteNode,
    Node,
    ProceduralGraph,
    Relation,
)
from tests.conftest import (
    FakeLLM,
    InMemoryRepository,
    SequenceSolver,
    StaticSolver,
    make_sample_graph,
)

# ----------------------------------------------------------------------------
# Helpers: score, mean_score
# ----------------------------------------------------------------------------


class TestScoreHelpers:
    def test_score_success_is_one(self) -> None:
        traj = Trajectory(task=Task(query="q"), steps=(), score=0.5)
        assert score(RolloutResult(trajectory=traj, success=True)) == 1.0

    def test_score_failure_is_zero(self) -> None:
        traj = Trajectory(task=Task(query="q"), steps=(), score=0.5)
        assert score(RolloutResult(trajectory=traj, success=False)) == 0.0

    def test_mean_score_empty(self) -> None:
        assert mean_score([]) == 0.0

    def test_mean_score_mixed(self) -> None:
        traj = Trajectory(task=Task(query="q"), steps=(), score=0.0)
        results = [
            RolloutResult(trajectory=traj, success=True),
            RolloutResult(trajectory=traj, success=False),
            RolloutResult(trajectory=traj, success=True),
        ]
        assert mean_score(results) == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------
# tail_concat
# ----------------------------------------------------------------------------


class TestTailConcat:
    def test_empty_traces(self) -> None:
        assert tail_concat([], max_tokens=100) == ""

    def test_under_budget_returns_all(self) -> None:
        t = Trajectory(task=Task(query="q"), steps=(("a", "b"),), score=1.0)
        text = tail_concat([t], max_tokens=1000)
        assert "Task: q" in text
        assert "A: a" in text
        assert "O: b" in text

    def test_over_budget_truncates_from_beginning(self) -> None:
        # Build a long trajectory.
        steps = tuple((f"a{i}", f"o{i}") for i in range(1000))
        t = Trajectory(task=Task(query="q"), steps=steps, score=1.0)
        text = tail_concat([t], max_tokens=10)  # char_budget = 40
        assert "Task: q" not in text  # truncated from beginning
        # Last step should be preserved
        assert "a999" in text

    def test_zero_max_tokens_returns_empty(self) -> None:
        t = Trajectory(task=Task(query="q"), steps=(("a", "b"),), score=1.0)
        assert tail_concat([t], max_tokens=0) == ""


# ----------------------------------------------------------------------------
# execute_action
# ----------------------------------------------------------------------------


class TestExecuteAction:
    async def test_returns_empty_string(self) -> None:
        assert await execute_action("search") == ""


# ----------------------------------------------------------------------------
# run_rollout
# ----------------------------------------------------------------------------


class TestRunRollout:
    async def test_max_steps_zero_does_nothing(self) -> None:
        solver = StaticSolver(action="FINISH")
        llm = FakeLLM()
        task = Task(query="q")
        result = await run_rollout(
            graph=make_sample_graph(),
            solver=solver,
            llm=llm,
            task=task,
            max_steps=0,
        )
        assert result.success is False
        assert result.trajectory.score == 0.0
        assert result.trajectory.steps == ()

    async def test_doom_loop_aborts(self) -> None:
        """Same action twice in a row → rollout aborts."""
        # Solver returns "search" twice — second triggers doom-loop.
        solver = SequenceSolver(actions=["search", "search", "FINISH"])
        llm = FakeLLM()
        result = await run_rollout(
            graph=make_sample_graph(),
            solver=solver,
            llm=llm,
            task=Task(query="q"),
            max_steps=10,
        )
        assert result.success is False

    async def test_terminate_success_marker(self) -> None:
        # execute_action returns "" by default; we need a custom
        # setup to inject TERMINATE_SUCCESS. Use a custom solver that
        # returns "FINISH" which we'll patch.
        class SuccessSolver:
            async def step(self, state: object) -> str:
                return "search"

        # Monkeypatch execute_action via patching the module
        # alias already at top

        original_executor = ev_mod.execute_action

        async def recording_executor(action: str) -> str:
            return TERMINATE_SUCCESS

        ev_mod.execute_action = recording_executor
        try:
            result = await run_rollout(
                graph=make_sample_graph(),
                solver=SuccessSolver(),
                llm=FakeLLM(),
                task=Task(query="q"),
                max_steps=10,
            )
            assert result.success is True
            assert result.trajectory.score == 1.0
        finally:
            ev_mod.execute_action = original_executor

    async def test_terminate_failure_marker(self) -> None:
        class FailureSolver:
            async def step(self, state: object) -> str:
                return "search"

        # alias already at top

        async def recording_executor(action: str) -> str:
            return TERMINATE_FAILURE

        original_executor = ev_mod.execute_action
        ev_mod.execute_action = recording_executor
        try:
            result = await run_rollout(
                graph=make_sample_graph(),
                solver=FailureSolver(),
                llm=FakeLLM(),
                task=Task(query="q"),
                max_steps=10,
            )
            assert result.success is False
            assert result.trajectory.score == 0.0
        finally:
            ev_mod.execute_action = original_executor


# ----------------------------------------------------------------------------
# propose_edits
# ----------------------------------------------------------------------------


class TestProposeEdits:
    async def test_parses_valid_json(self) -> None:
        edits_json = json.dumps(
            [
                {"kind": "add_node", "node": {"id": "verify", "description": "verify answer"}},
                {
                    "kind": "add_edge",
                    "edge": {
                        "src": "answer",
                        "dst": "verify",
                        "relation": "leads_to",
                        "attribute": {
                            "condition": "produced",
                            "guidance": "check",
                            "pitfalls": "skip",
                        },
                    },
                },
            ]
        )
        llm = FakeLLM(responses=[edits_json])
        graph = make_sample_graph()
        result = await propose_edits(
            llm=llm,
            graph=graph,
            traces=[],
            rejected=[],
        )
        assert len(result) == 2
        assert isinstance(result[0], EditAddNode)
        assert result[0].node.id == "verify"

    async def test_strips_markdown_fences(self) -> None:
        fenced = '```json\n[{"kind": "add_node", "node": {"id": "v"}}]\n```'
        result = await propose_edits(
            llm=FakeLLM(responses=[fenced]),
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert len(result) == 1

    async def test_invalid_json_returns_empty(self) -> None:
        llm = FakeLLM(responses=["not json at all"])
        result = await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert result == []

    async def test_non_list_returns_empty(self) -> None:
        llm = FakeLLM(responses=['{"not": "a list"}'])
        result = await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert result == []

    async def test_malformed_edit_dropped_with_warning(self) -> None:
        """A non-conforming item is dropped; the rest parse."""
        llm = FakeLLM(
            responses=[
                json.dumps(
                    [
                        {"kind": "add_node", "node": {"id": "ok"}},
                        {"kind": "add_node", "node": {"id": ".starts_with_dot"}},
                    ]
                )
            ]
        )
        result = await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert len(result) == 1
        assert result[0] is not None
        assert result[0].node.id == "ok"

    async def test_unknown_kind_dropped(self) -> None:
        llm = FakeLLM(
            responses=[
                json.dumps(
                    [
                        {"kind": "add_node", "node": {"id": "ok"}},
                        {"kind": "make_coffee", "intensity": "high"},
                    ]
                )
            ]
        )
        result = await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert len(result) == 1

    async def test_prompt_includes_graph_and_traces(self) -> None:
        llm = FakeLLM(responses=["[]"])
        t = Trajectory(task=Task(query="Q"), steps=(("a", "b"),), score=1.0)
        await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[t],
            rejected=[],
        )
        user_prompt = llm.calls[0]["user"]
        assert "Q" in user_prompt
        assert "A: a" in user_prompt
        assert "O: b" in user_prompt
        assert "test" in user_prompt  # graph id

    async def test_prompt_includes_rejection_history(self) -> None:
        llm = FakeLLM(responses=["[]"])
        rejection_edit = EditAddNode(node=Node(id="rej"))
        await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[(rejection_edit, 0.3)],
        )
        user_prompt = llm.calls[0]["user"]
        assert "REJECTED" in user_prompt
        assert "rej" in user_prompt

    async def test_system_prompt_is_set(self) -> None:
        llm = FakeLLM(responses=["[]"])
        await propose_edits(
            llm=llm,
            graph=make_sample_graph(),
            traces=[],
            rejected=[],
        )
        assert llm.calls[0]["system"] == REFINER_SYSTEM_PROMPT


# ----------------------------------------------------------------------------
# validate_candidate
# ----------------------------------------------------------------------------


class TestValidateCandidate:
    def test_returns_new_graph_on_success(self) -> None:
        graph = make_sample_graph()
        attr = Attribute(condition="verified", guidance="loop back", pitfalls="skip")
        # Add a new node with an edge to the terminal — every node can
        # still reach the terminal, so validation passes.
        candidate = validate_candidate(
            graph,
            [
                EditAddNode(node=Node(id="verify", description="verify answer")),
                EditAddEdge(
                    edge=Edge(
                        src="verify",
                        dst="answer",
                        relation=Relation.LEADS_TO,
                        attribute=attr,
                    )
                ),
            ],
        )
        assert candidate is not None
        assert "verify" in candidate.nodes
        # Original unchanged
        assert "verify" not in graph.nodes

    def test_returns_none_on_malformed_edit(self) -> None:
        graph = make_sample_graph()
        # Add a node that already exists → raises in apply_edits
        result = validate_candidate(
            graph,
            [EditAddNode(node=Node(id="search"))],  # duplicate
        )
        assert result is None

    def test_returns_none_on_introducing_cycle_when_disallowed(self) -> None:
        # Start → search → answer → start (cycle, no terminal reachable)
        attr = Attribute(condition="c", guidance="g", pitfalls="p")
        graph = ProceduralGraph(
            id="g",
            nodes={
                "start": Node(id="start"),
                "search": Node(id="search"),
                "answer": Node(id="answer"),
            },
            edges=[
                Edge(src="start", dst="search", relation=Relation.LEADS_TO, attribute=attr),
                Edge(src="search", dst="answer", relation=Relation.LEADS_TO, attribute=attr),
            ],
            terminal_ids={"answer"},
        )
        # Adding an answer→start edge creates a cycle AND breaks terminal reachability.
        cycle_edit = EditAddEdge(
            edge=Edge(
                src="answer",
                dst="start",
                relation=Relation.LEADS_TO,
                attribute=attr,
            )
        )
        candidate = validate_candidate(graph, [cycle_edit], allow_cycles=False)
        assert candidate is None

    def test_allows_cycles_when_enabled(self) -> None:
        attr = Attribute(condition="c", guidance="g", pitfalls="p")
        graph = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[
                Edge(src="a", dst="b", relation=Relation.LEADS_TO, attribute=attr),
                Edge(src="b", dst="a", relation=Relation.LEADS_TO, attribute=attr),
            ],
            terminal_ids={"a", "b"},
        )
        candidate = validate_candidate(
            graph,
            [EditDeleteNode(node_id="a")],
            allow_cycles=True,
        )
        assert candidate is not None  # node a deleted, still valid


# ----------------------------------------------------------------------------
# RejectionMemory
# ----------------------------------------------------------------------------


class TestRejectionMemory:
    def test_construction_rejects_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size must be positive"):
            RejectionMemory(max_size=0)
        with pytest.raises(ValueError, match="max_size must be positive"):
            RejectionMemory(max_size=-1)

    def test_empty_add_does_nothing(self) -> None:
        mem = RejectionMemory(max_size=10)
        mem.add([], 0.5)
        assert len(mem) == 0

    def test_records_marker(self) -> None:
        mem = RejectionMemory(max_size=10)
        edit = EditAddNode(node=Node(id="x"))
        mem.add([edit], 0.3)
        snap = mem.snapshot()
        assert len(snap) == 1
        assert snap[0][0] is not None
        assert snap[0][0].node.id == "x"
        assert snap[0][1] == 0.3

    def test_evicts_oldest_at_capacity(self) -> None:
        mem = RejectionMemory(max_size=2)
        e1 = EditAddNode(node=Node(id="a"))
        e2 = EditAddNode(node=Node(id="b"))
        e3 = EditAddNode(node=Node(id="c"))
        mem.add([e1], 0.1)
        mem.add([e2], 0.2)
        mem.add([e3], 0.3)
        snap = mem.snapshot()
        assert len(snap) == 2
        assert snap[0][0] is not None
        assert snap[0][0].node.id == "b"
        assert snap[1][0] is not None
        assert snap[1][0].node.id == "c"

    def test_len(self) -> None:
        mem = RejectionMemory(max_size=10)
        assert len(mem) == 0
        mem.add([EditAddNode(node=Node(id="a"))], 0.0)
        assert len(mem) == 1


# ----------------------------------------------------------------------------
# EvolutionEngine
# ----------------------------------------------------------------------------


class TestEvolutionEngineConstruction:
    def test_rejects_non_positive_k_rounds(self) -> None:
        with pytest.raises(ValueError, match="k_rounds must be positive"):
            EvolutionEngine(
                llm=FakeLLM(),
                repo=InMemoryRepository(),
                train_tasks=[],
                val_tasks=[],
                solver=StaticSolver(),
                k_rounds=0,
            )

    def test_rejects_non_positive_l_max(self) -> None:
        with pytest.raises(ValueError, match="l_max_tokens must be positive"):
            EvolutionEngine(
                llm=FakeLLM(),
                repo=InMemoryRepository(),
                train_tasks=[],
                val_tasks=[],
                solver=StaticSolver(),
                l_max_tokens=0,
            )

    def test_rejects_non_positive_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive"):
            EvolutionEngine(
                llm=FakeLLM(),
                repo=InMemoryRepository(),
                train_tasks=[],
                val_tasks=[],
                solver=StaticSolver(),
                max_steps=0,
            )


class ScriptedLLM(LLMClient):
    """LLM whose response is computed by a function over the call index.

    The callable receives `(call_index, user_prompt)`. Use the prompt
    content to discriminate refiner calls from rollout-guidance calls
    (refiner prompts contain "Propose a JSON array of edits.").
    """

    def __init__(self, fn: object) -> None:
        self.fn = fn
        self.calls: list[str] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: object | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(user)
        return self.fn(len(self.calls) - 1, user)

    def is_refiner_call(self, index: int) -> bool:
        return "Propose a JSON array of edits." in self.calls[index]


class TestEvolutionEngineRun:
    """Algorithm 1 behavior end-to-end with deterministic fakes."""

    async def test_accepts_initial_when_no_edits(self) -> None:
        # Refiner returns no edits → initial graph survives; engine
        # still runs K rounds and returns the same graph.
        llm = ScriptedLLM(lambda i, _: "[]")
        engine = EvolutionEngine(
            llm=llm,
            repo=InMemoryRepository(),
            train_tasks=[Task(query="t1"), Task(query="t2")],
            val_tasks=[Task(query="v1")],
            solver=StaticSolver(action="FINISH"),
            k_rounds=3,
        )
        graph = make_sample_graph()
        result = await engine.run(graph)
        # No edits proposed; same graph returned (deep copy).
        assert result.id == graph.id

    async def test_accepts_valid_candidate_on_improvement(self) -> None:
        # The refiner proposes a valid add_node + add_edge pair; the
        # rollout LLM is fed empty guidance text (no JSON).
        def script(i: int, user: str) -> str:
            if "Propose a JSON array of edits." in user:
                return json.dumps(
                    [
                        {"kind": "add_node", "node": {"id": "verify", "description": "verify"}},
                        {
                            "kind": "add_edge",
                            "edge": {
                                "src": "verify",
                                "dst": "answer",
                                "relation": "leads_to",
                                "attribute": {
                                    "condition": "verified",
                                    "guidance": "loop back",
                                    "pitfalls": "skip",
                                },
                            },
                        },
                    ]
                )
            return ""

        llm = ScriptedLLM(script)
        engine = EvolutionEngine(
            llm=llm,
            repo=InMemoryRepository(),
            train_tasks=[Task(query="t")],
            val_tasks=[Task(query="v")],
            solver=StaticSolver(action="FINISH"),
            k_rounds=1,
        )
        result = await engine.run(make_sample_graph())
        assert "verify" in result.nodes  # accepted

    async def test_rejects_structurally_invalid_candidate(self) -> None:
        # Refiner proposes a duplicate node (structurally invalid).
        def script(i: int, user: str) -> str:
            if "Propose a JSON array of edits." in user:
                return json.dumps(
                    [
                        {"kind": "add_node", "node": {"id": "search"}},  # duplicate
                    ]
                )
            return ""

        llm = ScriptedLLM(script)
        engine = EvolutionEngine(
            llm=llm,
            repo=InMemoryRepository(),
            train_tasks=[Task(query="t")],
            val_tasks=[Task(query="v")],
            solver=StaticSolver(action="FINISH"),
            k_rounds=1,
        )
        result = await engine.run(make_sample_graph())
        assert "search" in result.nodes  # original still there, no change
        assert len(engine.rejection) == 1  # rejected candidate recorded

    async def test_rejection_memory_included_in_next_prompt(self) -> None:
        # Round 1 refiner rejects a duplicate node; round 2 refiner
        # prompt must contain the REJECTED marker.
        def script(i: int, user: str) -> str:
            if "Propose a JSON array of edits." in user:
                return json.dumps(
                    [
                        {"kind": "add_node", "node": {"id": "search"}},  # duplicate
                    ]
                )
            return ""

        llm = ScriptedLLM(script)
        engine = EvolutionEngine(
            llm=llm,
            repo=InMemoryRepository(),
            train_tasks=[Task(query="t")],
            val_tasks=[Task(query="v")],
            solver=StaticSolver(action="FINISH"),
            k_rounds=2,
        )
        await engine.run(make_sample_graph())
        # Round-2 refiner is the second refiner call.
        refiner_indices = [i for i, call in enumerate(llm.calls) if llm.is_refiner_call(i)]
        assert len(refiner_indices) == 2, "expected 2 refiner calls"
        assert "REJECTED" in llm.calls[refiner_indices[1]]

    async def test_persists_final_graph(self) -> None:
        repo = InMemoryRepository()
        llm = ScriptedLLM(lambda i, _: "[]")
        engine = EvolutionEngine(
            llm=llm,
            repo=repo,
            train_tasks=[Task(query="t")],
            val_tasks=[Task(query="v")],
            solver=StaticSolver(action="FINISH"),
            k_rounds=1,
        )
        graph = make_sample_graph()
        result = await engine.run(graph)
        loaded = await repo.load_graph(result.id)
        assert loaded.id == result.id


class TestRefinerSystemPrompt:
    def test_prompt_mentions_required_keys(self) -> None:
        assert "JSON" in REFINER_SYSTEM_PROMPT
        assert "add_node" in REFINER_SYSTEM_PROMPT
        assert "add_edge" in REFINER_SYSTEM_PROMPT
        assert "leads_to" in REFINER_SYSTEM_PROMPT
        assert "attribute" in REFINER_SYSTEM_PROMPT


class TestSentinelConstants:
    def test_terminal_markers_distinct(self) -> None:
        assert TERMINATE_SUCCESS != TERMINATE_FAILURE
        assert TERMINATE_SUCCESS.startswith("__methodos_")
        assert TERMINATE_FAILURE.startswith("__methodos_")
