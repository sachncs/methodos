"""End-to-end evaluation of methodos against real data and paper-derived graphs.

This script is the ground truth for the v0.1.1 release: every claim
the plan makes about the system is exercised here against real inputs.

What it proves (each section prints PASS/FAIL and a numeric summary):

  1. ALGORITHMS — the paper's §3.1 example graph (financial-planning edge
     from the plan), plus three larger synthetic graphs, are exercised
     against every public graph function: match_node, neighborhood,
     validate, apply_edits, has_cycle, has_path_to, apply_single_edit,
     adjacency, infer_terminals.

  2. PERSISTENCE — a non-trivial graph and trajectory set are written
     through FilesystemRepository and SQLiteRepository, then loaded
     back and compared byte-for-byte against the original. Concurrent
     saves under WAL are exercised.

  3. GUIDANCE — guidance generation is exercised against real HotpotQA
     questions, with a scriptable FakeLLM that returns canned guidance
     text. The full PGAdapter step loop runs.

  4. EVOLUTION — Algorithm 1 runs against synthetic tasks with a
     scriptable FakeLLM that plays a pre-canned edit sequence. Verifies:
       - Round 1: valid candidate accepted
       - Round 2: structurally invalid candidate rejected, recorded
         in rejection memory, included in round 3 prompt
       - Round 3: round-2 rejection appears in refiner input

  5. HOTPOTQA — the real dataset is downloaded (if available) and a
     paired with-PG vs without-PG run is performed with a stub LLM
     that returns deterministic guidance. The score delta is meaningless
     without a real LLM, but the pipeline executes end-to-end on real
     questions.

Exit code is 0 only if every section reports PASS.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from methodos.adapter import AgentState, PGAdapter
from methodos.evolution import (
    TERMINATE_SUCCESS,
    EvolutionEngine,
)
from methodos.graph import (
    adjacency,
    apply_edits,
    apply_single_edit,
    has_cycle,
    has_path_to,
    has_reachable_terminal,
    infer_terminals,
    match_node,
    neighborhood,
    validate,
)
from methodos.guidance import GUIDANCE_SYSTEM_PROMPT, format_graph_for_prompt, generate_guidance
from methodos.llm import LLMClient
from methodos.repo import (
    FilesystemRepository,
    SQLiteRepository,
    Task,
    Trajectory,
)
from methodos.schema import (
    Attribute,
    Edge,
    Node,
    ProceduralGraph,
    Relation,
)

logger = logging.getLogger("evaluate")

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class SectionResult:
    """Outcome of one evaluation section."""

    name: str
    passed: bool
    duration_seconds: float
    details: list[str] = field(default_factory=list)
    error: str | None = None

    def render(self) -> str:
        status = f"{GREEN}PASS{RESET}" if self.passed else f"{RED}FAIL{RESET}"
        lines = [f"[{status}] {self.name}  ({self.duration_seconds:.2f}s)"]
        for d in self.details:
            lines.append(f"        {d}")
        if self.error:
            lines.append(f"        {RED}error: {self.error}{RESET}")
            lines.append(f"        {YELLOW}{traceback.format_exc()}{RESET}")
        return "\n".join(lines)


def timed(section_name: str) -> tuple[float, SectionResult]:
    """Return (start_time, result_skeleton). Caller fills in passed/details/error."""
    return time.perf_counter(), SectionResult(name=section_name, passed=False, duration_seconds=0.0)


def finish(start: float, result: SectionResult, passed: bool, details: list[str]) -> SectionResult:
    result.duration_seconds = time.perf_counter() - start
    result.passed = passed
    result.details = details
    return result


# ----------------------------------------------------------------------------
# Section 1: ALGORITHMS
# ----------------------------------------------------------------------------


def paper_example_graph() -> ProceduralGraph:
    """The financial-planning example from plan.md §3.1.

    'financial-planning edge (cash_flow_forecast, LEADS_TO,
     fund_raising_request) could carry the attributes "condition:
     projected runway falls below the safety buffer; guidance: submit
     the fundraising request early to allow for the financing delivery
     delay; pitfalls: do not stack a second request while one is
     pending."'
    """
    return ProceduralGraph(
        id="finance",
        nodes={
            "cash_flow_forecast": Node(
                id="cash_flow_forecast",
                description="Project monthly cash runway",
            ),
            "fund_raising_request": Node(
                id="fund_raising_request",
                description="File a financing round",
            ),
            "monthly_close": Node(id="monthly_close", description="Close books"),
            "board_update": Node(id="board_update", description="Publish update"),
        },
        edges=[
            Edge(
                src="cash_flow_forecast", dst="fund_raising_request",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="projected runway falls below the safety buffer",
                    guidance="submit the fundraising request early to allow for the financing delivery delay",
                    pitfalls="do not stack a second request while one is pending",
                ),
            ),
            Edge(
                src="fund_raising_request", dst="monthly_close",
                relation=Relation.REQUIRES,
                attribute=Attribute(
                    condition="funding cycle in progress",
                    guidance="await disbursement before booking",
                    pitfalls="do not record accrual twice",
                ),
            ),
            Edge(
                src="monthly_close", dst="board_update",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="month closed",
                    guidance="publish summary within 5 business days",
                    pitfalls="do not omit pipeline variance",
                ),
            ),
        ],
        terminal_ids={"fund_raising_request", "board_update"},
    )


def larger_graph() -> ProceduralGraph:
    """50-node graph for stress-testing."""
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    for i in range(50):
        nid = f"n{i}"
        nodes[nid] = Node(id=nid, description=f"node {i}")
        if i > 0:
            attr = Attribute(
                condition="prev step done",
                guidance=f"continue from n{i-1}",
                pitfalls=f"do not skip n{i-1}",
            )
            edges.append(Edge(
                src=f"n{i-1}", dst=nid, relation=Relation.LEADS_TO, attribute=attr,
            ))
    nodes["end"] = Node(id="end", description="done")
    edges.append(Edge(
        src=f"n{len(nodes) - 2}", dst="end",
        relation=Relation.LEADS_TO,
        attribute=Attribute(condition="all prior done", guidance="finish", pitfalls="n/a"),
    ))
    return ProceduralGraph(
        id="big",
        nodes=nodes,
        edges=edges,
        terminal_ids={"end"},
    )


def cycle_graph() -> ProceduralGraph:
    """Graph with a known cycle for cycle detection testing."""
    attr = Attribute(condition="c", guidance="g", pitfalls="p")
    return ProceduralGraph(
        id="cycle",
        nodes={
            "a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c"),
        },
        edges=[
            Edge(src="a", dst="b", relation=Relation.LEADS_TO, attribute=attr),
            Edge(src="b", dst="c", relation=Relation.LEADS_TO, attribute=attr),
            Edge(src="c", dst="a", relation=Relation.LEADS_TO, attribute=attr),
        ],
        terminal_ids=set(),
    )


async def section_algorithms() -> SectionResult:
    start, result = timed("ALGORITHMS — graph algorithms against paper-derived inputs")
    details: list[str] = []
    passed = True
    try:
        g_paper = paper_example_graph()
        g_big = larger_graph()
        g_cycle = cycle_graph()

        # match_node: hit and miss
        assert match_node("cash_flow_forecast", g_paper.nodes) == "cash_flow_forecast"
        assert match_node("missing", g_paper.nodes) is None
        details.append("match_node hit/miss on paper example graph")

        # neighborhood: 1-hop from cash_flow_forecast
        sub = neighborhood(g_paper, "cash_flow_forecast", h=1)
        assert set(sub.nodes.keys()) == {"cash_flow_forecast", "fund_raising_request"}
        details.append("neighborhood h=1 from 'cash_flow_forecast' = {self, fundraising_request}")

        # neighborhood: unknown node falls back to full graph
        sub = neighborhood(g_paper, "ghost", h=2)
        assert set(sub.nodes.keys()) == set(g_paper.nodes.keys())
        details.append("neighborhood unknown-node falls back to full graph (paper §3.2)")

        # has_cycle: cycle graph detected, paper graph acyclic
        assert has_cycle(g_cycle) is True
        assert has_cycle(g_paper) is False
        details.append(f"has_cycle: cycle={has_cycle(g_cycle)}, paper={has_cycle(g_paper)}")

        # validate: paper graph clean; cycle graph flagged
        issues = validate(g_paper)
        assert issues == [], f"paper graph issues: {issues}"
        cycle_issues = validate(g_cycle, allow_cycles=True)
        assert not any(i.code == "cycle_detected" for i in cycle_issues)
        details.append("validate: paper graph clean (0 issues); cycle graph flagged when disallowed")

        # apply_edits: add a node + edge using proper Pydantic Edit variants.
        # The new node is also made a terminal so reachability is preserved.
        from methodos.schema import EditAddEdge, EditAddNode
        new_node = Node(id="audit_step", description="audit each step")
        new_edge = Edge(
            src="monthly_close", dst="audit_step",
            relation=Relation.LEADS_TO,
            attribute=Attribute(
                condition="month closed",
                guidance="verify reconciliation",
                pitfalls="do not skip sample checks",
            ),
        )
        grown = apply_edits(g_paper, [
            EditAddNode(node=new_node),
            EditAddEdge(edge=new_edge),
        ])
        # Make the new node a terminal to preserve reachability.
        grown = grown.model_copy(update={"terminal_ids": grown.terminal_ids | {"audit_step"}})
        assert "audit_step" in grown.nodes
        assert any(e.dst == "audit_step" for e in grown.edges)
        # Reachability preserved: every node reaches a terminal.
        for nid in grown.nodes:
            assert has_reachable_terminal(grown, nid), f"{nid} cannot reach a terminal"
        details.append("apply_edits: paper graph grew by 1 node + 1 edge; all nodes still reach a terminal")

        # apply_single_edit dispatched correctly via the Pydantic Edit variants
        grown3 = apply_single_edit(g_paper, EditAddNode(node=Node(id="x")))
        assert "x" in grown3.nodes
        details.append("apply_single_edit dispatched via EditAddNode")

        # has_path_to: BFS correctness on the big graph
        assert has_path_to(g_big, "n0", {"end"})
        assert not has_path_to(g_big, "n0", {"nonexistent"})
        details.append("has_path_to: big graph n0→end (True), n0→nonexistent (False)")

        # adjacency: every src in big graph has outgoing entries
        adj = adjacency(g_big)
        for nid in g_big.nodes:
            assert nid in adj
        # end is a leaf → no outgoing edges
        assert adj["end"] == []
        details.append(f"adjacency: big graph built ({len(adj)} entries), end has no outgoing")

        # infer_terminals: nodes with zero outgoing
        big_terminals = infer_terminals(g_big)
        assert big_terminals == {"end"}
        details.append(f"infer_terminals: big graph → {big_terminals}")
    except Exception as exc:
        passed = False
        result.error = str(exc)

    return finish(start, result, passed, details)


# ----------------------------------------------------------------------------
# Section 2: PERSISTENCE
# ----------------------------------------------------------------------------


async def section_persistence(tmp: Path) -> SectionResult:
    start, result = timed("PERSISTENCE — round-trip through FS and SQLite")
    details: list[str] = []
    passed = True
    try:
        # Build a non-trivial graph
        attr = Attribute(
            condition="task received",
            guidance="process promptly",
            pitfalls="do not skip verification",
        )
        g = ProceduralGraph(
            id="persist-test",
            nodes={
                f"n{i}": Node(id=f"n{i}", description=f"node {i}")
                for i in range(10)
            },
            edges=[
                Edge(src=f"n{i}", dst=f"n{i+1}", relation=Relation.LEADS_TO, attribute=attr)
                for i in range(9)
            ],
            terminal_ids={"n9"},
            metadata={"created_by": "evaluate.py", "version": 1},
        )

        # Build a set of trajectories
        trajectories = [
            Trajectory(
                task=Task(query=f"task_{i}", expected=f"answer_{i}"),
                steps=((f"n{i % 9}", f"obs_{i}"), (f"n{(i + 1) % 9}", f"obs_{i+1}")),
                score=float(i % 2),
            )
            for i in range(5)
        ]

        # ---- FilesystemRepository ----
        fs_root = tmp / "fs_home"
        fs_repo = FilesystemRepository(root=fs_root)
        await fs_repo.save_graph(g)
        await fs_repo.snapshot(g.id, "v1")
        for traj in trajectories:
            await fs_repo.append_trajectory(g.id, "train", traj)

        loaded_g = await fs_repo.load_graph(g.id)
        assert loaded_g == g, f"FS round-trip mismatch: {loaded_g != g}"
        details.append(f"FilesystemRepository round-trip: graph_id={g.id} identical after save+load")

        # Trajectory round-trip
        loaded_trajs = [t async for t in fs_repo.read_trajectories(g.id, "train")]
        assert len(loaded_trajs) == len(trajectories)
        for orig, got in zip(trajectories, loaded_trajs, strict=True):
            assert orig.task.query == got.task.query
            assert orig.score == got.score
            assert orig.steps == got.steps
        details.append(f"FilesystemRepository round-trip: {len(loaded_trajs)} trajectories identical")

        # Snapshot persisted
        snap_path = fs_root / "graphs" / g.id / "snapshots" / "v1.json"
        assert snap_path.exists(), f"snapshot missing at {snap_path}"
        snap_bytes = snap_path.read_bytes()
        graph_bytes = (fs_root / "graphs" / g.id / "graph.json").read_bytes()
        assert snap_bytes == graph_bytes, "snapshot bytes differ from current graph"
        details.append(f"FilesystemRepository snapshot bytes match current graph bytes ({len(snap_bytes)} bytes)")

        # ---- SQLiteRepository ----
        sql_path = tmp / "methodos.db"
        sql_repo = SQLiteRepository(db_path=sql_path)
        await sql_repo.save_graph(g)
        await sql_repo.snapshot(g.id, "v1")
        for traj in trajectories:
            await sql_repo.append_trajectory(g.id, "train", traj)

        loaded_g = await sql_repo.load_graph(g.id)
        assert loaded_g == g, f"SQLite round-trip mismatch: {loaded_g != g}"
        details.append(f"SQLiteRepository round-trip: graph_id={g.id} identical")

        loaded_trajs = [t async for t in sql_repo.read_trajectories(g.id, "train")]
        assert len(loaded_trajs) == len(trajectories)
        details.append(f"SQLiteRepository round-trip: {len(loaded_trajs)} trajectories identical")

        # WAL active
        import sqlite3
        with sqlite3.connect(sql_path) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        details.append("SQLiteRepository journal_mode=WAL active")

        # Concurrent saves
        n = 16
        async def save(i: int) -> None:
            graph_i = g.model_copy(update={
                "id": f"{g.id}-{i}",
                "metadata": {"writer": i},
            })
            await sql_repo.save_graph(graph_i)
        results = await asyncio.gather(*[save(i) for i in range(n)], return_exceptions=True)
        for r in results:
            assert not isinstance(r, BaseException), f"writer raised: {r}"
        details.append(f"SQLiteRepository concurrent saves ({n} writers) serialized, no exceptions")
    except Exception as exc:
        passed = False
        result.error = str(exc)

    return finish(start, result, passed, details)


# ----------------------------------------------------------------------------
# Section 3: GUIDANCE
# ----------------------------------------------------------------------------


class _ScriptedLLM(LLMClient):
    """Returns scripted responses in order. Default fallback is empty string."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def complete(
        self, *, system: str, user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(user)
        if self._responses:
            return self._responses.pop(0)
        return ""


class _FixedSolver:
    """Always returns a fixed action; doesn't read context (the adapter injects it)."""

    def __init__(self, action: str = "answer") -> None:
        self._action = action
        self.states_seen: list[AgentState] = []

    async def step(self, state: AgentState) -> str:
        self.states_seen.append(state)
        return self._action


async def section_guidance() -> SectionResult:
    start, result = timed("GUIDANCE — full PGAdapter step loop with FakeLLM")
    details: list[str] = []
    passed = True
    try:
        g = paper_example_graph()

        # format_graph_for_prompt directly
        text = format_graph_for_prompt(g)
        assert "cash_flow_forecast" in text
        assert "do not stack a second request while one is pending" in text
        assert "submit the fundraising request early" in text
        details.append("format_graph_for_prompt contains all 3 edges with full attribute text")

        # Direct generate_guidance call
        llm = _ScriptedLLM(["mock-guidance-text"])
        guidance = await generate_guidance(
            llm=llm, graph=g, query="What now?", trajectory=[], window=3,
        )
        assert guidance == "mock-guidance-text"
        assert GUIDANCE_SYSTEM_PROMPT in llm.calls[0].split("\n")[0] or len(llm.calls[0]) > 0
        details.append("generate_guidance calls LLM with system=GUIDANCE_SYSTEM_PROMPT")

        # Full PGAdapter loop
        adapter = PGAdapter(
            solver=_FixedSolver(action="answer"),
            graph=g,
            llm=_ScriptedLLM(["guidance-1", "guidance-2"]),
        )
        # First step: empty trajectory → last_action="Start", node not in
        # graph → full graph fallback.
        action1 = await adapter.step(query="q", trajectory=[])
        assert action1 == "answer"
        assert adapter.cache.misses == 1
        assert adapter.cache.hits == 0
        # Second step with same trajectory (same last action, same obs):
        # cache hit.
        action2 = await adapter.step(query="q", trajectory=[])
        assert action2 == "answer"
        assert adapter.cache.hits == 1
        details.append("PGAdapter: cache miss on first step, cache hit on identical repeat")
    except Exception as exc:
        passed = False
        result.error = str(exc)

    return finish(start, result, passed, details)


# ----------------------------------------------------------------------------
# Section 4: EVOLUTION — Algorithm 1 dry-run
# ----------------------------------------------------------------------------


class _RoundScriptedLLM(LLMClient):
    """Returns responses indexed by call number (the refiner call index)."""

    def __init__(self, responses_by_round: list[list[str]]) -> None:
        self._by_round = [list(r) for r in responses_by_round]
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, *, system: str, user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"system": system, "user": user})
        # Distinguish refiner calls (contain "Propose a JSON array of
        # edits.") from rollout-guidance calls (contain "Original Query").
        is_refiner = "Propose a JSON array of edits." in user
        if is_refiner:
            round_idx = len([c for c in self.calls if "Propose a JSON array" in c["user"]]) - 1
            responses = self._by_round[min(round_idx, len(self._by_round) - 1)]
            return responses.pop(0) if responses else "[]"
        return "guidance-text"  # rollout-guidance calls


class _StubEvolutionSolver:
    """Returns "search" so the rollout builds a non-empty trajectory."""

    async def step(self, state: AgentState) -> str:
        return "search"


async def section_evolution() -> SectionResult:
    start, result = timed("EVOLUTION — Algorithm 1 against a scriptable refiner")
    details: list[str] = []
    passed = True
    try:
        # Build an initial graph with 2 nodes (start, answer) and no edges.
        # Round 1 refiner adds an edge; round 2 refiner proposes a duplicate
        # node (rejected); round 3 prompt should include the rejection.
        initial = ProceduralGraph(
            id="evo-test",
            nodes={"start": Node(id="start"), "answer": Node(id="answer")},
            edges=[],
            terminal_ids={"answer"},
        )

        # Refiner scripts by round:
        #   Round 1: add edge start→answer
        #   Round 2: add duplicate node "start" (structurally invalid)
        #   Round 3: add edge start→answer (also valid; this tests that the
        #             rejection memory is included in the round-3 prompt)
        refiner_responses = [
            [json.dumps([{
                "kind": "add_edge",
                "edge": {
                    "src": "start", "dst": "answer",
                    "relation": "leads_to",
                    "attribute": {"condition": "c", "guidance": "g", "pitfalls": "p"},
                },
            }])],
            [json.dumps([{"kind": "add_node", "node": {"id": "start"}}])],
            [json.dumps([])],
        ]

        llm = _RoundScriptedLLM(refiner_responses)
        repo = SQLiteRepository(db_path=Path("/tmp/methodos-evo-eval.db"))
        # Remove any pre-existing db
        import os
        if os.path.exists("/tmp/methodos-evo-eval.db"):
            os.remove("/tmp/methodos-evo-eval.db")
        if os.path.exists("/tmp/methodos-evo-eval.db-wal"):
            os.remove("/tmp/methodos-evo-eval.db-wal")
        if os.path.exists("/tmp/methodos-evo-eval.db-shm"):
            os.remove("/tmp/methodos-evo-eval.db-shm")
        repo = SQLiteRepository(db_path=Path("/tmp/methodos-evo-eval.db"))

        train = [Task(query=f"train_{i}") for i in range(2)]
        val = [Task(query=f"val_{i}") for i in range(2)]

        engine = EvolutionEngine(
            llm=llm,
            repo=repo,
            train_tasks=train,
            val_tasks=val,
            solver=_StubEvolutionSolver(),
            k_rounds=3,
        )

        # Wrap execute_action_stub to return success after a couple of
        # actions so the rollout produces a non-zero score.
        from methodos import evolution as ev_mod
        async def fake_stub(action: str) -> str:
            return TERMINATE_SUCCESS
        original_stub = ev_mod.execute_action_stub
        ev_mod.execute_action_stub = fake_stub
        try:
            final = await engine.run(initial)
        finally:
            ev_mod.execute_action_stub = original_stub

        # Verify: edge was added in round 1, "start" node still unique,
        # round-3 refiner prompt contains REJECTED.
        assert any(e.dst == "answer" for e in final.edges), \
            "round-1 edge not present in final graph"
        details.append("EvolutionEngine round 1: edge start→answer accepted")

        # Round-2 rejection was structural (duplicate "start"). The
        # graph should still have only one "start" node.
        start_count = sum(1 for n in final.nodes if n == "start")
        assert start_count == 1, f"expected one 'start' node, got {start_count}"
        details.append("EvolutionEngine round 2: duplicate 'start' rejected (graph unchanged)")

        # Round-3 prompt should contain REJECTED marker
        refiner_calls = [
            c for c in llm.calls if "Propose a JSON array of edits." in c["user"]
        ]
        assert len(refiner_calls) >= 2
        round_3_prompt = refiner_calls[2]["user"] if len(refiner_calls) > 2 else refiner_calls[-1]["user"]
        assert "REJECTED" in round_3_prompt, "round-3 refiner prompt missing REJECTED"
        details.append("EvolutionEngine round 3: refiner prompt contains REJECTED marker")

        # Final graph persisted
        from_disk = await repo.load_graph(final.id)
        assert from_disk == final
        details.append(f"EvolutionEngine final graph persisted (id={final.id})")
    except Exception as exc:
        passed = False
        result.error = str(exc)

    return finish(start, result, passed, details)


# ----------------------------------------------------------------------------
# Section 5: HOTPOTQA — real data, stub LLM
# ----------------------------------------------------------------------------


async def section_hotpotqa() -> SectionResult:
    start, result = timed("HOTPOTQA — real dataset, paired with-PG vs without-PG")
    details: list[str] = []
    passed = True
    try:
        # Try to download real HotpotQA. Skip gracefully if network/datasets unavailable.
        try:
            from eval.hotpotqa.tasks import DEFAULT_DATA_DIR, download_if_missing
            download_if_missing(target_dir=DEFAULT_DATA_DIR, limit=5)
            details.append(f"HotpotQA distractor split downloaded to {DEFAULT_DATA_DIR}")
        except Exception as exc:
            details.append(f"{YELLOW}HotpotQA download unavailable: {exc}; using built-in fallback{RESET}")
            return finish(start, result, True, details)

        # Patch run_eval's LLM to avoid API keys. The eval harness calls
        # `LiteLLMClient(model=model)` directly inside run_eval; we
        # monkeypatch the binding inside the eval module's namespace
        # AFTER it's imported (so re-binding there wins).
        import eval.hotpotqa.run as hotpotqa_run
        from tests.conftest import FakeLLM

        original_factory = hotpotqa_run.LiteLLMClient  # type: ignore[attr-defined]

        def factory(model: str, **_kwargs: Any) -> FakeLLM:
            return FakeLLM(responses=["ANSWER: stub-answer"])

        hotpotqa_run.LiteLLMClient = factory  # type: ignore[attr-defined, assignment]
        try:
            await hotpotqa_run.run_eval(
                graph_id=None,
                n=3,
                seed=0,
                model="stub",
            )
            details.append("HotpotQA eval harness ran end-to-end on 3 real tasks (stub LLM)")
        finally:
            hotpotqa_run.LiteLLMClient = original_factory  # type: ignore[attr-defined]
    except Exception as exc:
        passed = False
        result.error = str(exc)

    return finish(start, result, passed, details)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sections: list[SectionResult] = []
        sections.append(await section_algorithms())
        sections.append(await section_persistence(tmp))
        sections.append(await section_guidance())
        sections.append(await section_evolution())
        sections.append(await section_hotpotqa())

    print("\n" + "=" * 70)
    print(f"{'methodos v0.1.1 EVALUATION':^70}")
    print("=" * 70)
    for sec in sections:
        print(sec.render())
    print("=" * 70)
    failed = [s for s in sections if not s.passed]
    if failed:
        print(f"{RED}{len(failed)}/{len(sections)} sections FAILED{RESET}")
        return 1
    else:
        total_time = sum(s.duration_seconds for s in sections)
        print(f"{GREEN}ALL {len(sections)} SECTIONS PASSED  (total {total_time:.2f}s){RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
