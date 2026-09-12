"""Self-evolving procedural graph: paper §3.3 Algorithm 1.

This module implements the offline self-evolution loop:

  For k in 1..K:
    1. Diagnostic rollout: run the adapter+solver on train tasks; record
       trajectories + scores.
    2. Refiner: ask an LLM to propose Edit operations on the graph based
       on success vs failure patterns.
    3. Validate candidate: apply edits structurally; check reachability,
       cycles, etc.
    4. Accept iff validation score did not regress; otherwise record the
       candidate in rejection memory.

Engineering:
- Pure functions for each step; `EvolutionEngine` orchestrates them.
- All attributes are public (no `self._foo` markers).
- No lazy imports: `PGAdapter` is imported at module level (no cycle).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from methodos.adapter import PGAdapter, Solver
from methodos.graph import apply_edits
from methodos.graph import validate as graph_validate
from methodos.llm import LLMClient
from methodos.repo import Repository, Task, Trajectory
from methodos.schema import (
    Edit,
    EditAddEdge,
    EditAddNode,
    EditDeleteEdge,
    EditDeleteNode,
    EditUpdateAttr,
    ProceduralGraph,
)

logger = logging.getLogger(__name__)


REFINER_SYSTEM_PROMPT: str = """\
You are a procedural-graph refiner. Given successful and failed execution \
traces of an agent on a task, propose edits to the procedural graph that \
will reduce future failures.

Allowed edit operations (return as a JSON array of these objects):
- {"kind": "add_node", "node": {"id": "<id>", "description": "<text>"}}
- {"kind": "delete_node", "node_id": "<id>"}
- {"kind": "add_edge", "edge": {"src": "<id>", "dst": "<id>",
  "relation": "leads_to|requires|replaces",
  "attribute": {"condition": "<text>", "guidance": "<text>", "pitfalls": "<text>"}}}
- {"kind": "delete_edge", "src": "<id>", "dst": "<id>", "relation": "leads_to|requires|replaces"}
- {"kind": "update_attr", "src": "<id>", "dst": "<id>",
  "relation": "leads_to|requires|replaces",
  "attribute": {"condition": "<text>", "guidance": "<text>", "pitfalls": "<text>"}}

Constraints:
- All referenced nodes must exist (or be added in the same proposal).
- Every edge must reference real nodes.
- Edits must be minimal and targeted; do not propose large rewrites.
- The agent's reasoning freedom must be preserved.

Return ONLY the JSON array. No prose, no markdown fences.
"""


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Outcome of running the agent on a single task."""

    trajectory: Trajectory
    success: bool


# ----------------------------------------------------------------------------
# Pure functions
# ----------------------------------------------------------------------------


def score(result: RolloutResult) -> float:
    """Score a rollout: 1.0 for success, 0.0 otherwise."""
    return 1.0 if result.success else 0.0


def mean_score(results: Iterable[RolloutResult]) -> float:
    """Mean score across rollouts. Returns 0.0 for empty input."""
    materialized = list(results)
    if not materialized:
        return 0.0
    return sum(score(r) for r in materialized) / len(materialized)


async def run_rollout(
    *,
    graph: ProceduralGraph,
    solver: Solver,
    llm: LLMClient,
    task: Task,
    max_steps: int = 50,
    guidance_hops: int = 2,
    trajectory_window: int = 3,
) -> RolloutResult:
    """Run the agent on a task via `PGAdapter` until success, failure, or max_steps."""
    adapter = PGAdapter(
        solver=solver,
        graph=graph,
        llm=llm,
        guidance_hops=guidance_hops,
        trajectory_window=trajectory_window,
    )
    steps: list[tuple[str, str]] = []
    last_action = "Start"
    success = False

    for _ in range(max_steps):
        action = await adapter.step(query=task.query, trajectory=steps)
        if action == last_action and steps:
            logger.warning("doom loop on task; aborting rollout")
            break
        observation = await execute_action(action)
        steps.append((action, observation))
        last_action = action
        if observation == TERMINATE_SUCCESS:
            success = True
            break
        if observation == TERMINATE_FAILURE:
            break

    final_score = 1.0 if success else 0.0
    trajectory = Trajectory(task=task, steps=tuple(steps), score=final_score)
    return RolloutResult(trajectory=trajectory, success=success)


TERMINATE_SUCCESS: str = "__methodos_success__"
TERMINATE_FAILURE: str = "__methodos_failure__"


async def execute_action(action: str) -> str:
    """Stand-in action executor for evolution rollouts.

    Real deployments wire this to their environment (search tools, code
    execution, etc.). The default returns an empty observation, which the
    rollout interprets as "no terminal signal" — the agent loop continues
    until `max_steps` is reached or a doom loop is detected.
    """
    return ""


def tail_concat(traces: Iterable[Trajectory], max_tokens: int) -> str:
    """Concatenate trajectories preserving the END (paper's `Tail_{L_max}`)."""
    if max_tokens <= 0:
        return ""
    chunks: list[str] = []
    for traj in traces:
        chunks.append(f"# Task: {traj.task.query} | score: {traj.score}\n")
        for action, obs in traj.steps:
            chunks.append(f"  A: {action}\n  O: {obs}\n")
    full = "\n".join(chunks)
    char_budget = max_tokens * 4
    if len(full) <= char_budget:
        return full
    return full[-char_budget:]


async def propose_edits(
    *,
    llm: LLMClient,
    graph: ProceduralGraph,
    traces: Sequence[Trajectory],
    rejected: Sequence[tuple[Edit, float]],
    context_tokens: int = 6000,
) -> list[Edit]:
    """Ask the LLM refiner to propose edits to the graph."""
    tail_text = tail_concat(traces, max_tokens=context_tokens)
    rejected_text = (
        "\n".join(
            f"REJECTED: {edit.model_dump_json()} (val_score={val_score:.3f})"
            for edit, val_score in rejected
        )
        or "(no rejected edits yet)"
    )

    user_prompt = (
        f"# Current graph (id={graph.id}, "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges)\n"
        f"{graph.model_dump_json(indent=2)}\n\n"
        f"# Recent trajectories (newest at the end)\n{tail_text}\n\n"
        f"# Rejection history\n{rejected_text}\n\n"
        "Propose a JSON array of edits."
    )

    raw = await llm.complete(
        system=REFINER_SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
    )

    payload = parse_refiner_response(raw, logger)
    edits: list[Edit] = []
    for item in payload:
        edits.extend(item_to_edit(item))
    return edits


def parse_refiner_response(raw: str, log: logging.Logger) -> list[dict[str, Any]]:
    """Parse the refiner's JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        log.warning("refiner returned non-JSON: %s", text[:200])
        return []
    if not isinstance(payload, list):
        log.warning("refiner returned non-list: %s", type(payload).__name__)
        return []
    return [item for item in payload if isinstance(item, dict)]


def item_to_edit(item: dict[str, Any]) -> list[Edit]:
    """Convert one refiner output dict to 0+ Edit instances."""
    kind = item.get("kind")
    try:
        if kind == "add_node":
            return [EditAddNode.model_validate(item)]
        if kind == "delete_node":
            return [EditDeleteNode.model_validate(item)]
        if kind == "add_edge":
            return [EditAddEdge.model_validate(item)]
        if kind == "delete_edge":
            return [EditDeleteEdge.model_validate(item)]
        if kind == "update_attr":
            return [EditUpdateAttr.model_validate(item)]
    except Exception as exc:
        logger.warning("refiner produced malformed edit %r: %s", item, exc)
        return []
    logger.warning("unknown edit kind from refiner: %r", kind)
    return []


def validate_candidate(
    graph: ProceduralGraph, edits: Sequence[Edit], *, allow_cycles: bool = False
) -> ProceduralGraph | None:
    """Apply edits to a copy; return the candidate if structurally valid."""
    try:
        candidate = apply_edits(graph, list(edits))
    except ValueError as exc:
        logger.info("structural rejection: %s", exc)
        return None

    issues = graph_validate(candidate, allow_cycles=allow_cycles)
    if issues:
        codes = sorted({i.code for i in issues})
        logger.info("candidate rejected by structural checks: %s", codes)
        return None
    return candidate


# ----------------------------------------------------------------------------
# RejectionMemory + EvolutionEngine
# ----------------------------------------------------------------------------


class RejectionMemory:
    """Bounded FIFO of rejected candidates (paper §3.3 step 4)."""

    def __init__(self, max_size: int = 32) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.store: deque[tuple[Edit, float]] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.store)

    def add(self, edits: Sequence[Edit], val_score: float) -> None:
        """Record the first rejected edit as a marker for this rejection."""
        if not edits:
            return
        self.store.append((edits[0], val_score))

    def snapshot(self) -> list[tuple[Edit, float]]:
        return list(self.store)


class EvolutionEngine:
    """Algorithm 1: greedy self-evolution with rejection memory."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        repo: Repository,
        train_tasks: Sequence[Task],
        val_tasks: Sequence[Task],
        solver: Solver,
        k_rounds: int = 10,
        l_max_tokens: int = 8000,
        rejection_memory_size: int = 32,
        allow_cycles: bool = False,
        max_steps: int = 50,
    ) -> None:
        if k_rounds <= 0:
            raise ValueError(f"k_rounds must be positive, got {k_rounds}")
        if l_max_tokens <= 0:
            raise ValueError(f"l_max_tokens must be positive, got {l_max_tokens}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.llm = llm
        self.repo = repo
        self.train: list[Task] = list(train_tasks)
        self.val: list[Task] = list(val_tasks)
        self.solver = solver
        self.rounds = k_rounds
        self.context_tokens = l_max_tokens
        self.allow_cycles = allow_cycles
        self.max_steps = max_steps
        self.rejection = RejectionMemory(max_size=rejection_memory_size)

    async def run(self, graph: ProceduralGraph) -> ProceduralGraph:
        """Run K rounds of evolution; return the retained graph."""
        current = graph
        current_score = await self.score_validation(current)
        logger.info("initial validation score: %.3f", current_score)

        for round_idx in range(1, self.rounds + 1):
            traces = await self.collect_diagnostic_traces(current)
            edits = await propose_edits(
                llm=self.llm,
                graph=current,
                traces=traces,
                rejected=self.rejection.snapshot(),
                context_tokens=self.context_tokens,
            )
            candidate = validate_candidate(
                current,
                edits,
                allow_cycles=self.allow_cycles,
            )
            if candidate is None:
                self.rejection.add(edits, current_score)
                logger.info("round %d: rejected (structural)", round_idx)
                continue

            candidate_score = await self.score_validation(candidate)
            if candidate_score >= current_score:
                current = candidate
                current_score = candidate_score
                logger.info(
                    "round %d: accepted (val_score=%.3f)",
                    round_idx,
                    candidate_score,
                )
            else:
                self.rejection.add(edits, candidate_score)
                logger.info(
                    "round %d: rejected (val_score=%.3f < %.3f)",
                    round_idx,
                    candidate_score,
                    current_score,
                )

        await self.repo.save_graph(current)
        return current

    async def collect_diagnostic_traces(self, graph: ProceduralGraph) -> list[Trajectory]:
        traces: list[Trajectory] = []
        for task in self.train:
            result = await run_rollout(
                graph=graph,
                solver=self.solver,
                llm=self.llm,
                task=task,
                max_steps=self.max_steps,
            )
            traces.append(result.trajectory)
        return traces

    async def score_validation(self, graph: ProceduralGraph) -> float:
        scores: list[float] = []
        for task in self.val:
            result = await run_rollout(
                graph=graph,
                solver=self.solver,
                llm=self.llm,
                task=task,
                max_steps=self.max_steps,
            )
            scores.append(score(result))
        return sum(scores) / max(1, len(scores))


__all__ = [
    "REFINER_SYSTEM_PROMPT",
    "TERMINATE_FAILURE",
    "TERMINATE_SUCCESS",
    "EvolutionEngine",
    "RejectionMemory",
    "RolloutResult",
    "execute_action",
    "item_to_edit",
    "mean_score",
    "parse_refiner_response",
    "propose_edits",
    "run_rollout",
    "score",
    "tail_concat",
    "validate_candidate",
]
