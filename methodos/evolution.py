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
- No `_foo()` markers. Helpers are public where useful.
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
will reduce future failures. The refiner is one half of paper Algorithm 1 \
(App. B.6, line 9) and the seven rules below mirror paper §B.5 verbatim.

Allowed edit operations (return as a JSON array):
- {"kind": "add_node", "node": {"id": "<id>", "description": "<text>"}}
- {"kind": "delete_node", "node_id": "<id>"}
- {"kind": "add_edge", "edge": {"src": "<id>", "dst": "<id>",
  "relation": "leads_to|triggers|requires|converges_to|replaces",
  "attribute": {"condition": "<text>", "guidance": "<text>", "pitfalls": "<text>"}}}
- {"kind": "delete_edge", "src": "<id>", "dst": "<id>",
  "relation": "leads_to|triggers|requires|converges_to|replaces"}
- {"kind": "update_attr", "src": "<id>", "dst": "<id>",
  "relation": "leads_to|triggers|requires|converges_to|replaces",
  "attribute": {"condition": "<text>", "guidance": "<text>", "pitfalls": "<text>"}}

Refiner rules (paper §B.5, all seven):

1. ACTION-NODE MATCHING. Any node of kind "ACTION" must match one of the
   action/tool names in the `Available Tool Actions` list. STATUS nodes
   (markers like `Start`, `End`, progress indicators) do not need to match.

2. TRANSITION CONDITIONS. Provide a natural-language semantic
   precondition. Use the string "null" if unconditional. Example:
   "When dialogue history has been parsed but target constraints are
   unknown."

3. EXECUTION GUIDANCE (mandatory on every added edge). Must detail exactly
   what action to take next and the strategic rationale. The guidance text
   is the most important field for downstream solver behavior.

4. PITFALLS (mandatory on every added edge). Warns against premature
   actions, forbidden words, common formatting pitfalls. This is the
   second-most-important field after guidance.

5. GENERALITY & LEAK PREVENTION. The updated Procedural Graph must guide
   the agent effectively without overfitting to specific details of a
   single trajectory. Use high-level conceptual descriptions.

6. NODE-ID COMPATIBILITY (static modes). In static modes, you must preserve
   existing node IDs (e.g., `Month_Start`, `Decide_Capital`, tool names) so
   they remain compatible with the environment's state tracker. Do not
   rename them. In scratch modes you may propose new node IDs.

7. GRAPH STRUCTURE. Follow the task's configured cycle policy. Every edge
   must reference existing nodes, and every node must have a directed
   path to a terminal node. The environment loop handles repetition
   across simulation cycles.

Other constraints:
- All referenced nodes must exist (or be added in the same proposal).
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
    """Run the agent on a task via `PGAdapter` until success, failure, or max_steps.

    Returns a `RolloutResult` with the full trajectory and a boolean success
    flag. The trajectory is also wrapped in a `Trajectory` for the
    repository contract.
    """
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
        # Doom-loop detection: same action twice → bail.
        if action == last_action and steps:
            logger.warning("doom loop on task; aborting rollout")
            break
        # Sentinel observations signal terminal conditions. Real
        # environments (search tools, etc.) are wired by the host.
        observation = await execute_action_stub(action)
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


async def execute_action_stub(action: str) -> str:
    """Stand-in action executor for evolution rollouts.

    Real deployments wire this to their environment (search tools, code
    execution, etc.). The stub returns an empty observation, which the
    rollout interprets as "no terminal signal" — the agent loop continues
    until `max_steps` is reached or a doom loop is detected.
    """
    return ""


def tail_concat(traces: Iterable[Trajectory], max_tokens: int) -> str:
    """Concatenate trajectories preserving the END (paper's `Tail_{L_max}`).

    Greedy truncation from the BEGINNING; if the total exceeds the token
    budget, only the last `max_tokens * 4` characters are returned.

    Token approximation: 1 token ≈ 4 characters. The refiner prompt uses
    this to keep context window bounded.
    """
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
    """Ask the LLM refiner to propose edits to the graph.

    Builds a user prompt containing the current graph, the tail of the
    recent trajectories, and the rejection history. Returns a parsed
    list of `Edit` objects. Malformed items are dropped with a
    `logger.warning`; an empty list is returned on any failure.
    """
    tail_text = tail_concat(traces, max_tokens=context_tokens)
    rejected_text = "\n".join(
        f"REJECTED: {edit.model_dump_json()} (val_score={val_score:.3f})"
        for edit, val_score in rejected
    ) or "(no rejected edits yet)"

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

    payload = _parse_refiner_response(raw, logger)
    edits: list[Edit] = []
    for item in payload:
        edits.extend(_item_to_edit(item))
    return edits


def _parse_refiner_response(raw: str, log: logging.Logger) -> list[dict[str, Any]]:
    """Parse the refiner's JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
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


def _item_to_edit(item: dict[str, Any]) -> list[Edit]:
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
    """Apply edits to a copy; return the candidate if structurally valid.

    Returns `None` on any failure (apply error or structural check).
    """
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
    """Bounded FIFO of rejected candidates (paper §3.3 step 4).

    Each entry records the rejected edits and the candidate's validation
    score at the time of rejection. The refiner prompt receives a
    snapshot of these to discourage repeated unsuccessful proposals.
    """

    def __init__(self, max_size: int = 32) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max = max_size
        self._store: deque[tuple[Edit, float]] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._store)

    def add(self, edits: Sequence[Edit], val_score: float) -> None:
        """Record the first rejected edit as a marker for this rejection."""
        if not edits:
            return
        # Mark this rejection with a representative single edit; the refiner
        # only needs a hint, not the full batch.
        self._store.append((edits[0], val_score))

    def snapshot(self) -> list[tuple[Edit, float]]:
        return list(self._store)


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
        self._llm = llm
        self._repo = repo
        self._train = list(train_tasks)
        self._val = list(val_tasks)
        self._solver = solver
        self._k = k_rounds
        self._l_max = l_max_tokens
        self._allow_cycles = allow_cycles
        self._max_steps = max_steps
        self._rejection = RejectionMemory(max_size=rejection_memory_size)

    async def run(self, graph: ProceduralGraph) -> ProceduralGraph:
        """Run K rounds of evolution; return the retained graph."""
        current = graph
        current_score = await self.score_validation(current)
        logger.info("initial validation score: %.3f", current_score)

        for round_idx in range(1, self._k + 1):
            traces = await self._collect_diagnostic_traces(current)
            edits = await propose_edits(
                llm=self._llm,
                graph=current,
                traces=traces,
                rejected=self._rejection.snapshot(),
                context_tokens=self._l_max,
            )
            candidate = validate_candidate(
                current, edits, allow_cycles=self._allow_cycles,
            )
            if candidate is None:
                self._rejection.add(edits, current_score)
                logger.info("round %d: rejected (structural)", round_idx)
                continue

            candidate_score = await self.score_validation(candidate)
            if candidate_score >= current_score:
                current = candidate
                current_score = candidate_score
                logger.info(
                    "round %d: accepted (val_score=%.3f)", round_idx, candidate_score,
                )
            else:
                self._rejection.add(edits, candidate_score)
                logger.info(
                    "round %d: rejected (val_score=%.3f < %.3f)",
                    round_idx, candidate_score, current_score,
                )

        await self._repo.save_graph(current)
        return current

    async def _collect_diagnostic_traces(
        self, graph: ProceduralGraph
    ) -> list[Trajectory]:
        traces: list[Trajectory] = []
        for task in self._train:
            result = await run_rollout(
                graph=graph,
                solver=self._solver,
                llm=self._llm,
                task=task,
                max_steps=self._max_steps,
            )
            traces.append(result.trajectory)
        return traces

    async def score_validation(self, graph: ProceduralGraph) -> float:
        scores: list[float] = []
        for task in self._val:
            result = await run_rollout(
                graph=graph,
                solver=self._solver,
                llm=self._llm,
                task=task,
                max_steps=self._max_steps,
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
    "execute_action_stub",
    "mean_score",
    "propose_edits",
    "run_rollout",
    "score",
    "tail_concat",
    "validate_candidate",
]
