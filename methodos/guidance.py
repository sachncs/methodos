"""Procedural guidance generation: paper §3.2.

`generate_guidance` translates a localized subgraph plus the agent's
recent trajectory into a concise situational paragraph that biases the
solver's next action without dictating it.

The function is intentionally side-effect-free aside from the LLM call;
the graph is frozen during inference (per the paper).

Engineering notes:
- All names are public. No `_foo()` markers — the helpers are useful
  primitives that callers may want to import directly for custom
  guidance pipelines.
- No lazy imports; LLMClient is a top-level import.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from methodos.llm import LLMClient
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)


GUIDANCE_SYSTEM_PROMPT: str = """\
You translate a procedural subgraph into a brief situational hint for an \
LLM agent.

Given:
- The agent's recent trajectory (last few action/observation pairs).
- The user's original query.
- A local subgraph of admissible next procedures (with textual attributes).

Produce ONE concise guidance paragraph (under 120 words). The guidance must:
1. Identify what the agent is currently trying to accomplish, based on the \
trajectory.
2. List the admissible next procedures in priority order, citing their \
condition/guidance/pitfalls.
3. Warn about any pitfall from the subgraph that applies to the current \
trajectory context.
4. NEVER invent procedures not present in the subgraph.
5. NEVER dictate the exact action — the agent retains reasoning freedom.

Respond with the guidance paragraph only. No preamble, no JSON, no markdown.
"""


def format_graph_for_prompt(graph: ProceduralGraph) -> str:
    """Serialize a subgraph into a token-efficient textual form.

    Format:
        # Subgraph: <id>
        # Nodes (<n>):
          - <id>: <description>
        # Edges (<n>):
          - <src> --<relation>--> <dst>
              condition: <text>
              guidance:   <text>
              pitfalls:   <text>
    """
    lines: list[str] = []
    lines.append(f"# Subgraph: {graph.id}")
    lines.append(f"# Nodes ({len(graph.nodes)}):")
    for node_id, node in graph.nodes.items():
        description = node.description or "(no description)"
        lines.append(f"  - {node_id}: {description}")
    lines.append(f"# Edges ({len(graph.edges)}):")
    for edge in graph.edges:
        attr = edge.attribute
        lines.append(
            f"  - {edge.src} --{edge.relation.value}--> {edge.dst}\n"
            f"      condition: {attr.condition}\n"
            f"      guidance:   {attr.guidance}\n"
            f"      pitfalls:   {attr.pitfalls}"
        )
    return "\n".join(lines)


def format_trajectory_window(trajectory: Iterable[tuple[str, str]], window: int) -> str:
    """Format the last `window` action/observation pairs for the prompt."""
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    materialized = list(trajectory)
    recent = materialized[-window:] if window > 0 else []
    if not recent:
        return "(no prior actions)"
    lines: list[str] = []
    for index, (action, observation) in enumerate(recent, start=1):
        lines.append(f"Step {index}: ACTION: {action}")
        lines.append(f"         OBS:    {observation}")
    return "\n".join(lines)


async def generate_guidance(
    *,
    llm: LLMClient,
    graph: ProceduralGraph,
    query: str,
    trajectory: Iterable[tuple[str, str]],
    window: int = 3,
) -> str:
    """Compute guidance `g_t` for the current step (paper Eq. 2).

    Implements the "extract → generate" half of the locate→extract→generate
    pipeline. `locate` is performed by the caller (typically
    `methodos.graph.match_node`) so the graph passed here is already the
    localized subgraph (or the full graph as fallback per paper §3.2).
    """
    graph_text = format_graph_for_prompt(graph)
    trajectory_text = format_trajectory_window(trajectory, window)
    user_prompt = (
        f"# Original Query\n{query}\n\n# Recent Trajectory\n{trajectory_text}\n\n{graph_text}"
    )

    response = await llm.complete(
        system=GUIDANCE_SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.0,
    )
    logger.debug("generated guidance: %d chars", len(response))
    return response


__all__ = [
    "GUIDANCE_SYSTEM_PROMPT",
    "format_graph_for_prompt",
    "format_trajectory_window",
    "generate_guidance",
]
