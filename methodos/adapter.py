"""`PGAdapter`: wraps a `Solver` with on-demand procedural guidance.

This module implements the runtime inference half of the paper (paper §3.2):
locate → extract → generate. The graph passed in is FROZEN for the
lifetime of the adapter; evolution produces a new graph + a new adapter.

Engineering notes:
- `AgentState` is a frozen slotted dataclass: hashable, immutable, low
  memory overhead. Solvers receive it as the only mutable surface.
- `Solver` is a `typing.Protocol` — any object with `async def step(state)`
  satisfying the signature is accepted. No ABCs.
- `GuidanceCache` is an LRU keyed by `(hash(subgraph), last_action, last_obs)`.
  The hash is content-based so that mutations to the same graph_id
  produce different keys (see `graph_content_fingerprint`).
- No lazy imports; no `_foo()` markers. The cache is exposed publicly
  so callers can introspect hit/miss behavior if desired.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from methodos.graph import match_node, neighborhood
from methodos.guidance import generate_guidance
from methodos.llm import LLMClient
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)


def graph_content_fingerprint(graph: ProceduralGraph) -> int:
    """Stable content-based fingerprint of a procedural graph.

    Two graphs with identical content produce equal fingerprints;
    mutations to nodes, edges, terminal_ids, or metadata change the hash.
    The fingerprint is order-independent within each collection (sorted
    before hashing) and deterministic across Python processes.

    Used as the first element of `GuidanceCache` keys so that mutating a
    graph invalidates the cached guidance without requiring a new `id`.
    """
    nodes_part = tuple(
        sorted((nid, n.description) for nid, n in graph.nodes.items())
    )
    edges_part = tuple(sorted(
        (
            e.src,
            e.dst,
            e.relation.value,
            e.attribute.condition,
            e.attribute.guidance,
            e.attribute.pitfalls,
        )
        for e in graph.edges
    ))
    return hash((
        nodes_part,
        edges_part,
        frozenset(graph.terminal_ids),
        tuple(sorted(graph.metadata.items())),
    ))


@dataclass(frozen=True, slots=True)
class AgentState:
    """State passed to a Solver at each step.

    Attributes:
        query: the original user query, unchanged across the run.
        trajectory: ((action, observation), ...) tuples in order.
        context: pre-formatted context block the solver should include
            when constructing its prompt. methodos injects the procedural
            guidance here; hosts may append tool schemas or other context.
    """
    query: str
    trajectory: tuple[tuple[str, str], ...]
    context: str


@runtime_checkable
class Solver(Protocol):
    """Contract for the host agent's decision function.

    Any object with an async `step(state) -> str` method that returns the
    next action name satisfies this Protocol. methodos does not subclass
    the solver; it composes with it.
    """

    async def step(self, state: AgentState) -> str: ...


class GuidanceCache:
    """LRU cache for generated guidance.

    Keyed by `(hash(subgraph), last_action, last_obs)`. The subgraph hash
    is content-based (see `graph_content_fingerprint`), so mutations to
    the graph invalidate the cache automatically. Defaults to 256 entries
    with FIFO eviction on overflow.
    """

    def __init__(self, max_size: int = 256) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._store: OrderedDict[tuple[int, str, str], str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[int, str, str]) -> str | None:
        """Return cached guidance or `None`; bump LRU position on hit."""
        if key not in self._store:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def put(self, key: tuple[int, str, str], value: str) -> None:
        """Insert; evict the least-recently-used entry if over capacity."""
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop all entries (does not reset hit/miss counters)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class PGAdapter:
    """Wrap a Solver with on-demand procedural guidance.

    Args:
        solver: any object satisfying the `Solver` Protocol.
        graph: the (frozen) procedural graph to navigate.
        llm: backend used to translate subgraphs into guidance paragraphs.
        guidance_hops: neighborhood radius (paper §3.2 default 2).
        trajectory_window: number of recent trajectory steps to include
            in the guidance prompt (default 3).
        cache: optional `GuidanceCache`; defaults to a 256-entry LRU.
    """

    def __init__(
        self,
        *,
        solver: Solver,
        graph: ProceduralGraph,
        llm: LLMClient,
        guidance_hops: int = 2,
        trajectory_window: int = 3,
        cache: GuidanceCache | None = None,
    ) -> None:
        if guidance_hops < 0:
            raise ValueError(f"guidance_hops must be non-negative, got {guidance_hops}")
        if trajectory_window < 0:
            raise ValueError(
                f"trajectory_window must be non-negative, got {trajectory_window}"
            )
        self._solver = solver
        self._graph = graph
        self._llm = llm
        self._guidance_hops = guidance_hops
        self._trajectory_window = trajectory_window
        self._cache: GuidanceCache = cache if cache is not None else GuidanceCache()

    @property
    def graph(self) -> ProceduralGraph:
        """The (frozen) graph this adapter navigates. Read-only reference."""
        return self._graph

    @property
    def cache(self) -> GuidanceCache:
        """Direct access to the guidance cache (for tests and observability)."""
        return self._cache

    def cache_key(
        self, subgraph: ProceduralGraph, last_action: str, last_obs: str
    ) -> tuple[int, str, str]:
        """Compute the cache key for a (subgraph, last_action, last_obs) tuple.

        The subgraph hash is content-based, so mutations to the subgraph
        automatically invalidate any cached guidance for it.
        """
        return (graph_content_fingerprint(subgraph), last_action, last_obs)

    async def step(self, *, query: str, trajectory: list[tuple[str, str]]) -> str:
        """Run one agent step: guidance → solver.

        Implements paper Eq. 2: locate the active node via `match_node`,
        extract its h-hop neighborhood, generate the guidance paragraph
        (cached on `(hash(subgraph), last_action, last_obs)`), then call the
        solver with an `AgentState` carrying the formatted context.

        On a `match_node` miss, the full graph is used as fallback (paper
        §3.2). When `trajectory` is empty, the synthetic action `"Start"`
        is used as the lookup key.
        """
        last_action, last_obs = self._tail_key(trajectory)
        node_id = match_node(last_action, self._graph.nodes)
        sub = (
            neighborhood(self._graph, node_id, h=self._guidance_hops)
            if node_id is not None
            else self._graph
        )

        key = self.cache_key(sub, last_action, last_obs)
        cached = self._cache.get(key)
        if cached is not None:
            guidance = cached
        else:
            guidance = await generate_guidance(
                llm=self._llm,
                graph=sub,
                query=query,
                trajectory=trajectory,
                window=self._trajectory_window,
            )
            self._cache.put(key, guidance)

        state = AgentState(
            query=query,
            trajectory=tuple(trajectory),
            context=f"### PROCEDURAL GUIDANCE\n{guidance}",
        )
        action = await self._solver.step(state)
        logger.debug("PGAdapter step: action=%r", action)
        return action

    @staticmethod
    def _tail_key(trajectory: list[tuple[str, str]]) -> tuple[str, str]:
        """Last (action, observation) pair; synthetic Start on empty."""
        if not trajectory:
            return ("Start", "")
        return trajectory[-1]


__all__ = [
    "AgentState",
    "GuidanceCache",
    "PGAdapter",
    "Solver",
    "graph_content_fingerprint",
]
