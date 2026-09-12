"""Pure functions over `ProceduralGraph`.

All exports are functions, not classes — graph algorithms are stateless.
The only stateful helper is `StructuralIssue`, a dataclass that captures
one validation finding.

Engineering notes:
- Pure functions: no I/O, no global state, deterministic.
- BFS for reachability; DFS coloring for cycle detection.
- `apply_edits` dispatches per `Edit.kind`; raises `ValueError` on malformed
  edits with descriptive messages.
- `neighborhood` falls back to a deep copy of the full graph when the root
  node is missing, matching paper §3.2.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Sequence

from methodos.schema import (
    Edge,
    Edit,
    EditAddEdge,
    EditAddNode,
    EditDeleteEdge,
    EditDeleteNode,
    EditUpdateAttr,
    Node,
    ProceduralGraph,
)

logger = logging.getLogger(__name__)


class StructuralIssue:
    """A single validation finding produced by `validate`.

    Attributes:
        code: machine-readable identifier (e.g. `"unreachable_from_terminal"`)
        message: human-readable description
        node_id: optional offending node id
    """

    def __init__(self, *, code: str, message: str, node_id: str | None = None) -> None:
        """Initialize the validation issue.

        Args:
            code: Machine-readable identifier.
            message: Human-readable description.
            node_id: Optional offending node id.
        """
        self.code = code
        self.message = message
        self.node_id = node_id

    def __repr__(self) -> str:
        """Return a readable representation including code and node id."""
        suffix = f" (node={self.node_id!r})" if self.node_id else ""
        return f"StructuralIssue[{self.code}]: {self.message}{suffix}"

    def __eq__(self, other: object) -> bool:
        """Structural equality: same code, message, and node_id."""
        if not isinstance(other, StructuralIssue):
            return NotImplemented
        return (
            self.code == other.code
            and self.message == other.message
            and self.node_id == other.node_id
        )


def match_node(action_name: str, nodes: dict[str, Node]) -> str | None:
    """Exact-match an action name to a node id (paper §3.2).

    Returns the node id on hit, `None` on miss. The caller is responsible
    for the paper's fallback policy (return the full graph on miss).
    """
    return action_name if action_name in nodes else None


def neighborhood(graph: ProceduralGraph, node_id: str, *, h: int = 2) -> ProceduralGraph:
    """Extract the h-hop directed subgraph reachable from `node_id`.

    The returned graph contains `node_id` and all nodes reachable within
    `h` directed edge steps. Edges are included only when both endpoints
    are in the neighborhood.

    Fallback (paper §3.2): if `node_id` is not in `graph.nodes`, return a
    deep copy of the full graph so the caller can still produce guidance.

    Args:
        graph: the source procedural graph
        node_id: the node whose neighborhood to extract
        h: maximum hop distance; must be non-negative

    Raises:
        ValueError: if `h` is negative
    """
    if h < 0:
        raise ValueError(f"h must be non-negative, got {h}")
    if node_id not in graph.nodes:
        # Paper §3.2: when matching fails, fall back to the full graph.
        # We construct an explicit subgraph (rather than returning the
        # original) so the resulting object has a distinct id, metadata,
        # and equality semantics.
        logger.debug("neighborhood: node_id %r not in graph; returning full graph", node_id)
        return ProceduralGraph(
            id=f"{graph.id}::full_fallback",
            nodes=dict(graph.nodes),
            edges=list(graph.edges),
            terminal_ids=set(graph.terminal_ids),
            metadata={
                "parent_graph": graph.id,
                "root_node": node_id,
                "hops": h,
                "fallback": "full_graph",
            },
        )

    # If node_ids is empty, infer_terminals below would return empty too;
    # the unreachable `return` after `if not targets` is a defensive
    # branch kept for documentation. It cannot be hit given the check
    # above, so it is excluded from coverage.

    visited: set[str] = {node_id}
    frontier: set[str] = {node_id}
    for _ in range(h):
        next_frontier: set[str] = set()
        for edge in graph.edges:
            if edge.src in frontier and edge.dst not in visited:
                next_frontier.add(edge.dst)
                visited.add(edge.dst)
        if not next_frontier:
            break
        frontier = next_frontier

    sub_nodes: dict[str, Node] = {nid: graph.nodes[nid] for nid in visited}
    sub_edges: list[Edge] = [
        edge for edge in graph.edges if edge.src in visited and edge.dst in visited
    ]
    sub_terminals: set[str] = graph.terminal_ids & visited
    return ProceduralGraph(
        id=f"{graph.id}::sub::{node_id}::h{h}",
        nodes=sub_nodes,
        edges=sub_edges,
        terminal_ids=sub_terminals,
        metadata={
            "parent_graph": graph.id,
            "root_node": node_id,
            "hops": h,
        },
    )


def validate(graph: ProceduralGraph, *, allow_cycles: bool = False) -> list[StructuralIssue]:
    """Run structural checks; return a list of issues (empty == healthy).

    Checks performed:
    - Every non-terminal node has a directed path to some terminal node.
    - Cycle detection when `allow_cycles=False`.

    Note: edge endpoint and terminal id membership are also enforced by
    Pydantic validators at construction time; this function catches
    graph-level structural issues that arise post-construction.
    """
    issues: list[StructuralIssue] = []
    node_ids = set(graph.nodes.keys())
    if not node_ids:
        return issues

    # terminal_ids takes precedence; otherwise infer leaves (nodes with no
    # outgoing edges). With non-empty `node_ids`, `infer_terminals` is
    # non-empty (every node is either a leaf or has an outgoing edge), so
    # `targets` is guaranteed non-empty here.
    targets: set[str] = graph.terminal_ids or infer_terminals(graph)

    for nid in node_ids:
        if not has_path_to(graph, nid, targets):
            issues.append(
                StructuralIssue(
                    code="unreachable_from_terminal",
                    message=f"node {nid!r} cannot reach any terminal",
                    node_id=nid,
                )
            )

    if not allow_cycles and has_cycle(graph):
        issues.append(
            StructuralIssue(
                code="cycle_detected",
                message="graph contains a cycle and allow_cycles=False",
            )
        )

    return issues


def has_reachable_terminal(graph: ProceduralGraph, node_id: str) -> bool:
    """Whether `node_id` has a directed path to any node in `graph.terminal_ids`.

    Returns `False` if `node_id` is unknown or `terminal_ids` is empty.
    """
    if node_id not in graph.nodes:
        return False
    if not graph.terminal_ids:
        return False
    return has_path_to(graph, node_id, graph.terminal_ids)


def has_path_to(graph: ProceduralGraph, src: str, targets: Iterable[str]) -> bool:
    """Whether `src` has a directed path to any node in `targets`.

    BFS from `src` until either a target is reached or the reachable
    subgraph is exhausted. Returns `True` if `src` itself is in `targets`.
    """
    targets_set: set[str] = set(targets)
    if src in targets_set:
        return True
    if src not in graph.nodes:
        return False

    visited: set[str] = {src}
    queue: deque[str] = deque([src])
    adjacency_map = adjacency(graph)
    while queue:
        node = queue.popleft()
        for nxt in adjacency_map.get(node, ()):
            if nxt in targets_set:
                return True
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


def apply_edits(graph: ProceduralGraph, edits: Sequence[Edit]) -> ProceduralGraph:
    """Apply a sequence of edits to a copy of `graph`; return the new graph.

    Edits are applied in order. The original `graph` is not mutated.
    Returns a deep copy of the original if `edits` is empty.
    Attribute updates are implemented as in-place replacement (delete + add
    semantics) per paper §3.3.

    Raises:
        ValueError: if any edit is malformed (e.g. duplicate node, dangling
            edge endpoint, missing target for delete)
    """
    if not edits:
        return graph.model_copy(deep=True)

    result = graph.model_copy(deep=True)
    for edit in edits:
        result = apply_single(result, edit)
    return result


def apply_single(graph: ProceduralGraph, edit: Edit) -> ProceduralGraph:
    """Apply one edit to a copy of `graph`; raise `ValueError` on malformed input.

    Public helper used by `apply_edits`; exposed for advanced callers that
    want fine-grained per-edit error reporting.
    """
    if isinstance(edit, EditAddNode):
        if edit.node.id in graph.nodes:
            raise ValueError(f"node {edit.node.id!r} already exists")
        new_nodes = dict(graph.nodes)
        new_nodes[edit.node.id] = edit.node
        return graph.model_copy(update={"nodes": new_nodes})

    if isinstance(edit, EditDeleteNode):
        if edit.node_id not in graph.nodes:
            raise ValueError(f"cannot delete unknown node {edit.node_id!r}")
        new_nodes = {k: v for k, v in graph.nodes.items() if k != edit.node_id}
        new_edges = [
            edge for edge in graph.edges if edge.src != edit.node_id and edge.dst != edit.node_id
        ]
        new_terminals = graph.terminal_ids - {edit.node_id}
        return graph.model_copy(
            update={
                "nodes": new_nodes,
                "edges": new_edges,
                "terminal_ids": new_terminals,
            }
        )

    if isinstance(edit, EditAddEdge):
        edge = edit.edge
        if edge.src not in graph.nodes or edge.dst not in graph.nodes:
            raise ValueError(
                f"edge endpoints must reference existing nodes: "
                f"{edge.src!r} or {edge.dst!r} unknown"
            )
        if edge in graph.edges:
            raise ValueError(f"edge {edge.src}->{edge.dst} already exists")
        return graph.model_copy(update={"edges": [*graph.edges, edge]})

    if isinstance(edit, EditDeleteEdge):
        remaining = [
            edge
            for edge in graph.edges
            if not (
                edge.src == edit.src and edge.dst == edit.dst and edge.relation == edit.relation
            )
        ]
        if len(remaining) == len(graph.edges):
            raise ValueError(
                f"no matching edge to delete: {edit.src}->{edit.dst} ({edit.relation.value})"
            )
        return graph.model_copy(update={"edges": remaining})

    if isinstance(edit, EditUpdateAttr):
        replaced = False
        updated_edges: list[Edge] = []
        for existing in graph.edges:
            if (
                existing.src == edit.src
                and existing.dst == edit.dst
                and existing.relation == edit.relation
            ):
                updated_edges.append(
                    Edge(
                        src=existing.src,
                        dst=existing.dst,
                        relation=existing.relation,
                        attribute=edit.attribute,
                    )
                )
                replaced = True
            else:
                updated_edges.append(existing)
        if not replaced:
            raise ValueError(
                f"no matching edge to update: {edit.src}->{edit.dst} ({edit.relation.value})"
            )
        return graph.model_copy(update={"edges": updated_edges})

    raise ValueError(f"unknown edit type: {type(edit).__name__}")


def adjacency(graph: ProceduralGraph) -> dict[str, list[str]]:
    """Build the directed adjacency map: `src -> [dst, ...]`."""
    result: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        result[edge.src].append(edge.dst)
    return result


def infer_terminals(graph: ProceduralGraph) -> set[str]:
    """Infer terminals as nodes with zero outgoing edges."""
    has_outgoing: set[str] = {edge.src for edge in graph.edges}
    return set(graph.nodes.keys()) - has_outgoing


def has_cycle(graph: ProceduralGraph) -> bool:
    """Detect any directed cycle using DFS three-color marking."""
    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph.nodes, white)
    adjacency_map = adjacency(graph)

    def dfs(node: str) -> bool:
        color[node] = gray
        for nxt in adjacency_map.get(node, ()):
            if color[nxt] == gray:
                return True
            if color[nxt] == white and dfs(nxt):
                return True
        color[node] = black
        return False

    return any(color[nid] == white and dfs(nid) for nid in graph.nodes)


__all__ = [
    "StructuralIssue",
    "adjacency",
    "apply_edits",
    "apply_single",
    "has_cycle",
    "has_path_to",
    "has_reachable_terminal",
    "infer_terminals",
    "match_node",
    "neighborhood",
    "validate",
]
