"""Tests for `methodos.graph` pure functions."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from methodos.graph import (
    StructuralIssue,
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
from methodos.schema import (
    Attribute,
    Edge,
    Edit,
    EditAddEdge,
    EditAddNode,
    EditDeleteEdge,
    EditDeleteNode,
    EditUpdateAttr,
    Node,
    ProceduralGraph,
    Relation,
)


def _attr() -> Attribute:
    """Standard edge attribute for tests."""
    return Attribute(condition="c", guidance="g", pitfalls="p")


def _edge(src: str, dst: str, *, relation: Relation = Relation.LEADS_TO) -> Edge:
    """Edge helper."""
    return Edge(src=src, dst=dst, relation=relation, attribute=_attr())


class TestMatchNode:
    """`match_node` exact-string matching."""

    def test_hit_returns_node_id(self) -> None:
        nodes = {"foo": Node(id="foo"), "bar": Node(id="bar")}
        assert match_node("foo", nodes) == "foo"

    def test_miss_returns_none(self) -> None:
        nodes = {"foo": Node(id="foo")}
        assert match_node("quux", nodes) is None

    def test_empty_nodes_returns_none(self) -> None:
        assert match_node("foo", {}) is None

    def test_hit_on_empty_string_id_rejected_at_construction(self) -> None:
        # Node ids must be non-empty per Pydantic validation, so this
        # # state cannot be constructed directly. Verify the contract.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Node(id="")


class TestNeighborhood:
    """`neighborhood` BFS extraction with fallback."""

    def test_zero_hops_returns_only_root(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )
        sub = neighborhood(g, "a", h=0)
        assert set(sub.nodes.keys()) == {"a"}
        assert sub.edges == []

    def test_one_hop_includes_root_and_neighbors(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={
                "a": Node(id="a"),
                "b": Node(id="b"),
                "c": Node(id="c"),
            },
            edges=[_edge("a", "b"), _edge("b", "c")],
            terminal_ids={"c"},
        )
        sub = neighborhood(g, "a", h=1)
        assert set(sub.nodes.keys()) == {"a", "b"}
        assert [e.src for e in sub.edges] == ["a"]

    def test_two_hops_default(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={
                "a": Node(id="a"),
                "b": Node(id="b"),
                "c": Node(id="c"),
                "d": Node(id="d"),
            },
            edges=[
                _edge("a", "b"),
                _edge("b", "c"),
                _edge("c", "d"),
            ],
            terminal_ids={"d"},
        )
        sub = neighborhood(g, "a", h=2)
        assert set(sub.nodes.keys()) == {"a", "b", "c"}

    def test_bfs_does_not_revisit(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("a", "c"), _edge("b", "c")],
            terminal_ids={"c"},
        )
        sub = neighborhood(g, "a", h=2)
        assert set(sub.nodes.keys()) == {"a", "b", "c"}

    def test_unknown_node_falls_back_to_full_graph(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )
        sub = neighborhood(g, "missing", h=2)
        assert set(sub.nodes.keys()) == {"a", "b"}
        assert sub.id == "g::full_fallback"
        assert sub.metadata["fallback"] == "full_graph"

    def test_negative_h_rejected(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")}, terminal_ids={"a"})
        with pytest.raises(ValueError, match="h must be non-negative"):
            neighborhood(g, "a", h=-1)

    def test_returns_typed_procedural_graph(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")}, terminal_ids={"a"})
        sub = neighborhood(g, "a", h=0)
        assert isinstance(sub, ProceduralGraph)
        assert sub.schema_version == 1

    def test_metadata_preserves_parent(self) -> None:
        g = ProceduralGraph(id="parent", nodes={"a": Node(id="a")}, terminal_ids={"a"})
        sub = neighborhood(g, "a", h=0)
        assert sub.metadata["parent_graph"] == "parent"
        assert sub.metadata["root_node"] == "a"
        assert sub.metadata["hops"] == 0

    def test_terminals_intersected(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"a", "b"},
        )
        sub = neighborhood(g, "b", h=0)
        # Only "b" is in the neighborhood; "a" should not appear in sub.terminal_ids
        assert sub.terminal_ids == {"b"}


class TestValidate:
    """`validate` structural checks."""

    def test_healthy_graph_no_issues(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )
        assert validate(g) == []

    def test_empty_graph_no_issues(self) -> None:
        g = ProceduralGraph(id="g")
        assert validate(g) == []

    def test_unreachable_node_reported(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "orphan": Node(id="orphan")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )
        issues = validate(g)
        codes = [i.code for i in issues]
        assert "unreachable_from_terminal" in codes
        orphan_issues = [i for i in issues if i.node_id == "orphan"]
        assert len(orphan_issues) == 1

    def test_cycle_detected_when_disallowed(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b"), _edge("b", "a")],
            terminal_ids={"a"},
        )
        issues = validate(g, allow_cycles=False)
        assert any(i.code == "cycle_detected" for i in issues)

    def test_cycle_allowed_when_allowed(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b"), _edge("b", "a")],
            terminal_ids={"a"},
        )
        issues = validate(g, allow_cycles=True)
        assert not any(i.code == "cycle_detected" for i in issues)

    def test_inferred_terminals_when_explicit_empty(self) -> None:
        # terminal_ids empty; leaves (nodes with no outgoing edges) are inferred.
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids=set(),
        )
        # b is the leaf → inferred terminal; a reaches b → no issues.
        assert validate(g) == []

    def test_no_terminals_no_edges_returns_empty(self) -> None:
        # Isolated nodes with no edges: no inferred terminals → return early.
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[],
            terminal_ids=set(),
        )
        assert validate(g) == []


class TestHasPathTo:
    """`has_path_to` BFS reachability."""

    def test_src_is_target(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")})
        assert has_path_to(g, "a", {"a"})

    def test_direct_neighbor(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
        )
        assert has_path_to(g, "a", {"b"})

    def test_indirect_path(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("b", "c")],
        )
        assert has_path_to(g, "a", {"c"})

    def test_no_path(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b")],
        )
        assert not has_path_to(g, "c", {"a"})

    def test_unknown_src_returns_false(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")})
        assert not has_path_to(g, "missing", {"a"})

    def test_multi_target_first_hit(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("b", "c")],
        )
        assert has_path_to(g, "a", {"c", "b"})

    def test_handles_cycle(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("b", "a"), _edge("a", "c")],
        )
        assert has_path_to(g, "a", {"c"})

    def test_skips_revisited_nodes(self) -> None:
        # Diamond: a -> b, a -> c, b -> d, c -> d. BFS revisits d via two paths.
        g = ProceduralGraph(
            id="g",
            nodes={
                "a": Node(id="a"), "b": Node(id="b"),
                "c": Node(id="c"), "d": Node(id="d"),
            },
            edges=[
                _edge("a", "b"),
                _edge("a", "c"),
                _edge("b", "d"),
                _edge("c", "d"),
            ],
        )
        assert has_path_to(g, "a", {"d"})


class TestHasReachableTerminal:
    """`has_reachable_terminal` thin wrapper."""

    def test_true_when_terminal_reachable(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )
        assert has_reachable_terminal(g, "a") is True

    def test_false_when_no_terminal_declared(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a")},
            terminal_ids=set(),
        )
        assert has_reachable_terminal(g, "a") is False

    def test_false_when_unknown_node(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")}, terminal_ids={"a"})
        assert has_reachable_terminal(g, "missing") is False


class TestApplyEdits:
    """`apply_edits` dispatch across all five Edit variants."""

    def _base(self) -> ProceduralGraph:
        return ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )

    def test_empty_returns_deep_copy(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [])
        assert new_g == g
        assert new_g is not g
        assert new_g.nodes is not g.nodes

    def test_add_node(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [EditAddNode(node=Node(id="c", description="new"))])
        assert "c" in new_g.nodes
        assert g == self._base()  # original untouched

    def test_add_node_duplicate_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="already exists"):
            apply_edits(g, [EditAddNode(node=Node(id="a"))])

    def test_delete_node_removes_incident_edges(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [EditDeleteNode(node_id="a")])
        assert "a" not in new_g.nodes
        assert new_g.edges == []
        # b is still a terminal (it was declared as such); "a" is removed
        # from terminals (it was the non-terminal node).
        assert new_g.terminal_ids == {"b"}

    def test_delete_node_unknown_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="cannot delete unknown"):
            apply_edits(g, [EditDeleteNode(node_id="missing")])

    def test_add_edge(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [EditAddEdge(edge=_edge("b", "a"))])
        assert any(e.src == "b" and e.dst == "a" for e in new_g.edges)

    def test_add_edge_dangling_src_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="edge endpoints"):
            apply_edits(g, [EditAddEdge(edge=_edge("missing", "a"))])

    def test_add_edge_dangling_dst_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="edge endpoints"):
            apply_edits(g, [EditAddEdge(edge=_edge("a", "missing"))])

    def test_add_edge_duplicate_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="already exists"):
            apply_edits(g, [EditAddEdge(edge=_edge("a", "b"))])

    def test_delete_edge(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [EditDeleteEdge(
            src="a", dst="b", relation=Relation.LEADS_TO,
        )])
        assert new_g.edges == []

    def test_delete_edge_no_match_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="no matching edge to delete"):
            apply_edits(g, [EditDeleteEdge(
                src="a", dst="b", relation=Relation.REPLACES,
            )])

    def test_update_attr(self) -> None:
        g = self._base()
        new_attr = Attribute(condition="c2", guidance="g2", pitfalls="p2")
        new_g = apply_edits(g, [EditUpdateAttr(
            src="a", dst="b", relation=Relation.LEADS_TO, attribute=new_attr,
        )])
        assert new_g.edges[0].attribute == new_attr

    def test_update_attr_no_match_raises(self) -> None:
        g = self._base()
        with pytest.raises(ValueError, match="no matching edge to update"):
            apply_edits(g, [EditUpdateAttr(
                src="a", dst="b", relation=Relation.REPLACES,
                attribute=_attr(),
            )])

    def test_unknown_edit_type_raises(self) -> None:
        """A non-Edit object passed in raises ValueError from the dispatch fallback."""
        g = self._base()

        class NotAnEdit:
            kind = "add_node_with_cheese"

        with pytest.raises(ValueError, match="unknown edit type"):
            apply_edits(g, [NotAnEdit()])  # type: ignore[list-item]

    def test_sequential_edits_applied_in_order(self) -> None:
        g = self._base()
        new_g = apply_edits(g, [
            EditAddNode(node=Node(id="c")),
            EditAddEdge(edge=_edge("c", "a")),
        ])
        assert "c" in new_g.nodes
        assert any(e.src == "c" for e in new_g.edges)


class TestApplySingleEdit:
    """`apply_single_edit` per-variant dispatch."""

    def _base(self) -> ProceduralGraph:
        return ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
            terminal_ids={"b"},
        )

    def test_add_node(self) -> None:
        new_g = apply_single_edit(self._base(), EditAddNode(node=Node(id="c")))
        assert "c" in new_g.nodes

    def test_delete_node(self) -> None:
        new_g = apply_single_edit(self._base(), EditDeleteNode(node_id="a"))
        assert "a" not in new_g.nodes

    def test_add_edge(self) -> None:
        new_g = apply_single_edit(self._base(), EditAddEdge(edge=_edge("b", "a")))
        assert any(e.src == "b" for e in new_g.edges)

    def test_delete_edge(self) -> None:
        new_g = apply_single_edit(self._base(), EditDeleteEdge(
            src="a", dst="b", relation=Relation.LEADS_TO,
        ))
        assert new_g.edges == []

    def test_update_attr(self) -> None:
        new_attr = Attribute(condition="x", guidance="y", pitfalls="z")
        new_g = apply_single_edit(self._base(), EditUpdateAttr(
            src="a", dst="b", relation=Relation.LEADS_TO, attribute=new_attr,
        ))
        assert new_g.edges[0].attribute == new_attr


class TestAdjacency:
    """`adjacency` builds a directed adjacency map."""

    def test_empty(self) -> None:
        g = ProceduralGraph(id="g")
        assert adjacency(g) == {}

    def test_single_edge(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
        )
        adj = adjacency(g)
        assert adj == {"a": ["b"], "b": []}

    def test_multi_edges(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("a", "c")],
        )
        adj = adjacency(g)
        assert sorted(adj["a"]) == ["b", "c"]


class TestInferTerminals:
    """`infer_terminals` returns nodes with zero outgoing edges."""

    def test_simple(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b")],
        )
        assert infer_terminals(g) == {"b"}

    def test_two_leaves(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("a", "c")],
        )
        assert infer_terminals(g) == {"b", "c"}

    def test_no_edges_all_terminal(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
        )
        assert infer_terminals(g) == {"a", "b"}


class TestHasCycle:
    """`has_cycle` cycle detection via DFS coloring."""

    def test_acyclic_no_cycle(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("b", "c")],
        )
        assert has_cycle(g) is False

    def test_direct_cycle(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[_edge("a", "b"), _edge("b", "a")],
        )
        assert has_cycle(g) is True

    def test_three_node_cycle(self) -> None:
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b"), "c": Node(id="c")},
            edges=[_edge("a", "b"), _edge("b", "c"), _edge("c", "a")],
        )
        assert has_cycle(g) is True

    def test_empty_no_cycle(self) -> None:
        g = ProceduralGraph(id="g")
        assert has_cycle(g) is False

    def test_isolated_node_no_cycle(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")})
        assert has_cycle(g) is False


class TestStructuralIssue:
    """`StructuralIssue` dataclass equality and repr."""

    def test_repr_with_node_id(self) -> None:
        issue = StructuralIssue(code="x", message="y", node_id="z")
        assert "x" in repr(issue)
        assert "y" in repr(issue)
        assert "z" in repr(issue)

    def test_repr_without_node_id(self) -> None:
        issue = StructuralIssue(code="x", message="y")
        assert "x" in repr(issue)
        assert "y" in repr(issue)
        assert "node=" not in repr(issue)

    def test_equality(self) -> None:
        a = StructuralIssue(code="x", message="y", node_id="z")
        b = StructuralIssue(code="x", message="y", node_id="z")
        assert a == b

    def test_inequality(self) -> None:
        a = StructuralIssue(code="x", message="y")
        b = StructuralIssue(code="x", message="different")
        assert a != b

    def test_inequality_with_non_issue(self) -> None:
        issue = StructuralIssue(code="x", message="y")
        assert issue != "string"
        assert issue != 42
        assert issue != None  # noqa: E711


# ----------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ----------------------------------------------------------------------------

# Strategy: build a small acyclic graph
node_id_st = st.text(
    alphabet=st.characters(
        whitelist_categories=["L", "N"], max_codepoint=0x7E
    ),
    min_size=1,
    max_size=8,
).filter(lambda s: not s.startswith("."))


@st.composite
def small_acyclic_graph(draw: st.DrawFn) -> ProceduralGraph:
    """Generate a small acyclic graph with 1-5 nodes."""
    n = draw(st.integers(min_value=1, max_value=5))
    node_ids = draw(
        st.lists(
            node_id_st,
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    nodes = {nid: Node(id=nid) for nid in node_ids}
    edges: list[Edge] = []
    # Linear chain: node[0] -> node[1] -> node[2] -> ...
    for i in range(len(node_ids) - 1):
        edges.append(Edge(
            src=node_ids[i],
            dst=node_ids[i + 1],
            relation=Relation.LEADS_TO,
            attribute=Attribute(condition="c", guidance="g", pitfalls="p"),
        ))
    # last node is the terminal
    return ProceduralGraph(
        id="gen",
        nodes=nodes,
        edges=edges,
        terminal_ids={node_ids[-1]},
    )


class TestProperties:
    """Property-based invariants via hypothesis."""

    @given(small_acyclic_graph())
    @settings(max_examples=20, deadline=None)
    def test_acyclic_graph_has_no_cycle(self, g: ProceduralGraph) -> None:
        assert has_cycle(g) is False

    @given(small_acyclic_graph())
    @settings(max_examples=20, deadline=None)
    def test_first_node_reaches_terminal(self, g: ProceduralGraph) -> None:
        first = next(iter(g.nodes))
        terminal = next(iter(g.terminal_ids))
        assert has_path_to(g, first, {terminal})

    @given(small_acyclic_graph())
    @settings(max_examples=20, deadline=None)
    def test_validate_clean_on_acyclic(self, g: ProceduralGraph) -> None:
        assert validate(g) == []

    @given(small_acyclic_graph())
    @settings(max_examples=20, deadline=None)
    def test_apply_empty_edits_idempotent(self, g: ProceduralGraph) -> None:
        assert apply_edits(g, []) == g

    @given(small_acyclic_graph(), node_id_st)
    @settings(max_examples=20, deadline=None)
    def test_neighborhood_root_in_subgraph(self, g: ProceduralGraph, root: str) -> None:
        if root not in g.nodes:
            return
        sub = neighborhood(g, root, h=3)
        assert root in sub.nodes
