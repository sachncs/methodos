"""Tests for `methodos.schema` Pydantic models."""
from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

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


class TestRelation:
    """Relation is a string-valued Enum with stable wire format."""

    def test_leads_to_value(self) -> None:
        assert Relation.LEADS_TO.value == "leads_to"

    def test_requires_value(self) -> None:
        assert Relation.REQUIRES.value == "requires"

    def test_replaces_value(self) -> None:
        assert Relation.REPLACES.value == "replaces"

    def test_is_str_subclass(self) -> None:
        assert isinstance(Relation.LEADS_TO, str)
        assert isinstance(Relation.REQUIRES, str)
        assert isinstance(Relation.REPLACES, str)

    def test_membership_is_exactly_three(self) -> None:
        assert {r.value for r in Relation} == {"leads_to", "requires", "replaces"}


class TestAttribute:
    """Attribute is frozen and rejects unknown fields."""

    def test_minimum_valid(self) -> None:
        attr = Attribute(condition="x", guidance="y", pitfalls="z")
        assert attr.condition == "x"
        assert attr.guidance == "y"
        assert attr.pitfalls == "z"

    def test_frozen_blocks_assignment(self) -> None:
        attr = Attribute(condition="x", guidance="y", pitfalls="z")
        with pytest.raises(ValidationError):
            attr.condition = "new"

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="x", guidance="y", pitfalls="z", extra="nope")  # type: ignore[call-arg]

    def test_empty_condition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="", guidance="y", pitfalls="z")

    def test_empty_guidance_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="x", guidance="", pitfalls="z")

    def test_empty_pitfalls_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="x", guidance="y", pitfalls="")

    def test_oversize_condition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="x" * 2001, guidance="y", pitfalls="z")

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            Attribute(condition="x")  # type: ignore[call-arg]


class TestNode:
    """Node id validation and field constraints."""

    def test_minimum_valid(self) -> None:
        node = Node(id="abc")
        assert node.id == "abc"
        assert node.description == ""

    def test_description_default_empty(self) -> None:
        assert Node(id="a").description == ""

    def test_oversize_description_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="a", description="x" * 2001)

    def test_oversize_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="a" * 129)

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="")

    def test_id_with_spaces_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="has spaces")

    def test_id_with_dot_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id=".hidden")

    def test_id_with_slash_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="a/b")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Node(id="a", unknown="x")  # type: ignore[call-arg]


class TestEdge:
    """Edge self-loop and unknown-relation rejection."""

    def _attr(self) -> Attribute:
        return Attribute(condition="c", guidance="g", pitfalls="p")

    def test_minimum_valid(self) -> None:
        edge = Edge(src="a", dst="b", relation=Relation.LEADS_TO, attribute=self._attr())
        assert edge.src == "a"
        assert edge.dst == "b"
        assert edge.relation is Relation.LEADS_TO

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Edge(src="a", dst="a", relation=Relation.LEADS_TO, attribute=self._attr())

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Edge(
                src="a", dst="b", relation=Relation.LEADS_TO,
                attribute=self._attr(), extra="nope",  # type: ignore[call-arg]
            )


class TestProceduralGraph:
    """Graph validators and field defaults."""

    def test_minimum_valid(self) -> None:
        g = ProceduralGraph(id="g")
        assert g.schema_version == 1
        assert g.nodes == {}
        assert g.edges == []
        assert g.terminal_ids == set()
        assert g.metadata == {}

    def test_terminal_id_must_reference_node(self) -> None:
        with pytest.raises(ValidationError):
            ProceduralGraph(id="g", terminal_ids={"nonexistent"})

    def test_edge_src_must_reference_node(self) -> None:
        attr = Attribute(condition="c", guidance="g", pitfalls="p")
        with pytest.raises(ValidationError):
            ProceduralGraph(
                id="g",
                nodes={"a": Node(id="a")},
                edges=[Edge(src="unknown", dst="a", relation=Relation.LEADS_TO, attribute=attr)],
            )

    def test_edge_dst_must_reference_node(self) -> None:
        attr = Attribute(condition="c", guidance="g", pitfalls="p")
        with pytest.raises(ValidationError):
            ProceduralGraph(
                id="g",
                nodes={"a": Node(id="a")},
                edges=[Edge(src="a", dst="unknown", relation=Relation.LEADS_TO, attribute=attr)],
            )

    def test_assign_terminal_id_runs_validator(self) -> None:
        g = ProceduralGraph(id="g", nodes={"a": Node(id="a")}, terminal_ids={"a"})
        with pytest.raises(ValidationError):
            g.terminal_ids = {"missing"}

    def test_assign_node_then_break_edge_re_validates(self) -> None:
        attr = Attribute(condition="c", guidance="g", pitfalls="p")
        g = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a"), "b": Node(id="b")},
            edges=[Edge(src="a", dst="b", relation=Relation.LEADS_TO, attribute=attr)],
        )
        # Remove b by reassigning nodes — validator must catch dangling edge
        with pytest.raises(ValidationError):
            g.nodes = {"a": Node(id="a")}

    def test_schema_version_pinned(self) -> None:
        with pytest.raises(ValidationError):
            ProceduralGraph(id="g", schema_version=2)  # type: ignore[arg-type]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProceduralGraph(id="g", unknown="x")  # type: ignore[call-arg]


class TestEditUnion:
    """Edit discriminated union parsing via TypeAdapter."""

    adapter: ClassVar[TypeAdapter[Edit]] = TypeAdapter(Edit)

    def test_parse_add_node(self) -> None:
        raw = {"kind": "add_node", "node": {"id": "n1"}}
        parsed = self.adapter.validate_python(raw)
        assert isinstance(parsed, EditAddNode)
        assert parsed.node.id == "n1"

    def test_parse_delete_node(self) -> None:
        raw = {"kind": "delete_node", "node_id": "old"}
        parsed = self.adapter.validate_python(raw)
        assert isinstance(parsed, EditDeleteNode)
        assert parsed.node_id == "old"

    def test_parse_add_edge(self) -> None:
        raw = {
            "kind": "add_edge",
            "edge": {
                "src": "a", "dst": "b", "relation": "leads_to",
                "attribute": {"condition": "c", "guidance": "g", "pitfalls": "p"},
            },
        }
        parsed = self.adapter.validate_python(raw)
        assert isinstance(parsed, EditAddEdge)
        assert parsed.edge.src == "a"
        assert parsed.edge.relation is Relation.LEADS_TO

    def test_parse_delete_edge(self) -> None:
        raw = {"kind": "delete_edge", "src": "a", "dst": "b", "relation": "replaces"}
        parsed = self.adapter.validate_python(raw)
        assert isinstance(parsed, EditDeleteEdge)
        assert parsed.relation is Relation.REPLACES

    def test_parse_update_attr(self) -> None:
        raw = {
            "kind": "update_attr",
            "src": "a", "dst": "b", "relation": "leads_to",
            "attribute": {"condition": "c2", "guidance": "g2", "pitfalls": "p2"},
        }
        parsed = self.adapter.validate_python(raw)
        assert isinstance(parsed, EditUpdateAttr)
        assert parsed.attribute.condition == "c2"

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"kind": "add_node_with_cheese", "node": {"id": "x"}})

    def test_missing_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"node": {"id": "x"}})

    def test_add_node_self_loop_rejected_via_inner_node_validator(self) -> None:
        # EditAddNode has its own Node which has id validation
        raw = {"kind": "add_node", "node": {"id": ".hidden"}}
        with pytest.raises(ValidationError):
            self.adapter.validate_python(raw)

    def test_add_node_extra_rejected(self) -> None:
        raw = {"kind": "add_node", "node": {"id": "n1", "extra": "x"}}
        with pytest.raises(ValidationError):
            self.adapter.validate_python(raw)
