"""Pydantic v2 data models for procedural graphs.

This module is the single source of truth for all data shapes that flow
through methodos. It contains no behavior — only structure. Every public
class is a Pydantic model or a `str`-based Enum.

Engineering notes:
- All models use `ConfigDict(extra="forbid")` to reject unknown fields.
- All models with string fields use `min_length`/`max_length` constraints.
- `validate_assignment=True` is set on `ProceduralGraph` so that
  attribute reassignment re-runs validators.
- Cross-field validation (terminal_ids ∈ node_ids, edge endpoints ∈
  node_ids) lives on `ProceduralGraph` via `model_validator`.
- The Edit discriminated union has exactly five variants; adding a
  sixth requires updating `graph.apply_edits` dispatch.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Relation(StrEnum):
    """Edge relation types between procedure nodes.

    Values are stable wire format — do NOT rename. Adding new variants is
    safe; renaming or removing existing ones is a breaking change.
    """

    LEADS_TO = "leads_to"
    REQUIRES = "requires"
    REPLACES = "replaces"


class Attribute(BaseModel):
    """Textual attributes attached to every edge.

    condition: when this transition applies
    guidance: how to proceed along this transition
    pitfalls: what to avoid when taking this transition
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str = Field(min_length=1, max_length=2000)
    guidance: str = Field(min_length=1, max_length=2000)
    pitfalls: str = Field(min_length=1, max_length=2000)


class Node(BaseModel):
    """A procedure node — an abstract action, reasoning step, skill, or task state.

    Node ids must match `^[a-zA-Z0-9_\\-\\.]+$` and must NOT start with `.`.
    The leading-dot restriction prevents accidental id collisions with
    metadata keys when nodes are flattened for persistence.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    description: str = Field(default="", max_length=2000)

    @field_validator("id")
    @classmethod
    def id_must_not_start_with_dot(cls, value: str) -> str:
        """Reject ids starting with '.' to avoid serialization collisions."""
        if value.startswith("."):
            raise ValueError("node id cannot start with '.'")
        return value


class Edge(BaseModel):
    """A directed, attributed triplet between two procedure nodes."""

    model_config = ConfigDict(extra="forbid")

    src: str = Field(min_length=1, max_length=128)
    dst: str = Field(min_length=1, max_length=128)
    relation: Relation
    attribute: Attribute

    @model_validator(mode="after")
    def endpoints_must_be_distinct_validator(self) -> Edge:
        """Self-loops are not allowed; they create ambiguous guidance."""
        if self.src == self.dst:
            raise ValueError("edge endpoints must be distinct")
        return self


class ProceduralGraph(BaseModel):
    """An editable, attributable procedural graph.

    Invariants enforced by model validators:
    - All `terminal_ids` refer to existing nodes.
    - All `edges[].src` and `edges[].dst` refer to existing nodes.
    - `schema_version` is pinned to 1; migration runner deferred to v2.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1, max_length=128)
    schema_version: Literal[1] = 1
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    terminal_ids: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def terminals_must_be_nodes_validator(self) -> ProceduralGraph:
        """terminal_ids must reference nodes that exist."""
        missing = self.terminal_ids - set(self.nodes.keys())
        if missing:
            raise ValueError(f"terminal_ids refer to unknown nodes: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def edge_endpoints_must_be_nodes_validator(self) -> ProceduralGraph:
        """Every edge endpoint must reference an existing node."""
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.src not in node_ids:
                raise ValueError(f"edge src refers to unknown node: {edge.src!r}")
            if edge.dst not in node_ids:
                raise ValueError(f"edge dst refers to unknown node: {edge.dst!r}")
        return self


class EditAddNode(BaseModel):
    """Insert a new node. Fails if the id already exists."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["add_node"] = "add_node"
    node: Node


class EditDeleteNode(BaseModel):
    """Remove a node and all incident edges."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["delete_node"] = "delete_node"
    node_id: str = Field(min_length=1, max_length=128)


class EditAddEdge(BaseModel):
    """Insert a new edge. Fails if the exact triplet already exists."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["add_edge"] = "add_edge"
    edge: Edge


class EditDeleteEdge(BaseModel):
    """Remove a matching (src, dst, relation) triplet."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["delete_edge"] = "delete_edge"
    src: str = Field(min_length=1, max_length=128)
    dst: str = Field(min_length=1, max_length=128)
    relation: Relation


class EditUpdateAttr(BaseModel):
    """Replace the attribute on a matching (src, dst, relation) edge."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["update_attr"] = "update_attr"
    src: str = Field(min_length=1, max_length=128)
    dst: str = Field(min_length=1, max_length=128)
    relation: Relation
    attribute: Attribute


Edit = Annotated[
    EditAddNode | EditDeleteNode | EditAddEdge | EditDeleteEdge | EditUpdateAttr,
    Field(discriminator="kind"),
]


__all__ = [
    "Attribute",
    "Edge",
    "Edit",
    "EditAddEdge",
    "EditAddNode",
    "EditDeleteEdge",
    "EditDeleteNode",
    "EditUpdateAttr",
    "Node",
    "ProceduralGraph",
    "Relation",
]
