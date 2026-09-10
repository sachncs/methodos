"""methodos — self-evolving procedural graph adapter for LLM agents."""
from __future__ import annotations

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

__version__ = "0.1.0"

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
    "__version__",
]
