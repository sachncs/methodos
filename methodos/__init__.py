"""methodos — self-evolving procedural graph adapter for LLM agents."""

from __future__ import annotations

from methodos.adapter import AgentState, GuidanceCache, PGAdapter, Solver
from methodos.llm import LiteLLMClient, LLMClient
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
    "AgentState",
    "Attribute",
    "Edge",
    "Edit",
    "EditAddEdge",
    "EditAddNode",
    "EditDeleteEdge",
    "EditDeleteNode",
    "EditUpdateAttr",
    "GuidanceCache",
    "LLMClient",
    "LiteLLMClient",
    "Node",
    "PGAdapter",
    "ProceduralGraph",
    "Relation",
    "Solver",
    "__version__",
]
