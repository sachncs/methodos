"""methodos — self-evolving procedural graph adapter for LLM agents.

Re-exports the entire public API documented in ``docs/api.md``:

- Data models (schema.py): Relation, Attribute, Node, Edge, ProceduralGraph,
  Edit + its five variants.
- LLM client (llm.py): LLMClient Protocol + LiteLLMClient.
- Adapter (adapter.py): AgentState, Solver Protocol, GuidanceCache, PGAdapter.
- Evolution (evolution.py): RolloutResult, RejectionMemory, EvolutionEngine,
  REFINER_SYSTEM_PROMPT, TERMINATE_SUCCESS, TERMINATE_FAILURE,
  mean_score, score, tail_concat, execute_action_stub, run_rollout,
  propose_edits, validate_candidate.
- Persistence (repo.py): Task, Trajectory, ScoredMatch, NoOpVectorIndex,
  FilesystemRepository, SQLiteRepository, SqliteVecIndex, build_repository,
  tail_tokens.
- Guidance (guidance.py): GUIDANCE_SYSTEM_PROMPT.
- Graph (graph.py): StructuralIssue.
"""
from __future__ import annotations

from methodos.adapter import (
    AgentState,
    GuidanceCache,
    PGAdapter,
    Solver,
)
from methodos.evolution import (
    REFINER_SYSTEM_PROMPT,
    TERMINATE_FAILURE,
    TERMINATE_SUCCESS,
    EvolutionEngine,
    RejectionMemory,
    RolloutResult,
    execute_action_stub,
    mean_score,
    propose_edits,
    run_rollout,
    score,
    tail_concat,
    validate_candidate,
)
from methodos.graph import StructuralIssue
from methodos.guidance import GUIDANCE_SYSTEM_PROMPT, generate_guidance
from methodos.llm import LiteLLMClient, LLMClient, LLMError
from methodos.repo import (
    FilesystemRepository,
    NoOpVectorIndex,
    ScoredMatch,
    SQLiteRepository,
    SqliteVecIndex,
    Task,
    Trajectory,
    VectorIndex,
    build_repository,
)
from methodos.schema import (
    Attribute,
    Edge,
    Edit,
    EditAddEdge,
    EditAddNode,
    EditDeleteEdge,
    EditUpdateAttr,
    Node,
    ProceduralGraph,
    Relation,
)

__version__ = "0.1.0"

__all__ = [
    "GUIDANCE_SYSTEM_PROMPT",
    "REFINER_SYSTEM_PROMPT",
    "TERMINATE_FAILURE",
    "TERMINATE_SUCCESS",
    "AgentState",
    "Attribute",
    "Edge",
    "Edit",
    "EditAddEdge",
    "EditAddNode",
    "EditDeleteEdge",
    "EditUpdateAttr",
    "EvolutionEngine",
    "FilesystemRepository",
    "GuidanceCache",
    "LLMClient",
    "LLMError",
    "LiteLLMClient",
    "NoOpVectorIndex",
    "Node",
    "PGAdapter",
    "ProceduralGraph",
    "RejectionMemory",
    "Relation",
    "RolloutResult",
    "SQLiteRepository",
    "ScoredMatch",
    "Solver",
    "SqliteVecIndex",
    "StructuralIssue",
    "Task",
    "Trajectory",
    "VectorIndex",
    "__version__",
    "build_repository",
    "execute_action_stub",
    "generate_guidance",
    "mean_score",
    "propose_edits",
    "run_rollout",
    "score",
    "tail_concat",
    "validate_candidate",
]
