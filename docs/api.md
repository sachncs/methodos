# API Reference

This document covers the public API. Everything not listed here is
implementation detail and may change.

## Error envelope

Every API failure uses the same JSON shape and includes the request
correlation identifier:

```json
{
  "detail": "graph not found",
  "code": "not_found",
  "request_id": "7a6d..."
}
```

`code` is stable for clients (`invalid_request`, `unauthorized`,
`not_found`, `conflict`, `rate_limited`, `unavailable`,
`not_implemented`, `request_failed`, or `internal_error`). Validation
failures use a list in `detail`; internal errors never expose exception
messages or storage paths.

Graphs expose a non-negative `revision`. `PUT /v1/graphs/{graph_id}` uses
an integer `If-Match` header as a compare-and-swap token and returns `409`
when another writer has committed a newer revision. Clients should reload
the graph and retry their edit rather than blindly overwriting it.

When `METHODOS_TENANT_KEYS` is configured, clients authenticate with their
tenant's `X-API-Key`. The service derives the tenant from the key and scopes
graph storage to that tenant; `X-Tenant-ID`, if supplied, must match it.

## `methodos.schema` — Data Models

```python
from methodos.schema import (
    Relation,  # LEADS_TO, REQUIRES, REPLACES (StrEnum)
    Attribute,  # condition, guidance, pitfalls (all required, max 2000 chars)
    Node,  # id (pattern), description
    Edge,  # src, dst (≠ src), relation, attribute
    ProceduralGraph,  # id, revision, schema_version=1, nodes, edges, terminal_ids, metadata
    Edit,  # discriminated union over 5 variants
    EditAddNode,  # kind="add_node", node: Node
    EditDeleteNode,  # kind="delete_node", node_id: str
    EditAddEdge,  # kind="add_edge", edge: Edge
    EditDeleteEdge,  # kind="delete_edge", src, dst, relation
    EditUpdateAttr,  # kind="update_attr", src, dst, relation, attribute
)
```

All models use `ConfigDict(extra="forbid")`. `ProceduralGraph` uses
`validate_assignment=True` so attribute reassignment re-runs validators.

## `methodos.graph` — Algorithms

```python
def match_node(action_name: str, nodes: dict[str, Node]) -> str | None
def neighborhood(graph, node_id, *, h: int = 2) -> ProceduralGraph
def validate(graph, *, allow_cycles: bool = False) -> list[StructuralIssue]
def apply_edits(graph, edits: Sequence[Edit]) -> ProceduralGraph
def has_path_to(graph, src, targets) -> bool
def has_reachable_terminal(graph, node_id) -> bool
def apply_single_edit(graph, edit) -> ProceduralGraph
def adjacency(graph) -> dict[str, list[str]]
def infer_terminals(graph) -> set[str]
def has_cycle(graph) -> bool
```

`StructuralIssue` is a dataclass with `code`, `message`, `node_id`.

## `methodos.llm` — LLM Client

```python
class LLMClient(Protocol):
    async def complete(
        self, *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str: ...

class LiteLLMClient:
    def __init__(
        self, *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None: ...
    async def complete(...) -> str: ...

class LLMError(RuntimeError): ...

def is_retryable_provider_error(exc: Exception) -> bool: ...
```

## `methodos.guidance` — Guidance Generation

```python
GUIDANCE_SYSTEM_PROMPT: str  # Ψ prompt (paper §3.2)

def format_graph_for_prompt(graph: ProceduralGraph) -> str
def format_trajectory_window(trajectory: Iterable[tuple[str, str]], window: int) -> str
async def generate_guidance(
    *,
    llm: LLMClient,
    graph: ProceduralGraph,
    query: str,
    trajectory: Iterable[tuple[str, str]],
    window: int = 3,
) -> str: ...
```

## `methodos.adapter` — Runtime Adapter

```python
@dataclass(frozen=True, slots=True)
class AgentState:
    query: str
    trajectory: tuple[tuple[str, str], ...]
    context: str


class Solver(Protocol):
    async def step(self, state: AgentState) -> str: ...


class GuidanceCache:
    def __init__(self, max_size: int = 256) -> None: ...
    def get(self, key) -> str | None: ...
    def put(self, key, value) -> None: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...


class PGAdapter:
    def __init__(
        self,
        *,
        solver: Solver,
        graph: ProceduralGraph,
        llm: LLMClient,
        guidance_hops: int = 2,
        trajectory_window: int = 3,
        cache: GuidanceCache | None = None,
    ) -> None: ...
    @property
    def graph(self) -> ProceduralGraph: ...
    @property
    def cache(self) -> GuidanceCache: ...
    def cache_key(self, last_action, last_obs) -> tuple[str, str, str]: ...
    async def step(self, *, query: str, trajectory: list[tuple[str, str]]) -> str: ...
```

## `methodos.repo` — Persistence

```python
class Repository(Protocol):
    async def load_graph(self, graph_id) -> ProceduralGraph: ...
    async def save_graph(self, graph) -> None: ...
    async def snapshot(self, graph_id, tag) -> None: ...
    async def restore_snapshot(self, graph_id, tag, *, expected_revision=None): ...

class EvolutionEvent:
    graph_id: str
    round_index: int
    outcome: str
    current_score: float
    candidate_score: float | None
    edit_count: int
    created_at: float
    async def append_trajectory(self, graph_id, split, trajectory) -> None: ...
    def read_trajectories(self, graph_id, split) -> AsyncIterator[Trajectory]: ...


class VectorIndex(Protocol):
    def upsert(self, key, vector) -> None: ...
    def query(self, vector, k) -> list[ScoredMatch]: ...


@dataclass(frozen=True, slots=True)
class Task:
    query: str
    expected: Any = None


@dataclass(frozen=True, slots=True)
class Trajectory:
    task: Task
    steps: tuple[tuple[str, str], ...]
    score: float


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    key: str
    score: float


class SilentVectorIndex: ...


class FilesystemRepository:
    def __init__(self, *, root: Path) -> None: ...


class SQLiteRepository:
    def __init__(
        self,
        *,
        db_path: Path,
        vector_index: VectorIndex | None = None,
    ) -> None: ...


class SqliteVecIndex:
    def __init__(self, *, db_path: Path, dim: int) -> None: ...
    def initialize(self) -> None: ...
    def upsert(self, key, vector) -> None: ...
    def query(self, vector, k) -> list[ScoredMatch]: ...


def tail_tokens(text: str, max_tokens: int) -> str: ...
def build_repository() -> Repository: ...
```

## `methodos.evolution` — Self-Evolution

```python
REFINER_SYSTEM_PROMPT: str  # refiner prompt (paper §3.3 step 2)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    trajectory: Trajectory
    success: bool


TERMINATE_SUCCESS: str = "__methodos_success__"
TERMINATE_FAILURE: str = "__methodos_failure__"


def score(result: RolloutResult) -> float: ...
def mean_score(results: Iterable[RolloutResult]) -> float: ...
async def run_rollout(
    *,
    graph,
    solver,
    llm,
    task,
    max_steps: int = 50,
    guidance_hops: int = 2,
    trajectory_window: int = 3,
) -> RolloutResult: ...
async def execute_action_stub(action: str) -> str: ...
def tail_concat(traces: Iterable[Trajectory], max_tokens: int) -> str: ...
async def propose_edits(
    *,
    llm,
    graph,
    traces,
    rejected,
    context_tokens: int = 6000,
) -> list[Edit]: ...
def validate_candidate(
    graph,
    edits: Sequence[Edit],
    *,
    allow_cycles: bool = False,
) -> ProceduralGraph | None: ...


class RejectionMemory:
    def __init__(self, max_size: int = 32) -> None: ...
    def __len__(self) -> int: ...
    def add(self, edits, val_score) -> None: ...
    def snapshot(self) -> list[tuple[Edit, float]]: ...


class EvolutionEngine:
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
    ) -> None: ...
    async def run(self, graph: ProceduralGraph) -> ProceduralGraph: ...
```

## `methodos.service` — FastAPI App

```python
def create_app(*, repo: Repository | None = None, llm: LLMClient | None = None) -> FastAPI
```

Routers mounted by the factory:

- `GET /health` → `{"status": "ok"}`
- `GET /v1/graphs/{graph_id}` → `ProceduralGraph` (404 on miss)
- `POST /v1/graphs` (body: `GraphCreateRequest`) → `ProceduralGraph` (201)
- `POST /v1/graphs/{graph_id}/guidance` (body: `GuidanceRequest`) → `GuidanceResponse`
- `POST /v1/graphs/{graph_id}/rollback/{tag}` (header: `If-Match`) → `ProceduralGraph`
- `POST /v1/graphs/{graph_id}/evolve` → `202 {detail, job_id, status}` when an
  `EvolutionRunner` is injected; otherwise 501.
- `GET /v1/evolution/jobs/{job_id}` → `EvolutionJobStatus` through the
  injected runner.

Run with: `uvicorn methodos.service:app --host 0.0.0.0 --port 8000`

## `methodos.cli` — Typer CLI

```
methodos init [--graph-id GRAPH_ID] [--from-path PATH]
methodos inspect [--graph-id GRAPH_ID]
methodos serve [--host HOST] [--port PORT] [--reload]
methodos replay [--graph-id GRAPH_ID] [--split SPLIT] [--limit N]
methodos evolve [--graph-id ID] [--train-path P] [--val-path P] [--model M] [--k-rounds N]
methodos eval [--benchmark NAME] [--graph-id ID] [--n N] [--seed S] [--model M]
```

Use `-v`/`--verbose` for DEBUG logging.

## `eval.hotpotqa` — Eval Harness

```python
async def run_eval(
    *,
    graph_id: str | None,
    n: int,
    seed: int,
    model: str,
    data_dir: Path = ...,
    limit: int = 500,
) -> None
```

Runs paired with-PG vs without-PG on HotpotQA, prints EM/F1 deltas.
Requires `pip install methodos[eval]`. CLI: `python -m eval.hotpotqa.run`.

## Versioning

`methodos.__version__` is `"0.1.0"`. The `ProceduralGraph.schema_version`
field is pinned to `Literal[1]`; bumping it requires a migration runner
(deferred to v2 — see plan.md).

## Stability

| API | Stability |
|---|---|
| `schema.py` models | Stable (wire format) |
| `graph.py` function signatures | Stable |
| `llm.py` Protocol | Stable |
| `guidance.py` prompt constants | Stable (changes are documented) |
| `adapter.py` Adapter API | Stable |
| `evolution.py` Algorithm 1 | Stable |
| `repo.py` Protocols | Stable |
| `service.py` HTTP routes | Stable (additive only) |
| `cli.py` subcommands | Stable |
