# plan.md — `methodos` Production Build Plan

Self-evolving procedural graph adapter for LLM agents.
Implementer: build this file's contents in order; every function lands complete, no stubs.

## Locked-in Decisions

| Decision | Value |
|---|---|
| PyPI / import name | `methodos` |
| Repo / package directory | `/Users/sachin/repo/pgraph` (package `methodos/` lives at repo root, flat layout, no `src/`) |
| Python | 3.12+ |
| Type system | Pydantic v2 + `typing.Protocol` (no ABCs) |
| LLM provider | `litellm` (covers OpenAI / Anthropic / Google / Grok / local) |
| Persistence default | SQLite with WAL + FK + indices |
| Persistence opt-in | Filesystem via `PGRAPH_BACKEND=filesystem` |
| Vector search | `sqlite-vec` opt-in via `PGRAPH_VEC=1` |
| CLI | `typer` |
| Hosted service | `fastapi` + `uvicorn` |
| Style | Google Python Style Guide; no `_foo()` "protected" markers |
| Tests | pytest + pytest-asyncio + hypothesis; ≥95% line coverage on touched files per phase |
| Eval | `eval/hotpotqa/` only, gated by `methodos[eval]` extra |
| Container | multi-stage `uv`-based Dockerfile, non-root, single image |
| License | Apache-2.0 |

## Engineering Standards (applied to every file)

### Google Python Style

- `snake_case` for functions / methods / variables / modules
- `PascalCase` for classes and class-like type aliases
- `UPPER_SNAKE_CASE` for module-level constants
- Docstrings on every public symbol, imperative mood ("Match an action to a node.")
- `from __future__ import annotations` at top of every module
- Absolute imports only; no relative imports
- Type hints on every public function signature
- `pathlib.Path`, never `os.path`
- f-strings, never `%` or `.format`
- `logging.getLogger(__name__)` per module, never `print`
- Custom exceptions inherit `Error` suffix
- No mutable default arguments; use `field(default_factory=...)` or `None` sentinel
- `from __future__ import annotations` lets us write `list[X]` / `X | None` on 3.12+

### Visibility (no semi-private naming)

- No `_foo()` to mean "protected." Every public symbol is genuinely public.
- Module-private symbols: leading `_` per Python convention only.
- No `__dunder__` tricks.

### Type system

- Pydantic v2 for all data models with `ConfigDict(extra="forbid")`
- `typing.Protocol` for all interfaces (no ABCs)
- Generic types where they help (`Sequence[Edit]`, `Mapping[str, Node]`, `AsyncIterator[Trajectory]`)
- `mypy --strict` must pass on `methodos/`

### Async

- All I/O is `async def`; sync only for pure functions and `__init__`
- `asyncio` only, no threading
- `asyncio_mode = "auto"` in pytest config; `async def test_*` works without decorators

### Error handling

- Custom exceptions only; no bare `raise Exception`
- Catch specific exceptions; log with `logger.exception` (with traceback) or `logger.warning`
- Never silently swallow

### Logging

- `logger = logging.getLogger(__name__)` at module top
- `logger.info` for routine events
- `logger.warning` for recoverable errors
- `logger.exception` for unexpected errors
- `logger.debug` for diagnostics

### Testing

- pytest + pytest-asyncio + pytest-cov + respx + hypothesis
- Every public function has at least one test
- Hypothesis for graph invariants (cycle detection, reachability, edit application)
- Fakes defined in `tests/conftest.py` and reused

### Observability

- OTel spans in `adapter.step`, `evolution.run` rounds, `service` endpoints
- Prometheus counters/histograms for guidance latency, evolution accept/reject rates
- structlog for structured logs in service
- Hooks are in the relevant modules, not a separate package

## File-by-File Plan

### Source files — `methodos/` at repo root

| File | Contents |
|---|---|
| `methodos/__init__.py` | Re-export public API; `__version__ = "0.1.0"` |
| `methodos/schema.py` | Pydantic v2 models only: `Relation` (Enum), `Attribute`, `Node`, `Edge`, `ProceduralGraph`, Edit discriminated union (`EditAddNode` / `EditDeleteNode` / `EditAddEdge` / `EditDeleteEdge` / `EditUpdateAttr`) |
| `methodos/graph.py` | Pure functions: `match_node`, `neighborhood`, `validate`, `apply_edits`, `has_path_to`, `has_reachable_terminal`; helpers `_adjacency`, `_infer_terminals`, `_has_cycle`; `StructuralIssue` dataclass |
| `methodos/guidance.py` | `GUIDANCE_SYSTEM_PROMPT` constant; async `generate_guidance`; private `_format_graph_for_prompt`, `_format_trajectory_window` |
| `methodos/llm.py` | `LLMClient` Protocol; `LiteLLMClient` (litellm wrapper with retries) |
| `methodos/adapter.py` | `AgentState` (frozen slotted dataclass); `Solver` Protocol; `GuidanceCache` (LRU ordered-dict); `PGAdapter` class |
| `methodos/evolution.py` | `Task`, `Trajectory`, `RolloutResult` dataclasses; `RejectionMemory` (bounded FIFO via `deque(maxlen=...)`); `REFINER_SYSTEM_PROMPT`; pure functions `run_rollout`, `score`, `mean_score`, `tail_concat`, `propose_edits`, `validate_candidate`; `EvolutionEngine` class implementing Algorithm 1 |
| `methodos/repo.py` | `Repository` Protocol; `VectorIndex` Protocol; `ScoredMatch`; `NoOpVectorIndex`; `FilesystemRepository`; `SQLiteRepository`; `SqliteVecIndex`; `tail_tokens` helper; `build_repository()` env-driven factory |
| `methodos/service.py` | FastAPI app factory `create_app`; routers for `GET /v1/graphs/{id}`, `POST /v1/graphs`, `POST /v1/graphs/{id}/guidance`, `POST /v1/graphs/{id}/evolve`, `GET /health`; module-level `app` |
| `methodos/cli.py` | `cli = typer.Typer(...)` with subcommands `init`, `inspect`, `serve`, `evolve`, `replay`, `eval`; `main()` callback for `--verbose` |

### Tests — `tests/` mirrors source

| File | Coverage target |
|---|---|
| `tests/conftest.py` | `FakeLLM`, `ScriptedLLM`, `FakeSolver`, `SequenceSolver`, `InMemoryRepository`, fixtures (`sample_graph`, `tmp_path` repos) |
| `tests/test_schema.py` | All Pydantic validators, Edit discriminated union parsing |
| `tests/test_graph.py` | match, neighborhood (0/1/2 hop, unknown node, h=0), validate (cycle on/off, reachability), apply_edits (all 5 variants + edge cases + empty), has_path_to |
| `tests/test_guidance.py` | Prompt formatting, `generate_guidance` with `FakeLLM`, fallback on missing fields |
| `tests/test_llm.py` | `LiteLLMClient` retry with respx mock; one live test gated on `OPENAI_API_KEY` |
| `tests/test_adapter.py` | End-to-end with `FakeSolver` + `FakeLLM`; guidance cache hit/miss; fallback to full graph on miss |
| `tests/test_evolution.py` | Algorithm 1: accept on tie, reject on worse, structural reject skips validation, rejection memory eviction, `Tail_{L_max}` truncation |
| `tests/test_repo.py` | `FilesystemRepository` + `SQLiteRepository` round-trip; `NoOpVectorIndex` no-op; concurrent SQLite writes serialized |
| `tests/test_service.py` | FastAPI `TestClient` against `create_app(fake repo, fake llm)` for every endpoint; 404 on missing graph |
| `tests/test_cli.py` | Typer `CliRunner` for each subcommand; exit codes; error paths |

### Eval — `eval/hotpotqa/`, gated by `methodos[eval]` extra

| File | Contents |
|---|---|
| `eval/__init__.py` | Package marker |
| `eval/hotpotqa/__init__.py` | Package marker |
| `eval/hotpotqa/tasks.py` | `HotpotQATask` dataclass; `load_tasks(path, limit)`; `download_if_missing(target_dir, limit)` using `datasets.load_dataset("hotpot_qa", "distractor", split="validation")` |
| `eval/hotpotqa/solver.py` | `HotpotQASolver` ReAct-style with injected search callback; max-step bound |
| `eval/hotpotqa/run.py` | `run_eval(graph_id, n, seed, model)` paired comparison (with-PG vs without-PG); EM/F1 deltas; `main()` argparse for `python -m eval.hotpotqa.run` |

### Examples — `examples/`

| File | Demonstrates |
|---|---|
| `examples/simple_react.py` | ~50 lines: 3-node graph, wrap `EchoSolver` with `PGAdapter`, 3 in-process steps |
| `examples/hosted_service.py` | 5 lines: `uvicorn.run("methodos.service:app", ...)` |
| `examples/evolve_from_scratch.py` | Full Algorithm 1 smoke test with `FixedSolver` and synthetic tasks |

### Docs — `docs/`

| File | Contents |
|---|---|
| `docs/architecture.md` | Module map, data-flow mermaid diagram, request lifecycle |
| `docs/api.md` | Generated from docstrings + per-class examples |
| `docs/deployment.md` | Docker, env vars (`PGRAPH_BACKEND`, `PGRAPH_HOME`, `PGRAPH_VEC`, `PGRAPH_VEC_DIM`, `OPENAI_API_KEY`), production checklist |
| `docs/evaluation.md` | How to run HotpotQA; how to wire MultiChallenge / ALFWorld / τ-bench / BFCL / EnterpriseArena / GDPval yourself; paired-comparison methodology |
| `docs/paper-mapping.md` | Each paper section (3.1, 3.2, 3.3, 4, 5.1–5.5, App. B.6) → `file:line` |

### CI / Container

| File | Contents |
|---|---|
| `.github/workflows/ci.yml` | ruff check, mypy --strict, pytest --cov=methodos --cov-fail-under=95 |
| `docker/Dockerfile` | `python:3.12-slim`, multi-stage uv install, non-root, `VOLUME /data`, `EXPOSE 8000` |
| `docker/compose.yml` | `methodos` service + opt-in `otel-collector` and `prometheus` profiles |

### Config

| File | Contents |
|---|---|
| `pyproject.toml` | uv-managed, PEP 621; `[project.scripts] methodos = "methodos.cli:main"`; optional deps `eval` and `vec`; dev deps; ruff / mypy / pytest config |
| `.python-version` | `3.12` |
| `.gitignore` | Standard Python + `.venv/`, `dist/`, `__pycache__/`, `eval/hotpotqa/data/` |
| `AGENTS.md` | Build/test commands for opencode (`uv sync`, `uv run pytest`, etc.) |
| `LICENSE` | Apache-2.0 |
| `README.md` | Quickstart, install options, links to docs |

## Key Type Definitions (single source of truth)

### `methodos/schema.py`

```python
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Relation(str, Enum):
    LEADS_TO = "leads_to"
    REQUIRES = "requires"
    REPLACES = "replaces"


class Attribute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition: str = Field(min_length=1, max_length=2000)
    guidance: str = Field(min_length=1, max_length=2000)
    pitfalls: str = Field(min_length=1, max_length=2000)


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    description: str = Field(default="", max_length=2000)

    @field_validator("id")
    @classmethod
    def id_not_start_with_dot(cls, v: str) -> str:
        if v.startswith("."):
            raise ValueError("node id cannot start with '.'")
        return v


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    relation: Relation
    attribute: Attribute

    @model_validator(mode="after")
    def endpoints_distinct(self) -> Edge:
        if self.src == self.dst:
            raise ValueError("edge endpoints must be distinct")
        return self


class ProceduralGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str = Field(min_length=1, max_length=128)
    schema_version: Literal[1] = 1
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    terminal_ids: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _terminals_are_nodes(self) -> ProceduralGraph: ...
    @model_validator(mode="after")
    def _edges_endpoints_are_nodes(self) -> ProceduralGraph: ...


class EditAddNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["add_node"] = "add_node"
    node: Node


class EditDeleteNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["delete_node"] = "delete_node"
    node_id: str


class EditAddEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["add_edge"] = "add_edge"
    edge: Edge


class EditDeleteEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["delete_edge"] = "delete_edge"
    src: str
    dst: str
    relation: Relation


class EditUpdateAttr(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["update_attr"] = "update_attr"
    src: str
    dst: str
    relation: Relation
    attribute: Attribute


Edit = EditAddNode | EditDeleteNode | EditAddEdge | EditDeleteEdge | EditUpdateAttr
```

### `methodos/graph.py` (function contracts)

```python
def match_node(action_name: str, nodes: dict[str, Node]) -> str | None
def neighborhood(graph: ProceduralGraph, node_id: str, *, h: int = 2) -> ProceduralGraph
def validate(graph: ProceduralGraph, *, allow_cycles: bool = False) -> list[StructuralIssue]
def apply_edits(graph: ProceduralGraph, edits: Sequence[Edit]) -> ProceduralGraph
def has_path_to(graph: ProceduralGraph, src: str, targets: Iterable[str]) -> bool
def has_reachable_terminal(graph: ProceduralGraph, node_id: str) -> bool
```

- `match_node`: exact-string equality. Returns `None` on miss (caller falls back to full graph per paper §3.2).
- `neighborhood`: BFS up to `h` directed hops. Unknown node → return `graph.model_copy(deep=True)` (paper §3.2 fallback).
- `validate`: reachability to terminal + cycle detection. Returns `list[StructuralIssue]` (empty == healthy).
- `apply_edits`: dispatch per `Edit.kind`; deletes then adds per paper; returns `graph.model_copy(deep=True)` on empty input.
- `has_path_to`: BFS from `src` checking membership in `targets`.
- `has_reachable_terminal`: `has_path_to` against `graph.terminal_ids` (or inferred terminals if empty).

### `methodos/llm.py`

```python
class LLMClient(Protocol):
    async def complete(
        self, *, system: str, user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str: ...


class LiteLLMClient:
    def __init__(
        self, *, model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None: ...

    async def complete(
        self, *, system: str, user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        # litellm.acompletion with exponential retry up to max_retries+1
        # structured outputs via response_format json_schema when json_schema given
        # raises last exception if all attempts fail
        ...
```

### `methodos/adapter.py`

```python
@dataclass(frozen=True, slots=True)
class AgentState:
    query: str
    trajectory: tuple[tuple[str, str], ...]
    context: str


class Solver(Protocol):
    async def step(self, state: AgentState) -> str: ...


class GuidanceCache:
    """LRU cache, ordered-dict based, max_size default 256."""
    def __init__(self, max_size: int = 256) -> None: ...
    def get(self, key: tuple[int, str, str]) -> str | None: ...
    def put(self, key: tuple[int, str, str], value: str) -> None: ...


class PGAdapter:
    def __init__(
        self, *, solver: Solver, graph: ProceduralGraph, llm: LLMClient,
        guidance_hops: int = 2, trajectory_window: int = 3,
        cache: GuidanceCache | None = None,
    ) -> None: ...

    @property
    def graph(self) -> ProceduralGraph: ...   # current frozen graph

    async def step(self, *, query: str, trajectory: list[tuple[str, str]]) -> str:
        # 1. last_action = trajectory[-1][0] if trajectory else else "Start"
        # 2. node_id = match_node(last_action, graph.nodes)
        # 3. sub = neighborhood(graph, node_id, h=h) if node_id else graph
        # 4. cache_key = (hash(sub), last_action, last_obs)
        # 5. guidance = cache.get(cache_key) or await generate_guidance(...) and cache.put
        # 6. state = AgentState(query, tuple(trajectory), f"### PROCEDURAL GUIDANCE\n{guidance}")
        # 7. return await solver.step(state)
```

### `methodos/evolution.py`

Algorithm 1 verbatim:

```python
class EvolutionEngine:
    def __init__(
        self, *, llm: LLMClient, repo: Repository,
        train_tasks: Sequence[Task], val_tasks: Sequence[Task],
        solver: Solver,
        k_rounds: int = 10, l_max_tokens: int = 8000,
        rejection_memory_size: int = 32, allow_cycles: bool = False,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._train = list(train_tasks)
        self._val = list(val_tasks)
        self._solver = solver
        self._k = k_rounds
        self._l_max = l_max_tokens
        self._allow_cycles = allow_cycles
        self._rejection = RejectionMemory(max_size=rejection_memory_size)

    async def run(self, graph: ProceduralGraph) -> ProceduralGraph:
        current = graph
        current_score = await self._score_validation(current)
        for round_idx in range(1, self._k + 1):
            traces = await self._collect_diagnostic_traces(current)
            edits = await propose_edits(
                llm=self._llm, graph=current,
                traces=traces, rejected=self._rejection.snapshot(),
            )
            candidate = validate_candidate(current, edits, allow_cycles=self._allow_cycles)
            if candidate is None:
                self._rejection.add(edits, current_score)
                continue
            candidate_score = await self._score_validation(candidate)
            if candidate_score >= current_score:
                current = candidate
                current_score = candidate_score
            else:
                self._rejection.add(edits, candidate_score)
        await self._repo.save_graph(current)
        return current
```

`propose_edits` sends the refiner prompt + `tail_concat(traces, max_tokens=6000)` + rejection history to the LLM with `REFINER_SYSTEM_PROMPT`. Response is parsed as JSON array of `Edit` objects; malformed items are dropped with a `logger.warning`.

`validate_candidate` applies edits via `apply_edits` and returns the new graph if structurally valid (reachability + cycle check), else `None`.

`tail_concat` is paper's `Tail_{L_max}` — truncates from the beginning, preserving the last `max_tokens` characters (1 token ≈ 4 chars).

### `methodos/repo.py`

```python
class Repository(Protocol):
    async def load_graph(self, graph_id: str) -> ProceduralGraph: ...
    async def save_graph(self, graph: ProceduralGraph) -> None: ...
    async def snapshot(self, graph_id: str, tag: str) -> None: ...
    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None: ...
    async def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]: ...


class VectorIndex(Protocol):
    def upsert(self, key: str, vector: list[float]) -> None: ...
    def query(self, vector: list[float], k: int) -> list[ScoredMatch]: ...


@dataclass(frozen=True)
class ScoredMatch:
    key: str
    score: float


class NoOpVectorIndex:
    def upsert(self, key: str, vector: list[float]) -> None: pass
    def query(self, vector: list[float], k: int) -> list[ScoredMatch]: return []


class FilesystemRepository:
    """Layout:
        <root>/graphs/<graph_id>/graph.json         # current retained
        <root>/graphs/<graph_id>/trajectories/<split>.jsonl
        <root>/graphs/<graph_id>/rejection.jsonl
        <root>/graphs/<graph_id>/history/<tag>.json
    Atomic writes via os.replace(tmp, path)."""
    def __init__(self, *, root: Path) -> None: ...


class SQLiteRepository:
    """Default. WAL + FK on. Schema:
        graphs(id PK, schema_version, body, updated_at)
        trajectories(id PK, graph_id FK, split, score, body, ts)
            idx_trajectories_gid_split_ts
        snapshots(graph_id FK, tag, body, ts, PK(graph_id, tag))
        rejection_memory(id PK, graph_id FK, body, val_score, ts)
            idx_rejection_gid_ts
    """
    def __init__(self, *, db_path: Path, vector_index: VectorIndex | None = None) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_index: VectorIndex = vector_index or NoOpVectorIndex()
        self._ensure_schema()  # sync, called once in __init__


class SqliteVecIndex:
    """vec0 virtual table: vec_nodes(key TEXT PK, embedding float[N]).
    Loaded only when PGRAPH_VEC=1. Uses sqlite_vec.serialize_float32."""
    def __init__(self, *, db_path: Path, dim: int) -> None: ...
    def initialize(self) -> None: ...
    def upsert(self, key: str, vector: list[float]) -> None: ...
    def query(self, vector: list[float], k: int) -> list[ScoredMatch]: ...


def build_repository() -> Repository:
    backend = os.environ.get("PGRAPH_BACKEND", "sqlite")
    home = Path(os.environ.get("PGRAPH_HOME", Path.home() / ".methodos"))
    home.mkdir(parents=True, exist_ok=True)
    if backend == "filesystem":
        return FilesystemRepository(root=home)
    if backend == "sqlite":
        db_path = home / "methodos.db"
        if os.environ.get("PGRAPH_VEC") == "1":
            dim = int(os.environ.get("PGRAPH_VEC_DIM", "1536"))
            vec = SqliteVecIndex(db_path=db_path, dim=dim)
            vec.initialize()
            return SQLiteRepository(db_path=db_path, vector_index=vec)
        return SQLiteRepository(db_path=db_path)
    raise ValueError(f"unknown PGRAPH_BACKEND: {backend!r}")
```

### `methodos/service.py`

```python
def create_app(*, repo: Repository | None = None, llm: LiteLLMClient | None = None) -> FastAPI:
    """FastAPI app factory. Lifespan wires deps into app.state."""
    backend_repo = repo or build_repository()
    backend_llm = llm or LiteLLMClient(model="gpt-4o-mini")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.repo = backend_repo
        app.state.llm = backend_llm
        logger.info("methodos service started")
        yield
        logger.info("methodos service stopped")

    app = FastAPI(title="methodos", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]: return {"status": "ok"}

    @app.get("/v1/graphs/{graph_id}", response_model=ProceduralGraph)
    async def get_graph(...): ...

    @app.post("/v1/graphs", response_model=ProceduralGraph, status_code=201)
    async def create_graph(request: GraphCreateRequest, ...): ...

    @app.post("/v1/graphs/{graph_id}/guidance", response_model=GuidanceResponse)
    async def get_guidance(graph_id, request, ...): ...

    @app.post("/v1/graphs/{graph_id}/evolve", response_model=EvolveResponse)
    async def evolve_graph(...):
        # 501 — evolution requires a host Solver; use the SDK
        raise HTTPException(status_code=501, detail=...)

    return app


app = create_app()  # for `uvicorn methodos.service:app`
```

### `methodos/cli.py`

```python
import typer

cli = typer.Typer(name="methodos", help="...", no_args_is_help=True)


@cli.callback()
def main(verbose: bool = False) -> None:
    """Configure logging; --verbose sets DEBUG."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@cli.command()
def init(graph_id: str = "default", from_path: Path | None = None) -> None: ...

@cli.command()
def inspect(graph_id: str = "default") -> None: ...

@cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn
    uvicorn.run("methodos.service:app", host=host, port=port, reload=reload)

@cli.command()
def evolve(graph_id: str = "default", k_rounds: int = 10,
           train_path: Path = ..., val_path: Path = ...,
           model: str = "gpt-4o-mini") -> None: ...

@cli.command()
def replay(graph_id: str = "default", split: str = "train", limit: int = 10) -> None: ...

@cli.command()
def eval(benchmark: str = "hotpotqa", graph_id: str | None = None,
         n: int = 200, seed: int = 0, model: str = "gpt-4o-mini") -> None:
    if benchmark != "hotpotqa":
        typer.echo(f"unknown benchmark: {benchmark!r}", err=True); raise typer.Exit(1)
    from eval.hotpotqa.run import run_eval
    asyncio.run(run_eval(graph_id=graph_id, n=n, seed=seed, model=model))
```

## Build Phases

Each phase is **atomic** — ends with green CI, no broken intermediate state.

| Phase | Days | Output | Acceptance |
|---|---|---|---|
| **0** Scaffolding | 0.5 | Repo, `pyproject.toml`, ruff/mypy/pytest configs, CI, empty module tree, `AGENTS.md`, README skeleton | `uv sync` works; `uv run pytest` runs an empty suite; `uv run ruff check .` and `uv run mypy methodos` clean |
| **1** Schema + graph | 1.5 | `schema.py`, `graph.py` complete; `tests/test_schema.py`, `tests/test_graph.py` | All Pydantic validators and every graph algorithm tested; ≥95% coverage on these files |
| **2** LLM + guidance | 1.0 | `llm.py`, `guidance.py` complete; respx-mocked retry test | `tests/test_llm.py`, `tests/test_guidance.py` green |
| **3** Adapter | 1.0 | `adapter.py` complete; `PGAdapter`, `GuidanceCache`, fakes in `conftest.py` | `tests/test_adapter.py` green; cache hit/miss asserted |
| **4** Persistence | 1.0 | `repo.py` complete; `FilesystemRepository` + `SQLiteRepository` + `NoOpVectorIndex` (+ `SqliteVecIndex` available but optional in v1) | Round-trip tests pass; concurrent SQLite writes serialized |
| **5** Service + CLI | 1.0 | `service.py`, `cli.py` complete | `TestClient` + `CliRunner` tests green |
| **6** Evolution | 2.0 | `evolution.py` complete: Algorithm 1, all pure functions, `EvolutionEngine` | `tests/test_evolution.py` green; `examples/evolve_from_scratch.py` runs end-to-end with stub LLM |
| **7** Eval | 1.0 | `eval/hotpotqa/` complete | `pip install -e .[eval]`; `python -m eval.hotpotqa.run --n 5` runs with stub LLM |
| **8** Docs + Docker | 1.0 | 5 docs, Dockerfile, compose, README complete | `docker build .` succeeds; `docker compose up methodos` starts |
| **Total** | **10.0** | | |

## Out of Scope (explicit, deferred)

- Web UI / graph visualization (users render via `graphviz` / `networkx` from the Pydantic models)
- Multi-LLM orchestration beyond litellm (litellm covers 100+ providers already)
- Distributed / clustered evolution (paper is single-machine; production uses hosted service)
- Streaming responses per step (REST polling is sufficient for v1)
- Tool / function-calling schema validation (host agent validates its own tool schemas)
- WebSocket / SSE for streaming guidance (REST suffices; can add without breaking API)
- Schema migrations beyond v1 (`schema_version: Literal[1]`; bumping requires a migration runner, deferred)
- Hosting WebUI for traces

## Risk Register (with mitigations baked into the plan)

| Risk | Mitigation |
|---|---|
| Test suite becomes slow | Targeted unit tests for hot paths; future `pytest-xdist` |
| HotpotQA eval flakiness | `temperature=0`, fixed seeds, paired comparison cancels noise |
| `sqlite-vec` wheel availability | `PGRAPH_VEC=1` opt-in; `NoOpVectorIndex` fallback always available |
| Rejection memory unbounded | `deque(maxlen=...)` |
| Exact-match `Match` brittle in production | Phase 4 includes `SqliteVecIndex` (optional); production deployments enable it for fuzzy fallback |
| Per-step guidance latency | `GuidanceCache` in plan; future async streaming |
| Litellm version churn | Pinned in `uv.lock`; CI tests pinned version |
| 95% coverage gate too aggressive for early phases | Per-phase floor is "touched files ≥95%"; unused code can't land |
| OpenAI-style tools endpoint shape not stable across providers | Adapter doesn't speak tool JSON; host agent owns tool plumbing |

## Atomic Deliverable Boundaries

Each phase produces a shippable artifact:

- **After Phase 1**: `methodos.schema` and `methodos.graph` are usable as a standalone library.
- **After Phase 3**: A user can `import PGAdapter` and wrap their solver.
- **After Phase 4**: Graphs survive restarts; SQLite is the default backend.
- **After Phase 5**: Hosted service is deployable via `docker compose up`.
- **After Phase 6**: Full paper implementation (Algorithm 1 working end-to-end).
- **After Phase 7**: HotpotQA comparison numbers reproducible.
- **After Phase 8**: Production deployable with observability opt-in profile.

Nothing broken between phases.

## Open Questions (need answers before implementation)

1. **`schema_version: Literal[1]` with no migration runner** — acceptable for v1, deferred to v2? *(Recommend: yes.)*
2. **`methodos eval hotpotqa` as Typer command AND `python -m eval.hotpotqa.run` as module entry** — both? *(Recommend: yes.)*
3. **GitHub repository creation** — should the repo be pushed to `github.com/sachin/methodos` (using `sachncs@gmail.com`)? Or kept local until v1 ships?
4. **`pyproject.toml` author identity** — `name = "Sachin"`, `email = "sachncs@gmail.com"` — confirm?

## Build Command Reference

Once the build begins, the implementer runs these from the repo root:

```bash
# Phase 0
uv init --package --no-readme .           # or manually scaffold pyproject.toml
uv add pydantic litellm typer fastapi uvicorn httpx aiofiles structlog \
       opentelemetry-api opentelemetry-sdk prometheus-client
uv add --dev pytest pytest-asyncio pytest-cov pytest-mock hypothesis ruff mypy respx
uv add --optional eval datasets pandas tiktoken
uv add --optional vec sqlite-vec
uv sync

# Per phase
uv run pytest tests/ -x
uv run mypy methodos
uv run ruff check .
uv run pytest --cov=methodos --cov-fail-under=95

# Phase 5+
uv run uvicorn methodos.service:app --reload

# Phase 7+
uv sync --extra eval
python -m eval.hotpotqa.run --n 5

# Phase 8
docker build -t methodos:0.1.0 .
docker compose up methodos
```