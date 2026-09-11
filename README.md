# methodos

> *methodos — the way your agent proceeds*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-ruff%20%7C%20mypy%20%7C%20pytest-green)](.github/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](https://github.com/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

Self-evolving procedural graph adapter for LLM agents.

`methodos` wraps any ReAct-style agent with **queryable procedural knowledge**
that improves itself from execution feedback. The procedural graph is the
procedural counterpart to a knowledge graph: it answers *what-to-do*
questions by exposing a localized subgraph of admissible next procedures.
An LLM refiner edits the graph offline based on what worked and what
didn't — implementing paper Algorithm 1.

## At a glance

```python
from methodos import (
    Node,
    Edge,
    Attribute,
    Relation,
    ProceduralGraph,
    PGAdapter,
    Solver,
    AgentState,
    LiteLLMClient,
)


class MySolver:
    """Any object with an async step(state) -> str satisfies Solver Protocol."""

    async def step(self, state: AgentState) -> str:
        # Use state.context (which contains the procedural guidance)
        # when constructing your LLM prompt.
        return "search"


graph = ProceduralGraph(
    id="example",
    nodes={
        "start": Node(id="start"),
        "search": Node(id="search", description="search Wikipedia"),
        "answer": Node(id="answer"),
    },
    edges=[
        Edge(
            src="start",
            dst="search",
            relation=Relation.LEADS_TO,
            attribute=Attribute(
                condition="need information",
                guidance="call search with focused query",
                pitfalls="don't search with the full question",
            ),
        ),
        Edge(
            src="search",
            dst="answer",
            relation=Relation.LEADS_TO,
            attribute=Attribute(
                condition="have enough info",
                guidance="synthesize a concise answer",
                pitfalls="don't repeat observations",
            ),
        ),
    ],
    terminal_ids={"answer"},
)

adapter = PGAdapter(
    solver=MySolver(),
    graph=graph,
    llm=LiteLLMClient(model="gpt-4o-mini"),
)

trajectory: list[tuple[str, str]] = []
for _ in range(3):
    action = await adapter.step(
        query="What is the capital of France?",
        trajectory=trajectory,
    )
    trajectory.append((action, "(host provides observation)"))
```

## Install

```bash
pip install methodos              # core (SQLite default)
pip install methodos[eval]        # + datasets/pandas/tiktoken for HotpotQA eval
pip install methodos[vec]         # + sqlite-vec for fuzzy graph match
pip install methodos[dev]         # pytest/ruff/mypy/hypothesis
```

## CLI

```bash
methodos init                       # create a default graph
methodos inspect                    # graph summary
methodos serve                      # FastAPI on 127.0.0.1:8000
methodos replay                     # print recent trajectories
methodos eval hotpotqa --n 200       # paired with-PG vs without-PG eval
```

## What it does

**Online inference (per agent step):**

1. Match the agent's last action to a node via `match_node(last_action, graph.nodes)`.
2. Extract a 2-hop neighborhood (paper §3.2 default).
3. Cache the guidance key `(graph.id, last_action, last_obs)`.
4. Ask the LLM to translate the subgraph into a short guidance paragraph (cached).
5. Inject that paragraph into `AgentState.context` and call the host solver.

**Offline self-evolution (Algorithm 1):**

For each round:
1. Run the adapter on train tasks; record trajectories + scores.
2. Ask the refiner LLM to propose `Edit` operations (add/delete node or edge, update attribute).
3. Apply edits structurally; reject candidates with cycles or unreachable nodes.
4. Accept iff validation score did not regress; otherwise append to bounded rejection memory.

See `docs/paper-mapping.md` for section-by-section mapping to the paper.

## Architecture

```
methodos/
├── schema.py            Pydantic v2 data models + Edit discriminated union
├── graph.py             Pure functions: match, neighborhood, validate, apply_edits
├── llm.py               LLMClient Protocol + LiteLLMClient (litellm)
├── guidance.py          Ψ prompt + generate_guidance
├── adapter.py           AgentState, Solver Protocol, GuidanceCache, PGAdapter
├── evolution.py         Algorithm 1 + EvolutionEngine
├── repo.py              Repository + VectorIndex Protocols (SQLite, FS, sqlite-vec)
├── auth.py              APIKeyAuth (Bearer / X-API-Key)
├── rate_limit.py        TokenBucketLimiter
├── observability.py     Logging + Prometheus metrics
├── service.py           FastAPI app factory (REST: /v1/graphs, /guidance)
└── cli.py               Typer CLI

eval/
└── hotpotqa/            Paired with-PG vs without-PG eval on HotpotQA

examples/
├── simple_react.py         EchoSolver + 3-node graph + in-process loop
├── hosted_service.py       1-line uvicorn entrypoint
└── evolve_from_scratch.py  Full Algorithm 1 smoke test

tools/
├── bump_version.py      Update pyproject.toml + methodos/__init__.py in lockstep
└── evaluate.py          End-to-end evaluation of every public claim

docs/
├── architecture.md         Module map, data flow, persistence layout
├── api.md                  Full public API reference
├── deployment.md           Docker, env vars, runbook, prod checklist
├── evaluation.md           HotpotQA methodology + how to wire others
└── paper-mapping.md        Paper sections → file:line
```

## Hardening (production posture)

The hosted service is built for production deployments out of the box:

| Feature | How | Default |
|---|---|---|
| API key auth | `PGRAPH_API_KEY` env var | off (dev-friendly) |
| Rate limiting | `PGRAPH_RATE_LIMIT_*` env vars | 60 tokens / 1 per sec |
| Structured logs | `methodos serve --log-json` | text |
| Prometheus metrics | `GET /metrics` | exposed |
| Liveness probe | `GET /health/live` | always 200 when process up |
| Readiness probe | `GET /health/ready` | 200 when repo reachable |
| Sanitized errors | `LLMError` -> 502, generic -> 500 (no traceback in body) | always |

Full runbook (backups, upgrades, common failure modes) in
[`docs/deployment.md`](docs/deployment.md).

## Documentation

- [Architecture](docs/architecture.md) — module map and data flow
- [API reference](docs/api.md) — every public symbol
- [Deployment](docs/deployment.md) — Docker, env vars, prod checklist, runbook
- [Evaluation](docs/evaluation.md) — HotpotQA harness + extending to other benchmarks
- [Paper mapping](docs/paper-mapping.md) — paper sections → code locations
- [CHANGELOG](CHANGELOG.md) — release notes
- [SECURITY](SECURITY.md) — vulnerability reporting policy

## Development

```bash
# Sync deps
uv sync

# Lint, format, type-check
uv run ruff check .
uv run ruff format .
uv run mypy methodos

# Tests
uv run pytest                                          # full suite
uv run pytest --cov=methodos --cov-fail-under=95      # with coverage gate

# Service
uv run uvicorn methodos.service:app --reload --host 127.0.0.1 --port 8000

# Eval
uv sync --extra eval
python -m eval.hotpotqa.run --n 5

# Bump version
uv run python tools/bump_version.py 0.1.1

# Docker
docker build -t methodos:0.1.0 -f docker/Dockerfile --build-arg VERSION=0.1.0 .
docker compose up methodos
```

## Engineering standards

- **Google Python Style** — `snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`
- **No semi-private naming** — every name is genuinely public
- **No lazy imports** — top-of-file absolute imports
- **Highest polymorphism** — `typing.Protocol` everywhere, no ABCs
- **Minimal abstraction** — Pydantic models for data; classes only for stateful behavior
- **Highest standards** — `mypy --strict` clean, ≥95% coverage, comprehensive tests

## License

Apache-2.0
