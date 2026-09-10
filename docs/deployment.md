# Deployment

`methodos` ships as a single Python package plus a self-hosted FastAPI
service. Production deployment uses Docker; development uses `uv`.

## Requirements

- Python 3.12+ for the runtime
- An LLM API key (OpenAI, Anthropic, Google, Grok) — set as `OPENAI_API_KEY`
  or the provider-specific equivalent
- Optional: `sqlite-vec` for semantic graph node matching

## Install

```bash
pip install methodos              # core
pip install methodos[eval]        # adds datasets/pandas/tiktoken for HotpotQA
pip install methodos[vec]         # adds sqlite-vec for fuzzy match
pip install methodos[dev]         # pytest, ruff, mypy, hypothesis for development
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PGRAPH_BACKEND` | `sqlite` | `sqlite` or `filesystem` |
| `PGRAPH_HOME` | `~/.methodos` | data root |
| `PGRAPH_VEC` | `0` | `1` enables sqlite-vec |
| `PGRAPH_VEC_DIM` | `1536` | embedding dimension when `PGRAPH_VEC=1` |
| `OPENAI_API_KEY` | — | required for OpenAI / litellm |
| `ANTHROPIC_API_KEY` | — | for Anthropic via litellm |
| `GOOGLE_API_KEY` | — | for Google via litellm |
| `GROK_API_KEY` | — | for Grok via litellm |

## CLI

```bash
methodos init                   # create default graph
methodos inspect                # show summary
methodos serve                  # start FastAPI on 127.0.0.1:8000
methodos replay                 # print recent trajectories
methodos eval hotpotqa --n 200  # run paired HotpotQA eval
```

## Hosted Service (REST)

```bash
# Local
uvicorn methodos.service:app --reload --host 127.0.0.1 --port 8000

# Docker
docker compose up methodos
```

### Production checklist

- [ ] Set `OPENAI_API_KEY` (or equivalent) in the deployment environment.
- [ ] Mount a persistent volume at `/data` for `PGRAPH_HOME` (the SQLite
      database lives here).
- [ ] If using sqlite-vec, ensure the SQLite build in your image supports
      extensions (most pre-built Python wheels do).
- [ ] Behind a reverse proxy with TLS termination; the service itself is
      plaintext HTTP.
- [ ] Scrape `/health` for liveness probes.
- [ ] Send `methodos.service:app` access logs to your log aggregator
      (structured JSON via structlog; see observability profile).

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/v1/graphs/{id}` | — | `ProceduralGraph` or 404 |
| POST | `/v1/graphs` | `{id, from_graph_id?}` | `ProceduralGraph` (201) |
| POST | `/v1/graphs/{id}/guidance` | `{query, trajectory?, window?, guidance_hops?}` | `{guidance}` |
| POST | `/v1/graphs/{id}/evolve` | — | 501 (reserved; use SDK) |

### Example: request guidance

```bash
curl -sX POST http://localhost:8000/v1/graphs/my-graph/guidance \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the capital of France?",
    "trajectory": [["search", "Paris is the capital"]],
    "window": 3,
    "guidance_hops": 2
  }'
```

## Observability (opt-in)

The `docker/compose.yml` defines a profile `observability` that starts
OpenTelemetry Collector and Prometheus alongside the main service:

```bash
docker compose --profile observability up
```

This brings up:
- `methodos:8000` (the service)
- `otel-collector:4317` (OTLP gRPC)
- `prometheus:9090` (scrapes `/metrics` if exposed; not yet wired in
  this milestone — see plan.md).

## Persistence

Default backend: SQLite with WAL mode and FK constraints. WAL allows
concurrent reads during the evolution loop's diagnostic-rollout phase.
Each graph is stored as JSON in the `graphs` body column; trajectories
are stored as JSONL-equivalent in the `trajectories` body column.

To back up: copy `~/.methodos/methodos.db` (and the WAL/SHM sidecars
if present).

## Migration (planned for v2)

`schema_version: Literal[1]` is currently fixed. Bumping to v2 will add a
migration runner in `methodos.repo.migrate`. For now, treat schema_version
as the only supported version and rebuild graphs via the CLI on upgrade.

## Scaling

Single-process by default. The FastAPI app is stateless and can be
horizontally scaled; persistence is shared (SQLite via NFS, or move to
Postgres via the `Repository` Protocol — Postgres impl is on the roadmap).

The evolution loop is CPU/LLM-bound, not I/O-bound. For multi-tenant
deployments, run one EvolutionEngine per tenant to avoid
rejection-memory collisions.
