# Deployment

`methodos` ships as a single Python package plus a self-hosted FastAPI
service. Production deployment uses Docker; development uses `uv`.

## Requirements

- Python 3.12+ for the runtime
- An LLM API key (OpenAI, Anthropic, Google, Grok) — set as `OPENAI_API_KEY`
  or the provider-specific equivalent
- Optional: `sqlite-vec` for semantic graph node matching
- Optional: Postgres with `asyncpg` for horizontally scaled service deployments

## Install

```bash
pip install methodos              # core
pip install methodos[eval]        # adds datasets/pandas/tiktoken for HotpotQA
pip install methodos[vec]         # adds sqlite-vec for fuzzy match
pip install methodos[postgres]    # adds the pooled Postgres repository
pip install methodos[dev]         # pytest, ruff, mypy, hypothesis for development
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PGRAPH_BACKEND` | `sqlite` | `sqlite`, `filesystem`, or `postgres` |
| `PGRAPH_HOME` | `~/.methodos` | data root |
| `PGRAPH_POSTGRES_DSN` | — | required when `PGRAPH_BACKEND=postgres` |
| `PGRAPH_VEC` | `0` | `1` enables sqlite-vec |
| `PGRAPH_VEC_DIM` | `1536` | embedding dimension when `PGRAPH_VEC=1` |
| `OPENAI_API_KEY` | — | required for OpenAI / litellm |
| `ANTHROPIC_API_KEY` | — | for Anthropic via litellm |
| `GOOGLE_API_KEY` | — | for Google via litellm |
| `GROK_API_KEY` | — | for Grok via litellm |
| `METHODOS_API_KEY` | — | optional API key for non-public REST endpoints |
| `METHODOS_API_KEYS` | — | comma-separated overlapping keys for zero-downtime key rotation; takes precedence over `METHODOS_API_KEY` |
| `METHODOS_API_KEY_ROLES` | — | optional comma-separated `key=reader|writer|operator` role assignments; unspecified authenticated keys remain writers |
| `METHODOS_TENANT_KEYS` | — | optional comma-separated `tenant:key` pairs; enables tenant isolation |
| `METHODOS_TENANT_ID` | `default` | single-tenant identity when tenant keys are not configured |
| `METHODOS_RATE_LIMIT_PER_MINUTE` | `0` | request limit; Postgres uses a shared window, SQLite/filesystem use per-process limiting; `0` disables it |

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
- [ ] Set `METHODOS_API_KEY` or `METHODOS_API_KEYS` for any non-local deployment; `/health` and
      `/metrics` remain public for probes and scraping.
- [ ] For least privilege, set `METHODOS_API_KEY_ROLES` and assign readers,
      writers, and operators explicitly; unspecified keys retain the legacy
      writer role for backward compatibility.
- [ ] For multiple tenants, set `METHODOS_TENANT_KEYS` and send the matching
      `X-API-Key`; the server derives the tenant and rejects mismatched
      `X-Tenant-ID` headers. Do not use a shared `METHODOS_API_KEY` in this mode.
- [ ] Mount a persistent volume at `/data` for `PGRAPH_HOME` (the SQLite
      database lives here).
- [ ] For multiple service replicas, install `methodos[postgres]`, set
      `PGRAPH_BACKEND=postgres` and `PGRAPH_POSTGRES_DSN`, and run the database
      with managed backups, TLS, pooling limits, and migration review.
      For the included image, build with `--build-arg METHODOS_POSTGRES=1`.
- [ ] If using sqlite-vec, ensure the SQLite build in your image supports
      extensions (most pre-built Python wheels do).
- [ ] Behind a reverse proxy with TLS termination; the service itself is
      plaintext HTTP.
- [ ] Scrape `/health` for liveness and `/ready` for repository readiness.
- [ ] Send `methodos.service:app` access logs to your log aggregator and
      include the `x-request-id` response header in request traces.
- [ ] Configure an OpenTelemetry tracer provider/exporter in the deployment
      environment if distributed traces are required; the service emits
      correlated HTTP spans without forcing a vendor.

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/ready` | — | `{"status": "ready"}` or 503 |
| GET | `/v1/graphs/{id}` | — | `ProceduralGraph` or 404 |
| POST | `/v1/graphs` | `{id, from_graph_id?}` | `ProceduralGraph` (201) |
| PUT | `/v1/graphs/{id}` | `ProceduralGraph` + `If-Match` revision | updated graph or 409 |
| POST | `/v1/graphs/{id}/guidance` | `{query, trajectory?, window?, guidance_hops?}` | `{guidance}` |
| POST | `/v1/graphs/{id}/evolve` | — | 501 (reserved; use SDK) |
| POST | `/v1/graphs/{id}/rollback/{tag}` | `If-Match` | 200 / 404 / 409 |

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
- `prometheus:9090` (scrapes `/metrics`).

## Persistence

Default backend: SQLite with WAL mode and FK constraints. WAL allows
concurrent reads during the evolution loop's diagnostic-rollout phase.
Each graph is stored as JSON in the `graphs` body column; trajectories
are stored as JSONL-equivalent in the `trajectories` body column.

To create an atomic, integrity-checked backup of the configured SQLite store:

```bash
methodos backup /secure/backups/methodos-$(date +%Y%m%d%H%M%S).db
```

Schedule this command externally and retain copies according to your policy.
Before a restore, verify an artifact with:

```bash
methodos verify-backup /secure/backups/methodos-20261014090000.db
```

For Postgres, use the managed provider's backup and restore verification.

## Migration (planned for v2)

`schema_version: Literal[1]` is currently fixed. Repository startup uses a
forward-only migration runner, but no version 2 migration exists yet. Treat
schema_version as the only supported version and rebuild graphs via the CLI on
upgrade; do not manually edit production databases.

## Scaling

Single-process by default. SQLite is suitable for one service instance with a
local persistent volume; do not place the database on NFS or share it between
independent writers. For horizontal scaling, use the transactional Postgres
`Repository` backend and its shared rate-limit store.
Tenant-key mode currently isolates graph namespaces in the configured process;
a distributed deployment must carry that tenant scope into the shared
repository and identity provider.

The evolution loop is CPU/LLM-bound, not I/O-bound. For multi-tenant
deployments, run one EvolutionEngine per tenant to avoid
rejection-memory collisions.
