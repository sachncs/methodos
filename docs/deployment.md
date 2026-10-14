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
| `METHODOS_ENV` | `development` for source runs; `production` in the container image and Compose | `development`, `test`, `staging`, or `production`; staging/production fail closed without auth and durable persistence |
| `OPENAI_API_KEY` | — | required for OpenAI / litellm |
| `ANTHROPIC_API_KEY` | — | for Anthropic via litellm |
| `GOOGLE_API_KEY` | — | for Google via litellm |
| `GROK_API_KEY` | — | for Grok via litellm |
| `METHODOS_API_KEY` | — | optional API key for non-public REST endpoints |
| `METHODOS_API_KEYS` | — | comma-separated overlapping keys for zero-downtime key rotation; takes precedence over `METHODOS_API_KEY` |
| `METHODOS_API_KEY_ROLES` | — | comma-separated `key=reader|writer|operator` assignments; required for every key in staging/production |
| `METHODOS_TENANT_KEYS` | — | optional comma-separated `tenant:key` pairs; enables tenant isolation |
| `METHODOS_TENANT_ID` | `default` | single-tenant identity when tenant keys are not configured |
| `METHODOS_RATE_LIMIT_PER_MINUTE` | `0` | request limit; Postgres and SQLite use shared windows, filesystem remains per-process; `0` disables it |
| `METHODOS_MAX_BODY_BYTES` | `1048576` | maximum streamed HTTP request body size; oversized requests return `413` |
| `METHODOS_BODY_READ_TIMEOUT_SECONDS` | `10` | maximum wait for each request-body chunk; slow clients receive `408` |
| `METHODOS_TOKEN_BUDGET_TOTAL` | — | optional cumulative LLM token ceiling per `tenant:graph` scope; must be paired with input ceiling |
| `METHODOS_TOKEN_BUDGET_INPUT` | — | optional per-request estimated input-token ceiling per `tenant:graph` scope; must be paired with total ceiling |
| `METHODOS_TOKEN_BUDGET_COST_USD` | — | optional cumulative estimated USD ceiling; must be paired with both per-1K token prices |
| `METHODOS_TOKEN_PRICE_INPUT_PER_1K_USD` | — | input-token price used by the cost ceiling |
| `METHODOS_TOKEN_PRICE_OUTPUT_PER_1K_USD` | — | output-token price used by the cost ceiling |

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
METHODOS_API_KEYS='replace-with-a-long-key' \\
METHODOS_API_KEY_ROLES='replace-with-a-long-key=reader|writer|operator' \\
docker compose -f docker/compose.yml up methodos
```

The Compose file defaults to `METHODOS_ENV=production` and fails closed until
API keys and explicit role mappings are supplied. For an intentionally
unauthenticated local Compose sandbox, set `METHODOS_ENV=development` explicitly
and do not expose the port beyond the local host.

### Production checklist

- [ ] Set `OPENAI_API_KEY` (or equivalent) in the deployment environment.
- [ ] Set `METHODOS_API_KEY` or `METHODOS_API_KEYS` for any non-local deployment; `/health` and
      `/metrics` remain public for probes and scraping.
- [ ] Set `METHODOS_ENV=production` (or `staging`) so startup rejects missing
      authentication and process-local filesystem persistence.
- [ ] Use API and tenant keys with at least 16 characters; shared deployment
      modes reject shorter credentials at startup.
- [ ] For least privilege, set `METHODOS_API_KEY_ROLES` and assign readers,
      writers, and operators explicitly; unspecified keys retain the legacy
      writer role only in development/test. Staging/production reject missing
      role assignments.
- [ ] For multiple tenants, set `METHODOS_TENANT_KEYS` and send the matching
      `X-API-Key`; the server derives the tenant and rejects mismatched
      `X-Tenant-ID` headers. Do not use a shared `METHODOS_API_KEY` in this mode.
- [ ] For durable graph-scoped LLM quotas, set both token budget variables. Use
      SQLite for a single service process group or Postgres for horizontally
      scaled deployments; filesystem mode rejects this configuration.
- [ ] If spend control is required, set the USD ceiling and both provider price
      variables. Keep the price table aligned with the provider/model contract;
      partial pricing configuration fails startup.
- [ ] Mount a persistent volume at `/data` for `PGRAPH_HOME` (the SQLite
      database lives here).
- [ ] Keep the default API host binding on loopback (`127.0.0.1:8000`). Put an
      authenticated TLS reverse proxy or load balancer in front of it before
      intentionally exposing the service externally.
- [ ] Preserve the Compose runtime hardening: non-root user, read-only root
      filesystem, writable `/data` volume only, dropped Linux capabilities,
      `no-new-privileges`, a constrained `/tmp` tmpfs, an init process, a
      30-second graceful shutdown window, and a 256-process limit.
- [ ] Keep the optional observability profile's Prometheus and OTLP host ports
      bound to loopback, or replace them with an authenticated internal
      network/export path before exposing them outside the host.
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
| POST | `/v1/graphs/{id}/evolve` | `{k_rounds}` + optional `Idempotency-Key` | 202 queued / 429 quota exceeded / 501 without runner or idempotency capability |
| GET | `/v1/evolution/jobs/{job_id}` | — | job status / 404 / 501 |
| POST | `/v1/graphs/{id}/rollback/{tag}` | `If-Match` + operator role | 200 / 404 / 409 |

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
docker compose -f docker/compose.yml --profile observability up
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

To restore a verified SQLite backup into a new database without overwriting an
existing path:

```bash
methodos restore-backup /secure/backups/methodos-20261014090000.db /secure/recovery/methodos.db
```

The destination must not already exist. Point the application at the recovered
database only after validating it and completing the appropriate cutover.

### Durable evolution queue

The HTTP factory deliberately does not create a scheduler or worker implicitly.
Production hosts should construct a durable runner and supervise its worker
process explicitly. The store-backed runner can reject overload atomically per
tenant and per graph:

```python
from methodos.jobs import (
    EvolutionJobWorker,
    SQLiteEvolutionJobStore,
    StoreBackedEvolutionRunner,
)
from methodos.service import create_app

jobs = SQLiteEvolutionJobStore("/data/evolution-jobs.db")
runner = StoreBackedEvolutionRunner(
    jobs,
    max_active_jobs=16,
    max_active_jobs_per_graph=2,
)
app = create_app(evolution_runner=runner)
```

Use `PostgresEvolutionJobStore` for horizontally scaled deployments. Run an
`EvolutionJobWorker` with the same store in a separately supervised worker
process and inject an environment-specific `EvolutionJobExecutor`; the API
process should not execute LLM work inline.

## Migration (planned for v2)

`schema_version: Literal[1]` is currently fixed. Repository startup uses a
forward-only migration runner, but no version 2 migration exists yet. Treat
schema_version as the only supported version and rebuild graphs via the CLI on
upgrade; do not manually edit production databases.

## Scaling

Single-process by default. SQLite is suitable for one service instance with a
local persistent volume; do not place the database on NFS or share it between
independent writers. For horizontal scaling, use the transactional Postgres
`Repository` backend, its shared rate-limit store, and a durable tenant-aware
job runner. Tenant-key mode carries the authenticated tenant into graph
namespaces and evolution-job ownership; external identity lifecycle and
deployment policy remain host responsibilities.

The evolution loop is CPU/LLM-bound, not I/O-bound. For multi-tenant
deployments, run one EvolutionEngine per tenant to avoid
rejection-memory collisions.
