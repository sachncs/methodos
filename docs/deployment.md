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
| `PGRAPH_API_KEY` | — | when set, required on every `/v1/*` request as `Authorization: Bearer <key>` or `X-API-Key: <key>` |
| `PGRAPH_RATE_LIMIT_CAPACITY` | `60` | max tokens per bucket (per process) |
| `PGRAPH_RATE_LIMIT_REFILL_PER_SECOND` | `1.0` | token refill rate |
| `PGRAPH_RATE_LIMIT_DISABLED` | `0` | `1` disables rate limiting |
| `OPENAI_API_KEY` | — | required for OpenAI / litellm |
| `ANTHROPIC_API_KEY` | — | for Anthropic via litellm |
| `GOOGLE_API_KEY` | — | for Google via litellm |
| `GROK_API_KEY` | — | for Grok via litellm |

## CLI

```bash
methodos init                              # create default graph
methodos inspect                           # show summary
methodos serve --host 0.0.0.0 --port 8000  # start FastAPI
methodos serve --log-json                  # JSON-formatted logs
methodos replay                            # print recent trajectories
methodos eval hotpotqa --n 200             # run paired HotpotQA eval
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
- [ ] Set `PGRAPH_API_KEY` to a high-entropy value (>= 32 random bytes).
      Rotate by changing the env var and restarting the service.
- [ ] Mount a persistent volume at `/data` for `PGRAPH_HOME` (the SQLite
      database lives here).
- [ ] If using sqlite-vec, ensure the SQLite build in your image supports
      extensions (most pre-built Python wheels do).
- [ ] Behind a reverse proxy with TLS termination; the service itself is
      plaintext HTTP and the API key is sent in headers.
- [ ] Scrape `/health/live` for liveness probes, `/health/ready` for
      readiness probes (rejects with 503 when the repo is unreachable).
- [ ] Scrape `/metrics` (Prometheus exposition) every 15-30s.
- [ ] Send `methodos.service:app` access logs to your log aggregator
      (structured JSON via `--log-json`).

### Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | — | legacy liveness; returns `{"status": "ok"}` |
| GET | `/health/live` | — | process-up probe |
| GET | `/health/ready` | — | repo reachable (200) or 503 |
| GET | `/metrics` | — | Prometheus exposition |
| GET | `/v1/graphs/{id}` | yes | `ProceduralGraph` or 404 |
| POST | `/v1/graphs` | yes | `{id, from_graph_id?}` (201) |
| POST | `/v1/graphs/{id}/guidance` | yes + rate-limited | `{query, trajectory?, window?, guidance_hops?}` -> `{guidance}` |
| POST | `/v1/graphs/{id}/evolve` | yes | 501 (reserved; use SDK) |

Auth is enforced on every `/v1/*` route when `PGRAPH_API_KEY` is set.
Rate limiting is enforced on `/v1/graphs/{id}/guidance` when
`PGRAPH_RATE_LIMIT_DISABLED != 1`.

### Example: request guidance

```bash
curl -sX POST http://localhost:8000/v1/graphs/my-graph/guidance \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${PGRAPH_API_KEY}" \
  -d '{
    "query": "What is the capital of France?",
    "trajectory": [["search", "Paris is the capital"]],
    "window": 3,
    "guidance_hops": 2
  }'
```

A `Retry-After` header is returned with HTTP 429 responses from the rate
limiter; clients should respect it.

## Observability

### Prometheus metrics

`GET /metrics` returns the default `prometheus_client` registry. The
service exposes:

| Metric | Type | Labels |
|---|---|---|
| `methodos_requests_total` | counter | `method`, `path`, `status` |
| `methodos_request_duration_seconds` | histogram | `method`, `path` |
| `methodos_guidance_cache_total` | counter | `result` (`hit`/`miss`) |
| `methodos_llm_attempts_total` | counter | `outcome`, `model` |
| `methodos_evolution_rounds_total` | counter | `outcome` |
| `methodos_info` | gauge | `version`, `component` |

Path labels use FastAPI's route template (e.g., `/v1/graphs/{graph_id}`)
so high-cardinality graph IDs do not explode the metric cardinality.

### Structured logging

`methodos.service:app` emits one log line per request and one per LLM
failure. Default format is human-readable; pass `--log-json` to the
`serve` subcommand (or call `configure_logging(json_format=True)`)
to emit structured JSON suitable for Loki, ELK, or Datadog.

The JSON payload includes `timestamp` (UTC ISO-8601 with `Z` suffix),
`level`, `logger`, `message`, plus any `extra={...}` fields passed at
the call site.

### OpenTelemetry (compose profile)

The `docker/compose.yml` defines a profile `observability` that starts
OpenTelemetry Collector and Prometheus alongside the main service:

```bash
docker compose --profile observability up
```

This brings up:
- `methodos:8000` (the service, with `/metrics` exposed)
- `otel-collector:4317` (OTLP gRPC; the service is not yet exporting
  traces — Prometheus metrics cover the v0.1 observability surface)
- `prometheus:9090` (scrapes `/metrics`)

## Persistence

Default backend: SQLite with WAL mode and FK constraints. WAL allows
concurrent reads during the evolution loop's diagnostic-rollout phase.
Each graph is stored as JSON in the `graphs` body column; trajectories
are stored as JSONL-equivalent in the `trajectories` body column.

To back up: copy `~/.methodos/methodos.db` (and the WAL/SHM sidecars
if present) using `sqlite3 methodos.db ".backup backup.db"` for a
consistent snapshot.

To restore: place `methodos.db` (and sidecars) at `PGRAPH_HOME` and
restart the service.

## Runbook

### Startup

1. Choose a deployment mode: container (recommended) or bare Python.
2. Set `PGRAPH_HOME` to a persistent directory; the SQLite database
   is created on first write.
3. Set `OPENAI_API_KEY` (or the provider equivalent).
4. Optional: set `PGRAPH_API_KEY` to require auth on `/v1/*`.
5. Start the service. Container: `docker compose up methodos`. Bare:
   `uvicorn methodos.service:app --host 0.0.0.0 --port 8000`.

### Health

- `GET /health/live` — returns 200 when the process is up. Use this for
  Kubernetes `livenessProbe`.
- `GET /health/ready` — returns 200 when the repo is reachable, 503
  otherwise. Use this for `readinessProbe`. The body is
  `{"status": "ok"|"error", "checks": {"repo": "ok"|"error"}}`.

### Common failure modes

| Symptom | Likely cause | Mitigation |
|---|---|---|
| 502 on guidance endpoint | Upstream LLM provider error | Check `methodos.llm_attempts_total{outcome="exhausted"}`; the request has exhausted litellm retries |
| 401 on `/v1/*` | Missing/invalid `PGRAPH_API_KEY` | Set the same key on caller and server |
| 429 on guidance | Rate limit exceeded | Back off per `Retry-After`; raise `PGRAPH_RATE_LIMIT_CAPACITY` if legitimate traffic |
| 503 on `/health/ready` | SQLite unreachable | Check disk space, file permissions on `PGRAPH_HOME`; restart may be required |
| 500 on any route | Unhandled exception | Check container logs (`docker logs <id>`); the response body is sanitized to avoid leaking internals |
| Slow guidance responses | LLM latency dominates | Track `methodos_request_duration_seconds{path="/v1/graphs/{graph_id}/guidance"}`; consider caching guidance outputs in your application layer |

### Backups

Schedule a periodic snapshot of the SQLite database:

```bash
sqlite3 ~/.methodos/methodos.db ".backup /backups/methodos-$(date +%Y%m%d).db"
```

This works while the service is running (WAL mode holds a write lock
briefly and releases).

### Upgrades

1. Pull the new image / install the new wheel.
2. Stop the existing process (send `SIGTERM`, wait for `/health/live` to
   fail; uvicorn drains in-flight requests within ~30s).
3. Start the new process.
4. Verify with `GET /health/ready` and a smoke-test guidance call.

Rollback: stop the new process, start the old one. SQLite migrations
are forward-only in 0.x; rolling back to a much older version may
require rebuilding graphs via the CLI.

## Migration (planned for v2)

`schema_version: Literal[1]` is currently fixed. Bumping to v2 will add a
migration runner in `methodos.repo.migrate`. For now, treat schema_version
as the only supported version and rebuild graphs via the CLI on upgrade.

## Scaling

Single-process by default. The FastAPI app is stateless and can be
horizontally scaled; persistence is shared (SQLite via NFS, or move to
Postgres via the `Repository` Protocol — Postgres impl is on the roadmap).

The bundled rate limiter is **per-process**. Behind a load balancer, each
replica enforces its own quota; the aggregate limit is `N x per-replica`.
Operators needing a cluster-wide limit should front the service with a
shared store (e.g., Redis + `slowapi`).

The evolution loop is CPU/LLM-bound, not I/O-bound. For multi-tenant
deployments, run one EvolutionEngine per tenant to avoid
rejection-memory collisions.
