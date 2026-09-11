# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Service hardening
- `methodos.auth.APIKeyAuth`: optional bearer-token authentication, gated
  by `PGRAPH_API_KEY`. Supports `Authorization: Bearer <key>` and
  `X-API-Key: <key>` headers; constant-time comparison via
  `hmac.compare_digest`. Returns `HTTP 401` with `WWW-Authenticate: Bearer`
  on missing/invalid keys. No-op when `PGRAPH_API_KEY` is unset.
- `methodos.rate_limit.TokenBucketLimiter`: per-process token-bucket
  rate limiter with idle eviction. Default `60` capacity, `1.0`
  tokens/sec refill; configurable via `PGRAPH_RATE_LIMIT_*`. Subject is
  the API key when auth is enabled, otherwise the client IP. Returns
  `HTTP 429` with `Retry-After` on exhaustion.
- `methodos.observability`: structured JSON and human-readable log
  formatters (formerly `methodos.logging_setup`). JSON output emits
  `timestamp` (UTC ISO-8601 with `Z` suffix), `level`, `logger`,
  `message`, plus any `extra={...}` fields. Stdlib `logging` is used so
  existing call sites work unchanged. Selectable from the CLI via
  `methodos serve --log-json`.
- `methodos.observability`: Prometheus instruments exposed at `GET /metrics`
  (formerly `methodos.metrics`):
  - `methodos_requests_total` (counter, labels: `method`, `path`, `status`)
  - `methodos_request_duration_seconds` (histogram, labels: `method`, `path`)
  - `methodos_guidance_cache_total` (counter, labels: `result`)
  - `methodos_llm_attempts_total` (counter, labels: `outcome`, `model`)
  - `methodos_evolution_rounds_total` (counter, labels: `outcome`)
  - `methodos_info` (gauge, labels: `version`, `component`)
  Path labels use the FastAPI route template (e.g. `/v1/graphs/{graph_id}`)
  to keep cardinality bounded.
- New endpoints on `methodos.service:app`:
  - `GET /health/live` — process-up probe (always 200 when up).
  - `GET /health/ready` — repo-reachable probe (200 when reachable, 503
    otherwise). Body: `{"status": "...", "checks": {"repo": "..."}}`.
  - `GET /metrics` — Prometheus exposition format.
- Exception handlers in `methodos.service`:
  - `LLMError` → `HTTP 502 {"detail": "upstream LLM call failed"}`.
  - Unhandled `Exception` → `HTTP 500 {"detail": "internal server error"}`
    with sanitized body (no traceback leakage).
- HTTP request metrics middleware that records every request to the
  counter + histogram with route-template path labels.

#### Tooling
- `tools/bump_version.py`: scripted version bumps for `pyproject.toml`
  and `methodos/__init__.py` in lockstep. Supports `--bump major|minor|patch`,
  explicit version, and `--check` for CI.
- `.github/workflows/release.yml`: tag-triggered release workflow that
  builds and pushes a multi-arch (`linux/amd64`, `linux/arm64`) Docker
  image to `ghcr.io/<owner>/methodos`, optionally publishes to PyPI,
  and creates a GitHub release with auto-generated notes.
- `.dockerignore`: keeps the build context lean (excludes tests, docs,
  eval harness, venvs, caches).

#### Documentation
- `CHANGELOG.md` (this file).
- `SECURITY.md`: vulnerability reporting policy, hardening posture,
  threat model summary.
- `README.md`: badges, hardening table, updated module map.
- `docs/deployment.md`: full runbook (startup, health, common failure
  modes, backups, upgrades), extended env var table, observability
  surface.
- `docs/api.md`: public API reference extended with `methodos.auth`
  and `methodos.rate_limit` sections; stability table extended for new
  modules.

#### Docker
- `docker/Dockerfile`: `ARG VERSION` build arg plumbed into
  `METHODOS_VERSION` env var so the running image carries its release tag.

### Changed
- `methodos.cli.serve`: new `--log-json` flag toggles structured JSON
  log output for the FastAPI process.
- `methodos.service.create_app`: signature extended with optional
  `auth`, `rate_limiter`, `log_json`, and `log_level` kwargs for
  programmatic wiring.

### Tests
- New test modules:
  - `tests/test_auth.py` — APIKeyAuth, extract_candidate, env factory,
    integration through `create_app`.
  - `tests/test_rate_limit.py` — TokenBucketLimiter (capacity, refill,
    idle eviction, subject selection, env factory).
  - `tests/test_observability.py` — text + JSON formatters, level
    validation, extra fields, exception rendering, reserved keys,
    Prometheus instrument smoke tests.
  - `tests/test_bump_version.py` — version bump round-trip + CLI flags.
- `tests/test_service.py` extended with `/health/live`, `/health/ready`,
  `/metrics`, `LLMError`→502, rate-limit 429, and auth integration.
- Total: **441 tests, 97.20% line coverage**.

## [0.1.0] - 2026-09-11

The first public release of `methodos`, an open-source reimplementation of
the procedural-graph adapter described in the "Procedural Graph Learning"
paper.

### Added

#### Core library
- `methodos.schema`: Pydantic v2 models — `ProceduralGraph`, `Node`, `Edge`,
  `Relation` (LEADS_TO, REQUIRES, REPLACES), `Attribute`, and the `Edit`
  discriminated union (AddNode, DeleteNode, AddEdge, DeleteEdge, UpdateAttr).
  All models enforce `extra="forbid"`; `ProceduralGraph` validates on
  assignment.
- `methodos.graph`: structural algorithms — `match_node`, `neighborhood`,
  `validate`, `apply_edits`, `has_path_to`, `infer_terminals`, `has_cycle`.
- `methodos.llm`: `LLMClient` Protocol + `LiteLLMClient` (multi-provider via
  litellm) + `LLMError`.
- `methodos.guidance`: `GUIDANCE_SYSTEM_PROMPT` and `generate_guidance()`
  per paper §3.2.
- `methodos.adapter`: `PGAdapter`, `Solver` Protocol, `GuidanceCache`,
  `AgentState`. Reference runtime that ties guidance, caching, and the
  graph together.
- `methodos.evolution`: `EvolutionEngine`, `RejectionMemory`, `RolloutResult`,
  refiner prompt + parsers, `score`/`mean_score`/`tail_concat`,
  `propose_edits`, `validate_candidate`.
- `methodos.repo`: `Repository` Protocol with two implementations
  (`FilesystemRepository` for dev, `SQLiteRepository` for prod) and an
  optional `SqliteVecIndex` for vector recall.
- `methodos.service`: FastAPI app factory exposing `/health`,
  `/health/live`, `/health/ready`, `/metrics`, and the `/v1/graphs*` API.

#### Hardening (this release)
- Optional API key auth (`methodos.auth.APIKeyAuth`) gated by
  `PGRAPH_API_KEY`. Supports `Authorization: Bearer <key>` and
  `X-API-Key` headers; constant-time comparison.
- Optional per-process token-bucket rate limiter
  (`methodos.rate_limit.TokenBucketLimiter`) gated by
  `PGRAPH_RATE_LIMIT_*`. Subject is the API key when auth is enabled,
  otherwise the client IP.
- Structured JSON logging (`methodos.observability.configure_logging`)
  selectable via
  CLI flag `--log-json` or `configure_logging(json_format=True)`.
- Prometheus metrics (`methodos.observability`): request counter, latency
  histogram, cache hit/miss, LLM attempts, evolution rounds, build
  info. Exposed at `GET /metrics`.
- Exception handlers: `LLMError` → 502, generic unhandled → 500
  (sanitized detail).
- New health endpoints: `/health/live` (process up) and
  `/health/ready` (repo reachable).

#### Tooling
- Typer CLI: `init`, `inspect`, `serve`, `replay`, `evolve`, `eval`.
- Eval harness: `eval/hotpotqa` for paired comparison runs.
- Multi-stage `docker/Dockerfile` (python:3.12-slim, non-root user,
  `/data` volume, healthcheck on `/health`).
- `docker/compose.yml` with optional observability profile (otel-collector,
  prometheus).
- CI workflow (`.github/workflows/ci.yml`): ruff, mypy, pytest with
  `--cov-fail-under=95` on every push/PR to main.
- Release workflow (`.github/workflows/release.yml`): tag-triggered
  multi-arch Docker image push to `ghcr.io/<owner>/methodos`.
- `tools/bump_version.py`: scripted version bumps for `pyproject.toml`
  and `methodos/__init__.py`.

### Documentation
- `README.md`: install, quickstart, CLI walkthrough, links.
- `docs/api.md`: full public API reference.
- `docs/architecture.md`: module map, data flow, evolution loop.
- `docs/deployment.md`: environment variables, observability, runbook.
- `docs/evaluation.md`: paired-comparison eval methodology.
- `docs/paper-mapping.md`: mapping between paper sections and code.
- `SECURITY.md`: vulnerability reporting policy.

### Notes
- Test suite: 427 tests, 97.2% line coverage (`uv run pytest`).
- Type checking: `uv run mypy methodos` clean.
- Linting: `uv run ruff check .` clean.
