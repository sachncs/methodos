# Production-readiness audit

This document records the repository-wide review performed against the local
checkout. Items marked **done** are implemented in the current worktree;
items marked **open** are the remaining work required before calling the
system production-ready.

## Done in this worktree

- Static product site migrated from React/Vite/Tailwind to Astro.
- GitHub Pages uses the official artifact/deploy workflow; no Jekyll files or
  Jekyll deployment flags remain.
- Product page, responsive layout, graph visual, and logo were redesigned.
- Python tests, Ruff, mypy, Astro checks, npm audit, and Docker builds run in
  local verification and CI.
- API trajectory pairs, graph IDs, storage components, and request payloads
  have explicit validation.
- Duplicate graph creation uses an atomic repository capability where
  available and returns HTTP 409.
- SQLite rejects unsupported schema versions instead of silently operating on
  an unknown database.
- LLM requests have timeouts, bounded output, retry limits, and exponential
  backoff.
- Optional `sqlite-vec` no longer prevents the base production image from
  starting.
- `/health`, `/ready`, and `/metrics` are available; request IDs, security
  headers, request counters, and latency histograms are emitted.
- Optional API-key authentication and explicit CORS origins are supported.
- Optional bounded per-process request limiting returns 429 with `Retry-After`.
- Engineering boundaries and structural polymorphism are documented in
  `docs/engineering.md`.

## Open issues

### Runtime and API

1. The REST evolution endpoint still returns 501 because a remote host Solver
   contract has not been designed.
2. The built-in limiter is per-process only; there is no distributed rate
   limiter or quota enforcement.
3. API-key authentication is a shared static key, not identity, role, or
   tenant-based authorization.
4. There is no tenant isolation model or per-tenant graph ownership.
5. CORS is configurable but there is no deployment-time policy validation.
6. Request cancellation and downstream LLM cancellation are not explicitly
   coordinated.
7. Error responses do not yet use one documented application error schema.

### Persistence and data integrity

8. Graph revisions have no optimistic-concurrency token or compare-and-swap
   update API.
9. SQLite is intentionally single-instance; a production Postgres backend is
   not implemented.
10. Schema migration infrastructure is only a fail-closed version check;
    migrations themselves are not implemented.
11. Filesystem trajectory appends are not a multi-process journal and should
    remain development-only.
12. There is no backup scheduler, restore verification, retention policy, or
    disaster-recovery runbook.
13. Corrupt JSON/SQLite records do not have a repair/quarantine workflow.

### LLM and evolution

14. Retry classification is still broad for provider-specific transient error
    types; provider-aware retry policy should be added.
15. There is no circuit breaker or provider failover policy.
16. Token/cost budgets are not persisted or enforced per request, graph, or
    tenant.
17. Evolution proposals lack a durable approval/audit workflow.
18. Candidate graph rollback is not exposed as a first-class release action.
19. Live provider tests are skipped without API credentials.

### Observability and operations

20. Metrics are emitted, but traces and structured application logs are not
    fully correlated with the request ID.
21. OpenTelemetry collector configuration exists, but application spans are
    not instrumented end-to-end.
22. No SLOs, alert thresholds, dashboards, or error-budget policy are defined.
23. No load, soak, chaos, file-corruption, database-lock, or cancellation
    tests exist.
24. The Docker image is built and smoke-tested locally, but image scanning,
    SBOM generation, signing, and provenance attestation are not enforced.

### Delivery and governance

25. No automated release/versioning workflow exists.
26. No changelog or compatibility policy is enforced for schema and public API
    changes.
27. The public GitHub Pages URL must still receive a remote push before the
    redesigned site is externally visible.
28. Live browser regression testing is not yet part of CI.

## Remediation plan

### Phase 1 — release gate

- Push the Astro migration and verify the deployed URL.
- Add a browser smoke test for desktop/mobile layout, navigation, links, and
  accessibility landmarks.
- Add Docker image scanning, SBOM, and CI artifact retention.
- Require Python, frontend, container, and coverage checks on pull requests.

### Phase 2 — service contract

- Define a versioned error envelope and documented authentication model.
- Add rate limiting and quotas backed by a shared store.
- Add tenant ownership to graph records and enforce authorization at every
  repository operation.
- Design a safe REST evolution job model with idempotency keys and status
  polling.

### Phase 3 — durable data layer

- Add graph revision tokens and compare-and-swap writes.
- Introduce migration runners and migration tests.
- Implement a transactional Postgres repository.
- Add backup/restore verification and corruption handling.

### Phase 4 — dependable intelligence

- Classify provider errors, add circuit breakers and fallback providers.
- Enforce token/cost budgets and cancellation propagation.
- Add auditable evolution proposals, approvals, snapshots, rollback, and
  release promotion.

### Phase 5 — operations

- Instrument traces and structured logs with request correlation.
- Define SLOs and alerts for latency, error rate, LLM failures, queue depth,
  storage health, and evolution regressions.
- Add load, soak, fault-injection, and live-provider test environments.
