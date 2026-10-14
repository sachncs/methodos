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
- Python tests, Ruff, mypy, Astro checks, npm audit, browser smoke tests, and
  Docker builds run in local verification and CI.
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
- API errors use a stable envelope with error codes and request IDs.
- Graph updates use optimistic concurrency via revision tokens and `If-Match`.
- CI generates an SBOM, scans the container, retains the SBOM artifact, and
  runs desktop/mobile browser smoke tests.
- Tagged releases publish the runtime image to GHCR with BuildKit SBOM and
  provenance attestations, then sign the immutable digest with keyless Cosign.
- Tagged releases are validated and published by `.github/workflows/release.yml`.
- An optional pooled Postgres repository supports horizontal service replicas;
  its startup schema is idempotent and lifecycle-managed by FastAPI.
- Postgres CRUD, compare-and-swap, snapshots, trajectories, and cursor
  iteration run against a disposable database in CI and can be run locally.
- Postgres-backed request windows are shared atomically across service
  replicas; local backends retain the documented process-local limiter.
- Static API authentication supports overlapping keys for zero-downtime
  rotation; external identity and role management remain deployment work.
- The core CLI no longer imports optional benchmark dependencies until the
  `eval` command is invoked.
- SQLite has an online, atomic, integrity-checked backup capability exposed
  through `methodos backup`, plus `methodos verify-backup` for recovery
  validation; scheduling, retention, and restore drills remain operational
  responsibilities.
- CI cancels stale runs for the same branch or pull request so deployment
  evidence comes from the newest revision.
- Evolution rollouts accept a polymorphic `ActionExecutor` strategy, so hosts
  can wire tools without monkeypatching module globals.
- LLM attempts emit model/attempt OpenTelemetry spans and cancellation now has
  an explicit rollout regression test; collector/exporter policy remains a
  deployment concern.
- SQLite and Postgres repository bootstrap now use a contiguous, forward-only
  migration runner; future schema versions still require a reviewed migration
  and corresponding data-model compatibility tests.

## Open issues

### Runtime and API

1. The REST evolution endpoint still returns 501 because a remote host Solver
   contract has not been designed.
2. Distributed rate limiting is implemented for the Postgres backend; quota
   enforcement and shared limiting for SQLite/filesystem deployments remain
   open.
3. Tenant-key authentication and server-derived tenant identity now exist;
   roles and an external identity provider are still open.
4. Tenant graph namespaces are isolated in tenant-key mode; a shared durable
   repository and per-tenant ownership metadata are still required for
   horizontal deployments.
5. CORS origins are validated at startup; deployment-specific allowlists still
   need operational review when domains change.
6. The LLM client now enforces its timeout at the application boundary and
   preserves task cancellation; full HTTP disconnect testing remains open.
7. Error responses use a documented envelope with stable codes and request IDs.

### Persistence and data integrity

8. Graph revisions and a SQLite/filesystem compare-and-swap update API now
   prevent stale HTTP writers; distributed repositories still need the same
   contract.
9. Postgres support is implemented as an optional pooled backend with local
   and CI integration coverage; managed Postgres backups and restore drills
   remain operational responsibilities.
10. Migration infrastructure is implemented for forward-only SQL upgrades;
    no schema version 2 migration is currently needed or defined.
11. Filesystem trajectory appends are not a multi-process journal and should
    remain development-only.
12. SQLite backup creation and verification are available; scheduling,
    retention policy, managed-Postgres restore drills, and a full disaster-
    recovery runbook remain open.
13. Filesystem JSON graph/trajectory corruption now raises a stable error and
    preserves the bad record under `corrupt/`; SQLite corruption is classified
    but still requires operator recovery from a verified backup.

### LLM and evolution

14. Provider retry classification now distinguishes transient transport,
    throttling, and server errors from permanent client/auth failures; provider
    SDK-specific mappings remain open. An ordered failover client now provides
    an application-level fallback strategy.
15. A process-local circuit breaker exists and failover is covered in-process;
    shared-state breaker/failover policy across workers remains open.
16. Injectable per-client/tenant token budgets now enforce estimated input and
    cumulative request limits before dispatch; durable distributed accounting,
    provider-specific cost pricing, and graph-level quotas remain open.
17. Evolution proposals lack a durable approval/audit workflow.
18. Backend snapshot restore now provides revision-safe graph rollback; an
    authenticated release action, approval record, and promotion workflow are
    still open.
19. Live provider tests are skipped without API credentials.

### Observability and operations

20. Request metrics, OTEL spans, and structured completion events are now
    correlated with the request ID; exporter and retention policy remain
    deployment responsibilities.
21. HTTP, repository-boundary, and LLM spans are instrumented; collector
    configuration validation remains open.
22. No SLOs, alert thresholds, dashboards, or error-budget policy are defined.
23. Concurrent SQLite reader/CAS-writer and Postgres transactional tests now
    cover basic contention; load, soak, chaos, file-corruption, and broader
    cancellation tests remain open.
24. CI scans the Docker image; tagged releases now publish, attest, and
    keylessly sign the immutable GHCR image digest. Registry retention and
    independent signature verification remain operational controls.

### Delivery and governance

25. Release tags are validated and published automatically; dependency or
    version-number bump automation is not yet enforced.
26. Changelog and compatibility policy are documented in `CHANGELOG.md` and
    `docs/compatibility.md`; migration tooling for future schema versions is
    still open.
27. The redesigned site is deployed and publicly verified at the GitHub Pages
    URL.
28. Browser regression smoke tests are part of CI; broader visual-diff and
    accessibility-budget enforcement remain open.

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
- Add tenant ownership to durable graph records and enforce authorization at
  every repository operation in the shared backend.
- Design a safe REST evolution job model with idempotency keys and status
  polling.

### Phase 3 — durable data layer

- Extend graph revision tokens and compare-and-swap writes to every durable
  backend.
- Add reviewed schema migrations and data-model compatibility tests whenever
  schema version 2 is introduced.
- Add migration runners and backup/restore verification around the
  transactional repository; Postgres integration tests are now in place.
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
