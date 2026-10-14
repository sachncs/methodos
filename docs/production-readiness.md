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
- Dependabot tracks Python, site, and GitHub Actions dependencies weekly, and
  tagged releases verify the Git tag matches the Python package version.
- Tagged releases are validated and published by `.github/workflows/release.yml`.
- An optional pooled Postgres repository supports horizontal service replicas;
  its startup schema is idempotent and lifecycle-managed by FastAPI.
- Postgres CRUD, compare-and-swap, snapshots, trajectories, and cursor
  iteration run against a disposable database in CI and can be run locally.
- Postgres- and SQLite-backed request windows are shared atomically across
  service processes; filesystem retains the documented process-local limiter.
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
- Filesystem, SQLite, and Postgres repositories can persist immutable
  evolution round events, including outcome and score deltas; approval and
  promotion policy remains an explicit service concern.
- LLM attempts emit model/attempt OpenTelemetry spans and cancellation now has
  an explicit rollout regression test; collector/exporter policy remains a
  deployment concern.
- SQLite and Postgres repository bootstrap now use a contiguous, forward-only
  migration runner; future schema versions still require a reviewed migration
  and corresponding data-model compatibility tests.
- The local release gate exercises parallel HTTP reads and compare-and-swap
  writers through FastAPI, alongside concurrent evolution-job claims and a
  worker storage-outage recovery test.
- The Astro browser suite runs axe-core accessibility-budget checks for both
  desktop and mobile projects; the public product palette now satisfies the
  audited WCAG contrast checks.
- Playwright also stores deterministic desktop/mobile product screenshots with
  a small cross-platform raster tolerance, so layout drift is reviewed in CI.
- `SQLiteEvolutionJobStore` now persists queued-job state across process
  restarts and atomically deduplicates idempotency keys; the optional
  `PostgresEvolutionJobStore` provides the same contract for horizontal
  deployments. `EvolutionJobWorker` now atomically claims jobs and records
  success, failure, and cancellation around an injected executor; persisted
  leases, bounded retries, and expired-lease recovery prevent abandoned jobs
  from remaining stuck indefinitely. Queue counts are exposed through
  Prometheus metrics, and the worker has a cancellable supervised loop with
  heartbeat/error state that survives transient storage failures.
- The evolution API verifies graph ownership before queueing work and maps
  idempotency-key reuse conflicts to a stable HTTP 409 response. Durable
  SQLite/Postgres runners now bind jobs and idempotency keys to tenants and
  hide status from other tenants; configurable active-job quotas reject work
  atomically with HTTP 429 before a tenant or graph can overload the queue. External
  runners must implement the same capability before tenant mode accepts them.
- SQLite now uses an atomic WAL-backed rate-limit window shared by all local
  service processes; filesystem deployments remain process-local by design.
- `SQLiteTokenBudget` and the optional `PostgresTokenBudget` provide injectable,
  atomic durable token budgets for local and horizontally scaled deployments;
  `ScopedTokenBudget` adds explicit graph-capable scope routing, and the
  service wires durable scoped budgets from paired deployment settings.
  Provider-specific cost pricing remains open; durable active-job quotas can be
  configured per tenant and per graph.
- `SQLiteCircuitBreaker` and the optional `PostgresCircuitBreaker` provide
  shared async breaker state with an atomic half-open probe lease; gateway
  policy and provider-specific breaker tuning remain deployment concerns.

## Open issues

### Runtime and API

1. REST evolution now accepts an injected host-owned `EvolutionRunner`, returns
   a queued job contract, supports authenticated status polling, and exposes a
   fail-closed `Idempotency-Key` capability for runners that guarantee
   deduplication; the default app still returns 501 without a durable runner,
   and production scheduler ownership, external-runner tenant capability, and
   alert-threshold tuning remain open;
   starter queue and worker alerts are now versioned.
2. Distributed rate limiting and durable active-job quotas are implemented for
   Postgres and SQLite backends; token-cost quotas and shared limiting for
   filesystem deployments remain open.
3. Tenant-key authentication, server-derived tenant identity, and optional
   reader/writer/operator API-key roles now exist; external identity-provider
   integration and role lifecycle management remain open.
4. Tenant graph namespaces are isolated in tenant-key mode; a shared durable
   repository and per-tenant ownership metadata are still required for
   horizontal deployments.
5. CORS origins are validated at startup; deployment-specific allowlists still
   need operational review when domains change.
6. The LLM client now enforces its timeout at the application boundary and
   preserves task cancellation, covered by a direct cancellation test; full
   ASGI/HTTP disconnect testing remains open.
7. Error responses use a documented envelope with stable codes and request IDs.
   The service rejects oversized declared and streamed request bodies with a
   bounded `413` response; reverse-proxy limits remain an additional perimeter
   control.

### Persistence and data integrity

8. Graph revisions and compare-and-swap updates now prevent stale HTTP writers
   across filesystem, SQLite, and Postgres repositories; broader contention
   testing remains open.
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
    and now supports a non-overwriting `restore-backup` workflow that verifies
    the source and published destination.

### LLM and evolution

14. Provider retry classification now maps LiteLLM's transient connection,
    throttling, gateway, and server exception classes separately from
    permanent client/auth/request classes, then falls back to status codes;
    provider-specific policy tuning remains open. An ordered failover client
    provides an application-level fallback strategy.
15. Local and optional shared SQLite/Postgres circuit breakers now protect
    workers from retry storms with an atomic half-open probe; shared gateway
    policy and provider-specific breaker tuning remain open.
16. Injectable per-client/tenant token budgets now enforce estimated input and
    cumulative request limits before dispatch. `ScopedTokenBudget` adds an
    explicit graph-capable scope routed through the polymorphic LLM contract;
    SQLite-backed durable accounting and Postgres-backed accounting remain
    available; paired deployment settings now wire scoped budgets into the
    default service LLM, while provider-specific cost pricing remains open.
17. Evolution round decisions are now durably auditable across supported
    repositories; human approval, identity, and promotion policy are still
    open.
18. Backend snapshot restore now provides revision-safe graph rollback; an
    authenticated release action, approval record, and promotion workflow are
    still open.
19. Live provider tests are skipped without API credentials.

### Observability and operations

20. Request metrics, OTEL spans, and structured completion events are now
    correlated with the request ID; exporter and retention policy remain
    deployment responsibilities.
21. HTTP, repository-boundary, and LLM spans are instrumented; low-cardinality
    LLM attempt outcome counters are exported through `/metrics`. Prometheus
    and OpenTelemetry Collector configuration syntax are now validated in CI;
    collector deployment and exporter compatibility remain operational
    responsibilities.
22. Baseline SLOs, starter alert thresholds, and an incident runbook are now
    versioned in `docs/operations.md` and `docker/prometheus-rules.yml`;
    dashboards, paging ownership, retention, and error-budget policy remain
    deployment responsibilities.
23. Concurrent SQLite reader/CAS-writer and Postgres transactional tests now
    cover basic contention, concurrent job claiming, and a parallel HTTP smoke
    test plus deterministic worker fault injection; sustained load, soak,
    chaos, file-corruption, and broader cancellation tests remain open.
24. CI scans the Docker image; tagged releases now publish, attest, and
    keylessly sign the immutable GHCR image digest. Registry retention and
    independent signature verification remain operational controls.

### Delivery and governance

25. Release tags are validated and published automatically; Dependabot now
    tracks Python, npm, and GitHub Actions updates. Merges still require
    human review of dependency changes.
26. Changelog and compatibility policy are documented in `CHANGELOG.md` and
    `docs/compatibility.md`; migration tooling for future schema versions is
    still open.
27. The redesigned site is deployed and publicly verified at the GitHub Pages
    URL.
28. Browser regression smoke tests, axe-core accessibility-budget checks, and
    product-page visual snapshots run in CI for desktop and mobile; broader
    multi-page visual coverage remains open.

## Remediation plan

### Phase 1 — release gate

- Done: Astro migration, GitHub Pages deployment, responsive product redesign,
  logo, browser smoke tests, axe-core checks, visual snapshots, SBOM,
  vulnerability scanning, coverage, Google-style linting, enforced formatting,
  typing, and container gates.
- Remaining: add broader multi-page visual coverage if the site grows beyond
  the current product page; require branch protection and review ownership in
  repository settings.

### Phase 2 — service contract

- Done: stable error envelope, API-key authentication and roles, CORS
  validation, shared SQLite/Postgres rate-limit windows, tenant namespaces,
  tenant-aware jobs, active-job admission quotas, idempotency, and queued
  evolution status polling.
- Remaining: integrate an external identity provider, manage key/role
  lifecycle, configure provider pricing for scoped graph-level token budgets,
  and make production scheduler
  ownership explicit for each deployment.

### Phase 3 — durable data layer

- Done: forward-only migrations, revision-safe CAS, SQLite backup creation,
  verification and non-overwriting restore, snapshot restore, filesystem
  corruption quarantine, and
  SQLite/Postgres integration paths.
- Remaining: expand cross-process contention tests, run scheduled SQLite
  restore drills, document retention/RPO/RTO, and validate managed-Postgres
  restore procedures.

### Phase 4 — dependable intelligence

- Done: provider error classification, bounded retries, failover, local and
  shared circuit breakers, cancellation propagation, durable and scoped token budgets,
  audit events, snapshots, and revision-safe rollback.
- Remaining: add provider-SDK-specific mappings, pricing-aware cost budgets,
  human approval/identity records, signed release promotion, and explicit
  policy for which tenant may promote a graph.

### Phase 5 — operations

- Done: request-correlated structured logs, OTEL spans, Prometheus metrics,
  SLOs, alert rules, Prometheus and Collector syntax validation, runbook,
  fault-injection tests, and container signing/attestation.
- Remaining: assign dashboard/paging ownership, define retention and
  error-budget policy, run sustained load/soak/chaos tests, add full HTTP
  disconnect tests, enable credentialed live-provider suites, and independently
  verify release signatures.
