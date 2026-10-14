# Operations baseline

This is the minimum operational contract for a Methodos deployment. Values are
starting SLOs and should be tuned after observing representative traffic.

## SLOs

- Availability: 99.9% successful HTTP responses over a rolling 30-day window,
  excluding intentional 4xx responses and planned maintenance.
- Latency: 95th-percentile HTTP latency below one second over five minutes for
  requests that do not wait on an external evolution job.
- Provider reliability: fewer than 20% failed LLM attempts over a rolling
  five-minute window.
- Recovery: restore a verified SQLite backup or promote a healthy Postgres
  replica within the deployment's documented recovery-time objective.

The first three signals are emitted by `/metrics` and have starter alert rules
in [`docker/prometheus-rules.yml`](../docker/prometheus-rules.yml). A starter
Grafana dashboard is available at
[`docker/grafana-dashboard.json`](../docker/grafana-dashboard.json); import it
with a Prometheus datasource selected for `${DS_PROMETHEUS}`. Alert delivery,
paging ownership, retention, and error-budget burn calculations are deployment
responsibilities.

When a host configures `EvolutionJobWorker`, the same endpoint emits queue
backlog and worker-heartbeat signals. Enable the evolution alerts only for
deployments that run a worker; the REST service alone intentionally does not
claim to execute queued jobs.

## Recovery objectives and drill contract

Every deployment must record these values before accepting production traffic:

| Backend | RPO target | RTO target | Required recovery source |
|---|---|---|---|
| SQLite | backup interval plus the last successful backup age | volume restore plus service readiness | verified `methodos backup` artifact |
| Postgres | provider WAL/backup policy | provider recovery or replica promotion | provider-verified dump/restore or failover drill |

The targets are deployment commitments, not guarantees supplied by the
library. Record the owner, backup location, retention period, last successful
drill timestamp, measured restore duration, and recovered revision alongside
the release record. A drill is successful only when it proves all of the
following:

1. The source artifact is integrity-checked before it is used.
2. Recovery is performed into a separate destination or disposable database;
   the live database is never overwritten during the test.
3. A known canary graph and its latest expected revision can be loaded after
   recovery.
4. The service reaches `/ready` against the recovered store.
5. The measured RPO/RTO is within the deployment's recorded targets.

For SQLite, use the repository drill and record its output:

```bash
uv run python scripts/sqlite_restore_drill.py "$PGRAPH_HOME/methodos.db" \
  --graph-id "$METHODOS_CANARY_GRAPH_ID"
```

For managed Postgres, use the provider's disposable restore or replica
promotion workflow, then run the same canary/readiness checks against the
recovered DSN. Do not mark a provider backup as production-ready solely from
its dashboard status; retain the drill evidence with the release.

## Incident response

1. Check `/health`, `/ready`, `/metrics`, container logs, and the active
   `request_id` before restarting anything.
2. For provider failures, inspect LLM outcome counters and circuit-breaker
   state; failover is ordered and bounded, so do not increase retries during
   an incident without an explicit budget review.
3. For storage failures, stop writers, verify the latest backup, and restore
   to a separate path or database before promotion.
4. For corrupted filesystem records, preserve the generated `corrupt/` file,
   validate the replacement graph, and record the incident before resuming
   writes.
5. Record the triggering alert, request IDs, timeline, mitigation, and a
   follow-up test in the incident log.

## Required deployment checks

- Load the Prometheus rule file and verify alert expressions after every
  metrics-name change.
- Validate and review the dashboard queries after every metrics-name change;
  dashboard panels are intentionally low-cardinality and contain no request
  IDs, tenant IDs, graph IDs, or provider secrets.
- Run the bounded local HTTP load smoke check before release and after major
  service changes:

  ```bash
  uv run python scripts/http_load_smoke.py --requests 1000 --concurrency 25 --rounds 5
  ```

  Record throughput, p95/p99 latency, failure count, and the deployment revision;
  this is a bounded five-round gate. For a bounded local soak, use for example
  `--duration-seconds 60`; both modes enforce a 100,000-request cap. Neither is a
  substitute for a dedicated destructive test environment.
- Run a restore drill at least once per release or according to the service's
  recovery policy. For SQLite, the repository-level drill is:

  ```bash
  uv run python scripts/sqlite_restore_drill.py "$PGRAPH_HOME/methodos.db"
  ```

  Add `--graph-id` when a known canary graph should be read from the reopened
  backup. Managed Postgres providers still require an equivalent provider-side
  dump/restore drill.
- GitHub Actions also runs the SQLite canary drill weekly through
  `.github/workflows/recovery-drill.yml` and supports an explicit
  `workflow_dispatch` run. This verifies the repository's backup and reopen
  path independently of application feature changes; managed-Postgres drills
  still belong to the deployment environment.
- Keep API keys, provider credentials, signing identity, and telemetry export
  credentials outside source control.
- Define dashboard ownership and an error-budget policy before enabling paging
  in production.
- For worker deployments, define the scheduler/process supervisor, set a
  restart policy, and page on `MethodosEvolutionWorkerDown` before accepting
  evolution traffic.
