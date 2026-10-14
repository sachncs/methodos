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
  uv run python scripts/http_load_smoke.py --requests 1000 --concurrency 25
  ```

  Record throughput, p95 latency, failure count, and the deployment revision;
  use a dedicated load environment for sustained or destructive tests.
- Run a restore drill at least once per release or according to the service's
  recovery policy. For SQLite, the repository-level drill is:

  ```bash
  uv run python scripts/sqlite_restore_drill.py "$PGRAPH_HOME/methodos.db"
  ```

  Add `--graph-id` when a known canary graph should be read from the reopened
  backup. Managed Postgres providers still require an equivalent provider-side
  dump/restore drill.
- Keep API keys, provider credentials, signing identity, and telemetry export
  credentials outside source control.
- Define dashboard ownership and an error-budget policy before enabling paging
  in production.
- For worker deployments, define the scheduler/process supervisor, set a
  restart policy, and page on `MethodosEvolutionWorkerDown` before accepting
  evolution traffic.
