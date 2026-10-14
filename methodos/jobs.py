"""Durable evolution-job records for host-owned queue runners.

The service deliberately does not choose a queue or worker system. This module
provides the persistence port that a host-owned runner can use to make queued
jobs restart-safe and idempotent without keeping state in process memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvolutionJob:
    """Durable state for one evolution request."""

    job_id: str
    graph_id: str
    k_rounds: int
    status: str
    detail: str | None
    idempotency_key: str | None
    request_fingerprint: str
    created_at: float
    updated_at: float
    attempt_count: int
    lease_expires_at: float | None
    tenant_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvolutionJobCounts:
    """Low-cardinality queue counts suitable for metrics."""

    queued: int
    running: int
    failed: int


@dataclass(frozen=True, slots=True)
class EvolutionWorkerHealth:
    """Operational heartbeat state for a supervised worker."""

    running: bool
    last_heartbeat: float | None
    consecutive_errors: int
    last_error: str | None


class JobIdempotencyConflictError(RuntimeError):
    """Raised when a key is reused for a different evolution request."""


class JobNotFoundError(FileNotFoundError):
    """Raised when updating a job that does not exist."""


@runtime_checkable
class EvolutionJobStore(Protocol):
    """Persistence port for queue runners."""

    async def create_or_get(
        self,
        *,
        graph_id: str,
        k_rounds: int,
        idempotency_key: str | None,
        tenant_id: str | None = None,
    ) -> EvolutionJob:
        """Create a job or return the existing job for an idempotency key."""
        ...  # pragma: no cover

    async def get(self, job_id: str) -> EvolutionJob | None:
        """Return a job by id, or `None` when it is unknown."""
        ...  # pragma: no cover

    async def claim_next(self, *, lease_seconds: float = 300.0) -> EvolutionJob | None:
        """Atomically claim the oldest queued job with a lease."""
        ...  # pragma: no cover

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        detail: str | None = None,
    ) -> EvolutionJob:
        """Update status atomically and return the new record."""
        ...  # pragma: no cover

    async def counts(self) -> EvolutionJobCounts:
        """Return low-cardinality counts by operational state."""
        ...  # pragma: no cover


@runtime_checkable
class EvolutionJobLeaseStore(EvolutionJobStore, Protocol):
    """Optional store capability for recovering abandoned worker leases."""

    async def recover_expired(self, *, max_attempts: int) -> int:
        """Requeue expired jobs or fail them after bounded attempts."""
        ...  # pragma: no cover


@runtime_checkable
class EvolutionJobMetricsStore(EvolutionJobStore, Protocol):
    """Optional store capability for queue-depth metrics."""

    async def counts(self) -> EvolutionJobCounts:
        """Return low-cardinality counts by operational state."""
        ...


class SQLiteEvolutionJobStore:
    """SQLite-backed job store safe across process restarts on one host."""

    def __init__(self, database_path: Path) -> None:
        """Open or create the job database at `database_path`."""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived transaction connection with safe defaults."""
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _ensure_schema(self) -> None:
        """Create the job table and its idempotency index."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evolution_jobs (
                    job_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    k_rounds INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    idempotency_key TEXT UNIQUE,
                    request_fingerprint TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL,
                    tenant_id TEXT
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(evolution_jobs)")
            }
            if "attempt_count" not in columns:
                connection.execute(
                    "ALTER TABLE evolution_jobs ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "lease_expires_at" not in columns:
                connection.execute("ALTER TABLE evolution_jobs ADD COLUMN lease_expires_at REAL")
            if "tenant_id" not in columns:
                connection.execute("ALTER TABLE evolution_jobs ADD COLUMN tenant_id TEXT")

    @staticmethod
    def _fingerprint(graph_id: str, k_rounds: int, tenant_id: str | None = None) -> str:
        """Create a stable hash for the request fields protected by a key."""
        payload = json.dumps(
            {"graph_id": graph_id, "k_rounds": k_rounds, "tenant_id": tenant_id},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> EvolutionJob:
        """Convert a database row into the immutable public record."""
        return EvolutionJob(**dict(row))

    async def create_or_get(
        self,
        *,
        graph_id: str,
        k_rounds: int,
        idempotency_key: str | None,
        tenant_id: str | None = None,
    ) -> EvolutionJob:
        """Create a durable job, atomically deduplicating a supplied key."""
        if not graph_id or k_rounds <= 0:
            raise ValueError("graph_id must be non-empty and k_rounds must be positive")
        fingerprint = self._fingerprint(graph_id, k_rounds, tenant_id)
        storage_idempotency_key = (
            f"{tenant_id}:{idempotency_key}"
            if tenant_id is not None and idempotency_key
            else idempotency_key
        )

        def create() -> EvolutionJob:
            now = time.time()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    if storage_idempotency_key is not None:
                        row = connection.execute(
                            "SELECT * FROM evolution_jobs WHERE idempotency_key = ?",
                            (storage_idempotency_key,),
                        ).fetchone()
                        if row is not None:
                            if row["request_fingerprint"] != fingerprint:
                                raise JobIdempotencyConflictError(
                                    "idempotency key was used for a different request"
                                )
                            connection.commit()
                            return self._record(row)
                    job_id = str(uuid.uuid4())
                    connection.execute(
                        "INSERT INTO evolution_jobs "
                        "(job_id, graph_id, k_rounds, status, detail, idempotency_key, "
                        "request_fingerprint, created_at, updated_at, attempt_count, "
                        "lease_expires_at, tenant_id) "
                        "VALUES (?, ?, ?, 'queued', NULL, ?, ?, ?, ?, 0, NULL, ?)",
                        (
                            job_id,
                            graph_id,
                            k_rounds,
                            storage_idempotency_key,
                            fingerprint,
                            now,
                            now,
                            tenant_id,
                        ),
                    )
                    row = connection.execute(
                        "SELECT * FROM evolution_jobs WHERE job_id = ?", (job_id,)
                    ).fetchone()
                    connection.commit()
                    assert row is not None
                    return self._record(row)
                except Exception:  # pragma: no cover - defensive transaction rollback
                    connection.rollback()
                    raise

        return await asyncio.to_thread(create)

    async def get(self, job_id: str) -> EvolutionJob | None:
        """Return a durable job record without holding a connection open."""
        if not job_id:
            raise ValueError("job_id must be non-empty")

        def read() -> EvolutionJob | None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM evolution_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                return None if row is None else self._record(row)

        return await asyncio.to_thread(read)

    async def counts(self) -> EvolutionJobCounts:
        """Return queue counts from one consistent SQLite read."""

        def read_counts() -> EvolutionJobCounts:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT status, COUNT(*) AS count FROM evolution_jobs GROUP BY status"
                ).fetchall()
            values = {str(row["status"]): int(row["count"]) for row in rows}
            return EvolutionJobCounts(
                queued=values.get("queued", 0),
                running=values.get("running", 0),
                failed=values.get("failed", 0),
            )

        return await asyncio.to_thread(read_counts)

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        detail: str | None = None,
    ) -> EvolutionJob:
        """Persist a state transition and return the resulting job."""
        if status not in JOB_STATUSES:
            raise ValueError(f"unsupported evolution job status: {status}")

        def write() -> EvolutionJob:
            with self._connect() as connection:
                now = time.time()
                connection.execute("BEGIN IMMEDIATE")
                try:
                    cursor = connection.execute(
                        "UPDATE evolution_jobs SET status = ?, detail = ?, updated_at = ?, "
                        "lease_expires_at = NULL "
                        "WHERE job_id = ?",
                        (status, detail, now, job_id),
                    )
                    if cursor.rowcount != 1:
                        raise JobNotFoundError(f"evolution job {job_id!r} not found")
                    row = connection.execute(
                        "SELECT * FROM evolution_jobs WHERE job_id = ?", (job_id,)
                    ).fetchone()
                    connection.commit()
                    assert row is not None
                    return self._record(row)
                except Exception:
                    connection.rollback()
                    raise

        return await asyncio.to_thread(write)

    async def claim_next(self, *, lease_seconds: float = 300.0) -> EvolutionJob | None:
        """Atomically transition the oldest queued job to `running`."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        def claim() -> EvolutionJob | None:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    row = connection.execute(
                        "SELECT * FROM evolution_jobs WHERE status = 'queued' "
                        "ORDER BY created_at ASC, job_id ASC LIMIT 1"
                    ).fetchone()
                    if row is None:
                        connection.commit()
                        return None
                    connection.execute(
                        "UPDATE evolution_jobs SET status = 'running', updated_at = ?, "
                        "attempt_count = attempt_count + 1, lease_expires_at = ? "
                        "WHERE job_id = ? AND status = 'queued'",
                        (time.time(), time.time() + lease_seconds, row["job_id"]),
                    )
                    claimed = connection.execute(
                        "SELECT * FROM evolution_jobs WHERE job_id = ?", (row["job_id"],)
                    ).fetchone()
                    connection.commit()
                    assert claimed is not None
                    return self._record(claimed)
                except Exception:
                    connection.rollback()
                    raise

        return await asyncio.to_thread(claim)

    async def recover_expired(self, *, max_attempts: int) -> int:
        """Requeue expired jobs or mark them failed after `max_attempts`."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        def recover() -> int:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE evolution_jobs SET status = CASE WHEN attempt_count >= ? "
                    "THEN 'failed' ELSE 'queued' END, detail = CASE WHEN attempt_count >= ? "
                    "THEN 'worker lease expired; retry limit reached' "
                    "ELSE 'worker lease expired; retrying' END, updated_at = ?, "
                    "lease_expires_at = NULL WHERE status = 'running' "
                    "AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?",
                    (max_attempts, max_attempts, time.time(), time.time()),
                )
                return cursor.rowcount

        return await asyncio.to_thread(recover)


def _load_asyncpg() -> Any:  # pragma: no cover - covered by optional integration tests
    """Load the optional Postgres driver only when this store is used."""
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "asyncpg is not installed; install methodos[postgres] to enable Postgres"
        ) from exc
    return asyncpg


class PostgresEvolutionJobStore:  # pragma: no cover - exercised by optional integration tests
    """Postgres-backed job store for horizontally scaled queue runners."""

    def __init__(self, *, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        """Configure a pool without opening a network connection."""
        if not dsn.strip():
            raise ValueError("dsn must not be empty")
        if min_size < 1 or max_size < min_size:
            raise ValueError("pool sizes must satisfy 1 <= min_size <= max_size")
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any | None = None

    async def initialize(self) -> None:
        """Open the pool and create the idempotent job schema."""
        if self._pool is not None:
            return
        asyncpg = _load_asyncpg()
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
        )
        try:
            async with self._pool.acquire() as connection:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS methodos_evolution_jobs (
                        job_id TEXT PRIMARY KEY,
                        graph_id TEXT NOT NULL,
                        k_rounds INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        detail TEXT,
                        idempotency_key TEXT UNIQUE,
                        request_fingerprint TEXT NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        lease_expires_at DOUBLE PRECISION,
                        tenant_id TEXT
                    )
                    """
                )
                await connection.execute(
                    "ALTER TABLE methodos_evolution_jobs ADD COLUMN IF NOT EXISTS "
                    "attempt_count INTEGER NOT NULL DEFAULT 0"
                )
                await connection.execute(
                    "ALTER TABLE methodos_evolution_jobs ADD COLUMN IF NOT EXISTS "
                    "lease_expires_at DOUBLE PRECISION"
                )
                await connection.execute(
                    "ALTER TABLE methodos_evolution_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT"
                )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close the pool if it is open."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> Any:
        """Return an initialized pool."""
        await self.initialize()
        assert self._pool is not None
        return self._pool

    @staticmethod
    def _record(row: Any) -> EvolutionJob:
        """Convert an asyncpg record into the shared immutable model."""
        return EvolutionJob(**dict(row))

    async def create_or_get(
        self,
        *,
        graph_id: str,
        k_rounds: int,
        idempotency_key: str | None,
        tenant_id: str | None = None,
    ) -> EvolutionJob:
        """Create a job or atomically return the keyed existing job."""
        if not graph_id or k_rounds <= 0:
            raise ValueError("graph_id must be non-empty and k_rounds must be positive")
        fingerprint = SQLiteEvolutionJobStore._fingerprint(graph_id, k_rounds, tenant_id)
        storage_idempotency_key = (
            f"{tenant_id}:{idempotency_key}"
            if tenant_id is not None and idempotency_key
            else idempotency_key
        )
        pool = await self._get_pool()
        now = time.time()
        async with pool.acquire() as connection, connection.transaction():
            if storage_idempotency_key is not None:
                row = await connection.fetchrow(
                    "SELECT * FROM methodos_evolution_jobs WHERE idempotency_key = $1 FOR UPDATE",
                    storage_idempotency_key,
                )
                if row is not None:
                    if row["request_fingerprint"] != fingerprint:
                        raise JobIdempotencyConflictError(
                            "idempotency key was used for a different request"
                        )
                    return self._record(row)
            job_id = str(uuid.uuid4())
            row = await connection.fetchrow(
                "INSERT INTO methodos_evolution_jobs "
                "(job_id, graph_id, k_rounds, status, detail, idempotency_key, "
                "request_fingerprint, created_at, updated_at, attempt_count, lease_expires_at, "
                "tenant_id) VALUES ($1, $2, $3, 'queued', NULL, $4, $5, $6, $6, 0, NULL, $7) "
                "RETURNING *",
                job_id,
                graph_id,
                k_rounds,
                storage_idempotency_key,
                fingerprint,
                now,
                tenant_id,
            )
            assert row is not None
            return self._record(row)

    async def get(self, job_id: str) -> EvolutionJob | None:
        """Return a job by id, or `None` when it is unknown."""
        if not job_id:
            raise ValueError("job_id must be non-empty")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM methodos_evolution_jobs WHERE job_id = $1", job_id
            )
        return None if row is None else self._record(row)

    async def counts(self) -> EvolutionJobCounts:
        """Return queue counts from a shared Postgres read."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT status, COUNT(*) AS count FROM methodos_evolution_jobs GROUP BY status"
            )
        values = {str(row["status"]): int(row["count"]) for row in rows}
        return EvolutionJobCounts(
            queued=values.get("queued", 0),
            running=values.get("running", 0),
            failed=values.get("failed", 0),
        )

    async def claim_next(self, *, lease_seconds: float = 300.0) -> EvolutionJob | None:
        """Claim one queued job using row locks across workers."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM methodos_evolution_jobs WHERE status = 'queued' "
                "ORDER BY created_at ASC, job_id ASC FOR UPDATE SKIP LOCKED LIMIT 1"
            )
            if row is None:
                return None
            updated = await connection.fetchrow(
                "UPDATE methodos_evolution_jobs SET status = 'running', updated_at = $1, "
                "attempt_count = attempt_count + 1, lease_expires_at = $2 "
                "WHERE job_id = $3 AND status = 'queued' RETURNING *",
                time.time(),
                time.time() + lease_seconds,
                row["job_id"],
            )
            assert updated is not None
            return self._record(updated)

    async def recover_expired(self, *, max_attempts: int) -> int:
        """Requeue expired jobs or mark them failed after `max_attempts`."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            result = await connection.execute(
                "UPDATE methodos_evolution_jobs SET status = CASE WHEN attempt_count >= $1 "
                "THEN 'failed' ELSE 'queued' END, detail = CASE WHEN attempt_count >= $1 "
                "THEN 'worker lease expired; retry limit reached' "
                "ELSE 'worker lease expired; retrying' END, updated_at = $2, "
                "lease_expires_at = NULL WHERE status = 'running' "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at <= $2",
                max_attempts,
                time.time(),
            )
        return int(result.split()[-1])

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        detail: str | None = None,
    ) -> EvolutionJob:
        """Persist a state transition and return the resulting job."""
        if status not in JOB_STATUSES:
            raise ValueError(f"unsupported evolution job status: {status}")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "UPDATE methodos_evolution_jobs SET status = $1, detail = $2, "
                "updated_at = $3, lease_expires_at = NULL WHERE job_id = $4 RETURNING *",
                status,
                detail,
                time.time(),
                job_id,
            )
        if row is None:
            raise JobNotFoundError(f"evolution job {job_id!r} not found")
        return self._record(row)


@runtime_checkable
class EvolutionJobExecutor(Protocol):
    """Injected environment-specific evolution execution strategy."""

    async def execute(self, job: EvolutionJob) -> None:
        """Execute one claimed job or raise an operational failure."""
        ...


class EvolutionJobWorker:
    """Crash-aware worker that drives jobs from queued to terminal state."""

    def __init__(
        self,
        store: EvolutionJobStore,
        executor: EvolutionJobExecutor,
        *,
        max_attempts: int = 3,
        lease_seconds: float = 300.0,
    ) -> None:
        """Configure a worker with durable storage and an injected executor."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.store = store
        self.executor = executor
        self.max_attempts = max_attempts
        self.lease_seconds = lease_seconds
        self._running = False
        self._last_heartbeat: float | None = None
        self._consecutive_errors = 0
        self._last_error: str | None = None

    def health(self) -> EvolutionWorkerHealth:
        """Return a snapshot suitable for liveness metrics and probes."""
        return EvolutionWorkerHealth(
            running=self._running,
            last_heartbeat=self._last_heartbeat,
            consecutive_errors=self._consecutive_errors,
            last_error=self._last_error,
        )

    async def run_once(self) -> bool:
        """Claim and execute one job; return `False` when the queue is empty."""
        if isinstance(self.store, EvolutionJobLeaseStore):
            await self.store.recover_expired(max_attempts=self.max_attempts)
        job = await self.store.claim_next(lease_seconds=self.lease_seconds)
        if job is None:
            return False
        try:
            await self.executor.execute(job)
        except asyncio.CancelledError:
            await self.store.update(job.job_id, status="cancelled", detail="worker cancelled")
            raise
        except Exception:
            if job.attempt_count < self.max_attempts:
                await self.store.update(
                    job.job_id, status="queued", detail="evolution retry scheduled"
                )
            else:
                await self.store.update(
                    job.job_id, status="failed", detail="evolution execution failed"
                )
            return True
        await self.store.update(job.job_id, status="succeeded", detail="evolution completed")
        return True

    async def run_forever(
        self,
        stop_event: asyncio.Event,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        """Process jobs until `stop_event` is set or the task is cancelled."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._running = True
        try:
            while not stop_event.is_set():
                self._last_heartbeat = time.time()
                try:
                    processed = await self.run_once()
                    self._consecutive_errors = 0
                    self._last_error = None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._consecutive_errors += 1
                    self._last_error = type(exc).__name__
                    logger.exception("evolution worker loop failed")
                    processed = False
                if not processed:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                    except TimeoutError:
                        continue
        finally:
            self._running = False


class StoreBackedEvolutionRunner:
    """REST runner backed by a durable store and a separate worker process."""

    def __init__(self, store: EvolutionJobStore) -> None:
        """Configure the runner with a durable job store."""
        self.store = store

    async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
        """Persist a job and return its stable identifier."""
        del request_id
        job = await self.store.create_or_get(
            graph_id=graph_id,
            k_rounds=k_rounds,
            idempotency_key=None,
        )
        return job.job_id

    async def start_for_tenant(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        k_rounds: int,
        request_id: str,
    ) -> str:
        """Persist a job with durable tenant ownership metadata."""
        del request_id
        job = await self.store.create_or_get(
            graph_id=graph_id,
            k_rounds=k_rounds,
            idempotency_key=None,
            tenant_id=tenant_id,
        )
        return job.job_id

    async def start_idempotent(
        self,
        *,
        graph_id: str,
        k_rounds: int,
        request_id: str,
        idempotency_key: str,
    ) -> str:
        """Persist or deduplicate an idempotent REST request."""
        del request_id
        job = await self.store.create_or_get(
            graph_id=graph_id,
            k_rounds=k_rounds,
            idempotency_key=idempotency_key,
        )
        return job.job_id

    async def start_idempotent_for_tenant(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        k_rounds: int,
        request_id: str,
        idempotency_key: str,
    ) -> str:
        """Persist or deduplicate a tenant-owned evolution request."""
        del request_id
        job = await self.store.create_or_get(
            graph_id=graph_id,
            k_rounds=k_rounds,
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
        )
        return job.job_id

    async def status(self, job_id: str) -> Any:
        """Adapt a durable record to the service's public status DTO."""
        from methodos.service import EvolutionJobStatus

        job = await self.store.get(job_id)
        if job is None:
            return None
        return EvolutionJobStatus(job_id=job.job_id, status=job.status, detail=job.detail)

    async def status_for_tenant(self, *, tenant_id: str, job_id: str) -> Any:
        """Return status only when the durable job belongs to `tenant_id`."""
        from methodos.service import EvolutionJobStatus

        job = await self.store.get(job_id)
        if job is None or job.tenant_id != tenant_id:
            return None
        return EvolutionJobStatus(job_id=job.job_id, status=job.status, detail=job.detail)


__all__ = [
    "JOB_STATUSES",
    "EvolutionJob",
    "EvolutionJobExecutor",
    "EvolutionJobStore",
    "EvolutionJobWorker",
    "JobIdempotencyConflictError",
    "JobNotFoundError",
    "PostgresEvolutionJobStore",
    "SQLiteEvolutionJobStore",
    "StoreBackedEvolutionRunner",
]
