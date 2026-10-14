"""Durable evolution-job records for host-owned queue runners.

The service deliberately does not choose a queue or worker system. This module
provides the persistence port that a host-owned runner can use to make queued
jobs restart-safe and idempotent without keeping state in process memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})


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
    ) -> EvolutionJob:
        """Create a job or return the existing job for an idempotency key."""
        ...

    async def get(self, job_id: str) -> EvolutionJob | None:
        """Return a job by id, or `None` when it is unknown."""
        ...

    async def update(
        self,
        job_id: str,
        *,
        status: str,
        detail: str | None = None,
    ) -> EvolutionJob:
        """Update status atomically and return the new record."""
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
                    updated_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _fingerprint(graph_id: str, k_rounds: int) -> str:
        """Create a stable hash for the request fields protected by a key."""
        payload = json.dumps(
            {"graph_id": graph_id, "k_rounds": k_rounds},
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
    ) -> EvolutionJob:
        """Create a durable job, atomically deduplicating a supplied key."""
        if not graph_id or k_rounds <= 0:
            raise ValueError("graph_id must be non-empty and k_rounds must be positive")
        fingerprint = self._fingerprint(graph_id, k_rounds)

        def create() -> EvolutionJob:
            now = time.time()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    if idempotency_key is not None:
                        row = connection.execute(
                            "SELECT * FROM evolution_jobs WHERE idempotency_key = ?",
                            (idempotency_key,),
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
                        "request_fingerprint, created_at, updated_at) "
                        "VALUES (?, ?, ?, 'queued', NULL, ?, ?, ?, ?)",
                        (job_id, graph_id, k_rounds, idempotency_key, fingerprint, now, now),
                    )
                    row = connection.execute(
                        "SELECT * FROM evolution_jobs WHERE job_id = ?", (job_id,)
                    ).fetchone()
                    connection.commit()
                    assert row is not None
                    return self._record(row)
                except Exception:
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
                        "UPDATE evolution_jobs SET status = ?, detail = ?, updated_at = ? "
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


__all__ = [
    "JOB_STATUSES",
    "EvolutionJob",
    "EvolutionJobStore",
    "JobIdempotencyConflictError",
    "JobNotFoundError",
    "SQLiteEvolutionJobStore",
]
