"""Persistence layer for `methodos`.

This module provides:
- `Repository` Protocol: async interface for graph + trajectory storage.
- `VectorIndex` Protocol: sync interface for semantic search.
- `ScoredMatch`, `Task`, `Trajectory`: data classes used by both this
  module and `methodos.evolution`.
- `SilentVectorIndex`: default VectorIndex; satisfies the Protocol with no
  side effects. Production deployments opt into `SqliteVecIndex`.
- `FilesystemRepository`: JSON-based persistence for development.
- `SQLiteRepository`: production default with WAL + FK + indices.
- `PostgresRepository`: optional pooled backend for horizontally scaled service deployments.
- `SqliteVecIndex`: sqlite-vec integration for semantic search.
- `tail_tokens`: token-aware truncation helper (also used by evolution).
- `build_repository()`: env-driven factory selecting backend.

Engineering notes:
- Public repository configuration is documented; implementation-only state
  follows the one-leading-underscore convention.
- No lazy imports; `sqlite3`, `aiosqlite`, `aiofiles`, `sqlite_vec` are
  all top-level.
- SQLite schema is idempotent and uses FK + WAL for safety.
- `Task` and `Trajectory` live here (not in `evolution`) because they
  are persistence shapes; `evolution` imports them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import aiofiles
import aiosqlite

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - exercised by the production image
    sqlite_vec = None

from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Persistence-shape dataclasses (used by repo + evolution)
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Task:
    """A task the agent must solve.

    Attributes:
        query: the user-facing task description.
        expected: opaque expected outcome for scoring (callable, string,
            dict, etc.). Caller-defined.
    """

    query: str
    expected: Any = None


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A recorded agent trajectory on a task.

    Attributes:
        task: the task that was attempted.
        steps: ((action, observation), ...) in order.
        score: numeric outcome in [0.0, 1.0].
    """

    task: Task
    steps: tuple[tuple[str, str], ...]
    score: float


# ----------------------------------------------------------------------------
# Protocol definitions
# ----------------------------------------------------------------------------


@runtime_checkable
class Repository(Protocol):
    """Async persistence interface for graphs + trajectories.

    Implementations: `FilesystemRepository`, `SQLiteRepository`. The
    Protocol is duck-typed; no ABCs.

    `read_trajectories` is a non-async method that returns an async
    iterator (the standard Python pattern for async iteration).
    """

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        """Load the graph identified by `graph_id`."""
        ...  # pragma: no cover

    async def save_graph(self, graph: ProceduralGraph) -> None:
        """Persist `graph`, replacing any existing entry with the same id."""
        ...  # pragma: no cover

    async def snapshot(self, graph_id: str, tag: str) -> None:
        """Persist a tagged snapshot of the current state of `graph_id`."""
        ...  # pragma: no cover

    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None:
        """Append `trajectory` to the named `split` for `graph_id`."""
        ...  # pragma: no cover

    def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Yield trajectories for `graph_id` from `split`, oldest first."""
        ...  # pragma: no cover


@runtime_checkable
class VectorIndex(Protocol):
    """Sync semantic-search interface.

    Implementations: `SilentVectorIndex` (default), `SqliteVecIndex`.
    """

    def upsert(self, key: str, vector: list[float]) -> None:
        """Store `vector` under `key`, replacing any existing entry."""
        ...  # pragma: no cover

    def query(self, vector: list[float], k: int) -> list[ScoredMatch]:
        """Return up to `k` matches nearest to `vector` by cosine distance."""
        ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    """One result of a `VectorIndex.query` call."""

    key: str
    score: float


class GraphAlreadyExistsError(RuntimeError):
    """Raised when an atomic graph creation finds an existing id."""


class GraphRevisionConflictError(RuntimeError):
    """Raised when a compare-and-swap graph update sees a stale revision."""


@runtime_checkable
class AtomicRepository(Protocol):
    """Optional repository capability for race-free graph creation."""

    async def create_graph(self, graph: ProceduralGraph) -> None:
        """Persist `graph`, failing if its id already exists."""
        ...  # pragma: no cover


@runtime_checkable
class VersionedRepository(Protocol):
    """Optional capability for optimistic-concurrency graph updates."""

    async def save_graph_if_revision(
        self, graph: ProceduralGraph, expected_revision: int
    ) -> ProceduralGraph:
        """Save only when the stored revision equals the expected token."""
        ...  # pragma: no cover


@runtime_checkable
class LifecycleRepository(Protocol):  # pragma: no cover - structural typing only
    """Optional capability for repositories with an async connection lifecycle."""

    async def initialize(self) -> None:
        """Open pools and validate the storage schema."""
        ...  # pragma: no cover

    async def close(self) -> None:
        """Close pools and release storage resources."""
        ...  # pragma: no cover


@runtime_checkable
class RateLimitRepository(Protocol):  # pragma: no cover - capability contract
    """Optional capability for atomically shared request rate limits."""

    async def consume_rate_limit(self, key: str, limit: int) -> tuple[bool, int]:
        """Consume one request and return `(allowed, retry_after_seconds)`."""
        ...  # pragma: no cover


@runtime_checkable
class BackupRepository(Protocol):  # pragma: no cover - capability contract
    """Optional capability for creating verified storage backups."""

    async def backup_to(self, destination: Path) -> None:
        """Create an atomic, integrity-checked backup at `destination`."""
        ...  # pragma: no cover


# ----------------------------------------------------------------------------
# No-op VectorIndex
# ----------------------------------------------------------------------------


class SilentVectorIndex:
    """Default VectorIndex impl. Satisfies the Protocol with zero side effects."""

    def upsert(self, key: str, vector: list[float]) -> None:
        """No-op."""

    def query(self, vector: list[float], k: int) -> list[ScoredMatch]:
        """Always returns empty."""
        return []


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def tail_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` from the BEGINNING, preserving the end (max_tokens chars).

    Approximation: 1 token ≈ 4 characters. Used by both repository
    truncation (e.g., long trajectory logs) and by evolution's refiner
    prompt context (paper's `Tail_{L_max}`).
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")
    if max_tokens == 0:
        return ""
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[-char_budget:]


def validate_storage_component(value: str, *, field: str) -> str:
    """Validate a user-controlled name before using it in a storage path."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError(f"invalid {field}: {value!r}")
    return value


# ----------------------------------------------------------------------------
# FilesystemRepository
# ----------------------------------------------------------------------------


class FilesystemRepository:
    """JSON-based Repository for development and CI.

    Layout under `root`:
        graphs/<graph_id>/graph.json            # current retained graph
        graphs/<graph_id>/trajectories/<split>.jsonl
        graphs/<graph_id>/snapshots/<tag>.json
    Atomic writes via `tempfile` + `Path.replace`.
    """

    def __init__(self, *, root: Path) -> None:
        """Initialize the filesystem-backed repository.

        Args:
            root: Filesystem root under which `graphs/<id>/...` is stored.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def graph_dir(self, graph_id: str) -> Path:
        """Return the directory where artifacts for `graph_id` are stored."""
        return self.root / "graphs" / validate_storage_component(graph_id, field="graph_id")

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        """Load and return the graph identified by `graph_id`.

        Raises:
            FileNotFoundError: If `graph.json` does not exist for `graph_id`.
        """
        path = self.graph_dir(graph_id) / "graph.json"
        if not path.exists():
            raise FileNotFoundError(f"graph {graph_id!r} not found at {path}")
        async with aiofiles.open(path, encoding="utf-8") as f:
            data = json.loads(await f.read())
        return ProceduralGraph.model_validate(data)

    async def save_graph(self, graph: ProceduralGraph) -> None:
        """Persist `graph` atomically (temp-file + rename)."""
        graph_dir = self.graph_dir(graph.id)
        graph_dir.mkdir(parents=True, exist_ok=True)
        target = graph_dir / "graph.json"
        if target.exists():
            existing = await self.load_graph(graph.id)
            graph.revision = existing.revision + 1
        fd, tmp_path_str = tempfile.mkstemp(
            prefix="graph_",
            suffix=".json.tmp",
            dir=str(graph_dir),
        )
        tmp_path = Path(tmp_path_str)
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(graph.model_dump_json(indent=2))
                await f.flush()
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            os.close(fd)
        logger.debug("saved graph %s to %s", graph.id, target)

    async def create_graph(self, graph: ProceduralGraph) -> None:
        """Create a graph atomically without replacing an existing graph."""
        graph_dir = self.graph_dir(graph.id)
        graph_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            graph_dir.mkdir()
        except FileExistsError as exc:
            raise GraphAlreadyExistsError(f"graph {graph.id!r} already exists") from exc
        target = graph_dir / "graph.json"
        fd, tmp_path_str = tempfile.mkstemp(
            prefix="graph_",
            suffix=".json.tmp",
            dir=str(graph_dir),
        )
        tmp_path = Path(tmp_path_str)
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(graph.model_dump_json(indent=2))
                await f.flush()
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            os.close(fd)

    async def save_graph_if_revision(
        self, graph: ProceduralGraph, expected_revision: int
    ) -> ProceduralGraph:
        """Save a graph when its on-disk revision still matches."""
        existing = await self.load_graph(graph.id)
        if existing.revision != expected_revision:
            raise GraphRevisionConflictError(
                f"graph {graph.id!r} revision changed from {expected_revision} "
                f"to {existing.revision}"
            )
        updated = graph.model_copy(update={"revision": existing.revision + 1})
        await self.save_graph(updated)
        return updated

    async def snapshot(self, graph_id: str, tag: str) -> None:
        """Persist a tagged snapshot of `graph_id`'s current state."""
        validate_storage_component(tag, field="tag")
        graph_dir = self.graph_dir(graph_id)
        src = graph_dir / "graph.json"
        if not src.exists():
            raise FileNotFoundError(f"graph {graph_id!r} not found")
        snap_dir = graph_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        dest = snap_dir / f"{tag}.json"
        data = src.read_bytes()
        async with aiofiles.open(dest, "wb") as f:
            await f.write(data)
        logger.debug("snapshotted graph %s to tag %s", graph_id, tag)

    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None:
        """Append `trajectory` to `graph_id`'s `split` log."""
        validate_storage_component(split, field="split")
        graph_dir = self.graph_dir(graph_id)
        traj_dir = graph_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        path = traj_dir / f"{split}.jsonl"
        record = json.dumps(
            {
                "task_query": trajectory.task.query,
                "score": trajectory.score,
                "steps": [list(s) for s in trajectory.steps],
            }
        )
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(record + "\n")

    def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Return an async iterator over trajectories."""
        return self.read_trajectories_impl(graph_id, split)

    async def read_trajectories_impl(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Yield trajectories for `graph_id` from `split`, oldest first."""
        validate_storage_component(split, field="split")
        path = self.graph_dir(graph_id) / "trajectories" / f"{split}.jsonl"
        if not path.exists():
            return
        async with aiofiles.open(path, encoding="utf-8") as f:
            async for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                task = Task(query=record["task_query"])
                steps = tuple((s[0], s[1]) for s in record["steps"])
                yield Trajectory(task=task, steps=steps, score=record["score"])


# ----------------------------------------------------------------------------
# SQLiteRepository
# ----------------------------------------------------------------------------


SCHEMA_VERSION: int = 1


SQLITE_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS graphs (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    body TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id TEXT NOT NULL,
    split TEXT NOT NULL,
    score REAL NOT NULL,
    body TEXT NOT NULL,
    ts REAL NOT NULL,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trajectories_graph_split
    ON trajectories(graph_id, split, ts);

CREATE TABLE IF NOT EXISTS snapshots (
    graph_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    body TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (graph_id, tag),
    FOREIGN KEY (graph_id) REFERENCES graphs(id)
);

CREATE TABLE IF NOT EXISTS rejection_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id TEXT NOT NULL,
    body TEXT NOT NULL,
    val_score REAL NOT NULL,
    ts REAL NOT NULL,
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rejection_graph_ts
    ON rejection_memory(graph_id, ts);
"""


class SQLiteRepository:
    """SQLite-backed Repository with WAL + FK + indices. Production default."""

    def __init__(
        self,
        *,
        db_path: Path,
        vector_index: VectorIndex | None = None,
    ) -> None:
        """Initialize the SQLite-backed repository.

        Args:
            db_path: SQLite file path (shared with `SqliteVecIndex`).
            vector_index: Optional vector index; defaults to a silent
                no-op index. Set to a `SqliteVecIndex` to enable semantic
                recall.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index: VectorIndex = vector_index or SilentVectorIndex()
        self.ensure_schema()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL and FK enabled."""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        """Create tables and indices if they don't exist. Idempotent."""
        with self.connect() as conn:
            for stmt in SQLITE_SCHEMA.strip().split(";"):
                cleaned = stmt.strip()
                if cleaned:
                    conn.execute(cleaned)
            existing = conn.execute("SELECT version FROM schema_meta").fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_meta (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            elif existing["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    "unsupported database schema version "
                    f"{existing['version']}; expected {SCHEMA_VERSION}"
                )

    async def backup_to(self, destination: Path) -> None:
        """Create an atomic online backup and verify SQLite integrity."""
        target = Path(destination)
        if target.resolve() == self.db_path.resolve():
            raise ValueError("backup destination must differ from the source database")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(fd)

        def create_backup() -> None:
            try:
                with self.connect() as source, sqlite3.connect(temporary_path) as backup:
                    source.backup(backup)
                    integrity = backup.execute("PRAGMA integrity_check").fetchone()[0]
                    if integrity != "ok":
                        raise RuntimeError(  # pragma: no cover - defensive SQLite failure
                            f"SQLite backup integrity check failed: {integrity}"
                        )
                os.replace(temporary_path, target)
            finally:
                temporary = Path(temporary_path)
                if temporary.exists():
                    temporary.unlink()

        await asyncio.to_thread(create_backup)
        logger.info("created verified SQLite backup at %s", target)

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        """Load and return the graph identified by `graph_id`.

        Raises:
            FileNotFoundError: If `graph_id` has no row in `graphs`.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            async with db.execute(
                "SELECT body FROM graphs WHERE id = ?",
                (graph_id,),
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    raise FileNotFoundError(f"graph {graph_id!r} not found")
                return ProceduralGraph.model_validate_json(row["body"])

    async def save_graph(self, graph: ProceduralGraph) -> None:
        """Upsert `graph`; uses UPDATE-then-INSERT so snapshot rows survive."""
        ts = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT body FROM graphs WHERE id = ?",
                    (graph.id,),
                ) as cur:
                    existing = await cur.fetchone()
                if existing is None:
                    body = graph.model_dump_json()
                    await db.execute(
                        "INSERT INTO graphs (id, schema_version, body, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (graph.id, graph.schema_version, body, ts),
                    )
                else:
                    current = ProceduralGraph.model_validate_json(existing[0])
                    graph.revision = current.revision + 1
                    body = graph.model_dump_json()
                    await db.execute(
                        "UPDATE graphs SET schema_version = ?, body = ?, updated_at = ? "
                        "WHERE id = ?",
                        (graph.schema_version, body, ts, graph.id),
                    )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        logger.debug("saved graph %s", graph.id)

    async def save_graph_if_revision(
        self, graph: ProceduralGraph, expected_revision: int
    ) -> ProceduralGraph:
        """Compare and swap a graph in one SQLite write transaction."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT body FROM graphs WHERE id = ?", (graph.id,)
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    raise FileNotFoundError(f"graph {graph.id!r} not found")
                current = ProceduralGraph.model_validate_json(row[0])
                if current.revision != expected_revision:
                    raise GraphRevisionConflictError(
                        f"graph {graph.id!r} revision changed from {expected_revision} "
                        f"to {current.revision}"
                    )
                updated = graph.model_copy(update={"revision": current.revision + 1})
                await db.execute(
                    "UPDATE graphs SET schema_version = ?, body = ?, updated_at = ? "
                    "WHERE id = ?",
                    (
                        updated.schema_version,
                        updated.model_dump_json(),
                        time.time(),
                        graph.id,
                    ),
                )
                await db.commit()
                return updated
            except Exception:
                await db.rollback()
                raise

    async def create_graph(self, graph: ProceduralGraph) -> None:
        """Create a graph atomically without replacing an existing graph."""
        body = graph.model_dump_json()
        ts = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "INSERT INTO graphs (id, schema_version, body, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (graph.id, graph.schema_version, body, ts),
                )
                await db.commit()
            except sqlite3.IntegrityError as exc:
                await db.rollback()
                raise GraphAlreadyExistsError(f"graph {graph.id!r} already exists") from exc
            except Exception:
                await db.rollback()
                raise

    async def snapshot(self, graph_id: str, tag: str) -> None:
        """Persist a tagged snapshot of `graph_id`'s current state."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            async with db.execute(
                "SELECT body FROM graphs WHERE id = ?",
                (graph_id,),
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    raise FileNotFoundError(f"graph {graph_id!r} not found")
                body = row["body"]
            ts = time.time()
            await db.execute(
                "INSERT OR REPLACE INTO snapshots (graph_id, tag, body, ts) VALUES (?, ?, ?, ?)",
                (graph_id, tag, body, ts),
            )
            await db.commit()
        logger.debug("snapshotted graph %s to tag %s", graph_id, tag)

    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None:
        """Append `trajectory` to `graph_id`'s `split` log."""
        body = json.dumps(
            {
                "task_query": trajectory.task.query,
                "score": trajectory.score,
                "steps": [list(s) for s in trajectory.steps],
            }
        )
        ts = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute(
                "INSERT INTO trajectories (graph_id, split, score, body, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (graph_id, split, trajectory.score, body, ts),
            )
            await db.commit()

    def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Return an async iterator over trajectories."""
        return self.read_trajectories_impl(graph_id, split)

    async def read_trajectories_impl(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Yield trajectories for `graph_id` from `split`, oldest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA busy_timeout = 5000")
            async with db.execute(
                "SELECT body FROM trajectories WHERE graph_id = ? AND split = ? ORDER BY ts ASC",
                (graph_id, split),
            ) as cur:
                async for row in cur:
                    record = json.loads(row["body"])
                    task = Task(query=record["task_query"])
                    steps = tuple((s[0], s[1]) for s in record["steps"])
                    yield Trajectory(task=task, steps=steps, score=record["score"])


# ----------------------------------------------------------------------------
# PostgresRepository (opt-in via PGRAPH_BACKEND=postgres)
# ----------------------------------------------------------------------------


POSTGRES_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS methodos_schema_meta (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS methodos_graphs (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    revision BIGINT NOT NULL DEFAULT 0,
    body TEXT NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS methodos_trajectories (
    id BIGSERIAL PRIMARY KEY,
    graph_id TEXT NOT NULL REFERENCES methodos_graphs(id) ON DELETE CASCADE,
    split TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    body TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_methodos_trajectories_graph_split
    ON methodos_trajectories(graph_id, split, ts);
CREATE TABLE IF NOT EXISTS methodos_snapshots (
    graph_id TEXT NOT NULL REFERENCES methodos_graphs(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    body TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (graph_id, tag)
);
CREATE TABLE IF NOT EXISTS methodos_rate_limits (
    key TEXT PRIMARY KEY,
    window_started DOUBLE PRECISION NOT NULL,
    request_count INTEGER NOT NULL
);
"""


def _load_asyncpg() -> Any:  # pragma: no cover - exercised by the Postgres integration suite
    """Load the optional Postgres driver only when that backend is selected."""
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "asyncpg is not installed; install methodos[postgres] to enable Postgres"
        ) from exc
    return asyncpg


class PostgresRepository:  # pragma: no cover - exercised by the Postgres integration suite
    """Pooled Postgres Repository for horizontally scaled deployments."""

    def __init__(self, *, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        """Configure a pooled repository without opening network connections."""
        if not dsn.strip():
            raise ValueError("dsn must not be empty")
        if min_size < 1 or max_size < min_size:
            raise ValueError("pool sizes must satisfy 1 <= min_size <= max_size")
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any | None = None

    async def initialize(self) -> None:
        """Open the pool and create/validate the current schema."""
        if self._pool is not None:
            return
        asyncpg = _load_asyncpg()
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn, min_size=self.min_size, max_size=self.max_size
        )
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(POSTGRES_SCHEMA)
                version = await conn.fetchval("SELECT version FROM methodos_schema_meta LIMIT 1")
                if version is None:
                    await conn.execute(
                        "INSERT INTO methodos_schema_meta (version) VALUES ($1)",
                        SCHEMA_VERSION,
                    )
                elif version != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"unsupported database schema version {version}; expected {SCHEMA_VERSION}"
                    )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close the connection pool if it is open."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def consume_rate_limit(self, key: str, limit: int) -> tuple[bool, int]:
        """Atomically consume a request in a shared one-minute window."""
        if limit < 1:
            raise ValueError("limit must be positive")
        pool = await self._get_pool()
        now = time.time()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO methodos_rate_limits (key, window_started, request_count)
                VALUES ($1, $2, 1)
                ON CONFLICT (key) DO UPDATE SET
                    window_started = CASE WHEN $2 - methodos_rate_limits.window_started >= 60
                        THEN $2 ELSE methodos_rate_limits.window_started END,
                    request_count = CASE WHEN $2 - methodos_rate_limits.window_started >= 60
                        THEN 1 ELSE methodos_rate_limits.request_count + 1 END
                RETURNING window_started, request_count""",
                key,
                now,
            )
        window_started = float(row["window_started"])
        request_count = int(row["request_count"])
        retry_after = max(1, int(window_started + 60 - now))
        return request_count <= limit, retry_after

    async def _get_pool(self) -> Any:
        await self.initialize()
        assert self._pool is not None
        return self._pool

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        """Load a graph by public id."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            body = await conn.fetchval("SELECT body FROM methodos_graphs WHERE id = $1", graph_id)
        if body is None:
            raise FileNotFoundError(f"graph {graph_id!r} not found")
        return ProceduralGraph.model_validate_json(body)

    async def save_graph(self, graph: ProceduralGraph) -> None:
        """Upsert a graph and increment its revision atomically."""
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
                current = await conn.fetchrow(
                    "SELECT revision FROM methodos_graphs WHERE id = $1 FOR UPDATE", graph.id
                )
                graph.revision = 0 if current is None else int(current["revision"]) + 1
                await conn.execute(
                    """INSERT INTO methodos_graphs (id, schema_version, revision, body, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET schema_version = EXCLUDED.schema_version,
                    revision = EXCLUDED.revision, body = EXCLUDED.body, updated_at = EXCLUDED.updated_at""",
                    graph.id, graph.schema_version, graph.revision, graph.model_dump_json(), time.time(),
                )

    async def save_graph_if_revision(
        self, graph: ProceduralGraph, expected_revision: int
    ) -> ProceduralGraph:
        """Compare and swap a graph in one Postgres transaction."""
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
                current = await conn.fetchrow(
                    "SELECT revision FROM methodos_graphs WHERE id = $1 FOR UPDATE", graph.id
                )
                if current is None:
                    raise FileNotFoundError(f"graph {graph.id!r} not found")
                revision = int(current["revision"])
                if revision != expected_revision:
                    raise GraphRevisionConflictError(
                        f"graph {graph.id!r} revision changed from {expected_revision} to {revision}"
                    )
                updated = graph.model_copy(update={"revision": revision + 1})
                await conn.execute(
                    "UPDATE methodos_graphs SET schema_version = $1, revision = $2, body = $3, updated_at = $4 WHERE id = $5",
                    updated.schema_version, updated.revision, updated.model_dump_json(), time.time(), graph.id,
                )
                return updated

    async def create_graph(self, graph: ProceduralGraph) -> None:
        """Create a graph atomically without replacing an existing graph."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO methodos_graphs (id, schema_version, revision, body, updated_at) VALUES ($1, $2, $3, $4, $5)",
                    graph.id, graph.schema_version, graph.revision, graph.model_dump_json(), time.time(),
                )
        except Exception as exc:
            if exc.__class__.__name__ == "UniqueViolationError":
                raise GraphAlreadyExistsError(f"graph {graph.id!r} already exists") from exc
            raise

    async def snapshot(self, graph_id: str, tag: str) -> None:
        """Persist a tagged snapshot of the current graph."""
        validate_storage_component(tag, field="tag")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            body = await conn.fetchval("SELECT body FROM methodos_graphs WHERE id = $1", graph_id)
            if body is None:
                raise FileNotFoundError(f"graph {graph_id!r} not found")
            await conn.execute(
                """INSERT INTO methodos_snapshots (graph_id, tag, body, ts) VALUES ($1, $2, $3, $4)
                ON CONFLICT (graph_id, tag) DO UPDATE SET body = EXCLUDED.body, ts = EXCLUDED.ts""",
                graph_id, tag, body, time.time(),
            )

    async def append_trajectory(self, graph_id: str, split: str, trajectory: Trajectory) -> None:
        """Append a trajectory, enforcing the graph foreign key."""
        validate_storage_component(split, field="split")
        pool = await self._get_pool()
        body = json.dumps(
            {"task_query": trajectory.task.query, "score": trajectory.score, "steps": [list(s) for s in trajectory.steps]}
        )
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO methodos_trajectories (graph_id, split, score, body, ts) VALUES ($1, $2, $3, $4, $5)",
                graph_id, split, trajectory.score, body, time.time(),
            )

    def read_trajectories(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Return an async iterator over trajectories."""
        return self.read_trajectories_impl(graph_id, split)

    async def read_trajectories_impl(self, graph_id: str, split: str) -> AsyncIterator[Trajectory]:
        """Yield trajectories oldest first using a server-side cursor."""
        validate_storage_component(split, field="split")
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            async for row in conn.cursor(
                "SELECT body FROM methodos_trajectories WHERE graph_id = $1 AND split = $2 ORDER BY ts ASC",
                graph_id, split,
            ):
                record = json.loads(row["body"])
                yield Trajectory(
                    task=Task(query=record["task_query"]),
                    steps=tuple((s[0], s[1]) for s in record["steps"]),
                    score=record["score"],
                )


# ----------------------------------------------------------------------------
# SqliteVecIndex (opt-in via PGRAPH_VEC=1)
# ----------------------------------------------------------------------------


class SqliteVecIndex:
    """VectorIndex backed by sqlite-vec.

    Loaded only when `PGRAPH_VEC=1`. Uses sqlite-vec's vec0 virtual table
    with cosine distance. The `db_path` is shared with `SQLiteRepository`
    so a single file holds both relational + vector data.
    """

    def __init__(self, *, db_path: Path, dim: int) -> None:
        """Initialize the vector index.

        Args:
            db_path: SQLite file path (shared with `SQLiteRepository`).
            dim: Embedding dimensionality; must be positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if sqlite_vec is None:
            raise RuntimeError(
                "sqlite-vec is not installed; install methodos[vec] to enable vector search"
            )
        self.db_path = Path(db_path)
        self.dim = dim
        self.vec = sqlite_vec
        self.ensure_table()

    def connect(self) -> sqlite3.Connection:
        """Open a sqlite-vec-enabled connection for vector queries."""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        self.vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def ensure_table(self) -> None:
        """Create the `vec_nodes` virtual table if it does not exist."""
        with self.connect() as conn:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes "
                f"USING vec0(key TEXT PRIMARY KEY, embedding float[{self.dim}])"
            )

    def upsert(self, key: str, vector: list[float]) -> None:
        """Store a vector under `key`, replacing any existing entry.

        Args:
            key: Unique identifier for the vector.
            vector: Embedding values; must have length equal to `dim`.

        Raises:
            ValueError: If `len(vector) != dim`.
        """
        if len(vector) != self.dim:
            raise ValueError(f"vector length {len(vector)} != dim {self.dim}")
        packed = self.vec.serialize_float32(vector)
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vec_nodes (key, embedding) VALUES (?, ?)",
                (key, packed),
            )

    def query(self, vector: list[float], k: int) -> list[ScoredMatch]:
        """Return up to `k` vectors nearest to `vector` by cosine distance.

        Args:
            vector: Query embedding; must have length equal to `dim`.
            k: Maximum number of matches to return.

        Returns:
            Matches sorted by similarity (highest first). Empty when `k <= 0`,
            the query vector has wrong length, or the index is empty.
        """
        if len(vector) != self.dim:
            raise ValueError(f"vector length {len(vector)} != dim {self.dim}")
        if k <= 0:
            return []
        packed = self.vec.serialize_float32(vector)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT key, distance FROM vec_nodes "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (packed, k),
            ).fetchall()
        return [ScoredMatch(key=row["key"], score=1.0 - row["distance"]) for row in rows]


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_repository() -> Repository:
    """Construct the configured Repository from environment variables.

    Env vars:
        PGRAPH_BACKEND: "sqlite" (default), "filesystem", or "postgres"
        PGRAPH_HOME: data root (default: ~/.methodos)
        PGRAPH_POSTGRES_DSN: asyncpg DSN when PGRAPH_BACKEND=postgres
        PGRAPH_VEC: "1" to enable sqlite-vec (default: disabled)
        PGRAPH_VEC_DIM: embedding dimension when PGRAPH_VEC=1 (default: 1536)
    """
    backend = os.environ.get("PGRAPH_BACKEND", "sqlite").lower()
    home = Path(os.environ.get("PGRAPH_HOME", Path.home() / ".methodos")).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    if backend == "filesystem":
        return FilesystemRepository(root=home)

    if backend == "sqlite":
        db_path = home / "methodos.db"
        vector_index: VectorIndex = SilentVectorIndex()
        if os.environ.get("PGRAPH_VEC") == "1":
            dim = int(os.environ.get("PGRAPH_VEC_DIM", "1536"))
            vector_index = SqliteVecIndex(db_path=db_path, dim=dim)
        return SQLiteRepository(db_path=db_path, vector_index=vector_index)

    if backend == "postgres":
        dsn = os.environ.get("PGRAPH_POSTGRES_DSN", "").strip()
        if not dsn:
            raise ValueError("PGRAPH_POSTGRES_DSN is required when PGRAPH_BACKEND=postgres")
        return PostgresRepository(dsn=dsn)

    raise ValueError(f"unknown PGRAPH_BACKEND: {backend!r}")


__all__ = [
    "AtomicRepository",
    "FilesystemRepository",
    "GraphAlreadyExistsError",
    "GraphRevisionConflictError",
    "LifecycleRepository",
    "PostgresRepository",
    "Repository",
    "SQLiteRepository",
    "ScoredMatch",
    "SilentVectorIndex",
    "SqliteVecIndex",
    "Task",
    "Trajectory",
    "VectorIndex",
    "VersionedRepository",
    "build_repository",
    "tail_tokens",
    "validate_storage_component",
]
