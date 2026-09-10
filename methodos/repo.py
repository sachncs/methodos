"""Persistence layer for `methodos`.

This module provides:
- `Repository` Protocol: async interface for graph + trajectory storage.
- `VectorIndex` Protocol: sync interface for semantic search.
- `ScoredMatch`, `Task`, `Trajectory`: data classes used by both this
  module and `methodos.evolution`.
- `NoOpVectorIndex`: default VectorIndex; satisfies the Protocol with no
  side effects. Production deployments opt into `SqliteVecIndex`.
- `FilesystemRepository`: JSON-based persistence for development.
- `SQLiteRepository`: production default with WAL + FK + indices.
- `SqliteVecIndex`: sqlite-vec integration for semantic search.
- `tail_tokens`: token-aware truncation helper (also used by evolution).
- `build_repository()`: env-driven factory selecting backend.

Engineering notes:
- No `_foo()` markers. Every helper has a descriptive public name.
- No lazy imports; `sqlite3`, `aiosqlite`, `aiofiles`, `sqlite_vec`
  (when available) are all top-level.
- SQLite schema is idempotent and uses FK + WAL for safety.
- `Task` and `Trajectory` live here (not in `evolution`) because they
  are persistence shapes; `evolution` imports them.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import aiofiles
import aiosqlite

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

    async def load_graph(self, graph_id: str) -> ProceduralGraph: ...

    async def save_graph(self, graph: ProceduralGraph) -> None: ...

    async def snapshot(self, graph_id: str, tag: str) -> None: ...

    async def append_trajectory(
        self, graph_id: str, split: str, trajectory: Trajectory
    ) -> None: ...

    def read_trajectories(
        self, graph_id: str, split: str
    ) -> AsyncIterator[Trajectory]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Sync semantic-search interface.

    Implementations: `NoOpVectorIndex` (default), `SqliteVecIndex`.
    """

    def upsert(self, key: str, vector: list[float]) -> None: ...

    def query(self, vector: list[float], k: int) -> list[ScoredMatch]: ...


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    """One result of a `VectorIndex.query` call."""
    key: str
    score: float


# ----------------------------------------------------------------------------
# No-op VectorIndex
# ----------------------------------------------------------------------------


class NoOpVectorIndex:
    """Default VectorIndex impl. Satisfies the Protocol with zero side effects.

    Used when `PGRAPH_VEC != "1"`. Every `query` returns an empty list;
    every `upsert` is a no-op. Consumers must already handle empty
    results (which they do, because fuzzy match is opt-in).
    """

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

    Args:
        text: input string.
        max_tokens: maximum number of tokens to keep; must be ≥ 0.
            `0` returns empty string; values beyond `len(text)` are no-ops.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")
    if max_tokens == 0:
        return ""
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[-char_budget:]


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
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _graph_dir(self, graph_id: str) -> Path:
        return self._root / "graphs" / graph_id

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        path = self._graph_dir(graph_id) / "graph.json"
        if not path.exists():
            raise FileNotFoundError(f"graph {graph_id!r} not found at {path}")
        async with aiofiles.open(path, encoding="utf-8") as f:
            data = json.loads(await f.read())
        return ProceduralGraph.model_validate(data)

    async def save_graph(self, graph: ProceduralGraph) -> None:
        graph_dir = self._graph_dir(graph.id)
        graph_dir.mkdir(parents=True, exist_ok=True)
        target = graph_dir / "graph.json"
        # Atomic write: write to a sibling temp file, then replace.
        fd, tmp_path_str = tempfile.mkstemp(
            prefix="graph_", suffix=".json.tmp", dir=str(graph_dir),
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

    async def snapshot(self, graph_id: str, tag: str) -> None:
        graph_dir = self._graph_dir(graph_id)
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

    async def append_trajectory(
        self, graph_id: str, split: str, trajectory: Trajectory
    ) -> None:
        graph_dir = self._graph_dir(graph_id)
        traj_dir = graph_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        path = traj_dir / f"{split}.jsonl"
        record = json.dumps({
            "task_query": trajectory.task.query,
            "score": trajectory.score,
            "steps": [list(s) for s in trajectory.steps],
        })
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(record + "\n")

    def read_trajectories(
        self, graph_id: str, split: str
    ) -> AsyncIterator[Trajectory]:
        """Return an async iterator over trajectories.

        Synchronous method that returns an `AsyncIterator` (the standard
        Python pattern for async iteration; see PEP 492 / 525).
        """
        return self._read_trajectories_impl(graph_id, split)

    async def _read_trajectories_impl(
        self, graph_id: str, split: str
    ) -> AsyncIterator[Trajectory]:
        path = self._graph_dir(graph_id) / "trajectories" / f"{split}.jsonl"
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
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_index: VectorIndex = vector_index or NoOpVectorIndex()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables and indices if they don't exist. Idempotent."""
        with self._connect() as conn:
            for stmt in SQLITE_SCHEMA.strip().split(";"):
                cleaned = stmt.strip()
                if cleaned:
                    conn.execute(cleaned)
            # Record schema version if missing.
            existing = conn.execute(
                "SELECT version FROM schema_meta"
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_meta (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )

    async def load_graph(self, graph_id: str) -> ProceduralGraph:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                "SELECT body FROM graphs WHERE id = ?", (graph_id,),
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    raise FileNotFoundError(f"graph {graph_id!r} not found")
                return ProceduralGraph.model_validate_json(row["body"])

    async def save_graph(self, graph: ProceduralGraph) -> None:
        """Upsert the graph row. Uses UPDATE-then-INSERT (NOT `INSERT OR
        REPLACE`) so that snapshot rows referencing this graph_id are
        not cascade-deleted by SQLite's DELETE+INSERT implementation.
        """
        body = graph.model_dump_json()
        ts = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT 1 FROM graphs WHERE id = ?", (graph.id,),
                ) as cur:
                    existing = await cur.fetchone()
                if existing is None:
                    await db.execute(
                        "INSERT INTO graphs (id, schema_version, body, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (graph.id, graph.schema_version, body, ts),
                    )
                else:
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

    async def snapshot(self, graph_id: str, tag: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                "SELECT body FROM graphs WHERE id = ?", (graph_id,),
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    raise FileNotFoundError(f"graph {graph_id!r} not found")
                body = row["body"]
            ts = time.time()
            await db.execute(
                "INSERT OR REPLACE INTO snapshots (graph_id, tag, body, ts) "
                "VALUES (?, ?, ?, ?)",
                (graph_id, tag, body, ts),
            )
            await db.commit()
        logger.debug("snapshotted graph %s to tag %s", graph_id, tag)

    async def append_trajectory(
        self, graph_id: str, split: str, trajectory: Trajectory
    ) -> None:
        body = json.dumps({
            "task_query": trajectory.task.query,
            "score": trajectory.score,
            "steps": [list(s) for s in trajectory.steps],
        })
        ts = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO trajectories (graph_id, split, score, body, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (graph_id, split, trajectory.score, body, ts),
            )
            await db.commit()

    def read_trajectories(
        self, graph_id: str, split: str
    ) -> AsyncIterator[Trajectory]:
        """Return an async iterator over trajectories for (graph_id, split).

        Synchronous method that returns an `AsyncIterator` (the standard
        Python pattern for async iteration).
        """
        return self._read_trajectories_impl(graph_id, split)

    async def _read_trajectories_impl(
        self, graph_id: str, split: str
    ) -> AsyncIterator[Trajectory]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                "SELECT body FROM trajectories "
                "WHERE graph_id = ? AND split = ? ORDER BY ts ASC",
                (graph_id, split),
            ) as cur:
                async for row in cur:
                    record = json.loads(row["body"])
                    task = Task(query=record["task_query"])
                    steps = tuple((s[0], s[1]) for s in record["steps"])
                    yield Trajectory(task=task, steps=steps, score=record["score"])


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
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        try:
            import sqlite_vec
        except ImportError as exc:
            raise RuntimeError(
                "sqlite-vec is not installed. `pip install methodos[vec]`."
            ) from exc
        self._db_path = Path(db_path)
        self._dim = dim
        self._sqlite_vec = sqlite_vec
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        # Enable extension loading (required on macOS Python builds) and
        # load the sqlite-vec extension so `vec0` is available.
        conn.enable_load_extension(True)
        self._sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes "
                f"USING vec0(key TEXT PRIMARY KEY, embedding float[{self._dim}])"
            )

    def upsert(self, key: str, vector: list[float]) -> None:
        if len(vector) != self._dim:
            raise ValueError(
                f"vector length {len(vector)} != dim {self._dim}"
            )
        packed = self._sqlite_vec.serialize_float32(vector)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vec_nodes (key, embedding) VALUES (?, ?)",
                (key, packed),
            )

    def query(self, vector: list[float], k: int) -> list[ScoredMatch]:
        if len(vector) != self._dim:
            raise ValueError(
                f"vector length {len(vector)} != dim {self._dim}"
            )
        if k <= 0:
            return []
        packed = self._sqlite_vec.serialize_float32(vector)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, distance FROM vec_nodes "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (packed, k),
            ).fetchall()
        # Convert distance → similarity: smaller distance = more similar.
        return [ScoredMatch(key=row["key"], score=1.0 - row["distance"]) for row in rows]


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_repository() -> Repository:
    """Construct the configured Repository from environment variables.

    Env vars:
        PGRAPH_BACKEND: "sqlite" (default) or "filesystem"
        PGRAPH_HOME: data root (default: ~/.methodos)
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
        vector_index: VectorIndex = NoOpVectorIndex()
        if os.environ.get("PGRAPH_VEC") == "1":
            dim = int(os.environ.get("PGRAPH_VEC_DIM", "1536"))
            vector_index = SqliteVecIndex(db_path=db_path, dim=dim)
        return SQLiteRepository(db_path=db_path, vector_index=vector_index)

    raise ValueError(f"unknown PGRAPH_BACKEND: {backend!r}")


__all__ = [
    "FilesystemRepository",
    "NoOpVectorIndex",
    "Repository",
    "SQLiteRepository",
    "ScoredMatch",
    "SqliteVecIndex",
    "Task",
    "Trajectory",
    "VectorIndex",
    "build_repository",
    "tail_tokens",
]
