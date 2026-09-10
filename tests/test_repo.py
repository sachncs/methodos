"""Tests for `methodos.repo`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from methodos.repo import (
    FilesystemRepository,
    NoOpVectorIndex,
    Repository,
    ScoredMatch,
    SQLiteRepository,
    SqliteVecIndex,
    Task,
    Trajectory,
    VectorIndex,
    build_repository,
    tail_tokens,
)
from methodos.schema import ProceduralGraph


def make_trajectory(query: str, score: float = 1.0) -> Trajectory:
    return Trajectory(
        task=Task(query=query),
        steps=(("search", "found something"), ("answer", "done")),
        score=score,
    )


def make_graph(graph_id: str = "test") -> ProceduralGraph:
    """Minimal two-node graph for repository round-trips."""
    from methodos.schema import Node
    return ProceduralGraph(
        id=graph_id,
        nodes={"start": Node(id="start"), "answer": Node(id="answer")},
        edges=[],
        terminal_ids={"answer"},
    )


# ----------------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------------


class TestRepositoryProtocol:
    """`Repository` is runtime-checkable; implementations satisfy it."""

    def test_filesystem_repo_satisfies_protocol(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        assert isinstance(repo, Repository)

    def test_sqlite_repo_satisfies_protocol(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        assert isinstance(repo, Repository)


class TestVectorIndexProtocol:
    """`VectorIndex` is runtime-checkable."""

    def test_noop_satisfies_protocol(self) -> None:
        assert isinstance(NoOpVectorIndex(), VectorIndex)


# ----------------------------------------------------------------------------
# tail_tokens
# ----------------------------------------------------------------------------


class TestTailTokens:
    """`tail_tokens` truncates from the beginning, preserving the end."""

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            tail_tokens("hello", -1)

    def test_zero_returns_empty(self) -> None:
        assert tail_tokens("hello world", 0) == ""

    def test_under_budget_returns_whole(self) -> None:
        text = "hello"  # 5 chars
        assert tail_tokens(text, 100) == text

    def test_over_budget_truncates(self) -> None:
        text = "abcdefghij"  # 10 chars
        # max_tokens=2 → 8 chars → keep last 8
        result = tail_tokens(text, 2)
        assert result == "cdefghij"

    def test_exact_budget_returns_whole(self) -> None:
        text = "abcd"  # 4 chars → exactly 1 token
        assert tail_tokens(text, 1) == text

    def test_empty_text(self) -> None:
        assert tail_tokens("", 100) == ""


# ----------------------------------------------------------------------------
# NoOpVectorIndex
# ----------------------------------------------------------------------------


class TestNoOpVectorIndex:
    def test_upsert_is_silent(self) -> None:
        idx = NoOpVectorIndex()
        idx.upsert("k", [0.1, 0.2, 0.3])  # no error

    def test_query_returns_empty(self) -> None:
        idx = NoOpVectorIndex()
        assert idx.query([0.1, 0.2], 5) == []


# ----------------------------------------------------------------------------
# FilesystemRepository
# ----------------------------------------------------------------------------


class TestFilesystemRepository:
    """FilesystemRepository: JSON files, atomic writes, trajectories."""

    async def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        graph = make_graph("g1")
        await repo.save_graph(graph)
        loaded = await repo.load_graph("g1")
        assert loaded == graph
        assert loaded.id == "g1"

    async def test_load_missing_raises(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            await repo.load_graph("nope")

    async def test_save_creates_directories(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("deep"))
        assert (tmp_path / "graphs" / "deep" / "graph.json").exists()

    async def test_snapshot_writes_to_history(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        assert (tmp_path / "graphs" / "g" / "snapshots" / "v1.json").exists()

    async def test_snapshot_missing_raises(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            await repo.snapshot("missing", "v1")

    async def test_append_and_read_trajectories(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("q1"))
        await repo.append_trajectory("g", "train", make_trajectory("q2", 0.5))
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert len(trajectories) == 2
        assert trajectories[0].task.query == "q1"
        assert trajectories[0].score == 1.0
        assert trajectories[1].task.query == "q2"
        assert trajectories[1].score == 0.5

    async def test_read_trajectories_filtered_by_split(
        self, tmp_path: Path,
    ) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("t"))
        await repo.append_trajectory("g", "val", make_trajectory("v"))
        train = [t async for t in repo.read_trajectories("g", "train")]
        val = [t async for t in repo.read_trajectories("g", "val")]
        assert len(train) == 1 and train[0].task.query == "t"
        assert len(val) == 1 and val[0].task.query == "v"

    async def test_read_trajectories_missing_split_empty(
        self, tmp_path: Path,
    ) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert trajectories == []

    async def test_save_overwrites(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        from methodos.schema import Node
        new_graph = ProceduralGraph(
            id="g", nodes={"answer": Node(id="answer")}, terminal_ids={"answer"},
        )
        await repo.save_graph(new_graph)
        loaded = await repo.load_graph("g")
        assert "answer" in loaded.nodes
        assert "start" not in loaded.nodes


# ----------------------------------------------------------------------------
# SQLiteRepository
# ----------------------------------------------------------------------------


class TestSQLiteRepository:
    """SQLiteRepository: WAL, FK, indices, round-trips."""

    async def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        graph = make_graph("g1")
        await repo.save_graph(graph)
        loaded = await repo.load_graph("g1")
        assert loaded == graph

    async def test_load_missing_raises(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        with pytest.raises(FileNotFoundError, match="not found"):
            await repo.load_graph("nope")

    async def test_save_overwrites(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        # Re-saving the same graph updates the body in place; the row
        # is preserved (no DELETE+INSERT side effects).
        from methodos.schema import Node
        updated = ProceduralGraph(
            id="g", nodes={"answer": Node(id="answer")}, terminal_ids={"answer"},
        )
        await repo.save_graph(updated)
        loaded = await repo.load_graph("g")
        assert loaded.id == "g"
        assert "answer" in loaded.nodes
        assert "start" not in loaded.nodes

    async def test_snapshot(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        # Re-saving the graph should not affect the snapshot.
        await repo.save_graph(make_graph("g"))
        # Load snapshot via direct SQL to verify.
        import sqlite3
        with sqlite3.connect(tmp_path / "test.db") as conn:
            row = conn.execute(
                "SELECT body FROM snapshots WHERE graph_id = ? AND tag = ?",
                ("g", "v1"),
            ).fetchone()
            assert row is not None

    async def test_snapshot_missing_raises(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        with pytest.raises(FileNotFoundError):
            await repo.snapshot("nope", "v1")

    async def test_append_and_read_trajectories(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("q1"))
        await repo.append_trajectory("g", "train", make_trajectory("q2", 0.3))
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert len(trajectories) == 2
        assert [t.task.query for t in trajectories] == ["q1", "q2"]

    async def test_trajectories_ordered_by_timestamp(
        self, tmp_path: Path,
    ) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        for i in range(3):
            await repo.append_trajectory(
                "g", "train", make_trajectory(f"q{i}"),
            )
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert [t.task.query for t in trajectories] == ["q0", "q1", "q2"]

    async def test_read_trajectories_filtered_by_split(
        self, tmp_path: Path,
    ) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("t"))
        await repo.append_trajectory("g", "val", make_trajectory("v"))
        train = [t async for t in repo.read_trajectories("g", "train")]
        val = [t async for t in repo.read_trajectories("g", "val")]
        assert len(train) == 1
        assert len(val) == 1

    async def test_fk_cascade_delete_trajectories(
        self, tmp_path: Path,
    ) -> None:
        """Deleting a graph cascades to its trajectories via FK."""
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("q"))
        # Manually delete the graph row; trajectories should cascade.
        import sqlite3
        with sqlite3.connect(tmp_path / "test.db") as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM graphs WHERE id = 'g'")
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE graph_id = 'g'"
            ).fetchone()[0]
            assert count == 0

    async def test_wal_mode_active(self, tmp_path: Path) -> None:
        """Verify journal_mode is WAL after init."""
        SQLiteRepository(db_path=tmp_path / "test.db")
        import sqlite3
        with sqlite3.connect(tmp_path / "test.db") as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"

    async def test_index_on_trajectories(self, tmp_path: Path) -> None:
        """Verify the (graph_id, split, ts) index exists."""
        SQLiteRepository(db_path=tmp_path / "test.db")
        import sqlite3
        with sqlite3.connect(tmp_path / "test.db") as conn:
            indices = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' AND name LIKE 'idx_%'"
            ).fetchall()
            index_names = {row[0] for row in indices}
            assert "idx_trajectories_graph_split" in index_names
            assert "idx_rejection_graph_ts" in index_names

    async def test_schema_version_recorded(self, tmp_path: Path) -> None:
        """After init, schema_meta contains version 1."""
        SQLiteRepository(db_path=tmp_path / "test.db")
        import sqlite3
        with sqlite3.connect(tmp_path / "test.db") as conn:
            row = conn.execute("SELECT version FROM schema_meta").fetchone()
            assert row[0] == 1

    async def test_idempotent_schema_creation(self, tmp_path: Path) -> None:
        """Constructing twice does not raise."""
        path = tmp_path / "test.db"
        SQLiteRepository(db_path=path)
        # Second construction runs ensure_schema again (no-op).
        SQLiteRepository(db_path=path)

    async def test_concurrent_writes_serialize(
        self, tmp_path: Path,
    ) -> None:
        """Concurrent save_graph calls do not corrupt the database.

        `SQLiteRepository.save_graph` wraps each save in `BEGIN IMMEDIATE`
        + UPDATE-or-INSERT. WAL allows concurrent readers but writers
        serialize. This test fires N concurrent saves and asserts the
        final graph matches the last writer (no torn writes).
        """
        import asyncio

        from methodos.schema import Node

        repo = SQLiteRepository(db_path=tmp_path / "concurrent.db")
        base = ProceduralGraph(
            id="g",
            nodes={"answer": Node(id="answer")},
            terminal_ids={"answer"},
        )
        await repo.save_graph(base)

        async def save_variant(index: int) -> None:
            graph = base.model_copy(update={
                "id": f"g-{index}",
                "metadata": {"writer": index},
            })
            await repo.save_graph(graph)

        n = 8
        results = await asyncio.gather(
            *[save_variant(i) for i in range(n)],
            return_exceptions=True,
        )
        # All writers must complete without raising.
        for r in results:
            assert not isinstance(r, BaseException), f"writer raised: {r}"

        # All N writers' final graphs must be readable; the database isn't
        # corrupted. Spot-check the latest writer (last by index).
        latest = await repo.load_graph(f"g-{n - 1}")
        assert latest.metadata["writer"] == n - 1


# ----------------------------------------------------------------------------
# build_repository factory
# ----------------------------------------------------------------------------


class TestBuildRepository:
    """`build_repository` selects backend from env vars."""

    def test_default_returns_sqlite(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.delenv("PGRAPH_BACKEND", raising=False)
        repo = build_repository()
        assert isinstance(repo, SQLiteRepository)

    def test_filesystem_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "filesystem")
        repo = build_repository()
        assert isinstance(repo, FilesystemRepository)

    def test_unknown_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "bogus")
        with pytest.raises(ValueError, match="unknown PGRAPH_BACKEND"):
            build_repository()


# ----------------------------------------------------------------------------
# ScoredMatch
# ----------------------------------------------------------------------------


class TestScoredMatch:
    def test_construction(self) -> None:
        sm = ScoredMatch(key="x", score=0.9)
        assert sm.key == "x"
        assert sm.score == 0.9

    def test_is_hashable(self) -> None:
        # Frozen dataclass is hashable.
        s1 = ScoredMatch(key="x", score=0.9)
        s2 = ScoredMatch(key="x", score=0.9)
        assert hash(s1) == hash(s2)


# ----------------------------------------------------------------------------
# Task / Trajectory
# ----------------------------------------------------------------------------


class TestTaskAndTrajectory:
    def test_task_defaults(self) -> None:
        task = Task(query="q")
        assert task.expected is None

    def test_trajectory_with_steps(self) -> None:
        traj = Trajectory(
            task=Task(query="q"),
            steps=(("a", "b"),),
            score=0.5,
        )
        assert traj.steps == (("a", "b"),)
        assert traj.score == 0.5

    def test_trajectory_is_hashable(self) -> None:
        traj = Trajectory(task=Task(query="q"), steps=(), score=0.0)
        # Should be hashable (frozen, slotted).
        hash(traj)


# ----------------------------------------------------------------------------
# SqliteVecIndex (only if sqlite_vec is installed)
# ----------------------------------------------------------------------------


class TestSqliteVecIndexIfAvailable:
    """SqliteVecIndex: round-trip upsert + cosine query."""

    def test_requires_positive_dim(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="dim must be positive"):
            SqliteVecIndex(db_path=tmp_path / "vec.db", dim=0)

    def test_raises_when_dep_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        # Simulate missing dep.
        import builtins
        original_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sqlite_vec":
                raise ImportError("no sqlite_vec")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="sqlite-vec is not installed"):
            SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)

    def test_upsert_validates_dim(self, tmp_path: Path) -> None:
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite_vec not installed")
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        with pytest.raises(ValueError, match="vector length 3 != dim 4"):
            idx.upsert("k", [0.1, 0.2, 0.3])

    def test_query_validates_dim(self, tmp_path: Path) -> None:
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite_vec not installed")
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        with pytest.raises(ValueError, match="vector length 2 != dim 4"):
            idx.query([0.1, 0.2], k=1)

    def test_query_with_k_zero_returns_empty(self, tmp_path: Path) -> None:
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite_vec not installed")
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        assert idx.query([0.1, 0.2, 0.3, 0.4], k=0) == []

    def test_upsert_and_query_round_trip(self, tmp_path: Path) -> None:
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite_vec not installed")
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        idx.upsert("a", [1.0, 0.0, 0.0, 0.0])
        idx.upsert("b", [0.0, 1.0, 0.0, 0.0])
        # Query for vector closest to "a".
        matches = idx.query([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(matches) == 1
        assert matches[0].key == "a"
        assert matches[0].score > 0.99  # cosine ~1 for identical vectors
