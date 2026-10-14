"""Tests for `methodos.repo`."""

from __future__ import annotations

import asyncio
import builtins
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import methodos.repo as repo_module
from methodos.migrations import Migration, MigrationError, MigrationRunner
from methodos.repo import (
    BackupRepository,
    CorruptRecordError,
    EvolutionEvent,
    FilesystemRepository,
    GraphAlreadyExistsError,
    GraphRevisionConflictError,
    PostgresRepository,
    Repository,
    ScoredMatch,
    SilentVectorIndex,
    SQLiteRepository,
    SqliteVecIndex,
    Task,
    Trajectory,
    VectorIndex,
    build_repository,
    tail_tokens,
    validate_storage_component,
)
from methodos.schema import Node, ProceduralGraph


def make_trajectory(query: str, score: float = 1.0) -> Trajectory:
    return Trajectory(
        task=Task(query=query),
        steps=(("search", "found something"), ("answer", "done")),
        score=score,
    )


def make_graph(graph_id: str = "test") -> ProceduralGraph:
    """Minimal two-node graph for repository round-trips."""
    return ProceduralGraph(
        id=graph_id,
        nodes={"start": Node(id="start"), "answer": Node(id="answer")},
        edges=[],
        terminal_ids={"answer"},
    )


def test_migration_runner_applies_ordered_sqlite_upgrades() -> None:
    runner = MigrationRunner(
        (
            Migration(1, ("CREATE TABLE data (value TEXT)",)),
            Migration(2, ("ALTER TABLE data ADD COLUMN revision INTEGER NOT NULL DEFAULT 0",)),
        )
    )
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
        runner.apply_sqlite(connection, 0)
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 2
        columns = {row[1] for row in connection.execute("PRAGMA table_info(data)")}
        assert columns == {"value", "revision"}
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO schema_meta (version) VALUES (1)")
        connection.execute("CREATE TABLE data (value TEXT)")
        runner.apply_sqlite(connection, 1)
        assert "revision" in {row[1] for row in connection.execute("PRAGMA table_info(data)")}


async def test_migration_runner_applies_asyncpg_statements() -> None:
    runner = MigrationRunner((Migration(1, ("CREATE TABLE data (value TEXT)",)),))

    class FakeConnection:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple[object, ...]]] = []

        async def execute(self, statement: str, *args: object) -> None:
            self.statements.append((statement, args))

    connection = FakeConnection()
    await runner.apply_asyncpg(connection, 0, metadata_table="methodos_schema_meta")
    assert connection.statements == [
        ("CREATE TABLE data (value TEXT)", ()),
        ("INSERT INTO methodos_schema_meta (version) VALUES ($1)", (1,)),
    ]
    upgrade_runner = MigrationRunner((Migration(1, ()), Migration(2, ())))
    await upgrade_runner.apply_asyncpg(connection, 1, metadata_table="methodos_schema_meta")
    assert connection.statements[-1] == (
        "UPDATE methodos_schema_meta SET version = $1",
        (2,),
    )


def test_migration_runner_rejects_gaps_and_future_versions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MigrationRunner(())
    with pytest.raises(ValueError, match="contiguous"):
        MigrationRunner((Migration(2, ()),))
    runner = MigrationRunner((Migration(1, ()),))
    with pytest.raises(MigrationError, match="negative"):
        runner.pending(-1)
    with pytest.raises(MigrationError, match="unsupported"):
        runner.pending(2)


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

    def test_silent_satisfies_protocol(self) -> None:
        assert isinstance(SilentVectorIndex(), VectorIndex)


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


class TestStorageComponent:
    def test_accepts_safe_component(self) -> None:
        assert validate_storage_component("train.v1", field="split") == "train.v1"

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="invalid split"):
            validate_storage_component("../escape", field="split")

    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="invalid tag"):
            validate_storage_component("/tmp/escape", field="tag")


# ----------------------------------------------------------------------------
# SilentVectorIndex
# ----------------------------------------------------------------------------


class TestSilentVectorIndex:
    def test_upsert_is_silent(self) -> None:
        idx = SilentVectorIndex()
        idx.upsert("k", [0.1, 0.2, 0.3])  # no error

    def test_query_returns_empty(self) -> None:
        idx = SilentVectorIndex()
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

    async def test_corrupt_graph_is_quarantined(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        graph_path = tmp_path / "graphs" / "bad" / "graph.json"
        graph_path.parent.mkdir(parents=True)
        graph_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(CorruptRecordError, match="quarantined"):
            await repo.load_graph("bad")

        assert not graph_path.exists()
        assert len(list(graph_path.parent.joinpath("corrupt").iterdir())) == 1

    async def test_corrupt_graph_reports_quarantine_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = FilesystemRepository(root=tmp_path)
        graph_path = tmp_path / "graphs" / "bad" / "graph.json"
        graph_path.parent.mkdir(parents=True)
        graph_path.write_text("{not-json", encoding="utf-8")

        def fail_quarantine(path: Path) -> Path:
            del path
            raise OSError("read-only storage")

        monkeypatch.setattr(repo_module, "_quarantine_file", fail_quarantine)
        with pytest.raises(CorruptRecordError, match="quarantine failed"):
            await repo.load_graph("bad")

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

    async def test_snapshot_rejects_unsafe_tag(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        with pytest.raises(ValueError, match="invalid tag"):
            await repo.snapshot("g", "../escape")

    async def test_restore_snapshot_advances_revision(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        changed = make_graph("g")
        changed.nodes["start"].description = "changed"
        await repo.save_graph(changed)

        restored = await repo.restore_snapshot("g", "v1", expected_revision=1)

        assert restored.revision == 2
        assert restored.nodes["start"].description == ""

    async def test_restore_snapshot_rejects_stale_revision(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        await repo.save_graph(make_graph("g"))
        with pytest.raises(GraphRevisionConflictError):
            await repo.restore_snapshot("g", "v1", expected_revision=0)

    async def test_trajectory_rejects_unsafe_split(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        with pytest.raises(ValueError, match="invalid split"):
            await repo.append_trajectory("g", "../escape", make_trajectory("q"))

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

    async def test_evolution_events_round_trip(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        event = EvolutionEvent("g", 1, "accepted", 0.5, 1.0, 2, 1.0)
        await repo.append_evolution_event(event)
        events = [item async for item in repo.read_evolution_events("g")]
        assert events == [event]

    async def test_corrupt_trajectory_log_is_quarantined(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        trajectory_path = tmp_path / "graphs" / "g" / "trajectories" / "train.jsonl"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text("{not-json\n", encoding="utf-8")

        with pytest.raises(CorruptRecordError, match="trajectory log quarantined"):
            [item async for item in repo.read_trajectories("g", "train")]

        assert not trajectory_path.exists()

    async def test_corrupt_trajectory_reports_quarantine_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = FilesystemRepository(root=tmp_path)
        trajectory_path = tmp_path / "graphs" / "g" / "trajectories" / "train.jsonl"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text("{not-json\n", encoding="utf-8")

        def fail_quarantine(path: Path) -> Path:
            del path
            raise OSError("read-only storage")

        monkeypatch.setattr(repo_module, "_quarantine_file", fail_quarantine)
        with pytest.raises(CorruptRecordError, match="quarantine failed"):
            [item async for item in repo.read_trajectories("g", "train")]

    async def test_read_trajectories_filtered_by_split(
        self,
        tmp_path: Path,
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
        self,
        tmp_path: Path,
    ) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert trajectories == []

    async def test_save_overwrites(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        new_graph = ProceduralGraph(
            id="g",
            nodes={"answer": Node(id="answer")},
            terminal_ids={"answer"},
        )
        await repo.save_graph(new_graph)
        loaded = await repo.load_graph("g")
        assert "answer" in loaded.nodes
        assert "start" not in loaded.nodes

    async def test_create_rejects_duplicate(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.create_graph(make_graph("g"))
        with pytest.raises(GraphAlreadyExistsError):
            await repo.create_graph(make_graph("g"))

    async def test_compare_and_swap_updates_revision(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        await repo.save_graph(make_graph("g"))
        updated = await repo.save_graph_if_revision(make_graph("g"), expected_revision=0)
        assert updated.revision == 1
        with pytest.raises(GraphRevisionConflictError):
            await repo.save_graph_if_revision(make_graph("g"), expected_revision=0)

    async def test_concurrent_compare_and_swap_has_one_winner(self, tmp_path: Path) -> None:
        repo = FilesystemRepository(root=tmp_path)
        graph = make_graph("concurrent")
        await repo.save_graph(graph)

        outcomes = await asyncio.gather(
            *(
                repo.save_graph_if_revision(
                    graph.model_copy(update={"metadata": {"writer": index}}),
                    expected_revision=0,
                )
                for index in range(8)
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(outcome, ProceduralGraph) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, GraphRevisionConflictError) for outcome in outcomes) == 7
        assert (await repo.load_graph("concurrent")).revision == 1


# ----------------------------------------------------------------------------
# SQLiteRepository
# ----------------------------------------------------------------------------


class TestSQLiteRepository:
    async def test_rate_limit_is_shared_across_repository_instances(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "rate-limit.db"
        first = SQLiteRepository(db_path=database)
        second = SQLiteRepository(db_path=database)

        allowed, retry_after = await first.consume_rate_limit("shared", 1)
        assert allowed
        assert 1 <= retry_after <= 60
        allowed, retry_after = await second.consume_rate_limit("shared", 1)
        assert not allowed
        assert 1 <= retry_after <= 60

        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE rate_limits SET window_started = ? WHERE key = ?",
                (0.0, "shared"),
            )
        allowed, retry_after = await first.consume_rate_limit("shared", 1)
        assert allowed
        assert 1 <= retry_after <= 60

    async def test_rate_limit_concurrent_consumers_are_atomic(self, tmp_path: Path) -> None:
        database = tmp_path / "concurrent-rate-limit.db"
        repositories = [SQLiteRepository(db_path=database) for _ in range(4)]
        outcomes = await asyncio.gather(
            *(
                repository.consume_rate_limit("atomic", 1)
                for repository in repositories
                for _ in range(4)
            )
        )
        assert sum(allowed for allowed, _ in outcomes) == 1

    async def test_rate_limit_validates_inputs(self, tmp_path: Path) -> None:
        repository = SQLiteRepository(db_path=tmp_path / "rate-limit-validation.db")
        with pytest.raises(ValueError, match="key must not be empty"):
            await repository.consume_rate_limit("", 1)
        with pytest.raises(ValueError, match="limit must be positive"):
            await repository.consume_rate_limit("key", 0)

    async def test_concurrent_readers_and_cas_writers_are_safe(self, tmp_path: Path) -> None:
        repository = SQLiteRepository(db_path=tmp_path / "concurrent.db")
        graph = make_graph("concurrent")
        await repository.save_graph(graph)

        readers = await asyncio.gather(*(repository.load_graph("concurrent") for _ in range(32)))
        assert all(reader.id == "concurrent" for reader in readers)

        outcomes = await asyncio.gather(
            *(
                repository.save_graph_if_revision(
                    graph.model_copy(update={"metadata": {"writer": index}}),
                    expected_revision=0,
                )
                for index in range(8)
            ),
            return_exceptions=True,
        )
        successes = [outcome for outcome in outcomes if isinstance(outcome, ProceduralGraph)]
        conflicts = [
            outcome for outcome in outcomes if isinstance(outcome, GraphRevisionConflictError)
        ]
        assert len(successes) == 1
        assert len(conflicts) == 7

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

    async def test_corrupt_graph_record_raises_stable_error(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        with sqlite3.connect(tmp_path / "test.db") as connection:
            connection.execute(
                "INSERT INTO graphs (id, schema_version, body, updated_at) VALUES (?, ?, ?, ?)",
                ("bad", 1, "{not-json", 0.0),
            )
        with pytest.raises(CorruptRecordError, match="corrupt SQLite"):
            await repo.load_graph("bad")

    async def test_save_overwrites(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        # Re-saving the same graph updates the body in place; the row
        # is preserved (no DELETE+INSERT side effects).
        updated = ProceduralGraph(
            id="g",
            nodes={"answer": Node(id="answer")},
            terminal_ids={"answer"},
        )
        await repo.save_graph(updated)
        loaded = await repo.load_graph("g")
        assert loaded.id == "g"
        assert "answer" in loaded.nodes
        assert "start" not in loaded.nodes

    async def test_create_rejects_duplicate(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.create_graph(make_graph("g"))
        with pytest.raises(GraphAlreadyExistsError):
            await repo.create_graph(make_graph("g"))

    async def test_compare_and_swap_updates_revision(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        updated = make_graph("g")
        updated.nodes["answer"].description = "updated"
        result = await repo.save_graph_if_revision(updated, expected_revision=0)
        assert result.revision == 1
        assert (await repo.load_graph("g")).revision == 1
        with pytest.raises(GraphRevisionConflictError):
            await repo.save_graph_if_revision(make_graph("g"), expected_revision=0)

    async def test_compare_and_swap_missing_graph(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        with pytest.raises(FileNotFoundError):
            await repo.save_graph_if_revision(make_graph("missing"), expected_revision=0)

    async def test_snapshot(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        # Re-saving the graph should not affect the snapshot.
        await repo.save_graph(make_graph("g"))
        # Load snapshot via direct SQL to verify.
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

    async def test_restore_snapshot_advances_revision(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        changed = make_graph("g")
        changed.nodes["start"].description = "changed"
        await repo.save_graph(changed)

        restored = await repo.restore_snapshot("g", "v1", expected_revision=1)

        assert restored.revision == 2
        assert restored.nodes["start"].description == ""

    async def test_restore_snapshot_rejects_stale_revision(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.snapshot("g", "v1")
        with pytest.raises(GraphRevisionConflictError):
            await repo.restore_snapshot("g", "v1", expected_revision=1)

    async def test_append_and_read_trajectories(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("q1"))
        await repo.append_trajectory("g", "train", make_trajectory("q2", 0.3))
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert len(trajectories) == 2
        assert [t.task.query for t in trajectories] == ["q1", "q2"]

    async def test_evolution_events_round_trip(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        event = EvolutionEvent("g", 1, "rejected_score", 1.0, 0.5, 1, 1.0)
        await repo.append_evolution_event(event)
        events = [item async for item in repo.read_evolution_events("g")]
        assert events == [event]

    async def test_trajectories_ordered_by_timestamp(
        self,
        tmp_path: Path,
    ) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        for i in range(3):
            await repo.append_trajectory(
                "g",
                "train",
                make_trajectory(f"q{i}"),
            )
        trajectories = [t async for t in repo.read_trajectories("g", "train")]
        assert [t.task.query for t in trajectories] == ["q0", "q1", "q2"]

    async def test_read_trajectories_filtered_by_split(
        self,
        tmp_path: Path,
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
        self,
        tmp_path: Path,
    ) -> None:
        """Deleting a graph cascades to its trajectories via FK."""
        repo = SQLiteRepository(db_path=tmp_path / "test.db")
        await repo.save_graph(make_graph("g"))
        await repo.append_trajectory("g", "train", make_trajectory("q"))
        # Manually delete the graph row; trajectories should cascade.
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
        with sqlite3.connect(tmp_path / "test.db") as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"

    async def test_index_on_trajectories(self, tmp_path: Path) -> None:
        """Verify the (graph_id, split, ts) index exists."""
        SQLiteRepository(db_path=tmp_path / "test.db")
        with sqlite3.connect(tmp_path / "test.db") as conn:
            indices = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
            ).fetchall()
            index_names = {row[0] for row in indices}
            assert "idx_trajectories_graph_split" in index_names
            assert "idx_rejection_graph_ts" in index_names

    async def test_schema_version_recorded(self, tmp_path: Path) -> None:
        """After init, schema_meta contains version 1."""
        SQLiteRepository(db_path=tmp_path / "test.db")
        with sqlite3.connect(tmp_path / "test.db") as conn:
            row = conn.execute("SELECT version FROM schema_meta").fetchone()
            assert row[0] == 1

    async def test_backup_is_integrity_checked_and_loadable(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.db"
        backup_path = tmp_path / "backups" / "methodos.db"
        repository = SQLiteRepository(db_path=source_path)
        await repository.save_graph(make_graph("backup-graph"))
        assert isinstance(repository, BackupRepository)
        await repository.backup_to(backup_path)
        await repository.verify_backup(backup_path)
        restored = SQLiteRepository(db_path=backup_path)
        assert (await restored.load_graph("backup-graph")).id == "backup-graph"

    async def test_backup_rejects_source_path(self, tmp_path: Path) -> None:
        repository = SQLiteRepository(db_path=tmp_path / "source.db")
        with pytest.raises(ValueError, match="differ from the source"):
            await repository.backup_to(repository.db_path)

    async def test_verify_backup_rejects_missing_file(self, tmp_path: Path) -> None:
        repository = SQLiteRepository(db_path=tmp_path / "source.db")
        with pytest.raises(FileNotFoundError, match="backup database not found"):
            await repository.verify_backup(tmp_path / "missing.db")

    async def test_verify_backup_rejects_unsupported_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "unsupported.db"
        with sqlite3.connect(path) as database:
            database.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
            database.execute("INSERT INTO schema_meta (version) VALUES (99)")
        repository = SQLiteRepository(db_path=tmp_path / "source.db")
        with pytest.raises(RuntimeError, match="unsupported backup schema version"):
            await repository.verify_backup(path)

    async def test_verify_backup_rejects_incomplete_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.db"
        with sqlite3.connect(path) as database:
            database.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
            database.execute("INSERT INTO schema_meta (version) VALUES (1)")
        repository = SQLiteRepository(db_path=tmp_path / "source.db")
        with pytest.raises(RuntimeError, match="missing required tables"):
            await repository.verify_backup(path)

    async def test_restore_backup_to_creates_new_verified_database(self, tmp_path: Path) -> None:
        source_path = tmp_path / "source.db"
        backup_path = tmp_path / "backup.db"
        restored_path = tmp_path / "recovered" / "methodos.db"
        repository = SQLiteRepository(db_path=source_path)
        await repository.save_graph(make_graph("recovery-graph"))
        await repository.backup_to(backup_path)

        await repository.restore_backup_to(backup_path, restored_path)

        restored = SQLiteRepository(db_path=restored_path)
        assert (await restored.load_graph("recovery-graph")).id == "recovery-graph"

    async def test_restore_backup_to_rejects_existing_destination(self, tmp_path: Path) -> None:
        repository = SQLiteRepository(db_path=tmp_path / "source.db")
        backup_path = tmp_path / "backup.db"
        destination = tmp_path / "destination.db"
        await repository.backup_to(backup_path)
        destination.write_bytes(b"keep me")

        with pytest.raises(FileExistsError, match="already exists"):
            await repository.restore_backup_to(backup_path, destination)
        assert destination.read_bytes() == b"keep me"

    async def test_restore_backup_to_rejects_invalid_source_without_destination(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "invalid.db"
        destination = tmp_path / "restored.db"
        source.write_bytes(b"not sqlite")
        repository = SQLiteRepository(db_path=tmp_path / "source.db")

        with pytest.raises(sqlite3.DatabaseError):
            await repository.restore_backup_to(source, destination)
        assert not destination.exists()

    async def test_unsupported_schema_version_fails_closed(self, tmp_path: Path) -> None:
        db_path = tmp_path / "future.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO schema_meta (version) VALUES (99)")
        with pytest.raises(RuntimeError, match="unsupported database schema version"):
            SQLiteRepository(db_path=db_path)

    async def test_idempotent_schema_creation(self, tmp_path: Path) -> None:
        """Constructing twice does not raise."""
        path = tmp_path / "test.db"
        SQLiteRepository(db_path=path)
        # Second construction runs ensure_schema again (no-op).
        SQLiteRepository(db_path=path)


# ----------------------------------------------------------------------------
# build_repository factory
# ----------------------------------------------------------------------------


class TestBuildRepository:
    """`build_repository` selects backend from env vars."""

    def test_default_returns_sqlite(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.delenv("PGRAPH_BACKEND", raising=False)
        repo = build_repository()
        assert isinstance(repo, SQLiteRepository)

    def test_filesystem_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "filesystem")
        repo = build_repository()
        assert isinstance(repo, FilesystemRepository)

    def test_unknown_backend_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "bogus")
        with pytest.raises(ValueError, match="unknown PGRAPH_BACKEND"):
            build_repository()

    def test_postgres_backend_requires_dsn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "postgres")
        monkeypatch.delenv("PGRAPH_POSTGRES_DSN", raising=False)
        with pytest.raises(ValueError, match="PGRAPH_POSTGRES_DSN is required"):
            build_repository()

    def test_postgres_backend_constructs_without_connecting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.setenv("PGRAPH_BACKEND", "postgres")
        monkeypatch.setenv("PGRAPH_POSTGRES_DSN", "postgresql://localhost/db")
        repo = build_repository()
        assert isinstance(repo, PostgresRepository)

    def test_postgres_repository_validates_configuration(self) -> None:
        with pytest.raises(ValueError, match="dsn must not be empty"):
            PostgresRepository(dsn=" ")
        with pytest.raises(ValueError, match="pool sizes"):
            PostgresRepository(dsn="postgresql://localhost/db", min_size=0)


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

    def test_upsert_validates_dim(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        with pytest.raises(ValueError, match="vector length 3 != dim 4"):
            idx.upsert("k", [0.1, 0.2, 0.3])

    def test_query_validates_dim(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        with pytest.raises(ValueError, match="vector length 2 != dim 4"):
            idx.query([0.1, 0.2], k=1)

    def test_query_with_k_zero_returns_empty(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        assert idx.query([0.1, 0.2, 0.3, 0.4], k=0) == []

    def test_upsert_and_query_round_trip(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(db_path=tmp_path / "vec.db", dim=4)
        idx.upsert("a", [1.0, 0.0, 0.0, 0.0])
        idx.upsert("b", [0.0, 1.0, 0.0, 0.0])
        # Query for vector closest to "a".
        matches = idx.query([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(matches) == 1
        assert matches[0].key == "a"
        assert matches[0].score > 0.99  # cosine ~1 for identical vectors
