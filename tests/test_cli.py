"""Tests for `methodos.cli` (Typer subcommands)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from methodos.cli import cli
from methodos.repo import (
    FilesystemRepository,
    SQLiteRepository,
    Task,
    Trajectory,
)
from methodos.schema import ProceduralGraph


@pytest.fixture
def runner() -> CliRunner:
    """Typer's CliRunner for invoking subcommands."""
    return CliRunner()


@pytest.fixture
def seeded_filesystem_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> FilesystemRepository:
    """Filesystem repo seeded with one graph; PGRAPH_HOME set."""
    from methodos.schema import Node
    monkeypatch.setenv("PGRAPH_HOME", str(tmp_path / "home"))
    repo = FilesystemRepository(root=tmp_path / "home")
    import asyncio
    graph = ProceduralGraph(
        id="g",
        nodes={"start": Node(id="start"), "answer": Node(id="answer")},
        terminal_ids={"answer"},
    )
    asyncio.run(repo.save_graph(graph))
    # Also append one trajectory for replay tests
    asyncio.run(repo.append_trajectory(
        "g", "train",
        Trajectory(task=Task(query="tq"), steps=(("a", "b"),), score=0.7),
    ))
    return repo


@pytest.fixture
def seeded_sqlite_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> SQLiteRepository:
    """SQLite repo seeded; PGRAPH_HOME set to override default location."""
    from methodos.schema import Node
    monkeypatch.setenv("PGRAPH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PGRAPH_BACKEND", "sqlite")
    repo = SQLiteRepository(db_path=tmp_path / "home" / "test.db")
    import asyncio
    graph = ProceduralGraph(
        id="g",
        nodes={"start": Node(id="start"), "answer": Node(id="answer")},
        terminal_ids={"answer"},
    )
    asyncio.run(repo.save_graph(graph))
    return repo


# ----------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------


class TestInitCommand:
    def test_creates_empty_graph(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        result = runner.invoke(
            cli, ["init", "--graph-id", "new"],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "created graph 'new'" in result.stdout

    def test_imports_from_json(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from methodos.schema import Node
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        graph_path = tmp_path / "input.json"
        graph_path.write_text(
            json.dumps(ProceduralGraph(
                id="x", nodes={"a": Node(id="a")}, terminal_ids={"a"},
            ).model_dump(mode="json"))
        )
        result = runner.invoke(
            cli, ["init", "--graph-id", "y", "--from-path", str(graph_path)],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "created graph 'y'" in result.stdout

    def test_missing_from_path_fails(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        result = runner.invoke(
            cli, ["init", "--graph-id", "x", "--from-path", str(tmp_path / "nope.json")],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
        )
        assert result.exit_code != 0


# ----------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------


class TestInspectCommand:
    def test_prints_summary(
        self, runner: CliRunner, seeded_filesystem_repo: FilesystemRepository,
    ) -> None:
        import os
        result = runner.invoke(
            cli, ["inspect", "--graph-id", "g"],
            env={"PGRAPH_HOME": os.environ["PGRAPH_HOME"], "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (result.exit_code, result.stdout)
        assert "Graph 'g'" in result.stdout
        assert "nodes: 2" in result.stdout
        assert "edges: 0" in result.stdout

    def test_missing_graph_exits_nonzero(
        self, runner: CliRunner, seeded_filesystem_repo: FilesystemRepository,
    ) -> None:
        import os
        result = runner.invoke(
            cli, ["inspect", "--graph-id", "nope"],
            env={"PGRAPH_HOME": os.environ["PGRAPH_HOME"], "PGRAPH_BACKEND": "filesystem"},
        )
        assert result.exit_code != 0


# ----------------------------------------------------------------------------
# replay
# ----------------------------------------------------------------------------


class TestReplayCommand:
    def test_prints_trajectories(
        self, runner: CliRunner, seeded_filesystem_repo: FilesystemRepository,
    ) -> None:
        import os
        result = runner.invoke(
            cli, ["replay", "--graph-id", "g"],
            env={"PGRAPH_HOME": os.environ["PGRAPH_HOME"], "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "tq" in result.stdout

    def test_no_trajectories_prints_marker(
        self, runner: CliRunner, seeded_filesystem_repo: FilesystemRepository,
    ) -> None:
        import os
        result = runner.invoke(
            cli, ["replay", "--graph-id", "g", "--split", "val"],
            env={"PGRAPH_HOME": os.environ["PGRAPH_HOME"], "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "no trajectories" in result.stdout


# ----------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------


class TestEvalCommand:
    def test_unknown_benchmark_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--benchmark", "bogus"])
        assert result.exit_code == 1
        # typer echoes via stderr for error paths
        combined = (result.stdout or "") + (result.stderr or "")
        assert "unknown benchmark" in combined

    def test_eval_without_extra_fails_gracefully(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When [eval] extra is not installed, the import fails. The
        # CLI surfaces a clear message and exits non-zero.
        import builtins
        original_import = builtins.__import__

        def fake_import(
            name: str,
            globals: object | None = None,
            locals: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "eval.hotpotqa.run":
                raise ImportError("eval.hotpotqa.run not installed")
            return original_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = runner.invoke(cli, ["eval", "--benchmark", "hotpotqa"])
        assert result.exit_code == 1
        combined = (result.stdout or "") + (result.stderr or "")
        assert "Eval harness unavailable" in combined


# ----------------------------------------------------------------------------
# evolve
# ----------------------------------------------------------------------------


class TestEvolveCommand:
    def test_missing_paths_fails(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        # Typer rejects missing required options with exit code 2.
        result = runner.invoke(
            cli, ["evolve", "--graph-id", "x"],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
        )
        assert result.exit_code != 0

    def test_loads_tasks(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        from methodos.repo import FilesystemRepository
        from methodos.schema import Node, ProceduralGraph as _PG
        import asyncio
        repo = FilesystemRepository(root=tmp_path)
        asyncio.run(repo.save_graph(_PG(
            id="x", nodes={"start": Node(id="start"), "answer": Node(id="answer")},
            terminal_ids={"answer"},
        )))
        train = tmp_path / "train.jsonl"
        train.write_text(json.dumps({"query": "q1"}) + "\n")
        val = tmp_path / "val.jsonl"
        val.write_text(json.dumps({"query": "v1"}) + "\n")
        result = runner.invoke(
            cli, [
                "evolve",
                "--graph-id", "x",
                "--train-path", str(train),
                "--val-path", str(val),
            ],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        # Command loads tasks then prints a usage message; exits 0.
        assert result.exit_code == 0
        assert "Python SDK" in result.stdout


# ----------------------------------------------------------------------------
# help and root callback
# ----------------------------------------------------------------------------


class TestRootCallback:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "methodos" in result.stdout

    def test_no_args_exits_nonzero_but_shows_usage(
        self, runner: CliRunner,
    ) -> None:
        # Typer's `no_args_is_help=True` triggers a SystemExit(2) when no
        # args are provided; the usage is still printed to stdout.
        result = runner.invoke(cli, [])
        assert result.exit_code != 0
        assert "Usage:" in result.stdout or "methodos" in result.stdout

    def test_verbose_flag_accepted(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        result = runner.invoke(
            cli, ["-v", "init", "--graph-id", "v"],
            env={"PGRAPH_HOME": str(tmp_path), "PGRAPH_BACKEND": "filesystem"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
