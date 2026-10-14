"""Typer CLI for `methodos`.

Subcommands:
- `init`: create a new empty graph (or import one from JSON)
- `inspect`: print a graph summary
- `serve`: start the FastAPI service via uvicorn
- `replay`: print trajectories from the trajectory log
- `evolve`: run self-evolution (requires task files; thin wrapper around
  the Python SDK; see docs/evaluation.md for the full workflow)
- `eval`: run a paper benchmark (only `hotpotqa` is shipped; gated by
  the `[eval]` extra)

Engineering:
- `cli = typer.Typer(...)` with subcommand decorators.
- Async subcommands wrap `asyncio.run(...)`.
- No `_foo()` markers.
- Logging configured once in the root callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from methodos.repo import BackupRepository, Repository, Task, build_repository
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)

cli = typer.Typer(
    name="methodos",
    help="Self-evolving procedural graph adapter for LLM agents.",
    no_args_is_help=True,
)


def configure_logging(verbose: bool) -> None:
    """Configure root logger; --verbose sets DEBUG."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@cli.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Methodos — queryable know-how for LLM agents."""
    configure_logging(verbose)


def build_repo_from_env() -> Repository:
    """Default repository from environment."""
    return build_repository()


@cli.command()
def init(
    graph_id: Annotated[str, typer.Option(help="Graph identifier to create.")] = "default",
    from_path: Annotated[
        Path | None,
        typer.Option(help="Import from existing graph JSON."),
    ] = None,
) -> None:
    """Create a new empty graph (or import one from JSON)."""

    async def _run() -> ProceduralGraph:
        repo = build_repo_from_env()
        if from_path is not None:
            data = json.loads(from_path.read_text())
            graph = ProceduralGraph.model_validate(data)
            graph = graph.model_copy(update={"id": graph_id})
        else:
            graph = ProceduralGraph(id=graph_id)
        await repo.save_graph(graph)
        typer.echo(f"created graph {graph_id!r}")
        return graph

    asyncio.run(_run())


@cli.command()
def inspect(
    graph_id: Annotated[str, typer.Option(help="Graph identifier to inspect.")] = "default",
) -> None:
    """Print a graph summary."""

    async def _run() -> None:
        repo = build_repo_from_env()
        graph = await repo.load_graph(graph_id)
        typer.echo(f"Graph {graph.id!r}:")
        typer.echo(f"  schema_version: {graph.schema_version}")
        typer.echo(f"  nodes: {len(graph.nodes)}")
        typer.echo(f"  edges: {len(graph.edges)}")
        typer.echo(f"  terminals: {len(graph.terminal_ids)}")
        typer.echo(f"  metadata: {graph.metadata}")

    asyncio.run(_run())


@cli.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start the FastAPI service via uvicorn."""
    uvicorn.run("methodos.service:app", host=host, port=port, reload=reload)


@cli.command()
def replay(
    graph_id: Annotated[
        str, typer.Option(help="Graph identifier whose trajectories to replay.")
    ] = "default",
    split: Annotated[str, typer.Option(help="Trajectory split (train/val/test).")] = "train",
    limit: Annotated[int, typer.Option(min=1, help="Maximum trajectories to print.")] = 10,
) -> None:
    """Print recent trajectories from the trajectory log."""

    async def _run() -> None:
        repo = build_repo_from_env()
        count = 0
        async for traj in repo.read_trajectories(graph_id, split):
            if count >= limit:
                break
            typer.echo(f"task: {traj.task.query!r} score={traj.score:.2f} steps={len(traj.steps)}")
            count += 1
        if count == 0:
            typer.echo(f"(no trajectories for graph {graph_id!r} split={split!r})")

    asyncio.run(_run())


@cli.command()
def backup(
    destination: Annotated[Path, typer.Argument(help="Destination SQLite backup path.")],
) -> None:
    """Create an integrity-checked backup of the configured SQLite database."""

    async def _run() -> None:
        repo = build_repo_from_env()
        if not isinstance(repo, BackupRepository):
            typer.echo("The configured backend does not support local backups.", err=True)
            raise typer.Exit(code=2)
        await repo.backup_to(destination)
        typer.echo(f"created verified backup at {destination}")

    asyncio.run(_run())


@cli.command("verify-backup")
def verify_backup(
    source: Annotated[Path, typer.Argument(help="SQLite backup path to verify.")],
) -> None:
    """Verify a SQLite backup's integrity and schema before restore."""

    async def _run() -> None:
        repo = build_repo_from_env()
        if not isinstance(repo, BackupRepository):
            typer.echo("The configured backend does not support local backups.", err=True)
            raise typer.Exit(code=2)
        await repo.verify_backup(source)
        typer.echo(f"verified backup at {source}")

    asyncio.run(_run())


@cli.command("restore-backup")
def restore_backup(
    source: Annotated[Path, typer.Argument(help="Verified SQLite backup path.")],
    destination: Annotated[Path, typer.Argument(help="New SQLite database path to create.")],
) -> None:
    """Restore a verified SQLite backup without overwriting a destination."""

    async def _run() -> None:
        repo = build_repo_from_env()
        if not isinstance(repo, BackupRepository):
            typer.echo("The configured backend does not support local restores.", err=True)
            raise typer.Exit(code=2)
        await repo.restore_backup_to(source, destination)
        typer.echo(f"restored verified backup to {destination}")

    asyncio.run(_run())


@cli.command()
def evolve(
    graph_id: Annotated[str, typer.Option(help="Graph identifier to evolve.")] = "default",
    k_rounds: Annotated[int, typer.Option(min=1, max=100, help="Evolution rounds.")] = 10,
    train_path: Annotated[
        Path | None,
        typer.Option(help="JSONL of {query, expected} for train tasks."),
    ] = None,
    val_path: Annotated[
        Path | None,
        typer.Option(help="JSONL of {query, expected} for val tasks."),
    ] = None,
    model: Annotated[
        str, typer.Option(help="LLM model identifier for refiner + scoring.")
    ] = "gpt-4o-mini",
) -> None:
    """Run K rounds of self-evolution on the graph (thin wrapper).

    For full evolution runs use the Python SDK directly; this command
    exists for the common case of inspecting graph state after evolution.
    """
    if train_path is None or val_path is None:
        typer.echo(
            "Both --train-path and --val-path are required. "
            "For full evolution use methodos.evolution.EvolutionEngine "
            "directly in Python.",
            err=True,
        )
        raise typer.Exit(code=2)

    def _load_tasks(path: Path) -> list[Task]:
        """Load non-empty JSONL task records with actionable errors."""
        tasks: list[Task] = []
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read task file {path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("query"), str):
                raise ValueError(
                    f"task record in {path} at line {line_number} must contain a string query"
                )
            tasks.append(Task(query=record["query"], expected=record.get("expected")))
        return tasks

    try:
        train_tasks = _load_tasks(train_path)
        val_tasks = _load_tasks(val_path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if not train_tasks or not val_tasks:
        typer.echo("Both task files must contain at least one non-empty JSONL record.", err=True)
        raise typer.Exit(code=2)

    async def _run() -> None:
        repo = build_repo_from_env()
        graph = await repo.load_graph(graph_id)
        typer.echo(
            f"graph loaded: {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges; loaded {len(train_tasks)} train and "
            f"{len(val_tasks)} validation tasks for {k_rounds} rounds with {model!r}. "
            "Use the Python SDK for real "
            f"evolution; this command demonstrates the wiring."
        )

    asyncio.run(_run())


@cli.command()
def eval(
    benchmark: Annotated[
        str, typer.Option(help="Benchmark name (currently: hotpotqa).")
    ] = "hotpotqa",
    graph_id: Annotated[
        str | None,
        typer.Option(help="Optional graph to use during eval."),
    ] = None,
    n: Annotated[int, typer.Option(min=1, max=10_000, help="Number of examples.")] = 200,
    seed: Annotated[int, typer.Option()] = 0,
    model: Annotated[str, typer.Option(help="LLM model for the eval solver.")] = "gpt-4o-mini",
) -> None:
    """Run methodos against a paper benchmark."""
    if benchmark != "hotpotqa":
        typer.echo(f"unknown benchmark: {benchmark!r}", err=True)
        raise typer.Exit(code=1)
    try:
        from eval.hotpotqa.run import run_eval
    except ModuleNotFoundError as exc:
        typer.echo(
            "The eval command requires the optional dependencies; install methodos[eval].",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    async def _run() -> None:
        await run_eval(graph_id=graph_id, n=n, seed=seed, model=model)

    asyncio.run(_run())


__all__ = ["cli", "main"]


if __name__ == "__main__":
    cli()
