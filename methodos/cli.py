"""Typer CLI for `methodos`.

Subcommands:
- `init`: create a new empty graph (or import one from JSON)
- `inspect`: print a graph summary
- `serve`: start the FastAPI service via uvicorn
- `replay`: print trajectories from the trajectory log
- `evolve`: run self-evolution (requires train/val task files; uses a
  stub solver — production use is via the Python SDK)
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

from methodos.adapter import AgentState
from methodos.observability import configure_logging as _configure_root_logging
from methodos.repo import Repository, Task, build_repository

logger = logging.getLogger(__name__)

cli = typer.Typer(
    name="methodos",
    help="Self-evolving procedural graph adapter for LLM agents.",
    no_args_is_help=True,
)


def configure_logging(verbose: bool, json_format: bool = False) -> None:
    """Configure root logger; --verbose sets DEBUG.

    Args:
        verbose: When True, set log level to DEBUG; otherwise INFO.
        json_format: When True, emit structured JSON lines instead of
            human-readable text (suitable for log aggregators).
    """
    _configure_root_logging(
        level="DEBUG" if verbose else "INFO",
        json_format=json_format,
    )


@cli.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """methodos — queryable know-how for LLM agents."""
    configure_logging(verbose)


def build_repo_from_env() -> Repository:
    """Default repository from environment variables."""
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
    from methodos.schema import ProceduralGraph

    async def run() -> ProceduralGraph:
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

    asyncio.run(run())


@cli.command()
def inspect(
    graph_id: Annotated[str, typer.Option(help="Graph identifier to inspect.")] = "default",
) -> None:
    """Print a graph summary."""
    from methodos.schema import ProceduralGraph

    async def run() -> None:
        repo = build_repo_from_env()
        graph = await repo.load_graph(graph_id)
        if not isinstance(graph, ProceduralGraph):
            typer.echo(f"unexpected return type: {type(graph).__name__}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Graph {graph.id!r}:")
        typer.echo(f"  schema_version: {graph.schema_version}")
        typer.echo(f"  nodes: {len(graph.nodes)}")
        typer.echo(f"  edges: {len(graph.edges)}")
        typer.echo(f"  terminals: {len(graph.terminal_ids)}")
        typer.echo(f"  metadata: {graph.metadata}")

    asyncio.run(run())


@cli.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
    log_json: Annotated[
        bool,
        typer.Option(help="Emit structured JSON log lines."),
    ] = False,
) -> None:
    """Start the FastAPI service via uvicorn."""
    configure_logging(verbose=False, json_format=log_json)
    import uvicorn

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

    async def run() -> None:
        repo = build_repo_from_env()
        count = 0
        async for traj in repo.read_trajectories(graph_id, split):
            if count >= limit:
                break
            typer.echo(f"task: {traj.task.query!r} score={traj.score:.2f} steps={len(traj.steps)}")
            count += 1
        if count == 0:
            typer.echo(f"(no trajectories for graph {graph_id!r} split={split!r})")

    asyncio.run(run())


class _StubSolver:
    """Alternating-action solver so the rollout doesn't trigger a doom loop.

    This is a placeholder — production deployments supply a real LLM-backed
    solver via the Python SDK. The CLI exists to demonstrate the
    evolution loop wiring end-to-end without an API key.
    """

    async def step(self, state: AgentState) -> str:
        if not state.trajectory:
            return "start"
        last = state.trajectory[-1][0]
        if last == "start":
            return "answer"
        return "start"


def load_tasks(path: Path) -> list[Task]:
    """Read JSONL tasks from `path`. Each line: {"query": ..., "expected": ...}."""
    tasks: list[Task] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        tasks.append(Task(query=record["query"], expected=record.get("expected")))
    return tasks


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
    rejection_memory_size: Annotated[
        int, typer.Option(min=1, max=1000, help="Rejection memory capacity.")
    ] = 32,
    allow_cycles: Annotated[
        bool, typer.Option(help="Permit cycles in candidate graphs (off by default).")
    ] = False,
) -> None:
    """Run K rounds of self-evolution on the graph.

    This is a thin wrapper around `EvolutionEngine`. The CLI uses an
    alternating-action stub solver (no API key required) so the wiring
    can be exercised end-to-end. Production use is via the Python SDK
    with a real LLM-backed solver.
    """
    from methodos.evolution import EvolutionEngine
    from methodos.llm import LiteLLMClient

    async def run() -> None:
        repo = build_repo_from_env()
        if train_path is None or val_path is None:
            typer.echo(
                "Both --train-path and --val-path are required.",
                err=True,
            )
            raise typer.Exit(code=2)
        graph = await repo.load_graph(graph_id)
        train_tasks = load_tasks(train_path)
        val_tasks = load_tasks(val_path)
        llm = LiteLLMClient(model=model)
        engine = EvolutionEngine(
            llm=llm,
            repo=repo,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            solver=_StubSolver(),
            k_rounds=k_rounds,
            rejection_memory_size=rejection_memory_size,
            allow_cycles=allow_cycles,
        )
        typer.echo(
            f"starting evolution: graph={graph_id!r} k={k_rounds} "
            f"train={len(train_tasks)} val={len(val_tasks)} model={model!r}"
        )
        final = await engine.run(graph)
        typer.echo(f"evolution complete: {len(final.nodes)} nodes, {len(final.edges)} edges")

    asyncio.run(run())


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

    async def run() -> None:
        try:
            from eval.hotpotqa.run import run_eval
        except ImportError as exc:
            typer.echo(
                f"Eval harness unavailable: {exc}. Install with `pip install methodos[eval]`.",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        await run_eval(graph_id=graph_id, n=n, seed=seed, model=model)

    asyncio.run(run())


__all__ = ["cli", "main"]


if __name__ == "__main__":
    cli()
