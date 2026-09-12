"""HotpotQA evaluation runner: paired with-PG vs without-PG comparison."""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from eval.hotpotqa.solver import HotpotQASolver, echo_search
from eval.hotpotqa.tasks import DEFAULT_DATA_DIR, download_if_missing, load_tasks
from methodos.adapter import AgentState, PGAdapter
from methodos.llm import LiteLLMClient
from methodos.repo import Repository, build_repository
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One result row."""

    task_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    with_pg: bool
    em: float  # exact match
    f1: float  # token F1


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize(predicted) == normalize(gold) else 0.0


def token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = normalize(predicted).split()
    gold_tokens = normalize(gold).split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in predicted_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class EvalAggregates:
    em: float
    f1: float
    n: int


def aggregate(results: list[EvalResult]) -> EvalAggregates:
    if not results:
        return EvalAggregates(em=0.0, f1=0.0, n=0)
    return EvalAggregates(
        em=sum(r.em for r in results) / len(results),
        f1=sum(r.f1 for r in results) / len(results),
        n=len(results),
    )


async def run_one(
    *,
    question: str,
    task_id: str,
    expected: str,
    solver_factory: Callable[[], HotpotQASolver],
    adapter_factory: Callable[[HotpotQASolver, ProceduralGraph | None], PGAdapter | None],
    graph: ProceduralGraph | None,
) -> EvalResult:
    """Run one question with the given setup. Returns one EvalResult."""
    solver = solver_factory()
    adapter = adapter_factory(solver, graph)
    trajectory: list[tuple[str, str]] = []
    predicted = ""
    for _ in range(solver.max_steps):
        if adapter is not None:
            action = await adapter.step(query=question, trajectory=trajectory)
        else:
            action = await solver.step(AgentState(
                query=question,
                trajectory=tuple(trajectory),
                context="",
            ))
        if "ANSWER:" in action.upper():
            predicted = action.split(":", 1)[1].strip() if ":" in action else action
            break
        trajectory.append((action, ""))
    return EvalResult(
        task_id=task_id,
        question=question,
        expected_answer=expected,
        predicted_answer=predicted,
        with_pg=adapter is not None,
        em=exact_match(predicted, expected),
        f1=token_f1(predicted, expected),
    )


async def run_eval(
    *,
    graph_id: str | None,
    n: int,
    seed: int,
    model: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    limit: int = 500,
) -> None:
    """Run a paired with-PG vs without-PG evaluation over HotpotQA.

    Args:
        graph_id: optional graph id to load via `build_repository()`.
        n: number of examples to evaluate.
        seed: RNG seed for shuffling the task order.
        model: litellm model identifier for the solver LLM.
        data_dir: directory to store / find HotpotQA JSONL.
        limit: max examples to download on first run.
    """
    repo: Repository = build_repository()
    graph: ProceduralGraph | None = None
    if graph_id is not None:
        try:
            graph = await repo.load_graph(graph_id)
        except FileNotFoundError:
            logger.warning("graph %r not found; running without PG", graph_id)
            graph = None

    data_path = download_if_missing(target_dir=data_dir, limit=limit)
    tasks = list(load_tasks(path=data_path, limit=n))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    logger.info("running paired eval over %d HotpotQA examples", len(tasks))

    llm = LiteLLMClient(model=model)
    search = echo_search  # host wires a real backend for production

    def solver_factory() -> HotpotQASolver:
        return HotpotQASolver(llm=llm, search=search, max_steps=4)

    def adapter_factory(
        solver: HotpotQASolver, g: ProceduralGraph | None
    ) -> PGAdapter | None:
        if g is None:
            return None
        return PGAdapter(solver=solver, graph=g, llm=llm)

    with_results: list[EvalResult] = []
    without_results: list[EvalResult] = []

    for task in tasks:
        with_results.append(await run_one(
            question=task.question,
            task_id=task.id,
            expected=task.answer,
            solver_factory=solver_factory,
            adapter_factory=adapter_factory,
            graph=graph,
        ))
        without_results.append(await run_one(
            question=task.question,
            task_id=task.id,
            expected=task.answer,
            solver_factory=solver_factory,
            adapter_factory=adapter_factory,
            graph=None,
        ))

    with_agg = aggregate(with_results)
    without_agg = aggregate(without_results)
    print(
        f"HotpotQA eval (n={len(tasks)})\n"
        f"  with PG:    EM={with_agg.em:.4f}  F1={with_agg.f1:.4f}\n"
        f"  without PG: EM={without_agg.em:.4f}  F1={without_agg.f1:.4f}\n"
        f"  delta:      EM={with_agg.em - without_agg.em:+.4f}  "
        f"F1={with_agg.f1 - without_agg.f1:+.4f}\n"
    )


def main() -> None:
    """CLI entry: `python -m eval.hotpotqa.run --graph-id X --n 200`."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    asyncio.run(run_eval(
        graph_id=args.graph_id,
        n=args.n,
        seed=args.seed,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
