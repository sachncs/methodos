"""Run `methodos` self-evolution from scratch using synthetic tasks.

Demonstrates Algorithm 1 end-to-end. Useful as a smoke test or as a
template for wiring methodos into a real agent environment.

Requirements: `pip install methodos` and an OPENAI_API_KEY (or
equivalent) configured in your environment.
"""

from __future__ import annotations

import asyncio
import logging

from methodos import (
    Attribute,
    Edge,
    LiteLLMClient,
    Node,
    ProceduralGraph,
    Relation,
)
from methodos.adapter import AgentState, Solver
from methodos.evolution import EvolutionEngine
from methodos.repo import Task, build_repository


class StubSolver:
    """A minimal solver that always returns "answer".

    Replace this with a real LLM-backed solver that uses `state.context`
    (which contains the procedural guidance) when constructing its
    prompt. This stub demonstrates the wiring only.
    """

    async def step(self, state: AgentState) -> str:
        return "answer"


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    graph = ProceduralGraph(
        id="scratch-evo",
        nodes={
            "start": Node(id="start", description="begin task"),
            "answer": Node(id="answer", description="final answer"),
        },
        terminal_ids={"answer"},
    )

    repo = build_repository()
    await repo.save_graph(graph)
    llm = LiteLLMClient(model="gpt-4o-mini")

    train = [Task(query=f"Synthetic train task #{i}") for i in range(3)]
    val = [Task(query=f"Synthetic val task #{i}") for i in range(2)]

    engine = EvolutionEngine(
        llm=llm,
        repo=repo,
        train_tasks=train,
        val_tasks=val,
        solver=StubSolver(),  # type: ignore[arg-type]
        k_rounds=3,
    )
    final_graph = await engine.run(graph)
    print(f"final graph: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")


if __name__ == "__main__":
    asyncio.run(main())
