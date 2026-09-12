"""Run `methodos` self-evolution from scratch using synthetic tasks.

Demonstrates Algorithm 1 end-to-end with the live `LiteLLMClient`
backed by `litellm` and the production `EvolutionEngine`. Useful as a
smoke test or as a template for wiring methodos into a real agent
environment.

Requirements: `pip install methodos` and an OPENAI_API_KEY (or
equivalent) configured in your environment.

Run with: `python examples/evolve_from_scratch.py`
"""

from __future__ import annotations

import asyncio
import logging

from methodos.adapter import AgentState, Solver
from methodos.evolution import EvolutionEngine
from methodos.llm import LiteLLMClient
from methodos.repo import Task, build_repository
from methodos.schema import (
    Attribute,
    Edge,
    Node,
    ProceduralGraph,
    Relation,
)


class LocalAgent:
    """Minimal host agent that always returns "answer".

    Annotated as a `Solver` Protocol implementation so type checkers
    accept it in `EvolutionEngine(solver=...)`.

    Production hosts replace this with a real LLM-backed solver that
    uses `state.context` (which contains the procedural guidance) when
    constructing its prompt.
    """

    async def step(self, state: AgentState) -> str:
        """Return the next action name; this demo always returns `answer`."""
        return "answer"


async def main() -> None:
    """Run the example end-to-end (requires `OPENAI_API_KEY`)."""
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
        solver=LocalAgent(),
        k_rounds=3,
    )
    final_graph = await engine.run(graph)
    print(f"final graph: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")


if __name__ == "__main__":
    asyncio.run(main())
