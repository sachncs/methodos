"""Minimal in-process PGAdapter usage with a stub solver.

Demonstrates the public API end-to-end without any environment. The
`StubSolver` is a placeholder — replace with a real LLM-backed solver
in production.

Requires: `pip install methodos`
"""

from __future__ import annotations

import asyncio

from methodos import (
    AgentState,
    Attribute,
    Edge,
    LiteLLMClient,
    LLMClient,
    Node,
    PGAdapter,
    ProceduralGraph,
    Relation,
    Solver,
)


class StubSolver:
    """A placeholder solver that returns a fixed action.

    Annotated as a `Solver` Protocol implementation so type checkers
    accept it in `PGAdapter(solver=...)`.

    Real usage: implement `async def step(state) -> str` that consults
    `state.context` (which contains the procedural guidance) when
    constructing your LLM prompt, then returns the next action name.
    """

    async def step(self, state: AgentState) -> str:
        return "search"


async def main() -> None:
    graph = ProceduralGraph(
        id="example",
        nodes={
            "start": Node(id="start", description="begin task"),
            "search": Node(id="search", description="search Wikipedia"),
            "answer": Node(id="answer", description="final answer"),
        },
        edges=[
            Edge(
                src="start",
                dst="search",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="need information",
                    guidance="call search with focused query",
                    pitfalls="don't search with the full question",
                ),
            ),
            Edge(
                src="search",
                dst="answer",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="have enough info",
                    guidance="synthesize a concise answer",
                    pitfalls="don't repeat observations",
                ),
            ),
        ],
        terminal_ids={"answer"},
    )

    # Use a stub LLM; replace with LiteLLMClient(model="gpt-4o-mini") in prod.
    class StubLLM:
        """A placeholder LLM satisfying the `LLMClient` Protocol."""

        async def complete(
            self,
            *,
            system: str,
            user: str,
            json_schema=None,
            temperature: float = 0.0,
        ) -> str:
            return "Consider the most recent search result and answer."

    adapter = PGAdapter(
        solver=StubSolver(),
        graph=graph,
        llm=StubLLM(),
    )
    trajectory: list[tuple[str, str]] = []
    for _ in range(3):
        action = await adapter.step(
            query="What is the capital of France?",
            trajectory=trajectory,
        )
        trajectory.append((action, "stub observation"))


if __name__ == "__main__":
    asyncio.run(main())
