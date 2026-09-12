"""Minimal in-process PGAdapter usage with a live LLM.

Demonstrates the public API end-to-end using the real `LiteLLMClient`
backed by `litellm`. The local solver is a small Protocol-typed class
that uses the injected guidance context to choose actions.

Requires: `OPENAI_API_KEY` (or equivalent) in the environment.

Run with: `python examples/simple_react.py`
"""

from __future__ import annotations

import asyncio

from methodos.adapter import AgentState, PGAdapter, Solver
from methodos.llm import LiteLLMClient
from methodos.schema import (
    Attribute,
    Edge,
    Node,
    ProceduralGraph,
    Relation,
)


class LocalAgent:
    """Minimal host agent that consults guidance context to choose the next action.

    Annotated as a `Solver` Protocol implementation so type checkers
    accept it in `PGAdapter(solver=...)`.

    Production hosts replace this with a real LLM-backed solver that
    inspects `state.context` (which contains the procedural guidance)
    when constructing its prompt.
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

    adapter = PGAdapter(
        solver=LocalAgent(),
        graph=graph,
        llm=LiveDemoLLM(),
    )
    trajectory: list[tuple[str, str]] = []
    for _ in range(3):
        action = await adapter.step(
            query="What is the capital of France?",
            trajectory=trajectory,
        )
        trajectory.append((action, ""))


class LiveDemoLLM:
    """Demonstration LLM using LiteLLMClient under the hood.

    Annotated as an `LLMClient` Protocol implementation. The `complete`
    method delegates to the live LiteLLMClient (which requires a
    configured provider like OPENAI_API_KEY). This demonstrates wiring
    a Protocol-typed dependency around a real implementation.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = LiteLLMClient(model=model)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type | None = None,
        temperature: float = 0.0,
    ) -> str:
        return await self.client.complete(
            system=system,
            user=user,
            json_schema=json_schema,
            temperature=temperature,
        )


if __name__ == "__main__":
    asyncio.run(main())
