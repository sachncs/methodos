# methodos

Self-evolving procedural graph adapter for LLM agents.

> *methodos — the way your agent proceeds*

`methodos` is a framework that wraps any ReAct-style LLM agent with a
queryable **procedural graph** that encodes *know-how* as navigable
transitions, and **improves itself** from execution feedback.

The procedural graph is the procedural counterpart to a knowledge graph: it
answers *what-to-do* questions by exposing a localized subgraph of
admissible next procedures with textual attributes. A separate LLM call
translates that subgraph into step-level guidance for the agent.

The graph **self-evolves** offline: an LLM refiner contrasts successful and
failed trajectories and proposes edits to the graph's topology and
attributes; edits are accepted only if they don't degrade held-out
validation performance.

## Install

```bash
pip install methodos
```

With eval harness (HotpotQA):

```bash
pip install methodos[eval]
```

With sqlite-vec semantic search:

```bash
pip install methodos[vec]
```

## Quickstart

```python
from methodos import (
    Node, Edge, Attribute, Relation, ProceduralGraph,
    PGAdapter, Solver, AgentState,
    LiteLLMClient,
)
import asyncio


class MySolver:
    """Any object with an async step() method satisfies the Solver Protocol."""
    async def step(self, state: AgentState) -> str:
        # Your LLM-backed decision logic; `state.context` includes the guidance.
        return "search"


async def main() -> None:
    graph = ProceduralGraph(
        id="example",
        nodes={
            "start": Node(id="start"),
            "search": Node(id="search", description="search Wikipedia"),
            "answer": Node(id="answer"),
        },
        edges=[
            Edge(
                src="start", dst="search", relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="need information",
                    guidance="call search with focused query",
                    pitfalls="don't search with the full question",
                ),
            ),
            Edge(
                src="search", dst="answer", relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="have enough info",
                    guidance="synthesize a concise answer",
                    pitfalls="don't repeat observations",
                ),
            ),
        ],
        terminal_ids={"answer"},
    )

    llm = LiteLLMClient(model="gpt-4o-mini")
    adapter = PGAdapter(solver=MySolver(), graph=graph, llm=llm)

    trajectory: list[tuple[str, str]] = []
    for _ in range(3):
        action = await adapter.step(
            query="What is the capital of France?",
            trajectory=trajectory,
        )
        trajectory.append((action, "(host provides observation)"))


asyncio.run(main())
```

## CLI

```bash
methodos init                  # create a default graph
methodos inspect               # show graph summary
methodos serve                 # start FastAPI on 127.0.0.1:8000
methodos evolve --train-path train.jsonl --val-path val.jsonl --k-rounds 10
methodos replay --split train --limit 10
methodos eval hotpotqa --n 200 --model gpt-4o-mini
```

## Documentation

See `docs/` for the full reference:

- [Architecture](docs/architecture.md)
- [API reference](docs/api.md)
- [Deployment](docs/deployment.md)
- [Evaluation](docs/evaluation.md)
- [Paper mapping](docs/paper-mapping.md)

## License

Apache-2.0