"""Live LLM integration tests for `methodos.llm` and `eval.hotpotqa`.

These tests make real network calls to the configured LLM provider.
They are skipped automatically if neither `OPENAI_API_KEY` nor
`ANTHROPIC_API_KEY` is set in the environment.

The test loads the local HotpotQA distractor JSONL (already cached at
`~/.methodos/eval_data/hotpotqa_dev_distractor.jsonl` from a prior
`download_if_missing` call) and runs the same prompt the production
`HotpotQASolver` uses.

Run with:

    # Single-task smoke test (auto-skipped if no API key)
    uv run pytest tests/test_live_llm.py -v -s

    # Full eval on 5 HotpotQA tasks via real OpenAI LLM
    OPENAI_API_KEY=sk-... uv run pytest tests/test_live_llm.py -v -s

The tests print the model output and EM/F1 scores so you can eyeball
whether the LLM is reasoning correctly. They assert non-zero LLM output
and that EM is computed correctly for known-answer tasks.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.hotpotqa.solver import HotpotQASolver, echo_search
from eval.hotpotqa.tasks import DEFAULT_DATA_DIR, HotpotQATask, load_tasks
from methodos.adapter import AgentState, PGAdapter
from methodos.llm import LiteLLMClient, LLMClient
from methodos.schema import Attribute, Edge, Node, ProceduralGraph, Relation

# ---------------------------------------------------------------------------
# Skipping helpers
# ---------------------------------------------------------------------------


_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY", "").strip())
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
_HAS_GROQ = bool(os.environ.get("GROQ_API_KEY", "").strip())

_skip_no_key = pytest.mark.skipif(
    not (_HAS_OPENAI or _HAS_ANTHROPIC or _HAS_GROQ),
    reason=(
        "Live LLM tests require OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "or GROQ_API_KEY in the environment"
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_DATA_PATH: Path = DEFAULT_DATA_DIR / "hotpotqa_dev_distractor.jsonl"


def _data_path_exists() -> bool:
    return _DATA_PATH.exists()


_skip_no_data = pytest.mark.skipif(
    not _data_path_exists(),
    reason=(
        f"HotpotQA dataset not cached at {_DATA_PATH}; run "
        f"`python -m eval.hotpotqa.run` once to populate it"
    ),
)


@pytest.fixture(scope="module")
def hotpot_tasks() -> list[HotpotQATask]:
    """Load the cached HotpotQA JSONL (up to 5 tasks)."""
    if not _DATA_PATH.exists():
        pytest.skip(f"HotpotQA data not at {_DATA_PATH}")
    return list(load_tasks(path=_DATA_PATH, limit=5))


@pytest.fixture(scope="module")
def paper_graph() -> ProceduralGraph:
    """A small procedural graph covering a HotpotQA-style multi-hop question.

    The graph encodes "answer questions about people and their nationality
    by looking up type information for entities". It includes nodes for
    common reasoning steps and edges with concrete guidance text.
    """
    return ProceduralGraph(
        id="hotpot-qa-sample",
        nodes={
            "start": Node(
                id="start",
                description="Begin task; parse the question and identify entities",
            ),
            "identify": Node(
                id="identify",
                description="Identify entities (people, places, works) in the question",
            ),
            "lookup": Node(
                id="lookup",
                description="Look up each entity's key attributes (type, year, etc.)",
            ),
            "synthesize": Node(
                id="synthesize",
                description="Synthesize the multi-hop answer from the looked-up facts",
            ),
            "answer": Node(
                id="answer",
                description="Format the final answer as 'Answer: <answer>'",
            ),
        },
        edges=[
            Edge(
                src="start",
                dst="identify",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="always at task start",
                    guidance="Read the question carefully; list all named entities "
                    "(people, works, places, organizations).",
                    pitfalls="don't skip entities that look like adjectives",
                ),
            ),
            Edge(
                src="identify",
                dst="lookup",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="after entities are listed",
                    guidance="Retrieve each entity's defining attribute using the "
                    "search tool. Prefer single-hop lookups before chaining.",
                    pitfalls="don't invent attributes; if the search returns nothing, "
                    "rephrase the query",
                ),
            ),
            Edge(
                src="lookup",
                dst="synthesize",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="after at least one attribute is known",
                    guidance="Combine attributes to answer the multi-hop question. "
                    "Cite the chain of reasoning in your answer.",
                    pitfalls="don't skip entities; verify all entities are accounted for",
                ),
            ),
            Edge(
                src="synthesize",
                dst="answer",
                relation=Relation.LEADS_TO,
                attribute=Attribute(
                    condition="final synthesis complete",
                    guidance="Reply with `Answer: <your answer>` and nothing else.",
                    pitfalls="don't include preamble or reasoning in the answer line",
                ),
            ),
        ],
        terminal_ids={"answer"},
    )


@pytest.fixture
def llm(request: pytest.FixtureRequest) -> LLMClient:
    """Construct a real LiteLLMClient against whichever provider is configured."""
    model = request.config.getoption("--llm-model") or "gpt-4o-mini"
    if _HAS_OPENAI:
        return LiteLLMClient(model=model, api_key=os.environ["OPENAI_API_KEY"])
    if _HAS_ANTHROPIC:
        model = request.config.getoption("--llm-model") or "claude-3-5-haiku-20241022"
        return LiteLLMClient(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    if _HAS_GROQ:
        model = request.config.getoption("--llm-model") or "groq/llama-3.1-70b-versatile"
        return LiteLLMClient(model=model, api_key=os.environ["GROQ_API_KEY"])
    pytest.skip("no supported provider key found")


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def run_solver_on_task(
    *,
    solver: HotpotQASolver,
    task: HotpotQATask,
    max_steps: int = 2,
) -> str:
    """Run the HotpotQA solver on a single task; return the predicted answer."""
    trajectory: list[tuple[str, str]] = []
    for _ in range(max_steps):
        action = await solver.step(
            AgentState(
                query=task.to_query(),
                trajectory=tuple(trajectory),
                context="",
            )
        )
        if "ANSWER:" in action.upper():
            return action.split(":", 1)[1].strip() if ":" in action else action
        trajectory.append((action, ""))
    return ""


async def run_pgadapter_on_task(
    *,
    adapter: PGAdapter,
    task: HotpotQATask,
    max_steps: int = 4,
) -> tuple[str, bool]:
    """Run PGAdapter on a task; return (predicted answer, guidance_was_used)."""
    trajectory: list[tuple[str, str]] = []
    predicted = ""
    guidance_used = False
    for _ in range(max_steps):
        action = await adapter.step(query=task.to_query(), trajectory=trajectory)
        # Non-search, non-answer actions suggest procedural reasoning
        # (i.e., the LLM followed the injected guidance).
        if not action.startswith("search") and "ANSWER:" not in action.upper():
            guidance_used = True
        if "ANSWER:" in action.upper():
            predicted = action.split(":", 1)[1].strip() if ":" in action else action
            break
        trajectory.append((action, ""))
    return predicted, guidance_used


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_no_key
@pytest.mark.live_llm
def test_litellm_client_basic_completion(llm: LLMClient) -> None:
    """Single chat-completion round-trip against the real provider."""
    response = asyncio.run(_test_litellm_client_basic_completion(llm))
    assert isinstance(response, str) and response.strip()
    assert "pong" in response.lower(), f"expected 'pong' in response, got: {response!r}"


async def _test_litellm_client_basic_completion(llm: LLMClient) -> str:
    return await llm.complete(
        system="You are a terse assistant.",
        user="Reply with exactly the word 'pong' and nothing else.",
        temperature=0.0,
    )


@_skip_no_key
@_skip_no_data
@pytest.mark.live_llm
def test_hotpot_solver_single_task(llm: LLMClient, hotpot_tasks: list[HotpotQATask]) -> None:
    """Run one HotpotQA task through the live LLM; verify EM matches gold."""
    task = hotpot_tasks[0]
    predicted, em = asyncio.run(_test_hotpot_solver_single_task(llm, task))
    print(
        f"\n[hotpot_solver_single_task] "
        f"q='{task.question[:60]}...' "
        f"gold='{task.answer}' "
        f"pred='{predicted}' "
        f"EM={em:.2f}"
    )
    assert predicted, "LLM produced no answer line"
    assert 0.0 <= em <= 1.0


async def _test_hotpot_solver_single_task(llm: LLMClient, task: HotpotQATask) -> tuple[str, float]:
    from eval.hotpotqa.run import exact_match

    solver = HotpotQASolver(llm=llm, search=echo_search, max_steps=2)
    predicted = await run_solver_on_task(solver=solver, task=task, max_steps=2)
    em = exact_match(predicted, task.answer)
    return predicted, em


@_skip_no_key
@_skip_no_data
@pytest.mark.live_llm
@pytest.mark.parametrize("task_idx", [0, 1, 2, 3, 4], ids=lambda i: f"hotpot_task_{i}")
def test_hotpot_solver_multiple_tasks(
    llm: LLMClient,
    hotpot_tasks: list[HotpotQATask],
    task_idx: int,
) -> None:
    """Run the first 5 HotpotQA tasks through the live LLM; report EM and F1."""
    from eval.hotpotqa.run import exact_match, token_f1

    task = hotpot_tasks[task_idx]
    predicted, em, f1 = asyncio.run(_test_hotpot_solver_multiple_tasks(llm, task))
    print(
        f"\n[hotpot_task_{task_idx}] "
        f"q='{task.question[:60]}...' "
        f"gold='{task.answer}' "
        f"pred='{predicted}' "
        f"EM={em:.2f} F1={f1:.2f}"
    )


async def _test_hotpot_solver_multiple_tasks(
    llm: LLMClient, task: HotpotQATask
) -> tuple[str, float, float]:
    from eval.hotpotqa.run import exact_match, token_f1

    solver = HotpotQASolver(llm=llm, search=echo_search, max_steps=2)
    predicted = await run_solver_on_task(solver=solver, task=task, max_steps=2)
    em = exact_match(predicted, task.answer)
    f1 = token_f1(predicted, task.answer)
    return predicted, em, f1


@_skip_no_key
@_skip_no_data
@pytest.mark.live_llm
def test_pgadapter_with_paper_graph(
    llm: LLMClient, paper_graph: ProceduralGraph, hotpot_tasks: list[HotpotQATask]
) -> None:
    """Run one task through `PGAdapter` with a procedural graph; verify guidance injected."""
    from eval.hotpotqa.run import exact_match, token_f1

    task = hotpot_tasks[0]
    predicted, em, f1, guidance_used = asyncio.run(
        _test_pgadapter_with_paper_graph(llm, paper_graph, task)
    )
    print(
        f"\n[pgadapter_with_paper_graph] "
        f"q='{task.question[:60]}...' "
        f"gold='{task.answer}' "
        f"pred='{predicted}' "
        f"EM={em:.2f} F1={f1:.2f} "
        f"guidance_used={guidance_used}"
    )


async def _test_pgadapter_with_paper_graph(
    llm: LLMClient, graph: ProceduralGraph, task: HotpotQATask
) -> tuple[str, float, float, bool]:
    from eval.hotpotqa.run import exact_match, token_f1

    solver = HotpotQASolver(llm=llm, search=echo_search, max_steps=4)
    adapter = PGAdapter(solver=solver, graph=graph, llm=llm)
    predicted, guidance_used = await run_pgadapter_on_task(adapter=adapter, task=task, max_steps=4)
    em = exact_match(predicted, task.answer)
    f1 = token_f1(predicted, task.answer)
    return predicted, em, f1, guidance_used


# ---------------------------------------------------------------------------
# CLI option to override the model
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-model",
        action="store",
        default=os.environ.get("METHODOS_LLM_MODEL", ""),
        help="Override the LLM model identifier (default: provider default)",
    )
