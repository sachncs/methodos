"""Tests for `methodos.guidance`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from methodos.guidance import (
    GUIDANCE_SYSTEM_PROMPT,
    format_graph_for_prompt,
    format_trajectory_window,
    generate_guidance,
)
from methodos.llm import LLMClient
from methodos.schema import (
    Attribute,
    Edge,
    Node,
    ProceduralGraph,
    Relation,
)


def make_graph() -> ProceduralGraph:
    return ProceduralGraph(
        id="g1",
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
                    pitfalls="don't search with full question",
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


class FakeLLM(LLMClient):
    """Records calls and returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "json_schema": json_schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            return ""
        return self.responses.pop(0)


class TestFormatGraphForPrompt:
    """`format_graph_for_prompt` produces a stable, parseable serialization."""

    def test_includes_header(self) -> None:
        text = format_graph_for_prompt(make_graph())
        assert "# Subgraph: g1" in text
        assert "# Nodes (3):" in text
        assert "# Edges (2):" in text

    def test_includes_node_descriptions(self) -> None:
        text = format_graph_for_prompt(make_graph())
        assert "begin task" in text
        assert "search Wikipedia" in text

    def test_includes_node_without_description(self) -> None:
        graph = ProceduralGraph(
            id="g",
            nodes={"a": Node(id="a")},
            terminal_ids={"a"},
        )
        text = format_graph_for_prompt(graph)
        assert "(no description)" in text

    def test_includes_edge_attributes(self) -> None:
        text = format_graph_for_prompt(make_graph())
        # Each edge's three attributes appear.
        assert "need information" in text
        assert "call search with focused query" in text
        assert "don't search with full question" in text

    def test_includes_relation_token(self) -> None:
        text = format_graph_for_prompt(make_graph())
        assert "--leads_to-->" in text

    def test_empty_graph_renders(self) -> None:
        text = format_graph_for_prompt(ProceduralGraph(id="empty"))
        assert "# Subgraph: empty" in text
        assert "# Nodes (0):" in text
        assert "# Edges (0):" in text


class TestFormatTrajectoryWindow:
    """`format_trajectory_window` formats recent action/observation pairs."""

    def test_empty_window_returns_empty_marker(self) -> None:
        assert format_trajectory_window([], window=3) == "(no prior actions)"

    def test_window_zero_returns_empty_marker(self) -> None:
        out = format_trajectory_window([("a", "b"), ("c", "d")], window=0)
        assert out == "(no prior actions)"

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be non-negative"):
            format_trajectory_window([("a", "b")], window=-1)

    def test_includes_action_and_observation(self) -> None:
        out = format_trajectory_window([("search", "found X")], window=1)
        assert "ACTION: search" in out
        assert "OBS:    found X" in out

    def test_truncates_to_window(self) -> None:
        trajectory = [(f"a{i}", f"obs{i}") for i in range(10)]
        out = format_trajectory_window(trajectory, window=3)
        # Only the last 3 steps should appear.
        assert "ACTION: a7" in out
        assert "ACTION: a8" in out
        assert "ACTION: a9" in out
        assert "ACTION: a6" not in out

    def test_iterable_input(self) -> None:
        """Trajectory may be any iterable, not just a list."""

        def gen() -> Any:
            yield ("a", "b")
            yield ("c", "d")

        out = format_trajectory_window(gen(), window=5)
        assert "ACTION: a" in out
        assert "ACTION: c" in out


class TestGenerateGuidance:
    """`generate_guidance` end-to-end with a FakeLLM."""

    async def test_returns_llm_response(self) -> None:
        fake = FakeLLM(["guidance-text"])
        graph = make_graph()
        out = await generate_guidance(
            llm=fake,
            graph=graph,
            query="What is the capital of France?",
            trajectory=[],
            window=3,
        )
        assert out == "guidance-text"

    async def test_calls_llm_with_system_prompt(self) -> None:
        fake = FakeLLM(["x"])
        await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="q",
            trajectory=[],
        )
        assert fake.calls[0]["system"] == GUIDANCE_SYSTEM_PROMPT

    async def test_user_prompt_includes_query(self) -> None:
        fake = FakeLLM(["x"])
        await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="Tell me about Paris.",
            trajectory=[],
        )
        assert "Tell me about Paris." in fake.calls[0]["user"]

    async def test_user_prompt_includes_trajectory(self) -> None:
        fake = FakeLLM(["x"])
        await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="q",
            trajectory=[("search", "found Paris is the capital")],
        )
        assert "ACTION: search" in fake.calls[0]["user"]
        assert "OBS:    found Paris is the capital" in fake.calls[0]["user"]

    async def test_user_prompt_includes_graph(self) -> None:
        fake = FakeLLM(["x"])
        graph = make_graph()
        await generate_guidance(
            llm=fake,
            graph=graph,
            query="q",
            trajectory=[],
        )
        user_prompt = fake.calls[0]["user"]
        # Graph metadata appears
        assert "# Subgraph: g1" in user_prompt
        # Nodes and edges appear
        assert "search Wikipedia" in user_prompt
        assert "leads_to" in user_prompt

    async def test_temperature_is_zero(self) -> None:
        fake = FakeLLM(["x"])
        await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="q",
            trajectory=[],
        )
        assert fake.calls[0]["temperature"] == 0.0

    async def test_default_window_is_three(self) -> None:
        fake = FakeLLM(["x"])
        graph = make_graph()
        # 5-step trajectory
        trajectory = [(f"a{i}", f"obs{i}") for i in range(5)]
        await generate_guidance(
            llm=fake,
            graph=graph,
            query="q",
            trajectory=trajectory,
        )
        # Last 3 steps appear
        user_prompt = fake.calls[0]["user"]
        assert "ACTION: a2" in user_prompt
        assert "ACTION: a3" in user_prompt
        assert "ACTION: a4" in user_prompt
        assert "ACTION: a1" not in user_prompt

    async def test_explicit_window_overrides_default(self) -> None:
        fake = FakeLLM(["x"])
        graph = make_graph()
        trajectory = [(f"a{i}", f"obs{i}") for i in range(5)]
        await generate_guidance(
            llm=fake,
            graph=graph,
            query="q",
            trajectory=trajectory,
            window=2,
        )
        user_prompt = fake.calls[0]["user"]
        assert "ACTION: a3" in user_prompt
        assert "ACTION: a4" in user_prompt
        assert "ACTION: a2" not in user_prompt

    async def test_returns_string_even_on_empty_response(self) -> None:
        fake = FakeLLM([""])
        out = await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="q",
            trajectory=[],
        )
        assert out == ""

    async def test_no_json_schema_passed(self) -> None:
        """generate_guidance doesn't pass a structured-output schema."""
        fake = FakeLLM(["x"])
        await generate_guidance(
            llm=fake,
            graph=make_graph(),
            query="q",
            trajectory=[],
        )
        assert fake.calls[0]["json_schema"] is None


def test_guidance_system_prompt_is_non_empty() -> None:
    """The system prompt is part of the wire contract — assert it's substantive."""
    assert len(GUIDANCE_SYSTEM_PROMPT) > 200
    # It must instruct the model about its role.
    assert "procedural subgraph" in GUIDANCE_SYSTEM_PROMPT.lower()
    assert "trajectory" in GUIDANCE_SYSTEM_PROMPT.lower()
    # It must forbid certain failure modes.
    assert "never" in GUIDANCE_SYSTEM_PROMPT.lower()
