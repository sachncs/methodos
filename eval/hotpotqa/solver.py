"""HotpotQA ReAct-style solver.

Minimal implementation: calls an LLM to produce a thought/action/answer
sequence. The host wires the actual search backend via the `search`
callback. methodos guidance is injected through `state.context`.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from methodos.adapter import AgentState
from methodos.llm import LLMClient

logger = logging.getLogger(__name__)


REACT_PROMPT: str = """\
You are an agent answering multi-hop questions. Use the search tool to find \
information. End your response with "Answer: <answer>" when you have the final \
answer.

{context}

Question: {query}
"""


class HotpotQASolver:
    """ReAct-style solver that consults an LLM and delegates search to a callback.

    Args:
        llm: any `LLMClient` (e.g. `LiteLLMClient`).
        search: async callable taking a query string and returning a snippet.
        max_steps: maximum ReAct iterations per question.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        search: Callable[[str], Awaitable[str]],
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.search = search
        self.max_steps = max_steps

    async def step(self, state: AgentState) -> str:
        """Produce one ReAct step as a string action.

        The action is one of: a search query (then the caller is expected
        to provide an observation in the next iteration), or an ANSWER line.
        """
        prompt = REACT_PROMPT.format(context=state.context, query=state.query)
        response = await self.llm.complete(system="", user=prompt, temperature=0.0)
        logger.debug("hotpotqa solver response: %s", response[:200])
        return response.strip()


async def echo_search(query: str) -> str:
    """Default test-only search that echoes the query back.

    Production deployments replace this with a real backend (e.g.,
    Wikipedia API, BM25 index). The default returns the query as a
    placeholder so the eval harness can run without external services.
    """
    return f"(no results for: {query})"
