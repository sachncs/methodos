"""Sanity test for the package skeleton; replaced as phases land."""

from __future__ import annotations

import methodos


def test_version_is_string() -> None:
    assert isinstance(methodos.__version__, str)
    assert methodos.__version__ == "0.1.0"


def test_package_imports() -> None:
    assert methodos is not None
    for name in (
        "AgentState",
        "GuidanceCache",
        "LLMClient",
        "LiteLLMClient",
        "PGAdapter",
        "Solver",
    ):
        assert name in methodos.__all__
        assert hasattr(methodos, name)


def test_readme_primary_api_symbols_are_root_exports() -> None:
    """Keep the documented quick-start import path executable."""
    namespace: dict[str, object] = {}
    exec(
        "from methodos import AgentState, LiteLLMClient, PGAdapter, Solver",
        namespace,
    )
    assert all(name in namespace for name in ("AgentState", "LiteLLMClient", "PGAdapter", "Solver"))
