"""Sanity test for the package skeleton; replaced as phases land."""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

import methodos

PUBLIC_MODULES = (
    "methodos.adapter",
    "methodos.evolution",
    "methodos.graph",
    "methodos.guidance",
    "methodos.jobs",
    "methodos.llm",
    "methodos.migrations",
    "methodos.repo",
    "methodos.schema",
    "methodos.service",
)


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


def test_console_script_targets_typer_application() -> None:
    """The installed executable must invoke Typer, not its callback function."""
    project = Path(__file__).parents[1] / "pyproject.toml"
    with project.open("rb") as stream:
        config = tomllib.load(stream)
    assert config["project"]["scripts"]["methodos"] == "methodos.cli:cli"


def test_declared_public_exports_resolve() -> None:
    """Prevent a documented public symbol from becoming an import-time ghost."""
    for module_name in PUBLIC_MODULES:
        module = import_module(module_name)
        for symbol in module.__all__:
            assert hasattr(module, symbol), f"{module_name}.{symbol} is not defined"
