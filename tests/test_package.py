"""Sanity test for the package skeleton; replaced as phases land."""

from __future__ import annotations

import methodos


def test_version_is_string() -> None:
    assert isinstance(methodos.__version__, str)
    assert methodos.__version__ == "0.1.0"


def test_package_imports() -> None:
    # The package imports without raising even though modules are not yet
    # implemented (they will land in subsequent phases).
    assert methodos is not None
