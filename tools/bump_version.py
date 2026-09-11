"""Bump the package version across the project.

Updates:
- `pyproject.toml` (`[project] version = ...`)
- `methodos/__init__.py` (`__version__ = "..."`)
- `docker/Dockerfile` (no-op; `VERSION` is supplied at build time)

Usage:

    uv run python tools/bump_version.py 0.1.1
    uv run python tools/bump_version.py --current 0.1.0 --bump patch
    uv run python tools/bump_version.py --check   # verify all sources agree

Exits non-zero on any mismatch. Designed to run on a clean working tree
(no uncommitted changes to the version files).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "methodos" / "__init__.py"


def read_pyproject_version() -> str:
    text = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit('pyproject.toml: no `version = "..."` line found')
    return match.group(1)


def read_init_version() -> str:
    text = INIT_PY.read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit('methodos/__init__.py: no `__version__ = "..."` line found')
    return match.group(1)


def write_pyproject_version(version: str) -> None:
    text = PYPROJECT.read_text()
    new = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT.write_text(new)


def write_init_version(version: str) -> None:
    text = INIT_PY.read_text()
    new = re.sub(
        r"^(__version__\s*=\s*)\"[^\"]+\"",
        rf'\1"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    INIT_PY.write_text(new)


def bump(current: str, kind: str) -> str:
    parts = current.split(".")
    if len(parts) != 3:
        raise SystemExit(f"version {current!r} is not in MAJOR.MINOR.PATCH form")
    major, minor, patch = (int(p) for p in parts)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"unknown bump kind: {kind!r}; expected major|minor|patch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump the methodos package version.")
    parser.add_argument("explicit", nargs="?", help="New version, e.g. 0.1.1")
    parser.add_argument("--current", help="Current version (default: read from sources)")
    parser.add_argument(
        "--bump",
        choices=("major", "minor", "patch"),
        help="Bump kind (requires --current)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all sources agree and exit (no writes).",
    )
    args = parser.parse_args(argv)

    pyproject_v = read_pyproject_version()
    init_v = read_init_version()
    if pyproject_v != init_v:
        raise SystemExit(
            f"version mismatch: pyproject.toml={pyproject_v!r}, methodos/__init__.py={init_v!r}"
        )
    current = pyproject_v

    if args.check:
        print(f"versions agree at {current!r}")
        return 0

    if args.bump:
        if not args.current:
            args.current = current
        new_version = bump(args.current, args.bump)
    elif args.explicit:
        new_version = args.explicit
    else:
        parser.error("provide an explicit version or --bump <kind>")

    # Validate new_version shape
    if not re.fullmatch(r"\d+\.\d+\.\d+", new_version):
        raise SystemExit(f"new version {new_version!r} is not MAJOR.MINOR.PATCH")

    write_pyproject_version(new_version)
    write_init_version(new_version)
    print(f"bumped {current!r} -> {new_version!r} (pyproject.toml, methodos/__init__.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
