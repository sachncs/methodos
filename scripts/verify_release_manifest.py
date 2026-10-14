"""Validate a Methodos release provenance manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_HEX_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_PATTERN = re.compile(r"^ghcr\.io/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@sha256:[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class ReleaseManifestError(ValueError):
    """Raised when a release manifest is missing or malformed."""


def load_release_manifest(path: Path) -> dict[str, str]:
    """Load and validate a release manifest from ``path``."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"cannot read valid JSON from {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ReleaseManifestError("manifest must be a JSON object")
    expected_keys = {"package_version", "source_revision", "image"}
    if set(raw) != expected_keys:
        raise ReleaseManifestError(
            f"manifest keys must be exactly {sorted(expected_keys)}, got {sorted(raw)}"
        )
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise ReleaseManifestError("manifest values must be non-empty strings")

    manifest = {key: raw[key] for key in expected_keys}
    if not _VERSION_PATTERN.fullmatch(manifest["package_version"]):
        raise ReleaseManifestError("package_version must be a semantic version")
    if not _HEX_SHA_PATTERN.fullmatch(manifest["source_revision"]):
        raise ReleaseManifestError("source_revision must be a 40-character lowercase commit SHA")
    if not _IMAGE_PATTERN.fullmatch(manifest["image"]):
        raise ReleaseManifestError("image must be a GHCR image pinned to a sha256 digest")
    return manifest


def verify_release_manifest(
    path: Path,
    *,
    expected_package_version: str | None = None,
    expected_source_revision: str | None = None,
    expected_image: str | None = None,
) -> dict[str, str]:
    """Validate a manifest and optional release identity expectations."""
    manifest = load_release_manifest(path)
    expectations = {
        "package_version": expected_package_version,
        "source_revision": expected_source_revision,
        "image": expected_image,
    }
    for field, expected in expectations.items():
        if expected is not None and manifest[field] != expected:
            raise ReleaseManifestError(
                f"{field} mismatch: expected {expected!r}, got {manifest[field]!r}"
            )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--package-version")
    parser.add_argument("--source-revision")
    parser.add_argument("--image")
    return parser


def main() -> int:
    """Run the release manifest verifier."""
    args = _build_parser().parse_args()
    try:
        manifest = verify_release_manifest(
            args.manifest,
            expected_package_version=args.package_version,
            expected_source_revision=args.source_revision,
            expected_image=args.image,
        )
    except ReleaseManifestError as exc:
        _build_parser().error(str(exc))
    print(
        f"verified {manifest['package_version']} "
        f"({manifest['source_revision']}) -> {manifest['image']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
