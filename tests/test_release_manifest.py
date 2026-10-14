"""Tests for release provenance verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_release_manifest import ReleaseManifestError, verify_release_manifest

_MANIFEST = {
    "package_version": "0.1.0",
    "source_revision": "0123456789abcdef0123456789abcdef01234567",
    "image": "ghcr.io/sachncs/methodos@sha256:" + "a" * 64,
}


def _write_manifest(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_verifies_manifest_and_release_identity(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _MANIFEST)

    assert (
        verify_release_manifest(
            path,
            expected_package_version="0.1.0",
            expected_source_revision=_MANIFEST["source_revision"],
            expected_image=_MANIFEST["image"],
        )
        == _MANIFEST
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_version", "1"),
        ("source_revision", "not-a-sha"),
        ("image", "ghcr.io/sachncs/methodos:latest"),
    ],
)
def test_rejects_malformed_manifest_fields(tmp_path: Path, field: str, value: str) -> None:
    manifest = {**_MANIFEST, field: value}

    with pytest.raises(ReleaseManifestError):
        verify_release_manifest(_write_manifest(tmp_path, manifest))


def test_rejects_unexpected_keys(tmp_path: Path) -> None:
    with pytest.raises(ReleaseManifestError, match="exactly"):
        verify_release_manifest(_write_manifest(tmp_path, {**_MANIFEST, "extra": "value"}))


def test_rejects_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ReleaseManifestError, match="source_revision mismatch"):
        verify_release_manifest(
            _write_manifest(tmp_path, _MANIFEST), expected_source_revision="f" * 40
        )
