"""Tests for `tools/bump_version.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "methodos" / "__init__.py"


@pytest.fixture
def bump_module():
    """Load `tools/bump_version.py` as a module without polluting sys.path permanently."""
    spec = importlib.util.spec_from_file_location("bump_version", TOOLS_DIR / "bump_version.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def restore_versions():
    """Snapshot version strings before the test, restore on teardown."""
    pyproject_before = PYPROJECT.read_text()
    init_before = INIT_PY.read_text()
    yield
    PYPROJECT.write_text(pyproject_before)
    INIT_PY.write_text(init_before)


class TestBump:
    def test_bump_patch(self, bump_module) -> None:
        assert bump_module.bump("0.1.0", "patch") == "0.1.1"
        assert bump_module.bump("1.2.3", "patch") == "1.2.4"
        assert bump_module.bump("0.0.9", "patch") == "0.0.10"

    def test_bump_minor(self, bump_module) -> None:
        assert bump_module.bump("0.1.0", "minor") == "0.2.0"
        assert bump_module.bump("1.99.99", "minor") == "1.100.0"

    def test_bump_major(self, bump_module) -> None:
        assert bump_module.bump("0.1.0", "major") == "1.0.0"
        assert bump_module.bump("9.9.9", "major") == "10.0.0"

    def test_bump_invalid_version(self, bump_module) -> None:
        with pytest.raises(SystemExit):
            bump_module.bump("0.1", "patch")
        with pytest.raises(SystemExit):
            bump_module.bump("0.1.0.0", "minor")

    def test_bump_unknown_kind(self, bump_module) -> None:
        with pytest.raises(SystemExit):
            bump_module.bump("0.1.0", "feature")  # type: ignore[arg-type]


class TestReadWrite:
    def test_read_versions_agree(self, bump_module) -> None:
        pyproject_v = bump_module.read_pyproject_version()
        init_v = bump_module.read_init_version()
        assert pyproject_v == init_v

    def test_round_trip(self, bump_module, restore_versions) -> None:
        bump_module.write_pyproject_version("9.9.9")
        bump_module.write_init_version("9.9.9")
        assert bump_module.read_pyproject_version() == "9.9.9"
        assert bump_module.read_init_version() == "9.9.9"
        # restore_versions fixture will put them back


class TestMain:
    def test_check_passes_when_versions_agree(self, bump_module, capsys) -> None:
        code = bump_module.main(["--check"])
        assert code == 0
        captured = capsys.readouterr()
        assert "versions agree" in captured.out

    def test_check_fails_on_mismatch(self, bump_module, restore_versions) -> None:
        bump_module.write_init_version("9.9.9")
        with pytest.raises(SystemExit) as exc:
            bump_module.main(["--check"])
        assert "mismatch" in str(exc.value)

    def test_explicit_version(self, bump_module, restore_versions) -> None:
        code = bump_module.main(["1.2.3"])
        assert code == 0
        assert bump_module.read_pyproject_version() == "1.2.3"
        assert bump_module.read_init_version() == "1.2.3"

    def test_bump_kind(self, bump_module, restore_versions) -> None:
        original = bump_module.read_pyproject_version()
        code = bump_module.main(["--current", original, "--bump", "patch"])
        assert code == 0
        new = bump_module.read_pyproject_version()
        assert new != original

    def test_explicit_invalid(self, bump_module) -> None:
        with pytest.raises(SystemExit):
            bump_module.main(["not-a-version"])

    def test_no_args_errors(self, bump_module) -> None:
        with pytest.raises(SystemExit):
            bump_module.main([])
