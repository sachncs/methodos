"""Tests for `eval.hotpotqa` helpers (EM, F1, task loading)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.hotpotqa.run import _exact_match, _normalize, _token_f1
from eval.hotpotqa.tasks import (
    DEFAULT_DATA_DIR,
    HotpotQATask,
    load_tasks,
)


class TestNormalize:
    def test_lowercase_and_strip_whitespace(self) -> None:
        assert _normalize("  Hello   World  ") == "hello world"

    def test_handles_empty_string(self) -> None:
        assert _normalize("") == ""

    def test_collapses_multiple_spaces(self) -> None:
        assert _normalize("a\n\t b") == "a b"


class TestExactMatch:
    def test_identical_strings_match(self) -> None:
        assert _exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self) -> None:
        assert _exact_match("PARIS", "paris") == 1.0

    def test_whitespace_tolerant(self) -> None:
        assert _exact_match("  Paris  ", "Paris") == 1.0

    def test_different_strings_no_match(self) -> None:
        assert _exact_match("Paris", "London") == 0.0


class TestTokenF1:
    def test_identical_strings(self) -> None:
        assert _token_f1("Paris", "Paris") == 1.0

    def test_no_overlap(self) -> None:
        assert _token_f1("apple", "banana") == 0.0

    def test_partial_overlap(self) -> None:
        # "Paris France" vs "Paris Germany" → tokens "paris" overlaps
        f1 = _token_f1("Paris France", "Paris Germany")
        assert 0.0 < f1 < 1.0

    def test_empty_predicted(self) -> None:
        assert _token_f1("", "Paris") == 0.0

    def test_empty_gold(self) -> None:
        assert _token_f1("Paris", "") == 0.0


class TestLoadTasks:
    def test_loads_from_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        records = [
            {"_id": "1", "question": "Q1", "answer": "A1", "supporting_facts": ["fact1"]},
            {"_id": "2", "question": "Q2", "answer": "A2", "supporting_facts": []},
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))

        tasks = list(load_tasks(path=path))
        assert len(tasks) == 2
        assert isinstance(tasks[0], HotpotQATask)
        assert tasks[0].id == "1"
        assert tasks[0].question == "Q1"
        assert tasks[0].answer == "A1"
        assert tasks[0].supporting_facts == ("fact1",)
        assert tasks[1].supporting_facts == ()

    def test_respects_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        records = [
            {"_id": str(i), "question": f"Q{i}", "answer": f"A{i}", "supporting_facts": []}
            for i in range(10)
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))

        tasks = list(load_tasks(path=path, limit=3))
        assert len(tasks) == 3
        assert tasks[0].id == "0"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            json.dumps({"_id": "1", "question": "Q", "answer": "A", "supporting_facts": []})
            + "\n\n\n"
            + json.dumps({"_id": "2", "question": "Q", "answer": "A", "supporting_facts": []})
            + "\n"
        )
        tasks = list(load_tasks(path=path))
        assert len(tasks) == 2

    def test_default_data_dir_constant(self) -> None:
        assert Path.home() / ".methodos" / "eval_data" == DEFAULT_DATA_DIR


class TestHotpotQATaskToQuery:
    def test_returns_question(self) -> None:
        task = HotpotQATask(
            question="What is the capital of France?",
            answer="Paris",
            supporting_facts=(),
            id="1",
        )
        assert task.to_query() == "What is the capital of France?"


class TestHotpotQATaskIsFrozen:
    def test_dataclass_is_frozen(self) -> None:
        task = HotpotQATask(question="Q", answer="A", supporting_facts=(), id="1")
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            task.question = "new"  # type: ignore[misc]


class TestDownloadIfMissing:
    def test_returns_path_when_already_exists(  # type: ignore[no-untyped-def]
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """If JSONL is already at target_dir, return without calling datasets."""
        out = tmp_path / "hotpotqa_dev_distractor.jsonl"
        out.write_text('{"_id": "1", "question": "Q", "answer": "A", "supporting_facts": []}\n')

        # Verify datasets package is not even imported.
        import builtins
        original_import = builtins.__import__
        called: list[str] = []

        def fake_import(
            name: str,
            globals: object | None = None,
            locals: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            called.append(name)
            return original_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from eval.hotpotqa.tasks import download_if_missing
        path = download_if_missing(target_dir=tmp_path)
        assert path == out
        assert "datasets" not in called

    def test_raises_when_datasets_missing(  # type: ignore[no-untyped-def]
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """If JSONL is absent and datasets is not installed, raise."""
        # Block the datasets import.
        import builtins
        original_import = builtins.__import__

        def fake_import(
            name: str,
            globals: object | None = None,
            locals: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "datasets":
                raise ImportError("datasets not installed")
            return original_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from eval.hotpotqa.tasks import download_if_missing
        with pytest.raises(RuntimeError, match="datasets package required"):
            download_if_missing(target_dir=tmp_path)
