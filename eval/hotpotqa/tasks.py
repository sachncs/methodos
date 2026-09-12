"""HotpotQA task loading and download."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

DEFAULT_DATA_DIR: Path = Path.home() / ".methodos" / "eval_data"


@dataclass(frozen=True, slots=True)
class HotpotQATask:
    """One HotpotQA multi-hop question."""

    question: str
    answer: str
    supporting_facts: tuple[str, ...]
    id: str

    def to_query(self) -> str:
        """Return the question text for the agent."""
        return self.question


def download_if_missing(*, target_dir: Path = DEFAULT_DATA_DIR, limit: int = 500) -> Path:
    """Download HotpotQA distractor split if not present. Returns path to JSONL."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "hotpotqa_dev_distractor.jsonl"
    if out_path.exists():
        return out_path

    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= limit:
                break
            record = {
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "supporting_facts": [s[0] for s in row["supporting_facts"]],
            }
            f.write(json.dumps(record) + "\n")
    return out_path


def load_tasks(*, path: Path, limit: int | None = None) -> Iterator[HotpotQATask]:
    """Yield HotpotQA tasks from a JSONL file produced by `download_if_missing`."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            yield HotpotQATask(
                question=record["question"],
                answer=record["answer"],
                supporting_facts=tuple(record.get("supporting_facts", [])),
                id=record["_id"],
            )
            count += 1
            if limit is not None and count >= limit:
                break
