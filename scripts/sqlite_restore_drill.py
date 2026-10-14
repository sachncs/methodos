"""Verify that a configured SQLite database can be backed up and reopened."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from methodos.repo import SQLiteRepository


async def run_drill(source: Path, graph_id: str | None) -> None:
    """Create, verify, and reopen a temporary backup of ``source``."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SQLite source database not found: {source}")

    repository = SQLiteRepository(db_path=source)
    with tempfile.TemporaryDirectory(prefix="methodos-restore-drill-") as directory:
        backup = Path(directory) / "methodos-restore.db"
        await repository.backup_to(backup)
        await repository.verify_backup(backup)
        restored = SQLiteRepository(db_path=backup)
        if graph_id is not None:
            await restored.load_graph(graph_id)
    print(f"restore drill passed: verified and reopened {source}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SQLite database to verify and reopen.")
    parser.add_argument(
        "--graph-id",
        help="Optional graph id that must be readable from the reopened backup.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the restore drill and return a shell-friendly status."""
    args = parse_args()
    asyncio.run(run_drill(args.source, args.graph_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
