"""Run a bounded concurrent HTTP smoke/load check against a Methodos service."""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Sequence

import httpx


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a nearest-rank percentile from non-empty latency samples."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


async def run_check(
    *,
    url: str,
    requests: int,
    concurrency: int,
    timeout: float,
) -> int:
    """Issue requests and return a shell-friendly exit status."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures: list[str] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def request_once() -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.get(url)
                    elapsed = time.perf_counter() - started
                    latencies.append(elapsed)
                    if response.status_code != 200:
                        failures.append(f"HTTP {response.status_code}: {response.text[:120]}")
                except httpx.HTTPError as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")

        await asyncio.gather(*(request_once() for _ in range(requests)))

    succeeded = requests - len(failures)
    p95 = percentile(latencies, 0.95) if latencies else float("nan")
    print(
        f"requests={requests} succeeded={succeeded} failed={len(failures)} "
        f"p95_seconds={p95:.4f}"
    )
    for failure in failures[:5]:
        print(f"failure: {failure}", file=sys.stderr)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("requests and concurrency must be positive; timeout must be positive")
    return args


def main() -> int:
    """Run the load smoke check."""
    args = parse_args()
    return asyncio.run(
        run_check(
            url=args.url,
            requests=args.requests,
            concurrency=min(args.concurrency, args.requests),
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
