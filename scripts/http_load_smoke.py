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
    rounds: int = 1,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Issue bounded repeated request rounds and return a shell exit status."""
    if requests < 1 or concurrency < 1 or timeout <= 0 or rounds < 1:
        raise ValueError(
            "requests, concurrency, and rounds must be positive; timeout must be positive"
        )
    total_requests = requests * rounds
    if total_requests > 100_000:
        raise ValueError("total requests must not exceed 100000")
    concurrency = min(concurrency, requests)
    started_all = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures: list[str] = []

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:

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
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")

        for _ in range(rounds):
            await asyncio.gather(*(request_once() for _ in range(requests)))

    succeeded = total_requests - len(failures)
    elapsed_all = time.perf_counter() - started_all
    p95 = percentile(latencies, 0.95) if latencies else float("nan")
    p99 = percentile(latencies, 0.99) if latencies else float("nan")
    throughput = succeeded / elapsed_all if elapsed_all > 0 else 0.0
    print(
        f"requests={total_requests} rounds={rounds} succeeded={succeeded} failed={len(failures)} "
        f"throughput_per_second={throughput:.2f} p95_seconds={p95:.4f} p99_seconds={p99:.4f}"
    )
    for failure in failures[:5]:
        print(f"failure: {failure}", file=sys.stderr)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.rounds < 1 or args.timeout <= 0:
        parser.error("requests, concurrency, and rounds must be positive; timeout must be positive")
    if args.requests * args.rounds > 100_000:
        parser.error("requests multiplied by rounds must not exceed 100000")
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
            rounds=args.rounds,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
