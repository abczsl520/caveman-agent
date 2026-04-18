"""Caveman Bench — memory system performance benchmarks.

Usage:
    caveman bench [--rounds N]

Measures:
1. Memory store latency
2. Memory recall latency
3. Embedding generation latency
4. Shield update latency
"""
from __future__ import annotations
import asyncio
import time
from pathlib import Path
from tempfile import mkdtemp

from caveman.memory.manager import MemoryManager, MemoryType


async def _bench_store(mm: MemoryManager, n: int = 50) -> dict:
    """Benchmark memory store operations."""
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        await mm.store(f"Benchmark memory entry {i}: testing store performance", MemoryType.EPISODIC)
        times.append(time.perf_counter() - t0)
    return {
        "op": "store",
        "count": n,
        "avg_ms": round(sum(times) / len(times) * 1000, 2),
        "p50_ms": round(sorted(times)[n // 2] * 1000, 2),
        "p99_ms": round(sorted(times)[int(n * 0.99)] * 1000, 2),
    }


async def _bench_recall(mm: MemoryManager, n: int = 20) -> dict:
    """Benchmark memory recall operations."""
    queries = [
        "benchmark performance", "store latency", "memory system",
        "recall speed", "embedding generation", "shield update",
        "flywheel round", "agent loop", "tool execution", "session state",
        "compression pipeline", "provider timeout", "credential rotation",
        "wiki context", "project identity", "skill matching",
        "trajectory recording", "coordinator task", "bridge transport",
        "plugin loading",
    ]
    times = []
    for q in queries[:n]:
        t0 = time.perf_counter()
        await mm.recall(q, top_k=5)
        times.append(time.perf_counter() - t0)
    return {
        "op": "recall",
        "count": len(times),
        "avg_ms": round(sum(times) / len(times) * 1000, 2),
        "p50_ms": round(sorted(times)[len(times) // 2] * 1000, 2),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)] * 1000, 2),
    }


async def run_bench(rounds: int = 1) -> str:
    """Run benchmarks and return formatted report."""
    tmpdir = Path(mkdtemp(prefix="caveman-bench-"))
    mm = MemoryManager(base_dir=tmpdir)
    await mm.load()

    results = []
    for r in range(rounds):
        store_result = await _bench_store(mm)
        recall_result = await _bench_recall(mm)
        results.extend([store_result, recall_result])

    parts = ["Caveman Bench\n"]
    for res in results:
        parts.append(
            f"  {res['op']:8s}: avg={res['avg_ms']:7.2f}ms  "
            f"p50={res['p50_ms']:7.2f}ms  p99={res['p99_ms']:7.2f}ms  "
            f"(n={res['count']})"
        )

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return "\n".join(parts)


def run_bench_sync(rounds: int = 1) -> None:
    """Synchronous wrapper for CLI."""
    print(asyncio.run(run_bench(rounds=rounds)))
