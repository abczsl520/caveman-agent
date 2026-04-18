"""Tests for Round 99 performance optimizations.

Covers: memory cache, cache invalidation, batch store/recall,
metrics timing/summary, and dispatch timing logging.
"""
import asyncio
import time
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path


# ── Memory cache tests ──

@pytest.fixture
def memory_manager(tmp_path):
    from caveman.memory.manager import MemoryManager
    return MemoryManager(base_dir=tmp_path)


@pytest.mark.asyncio
async def test_memory_cache_hit(memory_manager):
    """Second recall with same query should hit cache."""
    from caveman.memory.types import MemoryType
    await memory_manager.store("Python is a programming language", MemoryType.SEMANTIC)

    # First call — cache miss
    r1 = await memory_manager.recall("Python programming")
    assert len(r1) == 1

    # Second call — should hit cache (same object returned)
    r2 = await memory_manager.recall("Python programming")
    assert r1 == r2
    assert memory_manager._recall_cache.size > 0


@pytest.mark.asyncio
async def test_memory_cache_invalidation(memory_manager):
    """Cache should be invalidated after store()."""
    from caveman.memory.types import MemoryType
    await memory_manager.store("Hello world", MemoryType.SEMANTIC)

    # Populate cache
    await memory_manager.recall("Hello")
    assert memory_manager._recall_cache.size == 1

    # Store invalidates cache
    await memory_manager.store("Goodbye world", MemoryType.SEMANTIC)
    assert memory_manager._recall_cache.size == 0


@pytest.mark.asyncio
async def test_memory_cache_ttl(memory_manager):
    """Cache entries should expire after TTL."""
    from caveman.memory.types import MemoryType
    from caveman.memory.recall_cache import RecallCache

    # Use a very short TTL for testing
    memory_manager._recall_cache = RecallCache(ttl=0.01)
    await memory_manager.store("Test content", MemoryType.SEMANTIC)

    await memory_manager.recall("Test")
    assert memory_manager._recall_cache.size == 1

    # Wait for TTL to expire
    await asyncio.sleep(0.02)

    # Should be a cache miss now (expired)
    result = memory_manager._recall_cache.get("Test", 5, None)
    assert result is None


# ── Batch operations tests ──

@pytest.mark.asyncio
async def test_batch_store(memory_manager):
    """store_batch should store multiple memories at once."""
    items = [
        {"content": "Memory one", "memory_type": "semantic"},
        {"content": "Memory two", "memory_type": "episodic"},
        {"content": "Memory three", "memory_type": "semantic"},
    ]
    ids = await memory_manager.store_batch(items)
    assert len(ids) == 3
    assert memory_manager.total_count == 3


@pytest.mark.asyncio
async def test_batch_recall(memory_manager):
    """recall_batch should return results for multiple queries."""
    from caveman.memory.types import MemoryType
    await memory_manager.store("Python programming", MemoryType.SEMANTIC)
    await memory_manager.store("JavaScript web dev", MemoryType.SEMANTIC)

    results = await memory_manager.recall_batch(["Python", "JavaScript"])
    assert len(results) == 2
    assert len(results[0]) >= 1
    assert len(results[1]) >= 1


# ── Metrics tests ──

def test_metrics_timing():
    """AgentMetrics should record and retrieve timings."""
    from caveman.agent.metrics import AgentMetrics
    m = AgentMetrics()
    m.record_timing("test_op", 0.5)
    m.record_timing("test_op", 1.0)
    m.record_timing("test_op", 1.5)

    summary = m.summary()
    assert "test_op" in summary["timings"]
    t = summary["timings"]["test_op"]
    assert t["count"] == 3
    assert abs(t["avg"] - 1.0) < 0.01
    assert t["min"] == 0.5
    assert t["max"] == 1.5


def test_metrics_summary():
    """Summary should include counters and percentiles."""
    from caveman.agent.metrics import AgentMetrics
    m = AgentMetrics()

    # Add enough data points for meaningful percentiles
    for i in range(100):
        m.record_timing("latency", i * 0.01)
    m.increment("requests", 100)

    summary = m.summary()
    assert summary["counters"]["requests"] == 100
    lat = summary["timings"]["latency"]
    assert lat["count"] == 100
    assert lat["p50"] >= 0.0
    assert lat["p95"] >= lat["p50"]
    assert lat["p99"] >= lat["p95"]


def test_metrics_reset():
    """Reset should clear all data."""
    from caveman.agent.metrics import AgentMetrics
    m = AgentMetrics()
    m.record_timing("x", 1.0)
    m.increment("y")
    m.reset()
    summary = m.summary()
    assert summary["timings"] == {}
    assert summary["counters"] == {}


def test_metrics_timer_context_manager():
    """Timer context manager should record elapsed time."""
    from caveman.agent.metrics import AgentMetrics
    m = AgentMetrics()
    with m.timer("sleep_test"):
        time.sleep(0.01)
    summary = m.summary()
    assert "sleep_test" in summary["timings"]
    assert summary["timings"]["sleep_test"]["count"] == 1
    assert summary["timings"]["sleep_test"]["avg"] >= 0.005


# ── Dispatch timing log test ──

@pytest.mark.asyncio
async def test_dispatch_timing_log(caplog):
    """Slow tool dispatch should log a warning."""
    from caveman.tools.registry import ToolRegistry, tool

    @tool(name="slow_tool", description="A slow tool", params={}, required=[])
    async def slow_tool():
        await asyncio.sleep(1.1)
        return {"ok": True}

    registry = ToolRegistry()
    registry.register_decorated(slow_tool)

    with caplog.at_level(logging.WARNING, logger="caveman.tools.registry"):
        await registry.dispatch("slow_tool", {})

    assert any("Slow tool dispatch" in r.message for r in caplog.records)
