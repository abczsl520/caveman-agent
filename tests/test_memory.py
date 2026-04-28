"""Tests for memory system."""
import pytest
import asyncio
import tempfile
from pathlib import Path

from caveman.memory.flywheel_metrics import FlywheelHealth
from caveman.memory.types import MemoryType, MemoryEntry
from caveman.memory.manager import MemoryManager


def _close_manager(mgr: MemoryManager) -> None:
    if mgr.backend:
        mgr.backend.close()


def test_memory_types():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.WORKING.value == "working"


@pytest.mark.asyncio
async def test_memory_store_recall():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            mid = await mgr.store("test content about python", MemoryType.EPISODIC)
            assert mid
            results = await mgr.recall("python")
            assert len(results) >= 1
            assert "python" in results[0].content
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_memory_nudge():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            await mgr.nudge()  # should not raise
            assert True  # Nudge completed without error
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_flywheel_health_uses_real_feedback_and_recall_counters():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            stale = await mgr.store("stale memory about docker compose", MemoryType.SEMANTIC)
            helpful = await mgr.store("helpful python deployment memory", MemoryType.SEMANTIC)

            results = await mgr.recall("python deployment")
            assert {r.id for r in results} >= {helpful}
            await mgr.backend.mark_helpful(helpful, helpful=True)

            health = await FlywheelHealth.diagnose(mgr.backend)

            assert health.total_memories == 2
            assert health.memories_never_recalled == 1
            assert health.recalled_memories == 1
            assert health.recall_rate == 0.5
            assert health.memories_with_feedback == 1
            assert health.feedback_rate == 0.5
            assert health.top_recalled[0]["id"] == helpful
            assert stale not in {item["id"] for item in health.top_recalled}
            assert "recall rate=50%" in health.summary()
        finally:
            _close_manager(mgr)
