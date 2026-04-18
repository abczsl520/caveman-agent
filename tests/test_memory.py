"""Tests for memory system."""
import pytest
import asyncio
import tempfile
from pathlib import Path
from caveman.memory.types import MemoryType, MemoryEntry
from caveman.memory.manager import MemoryManager


def test_memory_types():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.WORKING.value == "working"


@pytest.mark.asyncio
async def test_memory_store_recall():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager(base_dir=td)
        mid = await mgr.store("test content about python", MemoryType.EPISODIC)
        assert mid
        results = await mgr.recall("python")
        assert len(results) >= 1
        assert "python" in results[0].content


@pytest.mark.asyncio
async def test_memory_nudge():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager(base_dir=td)
        await mgr.nudge()  # should not raise
