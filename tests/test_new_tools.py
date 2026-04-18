"""Tests for memory_tool, process_tool, delegate_tool, and context injection."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caveman.memory.types import MemoryEntry, MemoryType
from caveman.tools.registry import ToolRegistry


# ── Fixtures ──

@pytest.fixture
def mock_memory_manager():
    mgr = MagicMock()
    mgr.recall = AsyncMock(return_value=[
        MemoryEntry(id="m1", content="test memory", memory_type=MemoryType.SEMANTIC,
                    created_at=datetime(2025, 1, 1), metadata={}),
    ])
    mgr.store = AsyncMock(return_value="new-id-123")
    mgr.recent = MagicMock(return_value=[
        MemoryEntry(id="m2", content="recent one", memory_type=MemoryType.EPISODIC,
                    created_at=datetime(2025, 6, 1), metadata={}),
    ])
    return mgr


@pytest.fixture
def registry_with_context(mock_memory_manager):
    reg = ToolRegistry()
    reg._register_builtins()
    reg.set_context("memory_manager", mock_memory_manager)
    reg.set_context("trajectory_recorder", MagicMock())
    return reg


# ── Memory tool tests ──

@pytest.mark.asyncio
async def test_memory_tool_search(registry_with_context, mock_memory_manager):
    result = await registry_with_context.dispatch("memory_search", {"query": "test", "limit": 3})
    assert isinstance(result, list)
    assert result[0]["id"] == "m1"
    assert result[0]["type"] == "semantic"
    mock_memory_manager.recall.assert_awaited_once_with("test", top_k=3)


@pytest.mark.asyncio
async def test_memory_tool_store(registry_with_context, mock_memory_manager):
    result = await registry_with_context.dispatch("memory_store", {"content": "remember this"})
    assert result["ok"] is True
    assert result["memory_id"] == "new-id-123"
    mock_memory_manager.store.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_tool_recent(registry_with_context, mock_memory_manager):
    result = await registry_with_context.dispatch("memory_recent", {"limit": 5})
    assert isinstance(result, list)
    assert result[0]["id"] == "m2"
    mock_memory_manager.recent.assert_called_once_with(limit=5)


# ── Process tool tests ──

@pytest.mark.asyncio
async def test_process_tool_lifecycle():
    from caveman.tools.builtin.process_tool import _PROCESSES
    _PROCESSES.clear()

    reg = ToolRegistry()
    reg._register_builtins()

    # Start
    result = await reg.dispatch("process_start", {"command": "echo hello", "label": "test-echo"})
    assert "pid" in result
    pid = result["pid"]

    # Wait for it to finish
    await _PROCESSES[pid]["proc"].wait()

    # List
    procs = await reg.dispatch("process_list", {})
    assert any(p["pid"] == pid for p in procs)

    # Output
    out = await reg.dispatch("process_output", {"pid": pid})
    assert "stdout" in out
    assert out["running"] is False

    # Kill (already dead, should still succeed)
    kill_result = await reg.dispatch("process_kill", {"pid": pid})
    assert kill_result["ok"] is True

    _PROCESSES.clear()


# ── Delegate tool tests ──

@pytest.mark.asyncio
async def test_delegate_tool():
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="sub-agent done")

    with patch("caveman.tools.builtin.delegate_tool.create_loop", return_value=mock_loop) as mock_create:
        reg = ToolRegistry()
        reg._register_builtins()
        result = await reg.dispatch("delegate", {"task": "do something", "max_iterations": 5})
        assert result == "sub-agent done"
        mock_create.assert_called_once_with(max_iterations=5)
        mock_loop.run.assert_awaited_once_with("do something")


# ── Context injection tests ──

@pytest.mark.asyncio
async def test_context_injection():
    """Verify _context is passed to tools that accept it."""
    reg = ToolRegistry()
    reg._register_builtins()
    reg.set_context("memory_manager", MagicMock())

    # memory_search accepts _context — should not crash
    mgr = reg._context["memory_manager"]
    mgr.recall = AsyncMock(return_value=[])
    result = await reg.dispatch("memory_search", {"query": "test"})
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_context_not_injected_to_plain_tools():
    """Tools without _context param should not receive it."""
    reg = ToolRegistry()
    reg._register_builtins()
    reg.set_context("memory_manager", MagicMock())

    # bash tool doesn't accept _context — should still work
    result = await reg.dispatch("bash", {"command": "echo ok"})
    assert "ok" in str(result)
