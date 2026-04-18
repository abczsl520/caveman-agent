"""Integration test: Tool Registry → Context Injection pipeline.

Verifies that ToolRegistry correctly injects context into tools and
that tools interact with real subsystems (MemoryManager, todo persistence).
"""
from __future__ import annotations

import json
import pytest

from caveman.tools.registry import ToolRegistry
from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryType


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg._register_builtins()
    return reg


@pytest.fixture
def memory_manager(tmp_path):
    return MemoryManager(base_dir=tmp_path / "memory")


@pytest.mark.asyncio
async def test_memory_search_via_registry(registry, memory_manager):
    """Dispatch memory_search through registry with injected MemoryManager."""
    registry.set_context("memory_manager", memory_manager)

    # Store some memories first
    await memory_manager.store("Python uses 0-based indexing", MemoryType.SEMANTIC)
    await memory_manager.store("Rust has zero-cost abstractions", MemoryType.SEMANTIC)

    result = await registry.dispatch("memory_search", {"query": "Python indexing", "limit": 5})
    assert isinstance(result, list)
    assert len(result) > 0
    assert any("Python" in r["content"] for r in result)


@pytest.mark.asyncio
async def test_memory_store_via_registry(registry, memory_manager):
    """Dispatch memory_store through registry and verify persistence."""
    registry.set_context("memory_manager", memory_manager)

    result = await registry.dispatch("memory_store", {
        "content": "Integration tests verify real component interactions",
        "memory_type": "semantic",
    })
    assert result["ok"] is True
    assert "memory_id" in result

    # Verify it's actually stored
    assert memory_manager.total_count == 1
    entries = memory_manager.all_entries()
    assert entries[0].content == "Integration tests verify real component interactions"


@pytest.mark.asyncio
async def test_memory_recent_via_registry(registry, memory_manager):
    """Dispatch memory_recent through registry."""
    registry.set_context("memory_manager", memory_manager)

    await memory_manager.store("First memory", MemoryType.EPISODIC)
    await memory_manager.store("Second memory", MemoryType.EPISODIC)

    result = await registry.dispatch("memory_recent", {"limit": 10})
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_todo_add_via_registry(registry, tmp_path, monkeypatch):
    """Dispatch todo_add through registry and verify disk persistence."""
    # Redirect todo file to tmp_path
    todo_file = tmp_path / "todos.json"
    import caveman.tools.builtin.todo_tool as todo_mod
    monkeypatch.setattr(todo_mod, "_TODO_FILE", todo_file)

    result = await registry.dispatch("todo_add", {"title": "Write integration tests", "priority": "high"})
    assert result["ok"] is True
    assert result["title"] == "Write integration tests"

    # Verify on disk
    assert todo_file.exists()
    data = json.loads(todo_file.read_text())
    assert len(data) == 1
    assert data[0]["title"] == "Write integration tests"
    assert data[0]["priority"] == "high"
    assert data[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_todo_lifecycle_via_registry(registry, tmp_path, monkeypatch):
    """Full todo lifecycle: add → list → done → list → remove."""
    todo_file = tmp_path / "todos.json"
    import caveman.tools.builtin.todo_tool as todo_mod
    monkeypatch.setattr(todo_mod, "_TODO_FILE", todo_file)

    # Add
    r1 = await registry.dispatch("todo_add", {"title": "Task A"})
    r2 = await registry.dispatch("todo_add", {"title": "Task B"})
    todo_id = r1["id"]

    # List pending
    pending = await registry.dispatch("todo_list", {"status": "pending"})
    assert len(pending) == 2

    # Mark done
    done_result = await registry.dispatch("todo_done", {"id": todo_id})
    assert done_result["ok"] is True

    # List pending (should be 1 now)
    pending = await registry.dispatch("todo_list", {"status": "pending"})
    assert len(pending) == 1

    # Remove
    remove_result = await registry.dispatch("todo_remove", {"id": r2["id"]})
    assert remove_result["ok"] is True

    # List all
    all_todos = await registry.dispatch("todo_list", {"status": "all"})
    assert len(all_todos) == 1  # only the done one remains


@pytest.mark.asyncio
async def test_context_injection_missing_manager(registry):
    """memory_search without injected manager returns error gracefully."""
    # Don't set memory_manager context
    result = await registry.dispatch("memory_search", {"query": "test"})
    assert isinstance(result, list)
    assert result[0].get("error") is not None


@pytest.mark.asyncio
async def test_file_write_via_registry(registry, tmp_path):
    """Dispatch file_write tool and verify the file is created."""
    test_file = tmp_path / "hello.txt"
    result = await registry.dispatch("file_write", {
        "path": str(test_file),
        "content": "Hello, integration test!",
    })
    assert test_file.exists()
    assert test_file.read_text() == "Hello, integration test!"
