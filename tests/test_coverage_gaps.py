"""Tests for previously untested core classes — closing coverage gaps.

Every public class should have at least one test proving it works correctly.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest


# ── AgentContext (Message class) ──

def test_message_dataclass():
    from caveman.agent.context import Message
    msg = Message(role="user", content="hello", tokens=10)
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tokens == 10


def test_context_utilization():
    from caveman.agent.context import AgentContext
    ctx = AgentContext(max_tokens=1000)
    ctx.add_message("user", "hello world test message", tokens=250)
    assert ctx.utilization == 0.25
    ctx.add_message("assistant", "response", tokens=250)
    assert ctx.utilization == 0.50


def test_context_clear():
    from caveman.agent.context import AgentContext
    ctx = AgentContext()
    ctx.add_message("user", "hello", tokens=100)
    assert ctx.total_tokens == 100
    ctx.clear()
    assert ctx.total_tokens == 0


# ── CompressionPipeline + Stats ──

def test_compression_stats():
    from caveman.compression.pipeline import CompressionStats
    stats = CompressionStats(original_tokens=1000, final_tokens=600)
    assert stats.ratio == 0.6


def test_compression_pipeline_micro():
    async def _run():
        from caveman.compression.pipeline import CompressionPipeline
        pipeline = CompressionPipeline()

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},  # duplicate
            {"role": "assistant", "content": "hi there"},
        ]
        result, stats = await pipeline.compress(messages, context_usage=0.0)
        assert len(result) == 2  # deduped
        assert stats.layer_applied == "micro"

    asyncio.run(_run())


def test_compression_pipeline_normal():
    async def _run():
        from caveman.compression.pipeline import CompressionPipeline
        pipeline = CompressionPipeline()

        messages = [
            {"role": "user", "content": "test"},
            {"role": "tool", "content": "x" * 2000},  # Long tool result
            {"role": "assistant", "content": "ok"},
        ]
        result, stats = await pipeline.compress(messages, context_usage=0.7)
        assert stats.layer_applied == "normal"
        # Tool result should be truncated
        tool_msg = [m for m in result if m.get("role") == "tool"]
        if tool_msg:
            assert len(tool_msg[0]["content"]) < 2000

    asyncio.run(_run())


def test_compression_pipeline_smart_no_provider():
    async def _run():
        from caveman.compression.pipeline import CompressionPipeline
        pipeline = CompressionPipeline(preserve_last_n=3)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result, stats = await pipeline.compress(messages, context_usage=0.9)
        assert stats.layer_applied == "smart"
        # Heuristic summary works without LLM
        assert stats.messages_summarized > 0
        assert len(result) < len(messages)

    asyncio.run(_run())


# ── SubTask ──

def test_subtask_to_dict():
    from caveman.coordinator.engine import SubTask, TaskStatus
    task = SubTask(task_id="t1", description="Build something", agent="claude-code")
    task.status = TaskStatus.COMPLETED
    task.result = "Done!"

    d = task.to_dict()
    assert d["task_id"] == "t1"
    assert d["status"] == "completed"
    assert d["result"] == "Done!"


# ── Memory concurrent safety ──

def test_memory_concurrent_store():
    """Verify concurrent stores don't corrupt data."""
    async def _run():
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType

        with tempfile.TemporaryDirectory() as td:
            mm = MemoryManager(base_dir=td)

            # Store 10 memories concurrently
            tasks = [
                mm.store(f"memory {i}", MemoryType.EPISODIC)
                for i in range(10)
            ]
            ids = await asyncio.gather(*tasks)

            assert len(ids) == 10
            assert len(set(ids)) == 10  # All unique IDs
            assert mm.total_count == 10

    asyncio.run(_run())


# ── SQLite context manager ──

def test_sqlite_context_manager():
    async def _run():
        from caveman.memory.sqlite_store import SQLiteMemoryStore
        from caveman.memory.types import MemoryType

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            async with SQLiteMemoryStore(db_path=db_path) as store:
                await store.store("The agent uses SQLite for persistent memory storage", MemoryType.SEMANTIC)
                assert store.total_count == 1
            # After exit, connection should be closed
            assert store._conn is None

    asyncio.run(_run())


# ── Display module (no circular import) ──

def test_display_functions_exist():
    """Verify display module exists and has all required functions."""
    from caveman.agent.display import (
        show_tool_call, show_tool_result, show_memory_nudge,
        show_skill_nudge, show_error, show_thinking,
    )
    # Should not crash — just log
    show_tool_call("bash", {"command": "ls"})
    show_tool_result("bash", "file.txt", success=True)
    show_memory_nudge()
    show_skill_nudge()
    show_error("test error")
    show_thinking()


def test_no_circular_import():
    """Verify the loop→tui→factory→loop cycle is broken."""
    # This should import without hanging or crashing
    from caveman.agent.loop import AgentLoop
    from caveman.agent.factory import create_loop
    from caveman.cli.tui import interactive_loop
    # All three imported successfully = no circular dependency at runtime


# ── Paths module ──

def test_paths_constants():
    from caveman.paths import (
        CAVEMAN_HOME, MEMORY_DIR, SKILLS_DIR, PLUGINS_DIR,
        TRAJECTORIES_DIR, CONFIG_PATH, MEMORY_DB_PATH,
    )
    assert CAVEMAN_HOME.name == ".caveman"
    assert MEMORY_DIR.parent == CAVEMAN_HOME
    assert CONFIG_PATH.suffix == ".yaml"
    assert MEMORY_DB_PATH.suffix == ".db"


# ── MemoryEntry ──

def test_memory_entry_dataclass():
    from caveman.memory.types import MemoryEntry, MemoryType
    entry = MemoryEntry(
        id="abc123", content="test content",
        memory_type=MemoryType.SEMANTIC, created_at=datetime.now(),
    )
    assert entry.id == "abc123"
    assert entry.memory_type == MemoryType.SEMANTIC
    assert entry.metadata == {}  # default empty dict


# ── Bridges (basic instantiation) ──

def test_hermes_bridge_init():
    from caveman.bridge.hermes_bridge import HermesBridge
    bridge = HermesBridge(base_url="http://localhost:8080", api_key="test")
    assert bridge.base_url == "http://localhost:8080"


def test_openclaw_bridge_init():
    from caveman.bridge.openclaw_bridge import OpenClawBridge
    bridge = OpenClawBridge(transport="cli", session_key="test-session")
    assert bridge.transport_name == "none"  # not connected yet
    assert bridge.session_key == "test-session"
    assert not bridge.is_connected


# ── PermissionLevel enum ──

def test_permission_level_values():
    from caveman.security.permissions import PermissionLevel
    assert PermissionLevel.AUTO.value == "auto"
    assert PermissionLevel.ASK.value == "ask"
    assert PermissionLevel.DENY.value == "deny"


# ── __all__ exports ──

def test_all_modules_have_exports():
    """Every __init__.py should define __all__."""
    from pathlib import Path
    init_files = list(Path("caveman").rglob("__init__.py"))
    for init in init_files:
        content = init.read_text()
        assert "__all__" in content, f"Missing __all__ in {init}"


def test_no_hardcoded_caveman_paths():
    """No module should hardcode ~/.caveman — use paths.py instead."""
    import ast
    from pathlib import Path
    violations = []
    for f in sorted(Path("caveman").rglob("*.py")):
        if "__pycache__" in str(f) or f.name == "paths.py":
            continue
        content = f.read_text()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        # Check all string literals in non-docstring positions
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "~/.caveman" in node.value:
                    # Skip module/class/function docstrings
                    violations.append(f"{f}:{node.lineno}: {node.value[:80]}")
    # Filter out docstrings by checking if parent is Expr (docstring)
    # Re-check with proper AST docstring detection
    real_violations = []
    for f in sorted(Path("caveman").rglob("*.py")):
        if "__pycache__" in str(f) or f.name == "paths.py":
            continue
        content = f.read_text()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        # Collect docstring line numbers
        docstring_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    ds = node.body[0]
                    for ln in range(ds.lineno, ds.end_lineno + 1):
                        docstring_lines.add(ln)
        # Check assignments, function args, etc.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "~/.caveman" in node.value and node.lineno not in docstring_lines:
                    real_violations.append(f"{f}:{node.lineno}: {node.value[:80]}")
    assert real_violations == [], f"Hardcoded paths found (use paths.py):\n" + "\n".join(real_violations)
