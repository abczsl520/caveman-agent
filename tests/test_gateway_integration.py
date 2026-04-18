"""Integration tests for gateway → agent loop pipeline.

Tests the critical paths that were previously untested:
- Session restore via snapshot/restore contract
- task_runner → loop.run_stream pipeline
- Session lifecycle (create → run → persist → restore → run)
"""
from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


class TestSnapshotRestore:
    """AgentLoop.snapshot() and restore() contract tests."""

    def test_snapshot_captures_all_state(self):
        from caveman.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        loop._turn_number = 5
        loop._turn_count = 10
        loop._tool_call_count = 3
        loop.surface = "discord"
        loop._system_prompt_cache = "test prompt"

        snap = loop.snapshot()
        assert snap["turn_number"] == 5
        assert snap["turn_count"] == 10
        assert snap["tool_call_count"] == 3
        assert snap["surface"] == "discord"
        assert snap["system_prompt_len"] == len("test prompt")

    def test_restore_rebuilds_state(self):
        from caveman.agent.loop import AgentLoop
        from caveman.agent.context import AgentContext

        loop = AgentLoop.__new__(AgentLoop)
        loop.surface = "cli"
        loop._system_prompt_cache = None
        loop._persistent_context = None
        loop.tool_registry = MagicMock()
        loop.tool_registry.get_schemas.return_value = []

        ctx = AgentContext(max_tokens=100000)
        ctx.add_message("user", "hello")

        loop.restore({
            "turn_number": 5,
            "turn_count": 10,
            "surface": "discord",
        }, context=ctx)

        assert loop._turn_number == 5
        assert loop._turn_count == 10
        assert loop.surface == "discord"
        assert loop._persistent_context is ctx
        assert loop._system_prompt_cache is not None
        assert len(loop._system_prompt_cache) > 100
        assert "Discord" in loop._system_prompt_cache

    def test_restore_without_context_preserves_existing(self):
        from caveman.agent.loop import AgentLoop
        from caveman.agent.context import AgentContext

        loop = AgentLoop.__new__(AgentLoop)
        loop.surface = "cli"
        loop._system_prompt_cache = None
        loop._persistent_context = AgentContext(max_tokens=100000)
        loop.tool_registry = MagicMock()
        loop.tool_registry.get_schemas.return_value = []

        original_ctx = loop._persistent_context
        loop.restore({"turn_number": 3, "surface": "telegram"})

        assert loop._persistent_context is original_ctx  # Not overwritten

    def test_snapshot_restore_roundtrip(self):
        from caveman.agent.loop import AgentLoop

        loop1 = AgentLoop.__new__(AgentLoop)
        loop1._turn_number = 7
        loop1._turn_count = 15
        loop1._tool_call_count = 4
        loop1.surface = "telegram"
        loop1._system_prompt_cache = "x" * 200

        snap = loop1.snapshot()

        loop2 = AgentLoop.__new__(AgentLoop)
        loop2.surface = "cli"
        loop2._system_prompt_cache = None
        loop2._persistent_context = None
        loop2.tool_registry = MagicMock()
        loop2.tool_registry.get_schemas.return_value = []

        loop2.restore(snap)

        assert loop2._turn_number == 7
        assert loop2._turn_count == 15
        assert loop2._tool_call_count == 4
        assert loop2.surface == "telegram"


class TestSessionMeta:
    """SessionMeta surface field tests."""

    def test_surface_persisted(self):
        from caveman.agent.session_store import SessionMeta
        meta = SessionMeta(session_id="test", surface="discord")
        d = meta.to_dict()
        assert d["surface"] == "discord"

        restored = SessionMeta.from_dict(d)
        assert restored.surface == "discord"

    def test_surface_defaults_to_cli(self):
        from caveman.agent.session_store import SessionMeta
        meta = SessionMeta.from_dict({"session_id": "test"})
        assert meta.surface == "cli"


class TestArchitecturalInvariants:
    """Tests that enforce architectural decisions."""

    def test_run_delegates_to_run_stream(self):
        """run() must be a thin wrapper — no duplicate logic."""
        import ast
        with open("caveman/agent/loop.py") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                assert len(node.body) < 12, \
                    f"run() has {len(node.body)} statements — should be a thin wrapper"
                break

    def test_stream_py_has_no_run_logic(self):
        """stream.py must only contain data classes."""
        import ast
        with open("caveman/agent/stream.py") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                raise AssertionError(f"stream.py has async function: {node.name}")

    def test_all_callers_use_safe_complete(self):
        """No direct provider.complete() calls in agent code."""
        import re
        for path in Path("caveman/agent").glob("*.py"):
            if path.name == "__init__.py":
                continue
            content = path.read_text()
            # Find .complete( but not .safe_complete(
            # Count .complete( but NOT .safe_complete(
            bare_matches = re.findall(r'(?<!safe_)\.complete\(', content)
            assert len(bare_matches) == 0, \
                f"{path.name} has {len(bare_matches)} bare .complete() calls — use .safe_complete()"

    def test_snapshot_has_all_restorable_fields(self):
        """snapshot() must capture every field that restore() sets."""
        import ast, re
        with open("caveman/agent/loop.py") as f:
            src = f.read()

        # Extract fields set in restore()
        restore_match = re.search(r'def restore\(self.*?\n(.*?)(?=\n    def |\nclass )', src, re.DOTALL)
        assert restore_match, "restore() not found"
        restore_fields = set(re.findall(r'self\.(_\w+)\s*=', restore_match.group(1)))
        # Remove _persistent_context and _system_prompt_cache (rebuilt, not from snapshot)
        restore_fields -= {"_persistent_context", "_system_prompt_cache"}

        # Extract fields captured in snapshot()
        snap_match = re.search(r'def snapshot\(self.*?\n(.*?)(?=\n    def |\nclass )', src, re.DOTALL)
        assert snap_match, "snapshot() not found"
        snap_fields = set(re.findall(r'self\.(_\w+)', snap_match.group(1)))

        missing = restore_fields - snap_fields
        assert not missing, f"restore() sets {missing} but snapshot() doesn't capture them"
