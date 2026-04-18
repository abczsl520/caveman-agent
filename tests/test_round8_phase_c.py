"""Tests for Round 8 Phase C: Recall Engine."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:
    yaml = None


def _run(coro):
    """Helper to run async in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════
# FR-104: Recall Engine
# ═══════════════════════════════════════════════════════════════════


class TestRecallEngine:
    """Recall Engine restores context from previous sessions."""

    def test_restore_empty_no_sessions(self):
        from caveman.engines.recall import RecallEngine
        tmp = Path(tempfile.mkdtemp())
        recall = RecallEngine(sessions_dir=tmp)
        result = _run(recall.restore("some task"))
        assert result == ""

    def test_restore_with_previous_session(self):
        from caveman.engines.recall import RecallEngine
        from caveman.engines.shield import CompactionShield
        if yaml is None:
            pytest.skip("pyyaml required")

        tmp = Path(tempfile.mkdtemp())

        # Create a previous session via Shield
        shield = CompactionShield(session_id="prev_session", store_dir=tmp)
        messages = [
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "I decided to use FastAPI. Created the project structure. TODO: add authentication."},
        ]
        _run(shield.update(messages, "build API"))
        _run(shield.save())

        # Now recall
        recall = RecallEngine(sessions_dir=tmp)
        result = _run(recall.restore("continue API work"))
        assert "prev_session" in result
        assert "Previous Session Context" in result

    def test_restore_with_multiple_sessions(self):
        from caveman.engines.recall import RecallEngine
        from caveman.engines.shield import CompactionShield
        if yaml is None:
            pytest.skip("pyyaml required")

        import time
        tmp = Path(tempfile.mkdtemp())

        # Create two sessions
        s1 = CompactionShield(session_id="session_old", store_dir=tmp)
        _run(s1.update([{"role": "assistant", "content": "Old session work"}], "old task"))
        _run(s1.save())

        time.sleep(0.1)

        s2 = CompactionShield(session_id="session_new", store_dir=tmp)
        _run(s2.update([{"role": "assistant", "content": "New session work"}], "new task"))
        _run(s2.save())

        recall = RecallEngine(sessions_dir=tmp, max_essences=2)
        result = _run(recall.restore("continue"))
        assert "Most Recent Session" in result
        assert "session_new" in result

    def test_restore_max_essences_limit(self):
        from caveman.engines.recall import RecallEngine
        from caveman.engines.shield import CompactionShield
        if yaml is None:
            pytest.skip("pyyaml required")

        import time
        tmp = Path(tempfile.mkdtemp())

        # Create 5 sessions
        for i in range(5):
            s = CompactionShield(session_id=f"session_{i}", store_dir=tmp)
            _run(s.update([{"role": "assistant", "content": f"Work {i}"}], f"task {i}"))
            _run(s.save())
            time.sleep(0.05)

        recall = RecallEngine(sessions_dir=tmp, max_essences=2)
        result = _run(recall.restore("continue"))
        # Should only include 2 most recent
        assert "session_4" in result
        assert "session_3" in result
        assert "session_0" not in result

    def test_has_previous_sessions_false(self):
        from caveman.engines.recall import RecallEngine
        tmp = Path(tempfile.mkdtemp())
        recall = RecallEngine(sessions_dir=tmp)
        assert _run(recall.has_previous_sessions()) is False

    def test_has_previous_sessions_true(self):
        from caveman.engines.recall import RecallEngine
        from caveman.engines.shield import CompactionShield
        if yaml is None:
            pytest.skip("pyyaml required")

        tmp = Path(tempfile.mkdtemp())
        shield = CompactionShield(session_id="exists", store_dir=tmp)
        _run(shield.update([{"role": "assistant", "content": "hello"}], "test"))
        _run(shield.save())

        recall = RecallEngine(sessions_dir=tmp)
        assert _run(recall.has_previous_sessions()) is True

    def test_restore_with_memory_manager(self):
        from caveman.engines.recall import RecallEngine
        from caveman.engines.shield import CompactionShield
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType
        if yaml is None:
            pytest.skip("pyyaml required")

        tmp_sessions = Path(tempfile.mkdtemp())
        tmp_memory = Path(tempfile.mkdtemp())

        # Create a session
        shield = CompactionShield(session_id="mem_test", store_dir=tmp_sessions)
        _run(shield.update([{"role": "assistant", "content": "Built the API"}], "build API"))
        _run(shield.save())

        # Store a memory
        manager = MemoryManager(base_dir=tmp_memory)
        _run(manager.store("Server runs on port 8080", MemoryType.SEMANTIC))

        recall = RecallEngine(sessions_dir=tmp_sessions, memory_manager=manager)
        result = _run(recall.restore("check server"))
        assert "mem_test" in result
        # Memory recall depends on search matching, may or may not appear

    def test_recall_integrated_in_loop(self):
        """Verify RecallEngine is instantiated in AgentLoop."""
        from caveman.agent.loop import AgentLoop
        loop = AgentLoop()
        assert hasattr(loop, "_recall")
        from caveman.engines.recall import RecallEngine
        assert isinstance(loop._recall, RecallEngine)
