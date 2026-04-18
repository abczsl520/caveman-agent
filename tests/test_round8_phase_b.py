"""Tests for Round 8 Phase B: Compaction Shield + Nudge LLM Integration."""
from __future__ import annotations

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════
# FR-101: Compaction Shield
# ═══════════════════════════════════════════════════════════════════


class TestSessionEssence:
    """SessionEssence data structure."""

    def test_create_default(self):
        from caveman.engines.shield import SessionEssence
        e = SessionEssence(session_id="test123")
        assert e.session_id == "test123"
        assert e.decisions == []
        assert e.progress == []
        assert e.stances == []
        assert e.key_data == {}
        assert e.open_todos == []
        assert e.turn_count == 0

    def test_to_dict_and_back(self):
        from caveman.engines.shield import SessionEssence
        e = SessionEssence(
            session_id="abc",
            decisions=["chose Python"],
            progress=["built CLI"],
            key_data={"path": "/tmp/test"},
            open_todos=["write tests"],
        )
        d = e.to_dict()
        assert d["session_id"] == "abc"
        assert d["decisions"] == ["chose Python"]
        assert isinstance(d["updated_at"], str)

        restored = SessionEssence.from_dict(d)
        assert restored.session_id == "abc"
        assert restored.decisions == ["chose Python"]
        assert restored.key_data == {"path": "/tmp/test"}

    def test_merge_deduplicates(self):
        from caveman.engines.shield import SessionEssence
        a = SessionEssence(session_id="a", decisions=["chose A", "chose B"])
        b = SessionEssence(session_id="b", decisions=["chose B", "chose C"])
        a.merge(b)
        assert a.decisions == ["chose A", "chose B", "chose C"]

    def test_summary_output(self):
        from caveman.engines.shield import SessionEssence
        e = SessionEssence(
            session_id="test",
            task="build a CLI",
            decisions=["use Typer"],
            progress=["created main.py"],
            open_todos=["add tests"],
            turn_count=5,
        )
        s = e.summary
        assert "test" in s
        assert "build a CLI" in s
        assert "use Typer" in s
        assert "created main.py" in s
        assert "add tests" in s


class TestCompactionShield:
    """CompactionShield heuristic + LLM extraction."""

    def test_heuristic_extracts_decisions(self):
        from caveman.engines.shield import CompactionShield
        shield = CompactionShield(session_id="test", store_dir=Path(tempfile.mkdtemp()))
        messages = [
            {"role": "user", "content": "What framework should we use?"},
            {"role": "assistant", "content": "I decided to use Flask because it's lightweight. Going with SQLite for the database."},
        ]
        loop = asyncio.new_event_loop()
        essence = loop.run_until_complete(shield.update(messages, "build API"))
        # Should extract decisions containing "use Flask" or "Going with SQLite"
        assert len(essence.decisions) > 0

    def test_heuristic_extracts_progress(self):
        from caveman.engines.shield import CompactionShield
        shield = CompactionShield(session_id="test", store_dir=Path(tempfile.mkdtemp()))
        messages = [
            {"role": "assistant", "content": "✅ Created the main.py file with Flask app. Done implementing the /health endpoint."},
        ]
        loop = asyncio.new_event_loop()
        essence = loop.run_until_complete(shield.update(messages, "build API"))
        assert len(essence.progress) > 0

    def test_heuristic_extracts_todos(self):
        from caveman.engines.shield import CompactionShield
        shield = CompactionShield(session_id="test", store_dir=Path(tempfile.mkdtemp()))
        messages = [
            {"role": "assistant", "content": "TODO: add authentication. Still need to write tests for the API."},
        ]
        loop = asyncio.new_event_loop()
        essence = loop.run_until_complete(shield.update(messages, "build API"))
        assert len(essence.open_todos) > 0

    def test_heuristic_extracts_key_data(self):
        from caveman.engines.shield import CompactionShield
        shield = CompactionShield(session_id="test", store_dir=Path(tempfile.mkdtemp()))
        messages = [
            {"role": "assistant", "content": "The project is at path: ~/projects/myapp. Server running on port: 8080. Version: 2.1.0"},
        ]
        loop = asyncio.new_event_loop()
        essence = loop.run_until_complete(shield.update(messages, "deploy"))
        assert len(essence.key_data) > 0

    def test_save_and_load(self):
        from caveman.engines.shield import CompactionShield, SessionEssence
        tmp = Path(tempfile.mkdtemp())
        shield = CompactionShield(session_id="persist_test", store_dir=tmp)
        messages = [
            {"role": "assistant", "content": "I decided to use React. Created the project scaffold."},
        ]
        loop = asyncio.new_event_loop()
        loop.run_until_complete(shield.update(messages, "build frontend"))
        loop.run_until_complete(shield.save())

        # Load it back
        loaded = loop.run_until_complete(CompactionShield.load("persist_test", store_dir=tmp))
        assert loaded is not None
        assert loaded.session_id == "persist_test"
        assert loaded.task == "build frontend"

    def test_load_latest(self):
        from caveman.engines.shield import CompactionShield
        import time
        tmp = Path(tempfile.mkdtemp())

        # Create two sessions
        s1 = CompactionShield(session_id="old_session", store_dir=tmp)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(s1.update([{"role": "assistant", "content": "old stuff"}], "old task"))
        loop.run_until_complete(s1.save())

        time.sleep(0.1)  # Ensure different mtime

        s2 = CompactionShield(session_id="new_session", store_dir=tmp)
        loop.run_until_complete(s2.update([{"role": "assistant", "content": "new stuff"}], "new task"))
        loop.run_until_complete(s2.save())

        latest = loop.run_until_complete(CompactionShield.load_latest(store_dir=tmp))
        assert latest is not None
        assert latest.session_id == "new_session"

    def test_load_nonexistent_returns_none(self):
        from caveman.engines.shield import CompactionShield
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            CompactionShield.load("nonexistent", store_dir=Path(tempfile.mkdtemp()))
        )
        assert result is None

    def test_llm_extraction(self):
        from caveman.engines.shield import CompactionShield
        tmp = Path(tempfile.mkdtemp())

        llm_response = json.dumps({
            "decisions": ["Use FastAPI over Flask for async support"],
            "progress": ["Set up project structure"],
            "stances": ["Prefer type hints everywhere"],
            "key_data": {"framework": "FastAPI", "python_version": "3.12"},
            "open_todos": ["Add authentication middleware"],
        })

        async def mock_llm(prompt: str) -> str:
            return llm_response

        shield = CompactionShield(session_id="llm_test", store_dir=tmp, llm_fn=mock_llm)
        messages = [
            {"role": "user", "content": "Build an API"},
            {"role": "assistant", "content": "I'll use FastAPI."},
        ]
        loop = asyncio.new_event_loop()
        essence = loop.run_until_complete(shield.update(messages, "build API"))
        assert "Use FastAPI over Flask for async support" in essence.decisions
        assert essence.key_data.get("framework") == "FastAPI"

    def test_llm_fallback_to_heuristic(self):
        from caveman.engines.shield import CompactionShield
        tmp = Path(tempfile.mkdtemp())

        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        shield = CompactionShield(session_id="fallback_test", store_dir=tmp, llm_fn=failing_llm)
        messages = [
            {"role": "assistant", "content": "I decided to use SQLite. Created the schema."},
        ]
        loop = asyncio.new_event_loop()
        # Should not raise, falls back to heuristic
        essence = loop.run_until_complete(shield.update(messages, "build DB"))
        assert essence is not None

    def test_merge_accumulates(self):
        from caveman.engines.shield import CompactionShield
        tmp = Path(tempfile.mkdtemp())
        shield = CompactionShield(session_id="merge_test", store_dir=tmp)
        loop = asyncio.new_event_loop()

        # First update
        loop.run_until_complete(shield.update(
            [{"role": "assistant", "content": "I decided to use Python."}], "task1"
        ))
        count1 = len(shield.essence.decisions)

        # Second update — should accumulate, not replace
        loop.run_until_complete(shield.update(
            [{"role": "assistant", "content": "I decided to use Docker for deployment."}], "task1"
        ))
        assert len(shield.essence.decisions) >= count1


# ═══════════════════════════════════════════════════════════════════
# FR-102: Nudge LLM Integration
# ═══════════════════════════════════════════════════════════════════


class TestNudgeLLMIntegration:
    """Nudge connected to LLM provider."""

    def test_nudge_with_llm_fn(self):
        from caveman.memory.nudge import MemoryNudge
        from caveman.memory.manager import MemoryManager
        import tempfile

        mem_dir = tempfile.mkdtemp()
        manager = MemoryManager(base_dir=mem_dir)

        llm_response = json.dumps([
            {"type": "semantic", "content": "pyenv is not pre-installed on macOS"},
            {"type": "procedural", "content": "Use brew install pyenv to install"},
        ])

        async def mock_llm(prompt: str) -> str:
            return llm_response

        nudge = MemoryNudge(memory_manager=manager, llm_fn=mock_llm, interval=1, first_nudge=1)
        turns = [
            {"role": "user", "content": "Install pyenv"},
            {"role": "assistant", "content": "pyenv is not pre-installed on macOS. Use brew install pyenv."},
        ]

        loop = asyncio.new_event_loop()
        created = loop.run_until_complete(nudge.run(turns, task="install pyenv"))
        assert len(created) >= 1

    @pytest.mark.xfail(reason="Nudge heuristic patterns changed in Round 97+")
    def test_nudge_heuristic_without_llm(self):
        from caveman.memory.nudge import MemoryNudge
        from caveman.memory.manager import MemoryManager
        import tempfile

        mem_dir = tempfile.mkdtemp()
        manager = MemoryManager(base_dir=mem_dir)

        nudge = MemoryNudge(memory_manager=manager, llm_fn=None, interval=1, first_nudge=1)
        turns = [
            {"role": "user", "content": "Deploy the app"},
            {"role": "assistant", "content": "Deployed successfully using bash and file_write tools."},
        ]

        loop = asyncio.new_event_loop()
        created = loop.run_until_complete(nudge.run(turns, task="deploy app"))
        # Heuristic should at least extract the task as episodic
        assert len(created) >= 1

    def test_nudge_llm_fallback_on_error(self):
        from caveman.memory.nudge import MemoryNudge
        from caveman.memory.manager import MemoryManager
        import tempfile

        mem_dir = tempfile.mkdtemp()
        manager = MemoryManager(base_dir=mem_dir)

        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("API down")

        nudge = MemoryNudge(memory_manager=manager, llm_fn=failing_llm, interval=1, first_nudge=1)
        turns = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Fixed the null pointer error in auth.py"},
        ]

        loop = asyncio.new_event_loop()
        # Should not raise, falls back to heuristic
        created = loop.run_until_complete(nudge.run(turns, task="fix bug"))
        assert len(created) >= 1

    def test_factory_creates_llm_fn(self):
        """Verify factory.py creates llm_fn from provider."""
        from caveman.agent.factory import _make_llm_fn
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        llm_fn = _make_llm_fn(mock_provider)
        assert callable(llm_fn)

    def test_nudge_should_nudge_timing(self):
        from caveman.memory.nudge import MemoryNudge
        from caveman.memory.manager import MemoryManager
        import tempfile

        manager = MemoryManager(base_dir=tempfile.mkdtemp())
        nudge = MemoryNudge(memory_manager=manager, interval=5, first_nudge=3)

        assert not nudge.should_nudge(0)
        assert not nudge.should_nudge(1)
        assert not nudge.should_nudge(2)
        assert nudge.should_nudge(3)  # First nudge at turn 3
