"""Tests for SQLite session DB and parallel tool execution."""
import asyncio
import json
import pytest
import time
from pathlib import Path


class TestSessionDB:
    """Test SQLite session store."""

    def test_create_and_load_meta(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        meta = SessionMeta(session_id="test1", model="claude", surface="discord",
                            started_at=time.time())
        db.save_meta(meta)
        loaded = db.load_meta("test1")
        assert loaded is not None
        assert loaded.session_id == "test1"
        assert loaded.model == "claude"
        db.close()

    def test_append_and_load_transcript(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        db.append_turn("s1", "user", "hello")
        db.append_turn("s1", "assistant", "hi there")
        transcript = db.load_transcript("s1")
        assert len(transcript) == 2
        assert transcript[0]["role"] == "user"
        assert transcript[1]["content"] == "hi there"
        db.close()

    def test_transcript_turn_count(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        db.append_turn("s1", "user", "a")
        db.append_turn("s1", "assistant", "b")
        db.append_turn("s1", "user", "c")
        assert db.transcript_turn_count("s1") == 3
        db.close()

    def test_update_meta(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        db.update_meta("s1", turn_count=5, total_tokens=1000)
        meta = db.load_meta("s1")
        assert meta.turn_count == 5
        assert meta.total_tokens == 1000
        db.close()

    def test_save_and_load_snapshot(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        snap = {"turn_number": 5, "tool_call_count": 20, "surface": "discord"}
        db.save_snapshot("s1", snap)
        loaded = db.load_snapshot("s1")
        assert loaded["turn_number"] == 5
        assert loaded["tool_call_count"] == 20
        db.close()

    def test_compaction(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        db.save_compaction("s1", "Summary of first 10 turns", 10)
        comps = db.load_compactions("s1")
        assert len(comps) == 1
        assert comps[0]["summary"] == "Summary of first 10 turns"
        meta = db.load_meta("s1")
        assert meta.compaction_count == 1
        db.close()

    def test_list_sessions(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        for i in range(3):
            db.save_meta(SessionMeta(session_id=f"s{i}", started_at=time.time(),
                                      last_active_at=time.time() + i))
        sessions = db.list_sessions()
        assert len(sessions) == 3
        assert sessions[0].session_id == "s2"  # Most recent first
        db.close()

    def test_delete_session(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", started_at=time.time()))
        db.append_turn("s1", "user", "hello")
        assert db.delete_session("s1")
        assert db.load_meta("s1") is None
        assert db.transcript_turn_count("s1") == 0
        db.close()

    def test_usage_summary(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        from caveman.agent.session_store import SessionMeta
        db = SessionDB(tmp_path / "test.db")
        db.save_meta(SessionMeta(session_id="s1", turn_count=10, total_tokens=5000,
                                  total_cost_usd=0.05, started_at=time.time()))
        db.save_meta(SessionMeta(session_id="s2", turn_count=20, total_tokens=10000,
                                  total_cost_usd=0.10, started_at=time.time()))
        summary = db.usage_summary()
        assert summary["total_sessions"] == 2
        assert summary["total_turns"] == 30
        assert summary["total_tokens"] == 15000
        db.close()

    def test_migrate_from_json(self, tmp_path):
        from caveman.agent.session_db import SessionDB
        # Create a fake JSON store
        json_dir = tmp_path / "json_sessions"
        session_dir = json_dir / "test_session"
        session_dir.mkdir(parents=True)
        meta = {"session_id": "test_session", "model": "claude", "started_at": time.time(),
                "last_active_at": time.time(), "turn_count": 2, "total_tokens": 100,
                "total_cost_usd": 0.01, "compaction_count": 0, "title": "", "tags": [],
                "surface": "discord"}
        (session_dir / "meta.json").write_text(json.dumps(meta))
        with (session_dir / "transcript.jsonl").open("w") as f:
            f.write(json.dumps({"role": "user", "content": "hello", "ts": time.time()}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "hi", "ts": time.time()}) + "\n")

        db = SessionDB(tmp_path / "test.db")
        migrated = db.migrate_from_json(json_dir)
        assert migrated == 1
        assert db.transcript_turn_count("test_session") == 2
        loaded = db.load_meta("test_session")
        assert loaded.model == "claude"
        db.close()


class TestParallelToolExecution:
    """Test parallel tool execution logic."""

    def test_can_parallelize_read_only(self):
        from caveman.agent.tools_exec import _can_parallelize
        calls = [
            {"name": "file_read", "id": "1", "input": {"path": "a.py"}},
            {"name": "file_read", "id": "2", "input": {"path": "b.py"}},
        ]
        assert _can_parallelize(calls) is True

    def test_cannot_parallelize_with_bash(self):
        from caveman.agent.tools_exec import _can_parallelize
        calls = [
            {"name": "file_read", "id": "1", "input": {"path": "a.py"}},
            {"name": "bash", "id": "2", "input": {"command": "ls"}},
        ]
        assert _can_parallelize(calls) is False

    def test_single_tool_not_parallel(self):
        from caveman.agent.tools_exec import _can_parallelize
        calls = [{"name": "file_read", "id": "1", "input": {"path": "a.py"}}]
        assert _can_parallelize(calls) is False

    def test_cannot_parallelize_write_tools(self):
        from caveman.agent.tools_exec import _can_parallelize
        calls = [
            {"name": "file_write", "id": "1", "input": {"path": "a.py", "content": "x"}},
            {"name": "file_write", "id": "2", "input": {"path": "b.py", "content": "y"}},
        ]
        assert _can_parallelize(calls) is False


class TestPromptSection:
    """Test PromptSection enum."""

    def test_enum_values(self):
        from caveman.agent.prompt import PromptSection
        assert PromptSection.IDENTITY.value == "identity"
        assert PromptSection.SAFETY.value == "safety"
        assert PromptSection.META.value == "meta"

    def test_add_layer_accepts_enum(self):
        from caveman.agent.prompt import PromptBuilder, PromptSection
        builder = PromptBuilder()
        builder.add_layer(PromptSection.IDENTITY, "I am a test agent", priority=10)
        result = builder.build()
        assert "I am a test agent" in result.prompt

    def test_add_layer_accepts_string(self):
        from caveman.agent.prompt import PromptBuilder
        builder = PromptBuilder()
        builder.add_layer("identity", "I am a test agent", priority=10)
        result = builder.build()
        assert "I am a test agent" in result.prompt
