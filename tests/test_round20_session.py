"""Tests for Round 20 Phase 3 — Session Store."""
from __future__ import annotations

import pytest
from caveman.agent.session_store import SessionStore, SessionMeta


class TestSessionMeta:
    def test_to_dict_roundtrip(self):
        meta = SessionMeta(
            session_id="s1", model="claude-opus-4-6",
            turn_count=10, total_tokens=5000, title="Test Session",
        )
        d = meta.to_dict()
        restored = SessionMeta.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.model == "claude-opus-4-6"
        assert restored.turn_count == 10


class TestSessionStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SessionStore(tmp_path / "sessions")

    def test_save_load_meta(self, store):
        meta = SessionMeta(session_id="s1", model="test", turn_count=5, title="Hello")
        store.save_meta(meta)
        loaded = store.load_meta("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.title == "Hello"

    def test_load_meta_nonexistent(self, store):
        assert store.load_meta("nonexistent") is None

    def test_list_sessions(self, store):
        store.save_meta(SessionMeta(session_id="s1", last_active_at=100))
        store.save_meta(SessionMeta(session_id="s2", last_active_at=200))
        sessions = store.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id == "s2"  # newest first

    def test_append_and_load_transcript(self, store):
        store.append_turn("s1", "user", "Hello")
        store.append_turn("s1", "assistant", "Hi there!")
        transcript = store.load_transcript("s1")
        assert len(transcript) == 2
        assert transcript[0]["role"] == "user"
        assert transcript[1]["content"] == "Hi there!"

    def test_transcript_turn_count(self, store):
        store.append_turn("s1", "user", "Hello")
        store.append_turn("s1", "assistant", "Hi")
        store.append_turn("s1", "user", "How are you?")
        assert store.transcript_turn_count("s1") == 3

    def test_empty_transcript(self, store):
        assert store.load_transcript("empty") == []
        assert store.transcript_turn_count("empty") == 0

    def test_save_load_compaction(self, store):
        store.save_compaction("s1", "Summary of first 10 turns", turns_compressed=10)
        store.save_compaction("s1", "Summary of next 10 turns", turns_compressed=10)
        compactions = store.load_compactions("s1")
        assert len(compactions) == 2
        assert compactions[0]["turns_compressed"] == 10

    def test_delete_session(self, store):
        store.save_meta(SessionMeta(session_id="s1"))
        store.append_turn("s1", "user", "test")
        assert store.delete_session("s1")
        assert store.load_meta("s1") is None
        assert not store.delete_session("s1")  # already deleted

    def test_unicode_content(self, store):
        store.append_turn("s1", "user", "你好世界 🦴")
        transcript = store.load_transcript("s1")
        assert transcript[0]["content"] == "你好世界 🦴"

    def test_extra_fields(self, store):
        store.append_turn("s1", "assistant", "Done", tool_calls=["bash"])
        transcript = store.load_transcript("s1")
        assert transcript[0]["tool_calls"] == ["bash"]
