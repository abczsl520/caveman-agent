"""Tests for session mirroring."""
import json
from pathlib import Path

import pytest

from caveman.gateway.mirror import mirror_to_session, _find_session_id


@pytest.fixture
def sessions_dir(tmp_path):
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    return sdir


def _create_session_index(sessions_dir, entries):
    """Create a sessions.json with given entries."""
    (sessions_dir / "sessions.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


class TestFindSessionId:
    def test_finds_matching_session(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "abc-123",
                "origin": {"platform": "discord", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        assert _find_session_id("discord", "12345", sessions_dir=sessions_dir) == "abc-123"

    def test_no_match(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "abc-123",
                "origin": {"platform": "telegram", "chat_id": "99999"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        assert _find_session_id("discord", "12345", sessions_dir=sessions_dir) is None

    def test_picks_most_recent(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "old-one",
                "origin": {"platform": "discord", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "s2": {
                "session_id": "new-one",
                "origin": {"platform": "discord", "chat_id": "123"},
                "updated_at": "2026-04-20T00:00:00",
            },
        })
        assert _find_session_id("discord", "123", sessions_dir=sessions_dir) == "new-one"

    def test_thread_id_filter(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "thread-session",
                "origin": {"platform": "discord", "chat_id": "123", "thread_id": "t1"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "s2": {
                "session_id": "main-session",
                "origin": {"platform": "discord", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            },
        })
        assert _find_session_id("discord", "123", thread_id="t1", sessions_dir=sessions_dir) == "thread-session"

    def test_no_index_file(self, sessions_dir):
        assert _find_session_id("discord", "123", sessions_dir=sessions_dir) is None

    def test_case_insensitive_platform(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "abc",
                "origin": {"platform": "Discord", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        assert _find_session_id("discord", "123", sessions_dir=sessions_dir) == "abc"


class TestMirrorToSession:
    def test_successful_mirror(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "test-session",
                "origin": {"platform": "telegram", "chat_id": "456"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        result = mirror_to_session("telegram", "456", "Hello from CLI", sessions_dir=sessions_dir)
        assert result is True

        # Verify JSONL was written
        jsonl_path = sessions_dir / "test-session.jsonl"
        assert jsonl_path.exists()
        entry = json.loads(jsonl_path.read_text().strip())
        assert entry["content"] == "Hello from CLI"
        assert entry["mirror"] is True
        assert entry["mirror_source"] == "cli"

    def test_no_matching_session(self, sessions_dir):
        _create_session_index(sessions_dir, {})
        result = mirror_to_session("discord", "999", "Hello", sessions_dir=sessions_dir)
        assert result is False

    def test_custom_source_label(self, sessions_dir):
        _create_session_index(sessions_dir, {
            "s1": {
                "session_id": "test-session",
                "origin": {"platform": "discord", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        mirror_to_session("discord", "123", "Cron message", source_label="cron", sessions_dir=sessions_dir)
        entry = json.loads((sessions_dir / "test-session.jsonl").read_text().strip())
        assert entry["mirror_source"] == "cron"
