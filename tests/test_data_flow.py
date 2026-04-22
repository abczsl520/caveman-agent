"""Tests for tool result persistence and ephemeral messages."""
import pytest
from pathlib import Path


class TestEphemeralMessages:
    """Test that ephemeral messages are excluded from persistence."""

    def test_add_ephemeral_message(self):
        from caveman.agent.context import AgentContext
        ctx = AgentContext(max_tokens=100000)
        ctx.add_message("user", "hello")
        ctx.add_message("system", "[Format: Discord]", ephemeral=True)
        ctx.add_message("assistant", "hi")
        assert len(ctx.messages) == 3

    def test_persistable_excludes_ephemeral(self):
        from caveman.agent.context import AgentContext
        ctx = AgentContext(max_tokens=100000)
        ctx.add_message("user", "hello")
        ctx.add_message("system", "[Format: Discord]", ephemeral=True)
        ctx.add_message("assistant", "hi")
        persistable = ctx.persistable_messages()
        assert len(persistable) == 2
        assert all(not m.ephemeral for m in persistable)
        assert persistable[0].content == "hello"
        assert persistable[1].content == "hi"

    def test_to_api_format_includes_ephemeral(self):
        from caveman.agent.context import AgentContext
        ctx = AgentContext(max_tokens=100000)
        ctx.add_message("user", "hello")
        ctx.add_message("system", "format hint", ephemeral=True)
        api = ctx.to_api_format()
        assert len(api) == 2
        assert api[1]["content"] == "format hint"


class TestToolResultPersistence:
    """Test tool result storage and budget enforcement."""

    def test_small_result_unchanged(self):
        from caveman.tools.result_storage import persist_tool_result
        result = persist_tool_result("short output", "bash", "id1")
        assert result == "short output"

    def test_large_result_persisted(self, tmp_path, monkeypatch):
        from caveman.tools import result_storage
        monkeypatch.setattr(result_storage, "STORAGE_DIR", tmp_path)
        large = "x" * 50_000
        result = result_storage.persist_tool_result(large, "bash", "test_id", threshold=30_000)
        assert "<persisted-output>" in result
        assert "test_id.txt" in result
        assert (tmp_path / "test_id.txt").exists()
        assert (tmp_path / "test_id.txt").read_text() == large

    def test_preview_size(self, tmp_path, monkeypatch):
        from caveman.tools import result_storage
        monkeypatch.setattr(result_storage, "STORAGE_DIR", tmp_path)
        large = "line\n" * 20_000
        result = result_storage.persist_tool_result(
            large, "bash", "preview_test", threshold=1000, preview_size=100
        )
        assert "<persisted-output>" in result
        assert len(result) < len(large)

    def test_turn_budget_enforcement(self, tmp_path, monkeypatch):
        from caveman.tools import result_storage
        monkeypatch.setattr(result_storage, "STORAGE_DIR", tmp_path)
        results = [
            {"content": "x" * 100_000, "tool_use_id": "big1"},
            {"content": "small", "tool_use_id": "small1"},
            {"content": "y" * 150_000, "tool_use_id": "big2"},
        ]
        result_storage.enforce_turn_budget(results, budget=50_000)
        # At least one should be persisted
        persisted = sum(1 for r in results if "<persisted-output>" in r["content"])
        assert persisted >= 1

    def test_cleanup_old_results(self, tmp_path, monkeypatch):
        import time
        from caveman.tools import result_storage
        monkeypatch.setattr(result_storage, "STORAGE_DIR", tmp_path)
        # Create a file with old mtime
        old_file = tmp_path / "old.txt"
        old_file.write_text("old")
        import os
        os.utime(old_file, (time.time() - 100_000, time.time() - 100_000))
        new_file = tmp_path / "new.txt"
        new_file.write_text("new")
        removed = result_storage.cleanup_old_results(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_generate_preview_truncates_at_newline(self):
        from caveman.tools.result_storage import _generate_preview
        content = "line1\nline2\nline3\nline4\nline5"
        preview, has_more = _generate_preview(content, max_chars=15)
        assert has_more
        assert preview.endswith("\n")
        assert len(preview) <= 15
