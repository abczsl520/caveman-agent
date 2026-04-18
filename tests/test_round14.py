"""Tests for Round 14 — Compaction Safeguard + Prompt Cache + Recall upgrade."""
from __future__ import annotations

import time
import copy
import pytest

from caveman.compression.safeguard import (
    CompactionSafeguard,
    SafeguardPhase,
    SafeguardState,
    extract_identifiers,
    audit_summary,
    build_compaction_instructions,
    REQUIRED_SECTIONS,
)
from caveman.providers.prompt_cache import apply_cache_control
from caveman.engines.recall import RecallEngine, RecallResult


# --- Compaction Safeguard ---

class TestSafeguardState:
    def test_idle_not_critical(self):
        s = SafeguardState()
        assert not s.is_critical

    def test_tool_executing_is_critical(self):
        s = SafeguardState(phase=SafeguardPhase.TOOL_EXECUTING)
        assert s.is_critical

    def test_thinking_not_critical(self):
        s = SafeguardState(phase=SafeguardPhase.THINKING)
        assert not s.is_critical

    def test_phase_duration(self):
        s = SafeguardState(phase_started_at=time.time() - 30)
        assert 29 <= s.phase_duration <= 31

    def test_should_force_after_timeout(self):
        s = SafeguardState(
            phase=SafeguardPhase.TOOL_EXECUTING,
            phase_started_at=time.time() - 120,
            max_wait_seconds=60.0,
        )
        assert s.should_force

    def test_no_force_within_timeout(self):
        s = SafeguardState(
            phase=SafeguardPhase.TOOL_EXECUTING,
            phase_started_at=time.time() - 10,
            max_wait_seconds=60.0,
        )
        assert not s.should_force


class TestCompactionSafeguard:
    def test_allow_when_idle(self):
        sg = CompactionSafeguard()
        assert sg.request_compaction()

    def test_defer_when_critical(self):
        sg = CompactionSafeguard()
        sg.enter_phase(SafeguardPhase.TOOL_EXECUTING)
        assert not sg.request_compaction()
        assert sg.has_pending()

    def test_force_after_timeout(self):
        sg = CompactionSafeguard(max_wait=0.01)
        sg.enter_phase(SafeguardPhase.TOOL_EXECUTING)
        time.sleep(0.02)
        assert sg.request_compaction()

    def test_pending_clears_on_completion(self):
        sg = CompactionSafeguard()
        sg.enter_phase(SafeguardPhase.TOOL_EXECUTING)
        sg.request_compaction()
        assert sg.has_pending()
        sg.compaction_completed()
        assert not sg.has_pending()

    def test_compaction_count(self):
        sg = CompactionSafeguard()
        assert sg.state.compaction_count == 0
        sg.compaction_completed()
        sg.compaction_completed()
        assert sg.state.compaction_count == 2

    def test_phase_transitions(self):
        sg = CompactionSafeguard()
        sg.enter_phase(SafeguardPhase.CODE_GENERATING)
        assert sg.state.phase == SafeguardPhase.CODE_GENERATING
        sg.enter_phase(SafeguardPhase.IDLE)
        assert sg.state.phase == SafeguardPhase.IDLE

    def test_all_critical_phases(self):
        sg = CompactionSafeguard()
        for phase in [
            SafeguardPhase.TOOL_EXECUTING,
            SafeguardPhase.CODE_GENERATING,
            SafeguardPhase.TESTING,
            SafeguardPhase.FILE_WRITING,
            SafeguardPhase.COMPACTING,
        ]:
            sg.enter_phase(phase)
            assert not sg.request_compaction()
            sg._state.pending_compaction = False  # Reset for next test


class TestExtractIdentifiers:
    def test_git_hash(self):
        ids = extract_identifiers("commit e59d0d0 and ba8396b")
        assert "e59d0d0" in ids
        assert "ba8396b" in ids

    def test_url(self):
        ids = extract_identifiers("see https://example.com/path")
        assert any("example.com" in i for i in ids)

    def test_ip_address(self):
        ids = extract_identifiers("server at 192.168.1.100")
        assert "192.168.1.100" in ids

    def test_file_path(self):
        ids = extract_identifiers("edit /src/main.py")
        assert "/src/main.py" in ids

    def test_version(self):
        ids = extract_identifiers("version v0.3.0")
        assert "v0.3.0" in ids

    def test_max_limit(self):
        text = " ".join(f"hash{i:07x}" for i in range(100))
        ids = extract_identifiers(text, max_ids=10)
        assert len(ids) <= 10


class TestAuditSummary:
    def test_complete_summary(self):
        summary = "\n".join(REQUIRED_SECTIONS) + "\n## Exact identifiers\n"
        passed, missing = audit_summary(summary)
        assert passed
        assert missing == []

    def test_missing_sections(self):
        summary = "## Goal\nSome content"
        passed, missing = audit_summary(summary)
        assert not passed
        assert len(missing) > 0

    def test_partial_summary(self):
        summary = "## Goal\n## Key Decisions\nSome content"
        passed, missing = audit_summary(summary)
        assert not passed
        assert "## Remaining Work" in missing


class TestBuildInstructions:
    def test_default(self):
        instructions = build_compaction_instructions()
        assert "## Goal" in instructions
        assert "## Key Decisions" in instructions
        assert "Exact identifiers" in instructions

    def test_with_custom(self):
        instructions = build_compaction_instructions(custom="Focus on API changes")
        assert "API changes" in instructions


# --- Prompt Cache ---

class TestPromptCache:
    def test_empty_messages(self):
        result = apply_cache_control([])
        assert result == []

    def test_system_gets_cache(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_cache_control(msgs)
        # System message content should be wrapped with cache_control
        sys_content = result[0]["content"]
        if isinstance(sys_content, list):
            assert "cache_control" in sys_content[-1]
        else:
            assert "cache_control" in result[0]

    def test_last_3_non_system(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        result = apply_cache_control(msgs)
        # Should have 4 breakpoints total (1 system + 3 last)
        cache_count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cache_count += 1
            elif "cache_control" in msg:
                cache_count += 1
        assert cache_count == 4

    def test_deep_copy(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = apply_cache_control(msgs)
        # Original should be unchanged
        assert isinstance(msgs[0]["content"], str)

    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "sys"}]
        result = apply_cache_control(msgs, cache_ttl="1h")
        content = result[0]["content"]
        if isinstance(content, list):
            assert content[-1]["cache_control"]["ttl"] == "1h"

    def test_no_system_message(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        result = apply_cache_control(msgs)
        # Should still work — 4 breakpoints on last 4 messages
        cache_count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cache_count += 1
        assert cache_count == 4


# --- Recall Upgrade ---

class TestRecallResult:
    def test_empty(self):
        r = RecallResult()
        assert not r.has_context
        assert r.as_prompt_text() == ""

    def test_with_essence(self):
        r = RecallResult(essence_text="## Previous Session\nDid X")
        assert r.has_context
        assert "Previous Session" in r.as_prompt_text()

    def test_with_both(self):
        r = RecallResult(
            essence_text="## Session\nA",
            memory_text="## Memories\nB",
        )
        text = r.as_prompt_text()
        assert "Session" in text
        assert "Memories" in text

    def test_token_estimate(self):
        r = RecallResult(
            essence_text="x" * 400,
            memory_text="y" * 400,
            total_tokens_est=200,
        )
        assert r.total_tokens_est == 200


class TestRecallEngine:
    @pytest.mark.asyncio
    async def test_empty_restore(self, tmp_path):
        engine = RecallEngine(sessions_dir=tmp_path / "sessions")
        result = await engine.restore("test task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_structured_restore_empty(self, tmp_path):
        engine = RecallEngine(sessions_dir=tmp_path / "sessions")
        result = await engine.restore_structured("test task")
        assert not result.has_context
        assert result.essences_loaded == 0

    @pytest.mark.asyncio
    async def test_has_previous_sessions_false(self, tmp_path):
        engine = RecallEngine(sessions_dir=tmp_path / "sessions")
        assert not await engine.has_previous_sessions()

    @pytest.mark.asyncio
    async def test_has_previous_sessions_true(self, tmp_path):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        (sessions / "test.yaml").write_text("summary: test\n")
        engine = RecallEngine(sessions_dir=sessions)
        assert await engine.has_previous_sessions()

    @pytest.mark.asyncio
    async def test_budget_params(self):
        engine = RecallEngine(essence_budget=1000, memory_budget=500)
        assert engine._essence_budget == 1000
        assert engine._memory_budget == 500
