"""Tests for P4 modules: dispatch, agent_memory, directives, status_panel,
browser_v2, skills_hub, tts_v2, web_fetch_v2."""
from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Dispatch Tests ──

class TestDispatch:
    def test_send_policy_allow(self):
        from caveman.gateway.dispatch import resolve_send_policy
        policy = resolve_send_policy({}, "s1", "c1")
        assert policy.allow

    def test_send_policy_deny_channel(self):
        from caveman.gateway.dispatch import resolve_send_policy
        policy = resolve_send_policy({"send_policies": {"channel:c1": "deny"}}, "s1", "c1")
        assert not policy.allow

    def test_send_policy_deny_type(self):
        from caveman.gateway.dispatch import resolve_send_policy
        policy = resolve_send_policy({"send_policies": {"type:group": "deny"}}, "s1", "c1", "group")
        assert not policy.allow

    @pytest.mark.asyncio
    async def test_dispatch_with_agent(self):
        from caveman.gateway.dispatch import MessageDispatcher, DispatchContext
        agent_fn = AsyncMock(return_value={"text": "hello", "tool_calls": 1})
        send_fn = AsyncMock()
        dispatcher = MessageDispatcher({}, agent_fn=agent_fn, send_fn=send_fn)
        ctx = DispatchContext(session_key="s1", body="hi")
        result = await dispatcher.dispatch(ctx)
        assert result.ok
        assert result.reply_queued
        send_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_before_hook_handled(self):
        from caveman.gateway.dispatch import MessageDispatcher, DispatchContext, HookRunner
        hooks = HookRunner()
        hooks.add_before_hook(lambda ctx: {"handled": True, "text": "hooked"})
        send_fn = AsyncMock()
        dispatcher = MessageDispatcher({}, hook_runner=hooks, send_fn=send_fn)
        ctx = DispatchContext(session_key="s1")
        result = await dispatcher.dispatch(ctx)
        assert result.reason == "before_dispatch_handled"

    @pytest.mark.asyncio
    async def test_dispatch_policy_deny(self):
        from caveman.gateway.dispatch import MessageDispatcher, DispatchContext
        dispatcher = MessageDispatcher({"send_policies": {"session:s1": "deny"}})
        ctx = DispatchContext(session_key="s1")
        result = await dispatcher.dispatch(ctx)
        assert not result.ok


# ── Agent Memory Tests ──

class TestAgentMemory:
    def test_token_budget(self):
        from caveman.gateway.agent_memory import TokenBudget
        budget = TokenBudget(context_window=200000, reserve_floor=20000)
        assert budget.available == 180000
        budget.update(150000, 10000)
        assert budget.usage_ratio == 0.8

    def test_should_compact(self):
        from caveman.gateway.agent_memory import TokenBudget
        budget = TokenBudget(context_window=200000, reserve_floor=20000, soft_threshold=4000)
        budget.update(180000, 0)
        assert budget.should_compact

    def test_resolve_context_window(self):
        from caveman.gateway.agent_memory import resolve_context_window
        assert resolve_context_window("claude-opus-4-6") == 200000
        assert resolve_context_window("gpt-4o") == 128000
        assert resolve_context_window("gemini-2.5-pro") == 1000000
        assert resolve_context_window("unknown-model") == 200000

    @pytest.mark.asyncio
    async def test_check_compaction(self):
        from caveman.gateway.agent_memory import AgentMemoryManager
        compact_fn = AsyncMock(return_value={"tokens_after": 50000})
        mgr = AgentMemoryManager(compact_fn=compact_fn)
        budget = mgr.get_budget("s1", "claude-opus-4-6")
        budget.update(190000, 0)  # Over threshold
        result = await mgr.check_compaction("s1")
        assert result
        compact_fn.assert_called_once()

    def test_get_stats(self):
        from caveman.gateway.agent_memory import AgentMemoryManager
        mgr = AgentMemoryManager()
        mgr.get_budget("s1", "gpt-4o")
        mgr.update_usage("s1", 1000, 500)
        stats = mgr.get_stats()
        assert stats["sessions"] == 1
        assert stats["total_tokens"] == 1500


# ── Directives Tests ──

class TestDirectives:
    def test_parse_model(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("/model claude-opus-4-6 hello")
        assert result.model_override == "claude-opus-4-6"
        assert "hello" in result.cleaned

    def test_parse_reasoning(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("/reasoning on tell me about X")
        assert result.reasoning_mode == "on"

    def test_parse_status(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("/status")
        assert result.has_status_directive

    def test_parse_reset(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("/reset")
        assert result.has_reset_directive

    def test_parse_skill(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("/skill weather London")
        assert result.skill_name == "weather"
        assert result.skill_input == "London"

    def test_resolve_with_alias(self):
        from caveman.gateway.directives import resolve_directives
        result = resolve_directives(
            "/model opus hello",
            default_model="gpt-4o",
            model_aliases={"opus": "anthropic/claude-opus-4-6"},
        )
        assert result.provider == "anthropic"
        assert result.model == "claude-opus-4-6"

    def test_resolve_group_no_mention(self):
        from caveman.gateway.directives import resolve_directives
        result = resolve_directives(
            "/model opus hello",
            default_model="gpt-4o",
            model_aliases={"opus": "anthropic/claude-opus-4-6"},
            is_group=True,
            was_mentioned=False,
        )
        # Should revert to default in group without mention
        assert result.model == "gpt-4o"

    def test_parse_not_command(self):
        from caveman.gateway.directives import parse_inline_directives
        result = parse_inline_directives("just a normal message")
        assert not result.model_override
        assert not result.has_status_directive


# ── Status Panel Tests ──

class TestStatusPanel:
    def test_build_system_status(self):
        from caveman.gateway.status_panel import build_system_status
        status = build_system_status(
            platforms=["discord", "telegram"],
            tools_count=42,
            start_time=time.monotonic() - 3600,
        )
        assert "discord" in status.connected_platforms
        assert status.tools_count == 42
        assert status.uptime_seconds > 0

    def test_format_text(self):
        from caveman.gateway.status_panel import (
            build_system_status, format_status_text, SessionStatus,
        )
        system = build_system_status(platforms=["discord"])
        session = SessionStatus(session_key="s1", model="claude-opus-4-6", total_tokens=5000)
        text = format_status_text(system, session)
        assert "Caveman" in text
        assert "claude-opus-4-6" in text

    def test_format_embed(self):
        from caveman.gateway.status_panel import build_system_status, format_status_embed
        system = build_system_status()
        embed = format_status_embed(system)
        assert embed["title"].startswith("🦴")
        assert isinstance(embed["fields"], list)


# ── Browser v2 Tests ──

class TestBrowserV2:
    def test_truncate_snapshot(self):
        from caveman.tools.builtin.browser_v2 import _truncate_snapshot
        short = "hello"
        assert _truncate_snapshot(short) == short
        long_text = "x" * 20000
        truncated = _truncate_snapshot(long_text, 8000)
        assert len(truncated) < 20000
        assert "truncated" in truncated

    def test_extract_relevant(self):
        from caveman.tools.builtin.browser_v2 import _extract_relevant_content
        snapshot = "heading: Title\nbutton: Click me\nparagraph: Some text\nlink: Go here"
        result = _extract_relevant_content(snapshot, "click")
        assert "Click me" in result

    def test_cleanup_inactive(self):
        from caveman.tools.builtin.browser_v2 import _sessions, _cleanup_inactive_sessions
        _sessions.clear()
        assert _cleanup_inactive_sessions() == 0


# ── Skills Hub Tests ──

class TestSkillsHub:
    def test_skill_meta(self):
        from caveman.tools.builtin.skills_hub import SkillMeta
        meta = SkillMeta(name="test", description="A test skill", tags=["test"])
        assert meta.name == "test"

    def test_skill_bundle_hash(self):
        from caveman.tools.builtin.skills_hub import SkillBundle, SkillMeta
        bundle = SkillBundle(
            meta=SkillMeta(name="test"),
            files={"SKILL.md": "# Test\nA test skill"},
        )
        h1 = bundle.compute_hash()
        assert len(h1) == 16
        # Same content = same hash
        bundle2 = SkillBundle(
            meta=SkillMeta(name="test"),
            files={"SKILL.md": "# Test\nA test skill"},
        )
        assert bundle2.compute_hash() == h1

    def test_local_source(self, tmp_path):
        from caveman.tools.builtin.skills_hub import LocalSource, SkillMeta
        # Create a test skill
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test\nA test skill for testing")
        source = LocalSource([str(tmp_path)])
        results = source.search("test")
        assert len(results) == 1
        assert results[0].name == "test_skill"

    def test_local_source_fetch(self, tmp_path):
        from caveman.tools.builtin.skills_hub import LocalSource
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")
        source = LocalSource([str(tmp_path)])
        bundle = source.fetch("my_skill")
        assert bundle is not None
        assert "SKILL.md" in bundle.files

    def test_hub_lock_file(self, tmp_path):
        from caveman.tools.builtin.skills_hub import HubLockFile, SkillMeta
        lock = HubLockFile(tmp_path / "test.lock")
        meta = SkillMeta(name="test", version="1.0")
        lock.record_install(meta, "abc123")
        installed = lock.get_installed("test")
        assert installed is not None
        assert installed["version"] == "1.0"
        lock.record_uninstall("test")
        assert lock.get_installed("test") is None

    def test_unified_search(self, tmp_path):
        from caveman.tools.builtin.skills_hub import unified_search, LocalSource
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Weather\nGet weather forecasts")
        sources = [LocalSource([str(tmp_path)])]
        results = unified_search("weather", sources)
        assert len(results) == 1


# ── TTS v2 Tests ──

class TestTTSV2:
    def test_cache_key(self):
        from caveman.tools.builtin.tts_v2 import _cache_key
        k1 = _cache_key("hello", "system", "")
        k2 = _cache_key("hello", "openai", "")
        assert k1 != k2

    def test_check_requirements(self):
        from caveman.tools.builtin.tts_v2 import check_tts_requirements
        reqs = check_tts_requirements()
        assert isinstance(reqs, dict)
        assert "system" in reqs


# ── Web Fetch v2 Tests ──

class TestWebFetchV2:
    def test_web_result(self):
        from caveman.tools.builtin.web_fetch_v2 import WebResult
        r = WebResult(ok=True, url="https://example.com", content="hello")
        assert r.ok

    def test_cache_key(self):
        from caveman.tools.builtin.web_fetch_v2 import _cache_key
        k1 = _cache_key("https://a.com")
        k2 = _cache_key("https://b.com")
        assert k1 != k2
        assert len(k1) == 16
