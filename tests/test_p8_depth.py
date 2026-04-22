"""Tests for P8 depth: web_fetch, TTS, browser, processor, session_cmd, allowlist, ACP, memory, cmd_reg, directives."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Web Fetch Depth ──

class TestWebFetchDepth:
    def test_check_url_secrets(self):
        from caveman.tools.builtin.web_fetch_depth import check_url_secrets
        assert check_url_secrets("https://evil.com/?key=sk-abc12345678901234567") is True
        assert check_url_secrets("https://example.com/page") is False

    def test_robots_txt_cache(self):
        from caveman.tools.builtin.web_fetch_depth import _robots_cache
        _robots_cache.clear()
        assert len(_robots_cache) == 0

    def test_clean_base64_images(self):
        from caveman.tools.builtin.web_fetch_depth import clean_base64_images
        text = "before data:image/png;base64," + "A" * 200 + " after"
        cleaned = clean_base64_images(text)
        assert "[base64-image-removed]" in cleaned
        assert "A" * 200 not in cleaned

    def test_extract_readable_content(self):
        from caveman.tools.builtin.web_fetch_depth import extract_readable_content
        html = "<html><script>evil()</script><p>Hello world</p><nav>nav</nav></html>"
        text = extract_readable_content(html)
        assert "Hello world" in text
        assert "evil()" not in text
        assert "nav" not in text

    def test_cache_roundtrip(self):
        from caveman.tools.builtin.web_fetch_depth import get_cached, set_cached, _CACHE_DIR
        import tempfile
        import caveman.tools.builtin.web_fetch_depth as mod
        old_dir = mod._CACHE_DIR
        mod._CACHE_DIR = Path(tempfile.mkdtemp()) / "cache"
        try:
            set_cached("https://test.com", "hello world")
            assert get_cached("https://test.com") == "hello world"
        finally:
            mod._CACHE_DIR = old_dir

    def test_search_backends(self):
        from caveman.tools.builtin.web_fetch_depth import SEARCH_BACKENDS
        assert len(SEARCH_BACKENDS) >= 3

    @pytest.mark.asyncio
    async def test_parallel_extract_blocks_secrets(self):
        from caveman.tools.builtin.web_fetch_depth import parallel_extract
        results = await parallel_extract(["https://evil.com/?key=sk-abc12345678901234567"])
        assert results[0].get("error")
        assert "secret" in results[0]["error"].lower()


# ── TTS Depth ──

class TestTTSDepth:
    def test_strip_markdown(self):
        from caveman.tools.builtin.tts_depth import strip_markdown_for_tts
        md = "# Hello\n**bold** and `code`\n```python\nprint('hi')\n```\n- item"
        text = strip_markdown_for_tts(md)
        assert "#" not in text
        assert "**" not in text
        assert "```" not in text

    def test_provider_config(self):
        from caveman.tools.builtin.tts_depth import load_tts_config
        config = load_tts_config({"provider": "openai", "voice": "nova"})
        assert config.name == "openai"
        assert config.voice == "nova"

    def test_provider_default(self):
        from caveman.tools.builtin.tts_depth import load_tts_config
        config = load_tts_config()
        assert config.name == "edge"

    def test_has_ffmpeg(self):
        from caveman.tools.builtin.tts_depth import has_ffmpeg
        # Just check it doesn't crash
        result = has_ffmpeg()
        assert isinstance(result, bool)


# ── Browser Depth ──

class TestBrowserDepth:
    def test_vision_no_file(self):
        from caveman.tools.builtin.browser_depth import browser_vision
        result = browser_vision(Path("/nonexistent.png"), "what is this?")
        assert not result["success"]

    def test_vision_no_llm(self, tmp_path):
        from caveman.tools.builtin.browser_depth import browser_vision
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = browser_vision(img, "what is this?")
        assert result["success"]
        assert "not configured" in result["analysis"].lower()

    def test_save_screenshot(self, tmp_path):
        from caveman.tools.builtin.browser_depth import save_screenshot
        import caveman.tools.builtin.browser_depth as mod
        old_dir = mod._SCREENSHOTS_DIR
        mod._SCREENSHOTS_DIR = tmp_path / "screenshots"
        try:
            path = save_screenshot(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            assert path.exists()
        finally:
            mod._SCREENSHOTS_DIR = old_dir

    def test_recording_session(self):
        from caveman.tools.builtin.browser_depth import RecordingSession
        rec = RecordingSession(session_id="test-123")
        rec.start()
        assert rec.is_recording
        result = rec.stop()
        assert not rec.is_recording
        assert result["duration_seconds"] >= 0

    def test_camofox_profiles(self):
        from caveman.tools.builtin.browser_depth import get_camofox_profile, PRESET_PROFILES
        assert len(PRESET_PROFILES) >= 3
        profile = get_camofox_profile("mac-safari")
        assert "Safari" in profile.user_agent

    def test_redact_secrets(self):
        from caveman.tools.builtin.browser_depth import _redact_secrets
        text = "Found key: sk-abc1234567890123456789 in the page"
        assert "[REDACTED]" in _redact_secrets(text)


# ── Processor Depth ──

class TestProcessorDepth:
    def test_retry_config(self):
        from caveman.gateway.processor_depth import RetryConfig
        config = RetryConfig(max_retries=3, base_delay=1.0, backoff_factor=2.0)
        assert config.delay_for(0) == 1.0
        assert config.delay_for(1) == 2.0
        assert config.delay_for(2) == 4.0

    @pytest.mark.asyncio
    async def test_process_with_retry_success(self):
        from caveman.gateway.processor_depth import StreamingProcessor
        async def process(msg, ctx):
            return {"text": "ok"}
        proc = StreamingProcessor(process_fn=process)
        result = await proc.process_with_retry("hello")
        assert result["text"] == "ok"

    @pytest.mark.asyncio
    async def test_process_with_retry_exhausted(self):
        from caveman.gateway.processor_depth import StreamingProcessor, RetryConfig
        call_count = 0
        async def process(msg, ctx):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("429 rate limit")
        proc = StreamingProcessor(
            process_fn=process,
            retry_config=RetryConfig(max_retries=2, base_delay=0.01),
        )
        with pytest.raises(RuntimeError):
            await proc.process_with_retry("hello")
        assert call_count == 3  # 1 + 2 retries

    def test_tool_progress_event(self):
        from caveman.gateway.processor_depth import ToolProgressEvent
        event = ToolProgressEvent(tool_name="bash", status="running", started_at=time.monotonic())
        assert event.tool_name == "bash"


# ── Session Commands Depth ──

class TestSessionCommandsDepth:
    def test_export_import(self, tmp_path):
        from caveman.gateway.session_commands_depth import export_session, import_session
        messages = [{"role": "user", "content": "hello"}]
        path = export_session("test:session", messages, output_dir=tmp_path)
        assert path.exists()
        imported = import_session(path)
        assert imported.session_key == "test:session"
        assert len(imported.messages) == 1

    def test_search_sessions(self):
        from caveman.gateway.session_commands_depth import search_sessions
        sessions = [
            {"session_key": "discord:channel:123", "model": "claude-opus-4-6"},
            {"session_key": "telegram:chat:456", "model": "gpt-4o"},
        ]
        results = search_sessions(sessions, "discord")
        assert len(results) == 1
        assert "discord" in results[0]["session_key"]

    def test_bulk_delete_dry_run(self):
        from caveman.gateway.session_commands_depth import bulk_delete_sessions
        result = bulk_delete_sessions(["s1", "s2"], lambda k: None, dry_run=True)
        assert len(result["deleted"]) == 2
        assert result["dry_run"]


# ── Allowlist Commands Depth ──

class TestAllowlistCommandsDepth:
    def test_bulk_add(self):
        from caveman.gateway.allowlist_commands_depth import AllowlistRule, bulk_add
        existing = [AllowlistRule(pattern="user1")]
        new_rules = [AllowlistRule(pattern="user1"), AllowlistRule(pattern="user2")]
        result = bulk_add(new_rules, existing)
        assert "user1" in result["skipped"]
        assert "user2" in result["added"]

    def test_bulk_remove(self):
        from caveman.gateway.allowlist_commands_depth import AllowlistRule, bulk_remove
        existing = [AllowlistRule(pattern="a"), AllowlistRule(pattern="b"), AllowlistRule(pattern="c")]
        result = bulk_remove(["a", "c", "d"], existing)
        assert set(result["removed"]) == {"a", "c"}
        assert "d" in result["not_found"]
        assert len(existing) == 1

    def test_export_import(self):
        from caveman.gateway.allowlist_commands_depth import AllowlistRule, export_allowlist, import_allowlist
        rules = [AllowlistRule(pattern="user*", label="test")]
        exported = export_allowlist(rules)
        imported = import_allowlist(exported)
        assert len(imported) == 1
        assert imported[0].pattern == "user*"

    def test_expired_cleanup(self):
        from caveman.gateway.allowlist_commands_depth import AllowlistRule, cleanup_expired
        rules = [
            AllowlistRule(pattern="active"),
            AllowlistRule(pattern="expired", expires_at=time.time() - 100),
        ]
        result = cleanup_expired(rules)
        assert result["removed"] == 1
        assert result["remaining"] == 1


# ── ACP Lifecycle Depth ──

class TestACPLifecycleDepth:
    def test_thread_binding(self):
        from caveman.gateway.acp_lifecycle_depth import ThreadBindingStore
        store = ThreadBindingStore()
        binding = store.bind("thread-1", "session-1", agent_id="claude")
        assert binding.session_id == "session-1"
        got = store.get("thread-1")
        assert got is not None
        assert got.message_count == 1

    def test_thread_unbind(self):
        from caveman.gateway.acp_lifecycle_depth import ThreadBindingStore
        store = ThreadBindingStore()
        store.bind("t1", "s1")
        assert store.unbind("t1")
        assert not store.unbind("t1")

    def test_persistent_session(self, tmp_path):
        from caveman.gateway.acp_lifecycle_depth import PersistentSessionStore
        store = PersistentSessionStore(base_dir=tmp_path / "sessions")
        session = store.create("s1", "claude")
        assert session.is_active
        store.update_state("s1", "completed")
        assert not store.get("s1").is_active

    def test_cleanup_completed(self, tmp_path):
        from caveman.gateway.acp_lifecycle_depth import PersistentSessionStore
        store = PersistentSessionStore(base_dir=tmp_path / "sessions")
        s = store.create("s1", "claude")
        s.state = "completed"
        s.last_activity = time.time() - 100000
        store._sessions["s1"] = s
        removed = store.cleanup_completed(max_age_hours=1)
        assert removed == 1


# ── Agent Memory Depth ──

class TestAgentMemoryDepth:
    def test_estimate_tokens(self):
        from caveman.gateway.agent_memory_depth import estimate_tokens_for_model
        tokens = estimate_tokens_for_model("Hello world, this is a test.", "claude")
        assert 5 < tokens < 20

    def test_estimate_transcript(self):
        from caveman.gateway.agent_memory_depth import estimate_transcript_tokens
        messages = [
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "World " * 100},
        ]
        tokens = estimate_transcript_tokens(messages, "gpt-4o")
        assert tokens > 50

    def test_should_compact(self):
        from caveman.gateway.agent_memory_depth import should_compact
        # Small transcript should not compact
        messages = [{"role": "user", "content": "hi"}]
        assert not should_compact(messages, "claude-opus-4-6")

    def test_context_window(self):
        from caveman.gateway.agent_memory_depth import get_context_window
        assert get_context_window("claude-opus-4-6") == 200_000
        assert get_context_window("gemini-2.5-pro") == 1_000_000

    def test_flush_and_load(self, tmp_path):
        from caveman.gateway.agent_memory_depth import flush_transcript, load_transcript, MemoryFlushConfig
        config = MemoryFlushConfig(base_dir=tmp_path / "flush")
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        path = flush_transcript("test:session", messages, config)
        assert path is not None
        loaded = load_transcript("test:session", config)
        assert len(loaded) == 2

    def test_prepare_compaction(self):
        from caveman.gateway.agent_memory_depth import prepare_compaction
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = prepare_compaction(messages, preserve_recent=5)
        assert result["needs_compaction"]
        assert len(result["to_preserve"]) == 5
        assert len(result["to_summarize"]) == 15


# ── Command Registry Depth ──

class TestCommandRegistryDepth:
    def test_register_and_resolve(self):
        from caveman.gateway.command_registry_depth import EnhancedCommandRegistry, EnhancedCommand
        reg = EnhancedCommandRegistry()
        reg.register(EnhancedCommand(name="test", aliases=["t"], description="Test command"))
        assert reg.resolve("test") is not None
        assert reg.resolve("t") is not None
        assert reg.resolve("unknown") is None

    def test_permission_check(self):
        from caveman.gateway.command_registry_depth import CommandPermission
        perm = CommandPermission(level="admin")
        assert not perm.check("user1", "user")
        assert perm.check("user1", "admin")
        assert perm.check("user1", "owner")

    def test_permission_allowlist(self):
        from caveman.gateway.command_registry_depth import CommandPermission
        perm = CommandPermission(level="admin", allowlist={"special-user"})
        assert perm.check("special-user", "user")

    def test_cooldown(self):
        from caveman.gateway.command_registry_depth import CommandCooldown
        cd = CommandCooldown(per_user_seconds=10)
        assert cd.check("user1") is None
        cd.record("user1")
        remaining = cd.check("user1")
        assert remaining is not None and remaining > 0

    def test_generate_help(self):
        from caveman.gateway.command_registry_depth import EnhancedCommandRegistry, EnhancedCommand
        reg = EnhancedCommandRegistry()
        reg.register(EnhancedCommand(name="test", description="A test", category="general"))
        reg.register(EnhancedCommand(name="secret", description="Hidden", hidden=True))
        help_text = reg.generate_help()
        assert "/test" in help_text
        assert "secret" not in help_text


# ── Directives Depth ──

class TestDirectivesDepth:
    def test_parse_extended(self):
        from caveman.gateway.directives_depth import parse_extended_directives
        text = "/help\n/config set model opus\nHello world"
        directives, remaining = parse_extended_directives(text)
        assert len(directives) == 2
        assert directives[0].name == "help"
        assert directives[1].name == "config"
        assert directives[1].args == "set model opus"
        assert remaining == "Hello world"

    def test_approve_directive(self):
        from caveman.gateway.directives_depth import ApproveDirective
        d = ApproveDirective.parse("abc123 allow-always")
        assert d.command_hash == "abc123"
        assert d.policy == "allow-always"

    def test_config_directive(self):
        from caveman.gateway.directives_depth import ConfigDirective
        d = ConfigDirective.parse("set model claude-opus-4-6")
        assert d.action == "set"
        assert d.key == "model"
        assert d.value == "claude-opus-4-6"

    def test_approval_store(self):
        from caveman.gateway.directives_depth import ApprovalStore, ApproveDirective
        store = ApprovalStore()
        store.add(ApproveDirective(command_hash="h1", policy="allow-once"))
        store.add(ApproveDirective(command_hash="h2", policy="allow-always"))
        # allow-once consumed after first check
        assert store.check("h1") is not None
        assert store.check("h1") is None
        # allow-always persists
        assert store.check("h2") is not None
        assert store.check("h2") is not None

    def test_generate_help(self):
        from caveman.gateway.directives_depth import generate_help
        text = generate_help()
        assert "/model" in text
        assert "/approve" in text
