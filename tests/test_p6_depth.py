"""Tests for P6 depth: enhanced preflight, MCP sampling, MCP lifecycle."""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from caveman.gateway.platform_types import MessageEvent, MessageType, SessionSource


# ── Enhanced Preflight Tests ──

class TestPreflightDepth:
    def _make_event(self, text="hello", user_id="u1", chat_type="dm",
                    chat_id="c1", thread_id="", reply_to="", reply_text="",
                    media_urls=None, media_types=None):
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform="discord", chat_id=chat_id, chat_type=chat_type,
                user_id=user_id, thread_id=thread_id,
            ),
            media_urls=media_urls or [],
            media_types=media_types or [],
            reply_to_message_id=reply_to,
            reply_to_text=reply_text,
        )

    def test_system_event_filtered(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event()
        event.message_type = "member_join"
        result = pf.check(event)
        assert not result.passed
        assert result.drop_reason == "system_event"

    def test_pluralkit_allowed(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(
            bot_user_id="bot1",
            pluralkit_bot_ids={"pk_bot"},
        ))
        event = self._make_event(user_id="pk_bot")
        event.source.is_bot = True
        result = pf.check(event)
        assert result.passed
        assert result.is_pluralkit

    def test_mention_by_display_name(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(
            bot_user_id="bot1",
            bot_display_name="Caveman",
        ))
        event = self._make_event(text="hey Caveman what's up", chat_type="group")
        result = pf.check(event)
        assert result.was_mentioned
        assert result.mention_kind == "explicit"

    def test_mention_by_reply(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event(
            text="thanks", reply_to="msg1", reply_text="here's the answer",
        )
        # reply detection needs _is_bot_reply to return True
        # For now just verify the quoted message is parsed
        result = pf.check(event)
        assert result.quoted is not None
        assert result.quoted.message_id == "msg1"

    def test_implicit_mention_in_dm(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(
            bot_user_id="bot1",
            implicit_mention_in_dm=True,
        ))
        event = self._make_event(text="hello", chat_type="dm")
        result = pf.check(event)
        assert result.was_mentioned
        assert result.mention_kind == "implicit"

    def test_implicit_mention_in_thread(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(
            bot_user_id="bot1",
            implicit_mention_in_thread=True,
        ))
        event = self._make_event(text="hello", chat_type="group", thread_id="t1")
        result = pf.check(event)
        assert result.was_mentioned
        assert result.is_thread

    def test_command_detection_with_args(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event(text="/model claude-opus-4-6")
        result = pf.check(event)
        assert result.command == "model"
        assert result.command_args == "claude-opus-4-6"

    def test_thread_starter_cache(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        pf.set_thread_starter("t1", "user_abc")
        event = self._make_event(text="hello", thread_id="t1", chat_type="group")
        result = pf.check(event)
        assert result.thread_starter_id == "user_abc"

    def test_forum_detection(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event(text="hello", chat_type="group")
        event.source.chat_topic = "Some Topic"
        result = pf.check(event)
        assert result.is_forum

    def test_enriched_text(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event(text="hello world")
        result = pf.check(event)
        assert result.enriched_text == "hello world"


# ── MCP Sampling Tests ──

class TestMCPSampling:
    def test_rate_limit(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler, SamplingConfig
        handler = SamplingHandler("test", SamplingConfig(max_rpm=2))
        assert handler.check_rate_limit()
        assert handler.check_rate_limit()
        assert not handler.check_rate_limit()

    def test_resolve_model_override(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler, SamplingConfig
        handler = SamplingHandler("test", SamplingConfig(model_override="gpt-4o"))
        assert handler.resolve_model() == "gpt-4o"

    def test_resolve_model_no_override(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler, SamplingConfig
        handler = SamplingHandler("test", SamplingConfig())
        assert handler.resolve_model() == ""

    def test_tool_loop_governance(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler, SamplingConfig
        handler = SamplingHandler("test", SamplingConfig(max_tool_rounds=2))
        assert handler.check_tool_loop() is None  # 1st
        assert handler.check_tool_loop() is None  # 2nd
        err = handler.check_tool_loop()  # 3rd = exceeded
        assert err is not None
        assert "limit" in err

    def test_tool_loop_reset(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler, SamplingConfig
        handler = SamplingHandler("test", SamplingConfig(max_tool_rounds=1))
        assert handler.check_tool_loop() is None
        handler.reset_tool_loop()
        assert handler.check_tool_loop() is None  # Reset allows another

    def test_convert_text_messages(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler
        handler = SamplingHandler("test")

        class FakeMsg:
            role = "user"
            content = "hello"

        result = handler.convert_messages([FakeMsg()])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_handle_sampling_no_llm(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler
        handler = SamplingHandler("test")
        result = await handler.handle_sampling(MagicMock())
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_sampling_success(self):
        from caveman.tools.builtin.mcp_sampling import SamplingHandler
        llm_fn = AsyncMock(return_value={"text": "response", "usage": {"total_tokens": 100}})
        handler = SamplingHandler("test", llm_fn=llm_fn)

        params = MagicMock()
        params.messages = []
        params.maxTokens = 1000
        params.modelPreferences = None

        result = await handler.handle_sampling(params)
        assert result["content"] == "response"


# ── MCP Lifecycle Tests ──

class TestMCPLifecycle:
    def test_build_safe_env(self):
        from caveman.tools.builtin.mcp_lifecycle import build_safe_env
        import os
        # Should include PATH but not secrets
        env = build_safe_env()
        assert "PATH" in env

    def test_build_safe_env_blocks_secrets(self):
        from caveman.tools.builtin.mcp_lifecycle import build_safe_env
        import os
        os.environ["TEST_SECRET_KEY_XYZ"] = "secret"
        try:
            env = build_safe_env()
            assert "TEST_SECRET_KEY_XYZ" not in env
        finally:
            os.environ.pop("TEST_SECRET_KEY_XYZ", None)

    def test_build_safe_env_user_override(self):
        from caveman.tools.builtin.mcp_lifecycle import build_safe_env
        env = build_safe_env({"CUSTOM_VAR": "value"})
        assert env["CUSTOM_VAR"] == "value"

    def test_server_health(self):
        from caveman.tools.builtin.mcp_lifecycle import ServerHealth
        health = ServerHealth(connected=True, uptime_start=time.monotonic() - 60)
        assert health.is_healthy
        health.record_failure()
        health.record_failure()
        health.record_failure()
        assert not health.is_healthy

    def test_server_health_dict(self):
        from caveman.tools.builtin.mcp_lifecycle import ServerHealth
        health = ServerHealth(connected=True, uptime_start=time.monotonic())
        d = health.to_dict()
        assert d["connected"]
        assert d["healthy"]

    def test_tool_allowed(self):
        from caveman.tools.builtin.mcp_lifecycle import MCPServerLifecycle, MCPServerConfig
        config = MCPServerConfig(
            name="test",
            allowed_tools={"read_file", "write_file"},
            blocked_tools={"dangerous_tool"},
        )
        lifecycle = MCPServerLifecycle(config)
        assert lifecycle.is_tool_allowed("read_file")
        assert not lifecycle.is_tool_allowed("exec_shell")
        assert not lifecycle.is_tool_allowed("dangerous_tool")

    def test_tool_allowed_no_filter(self):
        from caveman.tools.builtin.mcp_lifecycle import MCPServerLifecycle, MCPServerConfig
        config = MCPServerConfig(name="test")
        lifecycle = MCPServerLifecycle(config)
        assert lifecycle.is_tool_allowed("anything")

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from caveman.tools.builtin.mcp_lifecycle import MCPServerLifecycle, MCPServerConfig
        config = MCPServerConfig(name="test")
        lifecycle = MCPServerLifecycle(config)
        connect_fn = AsyncMock(return_value=True)
        disconnect_fn = AsyncMock()
        lifecycle.set_callbacks(connect_fn=connect_fn, disconnect_fn=disconnect_fn)
        result = await lifecycle.start()
        assert result
        assert lifecycle.health.connected
        await lifecycle.stop()
        disconnect_fn.assert_called_once()
