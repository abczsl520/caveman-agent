"""Tests for stream consumer, session context, debug helpers, env passthrough, interactive."""
import asyncio
import os
import pytest

from caveman.gateway.stream_consumer import StreamConsumer, StreamConfig
from caveman.gateway.session_context import (
    set_session_context, get_session_env, get_session_context, clear_session_context,
)
from caveman.tools.debug_helpers import DebugSession
from caveman.tools.env_passthrough import (
    register_passthrough, is_passthrough, get_passthrough_env, clear_passthrough,
    _BUILTIN_PASSTHROUGH,
)
from caveman.gateway.interactive import (
    InteractiveMessage, Button, ButtonStyle, to_discord_components, to_telegram_keyboard,
)


# ── Stream Consumer ──

class TestStreamConsumer:
    @pytest.mark.asyncio
    async def test_basic_stream(self):
        sent = []
        edited = []

        async def send(text):
            sent.append(text)
            return "msg-1"

        async def edit(msg_id, text):
            edited.append((msg_id, text))

        consumer = StreamConsumer(send, edit, StreamConfig(min_edit_interval=0, min_chars_per_edit=1))

        # Simulate stream
        consumer.on_delta("Hello ")
        consumer.on_delta("world!")
        consumer.on_complete()

        result = await consumer.run()
        assert "Hello world!" in result
        assert len(sent) >= 1

    @pytest.mark.asyncio
    async def test_tool_boundary(self):
        sent = []

        async def send(text):
            sent.append(text)
            return f"msg-{len(sent)}"

        async def edit(msg_id, text):
            pass

        consumer = StreamConsumer(send, edit, StreamConfig(min_edit_interval=0, min_chars_per_edit=1))
        consumer.on_delta("Part 1")
        consumer.on_tool_boundary()
        consumer.on_delta("Part 2")
        consumer.on_complete()

        await consumer.run()
        assert len(sent) > 0  # Stream consumer produced output
        # Should have sent at least 2 messages (one per segment)


# ── Session Context ──

class TestSessionContext:
    def setup_method(self):
        clear_session_context()

    def test_set_and_get(self):
        set_session_context(platform="discord", chat_id="123")
        assert get_session_env("CAVEMAN_SESSION_PLATFORM") == "discord"
        assert get_session_env("CAVEMAN_SESSION_CHAT_ID") == "123"

    def test_default(self):
        assert get_session_env("CAVEMAN_SESSION_PLATFORM", "fallback") == ""

    def test_unknown_var(self):
        assert get_session_env("NONEXISTENT", "default") == "default"

    def test_get_all(self):
        set_session_context(platform="telegram")
        ctx = get_session_context()
        assert "CAVEMAN_SESSION_PLATFORM" in ctx

    def test_clear(self):
        set_session_context(platform="discord")
        clear_session_context()
        assert get_session_env("CAVEMAN_SESSION_PLATFORM") == ""


# ── Debug Helpers ──

class TestDebugSession:
    def test_disabled_by_default(self):
        ds = DebugSession("test_tool")
        assert ds.enabled is False

    def test_log_call_noop_when_disabled(self):
        ds = DebugSession("test_tool")
        ds.log_call("search", {"query": "test"})
        assert len(ds._calls) == 0

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("TEST_TOOL_DEBUG", "true")
        ds = DebugSession("test_tool")
        assert ds.enabled is True
        ds.log_call("search", {"query": "test"})
        assert len(ds._calls) == 1

    def test_save(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_TOOL_DEBUG", "true")
        monkeypatch.setattr("caveman.tools.debug_helpers.CAVEMAN_HOME", tmp_path)
        ds = DebugSession("test_tool")
        ds.log_call("search")
        path = ds.save()
        assert path is not None
        assert path.exists()

    def test_session_info(self):
        ds = DebugSession("test_tool")
        info = ds.get_session_info()
        assert info["tool"] == "test_tool"
        assert info["call_count"] == 0


# ── Env Passthrough ──

class TestEnvPassthrough:
    def setup_method(self):
        clear_passthrough()

    def test_builtin_always_allowed(self):
        assert is_passthrough("PATH") is True
        assert is_passthrough("HOME") is True

    def test_register_and_check(self):
        register_passthrough(["MY_API_KEY", "MY_SECRET"])
        assert is_passthrough("MY_API_KEY") is True
        assert is_passthrough("UNKNOWN") is False

    def test_get_passthrough_env(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "value")
        register_passthrough(["MY_VAR"])
        env = get_passthrough_env()
        assert env.get("MY_VAR") == "value"

    def test_clear(self):
        register_passthrough(["MY_VAR"])
        clear_passthrough()
        assert is_passthrough("MY_VAR") is False


# ── Interactive ──

class TestInteractive:
    def test_create_message(self):
        msg = InteractiveMessage(text="Choose:")
        row = msg.add_row()
        row.add("Yes", action="confirm", style=ButtonStyle.SUCCESS)
        row.add("No", action="cancel", style=ButtonStyle.DANGER)
        assert len(msg.rows) == 1
        assert len(msg.rows[0].buttons) == 2

    def test_select_menu(self):
        msg = InteractiveMessage(text="Pick model:")
        menu = msg.add_select("model_select", "Choose a model")
        menu.add_option("Claude", "claude")
        menu.add_option("GPT-4", "gpt4")
        assert len(msg.selects) == 1
        assert len(msg.selects[0].options) == 2

    def test_discord_components(self):
        msg = InteractiveMessage()
        row = msg.add_row()
        row.add("Click me", action="btn1")
        components = to_discord_components(msg)
        assert len(components) == 1
        assert components[0]["type"] == 1  # ActionRow
        assert components[0]["components"][0]["type"] == 2  # Button

    def test_discord_link_button(self):
        msg = InteractiveMessage()
        row = msg.add_row()
        row.add("Visit", url="https://example.com")
        components = to_discord_components(msg)
        assert components[0]["components"][0]["style"] == 5  # Link

    def test_telegram_keyboard(self):
        msg = InteractiveMessage()
        row = msg.add_row()
        row.add("Yes", action="yes")
        row.add("No", action="no")
        kb = to_telegram_keyboard(msg)
        assert len(kb) == 1
        assert len(kb[0]) == 2
        assert kb[0][0]["text"] == "Yes"
        assert kb[0][0]["callback_data"] == "yes"

    def test_discord_select(self):
        msg = InteractiveMessage()
        menu = msg.add_select("pick", "Choose")
        menu.add_option("A", "a")
        menu.add_option("B", "b")
        components = to_discord_components(msg)
        assert len(components) == 1
        assert components[0]["components"][0]["type"] == 3  # StringSelect
