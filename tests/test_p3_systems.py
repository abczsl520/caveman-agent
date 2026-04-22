"""Tests for command registry, process registry, MCP client, send message."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Command Registry Tests ──

class TestCommandRegistry:
    def test_register_and_get(self):
        from caveman.gateway.command_registry import CommandRegistry, CommandDefinition
        reg = CommandRegistry()
        cmd = CommandDefinition(key="test", description="Test command")
        reg.register(cmd)
        assert reg.get("test") is cmd

    def test_parse_simple(self):
        from caveman.gateway.command_registry import CommandRegistry, CommandDefinition
        reg = CommandRegistry()
        reg.register(CommandDefinition(key="help"))
        parsed = reg.parse("/help")
        assert parsed.key == "help"
        assert parsed.valid

    def test_parse_with_args(self):
        from caveman.gateway.command_registry import (
            CommandRegistry, CommandDefinition, CommandArg,
        )
        reg = CommandRegistry()
        reg.register(CommandDefinition(
            key="model",
            args=[CommandArg(name="model", capture_remaining=True)],
        ))
        parsed = reg.parse("/model claude-opus-4-6")
        assert parsed.args["model"] == "claude-opus-4-6"

    def test_parse_unknown_command(self):
        from caveman.gateway.command_registry import CommandRegistry
        reg = CommandRegistry()
        parsed = reg.parse("/nonexistent")
        assert not parsed.valid
        assert "Unknown" in parsed.error

    def test_parse_not_command(self):
        from caveman.gateway.command_registry import CommandRegistry
        reg = CommandRegistry()
        assert reg.parse("hello world") is None

    def test_alias(self):
        from caveman.gateway.command_registry import CommandRegistry, CommandDefinition
        reg = CommandRegistry()
        reg.register(CommandDefinition(key="reset", aliases=["new"]))
        parsed = reg.parse("/new")
        assert parsed.key == "reset"

    @pytest.mark.asyncio
    async def test_dispatch(self):
        from caveman.gateway.command_registry import CommandRegistry, CommandDefinition
        reg = CommandRegistry()
        reg.register(CommandDefinition(key="ping", handler=lambda p, c: "pong"))
        result = await reg.dispatch("/ping")
        assert result == "pong"

    @pytest.mark.asyncio
    async def test_dispatch_async_handler(self):
        from caveman.gateway.command_registry import CommandRegistry, CommandDefinition
        reg = CommandRegistry()
        async def async_handler(parsed, ctx):
            return f"hello {parsed.remaining}"
        reg.register(CommandDefinition(key="greet", handler=async_handler))
        result = await reg.dispatch("/greet world")
        assert result == "hello world"

    def test_build_help(self):
        from caveman.gateway.command_registry import CommandRegistry, build_builtin_commands
        reg = CommandRegistry()
        for cmd in build_builtin_commands():
            reg.register(cmd)
        help_text = reg.build_help()
        assert "/help" in help_text
        assert "/model" in help_text
        assert "/reset" in help_text

    def test_list_commands(self):
        from caveman.gateway.command_registry import (
            CommandRegistry, CommandDefinition, CommandCategory,
        )
        reg = CommandRegistry()
        reg.register(CommandDefinition(key="a", category=CommandCategory.STATUS))
        reg.register(CommandDefinition(key="b", category=CommandCategory.TOOLS))
        reg.register(CommandDefinition(key="c", category=CommandCategory.STATUS, hidden=True))
        visible = reg.list_commands()
        assert len(visible) == 2
        status_only = reg.list_commands(category=CommandCategory.STATUS)
        assert len(status_only) == 1

    def test_choices_validation(self):
        from caveman.gateway.command_registry import (
            CommandRegistry, CommandDefinition, CommandArg,
        )
        reg = CommandRegistry()
        reg.register(CommandDefinition(
            key="mode",
            args=[CommandArg(name="mode", choices=["on", "off"])],
        ))
        parsed = reg.parse("/mode invalid")
        assert "Invalid" in parsed.error


# ── Process Registry Tests ──

class TestProcessRegistry:
    @pytest.mark.asyncio
    async def test_spawn_and_poll(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        session = await reg.spawn("echo hello")
        assert session.status == "running" or session.status == "completed"
        # Wait for completion
        await asyncio.sleep(0.5)
        result = reg.poll(session.id)
        assert result["status"] in ("completed", "running")

    @pytest.mark.asyncio
    async def test_spawn_with_timeout(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        session = await reg.spawn("sleep 10", timeout=0.5)
        await asyncio.sleep(1)
        result = reg.poll(session.id)
        assert result["status"] == "killed"

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        await reg.spawn("echo a")
        await reg.spawn("echo b")
        sessions = reg.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_kill(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        session = await reg.spawn("sleep 60")
        result = await reg.kill(session.id)
        assert result.get("ok") or result.get("status") == "killed"

    @pytest.mark.asyncio
    async def test_read_log(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        session = await reg.spawn("echo line1 && echo line2 && echo line3")
        await asyncio.sleep(0.5)
        result = reg.read_log(session.id)
        assert result.get("total_lines", 0) > 0

    def test_poll_nonexistent(self):
        from caveman.tools.builtin.process_registry import ProcessRegistry
        reg = ProcessRegistry()
        result = reg.poll("nonexistent")
        assert "error" in result


# ── MCP Client Tests ──

class TestMCPClient:
    def test_init(self):
        from caveman.tools.builtin.mcp_client import MCPClient
        client = MCPClient()
        assert client.list_tools() == []
        assert client.list_servers() == []

    def test_get_tool_not_found(self):
        from caveman.tools.builtin.mcp_client import MCPClient
        client = MCPClient()
        assert client.get_tool("nonexistent") is None

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        from caveman.tools.builtin.mcp_client import MCPClient
        client = MCPClient()
        result = await client.call_tool("nonexistent", {})
        assert "error" in result

    def test_server_properties(self):
        from caveman.tools.builtin.mcp_client import MCPServer
        stdio = MCPServer(name="test", command="node")
        assert stdio.is_stdio
        assert not stdio.is_http
        http = MCPServer(name="test2", url="http://localhost:3000")
        assert http.is_http
        assert not http.is_stdio


# ── Send Message Tests ──

class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_no_platform(self):
        from caveman.tools.builtin.send_message_tool import send_message
        result = await send_message("nonexistent", "ch1", "hello")
        assert not result["ok"]
        assert "not connected" in result["error"]

    @pytest.mark.asyncio
    async def test_send_with_platform(self):
        from caveman.tools.builtin.send_message_tool import send_message, register_platform
        from caveman.gateway.platform_types import SendResult
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
        register_platform("test_platform", mock_adapter)
        try:
            result = await send_message("test_platform", "ch1", "hello")
            assert result["ok"]
            assert result["message_id"] == "m1"
        finally:
            from caveman.tools.builtin import send_message_tool
            send_message_tool._platform_adapters.pop("test_platform", None)

    def test_list_platforms(self):
        from caveman.tools.builtin.send_message_tool import list_platforms
        # Should return whatever is registered
        platforms = list_platforms()
        assert isinstance(platforms, list)
