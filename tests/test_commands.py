"""Tests for the slash command system."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from caveman.commands.types import CommandDef, CommandContext
from caveman.commands.registry import (
    COMMAND_REGISTRY,
    resolve_command,
    get_by_category,
    rebuild_lookups,
)
from caveman.commands.dispatcher import parse_command, dispatch
from caveman.commands.completer import CommandCompleter
from caveman.commands.formatter import format_panel, format_table, format_help
from caveman.commands.gateway_adapter import discord_slash_commands, telegram_bot_commands


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_lookups():
    """Ensure lookups are rebuilt for each test."""
    rebuild_lookups()
    yield


def _make_agent():
    """Create a mock agent with common attributes."""
    agent = MagicMock()
    agent.model = "claude-opus-4-6"
    agent.tool_registry.get_schemas.return_value = [{"name": "bash"}, {"name": "read"}]
    agent.memory_manager.total_count = 42
    agent.max_iterations = 50
    agent._tool_call_count = 7
    agent.engine_flags.is_enabled.return_value = True
    agent.skill_manager.list_all.return_value = []
    agent._shield.essence.session_id = "test-session"
    agent._shield.essence.task = "testing"
    agent._shield.essence.turn_count = 3
    agent._shield.essence.decisions = []
    agent._shield.essence.progress = []
    agent._shield.essence.open_todos = []
    agent._shield.essence.key_data = {}
    agent._reflect.reflections = []
    agent.fast_mode = False
    agent.conversation = []
    return agent


# ── Registry integrity ────────────────────────────────────────

class TestRegistryIntegrity:
    def test_at_least_50_commands(self):
        assert len(COMMAND_REGISTRY) >= 50

    def test_no_duplicate_names(self):
        names = [c.name for c in COMMAND_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_no_duplicate_aliases(self):
        all_aliases: list[str] = []
        for c in COMMAND_REGISTRY:
            all_aliases.extend(c.aliases)
        assert len(all_aliases) == len(set(all_aliases)), "Duplicate aliases found"

    def test_no_alias_shadows_name(self):
        names = {c.name for c in COMMAND_REGISTRY}
        for c in COMMAND_REGISTRY:
            for alias in c.aliases:
                assert alias not in names, f"Alias '{alias}' shadows command name"

    def test_all_handlers_have_module_and_func(self):
        for c in COMMAND_REGISTRY:
            parts = c.handler.split(".")
            assert len(parts) == 2, f"Bad handler path: {c.handler}"

    def test_categories_present(self):
        cats = get_by_category()
        expected = {"session", "config", "tools", "info", "system", "caveman"}
        assert expected == set(cats.keys())

    def test_every_command_has_description(self):
        for c in COMMAND_REGISTRY:
            assert c.description, f"/{c.name} missing description"


# ── Resolve command ───────────────────────────────────────────

class TestResolveCommand:
    def test_resolve_by_name(self):
        cmd = resolve_command("model")
        assert cmd is not None
        assert cmd.name == "model"

    def test_resolve_by_alias(self):
        cmd = resolve_command("reset")
        assert cmd is not None
        assert cmd.name == "new"

    def test_resolve_with_slash(self):
        cmd = resolve_command("/help")
        assert cmd is not None
        assert cmd.name == "help"

    def test_resolve_unknown(self):
        assert resolve_command("nonexistent") is None

    def test_resolve_case_insensitive(self):
        cmd = resolve_command("MODEL")
        assert cmd is not None
        assert cmd.name == "model"

    def test_quit_aliases(self):
        for alias in ("exit", "q"):
            cmd = resolve_command(alias)
            assert cmd is not None
            assert cmd.name == "quit"


# ── Parse command ─────────────────────────────────────────────

class TestParseCommand:
    def test_simple(self):
        assert parse_command("/help") == ("help", "")

    def test_with_args(self):
        assert parse_command("/model claude-sonnet-4-6") == ("model", "claude-sonnet-4-6")

    def test_no_slash(self):
        assert parse_command("hello") == (None, "")

    def test_whitespace(self):
        assert parse_command("  /status  ") == ("status", "")

    def test_multi_word_args(self):
        name, args = parse_command("/title My Cool Project")
        assert name == "title"
        assert args == "My Cool Project"


# ── Dispatcher ────────────────────────────────────────────────

class TestDispatcher:
    @pytest.mark.asyncio
    async def test_dispatch_known_command(self):
        agent = _make_agent()
        responses = []
        result = await dispatch("/model", agent, respond_fn=responses.append)
        assert result == "handled"
        assert any("claude-opus-4-6" in r for r in responses)

    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self):
        agent = _make_agent()
        responses = []
        result = await dispatch("/xyzzy", agent, respond_fn=responses.append)
        assert result == "handled"
        assert any("Unknown" in r for r in responses)

    @pytest.mark.asyncio
    async def test_dispatch_quit(self):
        agent = _make_agent()
        responses = []
        result = await dispatch("/quit", agent, respond_fn=responses.append)
        assert result == "exit"

    @pytest.mark.asyncio
    async def test_dispatch_alias(self):
        agent = _make_agent()
        responses = []
        result = await dispatch("/think high", agent, respond_fn=responses.append)
        assert result == "handled"
        assert any("high" in r.lower() for r in responses)

    @pytest.mark.asyncio
    async def test_dispatch_model_switch(self):
        agent = _make_agent()
        responses = []
        await dispatch("/model claude-sonnet-4-6", agent, respond_fn=responses.append)
        assert agent.model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_dispatch_cli_only_on_gateway(self):
        agent = _make_agent()
        responses = []
        result = await dispatch("/quit", agent, surface="discord", respond_fn=responses.append)
        assert result == "handled"
        assert any("CLI" in r for r in responses)


# ── Handlers ──────────────────────────────────────────────────

class TestHandlers:
    @pytest.mark.asyncio
    async def test_handle_new(self):
        from caveman.commands.handlers.session import handle_new
        agent = _make_agent()
        agent.conversation = [{"role": "user", "content": "hi"}]
        responses = []
        ctx = CommandContext("new", "", agent, "cli", respond=responses.append)
        await handle_new(ctx)
        assert any("New session" in r for r in responses)

    @pytest.mark.asyncio
    async def test_handle_fast_on(self):
        from caveman.commands.handlers.config import handle_fast
        agent = _make_agent()
        responses = []
        ctx = CommandContext("fast", "on", agent, "cli", respond=responses.append)
        await handle_fast(ctx)
        assert agent.fast_mode is True

    @pytest.mark.asyncio
    async def test_handle_engines_list(self):
        from caveman.commands.handlers.tools import handle_engines
        agent = _make_agent()
        responses = []
        ctx = CommandContext("engines", "list", agent, "cli", respond=responses.append)
        await handle_engines(ctx)
        assert any("shield" in r for r in responses)

    @pytest.mark.asyncio
    async def test_handle_selftest(self):
        from caveman.commands.handlers.caveman import handle_selftest
        agent = _make_agent()
        responses = []
        ctx = CommandContext("selftest", "", agent, "cli", respond=responses.append)
        await handle_selftest(ctx)
        assert any("Self-test" in r for r in responses)

    @pytest.mark.asyncio
    async def test_handle_help_single_command(self):
        from caveman.commands.handlers.info import handle_help
        agent = _make_agent()
        responses = []
        ctx = CommandContext("help", "model", agent, "cli", respond=responses.append)
        await handle_help(ctx)
        assert any("model" in r.lower() for r in responses)

    @pytest.mark.asyncio
    async def test_handle_help_overview(self):
        from caveman.commands.handlers.info import handle_help
        agent = _make_agent()
        responses = []
        ctx = CommandContext("help", "", agent, "cli", respond=responses.append)
        await handle_help(ctx)
        assert any("Session" in r for r in responses)

    @pytest.mark.asyncio
    async def test_handle_commands_pagination(self):
        from caveman.commands.handlers.info import handle_commands
        agent = _make_agent()
        responses = []
        ctx = CommandContext("commands", "2", agent, "cli", respond=responses.append)
        await handle_commands(ctx)
        assert any("Page 2" in r for r in responses)

    @pytest.mark.asyncio
    async def test_handle_status(self):
        from caveman.commands.handlers.info import handle_status
        agent = _make_agent()
        responses = []
        ctx = CommandContext("status", "", agent, "cli", respond=responses.append)
        await handle_status(ctx)
        assert any("claude-opus-4-6" in r for r in responses)


# ── Completer ─────────────────────────────────────────────────

class TestCompleter:
    def test_command_prefix(self):
        c = CommandCompleter()
        results = c.complete("/mod")
        assert "/model" in results

    def test_subcommand(self):
        c = CommandCompleter()
        results = c.complete("/model ")
        # model doesn't have subcommands in registry, but reasoning does
        results2 = c.complete("/reasoning ")
        assert "high" in results2

    def test_no_match(self):
        c = CommandCompleter()
        results = c.complete("/xyzzy")
        assert results == []

    def test_non_slash(self):
        c = CommandCompleter()
        results = c.complete("hello")
        assert results == []


# ── Formatter ─────────────────────────────────────────────────

class TestFormatter:
    def test_panel_cli(self):
        result = format_panel("Title", "content", "cli")
        assert "Title" in result
        assert "content" in result

    def test_panel_discord(self):
        result = format_panel("Title", "content", "discord")
        assert "**Title**" in result

    def test_table_cli(self):
        result = format_table(["Name", "Value"], [["a", "1"]], "cli")
        assert "Name" in result
        assert "a" in result

    def test_table_markdown(self):
        result = format_table(["Name", "Value"], [["a", "1"]], "discord")
        assert "| Name | Value |" in result

    def test_help_format(self):
        cats = get_by_category()
        result = format_help(cats, "cli")
        assert "Session" in result
        assert "/model" in result


# ── Gateway adapter ───────────────────────────────────────────

class TestGatewayAdapter:
    def test_discord_commands(self):
        cmds = discord_slash_commands()
        assert len(cmds) > 0
        names = {c["name"] for c in cmds}
        assert "model" in names
        # CLI-only commands should be excluded
        assert "quit" not in names

    def test_telegram_commands(self):
        cmds = telegram_bot_commands()
        assert len(cmds) > 0
        assert all("command" in c and "description" in c for c in cmds)
