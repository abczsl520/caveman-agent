"""Tests for Round 8 Phase A: Feature Flags, EventStore, Workspace Loader, CLI Agent Runner."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# FR-107: Engine Feature Flags
# ═══════════════════════════════════════════════════════════════════

class TestEngineFlags:
    """EngineFlags config-driven + runtime toggle."""

    def test_default_all_enabled(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags()
        for engine in ("shield", "nudge", "ripple", "lint", "recall"):
            assert flags.is_enabled(engine) is True

    def test_config_disables_nudge(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags({"engines": {"nudge": {"enabled": False}}})
        assert flags.is_enabled("nudge") is False
        assert flags.is_enabled("shield") is True

    def test_config_disables_multiple(self):
        from caveman.engines.flags import EngineFlags
        cfg = {"engines": {"ripple": {"enabled": False}, "lint": {"enabled": False}}}
        flags = EngineFlags(cfg)
        assert flags.is_enabled("ripple") is False
        assert flags.is_enabled("lint") is False
        assert flags.is_enabled("recall") is True

    def test_runtime_enable_disable(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags({"engines": {"nudge": {"enabled": False}}})
        assert flags.is_enabled("nudge") is False
        flags.enable("nudge")
        assert flags.is_enabled("nudge") is True
        flags.disable("nudge")
        assert flags.is_enabled("nudge") is False

    def test_status_returns_all(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags()
        status = flags.status()
        assert set(status.keys()) == {"shield", "nudge", "ripple", "lint", "recall", "scheduler", "verification", "reflect"}
        assert all(isinstance(v, bool) for v in status.values())

    def test_unknown_engine_raises(self):
        from caveman.engines.flags import EngineFlags, EngineError
        flags = EngineFlags()
        with pytest.raises(EngineError, match="Unknown engine"):
            flags.is_enabled("nonexistent")
        with pytest.raises(EngineError):
            flags.enable("bad")
        with pytest.raises(EngineError):
            flags.disable("bad")

    def test_config_from_default_yaml(self):
        """EngineFlags should work with the actual default.yaml config shape."""
        from caveman.engines.flags import EngineFlags
        from caveman.config.loader import load_config
        import caveman.config.loader as loader_mod
        # Use default.yaml directly, not user config
        default_path = str(Path(__file__).parent.parent / "caveman" / "config" / "default.yaml")
        config = load_config(default_path)
        flags = EngineFlags(config)
        assert flags.is_enabled("shield") is True
        assert flags.is_enabled("ripple") is True  # enabled in Round 10
        assert flags.is_enabled("lint") is True  # enabled in Round 9

    def test_engine_error_is_caveman_error(self):
        from caveman.engines.flags import EngineError
        from caveman.errors import CavemanError
        assert issubclass(EngineError, CavemanError)


# ═══════════════════════════════════════════════════════════════════
# FR-108: EventStore — SQLite Persistence
# ═══════════════════════════════════════════════════════════════════

class TestEventStore:
    """EventStore SQLite persistence."""

    def _make_store(self, tmp_path: Path):
        from caveman.events_store import EventStore
        return EventStore(db_path=tmp_path / "test_events.db")

    def test_handle_and_query(self, tmp_path):
        from caveman.events import Event, EventType
        store = self._make_store(tmp_path)
        event = Event(type=EventType.TOOL_CALL.value, data={"name": "bash"}, source="test")
        store.handle(event)
        rows = store.query()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "tool.call"
        assert json.loads(rows[0]["data_json"])["name"] == "bash"
        store.close()

    def test_query_by_type(self, tmp_path):
        from caveman.events import Event, EventType
        store = self._make_store(tmp_path)
        store.handle(Event(type=EventType.TOOL_CALL.value, data={}, source="a"))
        store.handle(Event(type=EventType.LLM_REQUEST.value, data={}, source="b"))
        store.handle(Event(type=EventType.TOOL_CALL.value, data={}, source="c"))
        assert len(store.query(event_type="tool.call")) == 2
        assert len(store.query(event_type="llm.request")) == 1
        store.close()

    def test_query_since(self, tmp_path):
        from caveman.events import Event
        store = self._make_store(tmp_path)
        t1 = time.time() - 100
        t2 = time.time()
        store.handle(Event(type="old", data={}, timestamp=t1))
        store.handle(Event(type="new", data={}, timestamp=t2))
        rows = store.query(since=t2 - 1)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "new"
        store.close()

    def test_count(self, tmp_path):
        from caveman.events import Event
        store = self._make_store(tmp_path)
        for i in range(5):
            store.handle(Event(type="test", data={"i": i}))
        assert store.count() == 5
        assert store.count(event_type="test") == 5
        assert store.count(event_type="other") == 0
        store.close()

    def test_query_limit(self, tmp_path):
        from caveman.events import Event
        store = self._make_store(tmp_path)
        for i in range(10):
            store.handle(Event(type="test", data={"i": i}))
        assert len(store.query(limit=3)) == 3
        store.close()

    def test_create_default_bus_with_persistence(self, tmp_path):
        """create_default_bus with persistence attaches store to bus."""
        from caveman.events import create_default_bus
        with patch("caveman.events_store.DEFAULT_EVENTS_DB", tmp_path / "bus_events.db"):
            bus, metrics = create_default_bus(enable_persistence=True)
            assert hasattr(bus, "event_store")
            bus.event_store.close()

    def test_create_default_bus_without_persistence(self):
        """create_default_bus without persistence has no store."""
        from caveman.events import create_default_bus
        bus, metrics = create_default_bus(enable_persistence=False)
        assert not hasattr(bus, "event_store")


# ═══════════════════════════════════════════════════════════════════
# FR-201: Workspace Loader
# ═══════════════════════════════════════════════════════════════════

class TestWorkspaceLoader:
    """WorkspaceLoader reads workspace files into prompt layers."""

    def test_load_empty_dirs(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        loader = WorkspaceLoader(paths=[tmp_path / "nonexistent"])
        assert loader.load() == {}

    def test_load_soul_md(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("我是豆包")
        loader = WorkspaceLoader(paths=[ws])
        files = loader.load()
        assert "SOUL.md" in files
        assert files["SOUL.md"] == "我是豆包"

    def test_first_path_wins(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        ws1 = tmp_path / "caveman_ws"
        ws2 = tmp_path / "openclaw_ws"
        ws1.mkdir()
        ws2.mkdir()
        (ws1 / "SOUL.md").write_text("caveman soul")
        (ws2 / "SOUL.md").write_text("openclaw soul")
        loader = WorkspaceLoader(paths=[ws1, ws2])
        assert loader.load()["SOUL.md"] == "caveman soul"

    def test_load_memory_subdir(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        ws = tmp_path / "workspace"
        ws.mkdir()
        mem = ws / "memory"
        mem.mkdir()
        (mem / "notes.md").write_text("some notes")
        loader = WorkspaceLoader(paths=[ws])
        files = loader.load()
        assert "memory/notes.md" in files

    def test_build_prompt_layers_ordering(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("persona")
        (ws / "USER.md").write_text("user context")
        (ws / "AGENTS.md").write_text("rules")
        loader = WorkspaceLoader(paths=[ws])
        prompt = loader.build_prompt_layers()
        # SOUL (layer 0) should come before USER (layer 1) before AGENTS (layer 3)
        soul_pos = prompt.index("persona")
        user_pos = prompt.index("user context")
        agents_pos = prompt.index("rules")
        assert soul_pos < user_pos < agents_pos

    def test_build_prompt_layers_empty(self, tmp_path):
        from caveman.agent.workspace import WorkspaceLoader
        loader = WorkspaceLoader(paths=[tmp_path / "nope"])
        assert loader.build_prompt_layers() == ""

    def test_soul_replaces_default_persona(self, tmp_path):
        """If SOUL.md exists, it replaces the default persona in system prompt."""
        from caveman.agent.workspace import WorkspaceLoader
        from caveman.agent.prompt import build_system_prompt, BASE_SYSTEM_PROMPT
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("我是豆包")
        loader = WorkspaceLoader(paths=[ws])
        prompt = build_system_prompt(workspace_loader=loader)
        assert "<!-- SOUL.md -->\n我是豆包" in prompt
        # Default persona should NOT be present
        assert "You are Caveman, an AI agent" not in prompt


# ═══════════════════════════════════════════════════════════════════
# FR-202: CLI Agent Runner
# ═══════════════════════════════════════════════════════════════════

class TestCLIAgentRunner:
    """CLIAgentRunner subprocess invocation."""

    def test_available_checks_path(self):
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner()
        available = runner.available()
        # We can't guarantee claude/codex are installed, but the method should work
        assert isinstance(available, list)

    def test_available_with_mock_agents(self):
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner(agents={
            "echo_agent": {"cmd": ["echo"], "pty": False, "timeout": 10},
        })
        available = runner.available()
        assert "echo_agent" in available  # echo is always available

    def test_run_echo_agent(self):
        """Run a simple echo-based agent."""
        async def _run():
            from caveman.bridge.cli_agents import CLIAgentRunner
            runner = CLIAgentRunner(agents={
                "echo": {"cmd": ["echo"], "pty": False, "timeout": 10},
            })
            result = await runner.run("echo", "hello world")
            assert result.exit_code == 0
            assert "hello world" in result.output
            assert result.timed_out is False
            assert result.agent == "echo"
            assert result.duration >= 0
        asyncio.run(_run())

    def test_run_unknown_agent_raises(self):
        async def _run():
            from caveman.bridge.cli_agents import CLIAgentRunner, CLIAgentError
            runner = CLIAgentRunner()
            with pytest.raises(CLIAgentError, match="Unknown agent"):
                await runner.run("nonexistent_agent", "task")
        asyncio.run(_run())

    def test_run_missing_binary(self):
        """Missing binary should return exit_code 127."""
        async def _run():
            from caveman.bridge.cli_agents import CLIAgentRunner
            runner = CLIAgentRunner(agents={
                "fake": {"cmd": ["__nonexistent_binary_xyz__"], "pty": False, "timeout": 5},
            })
            result = await runner.run("fake", "task")
            assert result.exit_code == 127
            assert "not found" in result.output.lower()
        asyncio.run(_run())

    def test_run_timeout(self):
        """Agent that exceeds timeout should be killed."""
        async def _run():
            from caveman.bridge.cli_agents import CLIAgentRunner
            runner = CLIAgentRunner(agents={
                "sleeper": {"cmd": ["sleep"], "pty": False, "timeout": 1},
            })
            result = await runner.run("sleeper", "60", timeout=1)
            assert result.timed_out is True
        asyncio.run(_run())

    def test_cli_agent_result_dataclass(self):
        from caveman.bridge.cli_agents import CLIAgentResult
        r = CLIAgentResult(output="hi", exit_code=0, duration=1.5, timed_out=False, agent="test")
        assert r.output == "hi"
        assert r.exit_code == 0
        assert r.duration == 1.5
        assert r.timed_out is False
        assert r.agent == "test"

    def test_coding_agent_tool_registered(self):
        """coding_agent tool should be discoverable via registry."""
        from caveman.tools.registry import ToolRegistry
        registry = ToolRegistry()
        registry._register_builtins()
        assert "coding_agent" in registry.list_tools()

    def test_cli_agent_error_is_caveman_error(self):
        from caveman.bridge.cli_agents import CLIAgentError
        from caveman.errors import CavemanError
        assert issubclass(CLIAgentError, CavemanError)
