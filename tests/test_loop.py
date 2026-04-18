"""Tests for agent loop."""
import pytest
from caveman.agent.loop import AgentLoop


def test_agent_loop_init():
    loop = AgentLoop(model="claude-opus-4-6")
    assert loop.model == "claude-opus-4-6"
    assert loop.max_iterations == 50


class TestSystemPromptRestore:
    """Ensure system prompt is never empty — regression test for the
    'restored session has empty system prompt' bug."""

    def test_prepare_multi_turn_rebuilds_prompt(self):
        """When _system_prompt_cache is None, _prepare_multi_turn should rebuild it."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from caveman.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._system_prompt_cache = None
        loop.surface = "discord"
        loop.tool_registry = MagicMock()
        loop.tool_registry.get_schemas.return_value = []
        loop.skill_manager = MagicMock()
        loop.skill_manager.match.return_value = []
        loop.memory_manager = MagicMock()
        loop.memory_manager.recall = AsyncMock(return_value=[])
        loop.trajectory_recorder = MagicMock()
        loop.trajectory_recorder.record_turn = AsyncMock()
        loop.bus = MagicMock()
        loop.bus.emit = AsyncMock()

        from caveman.agent.context import AgentContext
        loop._persistent_context = AgentContext(max_tokens=100000)

        ctx, system, skills = asyncio.run(
            loop._prepare_multi_turn("test task", [])
        )

        assert system, "System prompt must not be empty after multi-turn rebuild"
        assert "Response Style" in system or "Caveman" in system, \
            "Rebuilt prompt should contain identity or style layers"
        assert loop._system_prompt_cache == system, "Cache should be populated"

    def test_stream_multi_turn_rebuilds_prompt(self):
        """run_stream_impl should rebuild system prompt when cache is empty."""
        from caveman.agent.prompt import build_system_prompt

        # Verify build_system_prompt with discord surface includes style
        prompt = build_system_prompt(surface="discord")
        assert "Response Style" in prompt, "Discord prompt must include Response Style"
        assert "Discord" in prompt, "Discord prompt must mention Discord"

        # Verify CLI surface
        prompt_cli = build_system_prompt(surface="cli")
        assert "CLI" in prompt_cli or "terminal" in prompt_cli, \
            "CLI prompt must include CLI-specific rules"

    def test_build_system_prompt_surface_default(self):
        """Default surface should be 'cli'."""
        from caveman.agent.prompt import build_system_prompt
        prompt = build_system_prompt()
        # Should not contain Discord-specific rules
        assert "Discord" not in prompt or "CLI" in prompt

    def test_no_duplicate_run_logic(self):
        """stream.py must NOT contain run logic — it lives in loop.py's run_stream().
        run() must delegate to run_stream() (single implementation)."""
        import ast

        # 1. stream.py should only have data classes, no run logic
        with open("caveman/agent/stream.py") as f:
            stream_src = f.read()
        stream_tree = ast.parse(stream_src)
        for node in ast.walk(stream_tree):
            if isinstance(node, ast.AsyncFunctionDef) and "run" in node.name:
                raise AssertionError(f"stream.py should not contain run logic, found: {node.name}")

        # 2. loop.py run() should delegate to run_stream()
        with open("caveman/agent/loop.py") as f:
            loop_src = f.read()
        loop_tree = ast.parse(loop_src)
        for node in ast.walk(loop_tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                # Should be short (< 15 lines) — just a delegation wrapper
                assert len(node.body) < 15, \
                    f"run() should be a thin wrapper over run_stream(), got {len(node.body)} statements"
                break

    def test_session_restore_rebuilds_prompt(self):
        """Simulates the gateway session restore path and verifies
        system prompt is rebuilt."""
        from caveman.agent.prompt import build_system_prompt

        # Simulate what runner.py does after restoring transcript
        prompt = build_system_prompt(
            tool_schemas=[{"name": "bash", "description": "Run commands"}],
            surface="discord",
        )
        assert len(prompt) > 100, f"Restored prompt too short: {len(prompt)} chars"
        assert "Response Style" in prompt, "Must include response style layer"
        assert "Discord" in prompt, "Must include Discord-specific rules"
        assert "Safety" in prompt or "safety" in prompt.lower(), "Must include safety rules"
