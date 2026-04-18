"""Integration test: Agent Loop → Tool Execution pipeline.

Verifies that AgentLoop with a mock LLM provider correctly executes
tool calls and feeds results back into the conversation.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from caveman.agent.loop import AgentLoop
from caveman.agent.context import AgentContext, Message
from caveman.providers.llm import LLMProvider
from caveman.tools.registry import ToolRegistry
from caveman.memory.manager import MemoryManager
from caveman.skills.manager import SkillManager
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.events import create_default_bus
from caveman.engines.flags import EngineFlags


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns a tool call on first request, then a final message."""

    def __init__(self, tmp_path: Path):
        self.model = "mock-model"
        self.max_tokens = 4096
        self._call_count = 0
        self._tmp_path = tmp_path

    @property
    def context_length(self) -> int:
        return 200_000

    def _get_client(self):
        return None

    def _build_params(self, messages, system=None, tools=None, **kwargs):
        return {}

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            # First call: return a file_write tool call
            yield {"type": "tool_call", "id": "tc_001", "name": "file_write",
                   "input": {"path": str(self._tmp_path / "test_output.txt"),
                             "content": "Hello from integration test!"}}
            yield {"type": "done", "stop_reason": "tool_use"}
        else:
            # Second call: return final text
            yield {"type": "delta", "text": "File written successfully."}
            yield {"type": "done", "stop_reason": "end_turn"}


@pytest.mark.asyncio
async def test_agent_loop_tool_execution(tmp_path, monkeypatch):
    """AgentLoop executes a tool call and terminates with final message."""
    # Suppress print output from the loop
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    provider = MockLLMProvider(tmp_path)
    memory_mgr = MemoryManager(base_dir=tmp_path / "memory")
    skill_mgr = SkillManager(skills_dir=tmp_path / "skills")
    traj_rec = TrajectoryRecorder(base_dir=tmp_path / "trajectories")

    # Disable engines that need LLM
    flags = EngineFlags({"engines": {
        "shield": {"enabled": False}, "recall": {"enabled": False},
        "nudge": {"enabled": False}, "reflect": {"enabled": False},
        "lint": {"enabled": False}, "ripple": {"enabled": False},
    }})

    loop = AgentLoop(
        model="mock-model",
        max_iterations=5,
        provider=provider,
        memory_manager=memory_mgr,
        skill_manager=skill_mgr,
        trajectory_recorder=traj_rec,
        engine_flags=flags,
    )

    result = await loop.run("Write a test file")

    # Verify the tool executed
    output_file = tmp_path / "test_output.txt"
    assert output_file.exists()
    assert output_file.read_text() == "Hello from integration test!"

    # Verify the loop returned the final message
    assert "File written" in result

    # Verify the LLM was called twice (tool call + final)
    assert provider._call_count == 2


class MockLLMProviderDirect(LLMProvider):
    """Mock LLM that returns a direct text response (no tools)."""

    def __init__(self):
        self.model = "mock-direct"
        self.max_tokens = 4096

    @property
    def context_length(self) -> int:
        return 200_000

    def _get_client(self):
        return None

    def _build_params(self, messages, system=None, tools=None, **kwargs):
        return {}

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        yield {"type": "delta", "text": "The answer is 42."}
        yield {"type": "done", "stop_reason": "end_turn"}


@pytest.mark.asyncio
async def test_agent_loop_direct_response(tmp_path, monkeypatch):
    """AgentLoop handles a direct text response without tool calls."""
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    provider = MockLLMProviderDirect()
    flags = EngineFlags({"engines": {
        "shield": {"enabled": False}, "recall": {"enabled": False},
        "nudge": {"enabled": False}, "reflect": {"enabled": False},
        "lint": {"enabled": False}, "ripple": {"enabled": False},
    }})

    loop = AgentLoop(
        model="mock-direct",
        max_iterations=5,
        provider=provider,
        memory_manager=MemoryManager(base_dir=tmp_path / "memory"),
        skill_manager=SkillManager(skills_dir=tmp_path / "skills"),
        trajectory_recorder=TrajectoryRecorder(base_dir=tmp_path / "traj"),
        engine_flags=flags,
    )

    result = await loop.run("What is the meaning of life?")
    assert "42" in result
