"""Regression tests for LLM idle-timeout handling.

The gateway previously treated an LLM streaming idle timeout as a normal final
assistant response `(LLM 无响应)`, which made unfinished work look completed.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _NeverYieldProvider:
    model = "test-model"
    context_length = 200_000
    usage_stats = {}

    async def safe_complete(self, messages, system=None, tools=None, stream=True, **kw):
        # Sleep forever from the perspective of DEFAULT_LLM_IDLE_TIMEOUT.
        await asyncio.sleep(3600)
        if False:  # pragma: no cover - keeps this an async generator
            yield {"type": "delta", "text": "unreachable"}


def _make_loop(provider):
    from caveman.agent.loop import AgentLoop
    from caveman.agent.iteration_budget import IterationBudget

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = provider
    loop.max_iterations = 5
    loop.budget = IterationBudget(5)
    loop._fallback_chain = None
    loop._last_activity_ts = 0.0
    loop._last_activity_desc = ""
    loop._current_tool = ""
    loop.bus = MagicMock()
    loop.bus.emit = AsyncMock()
    loop.bus.on = MagicMock()
    loop.bus.off = MagicMock()
    loop.skill_manager = MagicMock()
    loop.memory_manager = MagicMock()
    loop.trajectory_recorder = MagicMock()
    loop.trajectory_recorder.record_turn = AsyncMock()
    loop._recall = MagicMock()
    loop.engine_flags = MagicMock()
    loop.tool_registry = MagicMock()
    loop.tool_registry.get_schemas.return_value = []
    loop.permission_manager = MagicMock()
    loop.permission_manager.request = AsyncMock(return_value=True)
    loop._tool_call_count = 0
    loop._bg_skill_nudge = AsyncMock()
    loop._nudge_task_ref = ""
    loop._turn_number = 0
    loop._turn_count = 0
    loop._persistent_context = None
    loop._system_prompt_cache = None
    loop.surface = "cli"
    loop.metrics = MagicMock()
    loop._shield = None
    loop._reflect = None
    loop._nudge = MagicMock()
    loop._lint = None
    loop._ripple = None
    loop._llm_fn = AsyncMock()
    loop._check_termination = AsyncMock(return_value=True)
    loop._post_task_engines = AsyncMock()
    loop._offer_matching_skill = AsyncMock()
    loop._record_turn_metrics = MagicMock()
    loop._safe_bg = MagicMock()
    return loop


@pytest.mark.asyncio
async def test_llm_idle_timeout_is_error_not_fake_completion(monkeypatch):
    """Idle timeout must not produce a fake final answer or run finalizers."""
    monkeypatch.setattr("caveman.agent.loop.DEFAULT_LLM_IDLE_TIMEOUT", 0.01)

    mock_context = MagicMock()
    mock_context.should_compress.return_value = False
    mock_context.messages = []
    mock_context.to_api_format.return_value = []
    mock_context.utilization = 0.0

    finalize = AsyncMock(return_value="done")
    record_turn = MagicMock()

    with patch("caveman.agent.loop.phase_prepare", AsyncMock(return_value=(mock_context, "system", []))), \
         patch("caveman.agent.loop.phase_finalize", finalize), \
         patch("caveman.agent.loop.record_assistant_turn", record_turn):
        loop = _make_loop(_NeverYieldProvider())
        events = []
        async for event in loop.run_stream("complex task"):
            events.append(event)

    assert [e.type for e in events] == ["iteration_start", "error"]
    assert "LLM 无响应超时" in str(events[-1].data)
    assert "任务未完成" in str(events[-1].data)
    finalize.assert_not_awaited()
    record_turn.assert_not_called()
    loop._post_task_engines.assert_not_awaited()
    assert all("(LLM 无响应)" not in str(e.data) for e in events)
