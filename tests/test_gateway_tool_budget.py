"""Regression tests for gateway tool-call budget policy.

The old gateway had a hidden `_MAX_TOOL_CALLS = 80` hard cap. That silently
paused long-compounding flywheel/audit tasks even when they were making
progress. Tool-count budgets are now explicit opt-in policy.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from caveman.agent.stream import StreamEvent
from caveman.gateway.smart_buffer import _SmartBuffer
from caveman.gateway.task_runner import _TaskContext, _resolve_timeouts
from caveman.gateway.task_runner_helpers import _handle_tool_call


class _Router:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, gw_name, channel_id, message):
        self.sent.append(message)
        return {"message_id": str(len(self.sent))}

    async def edit(self, gw_name, channel_id, message_id, message):
        self.sent.append(message)
        return None


@pytest.mark.asyncio
async def test_gateway_default_has_no_80_tool_call_hard_cap():
    """81+ tool calls must not be stopped by a hidden safety limit."""
    router = _Router()
    ctx = _TaskContext("discord", "chan", router, _resolve_timeouts({}))
    buf = _SmartBuffer(router, "discord", "chan")
    buf.flush_interim = AsyncMock(return_value="")

    for i in range(81):
        event = StreamEvent(type="tool_call", data={"name": f"tool_{i}", "input": {"i": i}})
        should_break = await _handle_tool_call(event, ctx, buf)
        assert should_break is False
        assert ctx.shutdown_flag is False

    assert ctx.tool_call_count == 81
    assert not any("80" in msg or "安全上限" in msg for msg in router.sent)
    await ctx.cancel_all()


@pytest.mark.asyncio
async def test_gateway_explicit_tool_budget_pauses_as_incomplete_work():
    """If a user configures a tool budget, exhausting it pauses but doesn't mark success."""
    router = _Router()
    timeouts = _resolve_timeouts({"gateway": {"limits": {"max_tool_calls": 3}}})
    ctx = _TaskContext("discord", "chan", router, timeouts)
    buf = _SmartBuffer(router, "discord", "chan")
    buf.flush_interim = AsyncMock(return_value="")

    for i in range(2):
        event = StreamEvent(type="tool_call", data={"name": f"tool_{i}", "input": {"i": i}})
        assert await _handle_tool_call(event, ctx, buf) is False

    event = StreamEvent(type="tool_call", data={"name": "tool_2", "input": {"i": 2}})
    assert await _handle_tool_call(event, ctx, buf) is True

    assert ctx.shutdown_flag is True
    assert ctx.tool_call_count == 3
    assert any("工具预算" in msg and "任务未判定完成" in msg for msg in router.sent)
    await ctx.cancel_all()


def test_gateway_tool_budget_config_is_optional_and_unbounded_when_set():
    default_policy = _resolve_timeouts({})
    assert default_policy["max_tool_calls"] is None

    large_policy = _resolve_timeouts({"gateway": {"limits": {"max_tool_calls": 10_000}}})
    assert large_policy["max_tool_calls"] == 10_000


def test_gateway_tool_budget_validator_accepts_large_budget():
    from caveman.config.validator import validate_config

    assert validate_config(
        {"gateway": {"limits": {"max_tool_calls": 10_000}}},
        strict=False,
    ) == []


def test_gateway_tool_budget_validator_accepts_null_as_unlimited():
    from caveman.config.validator import validate_config

    assert validate_config(
        {"gateway": {"limits": {"max_tool_calls": None}}},
        strict=False,
    ) == []
