"""Regression tests for decomposed agent phases."""
from __future__ import annotations

import pytest

from caveman.agent.context import AgentContext
from caveman.agent.phases import phase_llm_call
from caveman.events import EventBus


class _Provider:
    def __init__(self, events):
        self._events = events

    async def safe_complete(self, **_kwargs):
        for event in self._events:
            yield event


class _Tools:
    def get_schemas(self):
        return []


@pytest.mark.asyncio
async def test_phase_llm_call_preserves_provider_stop_reason():
    context = AgentContext()
    context.add_message("user", "hello")
    provider = _Provider([
        {"type": "delta", "text": "hi"},
        {"type": "message_stop", "stop_reason": "max_tokens"},
    ])

    text, tool_calls, stop = await phase_llm_call(
        context=context,
        system="system prompt long enough for guardrails",
        provider=provider,
        tool_registry=_Tools(),
        bus=EventBus(),
    )

    assert text == "hi"
    assert tool_calls == []
    assert stop == "max_tokens"
