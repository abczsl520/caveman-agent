"""Regression tests for AgentLoop iteration exhaustion paths."""
from __future__ import annotations

import pytest

from caveman.agent.loop import AgentLoop
from caveman.agent.stream import is_result_event_type
from caveman.providers.llm import LLMProvider


class AlwaysMaxTokensProvider(LLMProvider):
    """Provider that forces continuation until the loop budget is exhausted."""

    model = "test-max-tokens"
    max_tokens = 100

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        yield {"type": "delta", "text": "partial response"}
        yield {"type": "message_stop", "stop_reason": "max_tokens"}

    @property
    def context_length(self) -> int:
        return 100_000

    def _get_client(self):
        return None

    def _build_params(self, messages, system=None, tools=None, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_run_stream_max_iterations_exhaustion_does_not_name_error():
    loop = AgentLoop(
        model="test-max-tokens",
        provider=AlwaysMaxTokensProvider(),
        max_iterations=1,
    )

    events = []
    async for event in loop.run_stream("force max iteration exhaustion"):
        events.append(event)

    assert any(is_result_event_type(event.type) for event in events)
    result_text = "\n".join(str(event.data) for event in events if is_result_event_type(event.type))
    assert "迭代上限" in result_text
    assert "没有被验证为完成" in result_text
    assert not any(
        event.type == "error" and "show_error" in str(event.data)
        for event in events
    )
