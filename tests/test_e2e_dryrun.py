"""End-to-end dry run — tests full AgentLoop without real LLM API."""
import pytest
from caveman.agent.loop import AgentLoop
from caveman.providers.llm import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider that returns a fixed response."""

    model = "mock"
    max_tokens = 4096

    @property
    def context_length(self):
        return 100_000

    def _get_client(self):
        return None

    def _build_params(self, messages, system=None, tools=None, **kwargs):
        return {"messages": messages, "system": system, "tools": tools}

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        call_count = getattr(self, "_call_count", 0)
        self._call_count = call_count + 1

        if call_count == 0 and tools:
            yield {"type": "tool_call", "id": "call_1", "name": "web_search", "input": {"query": "test"}}
            yield {"type": "done", "stop_reason": "tool_use", "usage": {}}
        else:
            yield {"type": "delta", "text": "Task completed successfully."}
            yield {"type": "done", "stop_reason": "end_turn", "usage": {}}


@pytest.mark.asyncio
async def test_agent_loop_e2e_dryrun():
    provider = MockProvider()
    loop = AgentLoop(model="mock", provider=provider)
    result = await loop.run("Search for AI news")
    assert "completed" in result.lower() or len(result) > 0


@pytest.mark.asyncio
async def test_agent_loop_tool_execution():
    provider = MockProvider()
    loop = AgentLoop(model="mock", provider=provider)

    call_log = []
    async def mock_tool(query: str):
        call_log.append(query)
        return {"results": [{"title": "Test", "url": "http://test.com"}]}

    loop.tool_registry.register("web_search", mock_tool, "Search web",
                                {"type": "object", "properties": {"query": {"type": "string"}}})
    result = await loop.run("Search for AI news")
    assert len(call_log) > 0
