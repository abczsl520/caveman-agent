"""End-to-end dry run — tests full AgentLoop without real LLM API."""
import pytest
from caveman.agent.loop import AgentLoop
from caveman.memory.manager import MemoryManager
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
            yield {"type": "message_stop", "stop_reason": "tool_use", "usage": {}}
        else:
            yield {"type": "delta", "text": "Task completed successfully."}
            yield {"type": "message_stop", "stop_reason": "end_turn", "usage": {}}
class PrematureClosingProvider(MockProvider):
    """Provider emits text claiming done and a tool call in the same turn."""

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        call_count = getattr(self, "_call_count", 0)
        self._call_count = call_count + 1

        if call_count == 0 and tools:
            yield {"type": "delta", "text": "All done.\n\n✅---本轮已完成---✅"}
            yield {"type": "tool_call", "id": "call_1", "name": "web_search", "input": {"query": "test"}}
            yield {"type": "message_stop", "stop_reason": "tool_use", "usage": {}}
        else:
            yield {"type": "delta", "text": "Actual work finished after tool execution."}
            yield {"type": "message_stop", "stop_reason": "end_turn", "usage": {}}


class ContinuationTerminalProvider(MockProvider):
    """Provider emits a stop-like final response for a keep-going task."""

    async def complete(self, messages, tools=None, stream=True, system=None, **kwargs):
        yield {"type": "delta", "text": "本轮已完成 output_validator 修复。\n\nDone."}
        yield {"type": "message_stop", "stop_reason": "end_turn", "usage": {}}


def make_loop(provider, tmp_path):
    """Create an AgentLoop with isolated JSON memory for deterministic tests."""
    return AgentLoop(model="mock", provider=provider, memory_manager=MemoryManager(base_dir=tmp_path / "memory"))


@pytest.mark.asyncio
async def test_agent_loop_e2e_dryrun(tmp_path):
    provider = MockProvider()
    loop = make_loop(provider, tmp_path)
    result = await loop.run("Search for AI news")
    assert "completed" in result.lower() or len(result) > 0


@pytest.mark.asyncio
async def test_agent_loop_tool_execution(tmp_path):
    provider = MockProvider()
    loop = make_loop(provider, tmp_path)

    call_log = []
    async def mock_tool(query: str):
        call_log.append(query)
        return {"results": [{"title": "Test", "url": "http://test.com"}]}

    loop.tool_registry.register("web_search", mock_tool, "Search web",
                                {"type": "object", "properties": {"query": {"type": "string"}}})
    result = await loop.run("Search for AI news")
    assert len(call_log) > 0


@pytest.mark.asyncio
async def test_premature_closing_marker_does_not_skip_tool_execution(tmp_path):
    provider = PrematureClosingProvider()
    loop = make_loop(provider, tmp_path)

    call_log = []
    async def mock_tool(query: str):
        call_log.append(query)
        return {"results": [{"title": "Test", "url": "http://test.com"}]}

    loop.tool_registry.register("web_search", mock_tool, "Search web",
                                {"type": "object", "properties": {"query": {"type": "string"}}})
    result = await loop.run("Search for AI news")

    assert call_log == ["test"]
    assert "Actual work finished" in result


@pytest.mark.asyncio
async def test_agent_loop_neutralizes_terminal_text_for_auto_continuation_task(tmp_path):
    provider = ContinuationTerminalProvider()
    loop = make_loop(provider, tmp_path)

    result = await loop.run("继续飞轮 (自动第 7/20 轮)")

    assert "Done" not in result
    assert "已完成" not in result
    assert "阶段性推进" in result
    assert "连续任务保持推进" in result


@pytest.mark.asyncio
async def test_agent_loop_preserves_terminal_text_for_normal_task(tmp_path):
    provider = ContinuationTerminalProvider()
    loop = make_loop(provider, tmp_path)

    result = await loop.run("修复 README 里的错别字")

    assert "Done" in result
    assert "已完成" in result
