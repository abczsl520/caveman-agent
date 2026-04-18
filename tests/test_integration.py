"""Integration tests — verify full system wiring."""
import pytest
from caveman.agent.loop import AgentLoop
from caveman.providers.llm import LLMProvider


class ToolTestProvider(LLMProvider):
    """Mock provider that exercises multiple tools sequentially."""

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
        call = getattr(self, "_call", 0)
        self._call = call + 1

        if call == 0 and tools:
            # Call bash tool
            yield {"type": "tool_call", "id": "c1", "name": "bash", "input": {"command": "echo caveman"}}
            yield {"type": "done", "stop_reason": "tool_use", "usage": {}}
        elif call == 1 and tools:
            # Call file_list
            yield {"type": "tool_call", "id": "c2", "name": "file_list", "input": {"path": "."}}
            yield {"type": "done", "stop_reason": "tool_use", "usage": {}}
        else:
            yield {"type": "delta", "text": "All tools executed successfully."}
            yield {"type": "done", "stop_reason": "end_turn", "usage": {}}


@pytest.mark.asyncio
async def test_builtin_tools_auto_registered():
    """AgentLoop should auto-register 6 builtin tools."""
    loop = AgentLoop(model="mock", provider=ToolTestProvider())
    schemas = loop.tool_registry.get_schemas()
    names = {s["name"] for s in schemas}
    assert "bash" in names
    assert "file_read" in names
    assert "file_write" in names
    assert "file_edit" in names
    assert "file_list" in names
    assert "web_search" in names
    assert "browser" in names
    assert len(names) >= 7


@pytest.mark.asyncio
async def test_multi_tool_chain():
    """Test sequential tool calls (bash → file_list → done)."""
    loop = AgentLoop(model="mock", provider=ToolTestProvider())
    result = await loop.run("List files after running a command")
    assert "successfully" in result.lower()


@pytest.mark.asyncio
async def test_memory_persists():
    """Test that memory is stored after a run."""
    import tempfile
    from caveman.memory.manager import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        mem = MemoryManager(base_dir=td)
        loop = AgentLoop(model="mock", provider=ToolTestProvider(), memory_manager=mem)
        await loop.run("test task")
        # Memory should have stored the episode
        entries = await mem.recall("test task")
        assert len(entries) >= 1


@pytest.mark.asyncio
async def test_trajectory_recorded():
    """Test that trajectory is recorded during a run."""
    from caveman.trajectory.recorder import TrajectoryRecorder
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rec = TrajectoryRecorder(base_dir=td)
        loop = AgentLoop(model="mock", provider=ToolTestProvider(), trajectory_recorder=rec)
        await loop.run("do something")
        traj = rec.to_sharegpt()
        assert len(traj) >= 2  # at least user + assistant
        assert traj[0]["from"] == "human"
