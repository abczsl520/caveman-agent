"""Tests for ACP server, client, and tool — Round 98."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── ACPServer direct API tests ──

@pytest.mark.asyncio
async def test_acp_task_lifecycle():
    """Create → get → verify completed."""
    from caveman.acp.server import ACPServer

    server = ACPServer()  # default echo mode (no agent_fn)
    msg = {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
    result = await server.handle_create_task(msg)

    assert result["status"] == "completed"
    assert result["id"].startswith("task-")
    assert result["result"]["parts"][0]["text"] == "Echo: hello"
    assert result["completed_at"] is not None

    # Get same task
    fetched = await server.handle_get_task(result["id"])
    assert fetched is not None
    assert fetched["id"] == result["id"]
    assert fetched["status"] == "completed"


@pytest.mark.asyncio
async def test_acp_task_cancel():
    """Cancel a pending/running task."""
    from caveman.acp.server import ACPServer, ACPTask

    server = ACPServer()
    # Manually insert a running task
    task = ACPTask(id="task-test123", status="running", message={"role": "user", "parts": []})
    server._tasks["task-test123"] = task

    result = await server.handle_cancel_task("task-test123")
    assert result["status"] == "cancelled"
    assert result["completed_at"] is not None


@pytest.mark.asyncio
async def test_acp_task_not_found():
    """Get/cancel non-existent task returns None."""
    from caveman.acp.server import ACPServer

    server = ACPServer()
    assert await server.handle_get_task("task-nonexistent") is None
    assert await server.handle_cancel_task("task-nonexistent") is None


@pytest.mark.asyncio
async def test_acp_task_with_agent_fn():
    """Task runs through a custom agent_fn."""
    from caveman.acp.server import ACPServer

    async def my_agent(msg: str) -> str:
        return f"Processed: {msg}"

    server = ACPServer(agent_fn=my_agent)
    msg = {"role": "user", "parts": [{"type": "text", "text": "do stuff"}]}
    result = await server.handle_create_task(msg)

    assert result["status"] == "completed"
    assert result["result"]["parts"][0]["text"] == "Processed: do stuff"


@pytest.mark.asyncio
async def test_acp_task_agent_fn_error():
    """Task fails gracefully when agent_fn raises."""
    from caveman.acp.server import ACPServer

    async def bad_agent(msg: str) -> str:
        raise RuntimeError("boom")

    server = ACPServer(agent_fn=bad_agent)
    msg = {"role": "user", "parts": [{"type": "text", "text": "fail"}]}
    result = await server.handle_create_task(msg)

    assert result["status"] == "failed"
    assert "boom" in result["result"]["parts"][0]["text"]


@pytest.mark.asyncio
async def test_acp_server_max_tasks():
    """LRU eviction when max tasks reached."""
    from caveman.acp.server import ACPServer, MAX_TASKS

    server = ACPServer()
    # Fill to max
    for i in range(MAX_TASKS):
        msg = {"role": "user", "parts": [{"type": "text", "text": f"task {i}"}]}
        await server.handle_create_task(msg)

    assert len(server._tasks) == MAX_TASKS
    first_id = list(server._tasks.keys())[0]

    # Add one more — should evict the oldest
    msg = {"role": "user", "parts": [{"type": "text", "text": "overflow"}]}
    await server.handle_create_task(msg)

    assert len(server._tasks) == MAX_TASKS
    assert first_id not in server._tasks


# ── ACPClient tests (mock httpx) ──

@pytest.mark.asyncio
async def test_acp_client_send():
    """Client send_task polls until completed."""
    from caveman.acp.client import ACPClient

    mock_client = AsyncMock()
    # First call: POST create → pending
    create_resp = MagicMock()
    create_resp.json.return_value = {"id": "task-abc", "status": "pending"}
    create_resp.raise_for_status = MagicMock()

    # Second call: GET poll → running
    poll1_resp = MagicMock()
    poll1_resp.json.return_value = {"id": "task-abc", "status": "running"}
    poll1_resp.raise_for_status = MagicMock()

    # Third call: GET poll → completed
    poll2_resp = MagicMock()
    poll2_resp.json.return_value = {
        "id": "task-abc",
        "status": "completed",
        "result": {"role": "assistant", "parts": [{"type": "text", "text": "done"}]},
    }
    poll2_resp.raise_for_status = MagicMock()

    mock_client.post = AsyncMock(return_value=create_resp)
    mock_client.get = AsyncMock(side_effect=[poll1_resp, poll2_resp])
    mock_client.aclose = AsyncMock()

    client = ACPClient("http://localhost:8766")
    client._client = mock_client

    result = await client.send_task("hello", poll_interval=0.01)
    assert result["status"] == "completed"
    assert result["result"]["parts"][0]["text"] == "done"

    await client.close()
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_acp_client_send_async():
    """Client send_task_async returns task ID immediately."""
    from caveman.acp.client import ACPClient

    mock_client = AsyncMock()
    resp = MagicMock()
    resp.json.return_value = {"id": "task-xyz"}
    resp.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.aclose = AsyncMock()

    client = ACPClient("http://localhost:8766")
    client._client = mock_client

    task_id = await client.send_task_async("hello")
    assert task_id == "task-xyz"
    await client.close()


# ── ACP tool tests ──

@pytest.mark.asyncio
async def test_acp_tool_send():
    """acp_send tool calls ACPClient correctly."""
    with patch("caveman.acp.client.ACPClient") as MockClient:
        instance = AsyncMock()
        instance.send_task = AsyncMock(return_value={
            "id": "task-t1", "status": "completed",
            "result": {"role": "assistant", "parts": [{"type": "text", "text": "ok"}]},
        })
        instance.close = AsyncMock()
        MockClient.return_value = instance

        from caveman.tools.builtin.acp_tool import acp_send
        result = await acp_send(url="http://localhost:8766", message="test")

        assert result["ok"] is True
        assert result["status"] == "completed"
        instance.send_task.assert_called_once_with("test")
        instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_acp_tool_status():
    """acp_status tool checks task status."""
    with patch("caveman.acp.client.ACPClient") as MockClient:
        instance = AsyncMock()
        instance.get_task = AsyncMock(return_value={"id": "task-t1", "status": "running"})
        instance.close = AsyncMock()
        MockClient.return_value = instance

        from caveman.tools.builtin.acp_tool import acp_status
        result = await acp_status(url="http://localhost:8766", task_id="task-t1")

        assert result["ok"] is True
        assert result["status"] == "running"


@pytest.mark.asyncio
async def test_acp_tool_cancel():
    """acp_cancel tool cancels a task."""
    with patch("caveman.acp.client.ACPClient") as MockClient:
        instance = AsyncMock()
        instance.cancel_task = AsyncMock(return_value={"id": "task-t1", "status": "cancelled"})
        instance.close = AsyncMock()
        MockClient.return_value = instance

        from caveman.tools.builtin.acp_tool import acp_cancel
        result = await acp_cancel(url="http://localhost:8766", task_id="task-t1")

        assert result["ok"] is True
        assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_acp_tool_send_error():
    """acp_send returns error dict on failure."""
    with patch("caveman.acp.client.ACPClient") as MockClient:
        instance = AsyncMock()
        instance.send_task = AsyncMock(side_effect=ConnectionError("refused"))
        instance.close = AsyncMock()
        MockClient.return_value = instance

        from caveman.tools.builtin.acp_tool import acp_send
        result = await acp_send(url="http://localhost:9999", message="test")

        assert "error" in result
        assert "refused" in result["error"]
