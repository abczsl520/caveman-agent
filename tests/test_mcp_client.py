"""Tests for MCP client, manager, and mcp_tool (Round 89)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caveman.mcp.client import MCPClient
from caveman.mcp.manager import MCPManager


# ── MCPClient stdio tests ──

@pytest.mark.asyncio
async def test_mcp_client_stdio_connect():
    """MCPClient connects via stdio, discovers tools."""
    client = MCPClient("test-server", command=["echo", "hi"])

    # Mock the subprocess
    mock_proc = AsyncMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock()

    # Prepare responses: initialize response, then tools/list response
    init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "serverInfo": {"name": "test", "version": "1.0"},
    }}).encode() + b"\n"
    tools_resp = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {
        "tools": [
            {"name": "add", "description": "Add numbers", "inputSchema": {"type": "object"}},
            {"name": "multiply", "description": "Multiply numbers", "inputSchema": {"type": "object"}},
        ],
    }}).encode() + b"\n"

    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.readline = AsyncMock(side_effect=[init_resp, tools_resp])
    mock_proc.stderr = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        await client.connect()

    assert len(client.list_tools()) == 2
    assert client._tools["add"]["description"] == "Add numbers"


@pytest.mark.asyncio
async def test_mcp_client_call_tool():
    """MCPClient.call_tool sends JSON-RPC and returns result."""
    client = MCPClient("test", command=["echo"])
    mock_proc = AsyncMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    client._process = mock_proc

    call_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "content": [{"type": "text", "text": "42"}],
    }}).encode() + b"\n"
    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.readline = AsyncMock(return_value=call_resp)

    result = await client.call_tool("add", {"a": 1, "b": 2})
    assert result["content"][0]["text"] == "42"


@pytest.mark.asyncio
async def test_mcp_client_disconnect():
    """MCPClient.disconnect terminates the process."""
    client = MCPClient("test", command=["echo"])
    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock()
    client._process = mock_proc

    await client.disconnect()
    mock_proc.terminate.assert_called_once()
    assert client._process is None


# ── MCPManager tests ──

@pytest.mark.asyncio
async def test_mcp_manager_add_and_list():
    """MCPManager.add_server + get_all_tools prefixes names."""
    mgr = MCPManager()

    mock_client = AsyncMock(spec=MCPClient)
    mock_client.name = "math"
    mock_client.list_tools.return_value = [
        {"name": "add", "description": "Add"},
        {"name": "sub", "description": "Subtract"},
    ]

    with patch.object(mgr, "add_server") as mock_add:
        # Simulate what add_server does
        mgr._clients["math"] = mock_client
        mock_add.return_value = mock_client

    tools = mgr.get_all_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "math__add"
    assert tools[0]["_server"] == "math"
    assert tools[0]["_original_name"] == "add"


@pytest.mark.asyncio
async def test_mcp_manager_call_tool():
    """MCPManager.call_tool routes to the right client."""
    mgr = MCPManager()
    mock_client = AsyncMock(spec=MCPClient)
    mock_client.call_tool.return_value = {"result": 42}
    mgr._clients["math"] = mock_client

    result = await mgr.call_tool("math__add", {"a": 1, "b": 2})
    mock_client.call_tool.assert_called_once_with("add", {"a": 1, "b": 2})
    assert result == {"result": 42}


@pytest.mark.asyncio
async def test_mcp_manager_call_unknown_tool():
    """MCPManager.call_tool raises on unknown prefix."""
    mgr = MCPManager()
    with pytest.raises(ValueError, match="Unknown MCP tool"):
        await mgr.call_tool("nope__add")


@pytest.mark.asyncio
async def test_mcp_manager_disconnect_all():
    """MCPManager.disconnect_all cleans up all clients."""
    mgr = MCPManager()
    c1 = AsyncMock(spec=MCPClient)
    c2 = AsyncMock(spec=MCPClient)
    mgr._clients = {"a": c1, "b": c2}

    await mgr.disconnect_all()
    c1.disconnect.assert_called_once()
    c2.disconnect.assert_called_once()
    assert len(mgr._clients) == 0


# ── mcp_tool tests ──

@pytest.mark.asyncio
async def test_mcp_tool_connect():
    """mcp_connect tool calls MCPManager.add_server."""
    from caveman.tools.builtin.mcp_tool import mcp_connect

    mock_mgr = AsyncMock(spec=MCPManager)
    mock_client = MagicMock()
    mock_client.list_tools.return_value = [{"name": "foo"}, {"name": "bar"}]
    mock_mgr.add_server.return_value = mock_client

    result = await mcp_connect(
        name="test", command="python -m myserver", _context={"mcp_manager": mock_mgr},
    )
    assert result["ok"] is True
    assert result["tools"] == ["foo", "bar"]
    mock_mgr.add_server.assert_called_once_with("test", command=["python", "-m", "myserver"], url=None)


@pytest.mark.asyncio
async def test_mcp_tool_list():
    """mcp_list_tools returns all tools from manager."""
    from caveman.tools.builtin.mcp_tool import mcp_list_tools

    mock_mgr = MagicMock(spec=MCPManager)
    mock_mgr.get_all_tools.return_value = [{"name": "s__t", "_server": "s"}]

    result = await mcp_list_tools(_context={"mcp_manager": mock_mgr})
    assert len(result) == 1
    assert result[0]["name"] == "s__t"


@pytest.mark.asyncio
async def test_mcp_tool_call():
    """mcp_call routes through MCPManager."""
    from caveman.tools.builtin.mcp_tool import mcp_call

    mock_mgr = AsyncMock(spec=MCPManager)
    mock_mgr.call_tool.return_value = {"answer": 42}

    result = await mcp_call(
        tool_name="math__add", arguments={"a": 1}, _context={"mcp_manager": mock_mgr},
    )
    assert result["ok"] is True
    assert result["result"]["answer"] == 42


@pytest.mark.asyncio
async def test_mcp_tool_disconnect():
    """mcp_disconnect calls MCPManager.remove_server."""
    from caveman.tools.builtin.mcp_tool import mcp_disconnect

    mock_mgr = AsyncMock(spec=MCPManager)

    result = await mcp_disconnect(name="test", _context={"mcp_manager": mock_mgr})
    assert result["ok"] is True
    mock_mgr.remove_server.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_mcp_tool_no_context():
    """mcp tools return error when context is missing."""
    from caveman.tools.builtin.mcp_tool import mcp_connect, mcp_list_tools, mcp_call, mcp_disconnect

    assert (await mcp_connect("x"))["error"]
    assert (await mcp_list_tools())[0]["error"]
    assert (await mcp_call("x"))["error"]
    assert (await mcp_disconnect("x"))["error"]
