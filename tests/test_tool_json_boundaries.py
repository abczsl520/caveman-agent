from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from caveman.bridge.cli_transport import CLITransport
from caveman.tools.builtin import checkpoint_tool, gateway_tool, mcp_tool, todo_tool
from caveman.tools.builtin.mcp_client import MCPClient, MCPServer
from caveman.trajectory.recorder import TrajectoryRecorder


def test_todo_load_fails_closed_for_mixed_items_and_ignores_wrong_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    todo_file = tmp_path / "todos.json"
    monkeypatch.setattr(todo_tool, "_TODO_FILE", todo_file)

    todo_file.write_text(json.dumps([{"id": "1", "status": "pending"}, "bad", 3]), encoding="utf-8")
    assert asyncio.run(todo_tool.todo_list("all")) == [{"error": "todos.json must contain a list of objects"}]
    assert asyncio.run(todo_tool.todo_add("new")) == {"error": "todos.json must contain a list of objects"}
    assert json.loads(todo_file.read_text(encoding="utf-8")) == [{"id": "1", "status": "pending"}, "bad", 3]

    todo_file.write_text(json.dumps({"id": "not-a-list"}), encoding="utf-8")
    assert asyncio.run(todo_tool.todo_list("all")) == []


def test_todo_finish_and_remove_tolerate_dicts_missing_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    todo_file = tmp_path / "todos.json"
    monkeypatch.setattr(todo_tool, "_TODO_FILE", todo_file)
    todo_file.write_text(json.dumps([{"title": "no id"}, {"id": "1", "status": "pending"}]), encoding="utf-8")

    assert asyncio.run(todo_tool.todo_finish("1")) == {"ok": True}
    data = json.loads(todo_file.read_text(encoding="utf-8"))
    assert data == [{"title": "no id"}, {"id": "1", "status": "finished"}]

    assert asyncio.run(todo_tool.todo_remove("missing")) == {"error": "Todo missing not found"}


@pytest.mark.parametrize("value", [[], "text", 123])
def test_trajectory_load_requires_json_object(tmp_path: Path, value: object) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        TrajectoryRecorder.load(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ('[1, 2]', {"result": [1, 2], "success": True}),
        ('plain text', {"result": "plain text", "success": True}),
    ],
)
def test_cli_transport_parse_result_normalizes_non_dict_json(raw: str, expected: dict[str, object]) -> None:
    assert CLITransport._parse_result({"stdout": raw, "stderr": "", "returncode": 0}) == expected


class _BadCheckpointManager:
    async def list_checkpoints(self, session_id: str | None) -> object:
        return {"not": "a-list"}


class _MixedCheckpointManager:
    async def list_checkpoints(self, session_id: str | None) -> object:
        return [{"id": "ok"}, "bad"]


def test_checkpoint_list_normalizes_manager_shape() -> None:
    assert asyncio.run(checkpoint_tool.checkpoint_list(_context={"checkpoint_manager": _BadCheckpointManager()})) == [
        {"error": "checkpoint_manager returned invalid checkpoint list"}
    ]
    assert asyncio.run(checkpoint_tool.checkpoint_list(_context={"checkpoint_manager": _MixedCheckpointManager()})) == [
        {"error": "checkpoint_manager returned malformed checkpoint item"}
    ]


class _BadGatewayRouter:
    def list_gateways(self) -> object:
        return {"not": "a-list"}


class _MixedGatewayRouter:
    def list_gateways(self) -> object:
        return [{"name": "discord"}, None]


def test_gateway_list_normalizes_router_shape() -> None:
    assert asyncio.run(gateway_tool.gateway_list(_context={"gateway_router": _BadGatewayRouter()})) == [
        {"error": "gateway_router returned invalid gateway list"}
    ]
    assert asyncio.run(gateway_tool.gateway_list(_context={"gateway_router": _MixedGatewayRouter()})) == [
        {"error": "gateway_router returned malformed gateway item"}
    ]


class _BadMCPManager:
    def get_all_tools(self) -> object:
        return {"not": "a-list"}


class _MixedMCPManager:
    def get_all_tools(self) -> object:
        return [{"name": "tool"}, object()]


def test_mcp_list_tools_normalizes_manager_shape() -> None:
    assert asyncio.run(mcp_tool.mcp_list_tools(_context={"mcp_manager": _BadMCPManager()})) == [
        {"error": "mcp_manager returned invalid tool list"}
    ]
    assert asyncio.run(mcp_tool.mcp_list_tools(_context={"mcp_manager": _MixedMCPManager()})) == [
        {"error": "mcp_manager returned malformed tool item"}
    ]


def test_mcp_send_request_rejects_non_dict_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stdin:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    server = MCPServer(name="s", command="cmd")
    server._process = SimpleNamespace(stdin=_Stdin())
    client = MCPClient()

    async def fake_wait_for(future: asyncio.Future[object], timeout: float) -> object:
        return ["not", "dict"]

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    result = asyncio.run(client._send_request(server, "tools/call", {}))

    assert result == {"error": "Invalid response for tools/call"}
    assert server._pending == {}
