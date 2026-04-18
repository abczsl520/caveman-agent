"""Tests for Round 90: Checkpoint system + Gateway abstraction layer."""
from __future__ import annotations
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Checkpoint tests ──

from caveman.agent.checkpoint import Checkpoint, CheckpointManager


@pytest.mark.asyncio
async def test_checkpoint_save_restore(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    cp = Checkpoint("sess1", [{"role": "user", "content": "hello"}], {"task": "test"})
    cp_id = await mgr.save(cp)
    assert cp_id.startswith("sess1_")
    # Restore latest
    restored = await mgr.restore("sess1")
    assert restored is not None
    assert restored.session_id == "sess1"
    assert len(restored.messages) == 1
    assert restored.messages[0]["content"] == "hello"
    assert restored.metadata["task"] == "test"


@pytest.mark.asyncio
async def test_checkpoint_restore_specific(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    # Manually create two checkpoints with distinct IDs
    for name, content in [("sess2_20250101_000001", "first"), ("sess2_20250101_000002", "second")]:
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "id": name, "session_id": "sess2",
            "messages": [{"role": "user", "content": content}],
            "metadata": {}, "created_at": "2025-01-01T00:00:00",
        }))
    # Restore specific (the first one)
    restored = await mgr.restore("sess2", "sess2_20250101_000001")
    assert restored.messages[0]["content"] == "first"


@pytest.mark.asyncio
async def test_checkpoint_restore_nonexistent(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    result = await mgr.restore("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_checkpoint_list(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    cp = Checkpoint("sess3", [{"role": "user", "content": "hi"}])
    await mgr.save(cp)
    items = await mgr.list_checkpoints("sess3")
    assert len(items) == 1
    assert items[0]["session_id"] == "sess3"


@pytest.mark.asyncio
async def test_checkpoint_cleanup(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    # Save 12 checkpoints — only 10 should survive
    for i in range(12):
        cp = Checkpoint("sess4", [{"role": "user", "content": f"msg{i}"}])
        # Manually create unique filenames to avoid timestamp collision
        cp_id = f"sess4_{i:04d}"
        path = tmp_path / f"{cp_id}.json"
        path.write_text(json.dumps({
            "id": cp_id, "session_id": "sess4",
            "messages": cp.messages, "metadata": {},
            "created_at": cp.created_at,
        }))
    # Trigger cleanup
    await mgr._cleanup("sess4", keep=10)
    remaining = list(tmp_path.glob("sess4_*.json"))
    assert len(remaining) == 10


# ── Gateway Router tests ──

from caveman.gateway.router import GatewayRouter
from caveman.gateway.base import Gateway


class FakeGateway(Gateway):
    """Minimal concrete gateway for testing."""

    def __init__(self, gw_name: str):
        self._name = gw_name
        self._running = False
        self._sent: list[tuple[str, str]] = []
        self._handler = None

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send_message(self, channel_id: str, text: str) -> None:
        self._sent.append((channel_id, text))

    async def on_message(self, handler) -> None:
        self._handler = handler


@pytest.mark.asyncio
async def test_gateway_router_register_send():
    router = GatewayRouter()
    gw = FakeGateway("test_gw")
    router.register(gw)
    result = await router.send("test_gw", "chan1", "hello")
    assert result["ok"] is True
    assert gw._sent == [("chan1", "hello")]


@pytest.mark.asyncio
async def test_gateway_router_list():
    router = GatewayRouter()
    gw1 = FakeGateway("discord")
    gw2 = FakeGateway("telegram")
    router.register(gw1)
    router.register(gw2)
    items = router.list_gateways()
    assert len(items) == 2
    names = {i["name"] for i in items}
    assert names == {"discord", "telegram"}


@pytest.mark.asyncio
async def test_gateway_router_unknown():
    router = GatewayRouter()
    with pytest.raises(ValueError, match="Unknown gateway"):
        await router.send("nope", "chan", "hi")


@pytest.mark.asyncio
async def test_gateway_router_start_stop():
    router = GatewayRouter()
    gw = FakeGateway("test")
    router.register(gw)
    await router.start_all()
    assert gw.is_running is True
    await router.stop_all()
    assert gw.is_running is False


# ── Gateway tool tests ──

from caveman.tools.builtin.gateway_tool import gateway_send, gateway_list


@pytest.mark.asyncio
async def test_gateway_tool_send():
    mock_router = AsyncMock()
    mock_router.send = AsyncMock(return_value={"ok": True, "gateway": "discord", "channel_id": "123"})
    ctx = {"gateway_router": mock_router}
    result = await gateway_send("discord", "123", "hello", _context=ctx)
    assert result["ok"] is True
    mock_router.send.assert_called_once_with("discord", "123", "hello")


@pytest.mark.asyncio
async def test_gateway_tool_list():
    mock_router = MagicMock()
    mock_router.list_gateways.return_value = [{"name": "discord", "running": True}]
    ctx = {"gateway_router": mock_router}
    result = await gateway_list(_context=ctx)
    assert len(result) == 1
    assert result[0]["name"] == "discord"


@pytest.mark.asyncio
async def test_gateway_tool_no_context():
    result = await gateway_send("discord", "123", "hi", _context=None)
    assert "error" in result


# ── Checkpoint tool tests ──

from caveman.tools.builtin.checkpoint_tool import checkpoint_save, checkpoint_restore, checkpoint_list


@pytest.mark.asyncio
async def test_checkpoint_tool_save(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    ctx = {"checkpoint_manager": mgr, "messages": [{"role": "user", "content": "hi"}]}
    result = await checkpoint_save("sess_tool", metadata={"x": 1}, _context=ctx)
    assert result["ok"] is True
    assert "checkpoint_id" in result


@pytest.mark.asyncio
async def test_checkpoint_tool_restore(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    cp = Checkpoint("sess_tool2", [{"role": "user", "content": "hi"}])
    await mgr.save(cp)
    ctx = {"checkpoint_manager": mgr}
    result = await checkpoint_restore("sess_tool2", _context=ctx)
    assert result["ok"] is True
    assert result["messages_count"] == 1


@pytest.mark.asyncio
async def test_checkpoint_tool_restore_not_found(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    ctx = {"checkpoint_manager": mgr}
    result = await checkpoint_restore("nope", _context=ctx)
    assert "error" in result


@pytest.mark.asyncio
async def test_checkpoint_tool_list(tmp_path):
    mgr = CheckpointManager(base_dir=tmp_path)
    cp = Checkpoint("sess_list", [{"role": "user", "content": "hi"}])
    await mgr.save(cp)
    ctx = {"checkpoint_manager": mgr}
    result = await checkpoint_list(session_id="sess_list", _context=ctx)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_checkpoint_tool_no_context():
    result = await checkpoint_save("x", _context=None)
    assert "error" in result
