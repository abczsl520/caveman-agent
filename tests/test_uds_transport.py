"""Tests for bridge/uds_transport.py — Unix Domain Socket transport."""
import asyncio
import os
import pytest
from caveman.bridge.uds_transport import UDSClient, UDSServer


@pytest.fixture
def sock_path():
    """Short socket path to avoid AF_UNIX length limit."""
    path = f"/tmp/cave_test_{os.getpid()}.sock"
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class TestUDSRoundtrip:
    @pytest.mark.asyncio
    async def test_call_and_response(self, sock_path):
        async def handler(method: str, params: dict) -> dict:
            if method == "ping":
                return {"pong": True}
            return {"method": method, "params": params}

        server = UDSServer(sock_path, handler)
        task = asyncio.create_task(server.start())
        await asyncio.sleep(0.15)

        client = UDSClient(sock_path)
        connected = await client.connect()
        assert connected is True
        assert client.is_connected

        result = await client.call("ping")
        assert result["pong"] is True

        result2 = await client.call("echo", {"msg": "hello"})
        assert result2["method"] == "echo"
        assert result2["params"]["msg"] == "hello"

        await client.disconnect()
        assert not client.is_connected

        await server.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestUDSClient:
    @pytest.mark.asyncio
    async def test_connect_nonexistent_returns_false(self):
        client = UDSClient("/tmp/cave_nonexistent.sock")
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_call_without_connect_raises(self):
        client = UDSClient("/tmp/cave_nonexistent.sock")
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.call("test")


class TestUDSServer:
    @pytest.mark.asyncio
    async def test_default_handler(self, sock_path):
        server = UDSServer(sock_path)  # Uses default handler
        task = asyncio.create_task(server.start())
        await asyncio.sleep(0.15)

        client = UDSClient(sock_path)
        await client.connect()
        result = await client.call("status")
        assert isinstance(result, dict)

        await client.disconnect()
        await server.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
