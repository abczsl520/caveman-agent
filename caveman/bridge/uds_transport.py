"""Unix Domain Socket (UDS) high-frequency bridge transport.

UDS = 0-latency IPC for local Caveman ↔ OpenClaw/Hermes communication.
HTTP bridge works but adds ~5ms overhead per call. UDS eliminates that.

Protocol: JSON-RPC 2.0 over UDS stream (same as MCP, different transport).

Usage:
  server = UDSServer("/tmp/caveman.sock", handler=my_handler)
  await server.start()

  client = UDSClient("/tmp/caveman.sock")
  result = await client.call("tools/list", {})
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

from caveman.paths import UDS_SOCK, OPENCLAW_SOCK

# Re-export for backward compat
CAVEMAN_SOCK = UDS_SOCK


class UDSClient:
    """JSON-RPC client over Unix Domain Socket."""

    def __init__(self, socket_path: str = CAVEMAN_SOCK, timeout: float = 30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id = 0
        self._connected = False

    async def connect(self) -> bool:
        """Connect to UDS server."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=self.timeout,
            )
            self._connected = True
            logger.info("UDS connected: %s", self.socket_path)
            return True
        except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.warning("UDS connect failed: %s (%s)", self.socket_path, e)
            return False

    async def disconnect(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception as e:
                logger.debug("Suppressed in disconnect: %s", e)
        self._connected = False

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict:
        """Send JSON-RPC request and wait for response."""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        # Send: JSON + newline delimiter
        data = json.dumps(request, ensure_ascii=False) + "\n"
        self._writer.write(data.encode())
        await self._writer.drain()

        # Read response
        try:
            line = await asyncio.wait_for(
                self._reader.readline(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"UDS call {method} timed out after {self.timeout}s")

        if not line:
            raise ConnectionError("UDS connection closed by server")

        response = json.loads(line.decode())

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"RPC error {err.get('code', -1)}: {err.get('message', 'unknown')}")

        return response.get("result", {})

    @property
    def is_connected(self) -> bool:
        return self._connected


class UDSServer:
    """JSON-RPC server over Unix Domain Socket."""

    def __init__(
        self,
        socket_path: str = CAVEMAN_SOCK,
        handler: Callable[[str, dict], Awaitable[dict]] | None = None,
    ):
        self.socket_path = socket_path
        self.handler = handler or self._default_handler
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start UDS server."""
        # Clean up stale socket
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path
        )
        # Set permissions: owner only
        from caveman.paths import UDS_SOCKET_MODE
        os.chmod(self.socket_path, UDS_SOCKET_MODE)
        logger.info("UDS server started: %s", self.socket_path)

    async def stop(self) -> None:
        """Stop server and clean up."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Clean socket file
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()
        # Cancel clients
        for task in self._clients:
            task.cancel()
        logger.info("UDS server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a connected client."""
        task = asyncio.current_task()
        if task:
            self._clients.add(task)

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.decode())
                except json.JSONDecodeError:
                    from caveman.paths import JSONRPC_PARSE_ERROR
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": JSONRPC_PARSE_ERROR, "message": "Parse error"},
                    }
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                req_id = request.get("id")
                method = request.get("method", "")
                params = request.get("params", {})

                try:
                    result = await self.handler(method, params)
                    response = {"jsonrpc": "2.0", "id": req_id, "result": result}
                except Exception as e:
                    from caveman.paths import JSONRPC_INTERNAL_ERROR
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": JSONRPC_INTERNAL_ERROR, "message": str(e)},
                    }

                writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            if task:
                self._clients.discard(task)

    @staticmethod
    async def _default_handler(method: str, params: dict) -> dict:
        """Default echo handler for testing."""
        return {"method": method, "params": params, "echo": True}
