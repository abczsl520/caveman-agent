"""MCP Client — connect to external MCP servers and use their tools.

Supports stdio (spawn process) and SSE (HTTP) transports.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from caveman.timeouts import MCP_PROCESS_KILL, MCP_PROCESS_STOP, MCP_READLINE

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCP_CLIENT_VERSION",
    "MCPClient",
]


logger = logging.getLogger(__name__)

# Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_CLIENT_VERSION = "0.1.0"


class MCPClient:
    """Connect to an MCP server and call its tools."""

    def __init__(self, name: str, command: list[str] | None = None, url: str | None = None):
        self.name = name
        self.command = command
        self.url = url
        self._process: asyncio.subprocess.Process | None = None
        self._http_client: Any = None
        self._tools: dict[str, dict] = {}
        self._request_id = 0
        self._stderr_task: asyncio.Task | None = None

    # ── Public API ──

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools."""
        if self.command:
            await self._connect_stdio()
        elif self.url:
            await self._connect_sse()
        else:
            raise ValueError("Need either command or url")

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        """Call a tool on the MCP server."""
        if self.command and self._process:
            resp = await self._send_and_recv("tools/call", {
                "name": name, "arguments": arguments or {},
            })
            if "error" in resp:
                raise RuntimeError(f"MCP tool '{name}' error: {resp['error']}")
            return resp.get("result", {})
        elif self.url and self._http_client:
            resp = await self._http_client.post("/mcp/v1", json={
                "jsonrpc": "2.0", "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            })
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP tool '{name}' error: {data['error']}")
            return data.get("result", {})
        raise RuntimeError("Not connected")

    def list_tools(self) -> list[dict]:
        return list(self._tools.values())

    async def disconnect(self) -> None:
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
            self._stderr_task = None
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=MCP_PROCESS_STOP)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=MCP_PROCESS_KILL)
                except (ProcessLookupError, asyncio.TimeoutError) as exc:
                    logger.debug("disconnect: suppressed %s", exc)
            self._process = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ── Stdio transport ──

    async def _connect_stdio(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Drain stderr in background to prevent pipe deadlock
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            # Initialize handshake
            await self._send_jsonrpc("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "caveman", "version": MCP_CLIENT_VERSION},
            })
            await self._recv_jsonrpc()  # initialize response
            # Send initialized notification
            await self._send_jsonrpc("notifications/initialized", {}, is_notification=True)
            # Discover tools
            tools_resp = await self._send_and_recv("tools/list", {})
            for t in tools_resp.get("result", {}).get("tools", []):
                self._tools[t["name"]] = t
        except (asyncio.TimeoutError, OSError, json.JSONDecodeError) as e:
            # Clean up the zombie process on handshake failure
            await self.disconnect()
            raise RuntimeError(f"MCP stdio handshake failed for '{self.name}': {e}") from e

    # ── SSE / HTTP transport ──

    async def _connect_sse(self) -> None:
        import httpx
        self._http_client = httpx.AsyncClient(base_url=self.url, timeout=MCP_PROCESS_KILL0)
        try:
            # Initialize
            resp = await self._http_client.post("/mcp/v1", json={
                "jsonrpc": "2.0", "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "caveman", "version": MCP_CLIENT_VERSION},
                },
            })
            resp.raise_for_status()
            # Discover tools
            tools_resp = await self._http_client.post("/mcp/v1", json={
                "jsonrpc": "2.0", "id": self._next_id(),
                "method": "tools/list", "params": {},
            })
            tools_resp.raise_for_status()
            for t in tools_resp.json().get("result", {}).get("tools", []):
                self._tools[t["name"]] = t
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"MCP SSE connection failed for '{self.name}': {e}") from e

    # ── JSON-RPC helpers ──

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_jsonrpc(self, method: str, params: dict, is_notification: bool = False) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not is_notification:
            msg["id"] = self._next_id()
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def _recv_jsonrpc(self) -> dict:
        line = await asyncio.wait_for(self._process.stdout.readline(), timeout=MCP_READLINE)
        return json.loads(line.decode())

    async def _send_and_recv(self, method: str, params: dict, timeout: float = 30) -> dict:
        """Send a JSON-RPC request and wait for a response with an id.

        Skips notifications (messages without 'id') that may arrive before
        the actual response. Accepts the first id-bearing response to stay
        compatible with servers that use their own id schemes.
        """
        await self._send_jsonrpc(method, params)
        request_id = self._request_id  # ID assigned by _send_jsonrpc
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"No response for request {request_id} within {timeout}s")
            line = await asyncio.wait_for(self._process.stdout.readline(), timeout=remaining)
            msg = json.loads(line.decode())
            # Skip notifications (no 'id' field)
            if "id" not in msg:
                continue
            # Accept response (ideally id matches, but some servers echo their own)
            return msg

    async def _drain_stderr(self) -> None:
        """Read stderr lines and log at debug level to prevent pipe deadlock."""
        try:
            while self._process and self._process.stderr:
                line = await self._process.stderr.readline()
                if not isinstance(line, (bytes, bytearray)):
                    break  # Not a real pipe (e.g. mock)
                if not line:
                    break
                logger.debug("[MCP %s stderr] %s", self.name, line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("_drain_stderr: suppressed %s", exc)
