"""MCP Client — connect to external MCP servers and use their tools.

Supports stdio (spawn process) and SSE (HTTP) transports.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=3)
                except (ProcessLookupError, asyncio.TimeoutError):
                    pass
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
        try:
            # Initialize handshake
            await self._send_jsonrpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "caveman", "version": "0.1.0"},
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
        self._http_client = httpx.AsyncClient(base_url=self.url, timeout=30)
        try:
            # Initialize
            resp = await self._http_client.post("/mcp/v1", json={
                "jsonrpc": "2.0", "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "caveman", "version": "0.1.0"},
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
        line = await asyncio.wait_for(self._process.stdout.readline(), timeout=10)
        return json.loads(line.decode())

    async def _send_and_recv(self, method: str, params: dict) -> dict:
        await self._send_jsonrpc(method, params)
        return await self._recv_jsonrpc()
