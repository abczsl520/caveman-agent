"""MCP Client — Model Context Protocol integration.

Extracted from Hermes mcp_tool.py (2195 lines).
Key patterns: stdio/SSE transport, tool discovery, sampling handler.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from caveman.timeouts import MCP_TOOL_CALL

__all__ = [
    "MCPTool",
    "MCPServer",
    "MCPClient",
]


logger = logging.getLogger("caveman.tools.mcp")


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServer:
    """An MCP server connection."""
    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    url: str = ""  # For HTTP/SSE transport
    env: Dict[str, str] = field(default_factory=dict)
    tools: List[MCPTool] = field(default_factory=list)
    status: str = "disconnected"  # disconnected | connecting | connected | error
    _process: Optional[Any] = field(default=None, repr=False)
    _reader_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _pending: Dict[int, asyncio.Future] = field(default_factory=dict, repr=False)
    _next_id: int = 1

    @property
    def is_stdio(self) -> bool:
        return bool(self.command)

    @property
    def is_http(self) -> bool:
        return bool(self.url)


class MCPClient:
    """Manages MCP server connections and tool dispatch."""

    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}
        self._tools: Dict[str, MCPTool] = {}  # tool_name → MCPTool

    # ── Server Management ──

    async def connect(self, server: MCPServer) -> bool:
        """Connect to an MCP server."""
        self._servers[server.name] = server

        if server.is_stdio:
            return await self._connect_stdio(server)
        elif server.is_http:
            return await self._connect_http(server)
        else:
            logger.error("MCP server %s: no transport configured", server.name)
            return False

    async def disconnect(self, name: str) -> bool:
        """Disconnect from an MCP server."""
        server = self._servers.get(name)
        if not server:
            return False

        if server._process:
            try:
                server._process.terminate()
                await asyncio.sleep(0.5)
                if server._process.returncode is None:
                    server._process.kill()
            except Exception as exc:
                logger.debug("disconnect: suppressed %s", exc)

        if server._reader_task:
            server._reader_task.cancel()

        # Remove tools from this server
        self._tools = {k: v for k, v in self._tools.items() if v.server_name != name}
        server.status = "disconnected"
        server.tools.clear()
        return True

    async def disconnect_all(self) -> None:
        for name in list(self._servers.keys()):
            await self.disconnect(name)

    # ── Tool Discovery ──

    def list_tools(self) -> List[MCPTool]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)

    # ── Tool Invocation ──

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool not found: {tool_name}"}

        server = self._servers.get(tool.server_name)
        if not server or server.status != "connected":
            return {"error": f"Server not connected: {tool.server_name}"}

        return await self._send_request(server, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

    # ── Stdio Transport ──

    async def _connect_stdio(self, server: MCPServer) -> bool:
        """Connect via stdio transport."""
        server.status = "connecting"
        env = dict(os.environ)
        env.update(server.env)

        try:
            proc = await asyncio.create_subprocess_exec(
                server.command, *server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            server._process = proc
            server._reader_task = asyncio.create_task(self._stdio_reader(server))

            # Initialize
            resp = await self._send_request(server, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "caveman", "version": "0.5.0"},
            })

            if "error" in resp:
                server.status = "error"
                return False

            # Send initialized notification
            await self._send_notification(server, "notifications/initialized", {})

            # Discover tools
            tools_resp = await self._send_request(server, "tools/list", {})
            for t in tools_resp.get("tools", []):
                mcp_tool = MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=server.name,
                )
                server.tools.append(mcp_tool)
                self._tools[t["name"]] = mcp_tool

            server.status = "connected"
            logger.info("MCP connected: %s (%d tools)", server.name, len(server.tools))
            return True

        except Exception as e:
            server.status = "error"
            logger.error("MCP connect failed for %s: %s", server.name, e)
            return False

    async def _stdio_reader(self, server: MCPServer) -> None:
        """Read JSON-RPC responses from stdout."""
        proc = server._process
        if not proc or not proc.stdout:
            return

        buffer = b""
        while True:
            try:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk

                # Parse JSON-RPC messages (newline-delimited)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        msg_id = msg.get("id")
                        if msg_id is not None and msg_id in server._pending:
                            server._pending[msg_id].set_result(msg.get("result", msg))
                    except json.JSONDecodeError:
                        pass  # intentional: Exception suppressed

            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _send_request(self, server: MCPServer, method: str, params: Dict) -> Dict:
        """Send a JSON-RPC request and wait for response."""
        if not server._process or not server._process.stdin:
            return {"error": "Not connected"}

        msg_id = server._next_id
        server._next_id += 1

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }) + "\n"

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        server._pending[msg_id] = future

        try:
            server._process.stdin.write(request.encode())
            await server._process.stdin.drain()
            result = await asyncio.wait_for(future, timeout=MCP_TOOL_CALL)
            return result
        except asyncio.TimeoutError:
            return {"error": f"Timeout waiting for {method}"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            server._pending.pop(msg_id, None)

    async def _send_notification(self, server: MCPServer, method: str, params: Dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not server._process or not server._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
        try:
            server._process.stdin.write(msg.encode())
            await server._process.stdin.drain()
        except Exception as exc:
            logger.debug("_send_notification: suppressed %s", exc)

    # ── HTTP/SSE Transport ──

    async def _connect_http(self, server: MCPServer) -> bool:
        """Connect via HTTP/SSE transport (streamable HTTP)."""
        server.status = "connecting"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Initialize
                async with session.post(
                    f"{server.url}/initialize",
                    json={
                        "jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "caveman", "version": "0.5.0"},
                        },
                    },
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status != 200:
                        server.status = "error"
                        return False

                # List tools
                async with session.post(
                    f"{server.url}/tools/list",
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    data = await resp.json()
                    for t in data.get("result", {}).get("tools", []):
                        mcp_tool = MCPTool(
                            name=t["name"],
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema", {}),
                            server_name=server.name,
                        )
                        server.tools.append(mcp_tool)
                        self._tools[t["name"]] = mcp_tool

            server.status = "connected"
            logger.info("MCP HTTP connected: %s (%d tools)", server.name, len(server.tools))
            return True

        except Exception as e:
            server.status = "error"
            logger.error("MCP HTTP connect failed: %s", e)
            return False

    # ── Info ──

    def list_servers(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name,
                "status": s.status,
                "transport": "stdio" if s.is_stdio else "http",
                "tools": len(s.tools),
            }
            for s in self._servers.values()
        ]
