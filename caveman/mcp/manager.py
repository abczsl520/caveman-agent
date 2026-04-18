"""MCP Manager — manage multiple MCP server connections."""
from __future__ import annotations

import logging
from typing import Any

from caveman.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class MCPManager:
    """Manage multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    async def add_server(
        self, name: str, command: list[str] | None = None, url: str | None = None,
    ) -> MCPClient:
        """Connect to an MCP server and register it."""
        client = MCPClient(name, command=command, url=url)
        await client.connect()
        self._clients[name] = client
        return client

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all connected servers, prefixed with server name."""
        tools = []
        for server_name, client in self._clients.items():
            for t in client.list_tools():
                tools.append({
                    **t,
                    "name": f"{server_name}__{t['name']}",
                    "_server": server_name,
                    "_original_name": t["name"],
                })
        return tools

    async def call_tool(self, prefixed_name: str, arguments: dict | None = None) -> Any:
        """Call a tool by its prefixed name (server__tool)."""
        parts = prefixed_name.split("__", 1)
        if len(parts) != 2 or parts[0] not in self._clients:
            raise ValueError(f"Unknown MCP tool: {prefixed_name}")
        return await self._clients[parts[0]].call_tool(parts[1], arguments)

    async def remove_server(self, name: str) -> None:
        """Disconnect and remove a server."""
        client = self._clients.pop(name, None)
        if client:
            await client.disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        errors = []
        for name, client in self._clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning("Failed to disconnect MCP server %s: %s", name, e)
                errors.append((name, e))
        self._clients.clear()
        if errors:
            logger.warning("Errors during disconnect_all: %s", errors)

    @property
    def server_names(self) -> list[str]:
        return list(self._clients.keys())
