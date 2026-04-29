"""MCP tools — connect to external MCP servers and use their tools."""
from __future__ import annotations

import logging
from typing import Any

from caveman.tools.registry import tool

__all__ = [
    "mcp_connect",
    "mcp_list_tools",
    "mcp_call",
    "mcp_disconnect",
]


logger = logging.getLogger(__name__)


@tool(
    name="mcp_connect",
    description="Connect to an external MCP server",
    params={
        "name": {"type": "string", "description": "Server name (used as prefix)"},
        "command": {"type": "string", "description": "Space-separated command to spawn stdio server"},
        "url": {"type": "string", "description": "URL for SSE/HTTP server"},
    },
    required=["name"],
)
async def mcp_connect(
    name: str, command: str = "", url: str = "", _context: dict | None = None,
) -> dict:
    """Connect to an MCP server and discover its tools."""
    mgr = (_context or {}).get("mcp_manager")
    if not mgr:
        return {"error": "mcp_manager not available"}
    cmd_list = command.split() if command else None
    url_val = url or None

    # OSV malware check for package-based MCP servers
    if cmd_list and len(cmd_list) >= 1:
        try:
            from caveman.security.osv_check import check_package_for_malware
            warning = check_package_for_malware(cmd_list[0], cmd_list[1:])
            if warning:
                return {"error": warning}
        except Exception:
            pass  # OSV check is best-effort

    try:
        client = await mgr.add_server(name, command=cmd_list, url=url_val)
    except Exception as e:
        logger.warning("mcp_connect failed for %s: %s", name, e)
        return {"error": f"Failed to connect to MCP server '{name}': {e}"}
    tool_names = [t["name"] for t in client.list_tools()]
    return {"ok": True, "server": name, "tools": tool_names}


@tool(
    name="mcp_list_tools",
    description="List all available MCP tools from connected servers",
    params={},
    required=[],
)
async def mcp_list_tools(_context: dict | None = None) -> list[dict[str, Any]]:
    """List all tools from all connected MCP servers."""
    mgr = (_context or {}).get("mcp_manager")
    if not mgr:
        return [{"error": "mcp_manager not available"}]
    tools = mgr.get_all_tools()
    if not isinstance(tools, list):
        return [{"error": "mcp_manager returned invalid tool list"}]
    if not all(isinstance(item, dict) for item in tools):
        return [{"error": "mcp_manager returned malformed tool item"}]
    return tools


@tool(
    name="mcp_call",
    description="Call a tool on a connected MCP server",
    params={
        "tool_name": {"type": "string", "description": "Prefixed tool name (server__tool)"},
        "arguments": {"type": "object", "description": "Tool arguments", "default": {}},
    },
    required=["tool_name"],
)
async def mcp_call(
    tool_name: str, arguments: dict | None = None, _context: dict | None = None,
) -> dict:
    """Call an MCP tool by its prefixed name."""
    mgr = (_context or {}).get("mcp_manager")
    if not mgr:
        return {"error": "mcp_manager not available"}
    try:
        result = await mgr.call_tool(tool_name, arguments or {})
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}
    return {"ok": True, "result": result}


@tool(
    name="mcp_disconnect",
    description="Disconnect from an MCP server",
    params={
        "name": {"type": "string", "description": "Server name to disconnect"},
    },
    required=["name"],
)
async def mcp_disconnect(name: str, _context: dict | None = None) -> dict:
    """Disconnect from a named MCP server."""
    mgr = (_context or {}).get("mcp_manager")
    if not mgr:
        return {"error": "mcp_manager not available"}
    try:
        await mgr.remove_server(name)
    except Exception as e:
        logger.warning("mcp_disconnect failed for %s: %s", name, e)
        return {"error": f"Failed to disconnect from '{name}': {e}"}
    return {"ok": True, "server": name}
