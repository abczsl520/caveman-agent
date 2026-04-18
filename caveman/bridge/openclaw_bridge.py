"""OpenClaw Bridge — connect Caveman to OpenClaw's messaging and tool ecosystem.

Architecture (Route C — layered parasitism):
  Layer 1: CLI subprocess via `openclaw agent` — simple, reliable, handles auth
  Layer 2: WebSocket gateway direct — lower latency, bidirectional events
  Layer 3 (goal): Caveman as MCP server for OpenClaw — reverse integration

Transport is abstracted behind OpenClawBridge so layers can be swapped without
changing callers.
"""
from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class OpenClawBridge:
    """Bridge between Caveman and OpenClaw.

    Transport selection:
      - "auto": try WebSocket first (fast), fall back to CLI
      - "websocket": direct Gateway WebSocket (Layer 2)
      - "cli": CLI subprocess (Layer 1, always works)
    """

    def __init__(
        self,
        transport: str = "auto",
        session_key: str = "caveman-bridge",
        agent_id: str = "caveman",
        gateway_url: str = "",
        gateway_port: int | None = None,
        token: str = "",
        password: str = "",
    ):
        self.session_key = session_key
        self.agent_id = agent_id
        self.gateway_url = gateway_url
        self.gateway_port = gateway_port
        self.token = token
        self.password = password
        self._transport: _Transport | None = None
        self._transport_name = transport
        self._connected = False

    async def connect(self) -> bool:
        """Initialize the transport and verify connectivity."""
        transport = self._transport_name

        if transport == "auto":
            ws = await self._try_websocket()
            if ws:
                self._transport = ws
                self._connected = True
                return True
            if shutil.which("openclaw"):
                transport = "cli"
            else:
                logger.error("No transport available")
                return False

        if transport == "websocket":
            ws = await self._try_websocket()
            if not ws:
                return False
            self._transport = ws
            self._connected = True
            return True

        if transport == "cli":
            self._transport = self._make_cli_transport()
            ok = await self._transport.connect()
            self._connected = ok
            return ok

        raise ValueError(f"Unknown transport: {transport}")

    async def _try_websocket(self) -> _Transport | None:
        try:
            from caveman.bridge.ws_transport import WSTransport
        except ImportError:
            return None

        url = self.gateway_url
        if not url and self.gateway_port:
            url = f"ws://127.0.0.1:{self.gateway_port}"
        if not url:
            from caveman.paths import DEFAULT_GATEWAY_URL
            url = DEFAULT_GATEWAY_URL

        adapter = _WSAdapter(
            url=url, token=self.token, password=self.password,
            session_key=self.session_key, agent_id=self.agent_id,
        )
        if await adapter.connect():
            return adapter
        return None

    def _make_cli_transport(self) -> _Transport:
        return _CLIAdapter(self.session_key, self.agent_id)

    async def disconnect(self) -> None:
        if self._transport:
            await self._transport.disconnect()
        self._connected = False

    async def send_message(self, message: str, channel: str | None = None, target: str | None = None) -> dict[str, Any]:
        self._check_connected()
        assert self._transport is not None
        return await self._transport.send_message(message, channel, target)

    async def agent_turn(self, message: str, **kwargs) -> dict[str, Any]:
        self._check_connected()
        assert self._transport is not None
        return await self._transport.agent_turn(message, **kwargs)

    async def list_tools(self) -> list[dict]:
        self._check_connected()
        assert self._transport is not None
        return await self._transport.list_tools()

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._check_connected()
        assert self._transport is not None
        return await self._transport.call_tool(tool_name, args)

    def get_tool_schemas(self) -> list[dict]:
        return self._transport.get_tool_schemas() if self._transport else []

    async def register_tools(self, registry) -> int:
        tools = await self.list_tools()
        count = 0
        for tool in tools:
            name = f"openclaw_{tool['name']}"
            desc = f"[OpenClaw] {tool.get('description', '')}"
            tool_name = tool["name"]

            async def call_wrapper(_tn=tool_name, **kwargs):
                return await self.call_tool(_tn, kwargs)

            registry.register(name, call_wrapper, desc, tool.get("inputSchema", {"type": "object"}))
            count += 1
        return count

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def transport_name(self) -> str:
        return self._transport.name if self._transport else "none"

    def _check_connected(self):
        if not self._connected or not self._transport:
            raise RuntimeError("Not connected to OpenClaw. Call connect() first.")


# ── Transport interface ──

class _Transport(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    async def connect(self) -> bool: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def send_message(self, message: str, channel: str | None, target: str | None) -> dict: ...
    @abstractmethod
    async def agent_turn(self, message: str, **kwargs) -> dict: ...
    @abstractmethod
    async def list_tools(self) -> list[dict]: ...
    @abstractmethod
    async def call_tool(self, tool_name: str, args: dict) -> dict: ...
    def get_tool_schemas(self) -> list[dict]:
        return []


# ── Adapters ──

class _CLIAdapter(_Transport):
    """Thin adapter wrapping CLITransport."""
    def __init__(self, session_key: str, agent_id: str):
        from caveman.bridge.cli_transport import CLITransport
        self._inner = CLITransport(session_key, agent_id)

    @property
    def name(self) -> str:
        return "cli"

    async def connect(self) -> bool:
        return await self._inner.connect()

    async def disconnect(self) -> None:
        await self._inner.disconnect()

    async def send_message(self, message, channel, target):
        return await self._inner.send_message(message, channel, target)

    async def agent_turn(self, message, **kwargs):
        return await self._inner.agent_turn(message, **kwargs)

    async def list_tools(self):
        return await self._inner.list_tools()

    async def call_tool(self, tool_name, args):
        return await self._inner.call_tool(tool_name, args)

    def get_tool_schemas(self):
        return self._inner.get_tool_schemas()


class _WSAdapter(_Transport):
    """Thin adapter wrapping WSTransport."""
    def __init__(self, url: str, token: str, password: str, session_key: str, agent_id: str):
        self._url = url
        self._token = token
        self._password = password
        self._session_key = session_key
        self._agent_id = agent_id
        self._ws = None

    @property
    def name(self) -> str:
        return "websocket"

    async def connect(self) -> bool:
        from caveman.bridge.ws_transport import WSTransport
        self._ws = WSTransport(
            url=self._url, token=self._token, password=self._password,
            session_key=self._session_key, agent_id=self._agent_id,
        )
        return await self._ws.connect()

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.disconnect()

    async def send_message(self, message, channel, target):
        assert self._ws
        return await self._ws.send_message(message, channel=channel or "discord", target=target or "")

    async def agent_turn(self, message, **kwargs):
        assert self._ws
        return await self._ws.agent_turn(message)

    async def list_tools(self):
        assert self._ws
        return await self._ws.list_tools()

    async def call_tool(self, tool_name, args):
        assert self._ws
        return await self._ws.call_tool(tool_name, args)

    def get_tool_schemas(self):
        if not self._ws:
            return []
        return [
            {
                "name": f"openclaw_{t['name']}",
                "description": f"[OpenClaw] {t.get('description', '')}",
                "input_schema": t.get("inputSchema", {"type": "object"}),
            }
            for t in self._ws._tools_cache
        ]
