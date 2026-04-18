"""WebSocket transport — direct connection to OpenClaw Gateway.

Layer 2 transport: bypasses CLI subprocess overhead (~3s → <100ms).
Speaks the OpenClaw Gateway protocol: JSON frames over WebSocket.

Protocol:
  Client → Gateway: {"type": "req", "id": "<uuid>", "method": "<method>", "params": {...}}
  Gateway → Client: {"type": "res", "id": "<uuid>", "result": {...}}
  Gateway → Client: {"type": "err", "id": "<uuid>", "error": {...}}
  Gateway → Client: {"type": "evt", "event": "<name>", "data": {...}}

Handshake:
  1. Connect WebSocket
  2. Gateway sends challenge or hello
  3. Client sends "connect" with auth + client metadata
  4. Gateway responds with HelloOk
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Default Gateway WebSocket URL (local)
from caveman.paths import DEFAULT_GATEWAY_URL
CONNECT_TIMEOUT = 10.0
REQUEST_TIMEOUT = 30.0


class WSTransportError(Exception):
    """WebSocket transport error."""
    pass


class WSTransport:
    """WebSocket transport for OpenClaw Gateway.

    Provides low-latency bidirectional communication with the Gateway.
    Handles connection, authentication, reconnection, and request/response.
    """

    def __init__(
        self,
        url: str = DEFAULT_GATEWAY_URL,
        token: str = "",
        password: str = "",
        session_key: str = "caveman-bridge",
        agent_id: str = "caveman",
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ):
        self.url = url
        self.token = token
        self.password = password
        self.session_key = session_key
        self.agent_id = agent_id
        self._on_event = on_event

        self._ws: Any = None  # websockets connection
        self._connected = False
        self._hello_ok = False
        self._pending: dict[str, asyncio.Future] = {}
        self._recv_task: asyncio.Task | None = None
        self._tools_cache: list[dict] = []

    async def connect(self) -> bool:
        """Connect to Gateway and complete handshake."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Run: pip install websockets")
            return False

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    max_size=25 * 1024 * 1024,
                    ping_interval=30,
                    ping_timeout=10,
                ),
                timeout=CONNECT_TIMEOUT,
            )
            logger.info("WebSocket connected to %s", self.url)
        except Exception as e:
            logger.error("WebSocket connect failed: %s", e)
            return False

        # Start message receiver
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Wait for hello/challenge, then send connect
        try:
            ok = await asyncio.wait_for(self._handshake(), timeout=CONNECT_TIMEOUT)
            if ok:
                self._connected = True
                logger.info("Gateway handshake complete")
            return ok
        except asyncio.TimeoutError:
            logger.error("Gateway handshake timed out")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._connected = False
        self._hello_ok = False
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("Suppressed in ws_transport: %s", e)
            self._ws = None
        # Reject all pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(WSTransportError("disconnected"))
        self._pending.clear()

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def request(
        self, method: str, params: dict | None = None, timeout: float = REQUEST_TIMEOUT,
    ) -> dict:
        """Send a request to Gateway and wait for response."""
        if not self.connected:
            raise WSTransportError("not connected")

        req_id = str(uuid.uuid4())
        frame = {
            "type": "req",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        fut: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(frame))
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise WSTransportError(f"request timeout: {method}")
        except Exception as e:
            self._pending.pop(req_id, None)
            raise WSTransportError(f"request failed: {e}") from e

    async def call_tool(self, tool_name: str, args: dict | None = None) -> dict:
        """Invoke a Gateway tool by name."""
        return await self.request("tools.invoke", {
            "tool": tool_name,
            "arguments": args or {},
            "sessionKey": self.session_key,
        })

    async def send_message(
        self, message: str, channel: str = "discord", target: str = "",
    ) -> dict:
        """Send a message through Gateway."""
        return await self.call_tool("message", {
            "action": "send",
            "message": message,
            "channel": channel,
            "to": target,
        })

    async def agent_turn(self, prompt: str) -> dict:
        """Run an agent turn through Gateway."""
        return await self.request("agent.turn", {
            "prompt": prompt,
            "sessionKey": self.session_key,
        }, timeout=120.0)

    async def list_tools(self) -> list[dict]:
        """List available Gateway tools."""
        if self._tools_cache:
            return self._tools_cache
        try:
            result = await self.request("tools.list")
            self._tools_cache = result.get("tools", [])
            return self._tools_cache
        except Exception as e:
            logger.warning("Failed to list tools: %s", e)
            return []

    # --- Internal ---

    async def _handshake(self) -> bool:
        """Complete the Gateway handshake."""
        # The Gateway protocol:
        # 1. On connect, Gateway may send a challenge or wait for "connect"
        # 2. Client sends "connect" with auth params
        # 3. Gateway responds with HelloOk

        connect_params: dict[str, Any] = {
            "clientName": "caveman",
            "clientVersion": "0.3.0",
            "platform": "python",
            "mode": "operator",
        }
        if self.token:
            connect_params["token"] = self.token
        if self.password:
            connect_params["password"] = self.password

        try:
            result = await self.request("connect", connect_params, timeout=CONNECT_TIMEOUT)
            self._hello_ok = True
            logger.debug("HelloOk: %s", result)
            return True
        except WSTransportError as e:
            logger.error("Handshake failed: %s", e)
            return False

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from Gateway."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from Gateway: %s", raw[:200])
                    continue

                msg_type = msg.get("type")

                if msg_type == "res":
                    # Response to a request
                    req_id = msg.get("id")
                    fut = self._pending.pop(req_id, None)
                    if fut and not fut.done():
                        fut.set_result(msg.get("result", {}))

                elif msg_type == "err":
                    # Error response
                    req_id = msg.get("id")
                    fut = self._pending.pop(req_id, None)
                    if fut and not fut.done():
                        error = msg.get("error", {})
                        fut.set_exception(WSTransportError(
                            error.get("message", "unknown error")
                        ))

                elif msg_type == "evt":
                    # Event from Gateway
                    if self._on_event:
                        try:
                            await self._on_event(msg)
                        except Exception:
                            logger.warning("Event handler error", exc_info=True)

                elif msg_type == "challenge":
                    # Gateway challenge — respond with connect
                    # Challenge is handled by _handshake, just log
                    logger.debug("Received challenge: %s", msg)

                else:
                    logger.debug("Unknown message type: %s", msg_type)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("WebSocket recv loop ended: %s", e)
            self._connected = False
            # Reject all pending requests so callers fail fast instead of hanging
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(WSTransportError(f"recv loop died: {e}"))
            self._pending.clear()
