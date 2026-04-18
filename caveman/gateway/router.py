"""Gateway router — route messages across multiple gateways."""
from __future__ import annotations
import logging
from typing import Callable

from .base import Gateway

logger = logging.getLogger(__name__)


class GatewayRouter:
    """Route messages across multiple gateways."""

    def __init__(self):
        self._gateways: dict[str, Gateway] = {}
        self._message_callback: Callable | None = None

    def register(self, gateway: Gateway) -> None:
        self._gateways[gateway.name] = gateway

    async def start_all(self) -> None:
        for name, gw in self._gateways.items():
            try:
                await gw.start()
            except Exception as e:
                logger.warning("Failed to start gateway %s: %s", name, e)

    async def stop_all(self) -> None:
        for name, gw in self._gateways.items():
            try:
                await gw.stop()
            except Exception as e:
                logger.warning("Failed to stop gateway %s: %s", name, e)

    async def send(self, gateway_name: str, channel_id: str, content: str) -> dict:
        gw = self._gateways.get(gateway_name)
        if not gw:
            raise ValueError(f"Unknown gateway: {gateway_name}")
        msg = await gw.send_message(channel_id, content)
        msg_id = getattr(msg, 'id', None) if msg else None
        return {"ok": True, "gateway": gateway_name, "channel_id": channel_id, "message_id": msg_id}

    async def edit(self, gateway_name: str, channel_id: str, message_id: int, content: str) -> dict:
        gw = self._gateways.get(gateway_name)
        if not gw or not hasattr(gw, 'edit_message'):
            return {"ok": False, "error": "edit not supported"}
        await gw.edit_message(int(channel_id), message_id, content)
        return {"ok": True}

    async def on_message(self, callback: Callable) -> None:
        self._message_callback = callback
        for gw in self._gateways.values():
            await gw.on_message(callback)

    def list_gateways(self) -> list[dict]:
        return [{"name": gw.name, "running": gw.is_running} for gw in self._gateways.values()]
