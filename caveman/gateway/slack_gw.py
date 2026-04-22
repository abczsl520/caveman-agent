"""Slack gateway — receive tasks from Slack, run Caveman, reply.

Uses Slack's Socket Mode (no public URL needed) or Events API.
Requires: slack_sdk (pip install slack-sdk)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from .base import Gateway
from caveman.utils import split_message

logger = logging.getLogger(__name__)


class SlackGateway(Gateway):
    """Slack bot gateway for Caveman."""

    def __init__(
        self,
        bot_token: str,
        app_token: str = "",
        allowed_channels: list[str] | None = None,
        max_message_len: int = 3000,
    ):
        self.bot_token = bot_token
        self.app_token = app_token
        self.allowed_channels = set(allowed_channels) if allowed_channels else None
        self.max_message_len = max_message_len
        self._handler: Callable | None = None
        self._client = None
        self._socket = None

    @property
    def name(self) -> str:
        return "slack"

    async def start(self) -> None:
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse
        except ImportError:
            logger.error("slack-sdk not installed. Run: pip install slack-sdk[socket_mode]")
            return

        self._client = AsyncWebClient(token=self.bot_token)
        self._running = True

        if self.app_token:
            # Socket Mode — no public URL needed
            self._socket = SocketModeClient(
                app_token=self.app_token,
                web_client=self._client,
            )

            async def handle_event(client, req: SocketModeRequest) -> None:
                if req.type == "events_api":
                    event = req.payload.get("event", {})
                    if event.get("type") == "message" and not event.get("bot_id"):
                        channel = event.get("channel", "")
                        if self.allowed_channels and channel not in self.allowed_channels:
                            return
                        text = event.get("text", "").strip()
                        if text and self._handler:
                            context = {
                                "channel_id": channel,
                                "user_id": event.get("user", ""),
                                "thread_ts": event.get("thread_ts") or event.get("ts"),
                                "gateway_name": "slack",
                            }
                            await self._handler(text, context)
                    response = SocketModeResponse(envelope_id=req.envelope_id)
                    await client.send_socket_mode_response(response)

            self._socket.socket_mode_request_listeners.append(handle_event)
            await self._socket.connect()
            logger.info("Slack gateway connected (Socket Mode)")

            while self._running:
                await asyncio.sleep(1)
        else:
            logger.error("Slack app_token required for Socket Mode")

    async def stop(self) -> None:
        self._running = False
        if self._socket:
            await self._socket.close()

    async def send_message(self, channel_id: str, text: str) -> None:
        if not self._client:
            return
        for chunk in split_message(text, self.max_message_len):
            try:
                await self._client.chat_postMessage(channel=channel_id, text=chunk)
            except Exception as e:
                logger.error("Slack send failed: %s", e)

    async def send_reply(self, channel_id: str, text: str, thread_ts: str) -> None:
        """Send a threaded reply."""
        if not self._client:
            return
        for chunk in split_message(text, self.max_message_len):
            try:
                await self._client.chat_postMessage(
                    channel=channel_id, text=chunk, thread_ts=thread_ts,
                )
            except Exception as e:
                logger.error("Slack reply failed: %s", e)

    async def on_message(self, handler: Callable) -> None:
        self._handler = handler

    def on_task(self, handler: Callable[[str, dict], Any]) -> None:
        """Register task handler (called by GatewayServer)."""
        self._handler = handler
