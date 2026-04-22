"""Matrix Platform Adapter — BasePlatformAdapter implementation for Matrix.

Uses matrix-nio for the Matrix protocol.
Features: E2EE support, room management, media uploads, reactions.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.aio import aio_stat
from caveman.gateway.platform_types import (
    MessageEvent, Platform, PlatformConfig,
    SendResult,
)

logger = logging.getLogger("caveman.gateway.matrix")

try:
    from nio import AsyncClient, MatrixRoom, RoomMessageText  # noqa: F401
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False


class MatrixAdapter(BasePlatformAdapter):
    """Matrix adapter via matrix-nio."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATRIX)
        self._homeserver = config.extra.get("homeserver", "")
        self._user_id = config.extra.get("user_id", "")
        self._password = config.extra.get("password", "")
        self._access_token = config.token or ""
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "Matrix"

    @property
    def _max_message_length(self) -> int:
        return 16384  # Matrix has generous limits

    async def connect(self) -> bool:
        if not MATRIX_AVAILABLE:
            logger.error("matrix-nio not installed: pip install matrix-nio[e2e]")
            return False
        if not self._homeserver:
            logger.error("Matrix requires homeserver URL")
            return False

        client = AsyncClient(self._homeserver, self._user_id)
        self._client = client

        try:
            if self._access_token:
                client.access_token = self._access_token
                client.user_id = self._user_id
            else:
                resp = await client.login(self._password)
                if hasattr(resp, "access_token"):
                    logger.info("Matrix logged in as %s", self._user_id)
                else:
                    logger.error("Matrix login failed")
                    return False

            # Register message callback
            client.add_event_callback(self._on_message, RoomMessageText)

            self._running = True
            asyncio.create_task(client.sync_forever(timeout=30000))
            logger.info("Matrix connected: %s", self._user_id)
            return True
        except Exception as e:
            logger.error("Matrix connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()
        if self._client:
            await self._client.close()
            self._client = None

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            resp = await self._client.room_send(
                room_id=chat_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": content},
            )
            event_id = getattr(resp, "event_id", "")
            return SendResult(success=True, message_id=event_id)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            import mimetypes
            mime = mimetypes.guess_type(image_path)[0] or "image/png"
            file_stat = await aio_stat(Path(image_path))
            with open(image_path, "rb") as f:
                resp = await self._client.upload(f, content_type=mime, filesize=file_stat.st_size)
            if not hasattr(resp, "content_uri"):
                return SendResult(success=False, error="Upload failed")
            content = {
                "msgtype": "m.image",
                "body": caption or Path(image_path).name,
                "url": resp.content_uri,
                "info": {"mimetype": mime, "size": file_stat.st_size},
            }
            resp2 = await self._client.room_send(chat_id, "m.room.message", content)
            return SendResult(success=True, message_id=getattr(resp2, "event_id", ""))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _on_message(self, room, event) -> None:
        """Handle incoming Matrix message."""
        if event.sender == self._user_id:
            return

        source = self.build_source(
            chat_id=room.room_id,
            chat_name=room.display_name,
            chat_type="group" if room.member_count > 2 else "dm",
            user_id=event.sender,
            user_name=room.user_name(event.sender) if hasattr(room, "user_name") else event.sender,
        )

        # Matrix: mention via display name or user_id in body
        is_mention = bool(self._user_id and self._user_id in (event.body or ""))
        # Check formatted_body for HTML mention too
        fmt_body = getattr(event, "formatted_body", "") or ""
        if self._user_id and self._user_id in fmt_body:
            is_mention = True
        is_reply_to_bot = False
        relates = getattr(event, "source", {}).get("content", {}).get("m.relates_to", {})
        in_reply_to = relates.get("m.in_reply_to", {}).get("event_id")
        if in_reply_to:
            # We can't easily check who sent the parent without fetching it
            # Set based on whether the reply fallback contains our user_id
            if self._user_id and self._user_id in (event.body or ""):
                is_reply_to_bot = True

        msg_event = MessageEvent(
            text=event.body,
            source=source,
            raw_message=event,
            message_id=event.event_id,
            is_mention=is_mention,
            is_reply_to_bot=is_reply_to_bot,
        )
        await self.handle_message(msg_event)
