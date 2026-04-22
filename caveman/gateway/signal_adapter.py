"""Signal Platform Adapter — BasePlatformAdapter implementation for Signal.

Uses signal-cli (REST API mode) for sending/receiving messages.
Requires: signal-cli running in JSON-RPC or REST mode.

Features:
- Text, image, audio, document sending
- Group and DM support
- Reaction support
- Quote/reply support
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.aio import aio_read_bytes
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    SendResult,
)

logger = logging.getLogger("caveman.gateway.signal")


class SignalAdapter(BasePlatformAdapter):
    """Signal adapter via signal-cli REST API."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)
        self._api_url = config.extra.get("api_url", "http://localhost:8080")
        self._phone_number = config.extra.get("phone_number", "")
        self._poll_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "Signal"

    @property
    def _max_message_length(self) -> int:
        return 4096

    async def connect(self) -> bool:
        if not self._phone_number:
            logger.error("Signal requires phone_number in config")
            return False

        # Verify signal-cli is reachable
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"{self._api_url}/v1/about"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.error("signal-cli not reachable at %s", self._api_url)
                        return False
        except Exception as e:
            logger.error("Signal connection failed: %s", e)
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_messages())
        logger.info("Signal connected (number: %s)", self._phone_number)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
            self._poll_task = None
        await self.cancel_background_tasks()

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            import aiohttp
            url = f"{self._api_url}/v2/send"
            payload: Dict[str, Any] = {
                "message": content,
                "number": self._phone_number,
                "recipients": [chat_id],
            }
            # Check if it's a group
            if chat_id.startswith("group."):
                payload = {
                    "message": content,
                    "number": self._phone_number,
                    "recipients": [],
                    "group_id": chat_id.replace("group.", ""),
                }
            if reply_to:
                payload["quote_message_id"] = reply_to

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        ts = str(data.get("timestamp", ""))
                        return SendResult(success=True, message_id=ts)
                    error = await resp.text()
                    return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_with_attachment(chat_id, image_path, caption)

    async def send_voice(
        self, chat_id: str, audio_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_with_attachment(chat_id, audio_path, caption)

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_with_attachment(chat_id, file_path, caption)

    async def _send_with_attachment(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
    ) -> SendResult:
        try:
            import aiohttp
            import base64
            url = f"{self._api_url}/v2/send"
            # Read file and base64 encode
            file_data = await aio_read_bytes(Path(file_path))
            b64 = base64.b64encode(file_data).decode()

            payload: Dict[str, Any] = {
                "message": caption or "",
                "number": self._phone_number,
                "recipients": [chat_id],
                "base64_attachments": [b64],
            }
            if chat_id.startswith("group."):
                payload["recipients"] = []
                payload["group_id"] = chat_id.replace("group.", "")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        return SendResult(success=True)
                    return SendResult(success=False, error=await resp.text())
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _poll_messages(self) -> None:
        """Poll signal-cli for new messages."""
        try:
            import aiohttp
            while self._running:
                try:
                    url = f"{self._api_url}/v1/receive/{self._phone_number}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            if resp.status == 200:
                                messages = await resp.json()
                                for msg in messages:
                                    event = self._parse_message(msg)
                                    if event:
                                        await self.handle_message(event)
                except asyncio.TimeoutError as exc:
                    logger.debug("_poll_messages: suppressed %s", exc)
                except Exception as e:
                    logger.debug("Signal poll error: %s", e)
                    await asyncio.sleep(5)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass  # intentional: Exception suppressed

    def _parse_message(self, raw: dict) -> Optional[MessageEvent]:
        """Parse signal-cli message into MessageEvent."""
        envelope = raw.get("envelope", {})
        data_msg = envelope.get("dataMessage")
        if not data_msg:
            return None

        sender = envelope.get("source", "")
        timestamp = str(data_msg.get("timestamp", ""))
        text = data_msg.get("message", "")
        group_info = data_msg.get("groupInfo")

        # Determine chat context
        if group_info:
            chat_id = f"group.{group_info.get('groupId', '')}"
            chat_type = "group"
        else:
            chat_id = sender
            chat_type = "dm"

        # Media
        media_urls: List[str] = []
        media_types: List[str] = []
        msg_type = MessageType.TEXT
        for att in data_msg.get("attachments", []):
            media_urls.append(att.get("id", ""))
            ct = att.get("contentType", "")
            media_types.append(ct)
            if ct.startswith("image/"):
                msg_type = MessageType.PHOTO
            elif ct.startswith("audio/"):
                msg_type = MessageType.AUDIO

        if not text and not media_urls:
            return None

        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=sender,
            user_name=envelope.get("sourceName", sender),
        )

        # Quote/reply context
        quote = data_msg.get("quote")
        reply_to_id = str(quote.get("id", "")) if quote else None
        reply_to_text = quote.get("text", "") if quote else None

        # Signal: mention detection via mentions list or text pattern
        is_mention = False
        mentions = data_msg.get("mentions", [])
        if mentions and self._phone:
            is_mention = any(m.get("uuid") == self._phone or m.get("number") == self._phone for m in mentions)
        is_reply_to_bot = False
        if quote and self._phone:
            quote_author = quote.get("author", "")
            is_reply_to_bot = quote_author == self._phone

        return MessageEvent(
            text=text or "",
            message_type=msg_type,
            source=source,
            raw_message=raw,
            message_id=timestamp,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            is_mention=is_mention,
            is_reply_to_bot=is_reply_to_bot,
        )
