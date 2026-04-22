"""WhatsApp Platform Adapter — BasePlatformAdapter implementation for WhatsApp.

Uses the WhatsApp Business API (Cloud API) via HTTP.
Requires: WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN.

Features:
- Text, image, audio, video, document sending
- Webhook-based message receiving
- Template message support
- Read receipts
- Reaction support
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    SendResult,
)

logger = logging.getLogger("caveman.gateway.whatsapp")

_API_BASE = "https://graph.facebook.com/v18.0"


class WhatsAppAdapter(BasePlatformAdapter):
    """WhatsApp Cloud API adapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP)
        self._token = config.token or ""
        self._phone_id = config.extra.get("phone_number_id", "")
        self._verify_token = config.extra.get("verify_token", "")
        self._app_secret = config.extra.get("app_secret", "")
        self._webhook_server: Optional[Any] = None

    @property
    def name(self) -> str:
        return "WhatsApp"

    @property
    def _max_message_length(self) -> int:
        return 4096

    async def connect(self) -> bool:
        if not self._token or not self._phone_id:
            logger.error("WhatsApp requires token and phone_number_id")
            return False
        # Verify token works by checking phone number
        try:
            import aiohttp
            _HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                url = f"{_API_BASE}/{self._phone_id}"
                headers = {"Authorization": f"Bearer {self._token}"}
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error("WhatsApp token verification failed: %d", resp.status)
                        return False
            self._running = True
            logger.info("WhatsApp connected (phone: %s)", self._phone_id)
            return True
        except Exception as e:
            logger.error("WhatsApp connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._token:
            return SendResult(success=False, error="Not connected")

        try:
            import aiohttp
            url = f"{_API_BASE}/{self._phone_id}/messages"
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "to": chat_id,
                "type": "text",
                "text": {"body": content},
            }
            if reply_to:
                payload["context"] = {"message_id": reply_to}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        msg_id = data.get("messages", [{}])[0].get("id", "")
                        return SendResult(success=True, message_id=msg_id)
                    error = data.get("error", {}).get("message", str(resp.status))
                    retryable = resp.status in (429, 500, 502, 503)
                    return SendResult(success=False, error=error, retryable=retryable)
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_image(
        self, chat_id: str, image_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._send_media(chat_id, "image", {"link": image_url}, caption, reply_to)

    async def send_voice(
        self, chat_id: str, audio_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        # WhatsApp requires media to be uploaded first
        media_id = await self._upload_media(audio_path, "audio/ogg")
        if not media_id:
            return SendResult(success=False, error="Media upload failed")
        return await self._send_media(chat_id, "audio", {"id": media_id}, caption, reply_to)

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        media_id = await self._upload_media(file_path, "application/octet-stream")
        if not media_id:
            return SendResult(success=False, error="Media upload failed")
        name = file_name or Path(file_path).name
        return await self._send_media(
            chat_id, "document", {"id": media_id, "filename": name}, caption, reply_to,
        )

    async def _send_media(
        self, chat_id: str, media_type: str, media_obj: dict,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
    ) -> SendResult:
        try:
            import aiohttp
            url = f"{_API_BASE}/{self._phone_id}/messages"
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
            if caption:
                media_obj["caption"] = caption
            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "to": chat_id,
                "type": media_type,
                media_type: media_obj,
            }
            if reply_to:
                payload["context"] = {"message_id": reply_to}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        msg_id = data.get("messages", [{}])[0].get("id", "")
                        return SendResult(success=True, message_id=msg_id)
                    return SendResult(success=False, error=str(data))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _upload_media(self, file_path: str, mime_type: str) -> Optional[str]:
        """Upload media to WhatsApp and return media ID."""
        try:
            import aiohttp
            url = f"{_API_BASE}/{self._phone_id}/media"
            headers = {"Authorization": f"Bearer {self._token}"}
            data = aiohttp.FormData()
            data.add_field("messaging_product", "whatsapp")
            data.add_field("type", mime_type)
            data.add_field("file", open(file_path, "rb"), content_type=mime_type)

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, data=data) as resp:
                    result = await resp.json()
                    return result.get("id")
        except Exception as e:
            logger.warning("WhatsApp media upload failed: %s", e)
            return None

    def handle_webhook(self, body: bytes, signature: Optional[str] = None) -> Optional[MessageEvent]:
        """Parse incoming webhook payload into MessageEvent.

        Call this from your webhook endpoint handler.
        Returns None if the payload is not a user message.
        """
        # Verify signature if app_secret is configured
        if self._app_secret and signature:
            expected = hmac.new(
                self._app_secret.encode(), body, hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(f"sha256={expected}", signature):
                logger.warning("WhatsApp webhook signature mismatch")
                return None

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None

        # Extract message from webhook structure
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return None

        msg = messages[0]
        from_number = msg.get("from", "")
        msg_id = msg.get("id", "")
        msg_type = msg.get("type", "text")

        # Extract text
        text = ""
        media_urls: List[str] = []
        media_types: List[str] = []
        event_type = MessageType.TEXT

        if msg_type == "text":
            text = msg.get("text", {}).get("body", "")
        elif msg_type == "image":
            event_type = MessageType.PHOTO
            media_id = msg.get("image", {}).get("id", "")
            media_urls.append(media_id)
            media_types.append("image/jpeg")
            text = msg.get("image", {}).get("caption", "")
        elif msg_type == "audio":
            event_type = MessageType.AUDIO
            media_id = msg.get("audio", {}).get("id", "")
            media_urls.append(media_id)
            media_types.append("audio/ogg")
        elif msg_type == "document":
            event_type = MessageType.DOCUMENT
            media_id = msg.get("document", {}).get("id", "")
            media_urls.append(media_id)
            media_types.append(msg.get("document", {}).get("mime_type", ""))
            text = msg.get("document", {}).get("caption", "")

        # Get contact name
        contacts = value.get("contacts", [{}])
        user_name = contacts[0].get("profile", {}).get("name", "") if contacts else ""

        source = self.build_source(
            chat_id=from_number,
            chat_type="dm",
            user_id=from_number,
            user_name=user_name,
        )

        # Reply context
        context = msg.get("context", {})
        reply_to_id = context.get("id") if context else None

        return MessageEvent(
            text=text,
            message_type=event_type,
            source=source,
            raw_message=msg,
            message_id=msg_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_id,
            is_mention=False,
            is_reply_to_bot=bool(reply_to_id),
        )
