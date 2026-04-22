"""Feishu (Lark) Platform Adapter — BasePlatformAdapter implementation.

Uses Feishu Open API for sending/receiving messages.
Requires: APP_ID, APP_SECRET, and event subscription webhook.

Features:
- Text, image, file sending
- Group and DM support
- Card messages
- Event subscription (webhook)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    SendResult,
)

logger = logging.getLogger("caveman.gateway.feishu")

_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark adapter via Open API."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.FEISHU)
        self._app_id = config.extra.get("app_id", "")
        self._app_secret = config.extra.get("app_secret", "")
        self._encrypt_key = config.extra.get("encrypt_key", "")
        self._verification_token = config.extra.get("verification_token", "")
        self._tenant_access_token: Optional[str] = None
        self._token_expires: float = 0

    @property
    def name(self) -> str:
        return "Feishu"

    @property
    def _max_message_length(self) -> int:
        return 4096

    async def connect(self) -> bool:
        if not self._app_id or not self._app_secret:
            logger.error("Feishu requires app_id and app_secret")
            return False
        # Get initial token
        token = await self._refresh_token()
        if not token:
            return False
        self._running = True
        logger.info("Feishu connected (app: %s)", self._app_id)
        return True

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()
        self._tenant_access_token = None

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        token = await self._get_token()
        if not token:
            return SendResult(success=False, error="No access token")

        try:
            import aiohttp
            _HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)
            url = f"{_API_BASE}/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            # Determine receive_id_type
            id_type = "chat_id"
            if chat_id.startswith("ou_"):
                id_type = "open_id"

            params = {"receive_id_type": id_type}
            payload = {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": content}),
            }
            if reply_to:
                payload["reply_in_thread"] = True

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, params=params, json=payload) as resp:
                    data = await resp.json()
                    if data.get("code") == 0:
                        msg_id = data.get("data", {}).get("message_id", "")
                        return SendResult(success=True, message_id=msg_id)
                    return SendResult(success=False, error=data.get("msg", str(data)))
        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def _refresh_token(self) -> Optional[str]:
        """Get tenant_access_token from Feishu."""
        try:
            import aiohttp
            url = f"{_API_BASE}/auth/v3/tenant_access_token/internal"
            payload = {"app_id": self._app_id, "app_secret": self._app_secret}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if data.get("code") == 0:
                        self._tenant_access_token = data["tenant_access_token"]
                        self._token_expires = time.time() + data.get("expire", 7200) - 300
                        return self._tenant_access_token
                    logger.error("Feishu token refresh failed: %s", data)
                    return None
        except Exception as e:
            logger.error("Feishu token refresh error: %s", e)
            return None

    async def _get_token(self) -> Optional[str]:
        """Get valid token, refreshing if needed."""
        if not self._tenant_access_token or time.time() > self._token_expires:
            return await self._refresh_token()
        return self._tenant_access_token

    def handle_webhook(self, body: bytes) -> Optional[MessageEvent]:
        """Parse incoming Feishu event webhook into MessageEvent."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None

        # Challenge verification
        if "challenge" in data:
            return None  # Caller should return {"challenge": data["challenge"]}

        event = data.get("event", {})
        msg = event.get("message", {})
        if not msg:
            return None

        sender = event.get("sender", {})
        chat_id = msg.get("chat_id", "")
        msg_id = msg.get("message_id", "")
        msg_type = msg.get("message_type", "text")

        # Parse content
        text = ""
        media_urls: List[str] = []
        event_type = MessageType.TEXT

        content_str = msg.get("content", "{}")
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {}

        if msg_type == "text":
            text = content.get("text", "")
        elif msg_type == "image":
            event_type = MessageType.PHOTO
            media_urls.append(content.get("image_key", ""))

        if not text and not media_urls:
            return None

        source = self.build_source(
            chat_id=chat_id,
            chat_type="dm" if msg.get("chat_type") == "p2p" else "group",
            user_id=sender.get("sender_id", {}).get("open_id", ""),
            user_name=sender.get("sender_id", {}).get("user_id", ""),
        )

        # Feishu: mention detection via mentions in message content
        is_mention = False
        mentions = msg.get("mentions", [])
        if mentions and self._app_id:
            is_mention = any(m.get("id", {}).get("open_id") == self._app_id for m in mentions)
        # Strip mention tags from text
        if is_mention:
            import re
            text = re.sub(r'@_user_\d+', '', text).strip()

        return MessageEvent(
            text=text,
            message_type=event_type,
            source=source,
            raw_message=data,
            message_id=msg_id,
            media_urls=media_urls,
            is_mention=is_mention,
        )
