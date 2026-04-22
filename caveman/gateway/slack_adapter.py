"""Slack Platform Adapter — BasePlatformAdapter implementation for Slack.

Features:
- Slack Bolt SDK integration
- Socket Mode (no public URL needed)
- Thread support
- Block Kit formatting
- File uploads
- Emoji reactions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    ProcessingOutcome, SendResult,
)

logger = logging.getLogger("caveman.gateway.slack")

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False


class SlackAdapter(BasePlatformAdapter):
    """Slack adapter built on BasePlatformAdapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._bot_token = config.token or ""
        self._app_token = config.extra.get("app_token", "")
        self._app: Optional[Any] = None
        self._handler: Optional[Any] = None
        self._bot_user_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "Slack"

    @property
    def _max_message_length(self) -> int:
        return 3000  # Slack limit is 4000 but leave room for formatting

    async def connect(self) -> bool:
        if not SLACK_AVAILABLE:
            logger.error("slack-bolt not installed: pip install slack-bolt")
            return False
        if not self._bot_token or not self._app_token:
            logger.error("Slack requires both bot_token and app_token")
            return False

        app = AsyncApp(token=self._bot_token)
        self._app = app

        @app.event("message")
        async def handle_message(event, say, client) -> None:
            await self._on_slack_message(event, say, client)

        try:
            # Get bot user ID for mention detection
            auth = await app.client.auth_test()
            self._bot_user_id = auth.get("user_id")

            handler = AsyncSocketModeHandler(app, self._app_token)
            self._handler = handler
            await handler.start_async()
            self._running = True
            logger.info("Slack connected (bot: %s)", self._bot_user_id)
            return True
        except Exception as e:
            logger.error("Slack connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()
        if self._handler:
            await self._handler.close_async()
            self._handler = None
        self._app = None

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            kwargs: Dict[str, Any] = {"channel": chat_id, "text": content}
            thread_ts = (metadata or {}).get("thread_id")
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            elif reply_to:
                kwargs["thread_ts"] = reply_to

            result = await self._app.client.chat_postMessage(**kwargs)
            msg_ts = result.get("ts", "")
            return SendResult(success=True, message_id=msg_ts)
        except Exception as e:
            error_str = str(e).lower()
            retryable = "ratelimited" in error_str or "timeout" in error_str
            return SendResult(success=False, error=str(e), retryable=retryable)

    async def edit_message(
        self, chat_id: str, message_id: str, content: str,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Not connected")
        try:
            await self._app.client.chat_update(
                channel=chat_id, ts=message_id, text=content,
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Slack doesn't have a typing API for bots — no-op."""

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._app:
            return SendResult(success=False, error="Not connected")
        try:
            name = file_name or Path(file_path).name
            result = await self._app.client.files_upload_v2(
                channel=chat_id, file=file_path,
                filename=name, initial_comment=caption or "",
            )
            return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self.send_document(chat_id, image_path, caption)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add 👀 reaction."""
        if not self._app or not event.message_id or not event.source:
            return
        try:
            await self._app.client.reactions_add(
                channel=event.source.chat_id,
                timestamp=event.message_id,
                name="eyes",
            )
        except Exception as exc:
            logger.debug("on_processing_start: suppressed %s", exc)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome,
    ) -> None:
        if not self._app or not event.message_id or not event.source:
            return
        try:
            await self._app.client.reactions_remove(
                channel=event.source.chat_id,
                timestamp=event.message_id,
                name="eyes",
            )
        except Exception as exc:
            logger.debug("on_processing_complete: suppressed %s", exc)
        emoji = "white_check_mark" if outcome == ProcessingOutcome.SUCCESS else "x"
        try:
            await self._app.client.reactions_add(
                channel=event.source.chat_id,
                timestamp=event.message_id,
                name=emoji,
            )
        except Exception as exc:
            logger.debug("on_processing_complete: suppressed %s", exc)

    async def _on_slack_message(self, event: dict, say, client) -> None:
        """Handle incoming Slack message."""
        # Ignore bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        text = event.get("text", "")
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts")
        msg_ts = event.get("ts", "")

        if not text and not event.get("files"):
            return

        # Strip bot mention
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        # Determine chat type
        channel_type = event.get("channel_type", "")
        chat_type = "dm" if channel_type == "im" else "channel"

        source = self.build_source(
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=thread_ts,
        )

        # Handle file attachments
        media_urls = []
        media_types = []
        msg_type = MessageType.TEXT
        for f in event.get("files", []):
            media_urls.append(f.get("url_private", ""))
            media_types.append(f.get("mimetype", ""))
            if f.get("mimetype", "").startswith("image/"):
                msg_type = MessageType.PHOTO

        msg_event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=event,
            message_id=msg_ts,
            media_urls=media_urls,
            media_types=media_types,
        )

        await self.handle_message(msg_event)
