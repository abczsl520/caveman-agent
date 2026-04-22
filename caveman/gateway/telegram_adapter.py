"""Telegram Platform Adapter — BasePlatformAdapter implementation for Telegram.

Features:
- python-telegram-bot 20.x integration
- MarkdownV2 formatting with safe fallback
- Voice message support (OGG/Opus)
- Photo/video/document sending
- Inline keyboard support (approvals, choices)
- Forum topic (DM topics) support
- Webhook and polling modes
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    SendResult,
)

logger = logging.getLogger("caveman.gateway.telegram")

try:
    from telegram import Update, Bot, InputFile  # noqa: F401
    from telegram.ext import Application, MessageHandler as TGMsgHandler, ContextTypes, filters  # noqa: F401
    from telegram.constants import ParseMode, ChatType  # noqa: F401
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    Bot = Any

# MarkdownV2 escape pattern
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def format_markdown_v2(content: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2.

    Handles: bold, italic, code, code blocks, links.
    Escapes everything else.
    """
    # Protect code blocks first
    blocks = []
    def _protect_block(m):
        blocks.append(m.group(0))
        return f"\x00BLOCK{len(blocks)-1}\x00"

    result = re.sub(r'```[\s\S]*?```', _protect_block, content)

    # Protect inline code
    codes = []
    def _protect_code(m):
        codes.append(m.group(0))
        return f"\x00CODE{len(codes)-1}\x00"

    result = re.sub(r'`[^`]+`', _protect_code, result)

    # Convert markdown formatting
    # Bold: **text** → *text*
    result = re.sub(r'\*\*(.+?)\*\*', lambda m: f"*{_escape_mdv2(m.group(1))}*", result)
    # Italic: _text_ → _text_
    result = re.sub(r'(?<!\w)_(.+?)_(?!\w)', lambda m: f"_{_escape_mdv2(m.group(1))}_", result)
    # Links: [text](url) → [text](url) — protect URL from escaping
    links = []
    def _protect_link(m):
        links.append((m.group(1), m.group(2)))
        return f"\x00LINK{len(links)-1}\x00"
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _protect_link, result)

    # Escape remaining special chars (outside protected regions)
    parts = re.split(r'(\x00(?:BLOCK|CODE|LINK)\d+\x00)', result)
    escaped_parts = []
    for part in parts:
        if part.startswith('\x00'):
            escaped_parts.append(part)
        else:
            escaped_parts.append(_MDV2_ESCAPE_RE.sub(r'\\\1', part))
    result = ''.join(escaped_parts)

    # Restore protected regions
    for i, block in enumerate(blocks):
        result = result.replace(f"\x00BLOCK{i}\x00", block)
    for i, code in enumerate(codes):
        result = result.replace(f"\x00CODE{i}\x00", code)
    for i, (text, url) in enumerate(links):
        result = result.replace(f"\x00LINK{i}\x00", f"[{_escape_mdv2(text)}]({url})")

    return result


class TelegramAdapter(BasePlatformAdapter):
    """Telegram adapter built on BasePlatformAdapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        self._token = config.token or ""
        self._app: Optional[Any] = None
        self._bot: Optional[Any] = None
        self._allowed_users = set(config.extra.get("allowed_users", []))
        self._use_webhook = config.extra.get("webhook", False)
        self._webhook_url = config.extra.get("webhook_url", "")

    @property
    def name(self) -> str:
        return "Telegram"

    @property
    def _max_message_length(self) -> int:
        return 4096

    # ── Abstract implementations ────────────────────────────────────────────

    async def connect(self) -> bool:
        if not TELEGRAM_AVAILABLE:
            logger.error("python-telegram-bot not installed: pip install python-telegram-bot")
            return False
        if not self._token:
            logger.error("No Telegram token configured")
            return False

        app = Application.builder().token(self._token).build()
        self._app = app
        self._bot = app.bot

        # Register message handler
        app.add_handler(TGMsgHandler(
            filters.ALL & ~filters.COMMAND,
            self._on_telegram_message,
        ))

        try:
            await app.initialize()
            await app.start()

            if self._use_webhook and self._webhook_url:
                await self._bot.set_webhook(self._webhook_url)
                logger.info("Telegram webhook set: %s", self._webhook_url)
            else:
                await app.updater.start_polling(drop_pending_updates=True)
                logger.info("Telegram polling started")

            self._running = True
            return True
        except Exception as e:
            logger.error("Telegram connection failed: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        await self.cancel_background_tasks()
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning("Telegram disconnect error: %s", e)
            self._app = None
            self._bot = None

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            # Try MarkdownV2 first
            kwargs: Dict[str, Any] = {"chat_id": int(chat_id)}
            thread_id = (metadata or {}).get("thread_id")
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)
            if reply_to:
                kwargs["reply_to_message_id"] = int(reply_to)

            try:
                formatted = format_markdown_v2(content)
                msg = await self._bot.send_message(
                    text=formatted, parse_mode=ParseMode.MARKDOWN_V2, **kwargs,
                )
            except Exception:
                # Fallback to plain text
                msg = await self._bot.send_message(text=content, **kwargs)

            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            error_str = str(e).lower()
            retryable = any(p in error_str for p in ("timeout", "network", "connection"))
            return SendResult(success=False, error=str(e), retryable=retryable)

    # ── Optional overrides ──────────────────────────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, content: str,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            await self._bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=content,
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        if not self._bot:
            return
        try:
            kwargs = {"chat_id": int(chat_id), "action": "typing"}
            thread_id = (metadata or {}).get("thread_id") if metadata else None
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)
            await self._bot.send_chat_action(**kwargs)
        except Exception as exc:
            logger.debug("send_typing: suppressed %s", exc)

    async def send_voice(
        self, chat_id: str, audio_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            with open(audio_path, "rb") as f:
                msg = await self._bot.send_voice(
                    chat_id=int(chat_id), voice=f, caption=caption,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            with open(image_path, "rb") as f:
                msg = await self._bot.send_photo(
                    chat_id=int(chat_id), photo=f, caption=caption,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(
        self, chat_id: str, image_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            msg = await self._bot.send_photo(
                chat_id=int(chat_id), photo=image_url, caption=caption,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_video(
        self, chat_id: str, video_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            with open(video_path, "rb") as f:
                msg = await self._bot.send_video(
                    chat_id=int(chat_id), video=f, caption=caption,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            name = file_name or Path(file_path).name
            with open(file_path, "rb") as f:
                msg = await self._bot.send_document(
                    chat_id=int(chat_id), document=f,
                    filename=name, caption=caption,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_animation(
        self, chat_id: str, animation_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            msg = await self._bot.send_animation(
                chat_id=int(chat_id), animation=animation_url, caption=caption,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if not self._bot:
            return {"name": chat_id, "type": "unknown"}
        try:
            chat = await self._bot.get_chat(int(chat_id))
            chat_type = "dm" if chat.type == "private" else chat.type
            return {"name": chat.title or chat.first_name or chat_id, "type": chat_type}
        except Exception:
            return {"name": chat_id, "type": "unknown"}

    def format_message(self, content: str) -> str:
        return format_markdown_v2(content)

    # ── Internal ────────────────────────────────────────────────────────────

    async def _on_telegram_message(self, update: Any, context: Any) -> None:
        """Handle incoming Telegram message → normalize → dispatch."""
        message = update.effective_message
        if not message:
            return

        user = update.effective_user
        if not user:
            return

        # Permission check
        if self._allowed_users and user.id not in self._allowed_users:
            return

        # Build text content
        text = message.text or message.caption or ""

        # Determine message type and collect media
        msg_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []

        if message.photo:
            msg_type = MessageType.PHOTO
            # Get highest resolution photo
            photo = message.photo[-1]
            file = await photo.get_file()
            media_urls.append(file.file_path)
            media_types.append("image/jpeg")
        elif message.voice:
            msg_type = MessageType.VOICE
            file = await message.voice.get_file()
            media_urls.append(file.file_path)
            media_types.append(message.voice.mime_type or "audio/ogg")
        elif message.audio:
            msg_type = MessageType.AUDIO
            file = await message.audio.get_file()
            media_urls.append(file.file_path)
            media_types.append(message.audio.mime_type or "audio/mpeg")
        elif message.video:
            msg_type = MessageType.VIDEO
            file = await message.video.get_file()
            media_urls.append(file.file_path)
            media_types.append(message.video.mime_type or "video/mp4")
        elif message.document:
            msg_type = MessageType.DOCUMENT
            file = await message.document.get_file()
            media_urls.append(file.file_path)
            media_types.append(message.document.mime_type or "application/octet-stream")
        elif message.sticker:
            msg_type = MessageType.STICKER
            text = text or f"[Sticker: {message.sticker.emoji or ''}]"
        elif message.location:
            msg_type = MessageType.LOCATION
            text = text or f"[Location: {message.location.latitude}, {message.location.longitude}]"

        if not text and not media_urls:
            return

        # Determine chat type
        chat = update.effective_chat
        chat_type = "dm" if chat.type == "private" else "group"
        if chat.type in ("supergroup", "channel"):
            chat_type = "channel"

        # Build source
        source = self.build_source(
            chat_id=str(chat.id),
            chat_name=chat.title or chat.first_name,
            chat_type=chat_type,
            user_id=str(user.id),
            user_name=user.username or user.first_name,
            thread_id=str(message.message_thread_id) if message.message_thread_id else None,
        )

        # Interaction flags
        is_mention = False
        if self._bot and message.entities:
            for ent in message.entities:
                if ent.type == "mention" and self._bot.username:
                    mentioned = (message.text or "")[ent.offset:ent.offset + ent.length]
                    if mentioned.lower() == f"@{self._bot.username.lower()}":
                        is_mention = True
                        break
        is_reply_to_bot = False
        if message.reply_to_message and self._bot:
            reply_author = message.reply_to_message.from_user
            if reply_author and reply_author.id == self._bot.id:
                is_reply_to_bot = True

        # Reply context
        reply_to_text = None
        reply_to_id = None
        if message.reply_to_message:
            reply_to_id = str(message.reply_to_message.message_id)
            reply_to_text = (message.reply_to_message.text or "")[:500]
        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.message_id),
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            is_mention=is_mention,
            is_reply_to_bot=is_reply_to_bot,
        )
        await self.handle_message(event)
