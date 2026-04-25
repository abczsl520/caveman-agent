"""Discord Platform Adapter — BasePlatformAdapter implementation for Discord.

This is the new-generation Discord adapter built on the unified platform
abstraction layer. It replaces the legacy DiscordGateway for new deployments.

Features:
- Native Discord.py integration (discord.py 2.x)
- Slash command registration
- Thread/DM/channel support
- Attachment handling (images, audio, video, documents)
- Reaction-based processing indicators (👀 while running; ❌ on failure/cancel)
- Message editing for streaming responses
- Rate limiting per user
- Permission filtering (channels + users)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    ProcessingOutcome, SendResult,
)

logger = logging.getLogger("caveman.gateway.discord")

try:
    import discord
    from discord import app_commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None


class DiscordAdapter(BasePlatformAdapter):
    """Discord adapter built on BasePlatformAdapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[Any] = None
        self._token = config.token or ""
        self._prefix = config.extra.get("prefix", "!cave")
        self._trigger = config.extra.get("trigger", "all")
        self._allowed_channels = set(config.extra.get("allowed_channels", []))
        self._allowed_users = set(config.extra.get("allowed_users", []))
        self._register_slash = config.extra.get("register_slash", True)
        self._user_timestamps: Dict[int, List[float]] = {}

    @property
    def name(self) -> str:
        return "Discord"

    @property
    def _max_message_length(self) -> int:
        return 1900  # Discord limit is 2000, leave room for formatting

    # ── Abstract implementations ────────────────────────────────────────────

    async def connect(self) -> bool:
        if not DISCORD_AVAILABLE:
            logger.error("discord.py not installed: pip install discord.py")
            return False
        if not self._token:
            logger.error("No Discord token configured")
            return False

        intents = discord.Intents.all()
        client = discord.Client(intents=intents)
        client.tree = app_commands.CommandTree(client)
        self._client = client

        @client.event
        async def on_ready() -> None:
            logger.info("Discord connected: %s", client.user)
            self._running = True
            if self._register_slash:
                await self._sync_slash_commands()

        @client.event
        async def on_message(message: discord.Message) -> None:
            await self._on_discord_message(message)

        try:
            await client.start(self._token)
            return True
        except Exception as e:
            logger.error("Discord connection failed: %s", e)
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

        channel = self._client.get_channel(int(chat_id))
        if not channel:
            # Cache miss — common for newly created threads.  Fall back to
            # an API call so we don't silently drop the reply.
            try:
                channel = await self._client.fetch_channel(int(chat_id))
            except Exception:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

        try:
            # Handle thread routing via metadata override
            thread_id = (metadata or {}).get("thread_id")
            if thread_id:
                thread = self._client.get_channel(int(thread_id))
                if not thread:
                    try:
                        thread = await self._client.fetch_channel(int(thread_id))
                    except Exception:
                        logger.debug("send: thread %s not fetchable", thread_id)
                if thread:
                    channel = thread

            # Reply to specific message if requested
            reference = None
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    reference = ref_msg
                except Exception as exc:
                    logger.debug("send: suppressed %s", exc)

            if reference:
                msg = await reference.reply(content)
            else:
                msg = await channel.send(content)

            return SendResult(success=True, message_id=str(msg.id))
        except discord.HTTPException as e:
            retryable = e.status in (429, 500, 502, 503, 504)
            return SendResult(success=False, error=str(e), retryable=retryable)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    # ── Optional overrides ──────────────────────────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, content: str,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return SendResult(success=False, error="Channel not found")
        try:
            msg = await channel.fetch_message(int(message_id))
            await msg.edit(content=content)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        if not self._client:
            return
        channel = self._client.get_channel(int(chat_id))
        if channel:
            try:
                await channel.typing()
            except Exception as exc:
                logger.debug("send_typing: suppressed %s", exc)

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return SendResult(success=False, error="Channel not found")
        try:
            file = discord.File(image_path)
            msg = await channel.send(content=caption, file=file)
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_voice(
        self, chat_id: str, audio_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return SendResult(success=False, error="Channel not found")
        try:
            file = discord.File(audio_path)
            msg = await channel.send(content=caption, file=file)
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return SendResult(success=False, error="Channel not found")
        try:
            name = file_name or Path(file_path).name
            file = discord.File(file_path, filename=name)
            msg = await channel.send(content=caption, file=file)
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if not self._client:
            return {"name": chat_id, "type": "unknown"}
        channel = self._client.get_channel(int(chat_id))
        if not channel:
            return {"name": chat_id, "type": "unknown"}
        chat_type = "dm" if isinstance(channel, discord.DMChannel) else "channel"
        if isinstance(channel, discord.Thread):
            chat_type = "thread"
        return {"name": getattr(channel, "name", chat_id), "type": chat_type}

    # ── Lifecycle hooks ─────────────────────────────────────────────────────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add 👀 reaction when processing starts."""
        if event.raw_message and hasattr(event.raw_message, "add_reaction"):
            try:
                await event.raw_message.add_reaction("👀")
            except Exception as exc:
                logger.debug("on_processing_start: suppressed %s", exc)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome,
    ) -> None:
        """Clear processing indicator and surface only non-success outcomes.

        A normal handler return means "message processing finished", not that the
        user's real task is objectively complete.  Do not add a success reaction
        here: it is user-visible and reads as premature task completion for
        long-running or multi-step work.  Keep failure/cancelled visible.
        """
        msg = event.raw_message
        if not msg or not hasattr(msg, "remove_reaction"):
            return
        try:
            if self._client and self._client.user:
                await msg.remove_reaction("👀", self._client.user)
        except Exception:
            pass  # intentional: Exception suppressed
        if outcome in (ProcessingOutcome.SUCCESS, ProcessingOutcome.NO_RESPONSE):
            return
        try:
            await msg.add_reaction("❌")
        except Exception as exc:
            logger.debug("on_processing_complete: suppressed %s", exc)

    # ── Internal ────────────────────────────────────────────────────────────

    async def _on_discord_message(self, message) -> None:
        """Handle incoming Discord message → normalize → dispatch."""
        if message.author.bot:
            return

        # Skip non-user messages: thread creation notices, pins, joins, etc.
        # Only process DEFAULT (normal) and REPLY message types.
        if discord and message.type not in (
            discord.MessageType.default, discord.MessageType.reply,
        ):
            return

        # Skip thread-starter messages that appear in the parent channel.
        # When a user creates a thread from a message, Discord attaches a
        # .thread reference to the original message and fires on_message
        # in the parent channel.  Processing it would cause the bot to
        # reply in both the channel AND the thread.
        if getattr(message, "thread", None) and not (
            discord and isinstance(message.channel, discord.Thread)
        ):
            return

        content = message.content.strip()
        has_attachments = bool(message.attachments)
        if not content and not has_attachments:
            return
        if not self._check_permissions(message):
            return
        if not self._check_rate_limit(message.author.id):
            try:
                await message.reply("⏳ 请求太频繁，请稍后再试。")
            except Exception as exc:
                logger.debug("_on_discord_message: suppressed %s", exc)
            return
        if not self._should_respond(content, message):
            return

        # Build normalized MessageEvent
        event = self._build_event(content, message)
        await self.handle_message(event)

    def _check_permissions(self, message) -> bool:
        if self._allowed_channels:
            ch_id = message.channel.id
            parent_id = getattr(message.channel, "parent_id", None)
            if ch_id not in self._allowed_channels and parent_id not in self._allowed_channels:
                return False
        if self._allowed_users and message.author.id not in self._allowed_users:
            return False
        return True

    def _check_rate_limit(self, user_id: int) -> bool:
        """Per-user rate limiting: 5 requests per 60s."""
        if self._allowed_users and user_id in self._allowed_users:
            return True  # Allowlisted users bypass rate limit
        import time
        now = time.monotonic()
        timestamps = self._user_timestamps.setdefault(user_id, [])
        timestamps[:] = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= 5:
            return False
        timestamps.append(now)
        return True

    def _should_respond(self, content: str, message) -> bool:
        if not self._client or not self._client.user:
            return False
        is_mention = self._client.user in message.mentions
        is_prefix = content.startswith(self._prefix)
        is_dm = isinstance(message.channel, discord.DMChannel) if discord else False
        is_thread = isinstance(message.channel, discord.Thread) if discord else False

        if self._trigger == "all":
            return True
        elif self._trigger == "thread":
            return is_thread or is_dm or is_mention or is_prefix
        return is_mention or is_prefix or is_dm

    def _build_event(self, content: str, message) -> MessageEvent:
        """Convert Discord message to normalized MessageEvent."""
        # Strip prefix/mention
        task = content
        if self._client and self._client.user:
            if self._client.user in message.mentions:
                task = task.replace(f"<@{self._client.user.id}>", "").strip()
        if content.startswith(self._prefix):
            task = task[len(self._prefix):].strip()

        # Determine chat type
        is_dm = isinstance(message.channel, discord.DMChannel) if discord else False
        is_thread = isinstance(message.channel, discord.Thread) if discord else False
        chat_type = "dm" if is_dm else ("thread" if is_thread else "channel")

        # Build source
        source = self.build_source(
            chat_id=str(message.channel.id),
            chat_name=getattr(message.channel, "name", None),
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=str(message.author),
            thread_id=str(message.channel.id) if is_thread else None,
            chat_topic=getattr(message.channel, "topic", None),
        )

        # Determine message type and collect media
        msg_type = MessageType.TEXT
        media_urls = []
        media_types = []
        if message.attachments:
            for att in message.attachments:
                ct = att.content_type or ""
                media_urls.append(att.url)
                media_types.append(ct)
                if ct.startswith("image/"):
                    msg_type = MessageType.PHOTO
                elif ct.startswith("audio/"):
                    msg_type = MessageType.AUDIO
                elif ct.startswith("video/"):
                    msg_type = MessageType.VIDEO

            # Append attachment descriptions to task
            parts = []
            for att in message.attachments:
                ct = att.content_type or ""
                if ct.startswith("image/"):
                    parts.append(f"[Image: {att.filename} ({att.url})]")
                elif ct.startswith("audio/"):
                    parts.append(f"[Audio: {att.filename} ({att.url})]")
                elif ct.startswith("video/"):
                    parts.append(f"[Video: {att.filename} ({att.url})]")
                else:
                    parts.append(f"[File: {att.filename} ({att.url})]")
            if parts:
                task = task + "\n" + "\n".join(parts) if task else "\n".join(parts)

        # Interaction flags
        is_mention = bool(self._client and self._client.user and self._client.user in message.mentions)
        is_reply_to_bot = False
        if message.reference and getattr(message.reference, "resolved", None):
            ref_author = getattr(message.reference.resolved, "author", None)
            if ref_author and self._client and self._client.user:
                is_reply_to_bot = ref_author.id == self._client.user.id

        # Reply context
        reply_to_text = None
        if message.reference and message.reference.resolved:
            ref = message.reference.resolved
            reply_to_text = ref.content[:500] if hasattr(ref, "content") and ref.content else None

        return MessageEvent(
            text=task or "",
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.id),
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=str(message.reference.message_id) if message.reference else None,
            reply_to_text=reply_to_text,
            is_mention=is_mention,
            is_reply_to_bot=is_reply_to_bot,
        )

    async def _sync_slash_commands(self) -> None:
        """Register native Discord Application Commands."""
        try:
            from caveman.gateway.discord_slash import sync_slash_commands
            await sync_slash_commands(self._client)
        except Exception as e:
            logger.warning("Slash command sync failed: %s", e)
