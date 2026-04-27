"""Discord gateway — receive tasks from Discord, run Caveman, reply.

Trigger modes:
  - "all": respond to every message (like a chat assistant)
  - "prefix": only respond to !cave prefix or @mentions (default)
  - "thread": respond to all messages in threads, prefix/mention in channels

Slash commands:
  - Native Discord Application Commands (registered on ready)
  - Text-based /commands intercepted in on_message
"""
from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable

from .base import Gateway
from caveman.gateway.mock_agent import GatewayMockAgent as _GatewayMockAgent
from caveman.utils import split_message

logger = logging.getLogger("caveman.gateway.discord")

class DiscordGateway(Gateway):
    """Discord bot gateway for Caveman."""

    def __init__(
        self,
        token: str,
        prefix: str = "!cave",
        trigger: str = "all",  # "all" | "prefix" | "thread"
        allowed_channels: list[int] | None = None,
        allowed_users: list[int] | None = None,
        max_message_len: int = 1900,
        register_slash: bool = True,
        locale: str = "en",
    ):
        self.token = token
        self.prefix = prefix
        self.trigger = trigger
        self.allowed_channels = set(allowed_channels) if allowed_channels else None
        self.allowed_users = set(allowed_users) if allowed_users else None
        self.max_message_len = max_message_len
        self.register_slash = register_slash
        self.locale = locale
        self._bot = None
        self._task_handler: Callable[[str, dict], Awaitable[str]] | None = None
        self._debouncer = None  # Initialized on start

    @property
    def name(self) -> str:
        return "discord"

    def on_task(self, handler: Callable[[str, dict], Awaitable[str]]) -> None:
        self._task_handler = handler

    async def _handle_slash_command(self, text: str, message) -> bool:
        """Intercept /commands and handle via dispatcher. Returns True if handled."""
        if not text.startswith("/"):
            return False

        from caveman.commands.dispatcher import parse_command, dispatch

        name, _ = parse_command(text)
        if name is None:
            return False

        from caveman.commands.registry import resolve_command
        cmd_def = resolve_command(name)
        if cmd_def is None:
            return False

        # Build a minimal mock agent for commands that need it
        agent = _GatewayMockAgent()

        # Chinese command name → zh locale
        cmd_word = text.lstrip("/").split()[0] if text.startswith("/") else ""
        locale = "zh" if any(ord(c) > 0x4E00 for c in cmd_word) else self.locale

        responses = []
        result = await dispatch(
            text, agent, surface="discord",
            respond_fn=lambda msg: responses.append(msg),
            locale=locale,
        )

        if result == "exit":
            # /quit doesn't make sense on Discord
            await message.reply("👋 Use this command in CLI mode.")
            return True

        if responses:
            for resp in responses:
                await self._send_split(message, resp)
        return True

    async def _sync_slash_commands(self, client):
        """Register native Discord Application Commands."""
        from caveman.gateway.discord_slash import sync_slash_commands
        await sync_slash_commands(client, locale=self.locale)


    def _save_locale(self, locale: str):
        """Persist locale to config file."""
        try:
            import yaml
            from caveman.config.loader import DEFAULT_CONFIG_PATH
            with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["locale"] = locale
            with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            logger.info("Locale saved to config: %s", locale)
        except Exception as e:
            logger.warning("Failed to save locale: %s", e)

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            raise ImportError("discord.py required: pip install discord.py")

        intents = discord.Intents.all()
        client = discord.Client(intents=intents)
        client.tree = discord.app_commands.CommandTree(client)
        self._bot = client

        @client.event
        async def on_ready() -> None:
            logger.info(f"Caveman Discord bot ready: {client.user}")
            for g in client.guilds:
                logger.info(f"  Guild: {g.name} ({g.id})")
            # Update runtime state — gateway is now actually connected
            try:
                from caveman.gateway.status import write_runtime_state
                write_runtime_state(state="running", platform="discord",
                                    platform_state="connected")
            except Exception:
                pass  # Non-critical — don't crash on status update failure
            # Sync slash commands
            if self.register_slash:
                await self._sync_slash_commands(client)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            content = message.content.strip()
            has_attachments = bool(message.attachments)
            if not content and not has_attachments:
                return
            if not self._check_permissions(message):
                return

            # Per-user rate limiting (max 5 requests per 60s)
            user_id = message.author.id
            now = asyncio.get_running_loop().time()
            if not hasattr(self, '_user_timestamps'):
                self._user_timestamps = {}
            timestamps = self._user_timestamps.setdefault(user_id, [])
            timestamps[:] = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= 5 and (not self.allowed_users or user_id not in self.allowed_users):
                try:
                    await message.reply("⏳ 请求太频繁，请稍后再试。")
                except Exception as exc:
                    logger.debug("on_message: suppressed %s", exc)
                return
            timestamps.append(now)

            if content.startswith("/"):
                if await self._handle_slash_command(content, message):
                    return

            if not self._should_respond(content, message, client.user):
                return

            task = self._extract_task(content, message, client.user)
            if not task:
                return
            if not self._task_handler:
                await message.reply("⚠️ No task handler configured.")
                return

            # Ack reaction — show we received the message
            try:
                await message.add_reaction("👀")
            except Exception:
                pass  # Reaction may fail in DMs or without permissions

            await self._run_task_with_typing(task, self._build_context(message), message)

            # Remove ack reaction after completion
            try:
                await message.remove_reaction("👀", client.user)
            except Exception:
                pass  # intentional: Exception suppressed

        try:
            self._running = True
            await client.start(self.token)
        except discord.LoginFailure:
            logger.error("Invalid Discord token")
            raise
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._bot:
            await self._bot.close()

    async def send_message(self, channel_id: str, text: str) -> None:
        return await self.send(int(channel_id), text)

    async def send_reply(self, channel_id: str, text: str, reply_to: int) -> None:
        """Send a message as a reply to a specific message."""
        if not self._bot:
            return None
        channel = self._bot.get_channel(int(channel_id))
        if not channel:
            return None
        try:
            ref_msg = await channel.fetch_message(reply_to)
            return await ref_msg.reply(text)
        except Exception:
            return await channel.send(text)

    async def send_confirm(self, channel_id: str, content: str) -> bool | None:
        """Send confirmation buttons. Returns True/False/None (timeout)."""
        if not self._bot:
            return None
        channel = self._bot.get_channel(int(channel_id))
        if not channel:
            return None
        from caveman.gateway.discord_buttons import ConfirmView, send_with_buttons
        view = ConfirmView()
        await send_with_buttons(channel, content, view)
        return await view.wait_for_result()

    async def send_choices(self, channel_id: str, content: str, choices: list[str]) -> str | None:
        """Send choice buttons. Returns selected choice or None."""
        if not self._bot:
            return None
        channel = self._bot.get_channel(int(channel_id))
        if not channel:
            return None
        from caveman.gateway.discord_buttons import ChoiceView, send_with_buttons
        view = ChoiceView(choices)
        await send_with_buttons(channel, content, view)
        return await view.wait_for_result()

    async def on_message(self, handler: Callable) -> None:
        self.on_task(handler)

    async def send(self, channel_id: int, content: str) -> None:
        if not self._bot:
            return None
        channel = self._bot.get_channel(channel_id)
        if channel:
            last_msg = None
            for chunk in split_message(content, self.max_message_len):
                last_msg = await channel.send(chunk)
            return last_msg
        return None

    async def edit_message(self, channel_id: int, message_id: int, content: str) -> None:
        """Edit a previously sent message."""
        if not self._bot:
            return
        channel = self._bot.get_channel(channel_id)
        if channel:
            try:
                msg = await channel.fetch_message(message_id)
                await msg.edit(content=content)
            except Exception as e:
                logger.warning("Failed to edit message %d: %s", message_id, e)

    async def _send_split(self, message, content: str):
        chunks = split_message(content, self.max_message_len)
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)
            except Exception as e:
                logger.warning("Failed to send chunk %d: %s", i, e)
                try:
                    await message.channel.send(chunk)
                except Exception:
                    break

    def _check_permissions(self, message) -> bool:
        """Check if message passes channel/user permission filters."""
        if self.allowed_channels and message.channel.id not in self.allowed_channels:
            parent_id = getattr(message.channel, "parent_id", None)
            if not parent_id or parent_id not in self.allowed_channels:
                return False
        if self.allowed_users and message.author.id not in self.allowed_users:
            return False
        return True

    def _should_respond(self, content: str, message, client_user) -> bool:
        """Determine if the bot should respond to this message."""
        import discord
        is_mention = client_user in message.mentions if client_user else False
        is_prefix = content.startswith(self.prefix)
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_thread = isinstance(message.channel, discord.Thread)

        if self.trigger == "all":
            return True
        elif self.trigger == "thread":
            return is_thread or is_dm or is_mention or is_prefix
        return is_mention or is_prefix or is_dm

    def _extract_task(self, content: str, message, client_user) -> str:
        """Strip prefix/mention from content to get the task text."""
        task = content
        is_mention = client_user in message.mentions if client_user else False
        if is_mention and client_user:
            task = task.replace(f"<@{client_user.id}>", "").strip()
        if content.startswith(self.prefix):
            task = task[len(self.prefix):].strip()

        # Append attachment descriptions
        if message.attachments:
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

        return task

    def _build_context(self, message) -> dict:
        """Build task context dict from a Discord message."""
        import discord
        is_thread = isinstance(message.channel, discord.Thread)
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mention = self._client.user in message.mentions if self._client and self._client.user else False
        is_reply_to_bot = False
        if message.reference and getattr(message.reference, "resolved", None):
            ref = message.reference.resolved
            if hasattr(ref, "author") and self._client and self._client.user:
                is_reply_to_bot = ref.author.id == self._client.user.id
        chat_type = "dm" if is_dm else ("thread" if is_thread else "channel")
        ctx = {
            "channel_id": message.channel.id,
            "user_id": message.author.id,
            "user_name": str(message.author),
            "username": str(message.author),
            "guild_id": message.guild.id if message.guild else None,
            "message_id": message.id,
            "is_thread": is_thread,
            "thread_id": str(message.channel.id) if is_thread else "",
            "chat_type": chat_type,
            "is_mention": is_mention,
            "is_reply_to_bot": is_reply_to_bot,
            "thread_name": message.channel.name if is_thread else None,
            "gateway_name": "discord",
        }
        # Include reply context if user is replying to a message
        if message.reference and message.reference.message_id:
            try:
                ref_msg = message.reference.resolved
                if ref_msg:
                    ctx["reply_to"] = {
                        "message_id": ref_msg.id,
                        "author": str(ref_msg.author),
                        "content": ref_msg.content[:500] if ref_msg.content else "",
                    }
            except Exception:
                ctx["reply_to"] = {"message_id": message.reference.message_id}

        # Include attachment metadata
        if message.attachments:
            ctx["attachments"] = [
                {
                    "filename": att.filename,
                    "url": att.url,
                    "content_type": att.content_type or "",
                    "size": att.size,
                }
                for att in message.attachments
            ]

        return ctx

    async def _run_task_with_typing(self, task: str, context: dict, message) -> None:
        """Run a task with typing indicator and send the result."""
        typing_task = asyncio.create_task(self._keep_typing(message.channel))
        try:
            result = await self._task_handler(task, context)
        except Exception as e:
            logger.exception("Task handler failed: %s", e)
            result = "⚠️ 出了点问题，请重试。"
        finally:
            typing_task.cancel()

        if result and result.strip():
            await self._send_split(message, result)

    @staticmethod
    async def _keep_typing(channel) -> None:
        """Send typing indicator every 8s until cancelled."""
        try:
            while True:
                await channel.typing()
                await asyncio.sleep(8)
        except asyncio.CancelledError:
            pass  # intentional: Exception suppressed
