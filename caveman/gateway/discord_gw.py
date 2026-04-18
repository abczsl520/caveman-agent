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

    @property
    def name(self) -> str:
        return "discord"

    def on_task(self, handler: Callable[[str, dict], Awaitable[str]]):
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
        try:
            import discord
            from discord import app_commands
            from caveman.commands.dispatcher import dispatch
            from caveman.commands.registry import COMMAND_REGISTRY
        except ImportError:
            return

        tree = client.tree
        guilds = client.guilds  # Available in on_ready

        # Chinese aliases — top 40 most useful (guild limit: 100 per guild)
        # 56 English + 40 Chinese + 2 language = 98 commands
        zh_aliases = {
            # Core
            "帮助": "help", "状态": "status", "模型": "model",
            "命令": "commands", "记忆": "memory", "工具": "tools",
            "技能": "skills", "诊断": "doctor", "自检": "selftest",
            "护盾": "shield", "飞轮": "flywheel", "配置": "config",
            # Session
            "新会话": "new", "历史": "history", "重试": "retry",
            "撤销": "undo", "停止": "stop", "后台": "background",
            # Config
            "提供商": "provider", "推理": "reasoning", "快速": "fast",
            "语音": "voice", "主题": "skin",
            # Tools
            "引擎": "engines", "定时": "cron", "插件": "plugins",
            "导入": "import", "浏览器": "browser",
            # Info
            "用量": "usage", "分析": "insights", "回忆": "recall",
            "反思": "reflect", "审计": "audit", "限速": "ratelimit",
            # System
            "重启": "restart", "更新": "update", "档案": "profile",
            # Caveman
            "轨迹": "trajectory", "基准": "bench", "维基": "wiki",
            # Language (special)
            "语言": None,
        }

        def _make_handler(cmd_name: str, locale: str = "en"):
            """Factory to avoid closure-over-loop-variable bug."""
            async def _handler(interaction: discord.Interaction, args: str = ""):
                await interaction.response.defer()
                agent = _GatewayMockAgent()
                responses = []
                # Pass args so subcommands work: /memory stats, /记忆 统计
                full_cmd = f"/{cmd_name} {args}".strip() if args else f"/{cmd_name}"
                await dispatch(
                    full_cmd, agent, surface="discord",
                    respond_fn=lambda msg: responses.append(msg),
                    locale=locale,
                )
                text = "\n".join(responses) if responses else "✅"
                chunks = split_message(text, 1900)
                await interaction.followup.send(chunks[0])
                ch = interaction.channel
                for chunk in chunks[1:]:
                    if ch:
                        await ch.send(chunk)
            return _handler

        # Register ALL eligible English commands
        for cmd_def in COMMAND_REGISTRY:
            if cmd_def.cli_only or cmd_def.hidden:
                continue
            handler = _make_handler(cmd_def.name, locale=self.locale)
            desc = cmd_def.desc(self.locale)[:100]
            dc = tree.command(name=cmd_def.name, description=desc, guilds=guilds)(handler)
            # Add optional args parameter for commands with subcommands
            if cmd_def.subcommands or cmd_def.args_hint:
                hint = cmd_def.args_hint or " ".join(cmd_def.subcommands)
                app_commands.describe(args=hint[:100])(dc)

        # Register Chinese alias commands — always locale="zh"
        for zh_name, en_name in zh_aliases.items():
            if en_name is None:
                continue  # skip /语言, registered below
            handler = _make_handler(en_name, locale="zh")
            en_cmd = next((c for c in COMMAND_REGISTRY if c.name == en_name), None)
            desc = en_cmd.desc("zh")[:100] if en_cmd else zh_name
            dc = tree.command(name=zh_name, description=desc, guilds=guilds)(handler)
            if en_cmd and (en_cmd.subcommands or en_cmd.args_hint):
                hint = en_cmd.args_hint or " ".join(en_cmd.subcommands)
                app_commands.describe(args=hint[:100])(dc)

        # /language and /语言 — switch locale
        _Choice = app_commands.Choice

        lang_choices = [
            _Choice(name="English", value="en"),
            _Choice(name="中文", value="zh"),
            _Choice(name="日本語", value="ja"),
            _Choice(name="한국어", value="ko"),
        ]

        @tree.command(name="language", description="Switch language / 切换语言", guilds=guilds)
        @app_commands.describe(lang="Language code: en, zh, ja, ko")
        @app_commands.choices(lang=lang_choices)
        async def _language_handler(interaction: discord.Interaction, lang: str):
            old = self.locale
            self.locale = lang
            try:
                self._save_locale(lang)
            except Exception as e:
                logger.debug("Suppressed in _language_handler: %s", e)
            name = next((c.name for c in lang_choices if c.value == lang), lang)
            await interaction.response.send_message(
                f"🌐 Language: {old} → {lang} ({name})\n"
                f"Slash command descriptions will update on next restart."
            )

        @tree.command(name="语言", description="切换语言 / Switch language", guilds=guilds)
        @app_commands.describe(lang="语言: en=英文, zh=中文, ja=日语, ko=韩语")
        @app_commands.choices(lang=lang_choices)
        async def _language_zh_handler(interaction: discord.Interaction, lang: str):
            old = self.locale
            self.locale = lang
            try:
                self._save_locale(lang)
            except Exception as e:
                logger.debug("Suppressed in _language_zh_handler: %s", e)
            name = next((c.name for c in lang_choices if c.value == lang), lang)
            await interaction.response.send_message(
                f"🌐 语言已切换: {old} → {lang} ({name})\n"
                f"斜杠命令说明将在下次重启后更新。"
            )

        try:
            # Clear stale global commands first (from previous versions)
            tree.clear_commands(guild=None)
            await tree.sync()
            logger.info("Cleared global commands")

            # Sync guild commands
            for guild in guilds:
                synced = await tree.sync(guild=guild)
                logger.info("Synced %d slash commands to guild %s", len(synced), guild.name)
        except Exception as e:
            logger.warning("Failed to sync slash commands: %s", e)

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

    async def start(self):
        try:
            import discord
        except ImportError:
            raise ImportError("discord.py required: pip install discord.py")

        intents = discord.Intents.all()
        client = discord.Client(intents=intents)
        client.tree = discord.app_commands.CommandTree(client)
        self._bot = client

        @client.event
        async def on_ready():
            logger.info(f"Caveman Discord bot ready: {client.user}")
            for g in client.guilds:
                logger.info(f"  Guild: {g.name} ({g.id})")
            # Sync slash commands
            if self.register_slash:
                await self._sync_slash_commands(client)

        @client.event
        async def on_message(message: discord.Message):
            if message.author.bot:
                return
            content = message.content.strip()
            if not content or not self._check_permissions(message):
                return

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

            await self._run_task_with_typing(task, self._build_context(message), message)

        try:
            self._running = True
            await client.start(self.token)
        except discord.LoginFailure:
            logger.error("Invalid Discord token")
            raise
        finally:
            self._running = False

    async def stop(self):
        self._running = False
        if self._bot:
            await self._bot.close()

    async def send_message(self, channel_id: str, text: str):
        return await self.send(int(channel_id), text)

    async def on_message(self, handler: Callable) -> None:
        self.on_task(handler)

    async def send(self, channel_id: int, content: str):
        if not self._bot:
            return None
        channel = self._bot.get_channel(channel_id)
        if channel:
            last_msg = None
            for chunk in split_message(content, self.max_message_len):
                last_msg = await channel.send(chunk)
            return last_msg
        return None

    async def edit_message(self, channel_id: int, message_id: int, content: str):
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
        import discord
        task = content
        is_mention = client_user in message.mentions if client_user else False
        if is_mention and client_user:
            task = task.replace(f"<@{client_user.id}>", "").strip()
        if content.startswith(self.prefix):
            task = task[len(self.prefix):].strip()
        return task

    def _build_context(self, message) -> dict:
        """Build task context dict from a Discord message."""
        import discord
        is_thread = isinstance(message.channel, discord.Thread)
        ctx = {
            "channel_id": message.channel.id,
            "user_id": message.author.id,
            "username": str(message.author),
            "guild_id": message.guild.id if message.guild else None,
            "message_id": message.id,
            "is_thread": is_thread,
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
        return ctx

    async def _run_task_with_typing(self, task: str, context: dict, message) -> None:
        """Run a task with typing indicator and send the result."""
        typing_task = asyncio.create_task(self._keep_typing(message.channel))
        try:
            result = await asyncio.wait_for(
                self._task_handler(task, context),
                timeout=2100.0,
            )
        except asyncio.TimeoutError:
            result = "⏸️ 任务超过 35 分钟，进度已保存。发消息可继续。"
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
            pass
