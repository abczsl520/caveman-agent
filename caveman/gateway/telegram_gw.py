"""Telegram gateway — receive tasks from Telegram, run Caveman, reply."""
from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable

from .base import Gateway
from caveman.utils import split_message

logger = logging.getLogger("caveman.gateway.telegram")


class TelegramGateway(Gateway):
    """Telegram bot gateway for Caveman."""

    def __init__(
        self,
        token: str,
        allowed_users: list[int] | None = None,
        max_message_len: int = 4000,
    ):
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self.max_message_len = max_message_len
        self._app = None
        self._task_handler: Callable[[str, dict], Awaitable[str]] | None = None

    @property
    def name(self) -> str:
        return "telegram"

    def on_task(self, handler: Callable[[str, dict], Awaitable[str]]) -> None:
        self._task_handler = handler

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
        except ImportError:
            raise ImportError("python-telegram-bot required: pip install python-telegram-bot")

        app = ApplicationBuilder().token(self.token).build()

        async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text("🦴 Caveman ready! Send me a task.")

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.message.text:
                return

            user_id = update.message.from_user.id
            if self.allowed_users and user_id not in self.allowed_users:
                await update.message.reply_text("⚠️ Not authorized.")
                return

            task = update.message.text.strip()
            if not task:
                return

            if not self._task_handler:
                await update.message.reply_text("⚠️ No handler configured.")
                return

            ctx = {
                "chat_id": update.message.chat_id,
                "channel_id": update.message.chat_id,
                "user_id": user_id,
                "username": update.message.from_user.username or str(user_id),
                "message_id": update.message.message_id,
                "gateway_name": "telegram",
            }

            # Send typing action
            await update.message.chat.send_action("typing")

            try:
                result = await self._task_handler(task, ctx)
            except Exception as e:
                logger.exception("Task handler failed: %s", e)
                result = f"❌ 任务处理链路异常：{type(e).__name__}: {str(e)[:300]}。已记录日志；这不是任务完成信号。"

            # Split long messages
            for chunk in split_message(result, self.max_message_len):
                await update.message.reply_text(chunk)

        app.add_handler(CommandHandler("start", handle_start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        self._app = app
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        self._running = True
        logger.info("Telegram gateway started")
        try:
            from caveman.gateway.status import write_runtime_state
            write_runtime_state(state="running", platform="telegram",
                                platform_state="connected")
        except Exception as exc:
            logger.debug("unknown: suppressed %s", exc)

    async def stop(self) -> None:
        self._running = False
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_message(self, channel_id: str, text: str) -> None:
        """Send a message to a chat (abstract method impl)."""
        if self._app and self._app.bot:
            for chunk in split_message(text, self.max_message_len):
                await self._app.bot.send_message(chat_id=int(channel_id), text=chunk)

    async def on_message(self, handler: Callable) -> None:
        """Register message handler (uses on_task internally)."""
        self._task_handler = handler
