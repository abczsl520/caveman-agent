"""Discord button views — interactive UI components.

Provides reusable button views for common interactions:
- ConfirmView: Yes/No confirmation
- ChoiceView: Multiple choice selection
- PaginationView: Page through results
"""
from __future__ import annotations
import asyncio
import logging

import discord

__all__ = [
    "BUTTON_TIMEOUT",
    "ConfirmView",
    "ChoiceView",
    "PaginationView",
    "send_with_buttons",
]


logger = logging.getLogger(__name__)

BUTTON_TIMEOUT = 120.0  # 2 minutes


class ConfirmView(discord.ui.View):
    """Yes/No confirmation buttons."""

    def __init__(self, timeout: float = BUTTON_TIMEOUT):
        super().__init__(timeout=timeout)
        self.value: bool | None = None
        self._event = asyncio.Event()

    @discord.ui.button(label="✅ 确认", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.value = True
        self._event.set()
        await interaction.response.edit_message(content=f"✅ 已确认", view=None)
        self.stop()

    @discord.ui.button(label="❌ 取消", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.value = False
        self._event.set()
        await interaction.response.edit_message(content=f"❌ 已取消", view=None)
        self.stop()

    async def wait_for_result(self) -> bool | None:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            logger.debug("wait_for_result: suppressed %s", exc)
        return self.value


class ChoiceView(discord.ui.View):
    """Multiple choice buttons (up to 5 options)."""

    def __init__(self, choices: list[str], timeout: float = BUTTON_TIMEOUT):
        super().__init__(timeout=timeout)
        self.value: str | None = None
        self._event = asyncio.Event()
        for i, choice in enumerate(choices[:5]):
            button = discord.ui.Button(
                label=choice[:80], style=discord.ButtonStyle.primary,
                custom_id=f"choice_{i}",
            )
            button.callback = self._make_callback(choice)
            self.add_item(button)

    def _make_callback(self, choice: str):
        async def callback(interaction: discord.Interaction) -> None:
            self.value = choice
            self._event.set()
            await interaction.response.edit_message(
                content=f"选择了: {choice}", view=None)
            self.stop()
        return callback

    async def wait_for_result(self) -> str | None:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            logger.debug("wait_for_result: suppressed %s", exc)
        return self.value


class PaginationView(discord.ui.View):
    """Page through results with ◀️ ▶️ buttons."""

    def __init__(self, pages: list[str], timeout: float = BUTTON_TIMEOUT):
        super().__init__(timeout=timeout)
        self.pages = pages
        self.current = 0

    @discord.ui.button(label="◀️", style=discord.ButtonStyle.secondary)
    async def prev_page(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.current = max(0, self.current - 1)
        await interaction.response.edit_message(
            content=self._page_content(), view=self)

    @discord.ui.button(label="▶️", style=discord.ButtonStyle.secondary)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.current = min(len(self.pages) - 1, self.current + 1)
        await interaction.response.edit_message(
            content=self._page_content(), view=self)

    def _page_content(self) -> str:
        return f"{self.pages[self.current]}\n\n📄 {self.current + 1}/{len(self.pages)}"


async def send_with_buttons(
    channel, content: str, view: discord.ui.View,
) -> discord.Message | None:
    """Send a message with button view attached."""
    try:
        return await channel.send(content, view=view)
    except Exception as e:
        logger.warning("Failed to send buttons: %s", e)
        return await channel.send(content)
