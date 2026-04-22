"""Display configuration for gateway platforms.

Controls how messages are formatted for different platforms:
- Discord: Markdown, code blocks, embeds
- Telegram: HTML or Markdown, inline keyboards
- CLI: Rich panels, ANSI colors
- Slack: mrkdwn, blocks
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Platform",
    "DisplayConfig",
    "DISCORD_CONFIG",
    "TELEGRAM_CONFIG",
    "SLACK_CONFIG",
    "CLI_CONFIG",
    "WEB_CONFIG",
    "get_display_config",
    "split_message",
]



class Platform(str, Enum):
    """Supported gateway platforms with their display constraints."""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    CLI = "cli"
    WEB = "web"


@dataclass(frozen=True)
class DisplayConfig:
    """Platform-specific display settings."""
    platform: Platform
    max_message_length: int = 2000
    supports_markdown: bool = True
    supports_code_blocks: bool = True
    supports_embeds: bool = False
    supports_buttons: bool = False
    supports_reactions: bool = False
    supports_threads: bool = False
    supports_voice: bool = False
    code_block_syntax: str = "```"  # ``` for Discord/Telegram, ~~~ for some
    newline: str = "\n"
    truncation_suffix: str = "... (truncated)"


# Platform presets
DISCORD_CONFIG = DisplayConfig(
    platform=Platform.DISCORD,
    max_message_length=2000,
    supports_embeds=True,
    supports_buttons=True,
    supports_reactions=True,
    supports_threads=True,
    supports_voice=True,
)

TELEGRAM_CONFIG = DisplayConfig(
    platform=Platform.TELEGRAM,
    max_message_length=4096,
    supports_buttons=True,
    supports_reactions=True,
)

SLACK_CONFIG = DisplayConfig(
    platform=Platform.SLACK,
    max_message_length=40000,
    supports_embeds=True,
    supports_buttons=True,
    supports_reactions=True,
    supports_threads=True,
)

CLI_CONFIG = DisplayConfig(
    platform=Platform.CLI,
    max_message_length=0,  # No limit
    supports_code_blocks=True,
)

WEB_CONFIG = DisplayConfig(
    platform=Platform.WEB,
    max_message_length=0,
    supports_embeds=True,
    supports_buttons=True,
)

_CONFIGS = {
    Platform.DISCORD: DISCORD_CONFIG,
    Platform.TELEGRAM: TELEGRAM_CONFIG,
    Platform.SLACK: SLACK_CONFIG,
    Platform.CLI: CLI_CONFIG,
    Platform.WEB: WEB_CONFIG,
}


def get_display_config(platform: str | Platform) -> DisplayConfig:
    """Get display config for a platform."""
    if isinstance(platform, str):
        try:
            platform = Platform(platform.lower())
        except ValueError:
            return CLI_CONFIG
    return _CONFIGS.get(platform, CLI_CONFIG)


def split_message(text: str, max_length: int) -> list[str]:
    """Split a message into chunks respecting max_length.

    Delegates to message_splitting.split_message for platform-aware splitting.
    """
    if max_length <= 0 or len(text) <= max_length:
        return [text]
    from caveman.gateway.message_splitting import split_message as _split
    return _split(text, max_length=max_length)
