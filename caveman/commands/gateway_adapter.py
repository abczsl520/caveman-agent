"""Platform adapters for Discord and Telegram command registration."""
from __future__ import annotations

from caveman.commands.registry import COMMAND_REGISTRY


def discord_slash_commands() -> list[dict]:
    """Generate Discord slash command registration payload.

    Discord requires: 1-32 chars, lowercase, no spaces. Hyphens and underscores allowed.
    """
    commands = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only or cmd.hidden:
            continue
        name = cmd.name.lower()[:32]
        entry: dict = {
            "name": name,
            "description": cmd.description[:100],
            "type": 1,  # CHAT_INPUT
        }
        if cmd.subcommands:
            entry["options"] = [
                {
                    "name": sub.lower()[:32],
                    "description": f"{cmd.name} {sub}"[:100],
                    "type": 1,  # SUB_COMMAND
                }
                for sub in cmd.subcommands
            ]
        commands.append(entry)
    return commands


def telegram_bot_commands() -> list[dict]:
    """Generate Telegram BotCommand menu entries.

    Telegram requires: 1-32 chars, lowercase a-z, digits 0-9, underscores only.
    Hyphens are converted to underscores.
    """
    commands = []
    for cmd in COMMAND_REGISTRY:
        if cmd.cli_only or cmd.hidden:
            continue
        # Telegram: no hyphens, lowercase only
        tg_name = cmd.name.lower().replace("-", "_")
        if len(tg_name) > 32:
            tg_name = tg_name[:32]
        commands.append({
            "command": tg_name,
            "description": cmd.description[:256],
        })
    return commands
