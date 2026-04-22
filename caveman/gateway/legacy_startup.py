"""Legacy gateway startup — backward-compatible Gateway ABC path.

Used when use_platform_adapters=False or as fallback when no new adapters match.
Will be removed once all platforms are migrated to BasePlatformAdapter.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("caveman.gateway")


async def start_legacy_gateways(
    gw_config: dict,
    config: dict,
    handle_task: Callable,
    router: Any,
) -> list:
    """Start gateways using the legacy Gateway ABC system."""
    gateways = []

    discord_cfg = gw_config.get("discord", {})
    if discord_cfg.get("enabled") and discord_cfg.get("token"):
        from caveman.gateway.discord_gw import DiscordGateway
        dg = DiscordGateway(
            token=discord_cfg["token"],
            prefix=discord_cfg.get("prefix", "!cave"),
            trigger=discord_cfg.get("trigger", "all"),
            allowed_channels=discord_cfg.get("allowed_channels"),
            allowed_users=discord_cfg.get("allowed_users"),
            locale=config.get("locale", "en"),
        )
        dg.on_task(handle_task)
        router.register(dg)
        gateways.append(("Discord", dg))

    telegram_cfg = gw_config.get("telegram", {})
    if telegram_cfg.get("enabled") and telegram_cfg.get("token"):
        from caveman.gateway.telegram_gw import TelegramGateway
        tg = TelegramGateway(
            token=telegram_cfg["token"],
            allowed_users=telegram_cfg.get("allowed_users"),
        )
        tg.on_task(handle_task)
        router.register(tg)
        gateways.append(("Telegram", tg))

    slack_cfg = gw_config.get("slack", {})
    if slack_cfg.get("enabled") and slack_cfg.get("bot_token"):
        from caveman.gateway.slack_gw import SlackGateway
        sg = SlackGateway(
            bot_token=slack_cfg["bot_token"],
            app_token=slack_cfg.get("app_token", ""),
            allowed_channels=slack_cfg.get("allowed_channels"),
        )
        sg.on_task(handle_task)
        router.register(sg)
        gateways.append(("Slack", sg))

    return gateways
