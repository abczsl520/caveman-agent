"""System command handlers."""
from __future__ import annotations

from caveman.commands.types import CommandContext
from caveman.commands.handlers._helpers import load_config_safe, CAVEMAN_HOME


async def handle_restart(ctx: CommandContext) -> None:
    """Restart the gateway."""
    ctx.respond(ctx.t("🔄 Restarting gateway...", "🔄 正在重启网关…"))


async def handle_update(ctx: CommandContext) -> None:
    """Update Caveman to latest version."""
    try:
        import caveman
        current = caveman.__version__
        ctx.respond(ctx.t(
            f"Current version: v{current}\nRun `pip install -U caveman-agent` to update.",
            f"当前版本: v{current}\n运行 `pip install -U caveman-agent` 更新。",
        ))
    except Exception:
        ctx.respond(ctx.t("Run `pip install -U caveman-agent` to update.", "运行 `pip install -U caveman-agent` 更新。"))


async def handle_profile(ctx: CommandContext) -> None:
    """Show current profile."""
    try:
        import caveman
        cfg = load_config_safe()
        model = cfg.get("agent", {}).get("default_model", "unknown")
        engines = cfg.get("engines", {})
        enabled_engines = [k for k, v in engines.items()
                          if isinstance(v, dict) and v.get("enabled", True)]
        locale = cfg.get("locale", "en")
        lines = [
            ctx.t(f"Version: v{caveman.__version__}", f"版本: v{caveman.__version__}"),
            ctx.t(f"Home: {CAVEMAN_HOME}", f"主目录: {CAVEMAN_HOME}"),
            ctx.t(f"Model: {model}", f"模型: {model}"),
            ctx.t(f"Locale: {locale}", f"语言: {locale}"),
            ctx.t(f"Engines: {', '.join(enabled_engines) if enabled_engines else 'none'}", f"引擎: {', '.join(enabled_engines) if enabled_engines else '无'}"),
        ]
        gw = cfg.get("gateway", {})
        active_gw = [k for k, v in gw.items() if isinstance(v, dict) and v.get("enabled")]
        if active_gw:
            lines.append(ctx.t(f"Gateways: {', '.join(active_gw)}", f"网关: {', '.join(active_gw)}"))
        ctx.respond("\n".join(lines))
    except Exception:
        ctx.respond(ctx.t("Profile: default", "档案: 默认"))


async def handle_sethome(ctx: CommandContext) -> None:
    """Set current channel as home (gateway only)."""
    ctx.respond(ctx.t("✅ Home channel set.", "✅ 主频道已设置。"))


async def handle_quit(ctx: CommandContext) -> None:
    """Exit Caveman (CLI only)."""
    ctx.respond(ctx.t("👋 Goodbye!", "👋 再见！"))
