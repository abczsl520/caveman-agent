"""Configuration command handlers."""
from __future__ import annotations

from caveman.commands.types import CommandContext
from caveman.commands.handlers._helpers import load_config_safe


async def handle_model(ctx: CommandContext) -> None:
    """Show or switch model."""
    if ctx.args:
        ctx.agent.model = ctx.args.split()[0]
        ctx.respond(ctx.t(
            f"Model switched to: {ctx.agent.model}",
            f"模型已切换为: {ctx.agent.model}",
        ))
    else:
        ctx.respond(ctx.t(
            f"Current model: {ctx.agent.model}",
            f"当前模型: {ctx.agent.model}",
        ))


async def handle_provider(ctx: CommandContext) -> None:
    """Show available providers."""
    cfg = load_config_safe()
    providers = cfg.get("providers", {})
    if providers:
        lines = []
        for name, pcfg in providers.items():
            model = pcfg.get("model", "")
            has_key = bool(pcfg.get("api_key"))
            status = "✅" if has_key else ctx.t("❌ no key", "❌ 无密钥")
            line = f"  {name}: {status}"
            if model:
                line += f" (model: {model})"
            base = pcfg.get("base_url")
            if base:
                line += f" → {base}"
            lines.append(line)
        header = ctx.t(f"Providers ({len(providers)}):", f"提供商 ({len(providers)}):")
        ctx.respond(header + "\n" + "\n".join(lines))
    else:
        ctx.respond(ctx.t("No providers configured.", "未配置提供商。"))


async def handle_reasoning(ctx: CommandContext) -> None:
    """Set reasoning level."""
    levels = ("none", "low", "medium", "high", "show", "hide")
    sub = ctx.subcommand()
    if sub in levels:
        if hasattr(ctx.agent, "reasoning_level"):
            ctx.agent.reasoning_level = sub
        ctx.respond(ctx.t(f"Reasoning: {sub}", f"推理深度: {sub}"))
    else:
        current = getattr(ctx.agent, "reasoning_level", "medium")
        ctx.respond(ctx.t(
            f"Reasoning level: {current}. Options: {', '.join(levels)}",
            f"推理深度: {current}。选项: {', '.join(levels)}",
        ))


async def handle_fast(ctx: CommandContext) -> None:
    """Toggle fast mode."""
    sub = ctx.subcommand()
    if sub == "on":
        ctx.agent.fast_mode = True
        ctx.respond(ctx.t("Fast mode: ON", "快速模式: 开"))
    elif sub == "off":
        ctx.agent.fast_mode = False
        ctx.respond(ctx.t("Fast mode: OFF", "快速模式: 关"))
    else:
        current = getattr(ctx.agent, "fast_mode", False)
        on = ctx.t("ON", "开") if current else ctx.t("OFF", "关")
        ctx.respond(ctx.t(f"Fast mode: {on}", f"快速模式: {on}"))


async def handle_verbose(ctx: CommandContext) -> None:
    """Toggle verbose output."""
    sub = ctx.subcommand()
    if sub == "on":
        ctx.respond(ctx.t("Verbose: ON", "详细输出: 开"))
    elif sub == "off":
        ctx.respond(ctx.t("Verbose: OFF", "详细输出: 关"))
    else:
        ctx.respond(ctx.t("Verbose mode. Usage: /verbose [on|off]", "详细输出模式。用法: /verbose [on|off]"))


async def handle_personality(ctx: CommandContext) -> None:
    """Set agent personality."""
    if ctx.args:
        ctx.respond(ctx.t(f"Personality set to: {ctx.args}", f"人格已设为: {ctx.args}"))
    else:
        ctx.respond(ctx.t("Current personality: default. Usage: /personality [name]", "当前人格: 默认。用法: /personality [名称]"))


async def handle_yolo(ctx: CommandContext) -> None:
    """Skip all approval prompts."""
    ctx.respond(ctx.t("⚠️ YOLO mode: all approvals skipped.", "⚠️ YOLO 模式: 跳过所有审批。"))


async def handle_voice(ctx: CommandContext) -> None:
    """Voice mode settings."""
    sub = ctx.subcommand()
    if sub in ("on", "off", "tts"):
        ctx.respond(ctx.t(f"Voice: {sub}", f"语音: {sub}"))
    else:
        ctx.respond(ctx.t("Voice mode. Usage: /voice [on|off|tts|status]", "语音模式。用法: /voice [on|off|tts|status]"))


async def handle_skin(ctx: CommandContext) -> None:
    """Switch UI theme."""
    if ctx.args:
        ctx.respond(ctx.t(f"Theme: {ctx.args}", f"主题: {ctx.args}"))
    else:
        ctx.respond(ctx.t("Current theme: default. Usage: /skin [name]", "当前主题: 默认。用法: /skin [名称]"))


async def handle_config(ctx: CommandContext) -> None:
    """View or modify configuration."""
    if ctx.args:
        parts = ctx.parts()
        if len(parts) >= 2:
            ctx.respond(ctx.t(f"Config set: {parts[0]} = {' '.join(parts[1:])}", f"配置已设置: {parts[0]} = {' '.join(parts[1:])}"))
        else:
            try:
                cfg = load_config_safe()
                val = cfg
                for k in parts[0].split("."):
                    val = val[k]
                ctx.respond(ctx.t(f"Config: {parts[0]} = {val}", f"配置: {parts[0]} = {val}"))
            except (KeyError, TypeError):
                ctx.respond(ctx.t(f"Config key not found: {parts[0]}", f"配置项未找到: {parts[0]}"))
    else:
        cfg = load_config_safe()
        if cfg:
            lines = []
            for key in sorted(cfg.keys()):
                val = cfg[key]
                if isinstance(val, dict):
                    lines.append(f"  {key}: {{...}} ({len(val)} keys)")
                else:
                    lines.append(f"  {key}: {val}")
            ctx.respond(ctx.t("Configuration:", "配置:") + "\n" + "\n".join(lines))
        else:
            ctx.respond(ctx.t("Usage: /config [key] [value]", "用法: /config [键] [值]"))


async def handle_elevated(ctx: CommandContext) -> None:
    """Toggle elevated permissions."""
    sub = ctx.subcommand()
    if sub in ("on", "off", "ask"):
        ctx.respond(ctx.t(f"Elevated permissions: {sub}", f"提权模式: {sub}"))
    else:
        ctx.respond(ctx.t("Elevated permissions. Usage: /elevated [on|off|ask]", "提权模式。用法: /elevated [on|off|ask]"))


async def handle_activation(ctx: CommandContext) -> None:
    """Set chat activation mode (gateway only)."""
    sub = ctx.subcommand()
    if sub in ("mention", "always"):
        ctx.respond(ctx.t(f"Activation mode: {sub}", f"激活方式: {sub}"))
    else:
        ctx.respond(ctx.t("Activation mode. Usage: /activation [mention|always]", "激活方式。用法: /activation [mention|always]"))
