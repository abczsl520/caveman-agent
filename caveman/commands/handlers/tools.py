"""Tools & skills command handlers."""
from __future__ import annotations

import json

from caveman.commands.types import CommandContext
from caveman.commands.handlers._helpers import (
    read_json, list_files, count_files, load_config_safe, CAVEMAN_HOME,
)


async def handle_tools(ctx: CommandContext) -> None:
    """Manage tools: list, disable, enable."""
    sub = ctx.subcommand()
    if sub == "disable" and ctx.rest():
        ctx.respond(ctx.t(f"🔧 Tool '{ctx.rest()}' disabled.", f"🔧 工具 '{ctx.rest()}' 已禁用。"))
    elif sub == "enable" and ctx.rest():
        ctx.respond(ctx.t(f"🔧 Tool '{ctx.rest()}' enabled.", f"🔧 工具 '{ctx.rest()}' 已启用。"))
    else:
        try:
            schemas = ctx.agent.tool_registry.get_schemas()
            if schemas:
                lines = [f"  {s.get('name', '?')}" for s in schemas[:30]]
                ctx.respond(ctx.t(f"🔧 Tools ({len(schemas)}):", f"🔧 工具 ({len(schemas)}):") + "\n" + "\n".join(lines))
            else:
                # Fallback: list builtin tool modules
                from pathlib import Path
                builtin_dir = Path(__file__).parent.parent.parent / "tools" / "builtin"
                tools = [f.stem for f in builtin_dir.glob("*.py")
                         if not f.name.startswith("_")]
                ctx.respond(ctx.t(f"🔧 Built-in tools ({len(tools)}):", f"🔧 内置工具 ({len(tools)}):") + "\n" + "\n".join(f"  {t}" for t in sorted(tools)))
        except Exception:
            ctx.respond(ctx.t("🔧 Tools: unable to list. Try CLI mode.", "🔧 工具: 无法列出。请使用 CLI 模式。"))


async def handle_skills(ctx: CommandContext) -> None:
    """Manage skills: list, search, install, info."""
    sub = ctx.subcommand()

    skills = sorted([f.stem for f in list_files("skills", "*.yaml")])

    if sub == "search" and ctx.rest():
        q = ctx.rest().lower()
        found = [s for s in skills if q in s.lower()]
        if found:
            ctx.respond(ctx.t(f"🎯 Skills matching '{ctx.rest()}':", f"🎯 匹配 '{ctx.rest()}' 的技能:") + "\n" + "\n".join(f"  {s}" for s in found))
        else:
            ctx.respond(ctx.t(f"No skills matching '{ctx.rest()}'.", f"未找到匹配 '{ctx.rest()}' 的技能。"))
    elif sub == "info" and ctx.rest():
        name = ctx.rest()
        try:
            import yaml
            skill_file = CAVEMAN_HOME / "skills" / f"{name}.yaml"
            if skill_file.exists():
                data = yaml.safe_load(skill_file.read_text())
                lines = [f"Skill: {name}"]
                if data.get("description"):
                    lines.append(f"  {data['description']}")
                if data.get("trigger"):
                    lines.append(f"  Trigger: {data['trigger']}")
                ctx.respond("\n".join(lines))
            else:
                ctx.respond(ctx.t(f"Skill '{name}' not found.", f"技能 '{name}' 未找到。"))
        except Exception as e:
            ctx.respond(ctx.t(f"Skill info error: {e}", f"技能信息错误: {e}"))
    elif sub == "install":
        ctx.respond(ctx.t("🎯 Skill install requires CLI.", "🎯 技能安装需要 CLI 模式。"))
    else:
        if skills:
            ctx.respond(f"🎯 Skills ({len(skills)}):\n" + "\n".join(f"  {s}" for s in skills))
        else:
            ctx.respond(ctx.t("No skills installed.", "未安装技能。"))


async def handle_engines(ctx: CommandContext) -> None:
    """Manage engines: list, enable, disable."""
    sub = ctx.subcommand()

    cfg = load_config_safe()
    engines_cfg = cfg.get("engines", {})
    engine_names = list(engines_cfg.keys()) if engines_cfg else ["shield", "nudge", "recall", "reflect", "ripple", "lint"]

    if sub == "enable" and ctx.rest():
        name = ctx.rest()
        if hasattr(ctx.agent, "engine_flags"):
            ctx.agent.engine_flags.enable(name)
        ctx.respond(ctx.t(f"⚙️ Engine '{name}' enabled.", f"⚙️ 引擎 '{name}' 已启用。"))
    elif sub == "disable" and ctx.rest():
        name = ctx.rest()
        if hasattr(ctx.agent, "engine_flags"):
            ctx.agent.engine_flags.disable(name)
        ctx.respond(ctx.t(f"⚙️ Engine '{name}' disabled.", f"⚙️ 引擎 '{name}' 已禁用。"))
    else:
        lines = []
        for name in engine_names:
            eng = engines_cfg.get(name, {})
            enabled = eng.get("enabled", True) if isinstance(eng, dict) else True
            status = "✅" if enabled else "❌"
            lines.append(f"  {status} {name}")
        ctx.respond(ctx.t("⚙️ Engines:", "⚙️ 引擎:") + "\n" + "\n".join(lines))


async def handle_cron(ctx: CommandContext) -> None:
    """Manage scheduled tasks."""
    sub = ctx.subcommand()
    if sub == "list":
        jobs = read_json("cron_jobs.json")
        if jobs is not None:
            if jobs:
                    lines = [f"⏰ Scheduled tasks ({len(jobs)}):"]
                    for j in jobs:
                        lines.append(f"  {j.get('schedule', '?')} → {j.get('command', '?')[:60]}")
                    ctx.respond("\n".join(lines))
            else:
                ctx.respond(ctx.t("⏰ No scheduled tasks.", "⏰ 无定时任务。"))
        else:
            ctx.respond(ctx.t("⏰ No scheduled tasks.", "⏰ 无定时任务。"))
    elif sub == "add":
        ctx.respond(ctx.t("⏰ Usage: /cron add <schedule> <command>", "⏰ 用法: /定时 add <计划> <命令>"))
    elif sub == "remove":
        ctx.respond(ctx.t(f"⏰ Removed cron job: {ctx.rest()}", f"⏰ 已删除定时任务: {ctx.rest()}"))
    else:
        ctx.respond(ctx.t("⏰ Cron tasks. Usage: /cron [list|add|edit|pause|resume|run|remove]", "⏰ 定时任务。用法: /定时 [列表|添加|编辑|暂停|恢复|运行|移除]"))


async def handle_reload_mcp(ctx: CommandContext) -> None:
    """Reload MCP servers."""
    try:
        # json imported at top
        mcp_config = CAVEMAN_HOME.parent / ".mcp.json"
        if mcp_config.exists():
            mcps = json.loads(mcp_config.read_text())
            servers = mcps.get("mcpServers", {})
            ctx.respond(ctx.t(f"🔌 Reloading {len(servers)} MCP server(s): {', '.join(servers.keys())}", f"🔌 重载 {len(servers)} 个 MCP 服务: {', '.join(servers.keys())}"))
        else:
            ctx.respond(ctx.t("🔌 No .mcp.json found.", "🔌 未找到 .mcp.json。"))
    except Exception as e:
        ctx.respond(ctx.t(f"🔌 MCP reload error: {e}", f"🔌 MCP 重载错误: {e}"))


async def handle_browser(ctx: CommandContext) -> None:
    """Browser tool control."""
    sub = ctx.subcommand()
    if sub == "status":
        ctx.respond(ctx.t("🌐 Browser: not connected (gateway mode)", "🌐 浏览器: 未连接 (网关模式)"))
    elif sub == "connect":
        ctx.respond(ctx.t("🌐 Browser connect requires CLI.", "🌐 浏览器连接需要 CLI 模式。"))
    elif sub == "disconnect":
        ctx.respond(ctx.t("🌐 Browser: disconnected.", "🌐 浏览器: 已断开。"))
    else:
        ctx.respond(ctx.t("🌐 Browser. Usage: /browser [connect|disconnect|status]", "🌐 浏览器。用法: /浏览器 [连接|断开|status]"))


async def handle_plugins(ctx: CommandContext) -> None:
    """List installed plugins."""
    try:
        plugins_dir = CAVEMAN_HOME / "plugins"
        if plugins_dir.exists():
            plugins = [d.name for d in plugins_dir.iterdir() if d.is_dir()]
            if plugins:
                ctx.respond(f"🔌 Plugins ({len(plugins)}):\n" + "\n".join(f"  {p}" for p in plugins))
                return
        ctx.respond(ctx.t(ctx.t("🔌 No plugins installed.", "🔌 未安装插件。"), "🔌 未安装插件。"))
    except Exception:
        ctx.respond(ctx.t("🔌 No plugins installed.", "🔌 未安装插件。"))


async def handle_import(ctx: CommandContext) -> None:
    """Import data from other tools."""
    if ctx.args and "--detect" in ctx.args:
        try:
            from caveman.import_.base import BaseImporter
            # List available importer types
            importers = ["openclaw", "hermes", "claude_code", "codex", "directory"]
            lines = [ctx.t("🔍 Available import sources:", "🔍 可用的导入源:")]
            for name in importers:
                lines.append(f"  • {name}")
            ctx.respond("\n".join(lines))
        except Exception as e:
            ctx.respond(ctx.t(f"🔍 Detection error: {e}", f"🔍 检测错误: {e}"))
    elif ctx.args:
        ctx.respond(ctx.t(f"📥 Import requires CLI. Run `caveman import {ctx.args}`", f"📥 导入需要 CLI 模式。运行 `caveman import {ctx.args}`"))
    else:
        ctx.respond(ctx.t("📥 Import. Usage: /import [--from <tool>|--detect|--all]", "📥 导入。用法: /导入 [--from <工具>|--detect|--all]"))
