"""Info & diagnostics command handlers."""
from __future__ import annotations

from caveman.commands.types import CommandContext
from caveman.commands.handlers._helpers import (
    read_json, count_files, memory_count, memory_stats, memory_search_fts,
    load_config_safe, CAVEMAN_HOME,
)


async def handle_status(ctx: CommandContext) -> None:
    """Show session status panel."""
    from caveman.commands.formatter import format_panel
    shield = getattr(ctx.agent, "_shield", None)
    is_gateway = ctx.surface in ("discord", "telegram", "gateway")

    lines = [
        ctx.t(f"Model: {ctx.agent.model}", f"模型: {ctx.agent.model}"),
        ctx.t(f"Max iterations: {ctx.agent.max_iterations}", f"最大迭代: {ctx.agent.max_iterations}"),
    ]

    # Tools — show count, real if available
    try:
        tool_count = len(ctx.agent.tool_registry.get_schemas())
        lines.append(ctx.t(f"Tools: {tool_count}", f"工具: {tool_count}"))
    except Exception:
        lines.append(ctx.t("Tools: N/A", "工具: N/A"))

    # Memories
    try:
        mem_count = ctx.agent.memory_manager.total_count
        lines.append(ctx.t(f"Memories: {mem_count}", f"记忆: {mem_count}"))
    except Exception:
        lines.append(ctx.t("Memories: N/A", "记忆: N/A"))

    # Skills
    try:
        skill_count = len(ctx.agent.skill_manager.list_all())
        if skill_count:
            lines.append(ctx.t(f"Skills: {skill_count}", f"技能: {skill_count}"))
    except Exception:
        pass  # intentional: non-critical

    lines.append(ctx.t(f"Tool calls: {getattr(ctx.agent, '_tool_call_count', 0)}", f"工具调用: {getattr(ctx.agent, '_tool_call_count', 0)}"))

    if is_gateway:
        lines.append(ctx.t(f"Surface: {ctx.surface}", f"平台: {ctx.surface}"))

    # Engines
    try:
        cfg = getattr(ctx.agent, '_cfg', {})
        engines = cfg.get("engines", {})
        if engines:
            enabled = [k for k, v in engines.items()
                       if isinstance(v, dict) and v.get("enabled", True)]
            if enabled:
                lines.append(ctx.t(f"Engines: {', '.join(enabled)}", f"引擎: {', '.join(enabled)}"))
    except Exception:
        pass  # intentional: non-critical

    if shield:
        lines.append(ctx.t(f"Shield turns: {shield.essence.turn_count}", f"护盾轮次: {shield.essence.turn_count}"))
        task = shield.essence.task
        if task:
            lines.append(ctx.t(f"Shield task: {task}", f"护盾任务: {task}"))

    reflect = getattr(ctx.agent, "_reflect", None)
    if reflect and reflect.reflections:
        lines.append(ctx.t(f"Reflections: {len(reflect.reflections)}", f"反思: {len(reflect.reflections)}"))

    ctx.respond(format_panel(ctx.t("Session Status", "会话状态"), "\n".join(lines), ctx.surface))


async def handle_help(ctx: CommandContext) -> None:
    """Show help or single command details."""
    from caveman.commands.registry import resolve_command, get_by_category, ZH_COMMAND_NAMES
    from caveman.commands.formatter import format_help, format_panel

    if ctx.args:
        cmd = resolve_command(ctx.args.strip())
        if cmd:
            zh = ctx.locale.startswith("zh")
            display = ZH_COMMAND_NAMES.get(cmd.name, cmd.name) if zh else cmd.name
            lines = [f"/{display} {cmd.args_hint}".strip()]
            lines.append(cmd.desc(ctx.locale))
            if cmd.aliases:
                lines.append(ctx.t(f"Aliases: {', '.join('/' + a for a in cmd.aliases)}", f"别名: {', '.join('/' + a for a in cmd.aliases)}"))
            if cmd.subcommands:
                lines.append(ctx.t(f"Subcommands: {', '.join(cmd.subcommands)}", f"子命令: {', '.join(cmd.subcommands)}"))
            if cmd.examples:
                lines.append(ctx.t("Examples:", "示例:"))
                for ex in cmd.examples:
                    lines.append(f"  {ex}")
            if cmd.dangerous:
                lines.append(ctx.t("⚠ Dangerous — requires confirmation", "⚠ 危险操作 — 需要确认"))
            ctx.respond(format_panel(f"/{display}", "\n".join(lines), ctx.surface))
        else:
            ctx.respond(ctx.t(f"Unknown command: {ctx.args}. Try /commands", f"未知命令: {ctx.args}。试试 /命令"))
    else:
        cats = get_by_category()
        visible = {
            cat: [c for c in cmds if not c.hidden]
            for cat, cmds in cats.items()
        }
        ctx.respond(format_help(visible, ctx.surface, locale=ctx.locale))


async def handle_commands(ctx: CommandContext) -> None:
    """Browse all commands, paginated."""
    from caveman.commands.registry import COMMAND_REGISTRY, ZH_COMMAND_NAMES
    page = int(ctx.parts()[0]) if ctx.parts() else 1
    per_page = 15
    start = (page - 1) * per_page
    total_pages = (len(COMMAND_REGISTRY) + per_page - 1) // per_page
    chunk = COMMAND_REGISTRY[start:start + per_page]
    lines = []
    for c in chunk:
        if ctx.locale.startswith("zh") and c.name in ZH_COMMAND_NAMES:
            display_name = ZH_COMMAND_NAMES[c.name]
        else:
            display_name = c.name
        lines.append(f"  /{display_name:<16} {c.desc(ctx.locale)}")
    lines.append(ctx.t(f"\nPage {page}/{total_pages}. /commands [page]", f"\n第 {page}/{total_pages} 页。/命令 [页码]"))
    ctx.respond("\n".join(lines))


async def handle_usage(ctx: CommandContext) -> None:
    """Show token usage."""
    sub = ctx.subcommand()
    count = getattr(ctx.agent, "_tool_call_count", 0)
    ctx.respond(ctx.t(f"Tool calls this session: {count}", f"本次会话工具调用: {count}"))


async def handle_insights(ctx: CommandContext) -> None:
    """Usage analytics."""
    try:
        lines = [ctx.t("📊 Usage Insights", "📊 使用分析")]

        fw = read_json("flywheel_stats.json") or []
        if fw:
            total_fixed = sum(s.get("fixed", 0) for s in fw)
            total_time = sum(s.get("duration_s", 0) for s in fw)
            lines.append(ctx.t(f"  Flywheel: {len(fw)} rounds, {total_fixed} fixes, {total_time/60:.0f}min", f"  飞轮: {len(fw)} 轮, {total_fixed} 修复, {total_time/60:.0f}分钟"))

        from caveman.commands.handlers._helpers import list_files
        traj_files = list_files("trajectories")
        if traj_files:
            traj_count = len(traj_files)
            total_size = sum(f.stat().st_size for f in traj_files)
            lines.append(ctx.t(f"  Trajectories: {traj_count} ({total_size/1024/1024:.1f}MB)", f"  轨迹: {traj_count} ({total_size/1024/1024:.1f}MB)"))

        mem_count = memory_count()
        if mem_count:
            lines.append(ctx.t(f"  Memories: {mem_count}", f"  记忆: {mem_count}"))

        sess_count = count_files("gateway_sessions", "*")
        if sess_count:
            lines.append(ctx.t(f"  Gateway sessions: {sess_count}", f"  网关会话: {sess_count}"))

        skill_count = count_files("skills", "*.yaml")
        if skill_count:
            lines.append(ctx.t(f"  Skills: {skill_count}", f"  技能: {skill_count}"))

        ctx.respond("\n".join(lines) if len(lines) > 1 else ctx.t("No usage data yet.", "暂无使用数据。"))
    except Exception as e:
        ctx.respond(ctx.t(f"Insights error: {e}", f"分析错误: {e}"))


async def handle_doctor(ctx: CommandContext) -> None:
    """Run health check."""
    try:
        from caveman.cli.doctor import run_doctor
        report = await run_doctor()
        ctx.respond(report.to_text())
    except Exception as e:
        ctx.respond(ctx.t(f"Doctor failed: {e}", f"诊断失败: {e}"))


async def handle_memory(ctx: CommandContext) -> None:
    """Memory management: stats, search, forget, recent, top."""
    from caveman.commands.handlers._helpers import (
        memory_stats, memory_search_fts, memory_recent,
        memory_top_retrieved, memory_high_trust,
    )
    sub = ctx.subcommand()

    if sub == "stats":
        total, cats = memory_stats()
        lines = [ctx.t(f"🧠 Memories: {total}", f"🧠 记忆: {total}")]
        if cats:
            type_labels = {"episodic": "经历", "semantic": "知识", "procedural": "技能", "working": "工作"}
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                label = type_labels.get(cat, cat) if ctx.locale.startswith("zh") else cat
                lines.append(f"  {label}: {count}")
        emb_file = CAVEMAN_HOME / "memory" / "_embeddings.json"
        if emb_file.exists():
            size_mb = emb_file.stat().st_size / 1024 / 1024
            lines.append(ctx.t(f"  Embeddings: {size_mb:.1f}MB", f"  向量: {size_mb:.1f}MB"))
        ctx.respond("\n".join(lines))

    elif sub == "search" and ctx.rest():
        query = ctx.rest()
        results = memory_search_fts(query)
        if results:
            lines = [ctx.t(f"🔍 '{query}' ({len(results)} results):", f"🔍 '{query}' ({len(results)} 条):")]
            for r in results:
                lines.append(f"  • {r[:120]}")
            ctx.respond("\n".join(lines))
        else:
            ctx.respond(ctx.t(f"No memories matching '{query}'.", f"未找到 '{query}'。"))

    elif sub == "recent":
        items = memory_recent(8)
        if items:
            type_emoji = {"episodic": "📖", "semantic": "💡", "procedural": "⚙️", "working": "📌"}
            lines = [ctx.t("🕐 Recent memories:", "🕐 最近记忆:")]
            for content, mtype, trust in items:
                emoji = type_emoji.get(mtype, "📝")
                trust_bar = "★" if trust >= 0.7 else "☆" if trust >= 0.4 else "○"
                lines.append(f"  {emoji}{trust_bar} {content[:100]}")
            ctx.respond("\n".join(lines))
        else:
            ctx.respond(ctx.t("No memories yet.", "还没有记忆。"))

    elif sub == "top":
        items = memory_top_retrieved(8)
        if items:
            lines = [ctx.t("🏆 Most used memories:", "🏆 最常用记忆:")]
            for content, count, trust in items:
                trust_bar = "★" if trust >= 0.7 else "☆" if trust >= 0.4 else "○"
                lines.append(f"  {trust_bar}[{count}x] {content[:100]}")
            ctx.respond("\n".join(lines))
        else:
            ctx.respond(ctx.t("No retrieved memories yet.", "还没有被调用过的记忆。"))

    elif sub == "trusted":
        items = memory_high_trust(8)
        if items:
            lines = [ctx.t("⭐ High-trust memories:", "⭐ 高信任记忆:")]
            for content, trust, created in items:
                date = created[:10] if created else ""
                lines.append(f"  ★{trust:.1f} [{date}] {content[:100]}")
            ctx.respond("\n".join(lines))
        else:
            ctx.respond(ctx.t("No high-trust memories yet.", "还没有高信任记忆。"))

    elif sub == "forget" and ctx.rest():
        ctx.respond(ctx.t(
            f"⚠️ Memory forget requires CLI. Run `caveman memory forget \"{ctx.rest()}\"`",
            f"⚠️ 删除记忆需要 CLI。运行 `caveman memory forget \"{ctx.rest()}\"`"))

    else:
        # Default: show a useful overview
        total, cats = memory_stats()
        type_labels = {"episodic": "经历", "semantic": "知识", "procedural": "技能", "working": "工作"}
        type_emoji = {"episodic": "📖", "semantic": "💡", "procedural": "⚙️", "working": "📌"}

        lines = [ctx.t(f"🧠 Memory Overview ({total} total)", f"🧠 记忆概览（共 {total} 条）")]

        # Category breakdown
        if cats:
            cat_parts = []
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                emoji = type_emoji.get(cat, "📝")
                label = type_labels.get(cat, cat) if ctx.locale.startswith("zh") else cat
                cat_parts.append(f"{emoji}{label} {count}")
            lines.append("  " + " | ".join(cat_parts))

        # Recent high-value
        trusted = memory_high_trust(3)
        if trusted:
            lines.append(ctx.t("\n⭐ Key memories:", "\n⭐ 核心记忆:"))
            for content, trust, _ in trusted:
                lines.append(f"  ★ {content[:100]}")

        # Most used
        top = memory_top_retrieved(3)
        if top:
            lines.append(ctx.t("\n🏆 Most used:", "\n🏆 最常用:"))
            for content, count, _ in top:
                lines.append(f"  [{count}x] {content[:80]}")

        lines.append(ctx.t(
            "\n/memory [stats|search|recent|top|trusted|forget]",
            "\n/记忆 [统计|搜索|最近|常用|信任|删除]"))
        ctx.respond("\n".join(lines))


async def handle_recall(ctx: CommandContext) -> None:
    """Restore previous context."""
    try:
        recall_engine = getattr(ctx.agent, "_recall", None)
        if recall_engine:
            text = await recall_engine.restore(ctx.args or "")
            if text:
                ctx.respond(text[:1500])  # recall content is language-neutral
            else:
                ctx.respond(ctx.t("No previous sessions to recall.", "无可恢复的历史会话。"))
        else:
            ctx.respond(ctx.t("Recall engine not available.", "回忆引擎不可用。"))
    except Exception as e:
        ctx.respond(ctx.t(f"Recall failed: {e}", f"回忆失败: {e}"))


async def handle_shield(ctx: CommandContext) -> None:
    """Show Shield essence state."""
    shield = getattr(ctx.agent, "_shield", None)
    if not shield:
        ctx.respond(ctx.t("Shield not available.", "护盾不可用。"))
        return
    e = shield.essence
    lines = [
        ctx.t(f"Session: {e.session_id}", f"会话: {e.session_id}"),
        ctx.t(f"Task: {e.task or '(none)'}", f"任务: {e.task or '(无)'}"),
        ctx.t(f"Turns: {e.turn_count}", f"轮次: {e.turn_count}"),
        ctx.t(f"Decisions: {len(e.decisions)}", f"决策: {len(e.decisions)}"),
        ctx.t(f"Progress: {len(e.progress)}", f"进度: {len(e.progress)}"),
        ctx.t(f"TODOs: {len(e.open_todos)}", f"待办: {len(e.open_todos)}"),
        ctx.t(f"Key data: {len(e.key_data)} fields", f"关键数据: {len(e.key_data)} 项"),
    ]
    from caveman.commands.formatter import format_panel
    ctx.respond(format_panel(ctx.t("Shield Essence", "护盾状态"), "\n".join(lines), ctx.surface))


async def handle_reflect(ctx: CommandContext) -> None:
    """Show post-task reflections."""
    reflect = getattr(ctx.agent, "_reflect", None)
    if not reflect or not reflect.reflections:
        ctx.respond(ctx.t("No reflections yet.", "暂无反思记录。"))
        return
    lines = []
    for r in reflect.reflections[-3:]:
        lines.append(ctx.t(f"Task: {r.task[:80]}", f"任务: {r.task[:80]}"))
        if r.effective_patterns:
            lines.append(ctx.t("  Patterns: ", "  模式: ") + "; ".join(r.effective_patterns[:3]))
        if r.lessons:
            lines.append(ctx.t("  Lessons: ", "  教训: ") + "; ".join(r.lessons[:3]))
    ctx.respond("\n".join(lines))


async def handle_audit(ctx: CommandContext) -> None:
    """Export audit log."""
    try:
        from caveman.audit import export_audit_log
        path = ctx.args or None
        out = await export_audit_log(ctx.agent.bus, path)
        ctx.respond(ctx.t(f"Audit log exported to {out}", f"审计日志已导出到 {out}"))
    except Exception as e:
        ctx.respond(ctx.t(f"Audit export failed: {e}", f"审计导出失败: {e}"))


async def handle_ratelimit(ctx: CommandContext) -> None:
    """Show rate limit status."""
    try:
        from caveman.providers.rate_limit import format_rate_limits
        state = getattr(ctx.agent.provider, "rate_limit_state", None)
        if state and state.has_data:
            ctx.respond(format_rate_limits(state))
        else:
            ctx.respond(ctx.t("No rate limit data yet.", "暂无限速数据。"))
    except Exception as e:
        ctx.respond(ctx.t(f"Rate limit display failed: {e}", f"限速显示失败: {e}"))
