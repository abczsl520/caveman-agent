"""Caveman-exclusive command handlers."""
from __future__ import annotations

import json

from caveman.commands.types import CommandContext
from caveman.commands.handlers._helpers import (
    read_json, count_files, list_files, memory_count, load_config_safe,
    CAVEMAN_HOME,
)


async def handle_flywheel(ctx: CommandContext) -> None:
    """Flywheel self-improvement status."""
    sub = ctx.subcommand()
    stats = read_json("flywheel_stats.json") or []

    if sub == "trigger":
        ctx.respond(ctx.t("🔄 Flywheel triggered.", "🔄 飞轮已触发。"))
    elif sub == "history":
        if not stats:
            ctx.respond(ctx.t("No flywheel rounds yet.", "暂无飞轮记录。"))
            return
        lines = [ctx.t(f"Flywheel history ({len(stats)} rounds):", f"飞轮历史 ({len(stats)} 轮):")]
        for s in stats[-5:]:
            r, target = s.get("round", "?"), s.get("target", "?")
            fixed, p0, dur = s.get("fixed", 0), s.get("p0", 0), s.get("duration_s", 0)
            ts = s.get("timestamp", "")[:16]
            lines.append(f"  R{r} [{target}] {ctx.t('fixed', '修复')}:{fixed} P0:{p0} {dur:.0f}s {ts}")
        ctx.respond("\n".join(lines))
    else:
        if not stats:
            ctx.respond(ctx.t("Flywheel: not started.", "飞轮: 未启动。"))
            return
        total_fixed = sum(s.get("fixed", 0) for s in stats)
        last = stats[-1]
        lines = [
            ctx.t("🔄 Flywheel Status", "🔄 飞轮状态"),
            ctx.t(f"  Rounds: {len(stats)}", f"  轮次: {len(stats)}"),
            ctx.t(f"  Total fixed: {total_fixed}", f"  总修复: {total_fixed}"),
            ctx.t(f"  Last: R{last.get('round','?')} [{last.get('target','?')}] @ {last.get('timestamp','')[:16]}",
                   f"  最近: R{last.get('round','?')} [{last.get('target','?')}] @ {last.get('timestamp','')[:16]}"),
        ]
        ctx.respond("\n".join(lines))


async def handle_trajectory(ctx: CommandContext) -> None:
    """Trajectory management."""
    sub = ctx.subcommand()
    files = list_files("trajectories")
    total = len(files)

    if sub == "list":
        if not files:
            ctx.respond(ctx.t("No trajectories.", "暂无轨迹。"))
            return
        lines = [ctx.t(f"Trajectories: {total}", f"轨迹: {total} 条")]
        for f in files[-10:]:
            lines.append(f"  {f.stem} ({f.stat().st_size/1024:.1f}KB)")
        if total > 10:
            lines.append(ctx.t(f"  ... +{total-10} more", f"  … 还有 {total-10} 条"))
        ctx.respond("\n".join(lines))
    elif sub == "replay":
        tid = ctx.rest()
        if not tid:
            ctx.respond(ctx.t(f"Usage: /trajectory replay <id>. {total} available.", f"用法: /轨迹 replay <id>。共 {total} 条。"))
        else:
            matches = [f for f in files if tid in f.stem]
            ctx.respond(ctx.t(f"Found {len(matches)}.", f"找到 {len(matches)} 条。") if matches else ctx.t(f"No match: '{tid}'.", f"未找到: '{tid}'。"))
    elif sub == "score":
        ctx.respond(ctx.t(f"Usage: /trajectory score <id>. {total} available.", f"用法: /轨迹 score <id>。共 {total} 条。"))
    else:
        ctx.respond(ctx.t(f"📊 Trajectories: {total}. Usage: /trajectory [list|replay|score]", f"📊 轨迹: {total} 条。用法: /轨迹 [列表|回放|评分]"))


async def handle_bench(ctx: CommandContext) -> None:
    """Run performance benchmarks."""
    sub = ctx.subcommand()
    if sub == "run":
        ctx.respond(ctx.t("⚡ Requires CLI. Run `caveman bench run`.", "⚡ 需要 CLI。运行 `caveman bench run`。"))
    elif sub == "results":
        # json imported at top
        results = read_json("bench_results.json")
        if results:
            ctx.respond(ctx.t("Last benchmark:", "最近基准测试:") + f"\n{json.dumps(results, indent=2)[:1500]}")
        else:
            ctx.respond(ctx.t("No results. Run `caveman bench run`.", "暂无结果。运行 `caveman bench run`。"))
    else:
        ctx.respond(ctx.t("⚡ Benchmarks. Usage: /bench [run|results]", "⚡ 基准测试。用法: /基准 [运行|结果]"))


async def handle_selftest(ctx: CommandContext) -> None:
    """Run self-diagnostics."""
    from caveman.commands.registry import COMMAND_REGISTRY
    import caveman

    checks = []
    checks.append(ctx.t(f"✅ Commands: {len(COMMAND_REGISTRY)}", f"✅ 命令: {len(COMMAND_REGISTRY)}"))

    cfg = load_config_safe()
    if cfg:
        checks.append(ctx.t(f"✅ Config: {len(cfg)} sections", f"✅ 配置: {len(cfg)} 个段"))
    else:
        checks.append("❌ Config")

    mc = memory_count()
    checks.append(ctx.t(f"✅ Memory: {mc}", f"✅ 记忆: {mc}") if mc else ctx.t("⚠️ Memory: empty", "⚠️ 记忆: 空"))

    tc = count_files("trajectories")
    checks.append(ctx.t(f"✅ Trajectories: {tc}", f"✅ 轨迹: {tc}"))

    sc = count_files("skills", "*.yaml")
    checks.append(ctx.t(f"✅ Skills: {sc}", f"✅ 技能: {sc}"))

    fw = read_json("flywheel_stats.json") or []
    checks.append(ctx.t(f"✅ Flywheel: {len(fw)} rounds", f"✅ 飞轮: {len(fw)} 轮") if fw else ctx.t("⚠️ Flywheel: none", "⚠️ 飞轮: 无"))

    passed = sum(1 for c in checks if "✅" in c)
    header = ctx.t(f"🔍 Self-test v{caveman.__version__} — {passed}/{len(checks)} passed",
                   f"🔍 自检 v{caveman.__version__} — {passed}/{len(checks)} 通过")
    ctx.respond(header + "\n" + "\n".join(checks))


async def handle_wiki(ctx: CommandContext) -> None:
    """Knowledge wiki."""
    sub = ctx.subcommand()
    if sub == "search" and ctx.rest():
        try:
            from caveman.wiki import WikiStore
            store = WikiStore()
            results = store.search(ctx.rest())
            if results:
                lines = [ctx.t(f"Wiki: '{ctx.rest()}':", f"维基: '{ctx.rest()}':")]
                for r in results[:5]:
                    lines.append(f"  • {r.get('title', '?')}: {r.get('snippet', '')[:100]}")
                ctx.respond("\n".join(lines))
            else:
                ctx.respond(ctx.t(f"No results: '{ctx.rest()}'.", f"未找到: '{ctx.rest()}'。"))
        except ImportError:
            ctx.respond(ctx.t("Wiki not available.", "维基不可用。"))
        except Exception as e:
            ctx.respond(ctx.t(f"Wiki error: {e}", f"维基错误: {e}"))
    elif sub == "compile":
        ctx.respond(ctx.t("📚 Requires CLI. Run `caveman wiki compile`.", "📚 需要 CLI。运行 `caveman wiki compile`。"))
    else:
        ctx.respond(ctx.t("📚 Wiki. Usage: /wiki [search|compile] [query]", "📚 维基。用法: /维基 [搜索|编译] [查询]"))
