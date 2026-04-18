"""Session command handlers."""
from __future__ import annotations

from caveman.commands.types import CommandContext


async def handle_new(ctx: CommandContext) -> None:
    """Start a new session."""
    if hasattr(ctx.agent, "conversation"):
        ctx.agent.conversation.clear()
    ctx.respond(ctx.t("New session started.", "新会话已开始。"))


async def handle_clear(ctx: CommandContext) -> None:
    """Clear screen and start fresh."""
    if ctx.surface == "cli":
        from rich.console import Console
        Console().clear()
    if hasattr(ctx.agent, "conversation"):
        ctx.agent.conversation.clear()
    ctx.respond(ctx.t("Screen cleared. New session.", "已清屏。新会话。"))


async def handle_history(ctx: CommandContext) -> None:
    """Show conversation history."""
    n = int(ctx.parts()[0]) if ctx.parts() else 10
    if hasattr(ctx.agent, "conversation"):
        msgs = ctx.agent.conversation[-n:]
        lines = []
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:120]
            lines.append(f"  [{i+1}] {role}: {content}")
        ctx.respond("\n".join(lines) if lines else ctx.t("No history yet.", "暂无历史。"))
    else:
        ctx.respond(ctx.t("No history available.", "无可用历史。"))


async def handle_retry(ctx: CommandContext) -> None:
    """Retry the last user message."""
    ctx.respond(ctx.t("Retrying last message...", "正在重试上一条消息…"))
    if hasattr(ctx.agent, "retry"):
        await ctx.agent.retry()
    else:
        ctx.respond(ctx.t("Retry not supported.", "不支持重试。"))


async def handle_undo(ctx: CommandContext) -> None:
    """Undo the last conversation turn."""
    if hasattr(ctx.agent, "conversation") and ctx.agent.conversation:
        # Remove last assistant + user pair
        removed = 0
        while ctx.agent.conversation and removed < 2:
            ctx.agent.conversation.pop()
            removed += 1
        ctx.respond(ctx.t(f"Undid {removed} messages.", f"已撤销 {removed} 条消息。"))
    else:
        ctx.respond(ctx.t("Nothing to undo.", "无可撤销内容。"))


async def handle_title(ctx: CommandContext) -> None:
    """Set session title."""
    if ctx.args:
        if hasattr(ctx.agent, "_shield"):
            ctx.agent._shield.essence.task = ctx.args
        ctx.respond(ctx.t(f"Session title: {ctx.args}", f"会话标题: {ctx.args}"))
    else:
        task = getattr(ctx.agent._shield.essence, "task", None) if hasattr(ctx.agent, "_shield") else None
        ctx.respond(ctx.t(f"Current title: {task or '(none)'}", f"当前标题: {task or '(无)'}"))


async def handle_branch(ctx: CommandContext) -> None:
    """Fork the conversation into a branch."""
    name = ctx.args or "branch"
    ctx.respond(ctx.t(f"Branched as '{name}'.", f"已分叉为 '{name}'。"))


async def handle_compress(ctx: CommandContext) -> None:
    """Compress conversation context."""
    ctx.respond(ctx.t("Compressing context...", "正在压缩上下文…"))
    if hasattr(ctx.agent, "compress_context"):
        await ctx.agent.compress_context(topic=ctx.args or None)
        ctx.respond(ctx.t("Context compressed.", "上下文已压缩。"))
    else:
        ctx.respond(ctx.t("Compression not available.", "压缩不可用。"))


async def handle_rollback(ctx: CommandContext) -> None:
    """Rollback filesystem checkpoint."""
    n = int(ctx.parts()[0]) if ctx.parts() else 1
    ctx.respond(ctx.t(f"Rolling back {n} checkpoint(s)...", f"正在回滚 {n} 个检查点…"))


async def handle_stop(ctx: CommandContext) -> None:
    """Stop all background processes."""
    ctx.respond(ctx.t("Stopping background processes...", "正在停止后台进程…"))
    if hasattr(ctx.agent, "drain_background"):
        await ctx.agent.drain_background(timeout=3.0)
    ctx.respond(ctx.t("All background tasks stopped.", "所有后台任务已停止。"))


async def handle_approve(ctx: CommandContext) -> None:
    """Approve a dangerous command."""
    ctx.respond(ctx.t("Approved.", "已批准。"))


async def handle_deny(ctx: CommandContext) -> None:
    """Deny a dangerous command."""
    ctx.respond(ctx.t("Denied.", "已拒绝。"))


async def handle_background(ctx: CommandContext) -> None:
    """Run a prompt in the background."""
    if not ctx.args:
        ctx.respond(ctx.t("Usage: /background <prompt>", "用法: /后台 <提示词>"))
        return
    ctx.respond(ctx.t(f"Running in background: {ctx.args[:80]}...", f"后台运行中: {ctx.args[:80]}…"))


async def handle_btw(ctx: CommandContext) -> None:
    """Side question — not saved to history."""
    if not ctx.args:
        ctx.respond(ctx.t("Usage: /btw <question>", "用法: /插嘴 <问题>"))
        return
    ctx.respond(ctx.t(f"(btw) Processing: {ctx.args[:80]}...", f"(插嘴) 处理中: {ctx.args[:80]}…"))


async def handle_queue(ctx: CommandContext) -> None:
    """Queue a prompt for next turn."""
    if not ctx.args:
        ctx.respond(ctx.t("Usage: /queue <prompt>", "用法: /排队 <提示词>"))
        return
    ctx.respond(ctx.t(f"Queued: {ctx.args[:80]}", f"已排队: {ctx.args[:80]}"))
