"""Command registry — central catalog of all slash commands.

Provides:
- COMMAND_REGISTRY: list of all CommandDef instances
- resolve_command(): name/alias → CommandDef lookup
- get_by_category(): grouped view
- rebuild_lookups(): refresh after plugin registration
"""
from __future__ import annotations

from caveman.commands.types import CommandDef

C = CommandDef  # shorthand

def _zh(zh: str) -> dict[str, str]:
    """Shorthand for Chinese i18n."""
    return {"zh": zh}

# ── Session (15) ──────────────────────────────────────────────
_SESSION = [
    C("new", "Start a new session", "session", "session.handle_new",
      aliases=("reset",), description_i18n=_zh("开始新会话")),
    C("clear", "Clear screen and start fresh", "session", "session.handle_clear",
      description_i18n=_zh("清屏并重新开始")),
    C("history", "Show conversation history", "session", "session.handle_history",
      aliases=("h",), args_hint="[n]", examples=("/history", "/history 5"),
      description_i18n=_zh("查看对话历史")),
    C("retry", "Retry the last message", "session", "session.handle_retry",
      description_i18n=_zh("重试上一条消息")),
    C("undo", "Undo the last conversation turn", "session", "session.handle_undo",
      description_i18n=_zh("撤销上一轮对话")),
    C("title", "Set session title", "session", "session.handle_title",
      args_hint="[name]", examples=("/title My Project",),
      description_i18n=_zh("设置会话标题")),
    C("branch", "Fork the conversation", "session", "session.handle_branch",
      aliases=("fork",), args_hint="[name]",
      description_i18n=_zh("分叉对话")),
    C("compress", "Compress conversation context", "session", "session.handle_compress",
      args_hint="[topic]", description_i18n=_zh("压缩对话上下文")),
    C("rollback", "Rollback filesystem checkpoint", "session", "session.handle_rollback",
      args_hint="[n]", dangerous=True, description_i18n=_zh("回滚文件系统检查点")),
    C("stop", "Stop all background processes", "session", "session.handle_stop",
      aliases=("abort",), description_i18n=_zh("停止所有后台进程")),
    C("approve", "Approve a dangerous command", "session", "session.handle_approve",
      args_hint="[session|always]", description_i18n=_zh("批准危险命令")),
    C("deny", "Deny a dangerous command", "session", "session.handle_deny",
      description_i18n=_zh("拒绝危险命令")),
    C("background", "Run a prompt in the background", "session", "session.handle_background",
      aliases=("bg",), args_hint="<prompt>",
      examples=("/background summarize this repo",),
      description_i18n=_zh("后台运行任务")),
    C("btw", "Side question (not saved to history)", "session", "session.handle_btw",
      args_hint="<question>", description_i18n=_zh("插嘴提问（不存历史）")),
    C("queue", "Queue a prompt for next turn", "session", "session.handle_queue",
      args_hint="<prompt>", description_i18n=_zh("排队下一轮任务")),
]

# ── Configuration (12) ────────────────────────────────────────
_CONFIG = [
    C("model", "Show or switch model", "config", "config.handle_model",
      args_hint="[name] [--global]",
      examples=("/model", "/model claude-sonnet-4-6"),
      description_i18n=_zh("查看或切换模型")),
    C("provider", "Show available providers", "config", "config.handle_provider",
      description_i18n=_zh("查看可用的模型提供商")),
    C("reasoning", "Set reasoning level", "config", "config.handle_reasoning",
      aliases=("think",), args_hint="[none|low|medium|high|show|hide]",
      subcommands=("none", "low", "medium", "high", "show", "hide"),
      examples=("/reasoning high", "/reasoning show"),
      description_i18n=_zh("设置推理深度")),
    C("fast", "Toggle fast mode", "config", "config.handle_fast",
      args_hint="[on|off|status]", subcommands=("on", "off", "status"),
      description_i18n=_zh("切换快速模式")),
    C("verbose", "Toggle verbose output", "config", "config.handle_verbose",
      args_hint="[on|off]", subcommands=("on", "off"),
      description_i18n=_zh("切换详细输出")),
    C("personality", "Set agent personality", "config", "config.handle_personality",
      args_hint="[name]", description_i18n=_zh("设置 Agent 人格")),
    C("yolo", "Skip all approval prompts", "config", "config.handle_yolo",
      dangerous=True, description_i18n=_zh("跳过所有审批确认")),
    C("voice", "Voice mode settings", "config", "config.handle_voice",
      args_hint="[on|off|tts|status]", subcommands=("on", "off", "tts", "status"),
      description_i18n=_zh("语音模式设置")),
    C("skin", "Switch UI theme", "config", "config.handle_skin",
      aliases=("theme",), args_hint="[name]",
      description_i18n=_zh("切换界面主题")),
    C("config", "View or modify configuration", "config", "config.handle_config",
      args_hint="[key] [value]", examples=("/config", "/config max_iterations 100"),
      description_i18n=_zh("查看或修改配置")),
    C("elevated", "Toggle elevated permissions", "config", "config.handle_elevated",
      args_hint="[on|off|ask]", subcommands=("on", "off", "ask"),
      description_i18n=_zh("切换提权模式")),
    C("activation", "Set chat activation mode", "config", "config.handle_activation",
      args_hint="[mention|always]", subcommands=("mention", "always"),
      gateway_only=True, description_i18n=_zh("设置聊天激活方式")),
]

# ── Tools & Skills (8) ────────────────────────────────────────
_TOOLS = [
    C("tools", "Manage tools", "tools", "tools.handle_tools",
      args_hint="[list|disable|enable] [name]",
      subcommands=("list", "disable", "enable"),
      examples=("/tools list", "/tools disable bash"),
      description_i18n=_zh("管理工具")),
    C("skills", "Manage skills", "tools", "tools.handle_skills",
      args_hint="[list|search|install|info] [name]",
      subcommands=("list", "search", "install", "info"),
      examples=("/skills list", "/skills search git"),
      description_i18n=_zh("管理技能")),
    C("engines", "Manage engines", "tools", "tools.handle_engines",
      args_hint="[list|enable|disable] [name]",
      subcommands=("list", "enable", "disable"),
      examples=("/engines list", "/engines disable nudge"),
      description_i18n=_zh("管理引擎")),
    C("cron", "Manage scheduled tasks", "tools", "tools.handle_cron",
      args_hint="[list|add|edit|pause|resume|run|remove]",
      subcommands=("list", "add", "edit", "pause", "resume", "run", "remove"),
      description_i18n=_zh("管理定时任务")),
    C("reload-mcp", "Reload MCP servers", "tools", "tools.handle_reload_mcp",
      description_i18n=_zh("重载 MCP 服务")),
    C("browser", "Browser tool control", "tools", "tools.handle_browser",
      args_hint="[connect|disconnect|status]",
      subcommands=("connect", "disconnect", "status"),
      description_i18n=_zh("浏览器工具控制")),
    C("plugins", "List installed plugins", "tools", "tools.handle_plugins",
      description_i18n=_zh("查看已安装插件")),
    C("import", "Import data from other tools", "tools", "tools.handle_import",
      args_hint="[--from|--detect|--all]",
      examples=("/import --detect", "/import --from claude-code"),
      description_i18n=_zh("从其他工具导入数据")),
]

# ── Info & Diagnostics (12) ───────────────────────────────────
_INFO = [
    C("status", "Show session status panel", "info", "info.handle_status",
      description_i18n=_zh("显示会话状态面板")),
    C("help", "Show help or command details", "info", "info.handle_help",
      aliases=("?",), args_hint="[command]",
      examples=("/help", "/help model"),
      description_i18n=_zh("显示帮助信息")),
    C("commands", "Browse all commands (paginated)", "info", "info.handle_commands",
      args_hint="[page]", description_i18n=_zh("浏览所有命令")),
    C("usage", "Show token usage", "info", "info.handle_usage",
      args_hint="[off|tokens|full]", subcommands=("off", "tokens", "full"),
      description_i18n=_zh("查看 Token 用量")),
    C("insights", "Usage analytics", "info", "info.handle_insights",
      args_hint="[days]", description_i18n=_zh("使用分析")),
    C("doctor", "Run health check", "info", "info.handle_doctor",
      description_i18n=_zh("运行健康检查")),
    C("memory", "Memory management", "info", "info.handle_memory",
      args_hint="[stats|search|recent|top|trusted|forget] [query]",
      subcommands=("stats", "search", "recent", "top", "trusted", "forget"),
      examples=("/memory stats", "/memory search git", "/memory top"),
      description_i18n=_zh("记忆管理")),
    C("recall", "Restore previous context", "info", "info.handle_recall",
      args_hint="[topic]", description_i18n=_zh("恢复之前的上下文")),
    C("shield", "Show Shield essence state", "info", "info.handle_shield",
      description_i18n=_zh("显示 Shield 状态")),
    C("reflect", "Show post-task reflections", "info", "info.handle_reflect",
      description_i18n=_zh("显示任务后反思")),
    C("audit", "Export audit log", "info", "info.handle_audit",
      args_hint="[path]", description_i18n=_zh("导出审计日志")),
    C("ratelimit", "Show rate limit status", "info", "info.handle_ratelimit",
      description_i18n=_zh("查看速率限制状态")),
]

# ── System (5) ────────────────────────────────────────────────
_SYSTEM = [
    C("restart", "Restart the gateway", "system", "system.handle_restart",
      dangerous=True, description_i18n=_zh("重启网关")),
    C("update", "Update Caveman to latest", "system", "system.handle_update",
      description_i18n=_zh("更新 Caveman")),
    C("profile", "Show current profile", "system", "system.handle_profile",
      description_i18n=_zh("显示当前配置档")),
    C("sethome", "Set current channel as home", "system", "system.handle_sethome",
      aliases=("set-home",), gateway_only=True,
      description_i18n=_zh("设置当前频道为主频道")),
    C("quit", "Exit Caveman", "system", "system.handle_quit",
      aliases=("exit", "q"), cli_only=True,
      description_i18n=_zh("退出 Caveman")),
]

# ── Caveman Exclusive (5) ─────────────────────────────────────
_CAVEMAN = [
    C("flywheel", "Flywheel self-improvement status", "caveman", "caveman.handle_flywheel",
      args_hint="[status|trigger|history]",
      subcommands=("status", "trigger", "history"),
      examples=("/flywheel status", "/flywheel trigger"),
      description_i18n=_zh("飞轮自我改进状态")),
    C("trajectory", "Trajectory management", "caveman", "caveman.handle_trajectory",
      args_hint="[list|replay|score]",
      subcommands=("list", "replay", "score"),
      description_i18n=_zh("轨迹管理")),
    C("bench", "Run performance benchmarks", "caveman", "caveman.handle_bench",
      args_hint="[run|results]", subcommands=("run", "results"),
      description_i18n=_zh("运行性能基准测试")),
    C("selftest", "Run self-diagnostics", "caveman", "caveman.handle_selftest",
      description_i18n=_zh("运行自检")),
    C("wiki", "Knowledge wiki", "caveman", "caveman.handle_wiki",
      args_hint="[search|compile] [query]",
      subcommands=("search", "compile"),
      examples=("/wiki search memory", "/wiki compile"),
      description_i18n=_zh("知识维基")),
]

# ── Master registry ───────────────────────────────────────────
COMMAND_REGISTRY: list[CommandDef] = (
    _SESSION + _CONFIG + _TOOLS + _INFO + _SYSTEM + _CAVEMAN
)

# ── Lookup indexes (rebuilt on demand) ────────────────────────
_name_index: dict[str, CommandDef] = {}
_alias_index: dict[str, CommandDef] = {}
_zh_reverse_index: dict[str, CommandDef] = {}  # Chinese name → CommandDef


def rebuild_lookups() -> None:
    """Rebuild name, alias, and Chinese reverse indexes."""
    _name_index.clear()
    _alias_index.clear()
    _zh_reverse_index.clear()
    for cmd in COMMAND_REGISTRY:
        _name_index[cmd.name] = cmd
        for alias in cmd.aliases:
            _alias_index[alias] = cmd
    # Build Chinese reverse index after ZH_COMMAND_NAMES is defined
    # (called lazily on first resolve_command)


def _ensure_zh_index() -> None:
    """Build Chinese reverse index on first use."""
    if _zh_reverse_index:
        return
    for en_name, zh_name in ZH_COMMAND_NAMES.items():
        cmd = _name_index.get(en_name)
        if cmd:
            _zh_reverse_index[zh_name] = cmd


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef."""
    if not _name_index:
        rebuild_lookups()
    name = name.lstrip("/").lower()
    result = _name_index.get(name) or _alias_index.get(name)
    if result:
        return result
    # O(1) Chinese reverse lookup
    _ensure_zh_index()
    return _zh_reverse_index.get(name)


def get_by_category() -> dict[str, list[CommandDef]]:
    """Group commands by category."""
    cats: dict[str, list[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        cats.setdefault(cmd.category, []).append(cmd)
    return cats


# Chinese command name mapping (English name → Chinese name)
# Used by handle_commands to show localized command names
ZH_COMMAND_NAMES: dict[str, str] = {
    "help": "帮助", "status": "状态", "model": "模型",
    "commands": "命令", "memory": "记忆", "tools": "工具",
    "skills": "技能", "doctor": "诊断", "selftest": "自检",
    "shield": "护盾", "flywheel": "飞轮", "config": "配置",
    "new": "新会话", "history": "历史", "retry": "重试",
    "undo": "撤销", "stop": "停止", "background": "后台",
    "provider": "提供商", "reasoning": "推理", "fast": "快速",
    "voice": "语音", "skin": "主题",
    "engines": "引擎", "cron": "定时", "plugins": "插件",
    "import": "导入", "browser": "浏览器",
    "usage": "用量", "insights": "分析", "recall": "回忆",
    "reflect": "反思", "audit": "审计", "ratelimit": "限速",
    "restart": "重启", "update": "更新", "profile": "档案",
    "trajectory": "轨迹", "bench": "基准", "wiki": "维基",
    "clear": "清屏", "title": "标题", "branch": "分叉",
    "compress": "压缩", "rollback": "回滚", "approve": "批准",
    "deny": "拒绝", "btw": "插嘴", "queue": "排队",
    # Previously missing
    "verbose": "详细", "personality": "人格", "yolo": "免审",
    "elevated": "提权", "activation": "激活",
    "reload-mcp": "重载mcp", "sethome": "设主频道", "quit": "退出",
}
