"""Multi-platform output formatting.

CLI uses Rich panels/tables; other surfaces get Markdown or plain text.
"""
from __future__ import annotations


def format_panel(title: str, content: str, surface: str = "cli") -> str:
    """Format a titled panel.

    CLI: Rich Panel markup. Others: Markdown block.
    """
    if surface == "cli":
        # Return Rich-compatible markup (caller wraps in Panel)
        return f"__panel__:{title}\n{content}"
    # Markdown for discord/telegram/gateway
    return f"**{title}**\n```\n{content}\n```"


def format_table(
    headers: list[str], rows: list[list[str]], surface: str = "cli"
) -> str:
    """Format a table.

    CLI: Rich Table markup. Others: Markdown table.
    """
    if surface == "cli":
        lines = ["  ".join(f"{h:<16}" for h in headers)]
        lines.append("  ".join("-" * 16 for _ in headers))
        for row in rows:
            lines.append("  ".join(f"{str(c):<16}" for c in row))
        return "\n".join(lines)
    # Markdown table
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def format_help(
    categories: dict[str, list], surface: str = "cli", locale: str = "en"
) -> str:
    """Format the help overview grouped by category."""
    from caveman.commands.registry import ZH_COMMAND_NAMES

    lines: list[str] = []
    is_zh = locale.startswith("zh")

    category_labels = {
        "session": "会话" if is_zh else "Session",
        "config": "配置" if is_zh else "Configuration",
        "tools": "工具与技能" if is_zh else "Tools & Skills",
        "info": "信息与诊断" if is_zh else "Info & Diagnostics",
        "system": "系统" if is_zh else "System",
        "caveman": "Caveman 专属" if is_zh else "Caveman Exclusive",
    }
    for cat, cmds in categories.items():
        label = category_labels.get(cat, cat.title())
        lines.append(f"\n{label}:")
        for cmd in cmds:
            display = ZH_COMMAND_NAMES.get(cmd.name, cmd.name) if is_zh else cmd.name
            alias_str = f" ({', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
            hint = f" {cmd.args_hint}" if cmd.args_hint else ""
            desc = cmd.desc(locale) if hasattr(cmd, 'desc') else cmd.description
            lines.append(f"  /{display}{hint}{alias_str} — {desc}")
    if is_zh:
        lines.append("\n输入 /帮助 <命令> 查看详情。/命令 查看分页列表。")
    else:
        lines.append("\nType /help <command> for details. /commands for paginated list.")
    return "\n".join(lines)
