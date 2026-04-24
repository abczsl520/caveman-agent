"""Behavior rules — single source of truth for agent output conventions.

All closing formats, output styles, and behavioral constants live here.
Other modules import via get_rule() instead of hardcoding values.
"""
from __future__ import annotations

_RULES: dict[str, str] = {
    "CLOSING_FORMAT": "✅---本轮已完成---✅",
    "AGENT_CLOSING_FORMAT": "✅ DONE",
    "OUTPUT_STYLE": (
        "- 直接开始干活，不说'让我来帮你'、'我来看看'这种客套话\n"
        "- 工具调用之间一两句话说在干什么\n"
        "- 代码块用 ``` + 语言标签\n"
        "- 加粗只用于文件名/路径/关键概念，不要每句话都加粗\n"
        "- 不用装饰性 emoji（🔍 📌 💡），只在状态标记时用 ✅ ❌\n"
        "- 短段落，空行分隔，不写长段落\n"
        "- 完成后 2-3 句总结，不长篇大论\n"
        "- 技术术语直接用英文\n"
        "- 不解释显而易见的事情，不重复自己"
    ),
}


def get_rule(name: str) -> str | None:
    """Get a behavior rule by name. Returns None if not found."""
    return _RULES.get(name)
