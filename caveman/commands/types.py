"""Core data types for the slash command system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CommandDef:
    """Definition of a slash command registered in the system."""

    name: str
    description: str
    category: str
    handler: str  # dotted path like "session.handle_new"
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    subcommands: tuple[str, ...] = ()
    cli_only: bool = False
    gateway_only: bool = False
    hidden: bool = False
    dangerous: bool = False
    cooldown_seconds: int = 0
    examples: tuple[str, ...] = ()
    since: str = "0.3.0"
    description_i18n: dict[str, str] = field(default_factory=dict)

    def desc(self, locale: str = "en") -> str:
        """Get description for locale, fallback to English."""
        if locale == "en" or not self.description_i18n:
            return self.description
        return self.description_i18n.get(locale, self.description)


# Chinese subcommand aliases → English canonical names
_ZH_SUBCOMMANDS: dict[str, str] = {
    "搜索": "search", "统计": "stats", "删除": "forget",
    "最近": "recent", "常用": "top", "信任": "trusted",
    "列表": "list", "添加": "add", "编辑": "edit",
    "暂停": "pause", "恢复": "resume", "运行": "run", "移除": "remove",
    "启用": "enable", "禁用": "disable",
    "连接": "connect", "断开": "disconnect",
    "编译": "compile", "回放": "replay", "评分": "score",
    "结果": "results", "开": "on", "关": "off",
}


@dataclass
class CommandContext:
    """Passed to every command handler."""

    command: str
    args: str
    agent: Any
    surface: str  # "cli" | "discord" | "telegram" | "gateway"
    locale: str = "en"  # "en" | "zh" | "ja" | "ko"
    session_key: str = ""
    respond: Callable[..., None] = field(default=lambda msg: None)

    @property
    def is_zh(self) -> bool:
        """Shortcut: is locale Chinese?"""
        return self.locale.startswith("zh")

    def t(self, en: str, zh: str = "") -> str:
        """Translate: return zh text if locale is zh and zh provided, else en."""
        if self.is_zh and zh:
            return zh
        return en

    def parts(self) -> list[str]:
        """Split args into parts."""
        return self.args.split() if self.args else []

    def subcommand(self) -> str:
        """Return first arg as subcommand (with zh→en alias resolution)."""
        p = self.parts()
        if not p:
            return ""
        return _ZH_SUBCOMMANDS.get(p[0], p[0])

    def rest(self) -> str:
        """Return args after the subcommand."""
        p = self.parts()
        return " ".join(p[1:]) if len(p) > 1 else ""
