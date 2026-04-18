"""Surface-aware response style rules.

Injected as a system prompt layer so the agent automatically adapts
its formatting to the output surface (Discord, Telegram, CLI, etc.).
"""
from __future__ import annotations

STYLES: dict[str, str] = {
    "discord": """## Response Style (Discord)
You are replying on Discord. Follow these format rules:
- Use blank lines between paragraphs — never wall-of-text
- Use emoji as section markers (✅ ❌ 📌 🔍 💡 ⚡ 🎯 🔧)
- Lists: use `•` or `-`, keep indentation clean
- Code: wrap in ``` with language tag
- Emphasis: use **bold** for key terms and conclusions
- NO markdown tables (Discord doesn't render them)
- NO # headers (Discord doesn't render them in bot messages)
- Long replies: break into sections with emoji + **bold** mini-headers
- Keep messages under 1900 chars when possible
- Tone: warm, direct, conversational""",

    "telegram": """## Response Style (Telegram)
Format rules for Telegram:
- Use blank lines between paragraphs
- Use emoji as section markers
- Lists: use • or -
- Code: wrap in ``` or ` for inline
- Emphasis: use **bold** or _italic_
- NO markdown tables
- Keep messages concise — Telegram users expect quick reads
- Tone: warm, direct, conversational""",

    "cli": """## Response Style (CLI)
Format rules for terminal:
- Use blank lines between sections
- Lists: use - or *
- Code: wrap in ``` with language tag
- Use ANSI-friendly formatting (no complex unicode)
- Be concise — terminal users want efficiency
- Show file paths, line numbers, and commands prominently""",
}

# Fallback for unknown surfaces
_DEFAULT = """## Response Style
- Use blank lines between paragraphs
- Use emoji as section markers where appropriate
- Lists: use • or -
- Code: wrap in ``` with language tag
- Be concise and well-structured"""


def get_response_style(surface: str) -> str:
    """Get response style rules for a given surface (system prompt layer)."""
    return STYLES.get(surface, _DEFAULT)


# Short format reminders for user message injection (recency bias)
_FORMAT_REMINDERS: dict[str, str] = {
    "discord": "[Format: Discord — emoji分隔, **bold**关键词, 空行分段, 禁止表格/标题]",
    "telegram": "[Format: Telegram — emoji, bold, 空行分段, 简洁]",
}


def get_format_reminder(surface: str) -> str:
    """Get a short format reminder to append to user messages.

    Uses recency bias: LLM pays more attention to the last thing it sees.
    Returns empty string for surfaces that don't need reminders (CLI).
    """
    return _FORMAT_REMINDERS.get(surface, "")
