"""Tool display metadata — emojis and labels for tool calls.

Used by heartbeat, progress indicators, and status displays.
"""
from __future__ import annotations

__all__ = ["tool_display",
    "format_tool_call"]

_TOOL_DISPLAY: dict[str, tuple[str, str]] = {
    # (emoji, label)
    "bash": ("💻", "Shell"),
    "file_read": ("📖", "Read"),
    "file_write": ("✏️", "Write"),
    "file_edit": ("🔧", "Edit"),
    "file_search": ("🔍", "Search"),
    "file_list": ("📂", "List"),
    "web_search": ("🌐", "Web Search"),
    "url_check": ("🔗", "URL Check"),
    "browser_navigate": ("🌍", "Browse"),
    "browser_snapshot": ("📸", "Snapshot"),
    "browser_click": ("👆", "Click"),
    "browser_type": ("⌨️", "Type"),
    "memory_store": ("💾", "Remember"),
    "memory_recall": ("🧠", "Recall"),
    "memory_recent": ("📋", "Recent"),
    "progress": ("📢", "Progress"),
    "delegate": ("🤖", "Delegate"),
    "vision_describe": ("👁️", "Vision"),
    "transcribe": ("🎤", "Transcribe"),
    "image_generate": ("🎨", "Image Gen"),
    "sandbox_exec": ("🧪", "Sandbox"),
    "process_start": ("⚙️", "Process"),
    "process_list": ("📊", "Processes"),
    "skill_list": ("📚", "Skills"),
    "skill_create": ("✨", "New Skill"),
    "session_search": ("🔎", "Session Search"),
    "cron_manage": ("⏰", "Cron"),
    "todo_manage": ("✅", "Todo"),
    "conversation_branch": ("🌿", "Branch"),
    "clarify": ("❓", "Clarify"),
    "gateway_send": ("📨", "Send"),
    "checkpoint_save": ("💾", "Checkpoint"),
    "checkpoint_restore": ("⏪", "Restore"),
}


def tool_display(name: str) -> tuple[str, str]:
    """Get (emoji, label) for a tool. Falls back to generic."""
    return _TOOL_DISPLAY.get(name, ("🔧", name))


def format_tool_call(name: str, compact: bool = False) -> str:
    """Format a tool call for display."""
    emoji, label = tool_display(name)
    return f"{emoji} {label}" if not compact else emoji
