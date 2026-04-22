"""Status Panel — session and system status display.

Extracted from OpenClaw status.ts (930 lines).
Generates rich status panels for CLI, Discord, and Telegram.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "SessionStatus",
    "SystemStatus",
    "build_system_status",
    "format_status_text",
    "format_status_embed",
]



@dataclass
class SessionStatus:
    """Status of a single session."""
    session_key: str = ""
    session_id: str = ""
    model: str = ""
    provider: str = ""
    total_tokens: int = 0
    messages: int = 0
    tool_calls: int = 0
    uptime_seconds: float = 0
    compaction_count: int = 0
    is_active: bool = False
    last_activity: float = 0
    reasoning_mode: str = "off"
    verbose_level: str = "off"


@dataclass
class SystemStatus:
    """Overall system status."""
    version: str = "0.5.0"
    hostname: str = ""
    os_info: str = ""
    python_version: str = ""
    uptime_seconds: float = 0
    active_sessions: int = 0
    total_sessions: int = 0
    connected_platforms: List[str] = field(default_factory=list)
    mcp_servers: int = 0
    tools_count: int = 0
    memory_entries: int = 0
    skills_count: int = 0
    process_count: int = 0


def build_system_status(
    sessions: Optional[Dict[str, Any]] = None,
    platforms: Optional[List[str]] = None,
    start_time: float = 0,
    tools_count: int = 0,
    mcp_servers: int = 0,
    memory_entries: int = 0,
    skills_count: int = 0,
) -> SystemStatus:
    """Build system status from current state."""
    import sys
    status = SystemStatus(
        hostname=platform.node(),
        os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        uptime_seconds=time.monotonic() - start_time if start_time else 0,
        connected_platforms=platforms or [],
        tools_count=tools_count,
        mcp_servers=mcp_servers,
        memory_entries=memory_entries,
        skills_count=skills_count,
    )

    if sessions:
        status.total_sessions = len(sessions)
        status.active_sessions = sum(
            1 for s in sessions.values()
            if isinstance(s, dict) and s.get("is_active", False)
        )

    return status


def format_status_text(
    system: SystemStatus,
    session: Optional[SessionStatus] = None,
    surface: str = "cli",
) -> str:
    """Format status for display."""
    lines = []

    # Header
    lines.append(f"🦴 Caveman v{system.version}")
    lines.append(f"Host: {system.hostname}")
    lines.append(f"OS: {system.os_info}")
    lines.append(f"Python: {system.python_version}")

    if system.uptime_seconds > 0:
        lines.append(f"Uptime: {_format_duration(system.uptime_seconds)}")

    lines.append("")

    # Platforms
    if system.connected_platforms:
        lines.append(f"Platforms: {', '.join(system.connected_platforms)}")
    else:
        lines.append("Platforms: none connected")

    # Resources
    lines.append(f"Sessions: {system.active_sessions}/{system.total_sessions}")
    lines.append(f"Tools: {system.tools_count}")
    if system.mcp_servers:
        lines.append(f"MCP Servers: {system.mcp_servers}")
    if system.memory_entries:
        lines.append(f"Memories: {system.memory_entries}")
    if system.skills_count:
        lines.append(f"Skills: {system.skills_count}")

    # Current session
    if session:
        lines.append("")
        lines.append("── Current Session ──")
        lines.append(f"Key: {session.session_key}")
        if session.model:
            model_display = session.model
            if session.provider:
                model_display = f"{session.provider}/{session.model}"
            lines.append(f"Model: {model_display}")
        lines.append(f"Tokens: {session.total_tokens:,}")
        lines.append(f"Messages: {session.messages}")
        lines.append(f"Tool calls: {session.tool_calls}")
        if session.compaction_count:
            lines.append(f"Compactions: {session.compaction_count}")
        lines.append(f"Reasoning: {session.reasoning_mode}")
        lines.append(f"Verbose: {session.verbose_level}")

    return "\n".join(lines)


def format_status_embed(
    system: SystemStatus,
    session: Optional[SessionStatus] = None,
) -> Dict[str, Any]:
    """Format status as a Discord embed."""
    fields = [
        {"name": "Host", "value": system.hostname, "inline": True},
        {"name": "Uptime", "value": _format_duration(system.uptime_seconds), "inline": True},
        {"name": "Platforms", "value": ", ".join(system.connected_platforms) or "none", "inline": True},
        {"name": "Sessions", "value": f"{system.active_sessions}/{system.total_sessions}", "inline": True},
        {"name": "Tools", "value": str(system.tools_count), "inline": True},
    ]

    if session:
        model = session.model
        if session.provider:
            model = f"{session.provider}/{session.model}"
        fields.extend([
            {"name": "Model", "value": model or "default", "inline": True},
            {"name": "Tokens", "value": f"{session.total_tokens:,}", "inline": True},
            {"name": "Reasoning", "value": session.reasoning_mode or "off", "inline": True},
        ])

    return {
        "title": f"🦴 Caveman v{system.version}",
        "color": 0x8B4513,  # SaddleBrown
        "fields": fields,
        "footer": {"text": f"{system.os_info} | Python {system.python_version}"},
    }


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"
