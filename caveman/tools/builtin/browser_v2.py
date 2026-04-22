"""Browser Tool v2 — full browser automation with vision, console, recording.

Extracted from Hermes browser_tool.py (2387 lines).
Adds: vision analysis, console access, scroll, back, press, recording,
session cleanup, screenshot management.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict

from caveman.tools.registry import tool
from caveman.tools.builtin.browser_manager import BrowserManager
from caveman.timeouts import BROWSER_SESSION

__all__ = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_console",
    "browser_get_images",
    "browser_close",
    "cleanup_all_browsers",
    # depth re-exports
    "VisionConfig",
    "browser_vision",
    "save_screenshot",
    "cleanup_old_screenshots",
    "RecordingSession",
    "CamofoxProfile",
    "PRESET_PROFILES",
    "get_camofox_profile",
    "cleanup_old_recordings",
]


logger = logging.getLogger("caveman.tools.browser_v2")

# ── Session Management ──

_sessions: Dict[str, Dict[str, Any]] = {}
_cleanup_interval = 300  # 5 minutes
_session_timeout = BROWSER_SESSION  # 30 minutes
_max_sessions = 10
_screenshots_dir = Path.home() / ".caveman" / "screenshots"
_recordings_dir = Path.home() / ".caveman" / "recordings"


def _get_session(task_id: str = "default") -> Dict[str, Any]:
    """Get or create a browser session."""
    if task_id not in _sessions:
        if len(_sessions) >= _max_sessions:
            _cleanup_inactive_sessions()
        _sessions[task_id] = {
            "manager": BrowserManager(),
            "last_activity": time.monotonic(),
            "screenshots": [],
            "recording": False,
        }
    else:
        _sessions[task_id]["last_activity"] = time.monotonic()
    return _sessions[task_id]


def _cleanup_inactive_sessions() -> int:
    """Remove sessions that have been inactive."""
    now = time.monotonic()
    to_remove = [
        tid for tid, s in _sessions.items()
        if now - s["last_activity"] > _session_timeout
    ]
    for tid in to_remove:
        _cleanup_session(tid)
    return len(to_remove)


def _cleanup_session(task_id: str) -> None:
    """Clean up a single session."""
    session = _sessions.pop(task_id, None)
    if session and session.get("manager"):
        try:
            mgr = session["manager"]
            if hasattr(mgr, "close"):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(mgr.close())
                except RuntimeError:
                    # No running loop — create one for sync cleanup
                    asyncio.run(mgr.close())
        except Exception:
            pass  # intentional: Exception suppressed


# ── Screenshot Management ──

def _save_screenshot(data: bytes, task_id: str = "default") -> str:
    """Save screenshot and return path."""
    path = save_screenshot(data, prefix=task_id)

    session = _sessions.get(task_id)
    if session:
        session["screenshots"].append(str(path))
        # Keep only last 20 screenshots per session
        if len(session["screenshots"]) > 20:
            old = session["screenshots"].pop(0)
            try:
                Path(old).unlink(missing_ok=True)
            except Exception:
                pass  # intentional: Exception suppressed

    return str(path)


def _cleanup_old_screenshots(max_age_hours: int = 24) -> int:
    """Remove screenshots older than max_age_hours."""
    return cleanup_old_screenshots(max_age_hours=max_age_hours)


# ── Content Extraction ──

def _truncate_snapshot(text: str, max_chars: int = 8000) -> str:
    """Truncate accessibility snapshot to fit context."""
    if len(text) <= max_chars:
        return text
    # Keep first and last portions
    half = max_chars // 2
    return text[:half] + "\n\n... [truncated] ...\n\n" + text[-half:]


def _extract_relevant_content(snapshot: str, query: str = "") -> str:
    """Extract relevant portions of a page snapshot."""
    if not query or len(snapshot) <= 4000:
        return snapshot

    lines = snapshot.split("\n")
    query_lower = query.lower()
    scored = []
    for i, line in enumerate(lines):
        score = 0
        if query_lower in line.lower():
            score += 10
        # Boost interactive elements
        if any(tag in line.lower() for tag in ["button", "link", "input", "select"]):
            score += 3
        # Boost headings
        if any(tag in line.lower() for tag in ["heading", "h1", "h2", "h3"]):
            score += 2
        scored.append((score, i, line))

    scored.sort(key=lambda x: -x[0])
    # Take top lines, preserving order
    top_indices = sorted(s[1] for s in scored[:100])
    return "\n".join(lines[i] for i in top_indices)


# ── Tool Definitions ──

@tool(
    name="browser_navigate",
    description="Navigate to a URL in the browser",
    params={
        "url": {"type": "string", "description": "URL to navigate to"},
        "task_id": {"type": "string", "description": "Browser session ID (optional)"},
    },
    required=["url"],
)
async def browser_navigate(url: str, task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    return await mgr.navigate(url)


@tool(
    name="browser_snapshot",
    description="Take an accessibility snapshot of the current page",
    params={
        "query": {"type": "string", "description": "Focus query to filter content"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
)
async def browser_snapshot(query: str = "", task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    result = await mgr.snapshot()
    if result.get("ok") and result.get("content"):
        content = result["content"]
        if query:
            content = _extract_relevant_content(content, query)
        result["content"] = _truncate_snapshot(content)
    return result


@tool(
    name="browser_click",
    description="Click an element by reference number",
    params={
        "ref": {"type": "string", "description": "Element reference from snapshot"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
    required=["ref"],
)
async def browser_click(ref: str, task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    return await session["manager"].click(ref)


@tool(
    name="browser_type",
    description="Type text into a focused element",
    params={
        "ref": {"type": "string", "description": "Element reference"},
        "text": {"type": "string", "description": "Text to type"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
    required=["ref", "text"],
)
async def browser_type(ref: str, text: str, task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    return await session["manager"].fill(ref, text)


@tool(
    name="browser_scroll",
    description="Scroll the page up or down",
    params={
        "direction": {"type": "string", "description": "up or down"},
        "amount": {"type": "integer", "description": "Pixels to scroll (default 500)"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
    required=["direction"],
)
async def browser_scroll(
    direction: str, amount: int = 500, task_id: str = "default",
) -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    tab = mgr._tabs.get(mgr._active_tab)
    if not tab or not tab.get("page"):
        return {"ok": False, "error": "No active tab"}
    page = tab["page"]
    try:
        delta = amount if direction.lower() == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        return {"ok": True, "direction": direction, "amount": amount}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser_back",
    description="Go back in browser history",
    params={"task_id": {"type": "string", "description": "Browser session ID"}},
)
async def browser_back(task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    tab = mgr._tabs.get(mgr._active_tab)
    if not tab or not tab.get("page"):
        return {"ok": False, "error": "No active tab"}
    try:
        await tab["page"].go_back()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser_press",
    description="Press a keyboard key",
    params={
        "key": {"type": "string", "description": "Key to press (Enter, Tab, Escape, etc.)"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
    required=["key"],
)
async def browser_press(key: str, task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    tab = mgr._tabs.get(mgr._active_tab)
    if not tab or not tab.get("page"):
        return {"ok": False, "error": "No active tab"}
    try:
        await tab["page"].keyboard.press(key)
        return {"ok": True, "key": key}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser_console",
    description="Execute JavaScript in the browser console",
    params={
        "expression": {"type": "string", "description": "JavaScript expression to evaluate"},
        "task_id": {"type": "string", "description": "Browser session ID"},
    },
    required=["expression"],
)
async def browser_console(expression: str, task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    tab = mgr._tabs.get(mgr._active_tab)
    if not tab or not tab.get("page"):
        return {"ok": False, "error": "No active tab"}
    try:
        result = await tab["page"].evaluate(expression)
        return {"ok": True, "result": str(result)[:5000]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser_get_images",
    description="Get all images on the current page",
    params={"task_id": {"type": "string", "description": "Browser session ID"}},
)
async def browser_get_images(task_id: str = "default") -> Dict[str, Any]:
    session = _get_session(task_id)
    mgr = session["manager"]
    tab = mgr._tabs.get(mgr._active_tab)
    if not tab or not tab.get("page"):
        return {"ok": False, "error": "No active tab"}
    try:
        images = await tab["page"].evaluate("""
            Array.from(document.images).map(img => ({
                src: img.src,
                alt: img.alt,
                width: img.naturalWidth,
                height: img.naturalHeight,
            })).filter(img => img.src && img.width > 50)
        """)
        return {"ok": True, "images": images[:50]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser_close",
    description="Close browser session and clean up",
    params={"task_id": {"type": "string", "description": "Browser session ID"}},
)
async def browser_close(task_id: str = "default") -> Dict[str, Any]:
    _cleanup_session(task_id)
    return {"ok": True}


def cleanup_all_browsers() -> int:
    """Clean up all browser sessions."""
    count = len(_sessions)
    for tid in list(_sessions.keys()):
        _cleanup_session(tid)
    _cleanup_old_screenshots()
    return count

from caveman.tools.builtin.browser_depth import (  # noqa: F401, E402  # depth wiring
    VisionConfig, browser_vision, save_screenshot, cleanup_old_screenshots,
    RecordingSession, CamofoxProfile, PRESET_PROFILES, get_camofox_profile,
    cleanup_old_recordings,
)
