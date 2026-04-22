"""Browser automation tool — @tool-decorated, lifecycle-aware.

Two backends:
  1. OpenClaw bridge: Use OpenClaw's browser tool via MCP
  2. Direct Playwright via BrowserManager: Multi-tab, SSRF-safe, cookies

Actions: navigate, snapshot, click, type, screenshot, evaluate,
         close, new_tab, close_tab, focus_tab, list_tabs, scroll,
         press, pdf, console
"""
from __future__ import annotations
import logging
from typing import Any

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_bridge = None
_manager = None  # BrowserManager instance


def set_bridge(bridge) -> None:
    global _bridge
    _bridge = bridge


async def _ensure_manager():
    """Lazy-init BrowserManager."""
    global _manager
    if _manager and _manager.active_page:
        return _manager
    from caveman.tools.builtin.browser_manager import BrowserManager
    _manager = BrowserManager(headless=True, ssrf_protection=True)
    await _manager.start()
    return _manager


async def close_browser() -> None:
    global _manager
    if _manager:
        await _manager.stop()
        _manager = None


def _mode() -> str:
    return "bridge" if _bridge else "standalone"


async def _bridge_call(action: str, **kwargs) -> dict[str, Any]:
    try:
        result = await _bridge.call_tool("browser", {"action": action, **kwargs})
        return {"ok": True, "data": result.get("result", "")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="browser",
    description="Browser automation: navigate, snapshot, click, type, screenshot, evaluate, tabs, scroll, pdf, console",
    params={
        "action": {
            "type": "string",
            "enum": [
                "navigate", "snapshot", "click", "type", "screenshot",
                "evaluate", "close", "new_tab", "close_tab", "focus_tab",
                "list_tabs", "scroll", "press", "pdf", "console",
            ],
            "description": "Browser action to perform",
        },
        "url": {"type": "string", "description": "URL for navigate"},
        "ref": {"type": "string", "description": "Element selector for click/type"},
        "text": {"type": "string", "description": "Text for type action"},
        "js": {"type": "string", "description": "JavaScript for evaluate"},
        "full_page": {"type": "boolean", "description": "Full page screenshot"},
        "compact": {"type": "boolean", "description": "Compact snapshot mode"},
        "tab_id": {"type": "string", "description": "Tab ID for tab operations"},
        "direction": {"type": "string", "description": "Scroll direction (up/down)"},
        "key": {"type": "string", "description": "Key for press action"},
        "path": {"type": "string", "description": "File path for pdf/screenshot save"},
    },
    required=["action"],
)
async def browser_dispatch(
    action: str,
    url: str = "",
    ref: str = "",
    text: str = "",
    js: str = "",
    full_page: bool = False,
    compact: bool = True,
    tab_id: str = "",
    direction: str = "down",
    key: str = "",
    path: str = "",
) -> dict:
    """Dispatch browser action."""
    # Bridge mode — delegate to OpenClaw
    if _mode() == "bridge" and action in ("navigate", "snapshot", "click", "type", "screenshot", "evaluate"):
        return await _bridge_call(action, url=url, ref=ref, text=text, js=js,
                                  full_page=full_page, compact=compact)

    # Standalone mode — use BrowserManager
    mgr = await _ensure_manager()

    if action == "navigate":
        return await mgr.navigate(url)
    if action == "snapshot":
        return await mgr.snapshot(compact)
    if action == "click":
        return await mgr.click(ref)
    if action == "type":
        return await mgr.fill(ref, text)
    if action == "screenshot":
        return await mgr.screenshot(full_page=full_page, path=path or None)
    if action == "evaluate":
        return await mgr.evaluate(js)
    if action == "scroll":
        return await mgr.scroll(direction)
    if action == "press":
        return await mgr.press(key)
    if action == "pdf":
        return await mgr.pdf(path or "/tmp/page.pdf")
    if action == "console":
        logs = mgr.get_console_logs(tab_id)
        return {"ok": True, "logs": logs[-50:], "total": len(logs)}
    if action == "new_tab":
        tid = await mgr.new_tab(tab_id)
        return {"ok": True, "tab_id": tid}
    if action == "close_tab":
        return {"ok": mgr.close_tab(tab_id) if tab_id else False}
    if action == "focus_tab":
        return {"ok": mgr.focus_tab(tab_id) if tab_id else False}
    if action == "list_tabs":
        return {"ok": True, "tabs": mgr.list_tabs()}
    if action == "close":
        await close_browser()
        return {"ok": True}

    return {"ok": False, "error": f"Unknown action: {action}"}

__all__ = [
    "set_bridge",
    "close_browser",
    "browser_dispatch",
]

