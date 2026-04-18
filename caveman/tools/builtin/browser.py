"""Browser automation tool — @tool-decorated, lifecycle-aware.

Provides browser control as a Caveman tool via the declarative @tool system:
  - Navigate to URLs
  - Take snapshots (accessibility tree)
  - Click, type, interact with elements
  - Screenshot for visual verification

Two backends:
  1. OpenClaw bridge: Use OpenClaw's browser tool via MCP
  2. Direct Playwright: Standalone browser automation (fallback)
"""
from __future__ import annotations
import logging
from typing import Any

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

# ── Module-level state (initialized via set_bridge / ensure_playwright) ──

_bridge = None
_playwright_ctx: dict[str, Any] = {"pw": None, "browser": None, "page": None}


def set_bridge(bridge) -> None:
    """Set the OpenClaw bridge for browser operations."""
    global _bridge
    _bridge = bridge


async def _ensure_playwright() -> Any:
    """Lazy-init Playwright, return page. Raises on missing dependency."""
    if _playwright_ctx["page"]:
        return _playwright_ctx["page"]
    try:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        _playwright_ctx.update(pw=pw, browser=browser, page=page)
        return page
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        )


async def close_browser() -> None:
    """Close browser resources. Called by lifecycle shutdown."""
    if _playwright_ctx["browser"]:
        await _playwright_ctx["browser"].close()
        _playwright_ctx["browser"] = None
    if _playwright_ctx["pw"]:
        await _playwright_ctx["pw"].stop()
        _playwright_ctx["pw"] = None
    _playwright_ctx["page"] = None


def _mode() -> str:
    return "bridge" if _bridge else "standalone"


# ── Bridge helpers ──

async def _bridge_call(action: str, **kwargs) -> dict[str, Any]:
    """Call OpenClaw browser tool via bridge."""
    try:
        result = await _bridge.call_tool("browser", {"action": action, **kwargs})
        return {"ok": True, "data": result.get("result", "")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── The @tool-decorated entry point ──

@tool(
    name="browser",
    description="Browser automation: navigate, snapshot, click, type, screenshot, evaluate",
    params={
        "action": {
            "type": "string",
            "enum": ["navigate", "snapshot", "click", "type", "screenshot", "evaluate", "close"],
            "description": "Browser action to perform",
        },
        "url": {"type": "string", "description": "URL for navigate"},
        "ref": {"type": "string", "description": "Element reference for click/type"},
        "text": {"type": "string", "description": "Text for type action"},
        "js": {"type": "string", "description": "JavaScript for evaluate"},
        "full_page": {"type": "boolean", "description": "Full page screenshot"},
        "compact": {"type": "boolean", "description": "Compact snapshot mode"},
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
) -> dict:
    """Dispatch browser action by name. Works with bridge or standalone Playwright."""
    dispatch_table = {
        "navigate": lambda: _do_navigate(url),
        "snapshot": lambda: _do_snapshot(compact),
        "click": lambda: _do_click(ref),
        "type": lambda: _do_type(ref, text),
        "screenshot": lambda: _do_screenshot(full_page),
        "evaluate": lambda: _do_evaluate(js),
        "close": lambda: _do_close(),
    }
    handler = dispatch_table.get(action)
    if not handler:
        return {"ok": False, "error": f"Unknown browser action: {action}"}
    return await handler()


# ── Action implementations (bridge-first, Playwright fallback) ──

async def _do_navigate(url: str) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("navigate", url=url)
    page = await _ensure_playwright()
    try:
        from caveman.paths import BROWSER_NAV_TIMEOUT
        await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_NAV_TIMEOUT)
        return {"ok": True, "url": page.url, "title": await page.title()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_snapshot(compact: bool) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("snapshot", compact=compact)
    page = await _ensure_playwright()
    try:
        tree = await page.accessibility.snapshot()
        return {"ok": True, "data": tree}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_click(ref: str) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("act", kind="click", ref=ref)
    page = await _ensure_playwright()
    try:
        from caveman.paths import BROWSER_CLICK_TIMEOUT
        await page.click(ref, timeout=BROWSER_CLICK_TIMEOUT)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_type(ref: str, text: str) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("act", kind="type", ref=ref, text=text)
    page = await _ensure_playwright()
    try:
        await page.fill(ref, text)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_screenshot(full_page: bool) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("screenshot", fullPage=full_page)
    page = await _ensure_playwright()
    try:
        buf = await page.screenshot(full_page=full_page)
        return {"ok": True, "size": len(buf)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_evaluate(js: str) -> dict:
    if _mode() == "bridge":
        return await _bridge_call("act", kind="evaluate", fn=js)
    page = await _ensure_playwright()
    try:
        result = await page.evaluate(js)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_close() -> dict:
    await close_browser()
    return {"ok": True}
