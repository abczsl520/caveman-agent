"""Browser Manager — multi-tab, cookie, SSRF protection, Canvas support.

Extends the basic browser tool with production-grade features:
- Multi-tab management (open/close/focus/list)
- Cookie persistence
- SSRF protection (block internal IPs)
- PDF save
- Console log capture
- Screenshot with base64 encoding

Learned from: OpenClaw extensions/browser (3450 lines)
Our version: Pure Python, Playwright-native, async.
"""
from __future__ import annotations

import base64
import ipaddress
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from caveman.aio import aio_exists, aio_mkdir, aio_read_text, aio_write_text

__all__ = ["is_ssrf_safe", "BrowserManager"]


logger = logging.getLogger("caveman.browser")

# ── SSRF Protection ──

_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "169.254.169.254"}
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
]


def is_ssrf_safe(url: str) -> bool:
    """Check if URL is safe from SSRF attacks."""
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if host.lower() in _BLOCKED_HOSTS:
            return False
        try:
            addr = ipaddress.ip_address(host)
            return not any(addr in net for net in _PRIVATE_RANGES)
        except ValueError:
            pass  # Not an IP, it's a hostname — allow
        return True
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return False


# ── Browser Manager ──

class BrowserManager:
    """Multi-tab browser manager with Playwright."""

    def __init__(self, headless: bool = True, ssrf_protection: bool = True):
        self._headless = headless
        self._ssrf_protection = ssrf_protection
        self._pw: Any = None
        self._browser: Any = None
        self._pages: Dict[str, Any] = {}  # tab_id → page
        self._active_tab: str = ""
        self._console_logs: Dict[str, List[str]] = {}
        self._cookies_path: Optional[Path] = None

    @property
    def active_page(self) -> Any:
        return self._pages.get(self._active_tab)

    @property
    def tab_count(self) -> int:
        return len(self._pages)

    async def start(self, cookies_path: Optional[str] = None) -> bool:
        """Launch browser. Returns True on success."""
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=self._headless)

            if cookies_path:
                self._cookies_path = Path(cookies_path)

            # Create initial tab
            await self.new_tab("main")

            # Restore cookies if available
            if self._cookies_path and await aio_exists(self._cookies_path):
                await self._load_cookies()

            logger.info("Browser started (headless=%s)", self._headless)
            return True
        except ImportError:
            logger.error("Playwright not installed: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            logger.error("Browser start failed: %s", e)
            return False

    async def stop(self) -> None:
        """Close browser and all tabs."""
        if self._cookies_path:
            await self._save_cookies()
        for tab_id in list(self._pages.keys()):
            try:
                await self._pages[tab_id].close()
            except Exception:
                pass  # intentional: Exception suppressed
        self._pages.clear()
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None

    # ── Tab Management ──

    async def new_tab(self, tab_id: str = "") -> str:
        """Open a new tab. Returns tab_id."""
        if not self._browser:
            raise RuntimeError("Browser not started")
        if not tab_id:
            tab_id = f"tab-{len(self._pages)}"
        context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
        page = await context.new_page()

        # Capture console logs
        self._console_logs[tab_id] = []
        page.on("console", lambda msg, tid=tab_id: self._console_logs.get(tid, []).append(
            f"[{msg.type}] {msg.text}"
        ))

        self._pages[tab_id] = page
        self._active_tab = tab_id
        return tab_id

    async def close_tab(self, tab_id: str) -> bool:
        page = self._pages.pop(tab_id, None)
        if not page:
            return False
        self._console_logs.pop(tab_id, None)
        await page.close()
        if self._active_tab == tab_id:
            self._active_tab = next(iter(self._pages), "")
        return True

    def focus_tab(self, tab_id: str) -> bool:
        if tab_id in self._pages:
            self._active_tab = tab_id
            return True
        return False

    def list_tabs(self) -> List[Dict[str, str]]:
        return [
            {"tab_id": tid, "url": p.url, "active": tid == self._active_tab}
            for tid, p in self._pages.items()
        ]

    # ── Navigation ──

    async def navigate(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        if self._ssrf_protection and not is_ssrf_safe(url):
            return {"ok": False, "error": f"SSRF blocked: {url}"}
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            return {"ok": True, "url": page.url, "title": await page.title()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Interaction ──

    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            await page.click(selector, timeout=timeout)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def fill(self, selector: str, text: str) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            await page.fill(selector, text)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def press(self, key: str) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            await page.keyboard.press(key)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Snapshot & Screenshot ──

    async def snapshot(self, compact: bool = True) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            tree = await page.accessibility.snapshot()
            return {"ok": True, "data": tree, "url": page.url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def screenshot(
        self, full_page: bool = False, path: Optional[str] = None,
    ) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            buf = await page.screenshot(full_page=full_page, path=path)
            return {
                "ok": True,
                "size": len(buf),
                "base64": base64.b64encode(buf).decode() if not path else None,
                "path": path,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def pdf(self, path: str) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            buf = await page.pdf(path=path)
            return {"ok": True, "path": path, "size": len(buf) if buf else 0}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── JavaScript ──

    async def evaluate(self, js: str) -> Dict[str, Any]:
        page = self.active_page
        if not page:
            return {"ok": False, "error": "No active tab"}
        try:
            result = await page.evaluate(js)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Console ──

    def get_console_logs(self, tab_id: str = "") -> List[str]:
        tid = tab_id or self._active_tab
        return self._console_logs.get(tid, [])

    def clear_console_logs(self, tab_id: str = "") -> None:
        tid = tab_id or self._active_tab
        if tid in self._console_logs:
            self._console_logs[tid].clear()

    # ── Cookies ──

    async def _save_cookies(self) -> None:
        if not self._cookies_path or not self._browser or not self._browser.contexts:
            return
        try:
            import json
            cookies = await self._browser.contexts[0].cookies()
            await aio_mkdir(self._cookies_path.parent, parents=True, exist_ok=True)
            await aio_write_text(self._cookies_path, json.dumps(cookies, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to save cookies", exc_info=True)

    async def _load_cookies(self) -> None:
        if not self._cookies_path or not await aio_exists(self._cookies_path):
            return
        try:
            import json
            cookies = json.loads(await aio_read_text(self._cookies_path, encoding="utf-8"))
            if self._browser and self._browser.contexts:
                await self._browser.contexts[0].add_cookies(cookies)
        except Exception:
            logger.debug("Failed to load cookies", exc_info=True)
