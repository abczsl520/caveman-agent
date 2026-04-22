"""Tests for BrowserManager — multi-tab, SSRF, cookies."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.tools.builtin.browser_manager import BrowserManager, is_ssrf_safe


class TestSSRFProtection:
    def test_safe_urls(self):
        assert is_ssrf_safe("https://google.com") is True
        assert is_ssrf_safe("https://example.com/path") is True
        assert is_ssrf_safe("http://api.github.com") is True

    def test_blocked_localhost(self):
        assert is_ssrf_safe("http://localhost:8080") is False
        assert is_ssrf_safe("http://127.0.0.1") is False
        assert is_ssrf_safe("http://127.0.0.1:3000/api") is False

    def test_blocked_private_ips(self):
        assert is_ssrf_safe("http://10.0.0.1") is False
        assert is_ssrf_safe("http://172.16.0.1") is False
        assert is_ssrf_safe("http://192.168.1.1") is False

    def test_blocked_metadata(self):
        assert is_ssrf_safe("http://169.254.169.254/latest/meta-data") is False
        assert is_ssrf_safe("http://metadata.google.internal") is False

    def test_empty_url(self):
        assert is_ssrf_safe("") is False

    def test_ipv6_loopback(self):
        assert is_ssrf_safe("http://[::1]") is False


class TestBrowserManager:
    def test_init(self):
        mgr = BrowserManager(headless=True)
        assert mgr.tab_count == 0
        assert mgr.active_page is None

    def test_list_tabs_empty(self):
        mgr = BrowserManager()
        assert mgr.list_tabs() == []

    @pytest.mark.asyncio
    async def test_navigate_ssrf_blocked(self):
        mgr = BrowserManager(ssrf_protection=True)
        # Manually set a mock page so we can test SSRF check
        mgr._pages["main"] = MagicMock()
        mgr._active_tab = "main"
        result = await mgr.navigate("http://169.254.169.254/latest/meta-data")
        assert not result["ok"]
        assert "SSRF" in result["error"]

    @pytest.mark.asyncio
    async def test_navigate_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.navigate("https://example.com")
        assert not result["ok"]
        assert "No active tab" in result["error"]

    @pytest.mark.asyncio
    async def test_click_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.click("#btn")
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_fill_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.fill("#input", "text")
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_snapshot_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.snapshot()
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_screenshot_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.screenshot()
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_evaluate_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.evaluate("1+1")
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_pdf_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.pdf("/tmp/test.pdf")
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_press_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.press("Enter")
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_scroll_no_tab(self):
        mgr = BrowserManager()
        result = await mgr.scroll("down")
        assert not result["ok"]

    def test_focus_tab_nonexistent(self):
        mgr = BrowserManager()
        assert mgr.focus_tab("nonexistent") is False

    @pytest.mark.asyncio
    async def test_close_tab_nonexistent(self):
        mgr = BrowserManager()
        assert await mgr.close_tab("nonexistent") is False

    def test_console_logs_empty(self):
        mgr = BrowserManager()
        assert mgr.get_console_logs() == []

    @pytest.mark.asyncio
    async def test_start_no_playwright(self):
        with patch("caveman.tools.builtin.browser_manager.BrowserManager.start") as mock_start:
            mock_start.return_value = False
            mgr = BrowserManager()
            result = await mgr.start()
            assert result is False
