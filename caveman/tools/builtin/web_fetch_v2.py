"""Web Tool v2 — comprehensive web interaction with multiple backends.

Extracted from Hermes web_tool.py (2100 lines).
Supports: Firecrawl, Jina Reader, raw HTTP, readability extraction.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.tools.registry import tool
from caveman.tools.builtin.web_tools_v2 import _is_ssrf_safe as check_ssrf, _clean_html as clean_html, _truncate_content as truncate_content

logger = logging.getLogger("caveman.tools.web_v2")

# Cache
_WEB_CACHE_DIR = Path.home() / ".caveman" / "web_cache"
_CACHE_TTL = 3600  # 1 hour


@dataclass
class WebResult:
    """Result of a web fetch."""
    ok: bool = True
    url: str = ""
    title: str = ""
    content: str = ""
    content_type: str = ""
    status_code: int = 0
    error: str = ""
    cached: bool = False
    backend: str = ""
    fetch_ms: float = 0


# ── Cache ──

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _get_cached(url: str) -> Optional[WebResult]:
    key = _cache_key(url)
    path = _WEB_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("cached_at", 0) > _CACHE_TTL:
            path.unlink(missing_ok=True)
            return None
        return WebResult(
            ok=True, url=url, title=data.get("title", ""),
            content=data.get("content", ""), cached=True,
            backend=data.get("backend", "cache"),
        )
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None


def _save_cache(url: str, result: WebResult) -> None:
    _WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(url)
    path = _WEB_CACHE_DIR / f"{key}.json"
    data = {
        "url": url, "title": result.title, "content": result.content[:50000],
        "backend": result.backend, "cached_at": time.time(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


# ── Backends ──

async def _fetch_raw(url: str, timeout: float = 15) -> WebResult:
    """Raw HTTP fetch with readability extraction."""
    start = time.monotonic()
    try:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; Caveman/0.5)",
                "Accept": "text/html,application/xhtml+xml,*/*",
            })
            content_type = resp.headers.get("content-type", "")
            text = resp.text

            if "html" in content_type:
                text = clean_html(text)

            return WebResult(
                ok=resp.status_code < 400,
                url=str(resp.url),
                content=truncate_content(text),
                content_type=content_type,
                status_code=resp.status_code,
                backend="raw",
                fetch_ms=(time.monotonic() - start) * 1000,
            )
    except Exception as e:
        return WebResult(ok=False, url=url, error=str(e), backend="raw",
                         fetch_ms=(time.monotonic() - start) * 1000)


async def _fetch_jina(url: str, timeout: float = 15) -> WebResult:
    """Fetch via Jina Reader API (free, no key needed)."""
    start = time.monotonic()
    try:
        import httpx
        reader_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(reader_url, headers={
                "Accept": "text/plain",
                "X-Return-Format": "text",
            })
            if resp.status_code == 200:
                content = resp.text
                # Extract title from first line
                lines = content.split("\n", 2)
                title = lines[0].strip("# ").strip() if lines else ""
                return WebResult(
                    ok=True, url=url, title=title,
                    content=truncate_content(content),
                    backend="jina",
                    fetch_ms=(time.monotonic() - start) * 1000,
                )
            return WebResult(ok=False, url=url, error=f"Jina returned {resp.status_code}",
                             backend="jina", fetch_ms=(time.monotonic() - start) * 1000)
    except Exception as e:
        return WebResult(ok=False, url=url, error=str(e), backend="jina",
                         fetch_ms=(time.monotonic() - start) * 1000)


async def _fetch_firecrawl(url: str, api_key: str = "", timeout: float = 30) -> WebResult:
    """Fetch via Firecrawl API."""
    key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return WebResult(ok=False, url=url, error="No Firecrawl API key", backend="firecrawl")

    start = time.monotonic()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {key}"},
                json={"url": url, "formats": ["markdown"]},
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("data", {}).get("markdown", "")
                title = data.get("data", {}).get("metadata", {}).get("title", "")
                return WebResult(
                    ok=True, url=url, title=title,
                    content=truncate_content(content),
                    backend="firecrawl",
                    fetch_ms=(time.monotonic() - start) * 1000,
                )
            return WebResult(ok=False, url=url, error=f"Firecrawl returned {resp.status_code}",
                             backend="firecrawl", fetch_ms=(time.monotonic() - start) * 1000)
    except Exception as e:
        return WebResult(ok=False, url=url, error=str(e), backend="firecrawl",
                         fetch_ms=(time.monotonic() - start) * 1000)


# ── Main Tool ──

@tool(
    name="web_fetch",
    description="Fetch and extract readable content from a URL",
    params={
        "url": {"type": "string", "description": "URL to fetch"},
        "backend": {"type": "string", "description": "Backend: auto/raw/jina/firecrawl"},
    },
    required=["url"],
)
async def web_fetch(url: str, backend: str = "auto") -> Dict[str, Any]:
    """Fetch a URL with automatic backend selection and caching."""
    # SSRF check
    if not check_ssrf(url):
        return {"ok": False, "error": "URL blocked by SSRF protection"}

    # Secrets exfiltration check
    if check_url_secrets(url):
        return {"ok": False, "url": url, "error": "URL contains embedded secrets — blocked for safety"}

    # Cache check
    cached = _get_cached(url)
    if cached:
        return {
            "ok": True, "url": url, "title": cached.title,
            "content": cached.content, "cached": True,
        }

    # Backend selection
    result = None
    if backend == "auto":
        # Try Jina first (free, good quality), fall back to raw
        result = await _fetch_jina(url)
        if not result.ok:
            result = await _fetch_raw(url)
    elif backend == "jina":
        result = await _fetch_jina(url)
    elif backend == "firecrawl":
        result = await _fetch_firecrawl(url)
    elif backend == "raw":
        result = await _fetch_raw(url)
    else:
        return {"ok": False, "error": f"Unknown backend: {backend}"}

    # Cache successful results
    if result.ok:
        _save_cache(url, result)

    return {
        "ok": result.ok,
        "url": result.url,
        "title": result.title,
        "content": result.content,
        "error": result.error,
        "backend": result.backend,
        "fetch_ms": round(result.fetch_ms, 1),
    }


# web_search_v2 removed — canonical version lives in web_tools_v2.py

from caveman.timeouts import WEB_FETCH_DEFAULT

from caveman.tools.builtin.web_fetch_depth import (  # noqa: F401  # depth wiring
    SearchBackend, SEARCH_BACKENDS, get_available_backend,
    check_url_secrets, check_robots_txt, clean_base64_images,
    extract_readable_content, cache_key, get_cached, set_cached,
    cleanup_cache, crawl_site, parallel_extract,
)

__all__ = [
    "WebResult",
    "web_fetch",
    # depth re-exports
    "SearchBackend",
    "SEARCH_BACKENDS",
    "get_available_backend",
    "check_url_secrets",
    "check_robots_txt",
    "clean_base64_images",
    "extract_readable_content",
    "cache_key",
    "get_cached",
    "set_cached",
    "cleanup_cache",
    "crawl_site",
    "parallel_extract",
]

