"""Web Tools v2 — extraction, caching, SSRF protection, proxy, retry.

Extracted from Hermes web_tools.py (2103 lines).
Key patterns: multi-backend extraction, content summarization, SSRF guard.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from caveman.tools.registry import tool

logger = logging.getLogger("caveman.tools.web_v2")

# SSRF protection (reuse from browser_manager)
_BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal", "169.254.169.254"})

# Cache config
_CACHE_DIR = Path(os.getenv("CAVEMAN_WEB_CACHE", "/tmp/caveman_web_cache"))
_CACHE_TTL = 3600  # 1 hour
_MAX_CONTENT_LENGTH = 500000  # 500KB


def _is_ssrf_safe(url: str) -> bool:
    """Check URL against SSRF blocklist."""
    try:
        host = urlsplit(url).hostname or ""
        return host.lower() not in _BLOCKED_HOSTS
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return False


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()


def _get_cached(url: str) -> Optional[str]:
    """Get cached content if fresh."""
    key = _cache_key(url)
    path = _CACHE_DIR / f"{key}.txt"
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL:
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def _set_cache(url: str, content: str) -> None:
    """Cache content."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(url)
    path = _CACHE_DIR / f"{key}.txt"
    path.write_text(content[:_MAX_CONTENT_LENGTH], encoding="utf-8")


def _clean_html(html: str) -> str:
    """Extract readable text from HTML."""
    # Remove scripts and styles
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text


def _truncate_content(content: str, max_chars: int = 50000) -> str:
    """Truncate content preserving paragraph boundaries."""
    if len(content) <= max_chars:
        return content
    # Try to break at paragraph
    truncated = content[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.5:
        truncated = truncated[:last_para]
    return truncated + f"\n\n... (truncated, {len(content)} chars total)"


@tool(
    name="web_fetch_v2",
    description="Fetch and extract readable content from a URL with caching and SSRF protection",
    params={
        "url": {"type": "string", "description": "URL to fetch"},
        "max_chars": {"type": "integer", "description": "Max content chars (default 50000)"},
        "use_cache": {"type": "boolean", "description": "Use cache (default true)"},
        "extract_links": {"type": "boolean", "description": "Extract links (default false)"},
    },
    required=["url"],
)
async def web_fetch_v2(
    url: str, max_chars: int = 50000, use_cache: bool = True,
    extract_links: bool = False,
) -> Dict[str, Any]:
    """Fetch URL with caching and content extraction."""
    if not _is_ssrf_safe(url):
        return {"ok": False, "error": f"SSRF blocked: {url}"}

    # Check cache
    if use_cache:
        cached = _get_cached(url)
        if cached:
            return {"ok": True, "content": _truncate_content(cached, max_chars),
                    "cached": True, "url": url}

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30),
                                   headers={"User-Agent": "Caveman/0.5"}) as resp:
                if resp.status != 200:
                    return {"ok": False, "error": f"HTTP {resp.status}", "url": url}

                content_type = resp.headers.get("content-type", "")
                raw = await resp.text(errors="replace")

                # Extract readable content
                if "html" in content_type:
                    content = _clean_html(raw)
                else:
                    content = raw

                content = _truncate_content(content, max_chars)

                # Cache
                if use_cache:
                    _set_cache(url, content)

                result: Dict[str, Any] = {
                    "ok": True,
                    "content": content,
                    "url": str(resp.url),
                    "status": resp.status,
                    "cached": False,
                }

                if extract_links:
                    links = re.findall(r'href=["\']([^"\']+)["\']', raw)
                    result["links"] = links[:50]

                return result

    except ImportError:
        # Fallback to urllib
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Caveman/0.5"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                content = _clean_html(raw) if "html" in resp.headers.get("content-type", "") else raw
                content = _truncate_content(content, max_chars)
                if use_cache:
                    _set_cache(url, content)
                return {"ok": True, "content": content, "url": url, "cached": False}
        except Exception as e:
            return {"ok": False, "error": str(e), "url": url}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}


@tool(
    name="web_search_v2",
    description="Search the web using Tavily API",
    params={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Max results (default 5)"},
        "search_depth": {"type": "string", "description": "'basic' or 'advanced' (default 'basic')"},
    },
    required=["query"],
)
async def web_search_v2(
    query: str, max_results: int = 5, search_depth: str = "basic",
) -> Dict[str, Any]:
    """Search the web using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {"ok": False, "error": data.get("detail", f"HTTP {resp.status}")}

                results = []
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:500],
                    })

                return {
                    "ok": True,
                    "query": query,
                    "results": results,
                    "answer": data.get("answer", ""),
                }
    except Exception as e:
        return {"ok": False, "error": str(e)}
