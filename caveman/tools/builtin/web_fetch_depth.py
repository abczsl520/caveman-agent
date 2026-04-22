"""Web Fetch Depth — readability, PDF, robots.txt, caching, crawl, multi-backend.

Supplements web_fetch_v2.py with deeper web extraction capabilities.
Extracted from Hermes web_tools.py (2103 lines).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

__all__ = ["SearchBackend", "SEARCH_BACKENDS", "get_available_backend", "check_url_secrets", "check_robots_txt", "clean_base64_images", "extract_readable_content", "cache_key", "get_cached", "set_cached", "cleanup_cache", "crawl_site", "parallel_extract"]


logger = logging.getLogger("caveman.tools.web_fetch_depth")

# ── Config ──

_CACHE_DIR = Path.home() / ".caveman" / "web_cache"
_CACHE_TTL = 3600  # 1 hour
_MAX_CONTENT_LENGTH = 500_000  # 500KB
_MIN_LENGTH_FOR_SUMMARIZATION = 5000
_SECRET_PREFIX_RE = re.compile(
    r"(sk-[a-zA-Z0-9]{20}|ghp_[a-zA-Z0-9]{36}|xoxb-|Bearer\s+[a-zA-Z0-9._-]{20,})",
    re.IGNORECASE,
)


# ── Multi-Backend Search ──

@dataclass
class SearchBackend:
    """Configuration for a search backend."""
    name: str
    api_key_env: str = ""
    base_url: str = ""
    priority: int = 0

    @property
    def is_available(self) -> bool:
        if self.api_key_env:
            return bool(os.environ.get(self.api_key_env))
        return True


SEARCH_BACKENDS = [
    SearchBackend("tavily", "TAVILY_API_KEY", "https://api.tavily.com"),
    SearchBackend("exa", "EXA_API_KEY", "https://api.exa.ai"),
    SearchBackend("firecrawl", "FIRECRAWL_API_KEY", "https://api.firecrawl.dev"),
]


def get_available_backend() -> Optional[SearchBackend]:
    """Get the first available search backend."""
    for backend in SEARCH_BACKENDS:
        if backend.is_available:
            return backend
    return None


# ── URL Safety ──

def check_url_secrets(url: str) -> bool:
    """Check if URL contains embedded secrets (exfiltration prevention)."""
    decoded = unquote(url)
    return bool(_SECRET_PREFIX_RE.search(url) or _SECRET_PREFIX_RE.search(decoded))


# ── Robots.txt ──

_robots_cache: Dict[str, Tuple[float, bool]] = {}


def check_robots_txt(url: str, user_agent: str = "CavemanBot") -> bool:
    """Check if URL is allowed by robots.txt. Returns True if allowed."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    # Check cache
    cache_key = f"{parsed.netloc}:{user_agent}"
    if cache_key in _robots_cache:
        ts, allowed = _robots_cache[cache_key]
        if time.time() - ts < 3600:
            return allowed

    try:
        import urllib.request
        req = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read().decode("utf-8", errors="replace")

        # Simple robots.txt parser
        allowed = True
        current_agent_matches = False
        for line in content.split("\n"):
            line = line.strip().lower()
            if line.startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                current_agent_matches = agent == "*" or agent == user_agent.lower()
            elif current_agent_matches and line.startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path and parsed.path.startswith(path):
                    allowed = False
                    break

        _robots_cache[cache_key] = (time.time(), allowed)
        return allowed

    except Exception:
        # If we can't fetch robots.txt, assume allowed
        _robots_cache[cache_key] = (time.time(), True)
        return True


# ── Content Cleaning ──

_BASE64_IMG_RE = re.compile(
    r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]{100,}", re.IGNORECASE,
)


def clean_base64_images(text: str) -> str:
    """Remove inline base64 images from content."""
    return _BASE64_IMG_RE.sub("[base64-image-removed]", text)


def extract_readable_content(html: str) -> str:
    """Extract readable content from HTML using simple heuristics."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Remove nav, header, footer, sidebar
    for tag in ("nav", "header", "footer", "aside"):
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert common elements to markdown
    html = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "#" * int(m.group(1)) + " " + m.group(2) + "\n\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove remaining tags
    html = re.sub(r"<[^>]+>", "", html)

    # Clean up whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r" {2,}", " ", html)

    return html.strip()


# ── Caching ──

def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_cached(url: str) -> Optional[str]:
    """Get cached content for URL."""
    key = cache_key(url)
    cache_file = _CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) > _CACHE_TTL:
            cache_file.unlink(missing_ok=True)
            return None
        return data.get("content")
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None


def set_cached(url: str, content: str) -> None:
    """Cache content for URL."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(url)
    cache_file = _CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(
            json.dumps({"url": url, "ts": time.time(), "content": content[:_MAX_CONTENT_LENGTH]}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("set_cached: suppressed %s", exc)


def cleanup_cache(max_age_hours: int = 24) -> int:
    """Remove old cache entries. Returns count removed."""
    if not _CACHE_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in _CACHE_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass  # intentional: Exception suppressed
    return removed


# ── Crawl ──

async def crawl_site(
    url: str,
    max_pages: int = 20,
    depth: str = "basic",
    instructions: str = "",
    summarize: bool = True,
    summarize_fn: Optional[Any] = None,
    min_length: int = _MIN_LENGTH_FOR_SUMMARIZATION,
) -> List[Dict[str, Any]]:
    """Crawl a website and extract content from multiple pages."""
    from caveman.tools.builtin.web_tools_v2 import _is_ssrf_safe

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"

    if not _is_ssrf_safe(url):
        return [{"url": url, "error": "Blocked: private/internal address"}]

    # Check robots.txt
    if not check_robots_txt(url):
        return [{"url": url, "error": "Blocked by robots.txt"}]

    # Check for secrets in URL
    if check_url_secrets(url):
        return [{"url": url, "error": "Blocked: URL contains embedded secrets"}]

    backend = get_available_backend()
    if not backend:
        return [{"url": url, "error": "No search backend available"}]

    if backend.name == "tavily":
        return await _crawl_tavily(url, max_pages, depth, backend)
    elif backend.name == "firecrawl":
        return await _crawl_firecrawl(url, max_pages, depth, instructions, backend)

    return [{"url": url, "error": f"Backend {backend.name} does not support crawl"}]


async def _crawl_tavily(
    url: str, max_pages: int, depth: str, backend: SearchBackend,
) -> List[Dict[str, Any]]:
    """Crawl via Tavily API."""
    import urllib.request
    api_key = os.environ.get(backend.api_key_env, "")
    payload = json.dumps({
        "url": url,
        "limit": max_pages,
        "extract_depth": depth,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        req = urllib.request.Request(
            f"{backend.base_url}/crawl",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        return [
            {
                "url": r.get("url", url),
                "title": r.get("title", ""),
                "content": r.get("raw_content", r.get("content", "")),
            }
            for r in results
        ]
    except Exception as e:
        return [{"url": url, "error": f"Tavily crawl failed: {e}"}]


async def _crawl_firecrawl(
    url: str, max_pages: int, depth: str, instructions: str, backend: SearchBackend,
) -> List[Dict[str, Any]]:
    """Crawl via Firecrawl API."""
    import urllib.request
    api_key = os.environ.get(backend.api_key_env, "")
    payload = json.dumps({
        "url": url,
        "limit": max_pages,
        "scrapeOptions": {"formats": ["markdown"]},
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        req = urllib.request.Request(
            f"{backend.base_url}/v1/crawl",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        # Firecrawl returns async job — poll for results
        job_id = data.get("id", "")
        if not job_id:
            return [{"url": url, "error": "Firecrawl returned no job ID"}]

        # Poll (up to 120s)
        for _ in range(24):
            await asyncio.sleep(5)
            poll_req = urllib.request.Request(
                f"{backend.base_url}/v1/crawl/{job_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(poll_req, timeout=15) as resp:
                status_data = json.loads(resp.read())
            if status_data.get("status") == "completed":
                return [
                    {
                        "url": r.get("metadata", {}).get("sourceURL", url),
                        "title": r.get("metadata", {}).get("title", ""),
                        "content": r.get("markdown", ""),
                    }
                    for r in status_data.get("data", [])
                ]
        return [{"url": url, "error": "Firecrawl crawl timed out"}]
    except Exception as e:
        return [{"url": url, "error": f"Firecrawl crawl failed: {e}"}]


# ── Parallel Extract ──

async def parallel_extract(
    urls: List[str],
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Extract content from multiple URLs in parallel."""
    from caveman.tools.builtin.web_tools_v2 import _is_ssrf_safe

    async def _extract_one(url: str) -> Dict[str, Any]:
        if check_url_secrets(url):
            return {"url": url, "error": "Blocked: embedded secrets"}
        if not _is_ssrf_safe(url):
            return {"url": url, "error": "Blocked: private address"}

        # Check cache first
        cached = get_cached(url)
        if cached:
            return {"url": url, "content": cached, "cached": True}

        try:
            import urllib.request
            req = urllib.request.Request(url, headers={
                "User-Agent": "CavemanBot/1.0",
                "Accept": "text/html,application/xhtml+xml,*/*",
            })
            loop = asyncio.get_running_loop()
            with await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=15)) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()

            if "pdf" in content_type.lower():
                return {"url": url, "content": "[PDF content — extraction not supported]", "type": "pdf"}

            text = raw.decode("utf-8", errors="replace")
            if "html" in content_type.lower():
                text = extract_readable_content(text)

            text = clean_base64_images(text)
            if len(text) > _MAX_CONTENT_LENGTH:
                text = text[:_MAX_CONTENT_LENGTH] + "\n\n[content truncated]"

            set_cached(url, text)
            return {"url": url, "content": text}

        except Exception as e:
            return {"url": url, "error": str(e)}

    tasks = [asyncio.wait_for(_extract_one(u), timeout=timeout) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if isinstance(r, dict) else {"url": urls[i], "error": str(r)}
        for i, r in enumerate(results)
    ]
