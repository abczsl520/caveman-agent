"""Link understanding — auto-detect URLs in messages and fetch content.

When a user sends a message containing URLs, this module:
1. Extracts URLs from the message text
2. Fetches readable content from each URL (with timeout + size limits)
3. Injects the content as context for the agent

Integrates with the agent loop's message preprocessing pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "DEFAULT_MAX_LINKS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_CONTENT",
    "extract_urls",
    "format_link_context",
    "fetch_url_content",
    "understand_links",
]


logger = logging.getLogger(__name__)

# Strip markdown link syntax to avoid double-matching
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*]\((https?://\S+?)\)", re.I)
_BARE_URL_RE = re.compile(r"https?://\S+", re.I)

# Domains to skip (images, videos, etc. that aren't readable text)
_SKIP_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".webm", ".avi", ".mov", ".mp3", ".wav", ".ogg",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dmg", ".apk", ".deb",
})

_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",
})

DEFAULT_MAX_LINKS = 3
DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_MAX_CONTENT = 8000  # chars per URL


def extract_urls(message: str, max_links: int = DEFAULT_MAX_LINKS) -> list[str]:
    """Extract unique URLs from a message, respecting limits.

    Strips markdown link syntax first, then finds bare URLs.
    Filters out image/video/binary URLs and blocked hosts.
    """
    if not message or not message.strip():
        return []

    # Collect markdown link targets first
    urls: list[str] = []
    seen: set[str] = set()

    for match in _MARKDOWN_LINK_RE.finditer(message):
        url = _clean_url(match.group(1))
        if url and url not in seen and _is_fetchable(url):
            urls.append(url)
            seen.add(url)

    # Remove markdown links from text, then find bare URLs
    bare_text = _MARKDOWN_LINK_RE.sub(" ", message)
    for match in _BARE_URL_RE.finditer(bare_text):
        url = _clean_url(match.group(0))
        if url and url not in seen and _is_fetchable(url):
            urls.append(url)
            seen.add(url)

    return urls[:max_links]


def format_link_context(results: list[dict[str, Any]]) -> str:
    """Format fetched link content for injection into agent context.

    Returns a string block suitable for prepending to the user message.
    """
    if not results:
        return ""

    parts = ["[Link content auto-fetched by Caveman]"]
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        content = r.get("content", "")
        error = r.get("error")

        if error:
            parts.append(f"\n--- {url} ---\n(fetch failed: {error})")
        elif content:
            header = f"\n--- {title or url} ---" if title else f"\n--- {url} ---"
            parts.append(f"{header}\n{content}")

    parts.append("\n[End of link content]\n")
    return "\n".join(parts)


async def fetch_url_content(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_content: int = DEFAULT_MAX_CONTENT,
) -> dict[str, Any]:
    """Fetch readable content from a URL.

    Returns dict with keys: url, title, content, error.
    Uses httpx for async HTTP. Extracts text via simple HTML stripping.
    """
    import httpx

    result: dict[str, Any] = {"url": url, "title": "", "content": "", "error": None}

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Caveman-Agent/1.0 (link-understanding)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                result["error"] = f"non-text content: {content_type.split(';')[0]}"
                return result

            html = resp.text
            title, text = _extract_readable(html)
            result["title"] = title
            result["content"] = text[:max_content]

    except httpx.TimeoutException:
        result["error"] = "timeout"
    except httpx.HTTPStatusError as e:
        result["error"] = f"HTTP {e.response.status_code}"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


async def understand_links(
    message: str,
    max_links: int = DEFAULT_MAX_LINKS,
    timeout: int = DEFAULT_TIMEOUT,
    max_content: int = DEFAULT_MAX_CONTENT,
) -> tuple[list[str], str]:
    """Full pipeline: extract URLs → fetch content → format context.

    Returns (urls_found, context_string).
    """
    urls = extract_urls(message, max_links=max_links)
    if not urls:
        return [], ""

    import asyncio
    tasks = [fetch_url_content(url, timeout=timeout, max_content=max_content) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            valid_results.append({"url": "unknown", "error": str(r)[:200]})
        else:
            valid_results.append(r)

    context = format_link_context(valid_results)
    return urls, context


def _clean_url(url: str) -> str:
    """Strip trailing punctuation that's likely not part of the URL."""
    # Remove trailing ), ], >, comma, period, semicolon
    url = url.rstrip(")],>;.")
    # Remove trailing quote marks
    url = url.strip("'\"")
    return url


def _is_fetchable(url: str) -> bool:
    """Check if a URL is worth fetching (not an image, not blocked, etc.)."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTS:
        return False

    # Check for private IP ranges
    if hostname.startswith("10.") or hostname.startswith("192.168."):
        return False

    # Check file extension
    path_lower = parsed.path.lower()
    for ext in _SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False

    return True


def _extract_readable(html: str) -> tuple[str, str]:
    """Extract title and readable text from HTML. Simple regex-based."""
    import re as _re

    # Title
    title_match = _re.search(r"<title[^>]*>(.*?)</title>", html, _re.I | _re.S)
    title = title_match.group(1).strip() if title_match else ""
    # Decode HTML entities in title
    title = _decode_entities(title)

    # Remove script, style, nav, header, footer
    text = _re.sub(r"<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>", "", html, flags=_re.I | _re.S)
    # Remove HTML tags
    text = _re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    text = _decode_entities(text)
    # Collapse whitespace
    text = _re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing noise
    text = text.strip()

    return title, text


def _decode_entities(text: str) -> str:
    """Decode common HTML entities."""
    replacements = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
        "&nbsp;": " ", "&#x27;": "'", "&#x2F;": "/",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    # Numeric entities
    import re as _re
    text = _re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = _re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    return text
