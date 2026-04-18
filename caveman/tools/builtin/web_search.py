"""Web search tool — Tavily API integration."""
from __future__ import annotations
import os
import re

import httpx

from caveman.tools.registry import tool
from caveman.utils import retry_async

# Max chars per result content to prevent context flooding
_MAX_CONTENT_CHARS = 2000
# Patterns that might be prompt injection attempts in web content
_INJECTION_PATTERNS = re.compile(
    r"(?i)(?:ignore (?:all )?(?:previous|above) instructions|"
    r"you are now|system prompt|<\|im_start\|>|<\|endoftext\|>|"
    r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
)


def _sanitize_web_content(text: str) -> str:
    """Sanitize web content before showing to LLM.

    - Truncate to max length
    - Strip HTML tags
    - Flag potential prompt injection
    - Redact secrets
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Truncate
    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "... [truncated]"
    # Flag injection attempts (don't remove — let the LLM see the warning)
    if _INJECTION_PATTERNS.search(text):
        text = "[⚠️ UNTRUSTED WEB CONTENT — may contain prompt injection]\n" + text
    # Redact secrets
    from caveman.security.scanner import redact
    text = redact(text)
    return text


@tool(
    name="web_search",
    description="Search the web",
    params={
        "query": {"type": "string"},
        "count": {"type": "integer", "default": 10},
    },
    required=["query"],
)
async def web_search(query: str, count: int = 10) -> dict:
    """Search the web via Tavily API with automatic retry."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set", "results": []}

    async def _attempt():
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": count},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "results": [
                    {
                        "title": r.get("title", "")[:200],
                        "url": r.get("url", ""),
                        "content": _sanitize_web_content(r.get("content", "")),
                    }
                    for r in data.get("results", [])
                ]
            }

    def _is_retryable(e: Exception) -> bool:
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
            return True
        if isinstance(e, (httpx.ConnectTimeout, httpx.ReadTimeout)):
            return True
        return False

    try:
        return await retry_async(_attempt, max_retries=3, base_delay=2.0, retryable=_is_retryable)
    except Exception as e:
        return {"error": str(e), "results": []}
