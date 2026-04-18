"""Auto-generate short session titles from the first user/assistant exchange.

Runs after the first response so it never adds latency to the user-facing reply.
Uses the auxiliary client (cheapest model) with heuristic fallback.

Ported from Hermes (MIT, Nous Research), adapted for Caveman.
"""
from __future__ import annotations

__all__ = ["generate_title", "auto_title_session"]

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

# Heuristic patterns for title extraction (no LLM needed)
_INTENT_PATTERNS = [
    # English
    (r"(?:help|how)\s+(?:me\s+)?(?:to\s+)?(.{10,60}?)(?:\?|$)", "Help: {}"),
    (r"(?:create|build|make|implement|write)\s+(.{5,50}?)(?:\.|$)", "Build: {}"),
    (r"(?:fix|debug|solve|resolve)\s+(.{5,50}?)(?:\.|$)", "Fix: {}"),
    (r"(?:explain|what\s+is|tell\s+me\s+about)\s+(.{5,50}?)(?:\?|$)", "About: {}"),
    # Chinese
    (r"(?:帮我|帮忙|请)(.{3,30})", "{}"),
    (r"(?:创建|构建|实现|写|做)(.{3,30})", "构建{}"),
    (r"(?:修复|调试|解决)(.{3,30})", "修复{}"),
    (r"(?:解释|什么是|介绍)(.{3,30})", "关于{}"),
]


def _heuristic_title(user_message: str) -> str | None:
    """Extract a title using regex patterns — no LLM needed."""
    text = user_message.strip()[:200]
    for pattern, template in _INTENT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            topic = m.group(1).strip().rstrip(".,;:!?")
            if len(topic) >= 3:
                return template.format(topic)[:80]
    # Fallback: first sentence or first N words
    first_line = text.split("\n")[0].strip()
    words = first_line.split()
    if 2 <= len(words) <= 8:
        return first_line[:80]
    if len(words) > 8:
        return " ".join(words[:7])[:80]
    return None


def _clean_title(raw: str) -> str | None:
    """Clean up a generated title."""
    title = raw.strip().strip('"\'')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title if title else None


async def generate_title(
    user_message: str,
    assistant_response: str = "",
    use_llm: bool = True,
) -> str | None:
    """Generate a session title from the first exchange.

    Tries LLM first (via auxiliary client), falls back to heuristic.
    """
    # Try heuristic first (fast, free)
    heuristic = _heuristic_title(user_message)

    if not use_llm:
        return heuristic

    # Try LLM via auxiliary client
    try:
        from caveman.agent.auxiliary import AuxiliaryClient
        aux = AuxiliaryClient()
        messages = [{"role": "user", "content": user_message}]
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})
        result = await aux.generate_title(messages)
        if result:
            return _clean_title(result)
    except Exception as e:
        logger.debug("LLM title generation failed: %s", e)

    return heuristic


async def auto_title_session(
    session_store: Any,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> str | None:
    """Generate and set a session title if one doesn't already exist.

    Returns the generated title or None.
    """
    if not session_store or not session_id:
        return None

    # Check if title already exists
    try:
        meta = session_store.load_meta(session_id)
        if meta and meta.title:
            return None
    except Exception as e:
        logger.debug("Suppressed in title_generator: %s", e)

    title = await generate_title(user_message, assistant_response)
    if not title:
        return None

    try:
        meta = session_store.load_meta(session_id)
        if meta:
            meta.title = title
            session_store.save_meta(meta)
        logger.debug("Auto-generated session title: %s", title)
        return title
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)
        return None
