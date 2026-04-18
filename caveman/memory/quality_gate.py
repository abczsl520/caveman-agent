"""Memory quality gate — prevents garbage from entering the flywheel.

This is the highest-leverage intervention in the entire system.
Every piece of garbage that enters costs:
  - Storage space forever
  - FTS5 index pollution (degrades recall quality)
  - Lint cycles wasted on obvious junk
  - HybridScorer noise (garbage competes with real knowledge)

PRD §6 Iron Law #1: Write quality >> Retrieve sophistication.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Minimum content length (characters). Below this, there's no knowledge.
_MIN_LENGTH = 15

# Maximum content length. Beyond this, it's a raw dump, not extracted knowledge.
_MAX_LENGTH = 3000

# Patterns that indicate zero-information content
_GARBAGE_PATTERNS = (
    # Test/benchmark noise
    re.compile(r"^Completed task:\s*.{0,30}$", re.IGNORECASE),
    re.compile(r"All tools executed successfully", re.IGNORECASE),
    re.compile(r"^Task completed\s*\.?\s*$", re.IGNORECASE),
    # Generic QA that LLM already knows (no project-specific value)
    re.compile(r"^Task:\s*(?:What is|Explain|Define|Convert|Write a (?:Python|JavaScript|bash))\s", re.IGNORECASE),
    # Empty/trivial results
    re.compile(r"^(?:Done|OK|Success|Acknowledged|Got it)\s*[.!]?\s*$", re.IGNORECASE),
    # Session metadata (not knowledge)
    re.compile(r"^# Session:\s*\d{4}-\d{2}-\d{2}"),
    # API keys / secrets leaked into memory
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    # Raw stack traces (store the fix, not the trace)
    re.compile(r"Traceback \(most recent call last\)"),
    # Base64 blobs
    re.compile(r"[A-Za-z0-9+/]{100,}={0,2}"),
    # Repeated characters (spam)
    re.compile(r"(.)\1{9,}"),
    # Pure URLs without context
    re.compile(r"^https?://\S+$"),
    # Chinese: pure interjections / filler
    re.compile(r"^[哈呵嘿嗯啊哦噢好的了吧呢嘛吗呀哇]{3,}[。！？.!?]*$"),
    # Chinese: acknowledgment without info
    re.compile(r"^(?:好的|收到|明白|了解|知道了|没问题|可以|行|嗯嗯|OK|ok)\s*[。！？.!?]*$"),
)

# Near-duplicate detection via content fingerprint
_FINGERPRINT_CACHE_SIZE = 500
_fingerprint_cache: set[str] = set()


def _fingerprint(content: str) -> str:
    """Create a normalized fingerprint for near-duplicate detection."""
    normalized = re.sub(r'\s+', ' ', content.lower().strip())[:200]
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized


def check_quality(content: str, trusted: bool = False) -> str | None:
    """Check if content meets quality threshold for storage.

    Returns None if quality is OK, or a rejection reason string.
    Trusted content (e.g., user explicit store) bypasses most checks.
    """
    if trusted:
        return None

    if len(content.strip()) < _MIN_LENGTH:
        return f"too_short ({len(content.strip())} < {_MIN_LENGTH})"

    for pat in _GARBAGE_PATTERNS:
        if pat.search(content):
            return f"garbage_pattern: {pat.pattern[:50]}"

    fp = _fingerprint(content)
    if fp in _fingerprint_cache:
        return "near_duplicate"
    if len(_fingerprint_cache) >= _FINGERPRINT_CACHE_SIZE:
        _fingerprint_cache.clear()
    _fingerprint_cache.add(fp)

    return None


def truncate_if_needed(content: str) -> str:
    """Truncate content to max length, preserving sentence boundaries."""
    if len(content) <= _MAX_LENGTH:
        return content
    truncated = content[:_MAX_LENGTH]
    for end in ('.', '\u3002', '!', '\n'):
        idx = truncated.rfind(end)
        if idx > _MAX_LENGTH // 2:
            return truncated[:idx + 1]
    return truncated + "..."


def reset_cache() -> None:
    """Reset fingerprint cache (for testing)."""
    _fingerprint_cache.clear()
