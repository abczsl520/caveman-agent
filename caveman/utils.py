"""Shared utilities — DRY primitives used across the codebase.

Every function here exists because it was duplicated 2+ times.
Adding here = commitment to never duplicate again.
"""
from __future__ import annotations
import asyncio
import logging
import math
from typing import TypeVar, Callable, Awaitable, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")

import re as _re_utils
_CJK_RE = _re_utils.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')


# ── Token estimation (CJK-aware, single source of truth) ──

def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (CJK-aware).

    Single source of truth for token estimation across the codebase.
    - English/code: ~4 chars/token
    - CJK (Chinese/Japanese/Korean): ~1 char/token
    - Mixed: weighted blend + 3 token overhead

    Used by: prompt.py, context.py, compression/, context_refs.py
    """
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    non_cjk_len = len(text) - cjk_count
    return cjk_count + (non_cjk_len // 4) + 3


# Average chars per token by model family (moved from gateway.agent_memory_depth)
_MODEL_CHARS_PER_TOKEN = {
    "claude": 3.5,
    "gpt": 4.0,
    "gemini": 3.8,
    "deepseek": 3.2,
    "default": 3.7,
}


def estimate_tokens_for_model(text: str, model: str = "") -> int:
    """Estimate token count for text based on model-specific chars-per-token ratios.

    Unlike estimate_tokens() (CJK-aware heuristic), this uses model-family
    ratios for budget planning.
    """
    if not text:
        return 0
    family = "default"
    model_lower = model.lower()
    for prefix in _MODEL_CHARS_PER_TOKEN:
        if prefix in model_lower:
            family = prefix
            break
    chars_per_token = _MODEL_CHARS_PER_TOKEN[family]
    return int(len(text) / chars_per_token)


# ── Math ──

try:
    import numpy as _np
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors (numpy fast path)."""
        if len(a) != len(b) or not a:
            return 0.0
        va, vb = _np.asarray(a, dtype=_np.float32), _np.asarray(b, dtype=_np.float32)
        na, nb = _np.linalg.norm(va), _np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(_np.dot(va, vb) / (na * nb))
except ImportError:
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors (pure Python fallback)."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ── Retry ──

async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, Exception], Any] | None = None,
    **kwargs,
) -> T:
    """Retry an async function with jittered exponential backoff.

    Delegates to ``providers.retry.jittered_backoff`` for decorrelated delays,
    keeping the simple call-site signature for non-provider code.

    Args:
        fn: Async function to call
        max_retries: Maximum number of attempts (not retries)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        retryable: Optional predicate — return True if exception is retryable
        on_retry: Optional callback(attempt, exception) on each retry

    Returns:
        Result of fn()

    Raises:
        Last exception if all retries exhausted
    """
    from caveman.providers.retry import jittered_backoff

    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if retryable and not retryable(e):
                raise  # Non-retryable, fail immediately

            if attempt == max_retries - 1:
                raise  # Last attempt, propagate

            delay = jittered_backoff(
                attempt, base_delay=base_delay, max_delay=max_delay,
            )
            if on_retry:
                on_retry(attempt, e)
            else:
                logger.warning(
                    "Retry %d/%d for %s after %s: %.1fs delay",
                    attempt + 1, max_retries, fn.__name__,
                    type(e).__name__, delay,
                )
            await asyncio.sleep(delay)

    raise last_exc  # Should never reach here, but type-safety


# ── LLM response parsing ──

import re

_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses.

    Handles: ```json ... ```, ``` ... ```, nested fences.
    Returns the inner content if fenced, or original text if not.
    """
    if not text:
        return text
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_from_llm(text: str, expect: str = "object") -> Any:
    """Parse JSON from an LLM response, handling code fences and preamble.

    Args:
        text: Raw LLM response
        expect: "object" to find {...} or "array" to find [...]

    Returns:
        Parsed JSON data, or None if parsing fails.
    """
    import json

    if not text:
        return None

    # Step 1: Strip code fences
    cleaned = strip_code_fences(text)

    # Step 2: Try direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass  # intentional: ValueError suppressed

    # Step 3: Find the outermost JSON structure
    open_char = "{" if expect == "object" else "["
    close_char = "}" if expect == "object" else "]"

    start = cleaned.find(open_char)
    if start < 0:
        return None

    # Find matching close by counting nesting
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None

    return None


# ── Text splitting ──

def split_message(text: str, max_length: int = 1900) -> list[str]:
    """Split text into chunks respecting max_length.

    Delegates to gateway.message_splitting (canonical implementation) which has:
    - Fence-parity counting (handles nested/multiple code fences correctly)
    - Language-tag preservation across chunks (```python -> close -> reopen)
    - Paragraph / line / sentence break-point hierarchy
    - Post-processing to auto-balance unclosed fences

    This thin wrapper keeps the utils.split_message API stable so all
    gateway modules (discord_gw, telegram_gw, slack_gw, outbound) benefit
    from a single source of truth.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    from caveman.gateway.message_splitting import split_message as _canonical
    return _canonical(text, max_length=max_length)


# ── Success detection (shared by phases.py + reflect.py) ──

import re as _re

__all__ = [
    "T",
    "estimate_tokens",
    "estimate_tokens_for_model",
    "retry_async",
    "strip_code_fences",
    "parse_json_from_llm",
    "split_message",
    "detect_success",
    "detect_outcome",
]


_SUCCESS_PATTERNS = [
    r"(?:✅|done|completed|finished|success|passed|fixed|resolved|created|built)",
    r"(?:all\s+\d+\s+(?:tests?\s+)?pass)",
    r"(?:here (?:is|are) (?:the|your))",
    r"(?:I've |I have |successfully )",
]

_FAILURE_PATTERNS = [
    r"(?:❌|FAILED)",
    r"\b(?:ERROR|TypeError|ValueError|ImportError|SyntaxError|RuntimeError)\b",
    r"(?:Traceback \(most recent)",
    r"(?:could not|unable to|cannot|impossible)",
    r"(?:I (?:couldn't|can't|was unable|failed to))",
    r"(?:unfortunately|sorry.*(?:can't|couldn't|unable))",
]

_ERROR_IN_SUCCESS_CONTEXT = _re.compile(
    r"(?:fixed|resolved|found|identified|debugged|handled|caught|diagnosed)\s+(?:the\s+)?error",
    _re.IGNORECASE,
)


def detect_success(text: str) -> bool:
    """Multi-signal success detection for confidence feedback loop.

    Used by phase_finalize (trust scoring) and Reflect (outcome detection).
    Replaces the old catastrophic `"error" not in text[:100]`.
    """
    if not text:
        return False

    sample = text[:500]
    success_signals = sum(
        1 for p in _SUCCESS_PATTERNS if _re.search(p, sample, _re.IGNORECASE)
    )
    failure_signals = sum(
        1 for p in _FAILURE_PATTERNS if _re.search(p, sample, _re.IGNORECASE)
    )

    if _ERROR_IN_SUCCESS_CONTEXT.search(sample):
        failure_signals = max(0, failure_signals - 1)
        success_signals += 1

    if success_signals == 0 and failure_signals == 0:
        # ANTI-HALLUCINATION: No signal ≠ success.
        # Returning True here caused trust inflation — every ambiguous output
        # boosted recalled memories' trust, making garbage float to the top.
        # Now returns False (unknown/neutral), so only genuinely successful
        # tasks boost trust. This is the conservative default.
        return False

    return success_signals >= failure_signals


def detect_outcome(text: str) -> str:
    """Detect task outcome as 'success' | 'partial' | 'failure'.

    Wrapper around detect_success for Reflect compatibility.
    """
    if not text:
        return "failure"
    if detect_success(text):
        return "success"
    # Check if there are ANY success signals (partial)
    sample = text[:500]
    has_success = any(
        _re.search(p, sample, _re.IGNORECASE) for p in _SUCCESS_PATTERNS
    )
    return "partial" if has_success else "failure"
