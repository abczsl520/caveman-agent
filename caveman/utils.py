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


_STRONG_SUCCESS_PATTERNS = [
    # Objective verification evidence. These are safe to use for trust/metrics.
    r"\b\d+\s+passed\b",
    r"\ball\s+tests?\s+pass(?:ed)?\b",
    r"\btests?\s+pass(?:ed)?\b",
    r"\bexit(?:ed)?\s+(?:with\s+)?(?:code\s+)?0\b",
    r"\breturn(?:ed)?\s+(?:code\s+)?0\b",
    r"\bpytest\b[^\n]{0,120}\bpassed\b",
    r"\bverified\b[^\n]{0,120}\b(?:pass(?:ed)?|working|fixed|resolved|green)\b",
    r"\bvalidated\b[^\n]{0,120}\b(?:pass(?:ed)?|working|fixed|resolved|green)\b",
    r"\bcommit(?:ted)?\b[^\n]{0,120}\b(?:[0-9a-f]{7,40}|changed|created)\b",
    r"\bno\s+P0\b",
    r"\bP0\s*[:=-]\s*(?:0|none|no\s+issues?)\b",
]

_WEAK_SUCCESS_PATTERNS = [
    # Human/model claims only. These may indicate partial progress but are not
    # sufficient evidence for trust-score boosts, metrics success, or flywheel
    # round success. This list intentionally includes the historically dangerous
    # tokens that caused premature done: done/completed/✅/successfully/fixed.
    r"(?:✅|done|completed|finished|success|successfully|passed|fixed|resolved|created|built)",
    r"(?:here (?:is|are) (?:the|your))",
    r"(?:I've |I have )",
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
    r"(?:fixed|resolved|found|identified|debugged|handled|caught|diagnosed)\b[^\n]{0,160}\b"
    r"(?:error|exception|traceback|typeerror|valueerror|importerror|syntaxerror|runtimeerror)",
    _re.IGNORECASE,
)


def _count_matches(patterns: list[str], sample: str) -> int:
    return sum(1 for p in patterns if _re.search(p, sample, _re.IGNORECASE))


def detect_success(text: str) -> bool:
    """Conservative success detection for trust/metrics feedback.

    This function is shared by phase_finalize, Reflect/outcome, and metrics.
    Natural-language claims of task closure or confidence are intentionally
    insufficient; only external verification evidence should raise trust.

    Returns True only when there is explicit verification evidence such as tests
    passing, exit code 0, a verified/validated result, a commit change, or an
    explicit "no P0" audit outcome. Absence of failure is not success.
    """
    if not text:
        return False

    sample = text[:1000]
    failure_signals = _count_matches(_FAILURE_PATTERNS, sample)
    strong_success_signals = _count_matches(_STRONG_SUCCESS_PATTERNS, sample)

    if failure_signals:
        # A verified fix may mention the original error. Only discount an error
        # word when there is also strong evidence; otherwise failure wins.
        if _ERROR_IN_SUCCESS_CONTEXT.search(sample) and strong_success_signals:
            failure_signals = max(0, failure_signals - 1)
        if failure_signals > 0:
            return False

    return strong_success_signals > 0


def detect_outcome(text: str) -> str:
    """Detect task outcome as 'success' | 'partial' | 'failure'.

    `success` requires the same objective evidence as detect_success(). Weak
    completion claims become `partial` so downstream systems do not equate
    response end / model confidence with verified task completion.
    """
    if not text:
        return "failure"
    if detect_success(text):
        return "success"
    sample = text[:1000]
    has_failure = _count_matches(_FAILURE_PATTERNS, sample) > 0
    has_weak_success = _count_matches(_WEAK_SUCCESS_PATTERNS, sample) > 0
    if has_weak_success and not has_failure:
        return "partial"
    if has_weak_success and has_failure:
        return "partial"
    return "failure"
