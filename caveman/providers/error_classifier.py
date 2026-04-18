"""API error classification for smart recovery.

Ported from Hermes error_classifier.py (MIT, Nous Research).
Classifies API errors into structured recovery recommendations.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""
    auth = "auth"
    auth_permanent = "auth_permanent"
    billing = "billing"
    rate_limit = "rate_limit"
    overloaded = "overloaded"
    server_error = "server_error"
    timeout = "timeout"
    context_overflow = "context_overflow"
    payload_too_large = "payload_too_large"
    model_not_found = "model_not_found"
    format_error = "format_error"
    unknown = "unknown"


@dataclass
class ClassifiedError:
    """Structured classification with recovery hints."""
    reason: FailoverReason
    status_code: Optional[int] = None
    provider: str = ""
    model: str = ""
    message: str = ""
    retryable: bool = True
    should_compress: bool = False
    should_rotate: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in (FailoverReason.auth, FailoverReason.auth_permanent)


# --- Pattern tables ---

_BILLING_PATTERNS = [
    "insufficient credits", "insufficient_quota", "credit balance",
    "credits have been exhausted", "payment required", "billing hard limit",
    "exceeded your current quota", "account is deactivated",
]

_RATE_LIMIT_PATTERNS = [
    "rate limit", "rate_limit", "too many requests", "throttled",
    "requests per minute", "tokens per minute", "try again in",
    "please retry after", "resource_exhausted",
]

_USAGE_LIMIT_PATTERNS = ["usage limit", "quota", "limit exceeded"]
_USAGE_TRANSIENT = ["try again", "retry", "resets at", "reset in", "wait"]

_CONTEXT_OVERFLOW_PATTERNS = [
    "context length", "context size", "maximum context", "token limit",
    "too many tokens", "reduce the length", "exceeds the limit",
    "context window", "prompt is too long", "max_tokens",
    "超过最大长度", "上下文长度",
]

_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model", "invalid model", "model not found",
    "model_not_found", "does not exist", "unknown model",
]

_AUTH_PATTERNS = [
    "invalid api key", "invalid_api_key", "authentication",
    "unauthorized", "forbidden", "invalid token", "access denied",
]

_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError", "ConnectionError",
    "ConnectionResetError", "BrokenPipeError", "TimeoutError",
    "APIConnectionError", "APITimeoutError",
})

_DISCONNECT_PATTERNS = [
    "server disconnected", "peer closed connection",
    "connection reset by peer", "unexpected eof",
]


def classify_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200_000,
) -> ClassifiedError:
    """Classify an API error into a recovery recommendation.

    Priority: status code → error code → message patterns → transport → unknown.
    """
    status = _extract_status(error)
    body = _extract_body(error)
    error_code = _extract_code(body)
    error_msg = _build_message(error, body)

    def _r(reason: FailoverReason, **kw) -> ClassifiedError:
        return ClassifiedError(
            reason=reason, status_code=status,
            provider=provider, model=model,
            message=_extract_display_msg(error, body),
            **kw,
        )

    # 1. Status code classification
    if status is not None:
        result = _by_status(status, error_msg, error_code, body,
                            approx_tokens, context_length, _r)
        if result:
            return result

    # 2. Error code
    if error_code:
        result = _by_error_code(error_code, _r)
        if result:
            return result

    # 3. Message patterns
    result = _by_message(error_msg, approx_tokens, context_length, _r)
    if result:
        return result

    # 4. Disconnect + large session → context overflow
    if any(p in error_msg for p in _DISCONNECT_PATTERNS) and not status:
        if approx_tokens > context_length * 0.6 or approx_tokens > 120_000:
            return _r(FailoverReason.context_overflow,
                      retryable=True, should_compress=True)
        return _r(FailoverReason.timeout, retryable=True)

    # 5. Transport errors
    error_type = type(error).__name__
    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(
        error, (TimeoutError, ConnectionError, OSError)
    ):
        return _r(FailoverReason.timeout, retryable=True)

    # 6. Unknown
    return _r(FailoverReason.unknown, retryable=True)


def _classify_400(msg, tokens, ctx_len, _r):
    """Classify HTTP 400 errors by message content."""
    if any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return _r(FailoverReason.context_overflow, retryable=True, should_compress=True)
    if any(p in msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return _r(FailoverReason.model_not_found, retryable=False, should_fallback=True)
    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return _r(FailoverReason.rate_limit, retryable=True, should_rotate=True)
    if tokens > ctx_len * 0.4 or tokens > 80_000:
        return _r(FailoverReason.context_overflow, retryable=True, should_compress=True)
    return _r(FailoverReason.format_error, retryable=False, should_fallback=True)


def _classify_402(msg, _r):
    """Classify HTTP 402: transient usage limit vs billing."""
    if (any(p in msg for p in _USAGE_LIMIT_PATTERNS)
            and any(p in msg for p in _USAGE_TRANSIENT)):
        return _r(FailoverReason.rate_limit, retryable=True,
                  should_rotate=True, should_fallback=True)
    return _r(FailoverReason.billing, retryable=False,
              should_rotate=True, should_fallback=True)


def _classify_403(msg, _r):
    """Classify HTTP 403: billing vs auth."""
    if "key limit" in msg or "spending limit" in msg:
        return _r(FailoverReason.billing, retryable=False,
                  should_rotate=True, should_fallback=True)
    return _r(FailoverReason.auth, retryable=False, should_fallback=True)


# Simple status → classification (no message inspection needed)
_SIMPLE_STATUS = {
    401: lambda _r: _r(FailoverReason.auth, retryable=False, should_rotate=True, should_fallback=True),
    404: lambda _r: _r(FailoverReason.model_not_found, retryable=False, should_fallback=True),
    413: lambda _r: _r(FailoverReason.payload_too_large, retryable=True, should_compress=True),
    429: lambda _r: _r(FailoverReason.rate_limit, retryable=True, should_rotate=True, should_fallback=True),
    500: lambda _r: _r(FailoverReason.server_error, retryable=True),
    502: lambda _r: _r(FailoverReason.server_error, retryable=True),
    503: lambda _r: _r(FailoverReason.overloaded, retryable=True),
    529: lambda _r: _r(FailoverReason.overloaded, retryable=True),
}


def _by_status(status, msg, code, body, tokens, ctx_len, _r):
    # Simple dispatch
    if status in _SIMPLE_STATUS:
        return _SIMPLE_STATUS[status](_r)
    # Complex cases needing message inspection
    if status == 400:
        return _classify_400(msg, tokens, ctx_len, _r)
    if status == 402:
        return _classify_402(msg, _r)
    if status == 403:
        return _classify_403(msg, _r)
    # Fallback ranges
    if 400 <= status < 500:
        return _r(FailoverReason.format_error, retryable=False, should_fallback=True)
    if 500 <= status < 600:
        return _r(FailoverReason.server_error, retryable=True)
    return None


def _by_error_code(code, _r):
    c = code.lower()
    if c in ("resource_exhausted", "throttled", "rate_limit_exceeded"):
        return _r(FailoverReason.rate_limit, retryable=True, should_rotate=True)
    if c in ("insufficient_quota", "billing_not_active", "payment_required"):
        return _r(FailoverReason.billing, retryable=False,
                  should_rotate=True, should_fallback=True)
    if c in ("model_not_found", "model_not_available", "invalid_model"):
        return _r(FailoverReason.model_not_found, retryable=False,
                  should_fallback=True)
    if c in ("context_length_exceeded", "max_tokens_exceeded"):
        return _r(FailoverReason.context_overflow, retryable=True,
                  should_compress=True)
    return None


def _by_message(msg, tokens, ctx_len, _r):
    # Usage limit disambiguation
    if any(p in msg for p in _USAGE_LIMIT_PATTERNS):
        if any(p in msg for p in _USAGE_TRANSIENT):
            return _r(FailoverReason.rate_limit, retryable=True,
                      should_rotate=True, should_fallback=True)
        return _r(FailoverReason.billing, retryable=False,
                  should_rotate=True, should_fallback=True)
    if any(p in msg for p in _BILLING_PATTERNS):
        return _r(FailoverReason.billing, retryable=False,
                  should_rotate=True, should_fallback=True)
    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return _r(FailoverReason.rate_limit, retryable=True,
                  should_rotate=True, should_fallback=True)
    if any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return _r(FailoverReason.context_overflow, retryable=True,
                  should_compress=True)
    if any(p in msg for p in _AUTH_PATTERNS):
        return _r(FailoverReason.auth, retryable=False,
                  should_rotate=True, should_fallback=True)
    if any(p in msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return _r(FailoverReason.model_not_found, retryable=False,
                  should_fallback=True)
    return None


# --- Extractors ---

def _extract_status(error: Exception) -> Optional[int]:
    current = error
    for _ in range(5):
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        cause = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_body(error: Exception) -> dict:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body
    resp = getattr(error, "response", None)
    if resp is not None:
        try:
            j = resp.json()
            if isinstance(j, dict):
                return j
        except Exception as e:
            logger.debug("Suppressed in error_classifier: %s", e)
    return {}


def _extract_code(body: dict) -> str:
    if not body:
        return ""
    err = body.get("error", {})
    if isinstance(err, dict):
        c = err.get("code") or err.get("type") or ""
        if isinstance(c, str) and c.strip():
            return c.strip()
    c = body.get("code") or body.get("error_code") or ""
    return str(c).strip() if c else ""


def _build_message(error: Exception, body: dict) -> str:
    parts = [str(error).lower()]
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            msg = (err.get("message") or "").lower()
            if msg and msg not in parts[0]:
                parts.append(msg)
    return " ".join(parts)


def _extract_display_msg(error: Exception, body: dict) -> str:
    if body:
        err = body.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
    return str(error)[:500]
