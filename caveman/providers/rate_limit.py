"""Rate limit tracking from API response headers.

Ported from Hermes rate_limit_tracker.py (MIT, Nous Research).
Captures x-ratelimit-* headers and provides formatted display.
Integrates with LLM Scheduler for dynamic rate adjustment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests per minute)."""
    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        return (self.used / self.limit * 100.0) if self.limit > 0 else 0.0

    @property
    def remaining_seconds_now(self) -> float:
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)


@dataclass
class RateLimitState:
    """Full rate-limit state from response headers."""
    requests_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at: float = 0.0
    provider: str = ""

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_rate_limit_headers(
    headers: Mapping[str, str], provider: str = "",
) -> Optional[RateLimitState]:
    """Parse x-ratelimit-* headers into RateLimitState."""
    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-ratelimit-") for k in lowered):
        return None

    now = time.time()

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket:
        tag = f"{resource}{suffix}"
        return RateLimitBucket(
            limit=_safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining=_safe_int(lowered.get(f"x-ratelimit-remaining-{tag}")),
            reset_seconds=_safe_float(lowered.get(f"x-ratelimit-reset-{tag}")),
            captured_at=now,
        )

    return RateLimitState(
        requests_min=_bucket("requests"),
        requests_hour=_bucket("requests", "-1h"),
        tokens_min=_bucket("tokens"),
        tokens_hour=_bucket("tokens", "-1h"),
        captured_at=now,
        provider=provider,
    )


# --- Formatting ---

def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 20) -> str:
    filled = max(0, min(width, int(pct / 100.0 * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def format_rate_limits(state: RateLimitState) -> str:
    """Format rate limit state for display."""
    if not state.has_data:
        return "No rate limit data yet."

    age = state.age_seconds
    freshness = "just now" if age < 5 else _fmt_seconds(age) + " ago"
    provider = state.provider.title() if state.provider else "Provider"

    def _line(label: str, b: RateLimitBucket) -> str:
        if b.limit <= 0:
            return f"  {label:<14}  (no data)"
        pct = b.usage_pct
        return (
            f"  {label:<14} {_bar(pct)} {pct:5.1f}%  "
            f"{_fmt_count(b.used)}/{_fmt_count(b.limit)} used  "
            f"({_fmt_count(b.remaining)} left, resets {_fmt_seconds(b.remaining_seconds_now)})"
        )

    lines = [
        f"{provider} Rate Limits (captured {freshness}):",
        "", _line("Requests/min", state.requests_min),
        _line("Requests/hr", state.requests_hour),
        "", _line("Tokens/min", state.tokens_min),
        _line("Tokens/hr", state.tokens_hour),
    ]

    # Warnings for hot buckets
    for label, b in [
        ("requests/min", state.requests_min),
        ("requests/hr", state.requests_hour),
        ("tokens/min", state.tokens_min),
        ("tokens/hr", state.tokens_hour),
    ]:
        if b.limit > 0 and b.usage_pct >= 80:
            lines.append(f"  ⚠ {label} at {b.usage_pct:.0f}% — resets {_fmt_seconds(b.remaining_seconds_now)}")

    return "\n".join(lines)


def format_compact(state: RateLimitState) -> str:
    """One-line compact summary."""
    if not state.has_data:
        return "No rate limit data."
    parts = []
    for label, b in [("RPM", state.requests_min), ("TPM", state.tokens_min)]:
        if b.limit > 0:
            parts.append(f"{label}: {b.remaining}/{b.limit}")
    return " | ".join(parts) or "No limits reported."
