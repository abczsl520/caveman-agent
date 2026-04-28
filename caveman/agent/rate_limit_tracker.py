"""Rate Limit Tracker — API rate limit tracking and backoff.

Tracks API rate limits per provider/model and implements
intelligent backoff. Extracted from Hermes agent/rate_limit_tracker.py.
"""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("caveman.agent.rate_limit_tracker")


@dataclass
class RateLimitState:
    """State for a rate-limited endpoint."""
    requests_remaining: int = -1  # -1 = unknown
    tokens_remaining: int = -1
    reset_at: float = 0
    retry_after: float = 0
    consecutive_429s: int = 0
    last_429_at: float = 0
    total_429s: int = 0

    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        if self.retry_after > 0 and time.time() < self.last_429_at + self.retry_after:
            return True
        if self.requests_remaining == 0 and time.time() < self.reset_at:
            return True
        return False

    @property
    def wait_seconds(self) -> float:
        """How long to wait before next request."""
        if not self.is_limited:
            return 0
        if self.retry_after > 0:
            remaining = (self.last_429_at + self.retry_after) - time.time()
            return max(0, remaining)
        if self.reset_at > 0:
            return max(0, self.reset_at - time.time())
        return 0

    def backoff_seconds(self) -> float:
        """Calculate exponential backoff based on consecutive 429s."""
        if self.consecutive_429s == 0:
            return 0
        base = 1.0
        return min(base * (2 ** (self.consecutive_429s - 1)), 60.0)


class RateLimitTracker:
    """Tracks rate limits across multiple API endpoints."""

    def __init__(self):
        self._states: Dict[str, RateLimitState] = {}
        self._lock = threading.Lock()

    def get_state(self, key: str) -> RateLimitState:
        """Get rate limit state for a key (provider:model)."""
        with self._lock:
            if key not in self._states:
                self._states[key] = RateLimitState()
            return self._states[key]

    def update_from_headers(self, key: str, headers: Dict[str, str]) -> None:
        """Update state from API response headers."""
        with self._lock:
            state = self._states.setdefault(key, RateLimitState())

            # OpenAI-style headers
            if "x-ratelimit-remaining-requests" in headers:
                state.requests_remaining = int(headers["x-ratelimit-remaining-requests"])
            if "x-ratelimit-remaining-tokens" in headers:
                state.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
            if "x-ratelimit-reset-requests" in headers:
                # Parse relative time (e.g., "1s", "6m0s")
                state.reset_at = time.time() + _parse_duration(headers["x-ratelimit-reset-requests"])

            # Anthropic-style headers
            if "anthropic-ratelimit-requests-remaining" in headers:
                state.requests_remaining = int(headers["anthropic-ratelimit-requests-remaining"])
            if "anthropic-ratelimit-tokens-remaining" in headers:
                state.tokens_remaining = int(headers["anthropic-ratelimit-tokens-remaining"])

            # Standard retry-after
            if "retry-after" in headers:
                try:
                    state.retry_after = float(headers["retry-after"])
                except ValueError:
                    state.retry_after = 60.0

    def record_429(self, key: str, retry_after: float = 0) -> None:
        """Record a 429 response."""
        with self._lock:
            state = self._states.setdefault(key, RateLimitState())
            state.consecutive_429s += 1
            state.total_429s += 1
            state.last_429_at = time.time()
            if retry_after > 0:
                state.retry_after = retry_after

    def record_success(self, key: str) -> None:
        """Record a successful request (resets consecutive 429 counter)."""
        with self._lock:
            state = self._states.get(key)
            if state:
                state.consecutive_429s = 0

    def should_wait(self, key: str) -> float:
        """Check if we should wait before making a request. Returns wait seconds."""
        state = self.get_state(key)
        if state.is_limited:
            return state.wait_seconds
        if state.consecutive_429s > 0:
            return state.backoff_seconds()
        return 0

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tracked endpoints."""
        with self._lock:
            return {
                key: {
                    "requests_remaining": s.requests_remaining,
                    "tokens_remaining": s.tokens_remaining,
                    "is_limited": s.is_limited,
                    "consecutive_429s": s.consecutive_429s,
                    "total_429s": s.total_429s,
                }
                for key, s in self._states.items()
            }

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key:
                self._states.pop(key, None)
            else:
                self._states.clear()


def _parse_duration(s: str) -> float:
    """Parse duration string like '1s', '6m0s', '1h2m3s'."""
    import re
    total = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)(h|m|s|ms)", s):
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "s":
            total += value
        elif unit == "ms":
            total += value / 1000
    return total or 60.0  # Default 60s if unparseable
