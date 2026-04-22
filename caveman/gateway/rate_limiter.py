"""Rate Limiter — token bucket algorithm for API rate limiting.

Provides per-key rate limiting with token bucket algorithm,
sliding window, and burst support.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TokenBucket:
    """Token bucket rate limiter."""
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0
    last_refill: float = 0

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = self.capacity
        if self.last_refill == 0:
            self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_consume(self, tokens: float = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: float = 1) -> float:
        """How long to wait before tokens are available."""
        self._refill()
        if self.tokens >= tokens:
            return 0
        deficit = tokens - self.tokens
        return deficit / self.refill_rate


class RateLimiter:
    """Multi-key rate limiter using token buckets."""

    def __init__(
        self,
        default_capacity: float = 60,
        default_rate: float = 1.0,  # tokens per second
    ):
        self._default_capacity = default_capacity
        self._default_rate = default_rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def configure(self, key: str, capacity: float, rate: float) -> None:
        """Configure rate limit for a specific key."""
        with self._lock:
            self._buckets[key] = TokenBucket(capacity=capacity, refill_rate=rate)

    def try_acquire(self, key: str, tokens: float = 1) -> bool:
        """Try to acquire tokens for a key."""
        with self._lock:
            bucket = self._get_or_create(key)
            return bucket.try_consume(tokens)

    def wait_time(self, key: str, tokens: float = 1) -> float:
        """Get wait time for a key."""
        with self._lock:
            bucket = self._get_or_create(key)
            return bucket.wait_time(tokens)

    def check(self, key: str) -> Dict[str, float]:
        """Check current state of a key's bucket."""
        with self._lock:
            bucket = self._get_or_create(key)
            bucket._refill()
            return {
                "tokens": round(bucket.tokens, 2),
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
            }

    def _get_or_create(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=self._default_capacity,
                refill_rate=self._default_rate,
            )
        return self._buckets[key]

    def reset(self, key: Optional[str] = None) -> None:
        """Reset a specific key or all keys."""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()
