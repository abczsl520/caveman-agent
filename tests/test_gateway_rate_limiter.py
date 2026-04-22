"""Tests for gateway rate limiter."""
import time

import pytest

from caveman.gateway.rate_limiter import TokenBucket, RateLimiter


class TestTokenBucket:
    def test_initial_tokens_equal_capacity(self):
        b = TokenBucket(capacity=10, refill_rate=1.0)
        assert b.tokens == 10

    def test_consume_reduces_tokens(self):
        b = TokenBucket(capacity=10, refill_rate=1.0)
        assert b.try_consume(3)
        assert b.tokens == 7

    def test_consume_fails_when_empty(self):
        b = TokenBucket(capacity=2, refill_rate=0.1)
        assert b.try_consume(2)
        assert not b.try_consume(1)

    def test_refill_over_time(self):
        b = TokenBucket(capacity=10, refill_rate=100.0)  # Fast refill
        b.try_consume(10)
        time.sleep(0.05)
        assert b.try_consume(1)  # Should have refilled some

    def test_tokens_capped_at_capacity(self):
        b = TokenBucket(capacity=5, refill_rate=1000.0)
        time.sleep(0.01)
        b._refill()
        assert b.tokens <= 5

    def test_wait_time_zero_when_available(self):
        b = TokenBucket(capacity=10, refill_rate=1.0)
        assert b.wait_time(1) == 0

    def test_wait_time_positive_when_empty(self):
        b = TokenBucket(capacity=1, refill_rate=1.0)
        b.try_consume(1)
        wt = b.wait_time(1)
        assert wt > 0


class TestRateLimiter:
    def test_default_allows_requests(self):
        rl = RateLimiter(default_capacity=10, default_rate=1.0)
        assert rl.try_acquire("key1")

    def test_exhaustion(self):
        rl = RateLimiter(default_capacity=2, default_rate=0.001)
        assert rl.try_acquire("key1")
        assert rl.try_acquire("key1")
        assert not rl.try_acquire("key1")

    def test_separate_keys(self):
        rl = RateLimiter(default_capacity=1, default_rate=0.001)
        assert rl.try_acquire("a")
        assert rl.try_acquire("b")  # Different key, separate bucket

    def test_configure_custom_limits(self):
        rl = RateLimiter()
        rl.configure("vip", capacity=1000, rate=100.0)
        info = rl.check("vip")
        assert info["capacity"] == 1000

    def test_check_returns_state(self):
        rl = RateLimiter(default_capacity=60, default_rate=1.0)
        info = rl.check("new_key")
        assert info["capacity"] == 60
        assert info["refill_rate"] == 1.0

    def test_reset_specific_key(self):
        rl = RateLimiter(default_capacity=1, default_rate=0.001)
        rl.try_acquire("k")
        assert not rl.try_acquire("k")
        rl.reset("k")
        assert rl.try_acquire("k")  # Fresh bucket

    def test_reset_all(self):
        rl = RateLimiter(default_capacity=1, default_rate=0.001)
        rl.try_acquire("a")
        rl.try_acquire("b")
        rl.reset()
        assert rl.try_acquire("a")
        assert rl.try_acquire("b")

    def test_wait_time(self):
        rl = RateLimiter(default_capacity=1, default_rate=1.0)
        rl.try_acquire("k")
        wt = rl.wait_time("k")
        assert wt > 0
