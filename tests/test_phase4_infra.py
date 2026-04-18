"""Tests for Phase 4 — rate limit, model routing, insights."""
from __future__ import annotations

import time
import pytest

from caveman.providers.rate_limit import (
    RateLimitBucket,
    RateLimitState,
    parse_rate_limit_headers,
    format_rate_limits,
    format_compact,
    _fmt_count,
    _fmt_seconds,
)
from caveman.providers.model_router import is_simple_turn, choose_model
from caveman.providers.insights import TurnUsage, SessionInsights


# --- Rate limit ---

class TestRateLimitBucket:
    def test_used(self):
        b = RateLimitBucket(limit=100, remaining=60)
        assert b.used == 40

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=60)
        assert b.usage_pct == 40.0

    def test_zero_limit(self):
        b = RateLimitBucket(limit=0, remaining=0)
        assert b.usage_pct == 0.0
        assert b.used == 0

    def test_remaining_seconds(self):
        b = RateLimitBucket(reset_seconds=60.0, captured_at=time.time() - 10)
        assert 49 <= b.remaining_seconds_now <= 51


class TestRateLimitState:
    def test_has_data(self):
        s = RateLimitState()
        assert not s.has_data
        s.captured_at = time.time()
        assert s.has_data

    def test_age(self):
        s = RateLimitState(captured_at=time.time() - 30)
        assert 29 <= s.age_seconds <= 31


class TestParseHeaders:
    def test_no_headers(self):
        assert parse_rate_limit_headers({}) is None

    def test_basic_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "55",
            "x-ratelimit-reset-requests": "30",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "90000",
            "x-ratelimit-reset-tokens": "60",
        }
        state = parse_rate_limit_headers(headers, provider="anthropic")
        assert state is not None
        assert state.provider == "anthropic"
        assert state.requests_min.limit == 60
        assert state.requests_min.remaining == 55
        assert state.tokens_min.limit == 100000

    def test_case_insensitive(self):
        headers = {"X-RateLimit-Limit-Requests": "100"}
        state = parse_rate_limit_headers(headers)
        assert state is not None
        assert state.requests_min.limit == 100


class TestFormatting:
    def test_fmt_count(self):
        assert _fmt_count(500) == "500"
        assert _fmt_count(5000) == "5.0K"
        assert _fmt_count(5_000_000) == "5.0M"

    def test_fmt_seconds(self):
        assert _fmt_seconds(30) == "30s"
        assert _fmt_seconds(90) == "1m 30s"
        assert _fmt_seconds(3700) == "1h 1m"

    def test_format_no_data(self):
        s = RateLimitState()
        assert "No rate limit" in format_rate_limits(s)

    def test_format_with_data(self):
        s = RateLimitState(
            requests_min=RateLimitBucket(limit=60, remaining=55, captured_at=time.time()),
            captured_at=time.time(),
            provider="anthropic",
        )
        output = format_rate_limits(s)
        assert "Anthropic" in output
        assert "Requests/min" in output

    def test_compact_no_data(self):
        assert "No rate limit" in format_compact(RateLimitState())

    def test_compact_with_data(self):
        s = RateLimitState(
            requests_min=RateLimitBucket(limit=60, remaining=55, captured_at=time.time()),
            captured_at=time.time(),
        )
        output = format_compact(s)
        assert "RPM" in output


# --- Model routing ---

class TestModelRouter:
    def test_simple_greeting(self):
        assert is_simple_turn("hello")
        assert is_simple_turn("hi there")
        assert is_simple_turn("thanks!")

    def test_complex_code(self):
        assert not is_simple_turn("```python\nprint('hello')\n```")
        assert not is_simple_turn("debug this error in my code")
        assert not is_simple_turn("implement a new feature for the API")

    def test_complex_long(self):
        assert not is_simple_turn("x " * 100)

    def test_complex_url(self):
        assert not is_simple_turn("check https://example.com")

    def test_complex_multiline(self):
        assert not is_simple_turn("line 1\nline 2\nline 3")

    def test_complex_chinese(self):
        assert not is_simple_turn("帮我调试这个错误")
        assert not is_simple_turn("实现一个新功能")

    def test_empty(self):
        assert not is_simple_turn("")

    def test_choose_model_disabled(self):
        model, reason = choose_model("hi", "opus", "haiku", routing_enabled=False)
        assert model == "opus"
        assert reason is None

    def test_choose_model_simple(self):
        model, reason = choose_model("hi", "opus", "haiku", routing_enabled=True)
        assert model == "haiku"
        assert reason == "simple_turn"

    def test_choose_model_complex(self):
        model, reason = choose_model("debug this error", "opus", "haiku", routing_enabled=True)
        assert model == "opus"
        assert reason is None

    def test_choose_model_no_cheap(self):
        model, reason = choose_model("hi", "opus", None, routing_enabled=True)
        assert model == "opus"


# --- Insights ---

class TestTurnUsage:
    def test_total_tokens(self):
        u = TurnUsage(input_tokens=1000, output_tokens=500)
        assert u.total_tokens == 1500

    def test_cost_estimation(self):
        u = TurnUsage(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
        cost = u.estimated_cost_usd
        assert cost > 0
        assert cost < 0.01  # Should be very cheap

    def test_cost_unknown_model(self):
        u = TurnUsage(model="unknown-model", input_tokens=1000, output_tokens=500)
        assert u.estimated_cost_usd > 0  # Uses default pricing


class TestSessionInsights:
    def test_empty(self):
        s = SessionInsights()
        assert s.turn_count == 0
        assert s.total_tokens == 0
        assert s.total_cost_usd == 0.0

    def test_record(self):
        s = SessionInsights()
        s.record(TurnUsage(model="gpt-4o", input_tokens=1000, output_tokens=500))
        s.record(TurnUsage(model="gpt-4o", input_tokens=2000, output_tokens=1000))
        assert s.turn_count == 2
        assert s.total_input_tokens == 3000
        assert s.total_output_tokens == 1500

    def test_routed_turns(self):
        s = SessionInsights()
        s.record(TurnUsage(model="opus", routed=False))
        s.record(TurnUsage(model="haiku", routed=True))
        s.record(TurnUsage(model="haiku", routed=True))
        assert s.routed_turns == 2

    def test_avg_tokens(self):
        s = SessionInsights()
        s.record(TurnUsage(input_tokens=100, output_tokens=100))
        s.record(TurnUsage(input_tokens=300, output_tokens=300))
        assert s.avg_tokens_per_turn == 400.0

    def test_format_summary(self):
        s = SessionInsights(session_start=time.time() - 120)
        s.record(TurnUsage(model="gpt-4o", input_tokens=5000, output_tokens=2000))
        s.record(TurnUsage(model="gpt-4o-mini", input_tokens=1000, output_tokens=500, routed=True))

        output = s.format_summary()
        assert "2 turns" in output
        assert "Input:" in output
        assert "Cost:" in output
        assert "Routed:" in output
        assert "Models:" in output

    def test_format_single_model(self):
        s = SessionInsights(session_start=time.time())
        s.record(TurnUsage(model="gpt-4o", input_tokens=1000, output_tokens=500))
        output = s.format_summary()
        assert "Models:" not in output  # Only one model, no breakdown
