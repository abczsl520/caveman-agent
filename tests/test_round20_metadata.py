"""Tests for Round 20 Phase 1 — Model Metadata + Usage Pricing."""
from __future__ import annotations

import pytest
from caveman.providers.model_metadata import (
    ModelInfo, get_model_info, strip_provider_prefix,
    is_local_endpoint, estimate_tokens, MINIMUM_CONTEXT_LENGTH,
)
from caveman.providers.usage_pricing import UsageTracker, TurnUsage, SessionUsage


class TestStripProviderPrefix:
    def test_slash_prefix(self):
        assert strip_provider_prefix("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_no_prefix(self):
        assert strip_provider_prefix("claude-opus-4-6") == "claude-opus-4-6"

    def test_colon_prefix(self):
        assert strip_provider_prefix("local:my-model") == "my-model"

    def test_ollama_tag_preserved(self):
        assert strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"

    def test_unknown_prefix(self):
        assert strip_provider_prefix("mycompany/my-model") == "mycompany/my-model"

    def test_http_url(self):
        assert strip_provider_prefix("http://localhost:11434") == "http://localhost:11434"


class TestModelInfo:
    def test_known_model(self):
        info = get_model_info("claude-opus-4-6")
        assert info.context_length == 1_000_000
        assert info.max_output_tokens == 32_000
        assert info.input_cost_per_mtok == 15.0
        assert info.supports_vision

    def test_with_provider_prefix(self):
        info = get_model_info("anthropic/claude-opus-4-6")
        assert info.context_length == 1_000_000

    def test_fuzzy_match(self):
        info = get_model_info("claude-opus-4-6-20260415")
        assert info.context_length == 1_000_000

    def test_unknown_model(self):
        info = get_model_info("totally-unknown-model-xyz")
        assert info.context_length == 128_000  # default

    def test_cost_estimation(self):
        info = get_model_info("claude-opus-4-6")
        cost = info.estimate_cost(1_000_000, 500_000)
        assert cost == pytest.approx(15.0 + 37.5, abs=0.01)

    def test_free_model(self):
        info = get_model_info("qwen3-coder-plus")
        assert info.estimate_cost(1_000_000, 1_000_000) == 0.0

    def test_gpt_model(self):
        info = get_model_info("gpt-4.1")
        assert info.context_length == 1_047_576

    def test_deepseek(self):
        info = get_model_info("deepseek-chat")
        assert info.context_length == 128_000
        assert info.input_cost_per_mtok == 0.27


class TestEstimateTokens:
    def test_english(self):
        tokens = estimate_tokens("Hello world, this is a test.")
        assert 5 <= tokens <= 10

    def test_chinese(self):
        tokens = estimate_tokens("你好世界这是一个测试")
        assert 4 <= tokens <= 10

    def test_mixed(self):
        tokens = estimate_tokens("Hello 你好 world 世界")
        assert tokens > 0

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestIsLocalEndpoint:
    def test_localhost(self):
        assert is_local_endpoint("http://localhost:11434")

    def test_127(self):
        assert is_local_endpoint("http://127.0.0.1:8080")

    def test_docker_internal(self):
        assert is_local_endpoint("http://host.docker.internal:11434")

    def test_private_10(self):
        assert is_local_endpoint("http://10.0.0.1:8080")

    def test_private_192(self):
        assert is_local_endpoint("http://192.168.1.1:8080")

    def test_public(self):
        assert not is_local_endpoint("https://api.anthropic.com")

    def test_empty(self):
        assert not is_local_endpoint("")


class TestUsageTracker:
    def test_record_turn(self):
        tracker = UsageTracker()
        turn = tracker.record("s1", "claude-opus-4-6", input_tokens=1000, output_tokens=500)
        assert turn.total_tokens == 1500
        assert turn.cost_usd > 0

    def test_session_aggregation(self):
        tracker = UsageTracker()
        tracker.record("s1", "claude-opus-4-6", input_tokens=1000, output_tokens=500)
        tracker.record("s1", "claude-opus-4-6", input_tokens=2000, output_tokens=1000)
        session = tracker.get_session("s1")
        assert session.total_input_tokens == 3000
        assert session.total_output_tokens == 1500
        assert session.turn_count == 2

    def test_multiple_sessions(self):
        tracker = UsageTracker()
        tracker.record("s1", "claude-opus-4-6", input_tokens=1000, output_tokens=500)
        tracker.record("s2", "gpt-4.1", input_tokens=2000, output_tokens=1000)
        assert len(tracker.all_sessions) == 2
        assert tracker.total_tokens == 4500

    def test_format_summary(self):
        tracker = UsageTracker()
        tracker.record("s1", "claude-opus-4-6", input_tokens=1000, output_tokens=500)
        summary = tracker.get_session("s1").format_summary()
        assert "1.0K" in summary or "1000" in summary
        assert "turn" in summary

    def test_format_total(self):
        tracker = UsageTracker()
        tracker.record("s1", "claude-opus-4-6", input_tokens=1_000_000, output_tokens=500_000)
        total = tracker.format_total()
        assert "1.5M" in total

    def test_free_model_zero_cost(self):
        tracker = UsageTracker()
        turn = tracker.record("s1", "qwen3-coder-plus", input_tokens=100000, output_tokens=50000)
        assert turn.cost_usd == 0.0

    def test_empty_session(self):
        tracker = UsageTracker()
        session = tracker.get_session("nonexistent")
        assert session.total_tokens == 0
        assert session.turn_count == 0


class TestTurnUsage:
    def test_to_dict(self):
        turn = TurnUsage(model="test", input_tokens=100, output_tokens=50, cost_usd=0.001)
        d = turn.to_dict()
        assert d["total_tokens"] == 150
        assert d["cost_usd"] == 0.001


class TestSessionUsage:
    def test_to_dict(self):
        session = SessionUsage(session_id="test")
        session.turns.append(TurnUsage(input_tokens=100, output_tokens=50))
        d = session.to_dict()
        assert d["total_tokens"] == 150
        assert d["turn_count"] == 1
