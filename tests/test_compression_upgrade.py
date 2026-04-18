"""Tests for the upgraded compression system (Hermes + OpenClaw port)."""
from __future__ import annotations

import asyncio
import pytest

from caveman.compression.smart import (
    SmartCompressor,
    SUMMARY_PREFIX,
)
from caveman.compression.utils import (
    sanitize_tool_pairs,
    estimate_tokens,
    align_forward,
    align_backward,
    build_template,
    IDENTIFIER_PRESERVATION,
    _PRUNED_TOOL_PLACEHOLDER,
)
from caveman.compression.pipeline import CompressionPipeline, CompressionStats


# --- Helper ---

def _make_messages(n: int, with_tools: bool = False) -> list[dict]:
    """Generate n alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append({"role": "user", "content": f"User message {i}"})
        else:
            if with_tools and i % 4 == 1:
                msgs.append({
                    "role": "assistant",
                    "content": f"Assistant {i}",
                    "tool_calls": [{"id": f"call_{i}", "function": {"name": "test", "arguments": "{}"}}],
                })
                msgs.append({
                    "role": "tool",
                    "content": f"Tool result for call_{i}" * 50,
                    "tool_call_id": f"call_{i}",
                })
            else:
                msgs.append({"role": "assistant", "content": f"Assistant message {i}"})
    return msgs


# --- SmartCompressor tests ---

class TestSmartCompressor:
    def test_init_defaults(self):
        sc = SmartCompressor()
        assert sc.context_length == 200_000
        assert sc.threshold_percent == 0.75
        assert sc.protect_first_n == 3
        assert sc.compression_count == 0
        assert sc._previous_summary is None

    def test_init_custom(self):
        sc = SmartCompressor(context_length=100_000, threshold_percent=0.5)
        assert sc.threshold_tokens == 50_000
        assert sc.tail_token_budget == int(50_000 * 0.20)

    def test_reset(self):
        sc = SmartCompressor()
        sc.compression_count = 5
        sc._previous_summary = "old"
        sc._cooldown_until = 999.0
        sc.reset()
        assert sc.compression_count == 0
        assert sc._previous_summary is None
        assert sc._cooldown_until == 0.0

    def test_prune_tool_results_small(self):
        """Small tool results should not be pruned."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "short result", "tool_call_id": "c1"},
        ]
        sc = SmartCompressor()
        pruned, count = sc.prune_tool_results(msgs)
        assert count == 0
        assert pruned[1]["content"] == "short result"

    def test_prune_tool_results_large(self):
        """Large old tool results should be pruned when outside tail budget."""
        # Put the large tool result early, with many recent messages after
        # so the tool result falls outside the tail token budget
        msgs = [
            {"role": "tool", "content": "x" * 500, "tool_call_id": "c1"},
        ] + [{"role": "user", "content": f"message number {i} with some content"} for i in range(20)] + [
            {"role": "assistant", "content": "final reply with details"},
        ]
        sc = SmartCompressor()
        # Small budget so the old tool result at index 0 is outside the tail
        pruned, count = sc.prune_tool_results(msgs, tail_token_budget=100)
        assert count == 1
        assert pruned[0]["content"] == _PRUNED_TOOL_PLACEHOLDER

    def test_find_tail_cut_basic(self):
        msgs = _make_messages(20)
        sc = SmartCompressor()
        cut = sc.find_tail_cut(msgs, head_end=3)
        assert 3 < cut < 20

    def test_find_tail_cut_small_conversation(self):
        """Small conversations should still allow compression."""
        msgs = _make_messages(8)
        sc = SmartCompressor()
        cut = sc.find_tail_cut(msgs, head_end=3)
        assert cut >= 4  # At least after head

    def test_serialize_turns(self):
        from caveman.compression.utils import serialize_turns
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "calc", "arguments": '{"expr": "2+2"}'}}
            ]},
            {"role": "tool", "content": "4", "tool_call_id": "c1"},
        ]
        text = serialize_turns(msgs)
        assert "[USER]:" in text
        assert "[ASSISTANT]:" in text
        assert "[TOOL RESULT c1]:" in text
        assert "calc(" in text

    def test_compress_too_few_messages(self):
        """Should return messages unchanged if too few."""
        msgs = _make_messages(4)
        sc = SmartCompressor()
        result = asyncio.run(sc.compress(msgs))
        assert result == msgs

    def test_compress_no_provider_fallback(self):
        """Without provider, heuristic summary kicks in → messages compressed."""
        msgs = _make_messages(20)
        sc = SmartCompressor(provider=None)
        result = asyncio.run(sc.compress(msgs))
        # Heuristic summary works without LLM
        assert len(result) < len(msgs)
        has_heuristic = any("Heuristic Summary" in str(m.get("content", "")) for m in result)
        assert has_heuristic

    def test_compress_increments_count(self):
        msgs = _make_messages(20)
        sc = SmartCompressor()
        asyncio.run(sc.compress(msgs))
        # Heuristic summary works → compression count increments
        assert sc.compression_count == 1

    def test_compress_system_note_first_time(self):
        """First compression with heuristic: system message gets compaction note."""
        msgs = [{"role": "system", "content": "You are helpful."}] + _make_messages(20)
        sc = SmartCompressor()
        result = asyncio.run(sc.compress(msgs))
        # Heuristic compression works → fewer messages
        assert len(result) < len(msgs)
        # System message should have compaction note
        assert "compacted" in result[0]["content"].lower() or "summary" in result[0]["content"].lower()

    def test_compress_preserves_head(self):
        """Head messages should be preserved."""
        msgs = _make_messages(20)
        sc = SmartCompressor(protect_first_n=3)
        result = asyncio.run(sc.compress(msgs))
        # First 3 messages should be preserved (content may have note appended)
        for i in range(min(3, len(result))):
            assert result[i]["role"] == msgs[i]["role"]


# --- Tool pair sanitization tests ---

class TestSanitizeToolPairs:
    def test_clean_messages_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = sanitize_tool_pairs(msgs)
        assert result == msgs

    def test_orphaned_tool_result_removed(self):
        """Tool result without matching assistant tool_call should be removed."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "orphan_id"},
            {"role": "assistant", "content": "ok"},
        ]
        result = sanitize_tool_pairs(msgs)
        assert len(result) == 2
        assert not any(m.get("role") == "tool" for m in result)

    def test_missing_tool_result_stubbed(self):
        """Assistant tool_call without result should get stub."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "test", "arguments": "{}"}}
            ]},
        ]
        result = sanitize_tool_pairs(msgs)
        assert len(result) == 3
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_1"
        assert "earlier" in result[2]["content"]

    def test_matched_pairs_preserved(self):
        """Properly matched tool pairs should be unchanged."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "test", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        result = sanitize_tool_pairs(msgs)
        assert len(result) == 2


# --- Boundary alignment tests ---

class TestBoundaryAlignment:
    def testalign_forward_skips_tools(self):
        msgs = [
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            {"role": "user", "content": "hi"},
        ]
        assert align_forward(msgs, 0) == 2

    def testalign_forward_no_tools(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert align_forward(msgs, 0) == 0

    def testalign_backward_pulls_past_tool_group(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "r1"},
            {"role": "user", "content": "next"},
        ]
        # Boundary at index 2 (tool) should pull back to 1 (assistant)
        assert align_backward(msgs, 2) == 1

    def testalign_backward_no_tool_group(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "next"},
        ]
        assert align_backward(msgs, 2) == 2


# --- Token estimation tests ---

class TestEstimateTokens:
    def test_basic(self):
        msgs = [{"role": "user", "content": "hello world"}]
        tokens = estimate_tokens(msgs)
        assert tokens > 0

    def test_with_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "c1", "function": {"name": "test", "arguments": '{"x": 1}' * 100}}
            ]},
        ]
        tokens = estimate_tokens(msgs)
        assert tokens > 100  # Should count tool call args

    def test_empty(self):
        assert estimate_tokens([]) == 1  # min 1


# --- Template tests ---

class TestTemplate:
    def test_basic_template(self):
        t = build_template(2000)
        assert "## Goal" in t
        assert "## Progress" in t
        assert "## Exact Identifiers" in t
        assert IDENTIFIER_PRESERVATION in t

    def test_focus_topic(self):
        t = build_template(2000, focus_topic="database migration")
        assert "database migration" in t
        assert "60-70%" in t


# --- Pipeline integration tests ---

class TestCompressionPipeline:
    def test_micro_only(self):
        msgs = _make_messages(10)
        pipeline = CompressionPipeline()
        result, stats = asyncio.run(pipeline.compress(msgs, context_usage=0.3))
        assert stats.layer_applied == "micro"

    def test_normal_layer(self):
        msgs = _make_messages(10)
        pipeline = CompressionPipeline()
        result, stats = asyncio.run(pipeline.compress(msgs, context_usage=0.65))
        assert stats.layer_applied == "normal"

    def test_smart_layer(self):
        msgs = _make_messages(20)
        pipeline = CompressionPipeline()
        result, stats = asyncio.run(pipeline.compress(msgs, context_usage=0.85))
        assert stats.layer_applied == "smart"
        # Heuristic summary works → messages compressed
        assert len(result) < len(msgs)

    def test_smart_compressor_accessible(self):
        pipeline = CompressionPipeline()
        assert isinstance(pipeline.smart_compressor, SmartCompressor)

    def test_focus_topic_passthrough(self):
        """focus_topic should be passed to smart compressor."""
        msgs = _make_messages(20)
        pipeline = CompressionPipeline()
        result, stats = asyncio.run(
            pipeline.compress(msgs, context_usage=0.85, focus_topic="test")
        )
        assert stats.layer_applied == "smart"

    def test_stats_ratio(self):
        stats = CompressionStats(original_tokens=1000, final_tokens=500)
        assert stats.ratio == 0.5

    def test_stats_ratio_zero(self):
        stats = CompressionStats(original_tokens=0, final_tokens=0)
        assert stats.ratio == 1.0

    def test_dedup_consecutive(self):
        """Micro layer should remove consecutive duplicate messages."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pipeline = CompressionPipeline()
        result, _ = asyncio.run(pipeline.compress(msgs, context_usage=0.1))
        assert len(result) == 2

    def test_normal_truncates_tool_results(self):
        """Normal layer should truncate long tool results."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "x" * 2000},
            {"role": "assistant", "content": "ok"},
        ] * 3 + [{"role": "user", "content": "recent"}] * 10
        pipeline = CompressionPipeline(max_tool_result_len=500)
        result, _ = asyncio.run(pipeline.compress(msgs, context_usage=0.65))
        for m in result:
            if m.get("role") == "tool":
                assert len(m["content"]) <= 600  # truncated + ellipsis
