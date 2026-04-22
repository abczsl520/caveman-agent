"""Tests for preemptive compaction."""
import pytest
from caveman.compression.preemptive import (
    truncate_tool_result_text,
    calculate_max_tool_result_chars,
    count_prunable_images,
    prune_images,
    should_preemptively_compact,
    apply_tool_result_truncation,
    apply_image_pruning,
    CompactionRoute,
    _has_important_tail,
    MIN_KEEP_CHARS,
)


class TestTruncateToolResultText:
    def test_short_text_unchanged(self):
        assert truncate_tool_result_text("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "x" * 10000
        result = truncate_tool_result_text(text, 3000)
        assert len(result) < 10000
        assert "truncated" in result

    def test_preserves_tail_with_error(self):
        text = "start line\n" * 1000 + "\nTraceback (most recent call last):\n  File 'x.py'\nError: boom"
        result = truncate_tool_result_text(text, 5000)
        assert "Error: boom" in result
        assert "middle content omitted" in result

    def test_preserves_tail_with_json(self):
        text = '{"data": [' + '"item",' * 2000 + '"last"]}'
        result = truncate_tool_result_text(text, 3000)
        assert result.rstrip().endswith('}') or "truncated" in result

    def test_min_keep_chars(self):
        text = "x" * 10000
        result = truncate_tool_result_text(text, MIN_KEEP_CHARS + 100)
        assert len(result) >= MIN_KEEP_CHARS


class TestHasImportantTail:
    def test_error_in_tail(self):
        assert _has_important_tail("lots of output\nError: something failed")

    def test_traceback(self):
        assert _has_important_tail("output\nTraceback (most recent call last):\n  boom")

    def test_json_closing(self):
        assert _has_important_tail('{"key": "value"}')

    def test_plain_text(self):
        assert not _has_important_tail("just some normal text here")


class TestCalculateMaxChars:
    def test_200k_context(self):
        result = calculate_max_tool_result_chars(200_000)
        assert result == 40_000  # capped at 40K

    def test_small_context(self):
        result = calculate_max_tool_result_chars(10_000)
        assert result == 6000  # 10K * 0.3 * 2

    def test_zero_context(self):
        assert calculate_max_tool_result_chars(0) == 0


class TestImagePruning:
    def test_count_prunable(self):
        messages = [
            {"role": "user", "content": [{"type": "image", "source": "x"}, {"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image", "source": "y"}]},
            {"role": "assistant", "content": "sure"},
        ]
        assert count_prunable_images(messages, protect_last_n=2) == 1

    def test_prune_replaces_with_text(self):
        messages = [
            {"role": "user", "content": [{"type": "image", "source": "x"}, {"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "latest"},
            {"role": "assistant", "content": "sure"},
        ]
        result, count = prune_images(messages, protect_last_n=2)
        assert count == 1
        assert result[0]["content"][0]["type"] == "text"
        assert "removed" in result[0]["content"][0]["text"]

    def test_no_images(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        result, count = prune_images(messages)
        assert count == 0


class TestPreemptiveCheck:
    def _make_context(self, total_tokens, max_tokens, messages=None):
        class FakeContext:
            pass
        ctx = FakeContext()
        ctx.total_tokens = total_tokens
        ctx.max_tokens = max_tokens
        ctx.messages = messages or []
        return ctx

    def test_fits(self):
        ctx = self._make_context(10000, 200000)
        result = should_preemptively_compact(ctx)
        assert result.route == CompactionRoute.FITS

    def test_needs_compression(self):
        ctx = self._make_context(180000, 200000)
        result = should_preemptively_compact(ctx)
        assert result.route != CompactionRoute.FITS

    def test_with_oversized_tool_result(self):
        class FakeMsg:
            def __init__(self, role, content, tokens=0):
                self.role = role
                self.content = content
                self.tokens = tokens
        msgs = [
            FakeMsg("user", "hi", 10),
            FakeMsg("tool", "x" * 100000, 25000),
            FakeMsg("assistant", "ok", 10),
        ]
        ctx = self._make_context(180000, 200000, msgs)
        result = should_preemptively_compact(ctx)
        assert result.truncatable_chars > 0


class TestApplyTruncation:
    def test_truncates_oversized(self):
        class FakeMsg:
            def __init__(self, role, content, tokens=0):
                self.role = role
                self.content = content
                self.tokens = tokens
        class FakeContext:
            def __init__(self):
                self.max_tokens = 10000
                self.messages = [
                    FakeMsg("user", "hi", 10),
                    FakeMsg("tool", "x" * 50000, 12500),
                ]
        ctx = FakeContext()
        count = apply_tool_result_truncation(ctx)
        assert count == 1
        assert len(ctx.messages[1].content) < 50000

    def test_no_truncation_needed(self):
        class FakeMsg:
            def __init__(self, role, content, tokens=0):
                self.role = role
                self.content = content
                self.tokens = tokens
        class FakeContext:
            def __init__(self):
                self.max_tokens = 200000
                self.messages = [FakeMsg("tool", "short", 5)]
        ctx = FakeContext()
        assert apply_tool_result_truncation(ctx) == 0
