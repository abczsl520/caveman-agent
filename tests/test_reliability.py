"""Tests for new reliability features: IterationBudget, FallbackChain, MessageSanitizer."""
import pytest
from caveman.agent.iteration_budget import IterationBudget


class TestIterationBudget:
    def test_consume_within_limit(self):
        b = IterationBudget(3)
        assert b.consume() is True
        assert b.consume() is True
        assert b.consume() is True
        assert b.consume() is False

    def test_refund(self):
        b = IterationBudget(2)
        b.consume()
        b.consume()
        assert b.exhausted
        b.refund(reason="test")
        assert not b.exhausted
        assert b.remaining == 1

    def test_remaining(self):
        b = IterationBudget(10)
        assert b.remaining == 10
        b.consume()
        assert b.remaining == 9
        assert b.used == 1

    def test_refund_at_zero(self):
        b = IterationBudget(5)
        b.refund()  # Should not go negative
        assert b.used == 0

    def test_repr(self):
        b = IterationBudget(50)
        b.consume()
        assert "1/50" in repr(b)


class TestFallbackChain:
    def test_empty_chain(self):
        from caveman.providers.fallback_chain import FallbackChain
        chain = FallbackChain([])
        assert not chain.has_fallbacks
        assert chain.exhausted
        assert chain.try_activate_next() is None

    def test_invalid_entries_skipped(self):
        from caveman.providers.fallback_chain import FallbackChain
        chain = FallbackChain([{"provider": "", "model": ""}])
        assert chain.try_activate_next() is None

    def test_reset(self):
        from caveman.providers.fallback_chain import FallbackChain
        chain = FallbackChain([{"provider": "unknown", "model": "x"}])
        chain.try_activate_next()
        assert chain.exhausted
        chain.reset()
        assert not chain.exhausted


class TestMessageSanitizer:
    def test_drop_invalid_roles(self):
        from caveman.providers.message_sanitizer import sanitize_messages
        msgs = [
            {"role": "system", "content": "hi"},
            {"role": "invalid", "content": "bad"},
            {"role": "user", "content": "hello"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 2
        assert all(m["role"] in ("system", "user") for m in result)

    def test_orphan_tool_result_removed(self):
        from caveman.providers.message_sanitizer import sanitize_messages
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "orphan_123"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 1

    def test_stub_injected_for_missing_result(self):
        from caveman.providers.message_sanitizer import sanitize_messages
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "bash", "arguments": "{}"}}
            ]},
        ]
        result = sanitize_messages(msgs)
        assert any(m.get("tool_call_id") == "call_1" for m in result)

    def test_surrogate_sanitization(self):
        from caveman.providers.message_sanitizer import sanitize_surrogates
        msgs = [{"role": "user", "content": "hello\ud800world"}]
        found = sanitize_surrogates(msgs)
        assert found
        assert "\ud800" not in msgs[0]["content"]
        assert "\ufffd" in msgs[0]["content"]

    def test_no_surrogates_noop(self):
        from caveman.providers.message_sanitizer import sanitize_surrogates
        msgs = [{"role": "user", "content": "hello world 你好"}]
        found = sanitize_surrogates(msgs)
        assert not found


class TestToolCallDedup:
    def test_dedup_same_name_args(self):
        from caveman.providers.message_sanitizer import deduplicate_tool_calls
        calls = [
            {"name": "bash", "input": {"command": "ls"}},
            {"name": "bash", "input": {"command": "ls"}},
            {"name": "file_read", "input": {"path": "a.py"}},
        ]
        result = deduplicate_tool_calls(calls)
        assert len(result) == 2

    def test_different_args_kept(self):
        from caveman.providers.message_sanitizer import deduplicate_tool_calls
        calls = [
            {"name": "bash", "input": {"command": "ls"}},
            {"name": "bash", "input": {"command": "pwd"}},
        ]
        result = deduplicate_tool_calls(calls)
        assert len(result) == 2

    def test_no_dupes_returns_original(self):
        from caveman.providers.message_sanitizer import deduplicate_tool_calls
        calls = [{"name": "bash", "input": {"command": "ls"}}]
        result = deduplicate_tool_calls(calls)
        assert result is calls  # Same object, no copy
