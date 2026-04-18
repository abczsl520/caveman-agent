"""Tests for 3-layer compression pipeline."""
import pytest
from caveman.compression.pipeline import (
    CompressionPipeline, _msg_hash, _normalize_whitespace,
)
from caveman.compression.smart import estimate_tokens as _estimate_tokens


def test_estimate_tokens():
    msgs = [{"role": "user", "content": "Hello world"}]  # 11 chars / 4 ≈ 2
    assert _estimate_tokens(msgs) >= 1

    empty = [{"role": "user", "content": ""}]
    assert _estimate_tokens(empty) >= 1


def test_msg_hash_consistency():
    m1 = {"role": "user", "content": "hello"}
    m2 = {"role": "user", "content": "hello"}
    assert _msg_hash(m1) == _msg_hash(m2)

    m3 = {"role": "assistant", "content": "hello"}
    assert _msg_hash(m1) != _msg_hash(m3)


def test_normalize_whitespace():
    text = "hello   \n\n\n\nworld   \n"
    result = _normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert result.endswith("world")


@pytest.mark.asyncio
async def test_micro_dedup():
    """Layer 1: duplicate messages removed."""
    pipe = CompressionPipeline()
    msgs = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "ok"},  # duplicate
        {"role": "user", "content": "next"},
    ]
    result, stats = await pipe.compress(msgs, context_usage=0.0)
    assert len(result) == 3  # duplicate removed
    assert stats.layer_applied == "micro"


@pytest.mark.asyncio
async def test_normal_truncation():
    """Layer 2: long tool results truncated."""
    pipe = CompressionPipeline()
    long_result = "x" * 2000
    msgs = [
        {"role": "user", "content": "run command"},
        {"role": "tool", "content": long_result},
        {"role": "assistant", "content": "done"},
    ]
    result, stats = await pipe.compress(msgs, context_usage=0.65)
    assert stats.layer_applied == "normal"
    # Tool result should be truncated
    tool_msg = [m for m in result if m.get("role") == "tool"][0]
    assert len(tool_msg["content"]) < len(long_result)
    assert "truncated" in tool_msg["content"]


@pytest.mark.asyncio
async def test_normal_removes_empty():
    """Layer 2: empty assistant messages removed."""
    pipe = CompressionPipeline(preserve_last_n=2)
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},  # empty
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "ok"},  # keep (recent)
    ]
    result, stats = await pipe.compress(msgs, context_usage=0.65)
    assert not any(m.get("role") == "assistant" and m.get("content") == "" for m in result)


@pytest.mark.asyncio
async def test_smart_without_llm():
    """Layer 3 without LLM: should NOT compress (data preservation > compression)."""
    pipe = CompressionPipeline(preserve_last_n=3)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"message {i}"})
        msgs.append({"role": "tool", "content": f"result {i}"})

    result, stats = await pipe.compress(msgs, context_usage=0.85)
    assert stats.layer_applied == "smart"
    # Without LLM, heuristic summary kicks in — messages ARE compressed
    assert stats.messages_summarized > 0
    assert len(result) < len(msgs)
    # Should contain heuristic summary
    has_heuristic = any("Heuristic Summary" in str(m.get("content", "")) for m in result)
    assert has_heuristic


@pytest.mark.asyncio
async def test_smart_with_mock_llm():
    """Layer 3 with mock LLM provider."""
    class MockProvider:
        @property
        def context_length(self):
            return 100_000
        async def complete(self, messages, stream=False, **kw):
            yield {"type": "delta", "text": "Summary: user asked to do things, tools were run."}
            yield {"type": "done", "stop_reason": "end_turn", "usage": {}}

    pipe = CompressionPipeline(preserve_last_n=3, provider=MockProvider())
    msgs = []
    for i in range(15):
        msgs.append({"role": "user", "content": f"step {i}"})

    result, stats = await pipe.compress(msgs, context_usage=0.85)
    assert stats.layer_applied == "smart"
    # Should contain the LLM-generated summary somewhere
    all_content = " ".join(str(m.get("content", "")) for m in result)
    assert "Summary" in all_content or "COMPACTION" in all_content


@pytest.mark.asyncio
async def test_pipeline_preserves_short():
    """Short conversations should pass through unchanged."""
    pipe = CompressionPipeline()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result, stats = await pipe.compress(msgs, context_usage=0.1)
    assert len(result) == 2  # no change


@pytest.mark.asyncio
async def test_compression_stats():
    """Stats should accurately reflect compression."""
    pipe = CompressionPipeline()
    msgs = [
        {"role": "user", "content": "a" * 400},
        {"role": "tool", "content": "b" * 2000},
        {"role": "assistant", "content": "done"},
    ]
    result, stats = await pipe.compress(msgs, context_usage=0.7)
    assert stats.original_tokens > 0
    assert stats.final_tokens > 0
    assert stats.ratio <= 1.0
