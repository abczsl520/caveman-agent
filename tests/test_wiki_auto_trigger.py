"""Tests for WikiAutoTrigger — auto-compilation on memory accumulation."""
import pytest
from unittest.mock import MagicMock, patch
from caveman.wiki.auto_trigger import WikiAutoTrigger


@pytest.fixture
def mock_compiler():
    c = MagicMock()
    result = MagicMock()
    result.entries_promoted = 2
    result.entries_expired = 1
    result.entries_processed = 10
    c.compile.return_value = result
    return c


@pytest.fixture
def trigger(mock_compiler):
    return WikiAutoTrigger(compiler=mock_compiler, threshold=3, cooldown=0)


def test_accumulation_below_threshold(trigger, mock_compiler):
    """Should not compile when below threshold."""
    assert trigger.on_nudge_extract(1) is False
    assert trigger.on_nudge_extract(1) is False
    mock_compiler.compile.assert_not_called()
    assert trigger.accumulated == 2


def test_accumulation_triggers_at_threshold(trigger, mock_compiler):
    """Should compile when threshold reached."""
    trigger.on_nudge_extract(1)
    trigger.on_nudge_extract(1)
    result = trigger.on_nudge_extract(1)
    assert result is True
    mock_compiler.compile.assert_called_once()
    assert trigger.accumulated == 0  # reset after compile


def test_batch_accumulation(trigger, mock_compiler):
    """Should handle batch counts."""
    result = trigger.on_nudge_extract(5)
    assert result is True
    mock_compiler.compile.assert_called_once()


def test_cooldown_prevents_rapid_compile():
    """Should respect cooldown between compilations."""
    compiler = MagicMock()
    result = MagicMock()
    result.entries_promoted = 0
    result.entries_expired = 0
    result.entries_processed = 5
    compiler.compile.return_value = result

    trigger = WikiAutoTrigger(compiler=compiler, threshold=1, cooldown=9999)
    # First compile should work
    assert trigger.on_nudge_extract(1) is True
    # Second should be blocked by cooldown
    assert trigger.on_nudge_extract(1) is False


def test_no_compiler():
    """Should handle missing compiler gracefully."""
    trigger = WikiAutoTrigger(compiler=None, threshold=1, cooldown=0)
    assert trigger.on_nudge_extract(1) is False


def test_force_compile(trigger, mock_compiler):
    """force_compile should bypass threshold."""
    assert trigger.accumulated == 0
    assert trigger.force_compile() is True
    mock_compiler.compile.assert_called_once()


def test_compiler_error_resilience():
    """Should handle compiler errors gracefully."""
    compiler = MagicMock()
    compiler.compile.side_effect = RuntimeError("disk full")
    trigger = WikiAutoTrigger(compiler=compiler, threshold=1, cooldown=0)
    assert trigger.on_nudge_extract(1) is False


def test_ingest_high_trust_memories(mock_compiler):
    """Should ingest high-trust, frequently-retrieved memories."""
    mem = MagicMock()
    mem.content = "Python uses 0-based indexing"
    mem.metadata = {"trust_score": 0.9, "retrieval_count": 5, "source": "nudge"}

    memory_mgr = MagicMock()
    memory_mgr.all_entries = [mem]

    trigger = WikiAutoTrigger(
        compiler=mock_compiler, memory_manager=memory_mgr,
        threshold=1, cooldown=0,
    )
    trigger.on_nudge_extract(1)
    mock_compiler.ingest.assert_called_once()
    call_kwargs = mock_compiler.ingest.call_args
    assert call_kwargs[1]["confidence"] == 0.9
    assert "auto-ingested" in call_kwargs[1]["tags"]
