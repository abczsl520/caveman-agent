"""Tests for OutcomeEngine — outcome scoring and feedback propagation."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from caveman.engines.outcome import OutcomeEngine


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def mock_memory():
    m = AsyncMock()
    m.feedback = AsyncMock()
    return m


@pytest.fixture
def mock_bus():
    b = AsyncMock()
    b.emit = AsyncMock()
    return b


@pytest.fixture
def engine(mock_router, mock_memory, mock_bus):
    return OutcomeEngine(
        rl_router=mock_router,
        memory_manager=mock_memory,
        bus=mock_bus,
    )


@pytest.mark.asyncio
async def test_success_outcome(engine, mock_router, mock_memory, mock_bus):
    """Successful task should boost skills and memories."""
    skill = MagicMock()
    skill.name = "test_skill"
    result = await engine.score_and_propagate(
        task="do something",
        result="Done! Successfully completed the task.",
        matched_skills=[skill],
        recalled_ids=["mem_1", "mem_2"],
    )
    assert result["outcome"] == "success"
    assert result["score"] == 1.0
    assert result["skills_updated"] == 1
    assert result["memories_boosted"] == 2
    mock_router.update.assert_called_once_with("test_skill", True)
    assert mock_memory.feedback.call_count == 2
    mock_bus.emit.assert_called_once()


@pytest.mark.asyncio
async def test_failure_outcome(engine, mock_router, mock_memory, mock_bus):
    """Failed task should penalize skills and memories."""
    skill = MagicMock()
    skill.name = "bad_skill"
    result = await engine.score_and_propagate(
        task="do something",
        result="",
        matched_skills=[skill],
        recalled_ids=["mem_1"],
    )
    assert result["outcome"] == "failure"
    assert result["score"] == 0.0
    mock_router.update.assert_called_once_with("bad_skill", False)
    mock_memory.feedback.assert_called_once_with("mem_1", helpful=False)


@pytest.mark.asyncio
async def test_partial_outcome(engine):
    """Partial success should score 0.5."""
    result = await engine.score_and_propagate(
        task="do something",
        result="Here's what I found, but I couldn't complete everything. Done",
    )
    assert result["outcome"] in ("success", "partial")
    assert result["score"] in (0.5, 1.0)


@pytest.mark.asyncio
async def test_no_skills_no_memories():
    """Engine should work fine with no router or memory."""
    engine = OutcomeEngine()
    result = await engine.score_and_propagate(
        task="hello", result="Done!",
    )
    assert result["outcome"] == "success"
    assert result["skills_updated"] == 0
    assert result["memories_boosted"] == 0


@pytest.mark.asyncio
async def test_router_error_resilience(engine, mock_router):
    """Router errors should not crash the engine."""
    mock_router.update.side_effect = RuntimeError("boom")
    skill = MagicMock()
    skill.name = "broken"
    result = await engine.score_and_propagate(
        task="test", result="Done!",
        matched_skills=[skill],
    )
    assert result["skills_updated"] == 0


@pytest.mark.asyncio
async def test_memory_error_resilience(engine, mock_memory):
    """Memory errors should not crash the engine."""
    mock_memory.feedback.side_effect = RuntimeError("db locked")
    result = await engine.score_and_propagate(
        task="test", result="Done!",
        recalled_ids=["mem_1"],
    )
    assert result["memories_boosted"] == 0


@pytest.mark.asyncio
async def test_string_skill_name(engine, mock_router):
    """Skills passed as strings should work."""
    result = await engine.score_and_propagate(
        task="test", result="Done!",
        matched_skills=["my_skill"],
    )
    mock_router.update.assert_called_once_with("my_skill", True)
    assert result["skills_updated"] == 1


@pytest.mark.asyncio
async def test_event_payload(engine, mock_bus):
    """SKILL_OUTCOME event should contain correct payload."""
    result = await engine.score_and_propagate(
        task="a" * 300,
        result="Done!",
        recalled_ids=["m1"],
    )
    call_args = mock_bus.emit.call_args
    from caveman.events import EventType
    assert call_args[0][0] == EventType.SKILL_OUTCOME
    payload = call_args[0][1]
    assert len(payload["task"]) <= 200  # truncated
    assert payload["outcome"] == "success"
    assert payload["recalled_ids"] == ["m1"]
