"""Test closing marker termination policy."""
import asyncio
import pytest
from caveman.agent.loop_engines import check_termination

CLOSING = "\u2705---\u672c\u8f6e\u5df2\u5b8c\u6210---\u2705"

@pytest.mark.asyncio
async def test_closing_marker_does_not_terminate_with_tool_calls():
    """Executable tool calls take precedence over premature closing text."""
    result = await check_termination(
        stop="tool_use",
        tool_calls=[{"name": "bash", "id": "1", "input": {}}],
        task="test",
        text=f"All done.\n\n{CLOSING}",
    )
    assert result is False

@pytest.mark.asyncio
async def test_closing_marker_terminates_without_tool_calls():
    """Closing marker with no tool_calls also terminates."""
    result = await check_termination(
        stop="end_turn",
        tool_calls=[],
        task="test",
        text=f"Summary here.\n\n{CLOSING}",
    )
    assert result is True

@pytest.mark.asyncio
async def test_no_closing_marker_with_tool_calls_continues():
    """Without closing marker, tool_calls means continue."""
    result = await check_termination(
        stop="tool_use",
        tool_calls=[{"name": "bash", "id": "1", "input": {}}],
        task="test",
        text="Still working on it...",
    )
    assert result is False

@pytest.mark.asyncio
async def test_backward_compat_no_text_arg():
    """Old callers without text= still work."""
    result = await check_termination(
        stop="end_turn",
        tool_calls=[],
        task="test",
    )
    assert result is True

@pytest.mark.asyncio
async def test_tool_calls_no_text():
    """tool_calls with empty text continues."""
    result = await check_termination(
        stop="tool_use",
        tool_calls=[{"name": "bash", "id": "1", "input": {}}],
        task="test",
        text="",
    )
    assert result is False
