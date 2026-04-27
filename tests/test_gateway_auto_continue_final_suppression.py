"""Regression tests for suppressing terminal-looking final text in auto-continuation."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from caveman.agent.stream import StreamEvent
from caveman.gateway.task_runner import run_single_task


class _FakeLoop:
    def __init__(self, text: str, *, emit_tool: bool = False):
        self.text = text
        self.emit_tool = emit_tool

    async def run_stream(self, task: str):
        if self.emit_tool:
            yield StreamEvent(type="tool_call", data={"name": "bash", "args": {}})
        yield StreamEvent(type="result", data=self.text)


def _session(text: str):
    return {"agent_loop": _FakeLoop(text)}


def _session_with_tool_then_result(text: str):
    return {"agent_loop": _FakeLoop(text, emit_tool=True)}

def _store():
    store = SimpleNamespace(saved_turns=[])

    def append_turn(*args, **kwargs):
        store.saved_turns.append(args)

    store.append_turn = append_turn
    store.save_meta = lambda *a, **k: None
    return store


@pytest.mark.asyncio
async def test_auto_continue_flag_sanitizes_but_does_not_send_terminal_final_text():
    """Auto flywheel rounds keep sanitized text for chaining, but do not emit final paragraphs."""
    router = SimpleNamespace(send=AsyncMock())
    store = _store()
    source_channel = {"_auto_continue": True, "_progress_sent": 1}

    result = await run_single_task(
        "继续飞轮 (自动第 8/20 轮)",
        _session("本轮已完成。\n\nDone."),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    assert "Done" not in result
    assert "已完成" not in result
    assert "连续任务保持推进" in result
    router.send.assert_not_called()


@pytest.mark.asyncio
async def test_normal_task_still_sends_final_text():
    """Suppressing auto final text must not break ordinary replies."""
    router = SimpleNamespace(send=AsyncMock())
    store = _store()
    source_channel = {"_progress_sent": 0}

    result = await run_single_task(
        "普通问题",
        _session("普通回答"),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    assert result == "普通回答"
    router.send.assert_called_once_with("discord", "chan", "普通回答")


@pytest.mark.asyncio
async def test_auto_continue_persists_sanitized_assistant_turn():
    """Raw terminal final text must not be restored into later auto rounds."""
    router = SimpleNamespace(send=AsyncMock())
    store = _store()
    source_channel = {"_auto_continue": True, "_progress_sent": 1}

    result = await run_single_task(
        "继续飞轮 (自动第 10/20 轮)",
        _session("修了一个点。全部修复完毕。"),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    assistant_turns = [args for args in store.saved_turns if len(args) >= 3 and args[1] == "assistant"]
    assert assistant_turns
    saved_text = assistant_turns[-1][2]
    assert "全部修复完毕" not in saved_text
    assert "全部修复完毕" not in result
    assert "阶段性推进" in saved_text


@pytest.mark.asyncio
async def test_auto_continue_does_not_edit_heartbeat_to_stopped():
    """Continuation rounds must not visually announce that the stream stopped."""
    router = SimpleNamespace(send=AsyncMock(return_value="hb-1"), edit=AsyncMock())
    store = _store()
    source_channel = {"_auto_continue": True, "_progress_sent": 1}

    await run_single_task(
        "继续飞轮 (自动第 11/20 轮)",
        _session_with_tool_then_result("还有排查继续推进。"),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    assert source_channel.get("_hb_msg_id") == "hb-1"
    router.edit.assert_not_called()


@pytest.mark.asyncio
async def test_auto_continue_quiet_round_sends_non_terminal_progress_pulse():
    """Suppressing auto final text must not make a round look silently stopped."""
    router = SimpleNamespace(send=AsyncMock())
    store = _store()
    source_channel = {"_auto_continue": True, "_progress_sent": 0}

    result = await run_single_task(
        "继续飞轮 (自动第 12/20 轮)",
        _session("本轮已完成。Done."),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    assert "Done" not in result
    assert router.send.await_count == 1
    sent = router.send.await_args.args[2]
    assert "自动续轮仍在推进" in sent
    assert "不代表已完成或停止" in sent


@pytest.mark.asyncio
async def test_auto_continue_with_existing_progress_does_not_emit_extra_quiet_pulse():
    """A visible progress tool call is already the pulse; don't add another message."""
    router = SimpleNamespace(send=AsyncMock())
    store = _store()
    source_channel = {"_auto_continue": True, "_progress_sent": 1}

    await run_single_task(
        "继续飞轮 (自动第 13/20 轮)",
        _session("阶段性推进，继续排查。"),
        "discord",
        "chan",
        source_channel,
        router,
        store,
        {},
    )

    router.send.assert_not_called()
