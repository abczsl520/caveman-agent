"""Regression tests for gateway false completion replies.

The gateway must not fabricate "Done." when the agent returns an empty result.
Long-running flywheel tasks rely on empty/partial/incomplete states being visible,
not converted into success-looking terminal replies.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from caveman.gateway import runner
from caveman.gateway.runner import GatewayServer
from caveman.gateway.message_pipeline import DedupeCache


@pytest.mark.asyncio
async def test_gateway_handle_task_does_not_fabricate_done_for_empty_agent_result(monkeypatch):
    """Empty run_single_task result means no extra reply, not "Done."."""
    server = GatewayServer.__new__(GatewayServer)
    server.sessions = {}
    server.session_locks = {}
    server.router = AsyncMock()
    server.store = AsyncMock()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())

    async def fake_get_or_create_session(key):
        return {"loop": SimpleNamespace(tool_registry=SimpleNamespace(set_context=lambda *a, **k: None)),
                "meta": SimpleNamespace(session_id="s1"),
                "task_count": 0}

    async def fake_run_single_task(*args, **kwargs):
        return ""

    monkeypatch.setattr(server, "_get_or_create_session", fake_get_or_create_session)
    monkeypatch.setattr("caveman.gateway.runner.run_single_task", fake_run_single_task)

    result = await server.handle_task("继续排查重要问题", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg1",
    })

    assert result == ""
    assert result != "Done."


@pytest.mark.asyncio
async def test_adapter_message_handler_forwards_attachments_to_handle_task(monkeypatch):
    """Discord image attachments must reach handle_task as structured attachments."""
    server = GatewayServer.__new__(GatewayServer)
    captured = {}

    async def fake_handle_task(task, context):
        captured["task"] = task
        captured["context"] = context
        return ""

    monkeypatch.setattr(server, "handle_task", fake_handle_task)

    event = SimpleNamespace(
        text="你这个问题为什么会这样",
        message_id="m1",
        media_urls=["https://cdn.discordapp.com/attachments/x/image.png"],
        media_types=["image/png"],
        reply_to_text=None,
        is_mention=False,
        is_reply_to_bot=False,
        source=SimpleNamespace(
            chat_id="c1",
            user_id="u1",
            user_name="元宝",
            platform=SimpleNamespace(value="discord"),
            chat_type="channel",
            thread_id="",
        ),
    )

    result = await server._adapter_message_handler(event)

    assert result == ""
    assert captured["context"]["attachments"] == [{
        "url": "https://cdn.discordapp.com/attachments/x/image.png",
        "content_type": "image/png",
    }]


def test_gateway_runner_imports_timeout_constants_for_runtime_paths():
    """Auto-continue and session reaper must not crash with NameError at runtime."""
    assert isinstance(runner.TASK_DEFAULT, (int, float))
    assert isinstance(runner.TASK_SHORT, (int, float))
    assert runner.TASK_DEFAULT > 0
    assert runner.TASK_SHORT > 0


def test_explicit_continue_flywheel_message_triggers_auto_mode():
    """User-visible auto-round prompts must re-enter the auto-continue path."""
    assert runner._AUTO_PATTERNS.search("继续飞轮 (自动第 6/20 轮)。上一轮结果摘要：")
    assert runner._AUTO_PATTERNS.search("继续 飞轮，继续下一个最高复利的改进")
    assert runner._AUTO_PATTERNS.search("自动第 12 / 20 轮")


def test_auto_continue_prompt_avoids_terminal_completion_wording():
    """Auto-continue prompt should not tell the model to emit terminal completion copy."""
    import inspect

    source = inspect.getsource(runner.GatewayServer._auto_continue)
    assert "完成后报告" not in source
    assert "终止性收尾" in source
    assert "不代表所有问题已完成" in source
