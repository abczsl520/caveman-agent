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
from caveman.gateway.session_context import get_session_env, set_session_context


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
    """Session reaper runtime constants must remain available."""
    assert isinstance(runner.TASK_SHORT, (int, float))
    assert runner.TASK_SHORT > 0


def test_explicit_continue_flywheel_message_triggers_auto_mode():
    """User-visible auto-round prompts must re-enter the auto-continue path."""
    assert runner._AUTO_PATTERNS.search("继续飞轮 (自动第 6/20 轮)。上一轮结果摘要：")
    assert runner._AUTO_PATTERNS.search("继续 飞轮，继续下一个最高复利的改进")
    assert runner._AUTO_PATTERNS.search("自动第 12 / 20 轮")


def test_running_task_nudges_are_not_interrupt_commands():
    """User impatience/status nudges must not be treated as stop-the-current-task commands."""
    positives = [
        "继续不要停",
        "怎么又停了",
        "咋做一半停了",
        "又停了",
        "没反应了",
        "是不是又失败了",
        "keep going",
    ]
    for text in positives:
        assert runner._is_non_interrupting_running_task_nudge(text), text

    negatives = [
        "换个方向，先查网关日志",
        "停止当前任务，做这个新任务",
        "帮我部署到服务器",
    ]
    for text in negatives:
        assert not runner._is_non_interrupting_running_task_nudge(text), text


def test_auto_continue_prompt_avoids_terminal_completion_wording():
    """Auto-continue prompt should not tell the model to emit terminal completion copy."""
    import inspect

    source = inspect.getsource(runner.GatewayServer._auto_continue)
    assert "完成后报告" not in source
    assert "终止性收尾" in source
    assert "asyncio.wait_for" not in source
    assert "TASK_DEFAULT" not in source


@pytest.mark.asyncio
async def test_running_continue_nudge_does_not_interrupt_active_task():
    """A follow-up like '继续不要停' should not set shutdown_flag on the active ctx."""
    server = GatewayServer.__new__(GatewayServer)
    lock = runner.asyncio.Lock()
    await lock.acquire()
    ctx = SimpleNamespace(shutdown_flag=False)
    active_key = "agent:main:discord:channel:chan"
    server.sessions = {
        active_key: {"_task_ctx": ctx, "_interrupt": False, "last_active": 0}
    }
    server.session_locks = {active_key: lock}
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())

    result = await server.handle_task("继续不要停", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg-nudge",
    })

    assert result == ""
    assert ctx.shutdown_flag is False
    assert server.sessions[active_key].get("_interrupt") is False
    sent_text = server.router.send.await_args.args[2]
    assert "不会中断" in sent_text
    assert "还在运行" in sent_text
    lock.release()


@pytest.mark.asyncio
async def test_regular_new_message_still_interrupts_active_task():
    """Non-nudge new tasks should keep the existing interrupt semantics."""
    server = GatewayServer.__new__(GatewayServer)
    lock = runner.asyncio.Lock()
    await lock.acquire()
    ctx = SimpleNamespace(shutdown_flag=False)
    active_key = "agent:main:discord:channel:chan"
    server.sessions = {
        active_key: {"_task_ctx": ctx, "_interrupt": False, "last_active": 0}
    }
    server.session_locks = {active_key: lock}
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())

    # Release shortly after the interrupt branch so handle_task can finish rather
    # than deadlocking the unit test on the pre-held lock.
    async def release_soon():
        await runner.asyncio.sleep(0)
        lock.release()

    runner.asyncio.create_task(release_soon())

    async def fake_get_or_create_session(key):
        return {"loop": SimpleNamespace(tool_registry=SimpleNamespace(set_context=lambda *a, **k: None)),
                "meta": SimpleNamespace(session_id="s1"),
                "task_count": 0}

    async def fake_run_single_task(*args, **kwargs):
        return ""

    from unittest.mock import patch
    with patch.object(server, "_get_or_create_session", fake_get_or_create_session), \
         patch.object(runner, "run_single_task", fake_run_single_task):
        result = await server.handle_task("换个方向，先查日志", {
            "gateway_name": "discord",
            "channel_id": "chan",
            "user_id": "user",
            "message_id": "msg-new-task",
        })

    assert result == ""
    assert ctx.shutdown_flag is True
    assert server.sessions[active_key].get("_interrupt") is True
    sent_text = server.router.send.await_args_list[0].args[2]
    assert "正在停止当前任务" in sent_text


@pytest.mark.asyncio
async def test_auto_continue_passes_suppression_flag_to_inner_task(monkeypatch):
    """Each auto round marks source_channel so final text is not user-emitted."""
    server = GatewayServer.__new__(GatewayServer)
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    seen = []

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        seen.append((task, dict(source_channel)))
        if len(seen) >= 2:
            raise KeyboardInterrupt("stop test after proving continuation")
        raise runner.AgentTaskError("stop after first captured round")

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    with pytest.raises(KeyboardInterrupt):
        await server._auto_continue(
            "上一轮摘要",
            {"auto_rounds": 0},
            "discord",
            "chan",
            {"_progress_sent": 99},
        )

    assert len(seen) == 2
    task, source = seen[0]
    assert "完成后报告" not in task
    assert source["_auto_continue"] is True
    assert source["_progress_sent"] == 0


@pytest.mark.asyncio
async def test_auto_continue_trigger_suppresses_initial_user_visible_final(monkeypatch):
    """The user-sent auto-round prompt is itself continuous, not a final-report turn."""
    server = GatewayServer.__new__(GatewayServer)
    server.sessions = {}
    server.session_locks = {}
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())

    captured_sources = []

    async def fake_get_or_create_session(key):
        return {"loop": SimpleNamespace(tool_registry=SimpleNamespace(set_context=lambda *a, **k: None)),
                "meta": SimpleNamespace(session_id="s1"),
                "task_count": 0}

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        captured_sources.append(dict(source_channel))
        return "像最终报告的文本"

    async def fake_auto_continue(result, session, gw_name, channel_id, source_channel):
        return result

    monkeypatch.setattr(server, "_get_or_create_session", fake_get_or_create_session)
    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)
    monkeypatch.setattr(server, "_auto_continue", fake_auto_continue)

    result = await server.handle_task("继续飞轮 (自动第 7/20 轮)。继续下一个最高复利的改进。", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg-auto-initial",
    })

    assert result == ""
    assert captured_sources
    assert captured_sources[0]["_auto_continue"] is True



@pytest.mark.asyncio
async def test_handle_task_exception_reports_diagnostic_not_generic_english(monkeypatch):
    """Gateway exceptions must be diagnosable and must not emit the old generic English error."""
    server = GatewayServer.__new__(GatewayServer)
    server.sessions = {}
    server.session_locks = {}
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())

    async def fake_get_or_create_session(key):
        return {"loop": SimpleNamespace(), "meta": SimpleNamespace(session_id="s1"), "task_count": 0}

    async def fake_run_single_task(*args, **kwargs):
        raise RuntimeError("boom-for-diagnostics")

    monkeypatch.setattr(server, "_get_or_create_session", fake_get_or_create_session)
    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    result = await server.handle_task("普通任务", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg-error",
    })

    assert "Something went wrong" not in result
    assert "Please try again" not in result
    assert "RuntimeError" in result
    assert "boom-for-diagnostics" in result
    assert "不是任务完成信号" in result


@pytest.mark.asyncio
async def test_auto_continue_unexpected_exception_reports_and_continues(monkeypatch):
    """One crashed auto round should not make the 20-round flywheel look stopped."""
    server = GatewayServer.__new__(GatewayServer)
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    calls = []

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        calls.append(task)
        if len(calls) == 1:
            raise RuntimeError("round-one-crash")
        if len(calls) == 2:
            raise runner.AgentTaskError("stop after second round")
        raise KeyboardInterrupt("stop test after proving continuation")

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    with pytest.raises(KeyboardInterrupt):
        await server._auto_continue(
            "上一轮摘要",
            {"auto_rounds": 0},
            "discord",
            "chan",
            {"_progress_sent": 99},
        )

    assert len(calls) == 3
    sent_texts = [call.args[2] for call in server.router.send.await_args_list]
    assert any("round-one-crash" in text or "运行异常" in text for text in sent_texts)
    assert any("agent 错误" in text for text in sent_texts)
    assert any("继续下一轮排查" in text for text in sent_texts)
    assert not any("暂停" in text for text in sent_texts)


@pytest.mark.asyncio
async def test_auto_continue_iteration_exhaustion_is_visible_and_continues(monkeypatch):
    """Iteration-budget exhaustion in an auto round is a visible non-terminal signal."""
    server = GatewayServer.__new__(GatewayServer)
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    calls = []

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        calls.append(task)
        if len(calls) == 1:
            return "⚠️ 已达到本轮迭代上限（50），任务没有被验证为完成。"
        if len(calls) == 2:
            raise runner.AgentTaskError("Max iterations (50) reached — budget exhausted")
        raise KeyboardInterrupt("stop test after proving continuation")

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    with pytest.raises(KeyboardInterrupt):
        await server._auto_continue(
            "上一轮摘要",
            {"auto_rounds": 0},
            "discord",
            "chan",
            {"_progress_sent": 99},
        )

    assert len(calls) == 3
    sent_texts = [call.args[2] for call in server.router.send.await_args_list]
    budget_msgs = [text for text in sent_texts if "触达迭代预算上限" in text]
    assert len(budget_msgs) >= 2
    assert any("继续下一轮排查" in text for text in sent_texts)
    assert not any("暂缓" in text or "暂停" in text for text in sent_texts)


@pytest.mark.asyncio
async def test_auto_continue_timeout_is_non_terminal_and_continues(monkeypatch):
    """A timed-out auto round must not break the continuous flywheel chain."""
    server = GatewayServer.__new__(GatewayServer)
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    calls = []

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        calls.append(task)
        if len(calls) == 1:
            raise runner.asyncio.TimeoutError()
        raise KeyboardInterrupt("stop test after proving continuation")

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    with pytest.raises(KeyboardInterrupt):
        await server._auto_continue(
            "上一轮摘要",
            {"auto_rounds": 0},
            "discord",
            "chan",
            {"_progress_sent": 99},
        )

    assert len(calls) == 2
    sent_texts = [call.args[2] for call in server.router.send.await_args_list]
    assert any("超时" in text and "继续下一轮排查" in text for text in sent_texts)
    assert not any("暂停" in text for text in sent_texts)


@pytest.mark.asyncio
async def test_auto_continue_runs_full_20_rounds_despite_per_round_failures(monkeypatch):
    """Continuable per-round failures must not prevent the configured 20-round chain."""
    server = GatewayServer.__new__(GatewayServer)
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    calls = []

    async def fake_run_single_task(task, session, gw_name, channel_id, source_channel, *args, **kwargs):
        calls.append((task, dict(source_channel)))
        n = len(calls)
        if n % 5 == 1:
            raise runner.AgentTaskError("Max iterations (50) reached — budget exhausted")
        if n % 5 == 2:
            raise runner.asyncio.TimeoutError()
        if n % 5 == 3:
            raise RuntimeError("transient-round-crash")
        return f"第 {n} 轮阶段性推进，继续排查。"

    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    result = await server._auto_continue(
        "上一轮摘要",
        {"auto_rounds": 0},
        "discord",
        "chan",
        {"_progress_sent": 99},
    )

    assert len(calls) == runner._AUTO_MAX_ROUNDS
    assert result
    assert all(source["_auto_continue"] is True for _, source in calls)
    assert all(source["_progress_sent"] == 0 for _, source in calls)
    sent_texts = [call.args[2] for call in server.router.send.await_args_list]
    assert sum("飞轮自动继续" in text for text in sent_texts) == runner._AUTO_MAX_ROUNDS
    assert any("触达迭代预算上限" in text for text in sent_texts)
    assert any("超时" in text and "继续下一轮排查" in text for text in sent_texts)
    assert any("运行异常" in text and "继续下一轮排查" in text for text in sent_texts)
    assert any("跑满" in text and "不代表所有问题已完成" in text for text in sent_texts)
    assert not any("暂停" in text for text in sent_texts)


@pytest.mark.asyncio
async def test_running_continue_nudge_clears_session_context():
    """Early-return nudge branch must not leak gateway context into later tasks."""
    server = GatewayServer.__new__(GatewayServer)
    lock = runner.asyncio.Lock()
    await lock.acquire()
    active_key = "agent:main:discord:channel:chan"
    server.sessions = {active_key: {"_task_ctx": SimpleNamespace(shutdown_flag=False)}}
    server.session_locks = {active_key: lock}
    server.router = SimpleNamespace(send=AsyncMock())
    server._dedupe_cache = DedupeCache()

    set_session_context(platform="stale", chat_id="stale", session_key="stale")

    result = await server.handle_task("继续不要停", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg-context-nudge",
    })

    assert result == ""
    assert get_session_env("CAVEMAN_SESSION_PLATFORM") == ""
    assert get_session_env("CAVEMAN_SESSION_CHAT_ID") == ""
    assert get_session_env("CAVEMAN_SESSION_KEY") == ""
    lock.release()


@pytest.mark.asyncio
async def test_successful_task_sets_and_clears_session_key_context(monkeypatch):
    """Live task execution should expose session_key to tools, then clean it up."""
    server = GatewayServer.__new__(GatewayServer)
    server.sessions = {}
    server.session_locks = {}
    server.router = SimpleNamespace(send=AsyncMock())
    server.store = SimpleNamespace()
    server._cached_config = {}
    server._queue_manager = SimpleNamespace(drain=lambda key: [])
    server._dedupe_cache = DedupeCache()
    server._infra = SimpleNamespace(emit_hook=AsyncMock())
    seen = {}

    async def fake_get_or_create_session(key):
        return {"loop": SimpleNamespace(tool_registry=SimpleNamespace(set_context=lambda *a, **k: None)),
                "meta": SimpleNamespace(session_id="s1"),
                "task_count": 0}

    async def fake_run_single_task(*args, **kwargs):
        seen["session_key"] = get_session_env("CAVEMAN_SESSION_KEY")
        seen["chat_id"] = get_session_env("CAVEMAN_SESSION_CHAT_ID")
        return ""

    monkeypatch.setattr(server, "_get_or_create_session", fake_get_or_create_session)
    monkeypatch.setattr(runner, "run_single_task", fake_run_single_task)

    result = await server.handle_task("普通任务", {
        "gateway_name": "discord",
        "channel_id": "chan",
        "user_id": "user",
        "message_id": "msg-context-run",
    })

    assert result == ""
    assert seen["session_key"] == "agent:main:discord:channel:chan"
    assert seen["chat_id"] == "chan"
    assert get_session_env("CAVEMAN_SESSION_KEY") == ""
    assert get_session_env("CAVEMAN_SESSION_CHAT_ID") == ""
