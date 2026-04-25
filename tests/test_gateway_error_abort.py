"""Regression tests for gateway handling of agent error events."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from caveman.agent.stream import StreamEvent
from caveman.gateway.task_runner import AgentTaskError, run_single_task


class _ErrorLoop:
    def __init__(self):
        self.budget = SimpleNamespace(reset=lambda: None)
        self.tool_registry = SimpleNamespace(set_context=lambda *a, **k: None)
        self._tool_call_count = 0
        self.provider = SimpleNamespace(usage_stats={})
        self.nudge = None
        self.nudge_task_ref = ""
        self.trajectory_recorder = SimpleNamespace()

    async def run_stream(self, task, attachments=None):
        yield StreamEvent(type="iteration_start", data={"iteration": 0, "max": 5, "remaining": 4})
        yield StreamEvent(type="error", data="LLM 无响应超时 (120s)。任务未完成，请重试或切换模型。")

    def snapshot(self):
        return {}


class _Router:
    def __init__(self):
        self.sent = []

    async def send(self, gw_name, channel_id, message):
        self.sent.append(message)
        return {"message_id": str(len(self.sent))}

    async def edit(self, gw_name, channel_id, message_id, message):
        self.sent.append(message)
        return None


class _Store:
    def __init__(self):
        self.turns = []
        self.saved_meta = []
        self.saved_snapshots = []

    def append_turn(self, session_id, role, content):
        self.turns.append((role, content))

    def save_meta(self, meta):
        self.saved_meta.append(meta)

    def save_snapshot(self, session_id, snap):
        self.saved_snapshots.append((session_id, snap))


@pytest.mark.asyncio
async def test_gateway_error_event_aborts_not_completed_or_persisted():
    meta = SimpleNamespace(session_id="s1", turn_count=0, last_active_at=0, total_tokens=0, total_cost_usd=0)
    session = {"loop": _ErrorLoop(), "meta": meta}
    router = _Router()
    store = _Store()

    with pytest.raises(AgentTaskError):
        await run_single_task(
            "complex task",
            session,
            "discord",
            "chan",
            {"_progress_sent": 0},
            router,
            store,
            config={"gateway": {"timeouts": {"absolute_max": 999, "idle_shutdown": 999, "idle_warning": 999}}},
        )

    assert any("LLM 无响应超时" in msg for msg in router.sent)
    assert store.turns == [("user", "complex task")]
    assert store.saved_meta == []
    assert store.saved_snapshots == []
    assert "_task_ctx" not in session
