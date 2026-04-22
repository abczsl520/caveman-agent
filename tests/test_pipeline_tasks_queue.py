"""Tests for message pipeline, task registry, reply queue, routing, and flows."""
import time
import pytest

from caveman.gateway.message_pipeline import (
    MessageContext, MessageAction, normalize_message, dedupe_message,
    route_message, execute_message, prepare_delivery, process_inbound,
    DedupeCache, _sanitize_system_tags,
)
from caveman.gateway.task_registry import (
    TaskRegistry, TaskStatus, TaskRuntime, DeliveryStatus,
)
from caveman.gateway.reply_queue import (
    ReplyQueue, QueueManager, QueuePolicy, QueuedMessage,
)
from caveman.gateway.routing import (
    resolve_route, parse_session_key, resolve_account_id, SessionBinding,
)
from caveman.gateway.flows import (
    create_flow, list_flows, ProviderSetupFlow, FlowStatus,
)


# ── Message Pipeline ──

class TestNormalize:
    def test_basic(self):
        ctx = MessageContext(body="hello\r\nworld")
        ctx = normalize_message(ctx)
        assert ctx.body == "hello\nworld"
        assert ctx.normalized is True

    def test_sanitize_system_tags(self):
        assert _sanitize_system_tags("[system:admin] hello") == "hello"
        assert _sanitize_system_tags("[[reply_to_current]] hi") == "hi"

    def test_sets_defaults(self):
        ctx = MessageContext(body="test", sender_name="Alice")
        ctx = normalize_message(ctx)
        assert ctx.sender_label == "Alice"
        assert ctx.received_at > 0

    def test_group_detection(self):
        ctx = MessageContext(body="test", chat_type="group")
        ctx = normalize_message(ctx)
        assert ctx.is_group is True


class TestDedupe:
    def test_first_message_passes(self):
        cache = DedupeCache()
        ctx = MessageContext(message_id="m1", platform="discord")
        ctx = dedupe_message(ctx, cache)
        assert ctx.action != MessageAction.REJECTED

    def test_duplicate_rejected(self):
        cache = DedupeCache()
        ctx1 = MessageContext(message_id="m1", platform="discord")
        dedupe_message(ctx1, cache)
        ctx2 = MessageContext(message_id="m1", platform="discord")
        ctx2 = dedupe_message(ctx2, cache)
        assert ctx2.action == MessageAction.REJECTED

    def test_different_messages_pass(self):
        cache = DedupeCache()
        ctx1 = MessageContext(message_id="m1", platform="discord")
        dedupe_message(ctx1, cache)
        ctx2 = MessageContext(message_id="m2", platform="discord")
        ctx2 = dedupe_message(ctx2, cache)
        assert ctx2.action != MessageAction.REJECTED

    def test_no_id_passes(self):
        cache = DedupeCache()
        ctx = MessageContext(body="no id")
        ctx = dedupe_message(ctx, cache)
        assert ctx.action != MessageAction.REJECTED


class TestRoute:
    def test_slash_command(self):
        ctx = MessageContext(body="/help")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx)
        assert ctx.action == MessageAction.COMMAND
        assert ctx.command_name == "help"

    def test_command_with_args(self):
        ctx = MessageContext(body="/model opus")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx)
        assert ctx.command_name == "model"
        assert ctx.command_args == "opus"

    def test_normal_message(self):
        ctx = MessageContext(body="hello there")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx)
        assert ctx.action == MessageAction.AGENT_RUN

    def test_group_mention(self):
        ctx = MessageContext(body="hey bot", chat_type="group", is_mention=True)
        ctx = normalize_message(ctx)
        ctx = route_message(ctx, group_activation="mention")
        assert ctx.action == MessageAction.AGENT_RUN

    def test_group_no_mention(self):
        ctx = MessageContext(body="hey everyone", chat_type="group")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx, group_activation="mention")
        assert ctx.action == MessageAction.SILENT

    def test_group_always(self):
        ctx = MessageContext(body="hey", chat_type="group")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx, group_activation="always")
        assert ctx.action == MessageAction.AGENT_RUN

    def test_group_thread(self):
        ctx = MessageContext(body="hey", chat_type="group", thread_id="t1")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx, group_activation="mention")
        assert ctx.action == MessageAction.AGENT_RUN

    def test_directives(self):
        ctx = MessageContext(body="@model:opus explain this")
        ctx = normalize_message(ctx)
        ctx = route_message(ctx)
        assert ctx.directives.get("model") == "opus"
        assert "explain this" in ctx.body


class TestExecute:
    @pytest.mark.asyncio
    async def test_agent_run(self):
        async def agent_fn(ctx):
            return f"Reply to: {ctx.body}"

        ctx = MessageContext(body="hello", action=MessageAction.AGENT_RUN)
        ctx = await execute_message(ctx, agent_fn=agent_fn)
        assert ctx.reply_text == "Reply to: hello"

    @pytest.mark.asyncio
    async def test_command(self):
        async def cmd_fn(name, args, ctx):
            return f"Command: {name} {args}"

        ctx = MessageContext(action=MessageAction.COMMAND, command_name="help", command_args="tools")
        ctx = await execute_message(ctx, command_fn=cmd_fn)
        assert "help tools" in ctx.reply_text

    @pytest.mark.asyncio
    async def test_silent_skipped(self):
        ctx = MessageContext(action=MessageAction.SILENT)
        ctx = await execute_message(ctx)
        assert ctx.reply_text == ""

    @pytest.mark.asyncio
    async def test_error_handling(self):
        async def bad_fn(ctx):
            raise ValueError("boom")

        ctx = MessageContext(body="test", action=MessageAction.AGENT_RUN)
        ctx = await execute_message(ctx, agent_fn=bad_fn)
        assert ctx.action == MessageAction.ERROR
        assert "boom" in ctx.error


class TestDelivery:
    def test_chunking(self):
        ctx = MessageContext(reply_text="a" * 5000)
        chunks = prepare_delivery(ctx, max_length=2000)
        assert len(chunks) >= 3
        assert all(len(c) <= 2000 for c in chunks)

    def test_empty_reply(self):
        ctx = MessageContext(reply_text="")
        assert prepare_delivery(ctx) == []


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_flow(self):
        async def agent_fn(ctx):
            return "Hello!"

        ctx = MessageContext(
            body="hi there", message_id="m1", platform="test",
            sender_name="Alice",
        )
        result = await process_inbound(ctx, agent_fn=agent_fn)
        assert result.reply_text == "Hello!"
        assert result.normalized is True

    @pytest.mark.asyncio
    async def test_command_flow(self):
        async def cmd_fn(name, args, ctx):
            return f"Ran {name}"

        ctx = MessageContext(body="/status", message_id="m2", platform="test")
        result = await process_inbound(ctx, command_fn=cmd_fn)
        assert "Ran status" in result.reply_text


# ── Task Registry ──

class TestTaskRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return TaskRegistry(persist_dir=tmp_path / "tasks")

    def test_create_task(self, registry):
        task = registry.create_task("Test task")
        assert task.task_id
        assert task.status == TaskStatus.CREATED

    def test_lifecycle(self, registry):
        task = registry.create_task("Test")
        assert registry.start_task(task.task_id)
        assert registry.get_task(task.task_id).status == TaskStatus.RUNNING
        assert registry.complete_task(task.task_id, "Done!")
        assert registry.get_task(task.task_id).status == TaskStatus.COMPLETED

    def test_fail_task(self, registry):
        task = registry.create_task("Test")
        registry.start_task(task.task_id)
        registry.fail_task(task.task_id, "oops")
        assert registry.get_task(task.task_id).status == TaskStatus.FAILED

    def test_cancel_task(self, registry):
        task = registry.create_task("Test")
        registry.cancel_task(task.task_id)
        assert registry.get_task(task.task_id).status == TaskStatus.CANCELLED

    def test_progress(self, registry):
        task = registry.create_task("Test")
        registry.start_task(task.task_id)
        registry.update_progress(task.task_id, 0.5, "Halfway")
        assert registry.get_task(task.task_id).progress == 0.5

    def test_list_tasks(self, registry):
        registry.create_task("A", session_id="s1")
        registry.create_task("B", session_id="s2")
        assert len(registry.list_tasks()) == 2
        assert len(registry.list_tasks(session_id="s1")) == 1

    def test_flows(self, registry):
        flow = registry.create_flow("Build feature")
        t1 = registry.create_task("Step 1", flow_id=flow.flow_id)
        t2 = registry.create_task("Step 2", flow_id=flow.flow_id)
        registry.start_task(t1.task_id)
        registry.complete_task(t1.task_id)
        registry.start_task(t2.task_id)
        registry.complete_task(t2.task_id)
        assert registry.get_flow(flow.flow_id).status == "completed"

    def test_delivery(self, registry):
        task = registry.create_task("Test")
        registry.start_task(task.task_id)
        registry.complete_task(task.task_id)
        pending = registry.get_pending_deliveries()
        assert len(pending) == 1
        registry.mark_delivered(task.task_id)
        assert len(registry.get_pending_deliveries()) == 0

    def test_summary(self, registry):
        registry.create_task("A")
        registry.create_task("B")
        s = registry.summary()
        assert s["total"] == 2

    def test_persistence(self, tmp_path):
        r1 = TaskRegistry(persist_dir=tmp_path / "tasks")
        r1.create_task("Persistent task")
        r1.save()
        r2 = TaskRegistry(persist_dir=tmp_path / "tasks")
        r2.load()
        assert len(r2.list_tasks()) == 1

    def test_mark_lost(self, registry):
        task = registry.create_task("Test")
        registry.start_task(task.task_id)
        task.started_at = time.time() - 7200
        count = registry.mark_lost_tasks(timeout=3600)
        assert count == 1
        assert registry.get_task(task.task_id).status == TaskStatus.LOST

    def test_observer(self, registry):
        events = []
        registry.on_event(lambda e, t: events.append(e))
        task = registry.create_task("Test")
        registry.start_task(task.task_id)
        assert "created" in events
        assert "started" in events


# ── Reply Queue ──

class TestReplyQueue:
    def test_fifo(self):
        q = ReplyQueue("s1", policy=QueuePolicy.FIFO)
        q.enqueue(QueuedMessage(body="first"))
        q.enqueue(QueuedMessage(body="second"))
        msgs = q.drain()
        assert len(msgs) == 2
        assert msgs[0].body == "first"

    def test_latest(self):
        q = ReplyQueue("s1", policy=QueuePolicy.LATEST)
        q.enqueue(QueuedMessage(body="first"))
        q.enqueue(QueuedMessage(body="second"))
        msgs = q.drain()
        assert len(msgs) == 1
        assert msgs[0].body == "second"

    def test_merge(self):
        q = ReplyQueue("s1", policy=QueuePolicy.MERGE)
        q.enqueue(QueuedMessage(body="hello"))
        q.enqueue(QueuedMessage(body="world"))
        msgs = q.drain()
        assert len(msgs) == 1
        assert "hello" in msgs[0].body
        assert "world" in msgs[0].body

    def test_empty_drain(self):
        q = ReplyQueue("s1")
        assert q.drain() == []

    def test_depth(self):
        q = ReplyQueue("s1")
        q.enqueue(QueuedMessage(body="a"))
        q.enqueue(QueuedMessage(body="b"))
        assert q.depth == 2

    def test_clear(self):
        q = ReplyQueue("s1")
        q.enqueue(QueuedMessage(body="a"))
        assert q.clear() == 1
        assert q.is_empty


class TestQueueManager:
    def test_per_session(self):
        mgr = QueueManager()
        mgr.enqueue("s1", QueuedMessage(body="a"))
        mgr.enqueue("s2", QueuedMessage(body="b"))
        assert len(mgr.drain("s1")) == 1
        assert len(mgr.drain("s2")) == 1

    def test_stats(self):
        mgr = QueueManager()
        mgr.enqueue("s1", QueuedMessage(body="a"))
        s = mgr.stats()
        assert s["total_queued"] == 1


# ── Routing ──

class TestRouting:
    def test_new_session(self):
        route = resolve_route("discord", "123")
        assert route.is_new_session
        assert "discord" in route.session_key

    def test_binding_match(self):
        bindings = {
            "discord:123": SessionBinding(platform="discord", chat_id="123", agent_id="custom")
        }
        route = resolve_route("discord", "123", bindings=bindings)
        assert route.agent_id == "custom"
        assert not route.is_new_session

    def test_thread_binding(self):
        bindings = {
            "discord:123:t1": SessionBinding(platform="discord", chat_id="123", thread_id="t1")
        }
        route = resolve_route("discord", "123", thread_id="t1", bindings=bindings)
        assert route.binding == "thread"

    def test_parse_session_key(self):
        result = parse_session_key("agent:main:discord:channel:123")
        assert result["agent_id"] == "main"
        assert result["chat_id"] == "123"

    def test_account_resolution(self):
        account_map = {"discord:user1": "premium"}
        assert resolve_account_id("user1", "discord", account_map) == "premium"
        assert resolve_account_id("user2", "discord", account_map) == "default"


# ── Flows ──

class TestFlows:
    def test_list_flows(self):
        flows = list_flows()
        assert "provider" in flows
        assert "channel" in flows

    def test_create_flow(self):
        flow = create_flow("provider")
        assert flow is not None

    def test_provider_flow(self):
        flow = ProviderSetupFlow()
        prompt = flow.start()
        assert "provider" in prompt.lower()

        result, done = flow.submit("anthropic")
        assert not done
        assert "key" in result.lower()

        result, done = flow.submit("sk-test-key")
        assert not done

        result, done = flow.submit("")  # default model
        assert not done

        result, done = flow.submit("")  # no proxy
        assert done
        assert "configured" in result

    def test_flow_validation(self):
        flow = ProviderSetupFlow()
        flow.start()
        flow.submit("anthropic")
        result, done = flow.submit("")  # empty API key
        assert not done
        assert "cannot be empty" in result.lower()

    def test_flow_cancel(self):
        flow = ProviderSetupFlow()
        flow.start()
        result = flow.cancel()
        assert "cancelled" in result.lower()
        assert flow.state.status == FlowStatus.CANCELLED

    def test_unknown_flow(self):
        assert create_flow("nonexistent") is None
