"""Tests for P1 gateway modules — threading, events, access, model selector."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Threading Tests ──

class TestThreading:
    def test_bind_and_get(self):
        from caveman.gateway.threading import ThreadManager
        mgr = ThreadManager()
        binding = mgr.bind("session1", "thread1", "channel1", starter_text="Hello")
        assert binding.session_key == "session1"
        assert mgr.get_binding("thread1") is binding
        assert mgr.get_thread_for_session("session1") == "thread1"
        assert mgr.is_bound_thread("thread1")

    def test_unbind(self):
        from caveman.gateway.threading import ThreadManager
        mgr = ThreadManager()
        mgr.bind("s1", "t1", "c1")
        mgr.unbind("t1")
        assert mgr.get_binding("t1") is None
        assert mgr.get_thread_for_session("s1") is None

    def test_auto_thread_trigger(self):
        from caveman.gateway.threading import ThreadManager, AutoThreadConfig
        mgr = ThreadManager(AutoThreadConfig(enabled=True, after_messages=3))
        assert not mgr.should_create_thread("ch1")
        assert not mgr.should_create_thread("ch1")
        assert mgr.should_create_thread("ch1")  # 3rd message

    def test_auto_thread_disabled(self):
        from caveman.gateway.threading import ThreadManager, AutoThreadConfig
        mgr = ThreadManager(AutoThreadConfig(enabled=False))
        for _ in range(10):
            assert not mgr.should_create_thread("ch1")

    def test_starter_context(self):
        from caveman.gateway.threading import ThreadManager
        mgr = ThreadManager()
        mgr.cache_starter("t1", "What is AI?", "Alice")
        ctx = mgr.build_starter_context("t1")
        assert "Alice" in ctx
        assert "What is AI?" in ctx

    def test_reap_stale(self):
        from caveman.gateway.threading import ThreadManager
        mgr = ThreadManager()
        b = mgr.bind("s1", "t1", "c1")
        b.last_active = 0  # Force stale
        count = mgr.reap_stale(max_idle=1)
        assert count == 1
        assert mgr.get_binding("t1") is None

    def test_sanitize_thread_name(self):
        from caveman.gateway.threading import sanitize_thread_name
        assert sanitize_thread_name("Hello **world**") == "Hello world"
        assert sanitize_thread_name("<@123456> help me") == "help me"
        assert sanitize_thread_name("") == "Thread"
        long = "x" * 200
        assert len(sanitize_thread_name(long)) <= 100


# ── Event Router Tests ──

class TestEventRouter:
    def test_reaction_command_mapping(self):
        from caveman.gateway.event_router import EventRouter
        router = EventRouter()
        assert router.resolve_reaction_command("🛑") == "/stop"
        assert router.resolve_reaction_command("👍") == "/approve"
        assert router.resolve_reaction_command("🎉") is None

    def test_custom_reaction_command(self):
        from caveman.gateway.event_router import EventRouter
        router = EventRouter()
        router.set_reaction_command("🎯", "/focus")
        assert router.resolve_reaction_command("🎯") == "/focus"
        router.remove_reaction_command("🎯")
        assert router.resolve_reaction_command("🎯") is None

    @pytest.mark.asyncio
    async def test_dispatch_handler(self):
        from caveman.gateway.event_router import EventRouter, EventType, PlatformEvent
        router = EventRouter()
        received = []
        async def handler(event):
            received.append(event)
        router.on(EventType.REACTION_ADD, handler)
        event = PlatformEvent(event_type=EventType.REACTION_ADD, user_id="u1")
        await router.dispatch(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_ignored_user(self):
        from caveman.gateway.event_router import EventRouter, EventType, PlatformEvent
        router = EventRouter()
        received = []
        router.on(EventType.REACTION_ADD, lambda e: received.append(e))
        router.ignore_user("bot1")
        event = PlatformEvent(event_type=EventType.REACTION_ADD, user_id="bot1")
        await router.dispatch(event)
        assert len(received) == 0

    def test_edit_tracking(self):
        from caveman.gateway.event_router import EventRouter
        router = EventRouter()
        router.track_message("m1", "original text")
        assert router.get_original_text("m1") == "original text"
        assert router.was_edited("m1", "new text")
        assert not router.was_edited("m1", "original text")

    @pytest.mark.asyncio
    async def test_handler_priority(self):
        from caveman.gateway.event_router import EventRouter, EventType, PlatformEvent
        router = EventRouter()
        order = []
        router.on(EventType.MESSAGE_EDIT, lambda e: order.append("low"), priority=0)
        router.on(EventType.MESSAGE_EDIT, lambda e: order.append("high"), priority=10)
        await router.dispatch(PlatformEvent(event_type=EventType.MESSAGE_EDIT))
        assert order == ["high", "low"]


# ── Access Control Tests ──

class TestAccessControl:
    def test_owner_access(self):
        from caveman.gateway.access_control import AccessController, AccessLevel
        ac = AccessController(owner_ids={"owner1"})
        assert ac.resolve_access("owner1") == AccessLevel.OWNER
        assert ac.is_owner("owner1")
        assert not ac.is_owner("other")

    def test_admin_access(self):
        from caveman.gateway.access_control import AccessController, AccessLevel
        ac = AccessController(admin_ids={"admin1"})
        assert ac.resolve_access("admin1") == AccessLevel.ADMIN
        assert ac.is_admin("admin1")

    def test_blocked_user(self):
        from caveman.gateway.access_control import AccessController, AccessLevel, AccessRule
        ac = AccessController(rules=[
            AccessRule(pattern="bad_user", level=AccessLevel.BLOCKED),
        ])
        assert ac.resolve_access("bad_user") == AccessLevel.BLOCKED
        assert not ac.is_allowed("bad_user")

    def test_role_based_access(self):
        from caveman.gateway.access_control import AccessController, AccessLevel, AccessRule
        ac = AccessController(rules=[
            AccessRule(pattern="role_admin", level=AccessLevel.ADMIN),
        ])
        assert ac.is_allowed("user1", role_ids={"role_admin"}, min_level=AccessLevel.ADMIN)

    def test_glob_pattern(self):
        from caveman.gateway.access_control import AccessController, AccessLevel, AccessRule
        ac = AccessController(rules=[
            AccessRule(pattern="vip_*", level=AccessLevel.ADMIN),
        ])
        assert ac.resolve_access("vip_alice") == AccessLevel.ADMIN
        assert ac.resolve_access("normal_bob") == AccessLevel.USER

    def test_dm_policy_disabled(self):
        from caveman.gateway.access_control import AccessController
        ac = AccessController(dm_policy="disabled")
        assert not ac.is_dm_allowed("anyone")

    def test_dm_policy_open(self):
        from caveman.gateway.access_control import AccessController
        ac = AccessController(dm_policy="open")
        assert ac.is_dm_allowed("anyone")

    def test_dm_policy_pairing(self):
        from caveman.gateway.access_control import AccessController
        ac = AccessController(dm_policy="pairing")
        assert not ac.is_dm_allowed("user1")
        ac.pair_user("user1")
        assert ac.is_dm_allowed("user1")

    def test_channel_config(self):
        from caveman.gateway.access_control import AccessController, ChannelConfig
        ac = AccessController()
        ac.set_channel_config(ChannelConfig(channel_id="ch1", model="opus", require_mention=True))
        config = ac.get_channel_config("ch1")
        assert config.model == "opus"
        assert config.require_mention

    def test_channel_enabled_default(self):
        from caveman.gateway.access_control import AccessController
        ac = AccessController()
        assert ac.is_channel_enabled("any_channel")


# ── Model Selector Tests ──

class TestModelSelector:
    def test_default_resolution(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector(default_model="claude-sonnet-4-20250514", default_provider="anthropic")
        provider, model = ms.resolve()
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_alias_resolution(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector()
        provider, model = ms.resolve(session_override="opus")
        assert provider == "anthropic"
        assert "opus" in model or "claude" in model

    def test_channel_override(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector(
            default_model="sonnet",
            channel_overrides={"ch1": "opus"},
        )
        provider, model = ms.resolve(channel_id="ch1")
        assert "opus" in model or "claude" in model

    def test_user_preference(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector(default_model="sonnet")
        ms.set_user_preference("user1", "opus")
        assert ms.get_user_preference("user1") == "opus"

    def test_session_override_highest_priority(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector(
            default_model="sonnet",
            channel_overrides={"ch1": "haiku"},
            user_preferences={"u1": "opus"},
        )
        # Session override beats everything
        provider, model = ms.resolve(channel_id="ch1", user_id="u1", session_override="gemini")
        assert provider == "google"

    def test_full_name_resolution(self):
        from caveman.gateway.model_selector import ModelSelector
        ms = ModelSelector()
        provider, model = ms.resolve(session_override="openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_register_model(self):
        from caveman.gateway.model_selector import ModelSelector, ModelEntry
        ms = ModelSelector()
        ms.register_model(ModelEntry(
            provider="custom", model="my-model", alias="mymodel",
            tier="premium", supports_vision=True,
        ))
        models = ms.list_models()
        assert any(m["model"] == "my-model" for m in models)

    def test_fallback_chain(self):
        from caveman.gateway.model_selector import ModelSelector, ModelEntry
        ms = ModelSelector()
        ms.register_model(ModelEntry(provider="a", model="m1", tier="standard"))
        ms.register_model(ModelEntry(provider="b", model="m2", tier="standard"))
        ms.register_model(ModelEntry(provider="c", model="m3", tier="premium"))
        chain = ms.resolve_fallback_chain("a", "m1")
        assert len(chain) >= 2
        assert chain[0] == ("a", "m1")
