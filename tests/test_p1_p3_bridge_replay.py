"""Tests for P1 Bridge + P3 EventBus Replay."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Bridge Tests ──


class TestOpenClawBridge:
    """Test OpenClawBridge transport selection and API."""

    def test_init_defaults(self):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        bridge = OpenClawBridge()
        assert bridge.transport_name == "none"
        assert not bridge.is_connected

    def test_init_with_params(self):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        bridge = OpenClawBridge(
            transport="cli",
            session_key="test-session",
            gateway_url="ws://localhost:9999",
            token="test-token",
            password="test-pass",
        )
        assert bridge.session_key == "test-session"
        assert bridge.gateway_url == "ws://localhost:9999"
        assert bridge.token == "test-token"

    def test_check_connected_raises(self):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        bridge = OpenClawBridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge._check_connected()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        bridge = OpenClawBridge()
        await bridge.disconnect()  # Should not raise
        assert not bridge.is_connected

    def test_get_tool_schemas_empty(self):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        bridge = OpenClawBridge()
        assert bridge.get_tool_schemas() == []


class TestWSTransport:
    """Test WebSocket transport."""

    def test_import(self):
        from caveman.bridge.ws_transport import WSTransport, WSTransportError
        assert WSTransport is not None
        assert WSTransportError is not None

    def test_init_defaults(self):
        from caveman.bridge.ws_transport import WSTransport
        ws = WSTransport()
        assert ws.url == "ws://127.0.0.1:18789"
        assert not ws.connected

    def test_init_custom(self):
        from caveman.bridge.ws_transport import WSTransport
        ws = WSTransport(
            url="ws://custom:9999",
            token="tok",
            password="pass",
            session_key="my-session",
        )
        assert ws.url == "ws://custom:9999"
        assert ws.token == "tok"
        assert ws.session_key == "my-session"

    @pytest.mark.asyncio
    async def test_request_not_connected(self):
        from caveman.bridge.ws_transport import WSTransport, WSTransportError
        ws = WSTransport()
        with pytest.raises(WSTransportError, match="not connected"):
            await ws.request("test.method")

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        from caveman.bridge.ws_transport import WSTransport
        ws = WSTransport()
        await ws.disconnect()  # Should not raise


class TestWSTransportAdapter:
    """Test the WS adapter in OpenClawBridge."""

    def test_adapter_name(self):
        from caveman.bridge.openclaw_bridge import _WSAdapter
        adapter = _WSAdapter(
            url="ws://test:1234", token="", password="",
            session_key="test", agent_id="caveman",
        )
        assert adapter.name == "websocket"


# ── ACP Client Tests ──


class TestACPClient:
    """Test ACP client fixes (keying, session extraction)."""

    def test_extract_session_key_from_result(self):
        from caveman.bridge.acp import ACPClient
        # Direct key
        assert ACPClient._extract_session_key({"sessionKey": "abc"}, "test") == "abc"
        assert ACPClient._extract_session_key({"session_key": "def"}, "test") == "def"
        assert ACPClient._extract_session_key({"key": "ghi"}, "test") == "ghi"

    def test_extract_session_key_nested(self):
        from caveman.bridge.acp import ACPClient
        result = {"result": {"sessionKey": "nested-key"}}
        assert ACPClient._extract_session_key(result, "test") == "nested-key"

    def test_extract_session_key_fallback(self):
        from caveman.bridge.acp import ACPClient
        key = ACPClient._extract_session_key({"random": "data"}, "claude-code")
        assert key.startswith("claude-code:")

    @pytest.mark.asyncio
    async def test_spawn_keys_by_session(self):
        from caveman.bridge.acp import ACPClient
        mock_bridge = AsyncMock()
        mock_bridge.call_tool.return_value = {"sessionKey": "session-1"}
        client = ACPClient(openclaw_bridge=mock_bridge)

        result = await client.spawn("task 1", agent_id="claude-code")
        assert result["session_key"] == "session-1"
        assert "session-1" in client._sessions

        # Spawn same agent type again — should NOT overwrite
        mock_bridge.call_tool.return_value = {"sessionKey": "session-2"}
        result2 = await client.spawn("task 2", agent_id="claude-code")
        assert result2["session_key"] == "session-2"
        assert len(client._sessions) == 2
        assert "session-1" in client._sessions
        assert "session-2" in client._sessions

    def test_list_spawned(self):
        from caveman.bridge.acp import ACPClient
        client = ACPClient()
        assert client.list_spawned() == []

    @pytest.mark.asyncio
    async def test_spawn_no_bridge_raises(self):
        from caveman.bridge.acp import ACPClient
        client = ACPClient()
        with pytest.raises(RuntimeError, match="bridge not configured"):
            await client.spawn("test")


# ── ACP Session Tests ──


class TestACPSession:
    """Test ACP session basics."""

    def test_session_creation(self):
        from caveman.bridge.acp import ACPSession
        s = ACPSession(agent_id="test")
        assert s.agent_id == "test"
        assert s.status == "active"
        assert len(s.session_id) > 0

    def test_add_message(self):
        from caveman.bridge.acp import ACPSession
        s = ACPSession()
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"

    def test_to_dict(self):
        from caveman.bridge.acp import ACPSession
        s = ACPSession(session_id="test-123", agent_id="caveman")
        d = s.to_dict()
        assert d["session_id"] == "test-123"
        assert d["agent_id"] == "caveman"
        assert d["message_count"] == 0


# ── CLI Agents PTY Fix Tests ──


class TestCLIAgentRunner:
    """Test CLI agent runner."""

    def test_available_agents(self):
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner()
        # Just verify it doesn't crash
        available = runner.available()
        assert isinstance(available, list)

    @pytest.mark.asyncio
    async def test_unknown_agent_raises(self):
        from caveman.bridge.cli_agents import CLIAgentRunner, CLIAgentError
        runner = CLIAgentRunner()
        with pytest.raises(CLIAgentError, match="Unknown agent"):
            await runner.run("nonexistent_agent", "test task")


# ── Hermes Bridge Tests ──


class TestHermesBridge:
    """Test Hermes bridge error handling fixes."""

    def test_init(self):
        from caveman.bridge.hermes_bridge import HermesBridge
        bridge = HermesBridge(base_url="http://test:8000")
        assert bridge.base_url == "http://test:8000"
        assert not bridge._connected

    @pytest.mark.asyncio
    async def test_connect_timeout_handled(self):
        from caveman.bridge.hermes_bridge import HermesBridge
        bridge = HermesBridge(base_url="http://192.0.2.1:9999")  # Non-routable
        # Should return False, not raise
        result = await bridge.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        from caveman.bridge.hermes_bridge import HermesBridge
        bridge = HermesBridge()
        await bridge.disconnect()
        await bridge.disconnect()  # Should not raise

    def test_ensure_connected_raises(self):
        from caveman.bridge.hermes_bridge import HermesBridge
        bridge = HermesBridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge._ensure_connected()


# ── EventStore Replay Tests ──


class TestEventStoreReplay:
    """Test EventStore replay functionality."""

    @pytest.fixture
    def store(self, tmp_path):
        from caveman.events_store import EventStore
        return EventStore(db_path=tmp_path / "test_events.db")

    @pytest.fixture
    def sample_events(self, store):
        """Insert sample events."""
        from caveman.events import Event
        store.set_session("test-session")
        base_ts = time.time()
        for i in range(5):
            event = Event(
                type=f"test.event.{i % 2}",
                data={"index": i, "value": f"data-{i}"},
                timestamp=base_ts + i,
                source="test",
            )
            store.handle(event)
        return base_ts

    def test_set_session(self, store):
        store.set_session("my-session")
        assert store._session_id == "my-session"

    def test_handle_with_session(self, store):
        from caveman.events import Event
        store.set_session("sess-1")
        store.handle(Event(type="test", data={}, timestamp=time.time(), source="t"))
        rows = store.query(session_id="sess-1")
        assert len(rows) == 1

    def test_query_ascending(self, store, sample_events):
        rows = store.query(ascending=True, limit=5)
        assert len(rows) == 5
        # Ascending: first event has lowest timestamp
        assert rows[0]["timestamp"] <= rows[-1]["timestamp"]

    def test_query_by_session(self, store, sample_events):
        rows = store.query(session_id="test-session")
        assert len(rows) == 5

    def test_query_by_type(self, store, sample_events):
        rows = store.query(event_type="test.event.0")
        assert len(rows) == 3  # indices 0, 2, 4

    def test_query_until(self, store, sample_events):
        rows = store.query(until=sample_events + 2.5)
        assert len(rows) == 3  # timestamps 0, 1, 2

    def test_distinct_types(self, store, sample_events):
        types = store.distinct_types()
        assert "test.event.0" in types
        assert "test.event.1" in types

    def test_sessions_list(self, store, sample_events):
        sessions = store.sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "test-session"
        assert sessions[0]["count"] == 5

    def test_summary(self, store, sample_events):
        s = store.summary()
        assert s["total"] == 5
        assert "test.event.0" in s["by_type"]

    def test_summary_by_session(self, store, sample_events):
        s = store.summary(session_id="test-session")
        assert s["total"] == 5
        assert s["session_id"] == "test-session"

    def test_purge(self, store, sample_events):
        deleted = store.purge(before=sample_events + 2.5)
        assert deleted == 3
        assert store.count() == 2

    def test_purge_by_type(self, store, sample_events):
        deleted = store.purge(event_type="test.event.1", before=time.time() + 100)
        assert deleted == 2  # indices 1, 3
        assert store.count() == 3

    def test_purge_no_criteria_safe(self, store, sample_events):
        deleted = store.purge()
        assert deleted == 0  # Safety: refuses to purge all
        assert store.count() == 5

    @pytest.mark.asyncio
    async def test_replay_instant(self, store, sample_events):
        from caveman.events import EventBus
        bus = EventBus()
        received = []
        bus.on("test.event.0", lambda e: received.append(e))
        bus.on("test.event.1", lambda e: received.append(e))

        count = await store.replay(bus, speed=0, limit=100)
        assert count == 5
        assert len(received) == 5

    @pytest.mark.asyncio
    async def test_replay_filtered(self, store, sample_events):
        from caveman.events import EventBus
        bus = EventBus()
        received = []
        bus.on("test.event.0", lambda e: received.append(e))

        count = await store.replay(bus, event_type="test.event.0", speed=0)
        assert count == 3

    @pytest.mark.asyncio
    async def test_replay_by_session(self, store, sample_events):
        from caveman.events import EventBus
        bus = EventBus()
        count = await store.replay(bus, session_id="test-session", speed=0)
        assert count == 5

    @pytest.mark.asyncio
    async def test_replay_empty(self, store):
        from caveman.events import EventBus
        bus = EventBus()
        count = await store.replay(bus, session_id="nonexistent", speed=0)
        assert count == 0

    @pytest.mark.asyncio
    async def test_replay_iter(self, store, sample_events):
        events = []
        async for e in store.replay_iter():
            events.append(e)
        assert len(events) == 5
        assert "data" in events[0]

    def test_close(self, store):
        store.close()  # Should not raise


# ── Bridge __init__ Tests ──


class TestBridgeInit:
    """Test bridge module exports."""

    def test_imports(self):
        from caveman.bridge import acp, cli_agents, hermes_bridge, openclaw_bridge, uds_transport
        assert acp is not None
        assert cli_agents is not None

    def test_ws_transport_import(self):
        from caveman.bridge import ws_transport
        assert ws_transport is not None
