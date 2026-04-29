"""Boundary tests for client/bridge JSON responses."""

import json
import time
from pathlib import Path

import pytest

from caveman.acp.client import ACPClient
from caveman.bridge.hermes_bridge import HermesBridge
from caveman.hub.client import HubClient
from caveman.mcp.oauth import is_token_expired, load_tokens


class FakeResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeSyncClient:
    def __init__(self, response):
        self.response = response

    async def get(self, *args, **kwargs):
        return self.response

    async def post(self, *args, **kwargs):
        return self.response


class FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def post(self, *args, **kwargs):
        return self.responses.pop(0)

    async def get(self, *args, **kwargs):
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_acp_send_task_handles_non_object_create_response():
    client = ACPClient("http://acp.example")
    client._client = FakeAsyncClient([FakeResponse(["not", "object"])])

    result = await client.send_task("hello")

    assert result["error"] == "ACP response must be a JSON object"


@pytest.mark.asyncio
async def test_hermes_bridge_list_skills_rejects_malformed_skill_items():
    bridge = HermesBridge("http://hermes.example")
    bridge._connected = True
    bridge._client = FakeSyncClient(FakeResponse({"skills": [{"name": "ok"}, "bad"]}))

    assert await bridge.list_skills() == []


@pytest.mark.asyncio
async def test_hermes_bridge_delegate_requires_string_result():
    bridge = HermesBridge("http://hermes.example")
    bridge._connected = True
    bridge._client = FakeSyncClient(FakeResponse({"result": {"not": "text"}}))

    assert await bridge.delegate("demo") == ""


def test_hub_client_search_cache_rejects_malformed_skill_items(tmp_path: Path):
    (tmp_path / "skills_cache.json").write_text(
        json.dumps([{"name": "ok"}, "bad"]), encoding="utf-8"
    )

    client = HubClient()
    client._cache_dir = tmp_path

    assert client._search_local_cache("ok") == []


@pytest.mark.asyncio
async def test_hub_client_stats_invalid_json_shape_is_offline(monkeypatch):
    class FakeAsyncContext:
        async def __aenter__(self):
            return FakeSyncClient(FakeResponse(["not", "object"]))

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("caveman.hub.client.httpx.AsyncClient", lambda *a, **k: FakeAsyncContext())
    client = HubClient("http://hub.example")

    stats = await client.hub_stats()

    assert stats["status"] == "offline"
    assert "invalid stats" in stats["note"]


def test_load_tokens_rejects_non_object_json(tmp_path: Path):
    (tmp_path / "demo.json").write_text("[]", encoding="utf-8")

    assert load_tokens("demo", tmp_path) is None


def test_is_token_expired_treats_malformed_times_as_expired():
    assert is_token_expired({"saved_at": "bad", "expires_in": 3600}) is True
    assert is_token_expired({"saved_at": time.time(), "expires_in": "bad"}) is True


@pytest.mark.asyncio
async def test_acp_send_task_async_requires_string_id():
    client = ACPClient("http://acp.example")
    client._client = FakeAsyncClient([FakeResponse({"id": 123})])

    with pytest.raises(ValueError, match="missing string id"):
        await client.send_task_async("hello")
