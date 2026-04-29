"""Boundary tests for small utility response/metadata helpers."""

import json
from pathlib import Path
from typing import Any

import pytest

from caveman.agent.workspace_memory_sync import WorkspaceMemorySync
from caveman.bridge.acp import ACPClient
from caveman.security.permissions import PermissionLevel, PermissionManager
from caveman.tools.builtin.skills_sync import sync_all_bundled


class _Bridge:
    def __init__(self, result: dict[str, Any]):
        self.result = result

    async def call_tool(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.result


@pytest.mark.asyncio
async def test_permission_callback_only_literal_true_grants() -> None:
    async def returns_truthy_string(action: str, description: str) -> object:
        return "yes"

    manager = PermissionManager({"danger": PermissionLevel.ASK})
    manager.set_approval_callback(returns_truthy_string)  # type: ignore[arg-type]

    assert await manager.request("danger", "delete something") is False


@pytest.mark.asyncio
async def test_permission_callback_literal_true_grants() -> None:
    async def approve(action: str, description: str) -> bool:
        return True

    manager = PermissionManager({"safe": PermissionLevel.ASK})
    manager.set_approval_callback(approve)

    assert await manager.request("safe", "read something") is True


@pytest.mark.asyncio
async def test_acp_send_stringifies_non_string_result_value() -> None:
    client = ACPClient(_Bridge({"result": {"message": "done"}}))

    assert await client.send("session-1", "hello") == "{'message': 'done'}"


@pytest.mark.asyncio
async def test_acp_send_stringifies_full_response_when_result_missing() -> None:
    client = ACPClient(_Bridge({"ok": True}))

    assert await client.send("session-1", "hello") == "{'ok': True}"


def test_workspace_manifest_scalar_falls_back_to_default(tmp_path: Path) -> None:
    sync = WorkspaceMemorySync(tmp_path, memory_manager=object())
    sync._manifest_path.parent.mkdir(parents=True)
    sync._manifest_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    assert sync._load_manifest() == {"version": 1, "files": {}}


def test_workspace_manifest_non_dict_files_is_normalized(tmp_path: Path) -> None:
    sync = WorkspaceMemorySync(tmp_path, memory_manager=object())
    sync._manifest_path.parent.mkdir(parents=True)
    sync._manifest_path.write_text(json.dumps({"version": 1, "files": []}), encoding="utf-8")

    manifest = sync._load_manifest()

    assert manifest["files"] == {}


def test_sync_all_bundled_empty_result_has_typed_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "caveman.tools.builtin.skills_sync._discover_bundled_skills",
        lambda bundled_dir: [],
    )

    result = sync_all_bundled(quiet=True)

    assert result["synced"] == 0
    assert result["unchanged"] == 0
    assert result["conflicts"] == 0
    assert result["errors"] == 0
    assert result["details"] == []
