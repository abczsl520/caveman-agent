"""Integration test: Checkpoint → Restore pipeline.

Verifies CheckpointManager save/restore/list/cleanup with real disk I/O.
"""
from __future__ import annotations

import asyncio
import json
import pytest

from caveman.agent.checkpoint import Checkpoint, CheckpointManager


@pytest.fixture
def cp_manager(tmp_path):
    return CheckpointManager(base_dir=tmp_path / "checkpoints")


SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "Write a test"},
    {"role": "assistant", "content": "Sure, here's a test..."},
]


@pytest.mark.asyncio
async def test_save_and_restore(cp_manager):
    """Save a checkpoint and restore it."""
    cp = Checkpoint(
        session_id="sess001",
        messages=SAMPLE_MESSAGES,
        metadata={"task": "integration test"},
    )
    cp_id = await cp_manager.save(cp)
    assert cp_id.startswith("sess001_")

    # Restore latest
    restored = await cp_manager.restore("sess001")
    assert restored is not None
    assert restored.session_id == "sess001"
    assert len(restored.messages) == 4
    assert restored.messages[0]["content"] == "Hello"
    assert restored.metadata["task"] == "integration test"


@pytest.mark.asyncio
async def test_restore_specific_checkpoint(cp_manager):
    """Restore a specific checkpoint by ID."""
    cp1 = Checkpoint("sess002", [{"role": "user", "content": "first"}])
    id1 = await cp_manager.save(cp1)

    await asyncio.sleep(1.1)  # ensure different timestamp

    cp2 = Checkpoint("sess002", [{"role": "user", "content": "second"}])
    id2 = await cp_manager.save(cp2)

    # Restore specific (first)
    restored = await cp_manager.restore("sess002", id1)
    assert restored is not None
    assert restored.messages[0]["content"] == "first"

    # Restore latest (should be second)
    latest = await cp_manager.restore("sess002")
    assert latest is not None
    assert latest.messages[0]["content"] == "second"


@pytest.mark.asyncio
async def test_list_checkpoints(cp_manager):
    """List checkpoints for a session."""
    for i in range(3):
        cp = Checkpoint("sess003", [{"role": "user", "content": f"msg {i}"}])
        await cp_manager.save(cp)
        await asyncio.sleep(1.1)

    listing = await cp_manager.list_checkpoints("sess003")
    assert len(listing) == 3
    # Should be reverse sorted (newest first)
    assert listing[0]["created_at"] >= listing[-1]["created_at"]


@pytest.mark.asyncio
async def test_cleanup_keeps_only_10(cp_manager, monkeypatch):
    """Saving 15 checkpoints should clean up to keep only 10."""
    from datetime import datetime as _dt
    from unittest.mock import patch
    import caveman.agent.checkpoint as cp_mod

    # Monkeypatch datetime.now() to return unique second-precision timestamps
    class FakeDatetime(_dt):
        _counter = [0]

        @classmethod
        def now(cls, tz=None):
            cls._counter[0] += 1
            base = _dt(2026, 1, 1, 0, 0, cls._counter[0])
            return base

    monkeypatch.setattr(cp_mod, "datetime", FakeDatetime)

    for i in range(15):
        cp = Checkpoint("sess004", [{"role": "user", "content": f"msg {i}"}])
        await cp_manager.save(cp)

    listing = await cp_manager.list_checkpoints("sess004")
    assert len(listing) == 10

    # Verify the files on disk
    files = list(cp_manager.base_dir.glob("sess004_*.json"))
    assert len(files) == 10


@pytest.mark.asyncio
async def test_restore_nonexistent_returns_none(cp_manager):
    """Restoring a nonexistent session returns None."""
    result = await cp_manager.restore("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_checkpoint_data_integrity(cp_manager):
    """Verify the JSON on disk matches what we saved."""
    messages = [
        {"role": "user", "content": "complex message with unicode: 你好世界"},
        {"role": "assistant", "content": "response with special chars: <>&\"'"},
    ]
    cp = Checkpoint("sess005", messages, metadata={"round": 94})
    cp_id = await cp_manager.save(cp)

    # Read raw JSON from disk
    path = cp_manager.base_dir / f"{cp_id}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["session_id"] == "sess005"
    assert data["messages"] == messages
    assert data["metadata"]["round"] == 94
