"""Regression tests for flywheel-observed prompt and metadata warnings."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from caveman.agent.prompt import build_system_prompt
from caveman.agent.prompt_contract import validate_layer
from caveman.agent.workspace import WorkspaceLoader
from caveman.engines.ripple import RippleEffect, RippleEngine
from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.types import MemoryEntry, MemoryType


def test_workspace_loader_excludes_format_layout_lines_from_workspace_layer(tmp_path, caplog):
    """User workspace can contain format notes, but workspace prompt layer must not own them."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "AGENTS.md").write_text(
        "# AGENTS\n"
        "## Workflow\n"
        "Always trace root cause before fixing.\n"
        "## Group Chats\n"
        "格式: Discord/WhatsApp不用markdown表格; Discord链接用`<>`。\n",
        encoding="utf-8",
    )
    loader = WorkspaceLoader(paths=[ws])

    workspace_content = loader.build_prompt_layers()

    assert "Always trace root cause" in workspace_content
    assert "不用markdown表格" not in workspace_content
    assert not validate_layer("workspace", workspace_content)

    caplog.set_level(logging.WARNING)
    build_system_prompt(workspace_loader=loader, surface="discord")
    assert "Prompt contract: Layer 'workspace' contains format_layout content" not in caplog.text


def test_ripple_conflict_dismissal_does_not_rewrite_invalid_transport_last_accessed(tmp_path, caplog):
    """Full metadata rewrites must not persist transport-only/invalid last_accessed values."""
    store = SQLiteMemoryStore(db_path=tmp_path / "mem.db")
    mid = asyncio.run(store.store(
        "Old memory about gateway port 4201.",
        MemoryType.SEMANTIC,
        metadata={"source": "test"},
    ))
    memory = type("MemoryFacade", (), {"update_metadata": store.update_metadata})()

    async def judge(_prompt: str) -> str:
        return "no"

    engine = RippleEngine(memory, llm_fn=judge)
    existing = MemoryEntry(
        id=mid,
        content="Old memory about gateway port 4201.",
        memory_type=MemoryType.SEMANTIC,
        created_at=datetime.now(),
        metadata={
            "source": "test",
            "superseded_by": "new-id",
            "superseded_at": datetime.now().isoformat(),
            "superseded_content": "new content",
            "last_accessed": 0,
        },
    )
    effect = RippleEffect(new_entry_id="new-id")
    effect.conflicts.append({
        "new": "New memory about gateway port 4201.",
        "existing": existing.content,
        "existing_id": mid,
        "_existing_entry": existing,
    })
    effect.stale_marked.append(mid)
    caplog.set_level(logging.WARNING)

    asyncio.run(engine._verify_conflicts_llm(effect))

    assert "Invalid update_metadata metadata[last_accessed]" not in caplog.text
    refreshed = asyncio.run(store.get_by_id(mid))
    assert refreshed is not None
    assert "last_accessed" not in refreshed.metadata
    assert "superseded_by" not in refreshed.metadata
