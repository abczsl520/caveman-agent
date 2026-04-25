"""Acceptance tests for PRD #22/#24 memory quality contracts."""
from __future__ import annotations

import asyncio
import json
import logging

from caveman.memory.manager import MemoryManager
from caveman.memory.provider import BuiltinMemoryProvider
from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.types import MemoryType, validate_metadata
from caveman.memory.quality_gate import reset_cache, get_stats


def test_validate_metadata_warns_and_drops_bad_known_key(caplog):
    caplog.set_level(logging.WARNING)

    sanitized = validate_metadata(
        {"trust_score": "high", "retrieval_count": -1, "custom_key": {"ok": True}},
        context="unit-test",
    )

    assert sanitized == {"custom_key": {"ok": True}}
    assert "Invalid unit-test metadata[trust_score]" in caplog.text
    assert "Invalid unit-test metadata[retrieval_count]" in caplog.text


def test_sqlite_store_validates_metadata_on_write(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    store = SQLiteMemoryStore(db_path=tmp_path / "mem.db")

    mid = asyncio.run(store.store(
        "The memory metadata registry drops malformed known keys before persistence.",
        MemoryType.SEMANTIC,
        metadata={"trust_score": "bad", "retrieval_count": 2, "source": "nudge"},
    ))

    assert mid
    row = store._get_conn().execute(
        "SELECT metadata_json, trust_score FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    meta = json.loads(row[0])
    assert "trust_score" not in meta
    assert meta["retrieval_count"] == 2
    assert meta["source"] == "nudge"
    assert row[1] == 0.5
    assert "Invalid sqlite store metadata[trust_score]" in caplog.text


def test_memory_manager_with_sqlite_wires_llm_quality_gate(tmp_path):
    reset_cache()
    calls: list[str] = []

    async def judge(prompt: str) -> str:
        calls.append(prompt)
        return '{"accept": false, "reason": "not durable"}'

    mm = MemoryManager.with_sqlite(
        base_dir=tmp_path,
        db_path=tmp_path / "mem.db",
        quality_llm_fn=judge,
        use_llm_quality_gate=True,
    )

    mid = asyncio.run(mm.store(
        "Temporary observation from a one-off scratch run with no reusable project value.",
        MemoryType.SEMANTIC,
    ))

    assert mid == ""
    assert len(calls) == 1
    stats = get_stats().as_dict()
    assert stats["llm_checked"] == 1
    assert stats["llm_rejected"] == 1


def test_builtin_provider_initialize_passes_quality_gate_options(tmp_path):
    reset_cache()

    async def judge(prompt: str) -> str:
        return '{"accept": false, "reason": "reject via provider wiring"}'

    provider = BuiltinMemoryProvider()
    asyncio.run(provider.initialize(
        "session",
        db_path=tmp_path / "provider.db",
        quality_llm_fn=judge,
        use_llm_quality_gate=True,
    ))

    mid = asyncio.run(provider.store(
        "A generic temporary note that should be rejected by the configured provider judge.",
        MemoryType.SEMANTIC,
    ))
    assert mid == ""
    assert get_stats().as_dict()["llm_rejected"] == 1
