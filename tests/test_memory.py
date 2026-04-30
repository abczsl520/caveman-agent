"""Tests for memory system."""
import pytest
import tempfile

from caveman.memory.flywheel_metrics import FlywheelHealth
from caveman.memory.types import MemoryType
from caveman.memory.manager import MemoryManager
from caveman.memory.quarantine import (
    list_quarantined,
    preview_restore_quarantined,
    restore_quarantined,
)


def _close_manager(mgr: MemoryManager) -> None:
    if mgr.backend:
        mgr.backend.close()


@pytest.mark.asyncio
async def test_quarantine_lifecycle_can_list_and_restore_entries_with_audit_reason():
    """Operators need a reversible path from dashboard quarantine evidence to active recall."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            quarantined_id = await store.store(
                "reversible openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                    "quarantined_at": "2026-04-30T00:00:00+00:00",
                },
                trusted=True,
            )
            active_id = await store.store(
                "active openclaw memory",
                MemoryType.SEMANTIC,
                metadata={"source": "import:openclaw"},
                trusted=True,
            )

            listed = list_quarantined(store, source="import:openclaw")

            assert [entry.id for entry in listed] == [quarantined_id]
            assert active_id not in {entry.id for entry in listed}

            restored = await restore_quarantined(
                store,
                quarantined_id,
                restored_by="operator",
                restore_reason="manual false positive",
            )
            assert restored is True
            restored_entry = await store.get_by_id(quarantined_id)
            assert restored_entry is not None
            assert restored_entry.metadata["governance_state"] == "active"
            assert restored_entry.metadata["previous_governance_state"] == "quarantined"
            assert restored_entry.metadata["restored_by"] == "operator"
            assert restored_entry.metadata["restore_reason"] == "manual false positive"
            assert "restored_at" in restored_entry.metadata
            assert list_quarantined(store, source="import:openclaw") == []
        finally:
            _close_manager(mgr)


def test_quarantine_cli_lists_and_restores_entries_with_audit_metadata(monkeypatch):
    """The operator-facing CLI should expose list/restore without deleting evidence."""
    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            import asyncio
            quarantined_id = asyncio.run(store.store(
                "cli reversible openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                    "quarantined_at": "2026-04-30T00:00:00+00:00",
                },
                trusted=True,
            ))
        finally:
            _close_manager(mgr)

        runner = CliRunner()
        listed = runner.invoke(app, ["memory-quarantine", "list", "--db", f"{td}/caveman.db", "--source", "import:openclaw"])
        assert listed.exit_code == 0, listed.output
        assert quarantined_id in listed.output
        assert "source_policy_low_signal_import" in listed.output

        restored = runner.invoke(app, [
            "memory-quarantine", "restore", quarantined_id,
            "--db", f"{td}/caveman.db",
            "--by", "operator",
            "--reason", "manual false positive",
        ])
        assert restored.exit_code == 0, restored.output
        assert "restored" in restored.output.lower()

        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            entry = asyncio.run(mgr.backend.get_by_id(quarantined_id))
            assert entry.metadata["governance_state"] == "active"
            assert entry.metadata["restored_by"] == "operator"
            assert entry.metadata["restore_reason"] == "manual false positive"
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantine_restore_preview_filters_without_mutating_rows():
    """Bulk restore needs a dry-run preview before any quarantined row becomes active."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            match_id = await store.store(
                "preview matching openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                },
                trusted=True,
            )
            other_source_id = await store.store(
                "preview hermes quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:hermes",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                },
                trusted=True,
            )
            other_reason_id = await store.store(
                "preview stale openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "stale_low_signal_import",
                },
                trusted=True,
            )
            active_id = await store.store(
                "preview active openclaw memory",
                MemoryType.SEMANTIC,
                metadata={"source": "import:openclaw"},
                trusted=True,
            )

            preview = preview_restore_quarantined(
                store,
                source="import:openclaw",
                reason="source_policy_low_signal_import",
                limit=10,
            )

            assert preview.total_matches == 1
            assert preview.by_source == {"import:openclaw": 1}
            assert preview.by_reason == {"source_policy_low_signal_import": 1}
            assert [entry.id for entry in preview.entries] == [match_id]
            for memory_id in [match_id, other_source_id, other_reason_id]:
                entry = await store.get_by_id(memory_id)
                assert entry is not None
                assert entry.metadata["governance_state"] == "quarantined"
            active = await store.get_by_id(active_id)
            assert active is not None
            assert active.metadata.get("governance_state") != "quarantined"
        finally:
            _close_manager(mgr)


def test_quarantine_cli_preview_is_dry_run_and_reports_impact(monkeypatch):
    """Operators should see restore impact by source/reason before running mutation."""
    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            import asyncio
            first_id = asyncio.run(store.store(
                "cli preview first openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                },
                trusted=True,
            ))
            second_id = asyncio.run(store.store(
                "cli preview second openclaw quarantine memory",
                MemoryType.SEMANTIC,
                metadata={
                    "source": "import:openclaw",
                    "governance_state": "quarantined",
                    "quarantine_reason": "source_policy_low_signal_import",
                },
                trusted=True,
            ))
        finally:
            _close_manager(mgr)

        runner = CliRunner()
        preview = runner.invoke(app, [
            "memory-quarantine", "preview-restore",
            "--db", f"{td}/caveman.db",
            "--source", "import:openclaw",
            "--reason", "source_policy_low_signal_import",
        ])

        assert preview.exit_code == 0, preview.output
        assert "would_restore=2" in preview.output
        assert "import:openclaw=2" in preview.output
        assert "source_policy_low_signal_import=2" in preview.output
        assert first_id in preview.output
        assert second_id in preview.output

        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            for memory_id in [first_id, second_id]:
                entry = asyncio.run(mgr.backend.get_by_id(memory_id))
                assert entry is not None
                assert entry.metadata["governance_state"] == "quarantined"
        finally:
            _close_manager(mgr)


def test_source_governance_cli_previews_policy_drift_without_mutating_rows(monkeypatch):
    """Operators need a copyable allowlist candidate preview before changing source policy."""
    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            import asyncio
            for idx in range(3):
                asyncio.run(store.store(
                    f"cli unmanaged low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:new-bulk-feed", "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        runner = CliRunner()
        preview = runner.invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

        assert preview.exit_code == 0, preview.output
        assert "candidate_count=1" in preview.output
        assert "candidate_policy_entry" in preview.output
        assert "import:new-bulk-feed" in preview.output
        assert "review_for_low_signal_allowlist" in preview.output

        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            entries = mgr.backend.all_entries()
            assert len(entries) == 3
            assert {entry.metadata.get("source") for entry in entries} == {"import:new-bulk-feed"}
            assert {entry.metadata.get("governance_state") for entry in entries} == {None}
        finally:
            _close_manager(mgr)


def test_source_governance_cli_reports_total_candidates_separately_from_limit(monkeypatch):
    """Limit must not make operators think hidden policy drift candidates do not exist."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for source, rows in [("import:largest", 5), ("import:middle", 4), ("import:smallest", 3)]:
                for idx in range(rows):
                    asyncio.run(store.store(
                        f"{source} low-signal source {idx}",
                        MemoryType.SEMANTIC,
                        metadata={"source": source, "trust_score": 0.05},
                        trusted=True,
                    ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
            "--limit", "1",
        ])

        assert preview.exit_code == 0, preview.output
        assert "candidate_count=3" in preview.output
        assert "showing_count=1" in preview.output
        assert "1. source='import:largest' total=5" in preview.output
        assert "import:middle" not in preview.output
        assert "import:smallest" not in preview.output


def test_source_governance_cli_prints_copy_paste_policy_workflow(monkeypatch):
    """Preview output should give operators an exact no-mutation allowlist patch workflow."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"copyable low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:operator-feed", "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

        assert preview.exit_code == 0, preview.output
        assert "Policy workflow (copy/paste):" in preview.output
        assert "1. Review source quality outside the CLI; this command is read-only." in preview.output
        assert "2. If approved, add to caveman.memory.sources.SOURCE_POLICY_LOW_SIGNAL_IMPORTS:" in preview.output
        assert "   'import:operator-feed'," in preview.output
        assert "3. Re-run: caveman source-governance preview-drift --db" in preview.output
        assert "--min-rows 3 --limit 8" in preview.output
        assert "auto_mutation=disabled" in preview.output


def test_source_governance_cli_escapes_copy_paste_policy_entries(monkeypatch):
    """Database-derived source labels must be emitted as safe Python literals."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    unsafe_source = 'import:operator"feed\\new'
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"unsafe copyable low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": unsafe_source, "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

        assert preview.exit_code == 0, preview.output
        assert f"   {unsafe_source!r}," in preview.output
        assert f'   "{unsafe_source}",' not in preview.output


def test_source_governance_cli_rerun_command_shell_quotes_custom_db_path(monkeypatch):
    """Operators should be able to copy the review command without losing the audited DB scope."""
    import asyncio
    import shlex

    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        data_dir = f"{td}/copy'quoted"
        mgr = MemoryManager.with_sqlite(base_dir=data_dir)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"custom-db low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:custom-db-feed", "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        db_path = f"{data_dir}/caveman.db"
        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", db_path,
            "--min-rows", "3",
            "--limit", "1",
        ])

        assert preview.exit_code == 0, preview.output
        assert f"3. Re-run: caveman source-governance preview-drift --db {shlex.quote(db_path)} --min-rows 3 --limit 1" in preview.output


def test_source_governance_cli_prints_review_checklist_for_each_candidate(monkeypatch):
    """Operators need a per-candidate checklist so preview output can be reviewed without losing items."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for source, rows in [("import:largest", 5), ("import:middle", 4)]:
                for idx in range(rows):
                    asyncio.run(store.store(
                        f"{source} checklist low-signal source {idx}",
                        MemoryType.SEMANTIC,
                        metadata={"source": source, "trust_score": 0.05},
                        trusted=True,
                    ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
            "--limit", "2",
        ])

        assert preview.exit_code == 0, preview.output
        assert "Review checklist:" in preview.output
        assert "   [ ] 'import:largest' — reason='unmanaged_low_signal_import' total=5" in preview.output
        assert "   [ ] 'import:middle' — reason='unmanaged_low_signal_import' total=4" in preview.output
        assert preview.output.index("Review checklist:") < preview.output.index("auto_mutation=disabled")


def test_source_governance_cli_checklist_escapes_control_characters(monkeypatch):
    """Checklist source labels must not let data-controlled control chars spoof output rows."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    unsafe_source = "import:operator\n\x1b[31mspoof"
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"unsafe checklist low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": unsafe_source, "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

        assert preview.exit_code == 0, preview.output
        checklist = preview.output.split("Review checklist:", 1)[1]
        assert "operator\n" not in checklist
        assert "\x1b[31m" not in checklist
        assert "'import:operator\\n\\x1b[31mspoof'" in checklist
def test_source_governance_cli_preview_rows_escape_control_characters(monkeypatch):
    """All data-derived preview rows should be safe against terminal/output spoofing."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app

    unsafe_source = "import:operator\n\x1b[31mspoof"
    unsafe_reason = "unmanaged\n\x1b[32mreason"
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"unsafe preview low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": unsafe_source, "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        from caveman.cli import source_governance

        original_collect = source_governance._collect_memory_source_policy_drift

        def collect_with_unsafe_reason(*args, **kwargs):
            rows = original_collect(*args, **kwargs)
            for row in rows:
                row["reason"] = unsafe_reason
            return rows

        monkeypatch.setattr(source_governance, "_collect_memory_source_policy_drift", collect_with_unsafe_reason)
        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

        assert preview.exit_code == 0, preview.output
        for line in preview.output.splitlines():
            assert "\x1b[" not in line
            assert line.strip() != "spoof"
            assert line.strip() != "reason"
        assert "source='import:operator\\n\\x1b[31mspoof'" in preview.output
        assert "reason='unmanaged\\n\\x1b[32mreason'" in preview.output


def test_source_governance_cli_uses_shared_literal_formatter(monkeypatch):
    """Operator-facing literals should share one formatter instead of scattering inline reprs."""
    import asyncio

    from typer.testing import CliRunner

    from caveman.cli.main import app
    from caveman.cli import source_governance

    seen = []

    def tracking_literal(value):
        seen.append(value)
        return f"SAFE<{str(value).replace(chr(10), chr(92) + 'n')}>"

    monkeypatch.setattr(source_governance, "_operator_literal", tracking_literal)
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("CAVEMAN_HOME", td)
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for idx in range(3):
                asyncio.run(store.store(
                    f"shared literal low-signal source {idx}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:shared-literal", "trust_score": 0.05},
                    trusted=True,
                ))
        finally:
            _close_manager(mgr)

        preview = CliRunner().invoke(app, [
            "source-governance", "preview-drift",
            "--db", f"{td}/caveman.db",
            "--min-rows", "3",
        ])

    assert preview.exit_code == 0, preview.output
    assert "source=SAFE<import:shared-literal>" in preview.output
    assert "reason=SAFE<unmanaged_low_signal_import>" in preview.output
    assert "   SAFE<import:shared-literal>," in preview.output
    assert "[ ] SAFE<import:shared-literal> — reason=SAFE<unmanaged_low_signal_import> total=3" in preview.output
    assert seen.count("import:shared-literal") >= 4
    assert "unmanaged_low_signal_import" in seen


def test_memory_types():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.WORKING.value == "working"


@pytest.mark.asyncio
async def test_memory_store_recall():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            mid = await mgr.store("test content about python", MemoryType.EPISODIC)
            assert mid
            results = await mgr.recall("python")
            assert len(results) >= 1
            assert "python" in results[0].content
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_memory_nudge():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            await mgr.nudge()  # should not raise
            assert True  # Nudge completed without error
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_flywheel_health_uses_real_feedback_and_recall_counters():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            stale = await mgr.store("stale memory about docker compose", MemoryType.SEMANTIC)
            helpful = await mgr.store("helpful python deployment memory", MemoryType.SEMANTIC)

            results = await mgr.recall("python deployment")
            assert {r.id for r in results} >= {helpful}
            await mgr.backend.mark_helpful(helpful, helpful=True)

            health = await FlywheelHealth.diagnose(mgr.backend)

            assert health.total_memories == 2
            assert health.memories_never_recalled == 1
            assert health.recalled_memories == 1
            assert health.recall_rate == 0.5
            assert health.memories_with_feedback == 1
            assert health.feedback_rate == 0.5
            assert health.top_recalled[0]["id"] == helpful
            assert stale not in {item["id"] for item in health.top_recalled}
            assert "recall rate=50%" in health.summary()
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_import_memories_are_excluded_from_recall_candidates():
    """Reversible quarantine must remove noisy imported memories from active recall."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            quarantined = await mgr.store(
                "docker compose restart troubleshooting",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )
            active = await mgr.store(
                "docker compose restart troubleshooting verified in current project",
                MemoryType.PROCEDURAL,
                metadata={"source": "nudge"},
                trusted=True,
            )

            results = await mgr.recall("docker compose restart troubleshooting", top_k=5)

            assert active in {entry.id for entry in results}
            assert quarantined not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_recall_fallback():
    """If all lexical matches are quarantined, high-trust fallback must not re-add them."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            quarantined = await mgr.store(
                "only quarantined docker compose restart memory",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )

            results = await mgr.recall("docker compose restart", top_k=5)

            assert quarantined not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_sync_search():
    """search_sync is also an active recall path and must honor quarantine."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            quarantined_id = await store.store(
                "sync docker compose restart memory",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )

            results = store.search_sync("sync docker compose restart", limit=5)

            assert quarantined_id not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_recent_or_all_entries():
    """List-style active memory APIs must not expose quarantined imports."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            quarantined_id = await store.store(
                "recent quarantined import memory",
                MemoryType.SEMANTIC,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )
            active_id = await store.store(
                "recent active nudge memory",
                MemoryType.SEMANTIC,
                metadata={"source": "nudge"},
                trusted=True,
            )

            recent_ids = {entry.id for entry in store.recent(limit=10)}
            all_ids = {entry.id for entry in store.all_entries()}

            assert active_id in recent_ids
            assert active_id in all_ids
            assert quarantined_id not in recent_ids
            assert quarantined_id not in all_ids
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantine_sql_filter_prevents_limited_fts_page_from_hiding_active_match():
    """SQL-side filtering must avoid quarantined first-page rows crowding out active matches."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for i in range(8):
                await store.store(
                    f"needle dominant quarantined import memory {i}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                    trusted=True,
                )
            active_id = await store.store(
                "needle active memory should survive quarantine crowding",
                MemoryType.SEMANTIC,
                metadata={"source": "nudge"},
                trusted=True,
            )

            results = await store.recall("needle", top_k=1)

            assert [entry.id for entry in results] == [active_id]
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_malformed_metadata_does_not_break_active_memory_queries():
    """Legacy/corrupt metadata rows must not crash SQL-side quarantine filtering."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            memory_id = await store.store(
                "legacy malformed metadata needle",
                MemoryType.SEMANTIC,
                metadata={"source": "legacy"},
                trusted=True,
            )
            store._get_conn().execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                ("not json", memory_id),
            )
            store._get_conn().commit()

            recall_ids = {entry.id for entry in await store.recall("needle", top_k=5)}
            recent_ids = {entry.id for entry in store.recent(limit=5)}
            all_ids = {entry.id for entry in store.all_entries()}

            assert memory_id in recall_ids
            assert memory_id in recent_ids
            assert memory_id in all_ids
        finally:
            _close_manager(mgr)
