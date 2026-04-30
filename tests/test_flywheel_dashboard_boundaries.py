"""Boundary tests for flywheel dashboard observability."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from caveman.training.flywheel_dashboard import FlywheelDashboard


def _make_memory_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    rows = [
        ("n1", "nudge low", "episodic", "2026-03-16T00:00:00+00:00", '{"source":"nudge"}', 0.2, 0, 0),
        ("n2", "nudge used", "episodic", "2026-03-16T00:00:00+00:00", '{"source":"nudge"}', 0.6, 3, 1),
        (
            "o1", "openclaw bulk", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:openclaw"}', 0.05, 0, 0,
        ),
        (
            "os-alias", "openclaw session alias", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:openclaw_sessions"}', 0.05, 0, 0,
        ),
        (
            "oq", "openclaw quarantined", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:openclaw","governance_state":"quarantined"}', 0.01, 0, 0,
        ),
        (
            "he", "hermes eligible after decay", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:hermes","governance_state":"active"}', 0.09, 0, 0,
        ),
        (
            "hn", "hermes too new", "semantic", "2026-04-29T00:00:00+00:00",
            '{"source":"import:hermes","governance_state":"active"}', 0.04, 0, 0,
        ),
        (
            "ho", "hermes too old", "semantic", "2025-12-01T00:00:00+00:00",
            '{"source":"import:hermes","governance_state":"active"}', 0.04, 0, 0,
        ),
        ("m1", "missing source", "procedural", "2026-03-16T00:00:00+00:00", '{}', 0.4, 1, 0),
        ("bad", "bad metadata", "semantic", "2026-03-16T00:00:00+00:00", '{', 0.1, 0, 0),
        (
            "long", "long source", "procedural", "2026-03-16T00:00:00+00:00",
            '{"source":"very-long-source-with-newline\\n-and-a-very-very-very-very-very-very-very-very-very-long-tail"}',
            0.3, 0, 0,
        ),
    ]
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_collect_memory_stats_stratifies_by_source_and_type(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _make_memory_db(memory_dir / "caveman.db")
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    dashboard = FlywheelDashboard()
    stats = dashboard.collect_memory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 11
    assert stats["avg_trust"] == 0.171
    assert stats["source_breakdown"][:3] == [
        {
            "label": "import:hermes",
            "total": 3,
            "avg_trust": 0.057,
            "never_recalled": 3,
            "never_recalled_pct": 1.0,
            "helpful": 0,
            "helpful_pct": 0.0,
            "active": 3,
            "quarantined": 0,
            "eligible_for_source_policy": 1,
        },
        {
            "label": "import:openclaw",
            "total": 3,
            "avg_trust": 0.037,
            "never_recalled": 3,
            "never_recalled_pct": 1.0,
            "helpful": 0,
            "helpful_pct": 0.0,
            "active": 2,
            "quarantined": 1,
            "eligible_for_source_policy": 2,
        },
        {
            "label": "nudge",
            "total": 2,
            "avg_trust": 0.4,
            "never_recalled": 1,
            "never_recalled_pct": 0.5,
            "helpful": 1,
            "helpful_pct": 0.5,
            "active": 2,
            "quarantined": 0,
            "eligible_for_source_policy": 0,
        },
    ]
    assert stats["type_breakdown"] == [
        {
            "label": "semantic",
            "total": 7,
            "avg_trust": 0.054,
            "never_recalled": 7,
            "never_recalled_pct": 1.0,
            "helpful": 0,
            "helpful_pct": 0.0,
        },
        {
            "label": "episodic",
            "total": 2,
            "avg_trust": 0.4,
            "never_recalled": 1,
            "never_recalled_pct": 0.5,
            "helpful": 1,
            "helpful_pct": 0.5,
        },
        {
            "label": "procedural",
            "total": 2,
            "avg_trust": 0.35,
            "never_recalled": 1,
            "never_recalled_pct": 0.5,
            "helpful": 0,
            "helpful_pct": 0.0,
        },
    ]
    assert stats["source_governance"][:2] == [
        {
            "label": "import:openclaw",
            "total": 3,
            "active": 2,
            "quarantined": 1,
            "eligible_for_source_policy": 2,
            "noise_score": 1.0,
            "recall_candidate_reduction_pct": 0.333,
        },
        {
            "label": "import:hermes",
            "total": 3,
            "active": 3,
            "quarantined": 0,
            "eligible_for_source_policy": 1,
            "noise_score": 1.0,
            "recall_candidate_reduction_pct": 0.0,
        },
    ]

    report = dashboard.format_report()
    assert "By source (top):" in report
    assert "nudge: n=2, active=2, quarantined=0, eligible=0, noise=25%, recall-reduction=0%" in report
    assert "import:openclaw: n=3, active=2, quarantined=1, eligible=2, noise=100%, recall-reduction=33%" in report
    assert "Source governance actions:" in report
    long_label = next(row["label"] for row in stats["source_breakdown"] if row["label"].startswith("very-long"))
    assert "\n" not in long_label
    assert len(long_label) == 80
    assert long_label.endswith("…")


def test_source_governance_includes_actionable_sources_beyond_source_breakdown_limit(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    rows = [
        (
            f"bulk-{idx}",
            f"bulk {idx}",
            "episodic",
            "2026-03-16T00:00:00+00:00",
            json.dumps({"source": f"bulk:{idx}"}),
            0.9,
            1,
            1,
        )
        for idx in range(13)
    ]
    rows.append(
        (
            "actionable",
            "small actionable source",
            "semantic",
            "2026-03-16T00:00:00+00:00",
            '{"source":"import:hermes-skill-ref"}',
            0.04,
            0,
            0,
        )
    )
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert all(row["label"] != "import:hermes-skill-ref" for row in stats["source_breakdown"])
    assert stats["source_governance"] == [
        {
            "label": "import:hermes-skill-ref",
            "total": 1,
            "active": 1,
            "quarantined": 0,
            "eligible_for_source_policy": 1,
            "noise_score": 1.0,
            "recall_candidate_reduction_pct": 0.0,
        }
    ]


def test_source_governance_uses_canonical_identity_not_display_truncation(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    long_source = "import:" + "x" * 120
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("long-src", "long source", "semantic", "2026-03-16T00:00:00+00:00", json.dumps({"source": long_source}), 0.04, 0, 0),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)
    monkeypatch.setattr(
        "caveman.training._flywheel_memory_diagnostics.SOURCE_POLICY_LOW_SIGNAL_IMPORTS",
        frozenset({long_source}),
    )

    stats = FlywheelDashboard().collect_memory_stats()

    assert stats["source_governance"][0]["eligible_for_source_policy"] == 1
    assert len(stats["source_governance"][0]["label"]) == 80


def test_collect_memory_stats_keeps_legacy_schema_working(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO memories (id, content, trust_score, retrieval_count, helpful_count) VALUES "
        "('legacy', 'old schema memory', 0.7, 1, 1)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 1
    assert stats["avg_trust"] == 0.7
    assert stats["source_breakdown"] == []
    assert stats["type_breakdown"] == []


def test_collect_memory_stats_keeps_partial_legacy_source_schema_working(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata_json TEXT DEFAULT '{}', "
        "trust_score REAL DEFAULT 0.5, retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO memories (id, content, metadata_json, trust_score, retrieval_count, helpful_count) VALUES "
        "('legacy-source', 'old schema memory', '{\"source\":\"import:hermes\"}', 0.04, 0, 0)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 1
    assert stats["source_breakdown"] == []
    assert stats.get("source_governance", []) == []


def test_memory_stats_include_decay_dry_run_operator_report(tmp_path, monkeypatch):
    eligible_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    prune_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    rows = [
        (
            "openclaw-eligible", "openclaw eligible", "semantic", eligible_date,
            '{"source":"import:openclaw"}', 0.05, 0, 0,
        ),
        (
            "hermes-quarantined", "hermes quarantined", "semantic", eligible_date,
            '{"source":"import:hermes","governance_state":"quarantined"}', 0.01, 0, 0,
        ),
        (
            "generic-prune", "generic prune", "semantic", prune_date,
            '{"source":"manual"}', 0.01, 0, 0,
        ),
        (
            "helpful-protected", "helpful protected", "semantic", eligible_date,
            '{"source":"import:openclaw"}', 0.05, 0, 1,
        ),
    ]
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    report = stats["decay_dry_run"]
    assert report["scanned"] == 4
    assert report["would_decay"] >= 2
    assert report["would_prune"] == 1
    assert report["would_quarantine"] == 1
    assert report["would_quarantine_by_source"] == {"import:openclaw": 1}
    assert report["eligible_by_source"] == {"import:openclaw": 1}
    assert stats["already_quarantined"] == 1
    formatted = FlywheelDashboard()
    formatted.metrics["memory"] = stats
    formatted.metrics["trajectories"] = {}
    formatted.metrics["rl_router"] = {}
    formatted.metrics["wiki"] = {}
    assert "Decay dry-run: scan=4, would_decay=" in formatted.format_report()
    assert "would_prune=1, would_quarantine=1" in formatted.format_report()

    persisted = sqlite3.connect(db_path).execute(
        "SELECT trust_score, metadata_json FROM memories WHERE id = ?",
        ("openclaw-eligible",),
    ).fetchone()
    assert persisted[0] == 0.05
    assert "governance_state" not in json.loads(persisted[1])


def test_memory_stats_skips_decay_preview_on_sqlite_error(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _make_memory_db(memory_dir / "caveman.db")
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    with patch("caveman.training.flywheel_dashboard.MemoryDecay") as decay_cls:
        decay_cls.return_value.run.side_effect = sqlite3.OperationalError("database is locked")
        stats = FlywheelDashboard().collect_memory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 11
    assert "decay_dry_run" not in stats


def test_collect_trajectory_stats_skips_malformed_and_normalizes_numeric_fields(tmp_path, monkeypatch):
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    (traj_dir / "good.json").write_text(
        json.dumps({"metadata": {"tool_calls": "2", "quality_score": "0.8"}}),
        encoding="utf-8",
    )
    (traj_dir / "bad_meta.json").write_text(json.dumps({"metadata": ["not", "dict"]}), encoding="utf-8")
    (traj_dir / "bad_numbers.json").write_text(
        json.dumps({"metadata": {"tool_calls": "many", "quality_score": "bad"}}),
        encoding="utf-8",
    )
    (traj_dir / "list_root.json").write_text(json.dumps([{"metadata": {}}]), encoding="utf-8")
    (traj_dir / "broken.json").write_text("{", encoding="utf-8")

    (traj_dir / "jsonl_entries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"quality_score": 0.9, "tool_calls": 1}),
                "not json",
                json.dumps(["not", "object"]),
                json.dumps({"metadata": {"quality_score": 0.3, "tool_calls": 0}}),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("caveman.training.flywheel_dashboard.TRAJECTORIES_DIR", traj_dir)

    stats = FlywheelDashboard().collect_trajectory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 5
    assert stats["with_tools"] == 2
    assert stats["high_quality"] == 2
    assert stats["low_quality"] == 1
    assert stats["avg_quality"] == 0.6


def test_collect_rl_router_stats_ignores_malformed_arms(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".rl_router_state.json").write_text(
        json.dumps(
            {
                "arms": {
                    "good": {"alpha": "3", "beta": "1"},
                    "bad_numbers": {"alpha": "x", "beta": "2"},
                    "negative": {"alpha": -1, "beta": 2},
                    "not_dict": [1, 2, 3],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("caveman.training.flywheel_dashboard.SKILLS_DIR", skills_dir)

    stats = FlywheelDashboard().collect_rl_router_stats()

    assert stats["status"] == "ok"
    assert stats["total_updates"] == 2
    assert stats["arms"] == {
        "good": {"alpha": 3.0, "beta": 1.0, "updates": 2, "win_rate": 0.75},
    }


def test_collect_rl_router_stats_accepts_direct_skill_router_state(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".rl_router_state.json").write_text(
        json.dumps(
            {
                "alpha_skill": {"alpha": 4, "beta": 2, "total": 4},
                "beta_skill": {"alpha": "2", "beta": "5", "total": "5"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("caveman.training.flywheel_dashboard.SKILLS_DIR", skills_dir)

    stats = FlywheelDashboard().collect_rl_router_stats()

    assert stats["status"] == "ok"
    assert stats["total_updates"] == 9
    assert stats["arms"] == {
        "alpha_skill": {"alpha": 4.0, "beta": 2.0, "updates": 4, "win_rate": 0.667},
        "beta_skill": {"alpha": 2.0, "beta": 5.0, "updates": 5, "win_rate": 0.286},
    }


def test_collect_rl_router_stats_reports_invalid_state_file(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".rl_router_state.json").write_text(json.dumps(["not", "object"]), encoding="utf-8")
    monkeypatch.setattr("caveman.training.flywheel_dashboard.SKILLS_DIR", skills_dir)

    stats = FlywheelDashboard().collect_rl_router_stats()

    assert stats["status"] == "error: invalid state file"
    assert stats["total_updates"] == 0



def test_source_policy_drift_flags_unmanaged_low_signal_import_sources(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    rows = [
        (
            f"cc-{idx}", f"claude code import {idx}", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:claude-code"}', 0.05, 0, 0,
        )
        for idx in range(3)
    ]
    rows.append(("manual", "manual low", "semantic", "2026-03-16T00:00:00+00:00", '{"source":"manual"}', 0.01, 0, 0))
    rows.append(("helpful", "helpful import", "semantic", "2026-03-16T00:00:00+00:00", '{"source":"import:rare"}', 0.05, 0, 1))
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert stats["source_policy_drift"] == [
        {
            "label": "import:claude-code",
            "total": 3,
            "active": 3,
            "never_recalled_pct": 1.0,
            "helpful_pct": 0.0,
            "avg_trust": 0.05,
            "reason": "unmanaged_low_signal_import",
            "recommended_action": "review_for_low_signal_allowlist",
            "candidate_policy_entry": "import:claude-code",
        }
    ]
    formatted = FlywheelDashboard()
    formatted.metrics["memory"] = stats
    formatted.metrics["trajectories"] = {}
    formatted.metrics["rl_router"] = {}
    formatted.metrics["wiki"] = {}
    assert "Source policy drift:" in formatted.format_report()
    assert "import:claude-code: unmanaged low-signal import source (n=3, never=100%, helpful=0%, candidate=import:claude-code)" in formatted.format_report()

def test_source_policy_drift_keeps_truncated_import_identities_separate(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    prefix = "import:" + ("same-prefix-" * 8)
    noisy_source = prefix + "noisy"
    helpful_source = prefix + "helpful"
    rows = [
        (
            f"noisy-{idx}", f"noisy import {idx}", "semantic", "2026-03-16T00:00:00+00:00",
            json.dumps({"source": noisy_source}), 0.05, 0, 0,
        )
        for idx in range(3)
    ]
    rows.extend(
        (
            f"helpful-{idx}", f"helpful import {idx}", "semantic", "2026-03-16T00:00:00+00:00",
            json.dumps({"source": helpful_source}), 0.05, 0, 1,
        )
        for idx in range(3)
    )
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert len({row["label"] for row in stats["source_breakdown"]}) == 1
    assert stats["source_policy_drift"] == [
        {
            "label": noisy_source[:79] + "…",
            "total": 3,
            "active": 3,
            "never_recalled_pct": 1.0,
            "helpful_pct": 0.0,
            "avg_trust": 0.05,
            "reason": "unmanaged_low_signal_import",
            "recommended_action": "review_for_low_signal_allowlist",
            "candidate_policy_entry": noisy_source,
        }
    ]


def test_collect_wiki_stats_accepts_list_and_object_shapes(tmp_path, monkeypatch):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "working.json").write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    (wiki_dir / "episodic.json").write_text(json.dumps({"entries": [{"a": 1}]}), encoding="utf-8")
    (wiki_dir / "semantic.json").write_text(json.dumps({"items": [{"a": 1}, {"b": 2}, {"c": 3}]}), encoding="utf-8")
    (wiki_dir / "procedural.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr("caveman.training.flywheel_dashboard.WIKI_DIR", wiki_dir)

    stats = FlywheelDashboard().collect_wiki_stats()

    assert stats["status"] == "ok"
    assert stats["tiers"] == {"working": 2, "episodic": 1, "semantic": 3, "procedural": 0}
    assert stats["total_entries"] == 6


def test_decay_dry_run_reports_restorable_quarantine_sources(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    rows = [
        (
            "q-openclaw", "quarantined openclaw", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:openclaw","governance_state":"quarantined","quarantine_reason":"source_policy_low_signal_import"}',
            0.01, 0, 0,
        ),
        (
            "q-hermes", "quarantined hermes", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:hermes","governance_state":"quarantined","quarantine_reason":"stale_low_signal_import"}',
            0.01, 0, 0,
        ),
        (
            "active-openclaw", "active openclaw", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:openclaw"}', 0.05, 0, 0,
        ),
    ]
    conn.executemany(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()

    assert stats["decay_dry_run"]["restorable_quarantine_by_source"] == {
        "import:hermes": 1,
        "import:openclaw": 1,
    }
    assert stats["decay_dry_run"]["restorable_quarantine_by_reason"] == {
        "source_policy_low_signal_import": 1,
        "stale_low_signal_import": 1,
    }
    formatted = FlywheelDashboard()
    formatted.metrics["memory"] = stats
    formatted.metrics["trajectories"] = {}
    formatted.metrics["rl_router"] = {}
    formatted.metrics["wiki"] = {}
    report = formatted.format_report()
    assert "Restorable quarantine: import:hermes=1, import:openclaw=1" in report


def test_restorable_quarantine_report_survives_decay_preview_failure(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "q-hermes", "quarantined hermes", "semantic", "2026-03-16T00:00:00+00:00",
            '{"source":"import:hermes","governance_state":"quarantined","quarantine_reason":"manual_review"}',
            0.01, 0, 0,
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    with patch("caveman.training.flywheel_dashboard.MemoryDecay") as decay_cls:
        decay_cls.return_value.run.side_effect = sqlite3.OperationalError("database is locked")
        stats = FlywheelDashboard().collect_memory_stats()

    assert "decay_dry_run" not in stats
    assert stats["restorable_quarantine_by_source"] == {"import:hermes": 1}
    assert stats["restorable_quarantine_by_reason"] == {"manual_review": 1}
    formatted = FlywheelDashboard()
    formatted.metrics["memory"] = stats
    formatted.metrics["trajectories"] = {}
    formatted.metrics["rl_router"] = {}
    formatted.metrics["wiki"] = {}
    assert "Restorable quarantine: import:hermes=1" in formatted.format_report()
