"""Boundary tests for flywheel dashboard observability."""

import json

from caveman.training.flywheel_dashboard import FlywheelDashboard


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


def test_collect_rl_router_stats_reports_invalid_state_file(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / ".rl_router_state.json").write_text(json.dumps(["not", "object"]), encoding="utf-8")
    monkeypatch.setattr("caveman.training.flywheel_dashboard.SKILLS_DIR", skills_dir)

    stats = FlywheelDashboard().collect_rl_router_stats()

    assert stats["status"] == "error: invalid state file"
    assert stats["total_updates"] == 0


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
