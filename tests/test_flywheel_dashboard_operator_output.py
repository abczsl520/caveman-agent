"""Boundary tests for flywheel dashboard source reports."""

from caveman.training.flywheel_dashboard import FlywheelDashboard


def _dashboard_with_memory(memory_stats):
    dashboard = FlywheelDashboard()
    dashboard.metrics["memory"] = memory_stats
    dashboard.metrics["trajectories"] = {}
    dashboard.metrics["rl_router"] = {}
    dashboard.metrics["wiki"] = {}
    return dashboard


def test_source_breakdown_report_escapes_source_labels():
    unsafe_label = "import:evil\nSPOOF_SOURCE\x1b[31m"
    dashboard = _dashboard_with_memory(
        {
            "total": 1,
            "avg_trust": 0.1,
            "recalled": 0,
            "never_recalled": 1,
            "helpful": 0,
            "prune_candidates": 0,
            "source_breakdown": [
                {
                    "label": unsafe_label,
                    "total": 1,
                    "active": 1,
                    "quarantined": 0,
                    "eligible_for_source_policy": 1,
                    "never_recalled_pct": 1.0,
                    "helpful_pct": 0.0,
                }
            ],
        }
    )

    report = dashboard.format_report()

    assert "'import:evil\\nSPOOF_SOURCE\\x1b[31m': n=1" in report
    assert "\nSPOOF_SOURCE" not in report
    assert "import:evil\n" not in report


def test_source_governance_report_escapes_source_labels():
    unsafe_label = "import:governance\nSPOOF_GOV\x1b[32m"
    dashboard = _dashboard_with_memory(
        {
            "total": 1,
            "avg_trust": 0.1,
            "recalled": 0,
            "never_recalled": 1,
            "helpful": 0,
            "prune_candidates": 0,
            "source_governance": [
                {
                    "label": unsafe_label,
                    "eligible_for_source_policy": 1,
                    "quarantined": 0,
                    "noise_score": 1.0,
                }
            ],
        }
    )

    report = dashboard.format_report()

    assert "'import:governance\\nSPOOF_GOV\\x1b[32m': eligible=1" in report
    assert "\nSPOOF_GOV" not in report
    assert "import:governance\n" not in report


def test_type_breakdown_report_escapes_type_labels():
    unsafe_label = "semantic\nSPOOF_TYPE\x1b[33m"
    dashboard = _dashboard_with_memory(
        {
            "total": 1,
            "avg_trust": 0.1,
            "recalled": 0,
            "never_recalled": 1,
            "helpful": 0,
            "prune_candidates": 0,
            "type_breakdown": [
                {
                    "label": unsafe_label,
                    "total": 1,
                    "avg_trust": 0.1,
                    "never_recalled_pct": 1.0,
                    "helpful_pct": 0.0,
                }
            ],
        }
    )

    report = dashboard.format_report()

    assert "'semantic\\nSPOOF_TYPE\\x1b[33m': n=1" in report
    assert "\nSPOOF_TYPE" not in report
    assert "semantic\n" not in report


def test_rl_router_report_escapes_skill_names():
    dashboard = _dashboard_with_memory({"total": 0, "avg_trust": 1.0})
    dashboard.metrics["rl_router"] = {
        "total_updates": 1,
        "arms": {
            "skill\nSPOOF_SKILL\x1b[35m": {
                "win_rate": 0.75,
                "alpha": 3,
                "beta": 1,
            }
        },
    }

    report = dashboard.format_report()

    assert "'skill\\nSPOOF_SKILL\\x1b[35m': win_rate=75.0%" in report
    assert "\nSPOOF_SKILL" not in report
    assert "skill\n" not in report


def test_wiki_report_escapes_tier_names():
    dashboard = _dashboard_with_memory({"total": 0, "avg_trust": 1.0})
    dashboard.metrics["wiki"] = {
        "total_entries": 1,
        "tiers": {"semantic\nSPOOF_WIKI\x1b[36m": 1},
    }

    report = dashboard.format_report()

    assert "'semantic\\nSPOOF_WIKI\\x1b[36m': 1" in report
    assert "\nSPOOF_WIKI" not in report
    assert "semantic\n" not in report
