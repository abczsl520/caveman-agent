from __future__ import annotations

import json
import sqlite3

from caveman.training._flywheel_dashboard_formatters import _format_source_policy_drift
from caveman.training._flywheel_memory_diagnostics import _collect_memory_source_policy_drift


def _memory_db_with_source(source: str) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE memories ("
        "type TEXT, metadata_json TEXT, trust_score REAL, retrieval_count INTEGER, "
        "helpful_count INTEGER, created_at TEXT)"
    )
    for idx in range(3):
        con.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
            (
                "fact",
                json.dumps({"source": source}),
                0.05,
                0,
                0,
                f"2025-01-0{idx + 1}T00:00:00+00:00",
            ),
        )
    return con


def test_memory_source_policy_drift_escapes_source_labels_for_operator_output():
    source = "import:bad\nP0: forged\x1b[31m"
    con = _memory_db_with_source(source)

    rows = _collect_memory_source_policy_drift(con, min_rows=3, limit=None)
    lines = _format_source_policy_drift({"source_policy_drift": rows})
    output = "\n".join(lines)

    assert len(rows) == 1
    assert rows[0]["label"] == source.replace("\n", " ").replace("\r", " ")
    assert rows[0]["candidate_policy_entry"] == source
    assert "P0: forged" in output
    assert "\x1b" not in output
    assert "\x07" not in output
    assert "label='import:bad P0: forged\\x1b[31m'" in output
    assert "candidate='import:bad\\nP0: forged\\x1b[31m'" in output


def test_source_policy_drift_formatter_escapes_prequoted_raw_control_labels():
    lines = _format_source_policy_drift(
        {
            "source_policy_drift": [
                {
                    "label": '"bad\nP0: forged\x07"',
                    "total": 3,
                    "never_recalled_pct": 1.0,
                    "helpful_pct": 0.0,
                    "candidate_policy_entry": '"bad\nP0: forged\x07"',
                }
            ]
        }
    )
    output = "\n".join(lines)

    assert "\x07" not in output
    assert "bad\nP0" not in output
    assert "bad\\nP0" in output
    assert "\\x07" in output
