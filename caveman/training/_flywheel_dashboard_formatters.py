from __future__ import annotations

from typing import Any


def _format_source_policy_drift(mem: dict[str, Any]) -> list[str]:
    source_policy_drift = mem.get("source_policy_drift", [])
    if not source_policy_drift:
        return []
    lines = ["   Source policy drift:"]
    lines.extend(
        f"      {row.get('label', '<missing>')}: unmanaged low-signal import source "
        f"(n={row.get('total', 0)}, never={float(row.get('never_recalled_pct', 0.0)):.0%}, "
        f"helpful={float(row.get('helpful_pct', 0.0)):.0%}, "
        f"candidate={row.get('candidate_policy_entry', row.get('label', '<missing>'))})"
        for row in source_policy_drift[:5]
    )
    return lines
