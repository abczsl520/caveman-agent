from __future__ import annotations

from typing import Any


def _format_source_policy_drift(mem: dict[str, Any]) -> list[str]:
    source_policy_drift = mem.get("source_policy_drift", [])
    if not source_policy_drift:
        return []
    lines = ["   Source policy drift:"]
    lines.extend(
        f"      {row['label']}: unmanaged low-signal import source "
        f"(n={row['total']}, never={row['never_recalled_pct']:.0%}, helpful={row['helpful_pct']:.0%})"
        for row in source_policy_drift[:5]
    )
    return lines
