"""Memory grounding — verify memories against current reality."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ground"]


def ground(
    memories: list[Any],
    check_paths: bool = False,
) -> list[tuple[Any, dict]]:
    """Verify memories against reality and return (memory, result) pairs.

    Each result dict contains:
        confidence_modifier: float (0.0-1.0) — how much to trust this memory
        reason: str — why the modifier was applied
    """
    results = []
    for mem in memories:
        modifier = 1.0
        reason = "ok"

        if check_paths:
            # If memory references a file path, check it still exists
            content = getattr(mem, "content", "") or getattr(mem, "text", "") or str(mem)
            for token in content.split():
                if token.startswith("/") and len(token) > 3 and not token.startswith("//"):
                    if not os.path.exists(token):
                        modifier = min(modifier, 0.3)
                        reason = f"referenced path missing: {token[:60]}"
                        break

        results.append((mem, {"confidence_modifier": modifier, "reason": reason}))
    return results
