"""Memory metadata contract validation.

Well-known metadata keys are intentionally permissive about unknown keys, but
known keys must have stable types because retrieval, lint, ripple and decay
engines read them directly.  Invalid known-key values are dropped with a
WARNING instead of being persisted and breaking later reads.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .types import MetadataKeys

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataSpec:
    """Runtime contract for one well-known metadata key."""

    expected: tuple[type, ...]
    validator: Callable[[Any], bool] | None = None
    description: str = ""

    def accepts(self, value: Any) -> bool:
        # bool is a subclass of int; metadata counters/scores must not accept it.
        if bool in (type(value),) and any(t in self.expected for t in (int, float)):
            return False
        if not isinstance(value, self.expected):
            return False
        return self.validator(value) if self.validator else True


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _known_source(value: Any) -> bool:
    # Backward-compatible union:
    # - PRD source labels: user/feedback/project/reference
    # - writer labels used by production code: nudge/user_tool/import/etc.
    if not isinstance(value, str) or not value.strip():
        return False
    return len(value) <= 80


def _trust_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


METADATA_REGISTRY: dict[str, MetadataSpec] = {
    MetadataKeys.TRUST_SCORE: MetadataSpec((int, float), _trust_score, "0..1 trust score"),
    MetadataKeys.RETRIEVAL_COUNT: MetadataSpec((int,), lambda v: v >= 0, "non-negative retrieval count"),
    MetadataKeys.LAST_ACCESSED: MetadataSpec((str,), _is_iso_datetime, "ISO-8601 datetime"),
    MetadataKeys.SOURCE: MetadataSpec((str,), _known_source, "memory writer/source label"),
    MetadataKeys.NUDGE_TRIGGER: MetadataSpec((str,), lambda v: bool(v.strip()), "nudge trigger label"),
    MetadataKeys.RELATED: MetadataSpec((list,), _str_list, "related memory IDs"),
    MetadataKeys.SUPERSEDES: MetadataSpec((str,), lambda v: bool(v.strip()), "superseded memory ID"),
    MetadataKeys.CONFLICT_WITH: MetadataSpec((str,), lambda v: bool(v.strip()), "conflicting memory ID/content"),
    MetadataKeys.GROUNDING_STATUS: MetadataSpec((str,), lambda v: v in {"verified", "unverified", "stale"}, "grounding status"),
    MetadataKeys.GROUNDING_CHECKED_AT: MetadataSpec((str,), _is_iso_datetime, "ISO-8601 datetime"),
    MetadataKeys.CONFIDENCE: MetadataSpec((int, float), _trust_score, "legacy confidence score"),
    MetadataKeys._FTS_RANK: MetadataSpec((int, float), lambda v: not isinstance(v, bool), "internal FTS rank"),
    MetadataKeys._VECTOR_SIM: MetadataSpec((int, float), lambda v: not isinstance(v, bool), "internal vector similarity"),
}


def validate_metadata(metadata: dict[str, Any] | None, *, context: str = "memory") -> dict[str, Any]:
    """Return a sanitized copy of metadata.

    Unknown keys are preserved. Known keys with invalid types/ranges are dropped
    and logged at WARNING so bad writes are visible without taking down the
    memory pipeline.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        logger.warning("Invalid %s metadata: expected dict, got %s; dropping metadata", context, type(metadata).__name__)
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            logger.warning("Invalid %s metadata key %r: expected str; dropping key", context, key)
            continue
        spec = METADATA_REGISTRY.get(key)
        if spec and not spec.accepts(value):
            expected = "/".join(t.__name__ for t in spec.expected)
            logger.warning(
                "Invalid %s metadata[%s]: expected %s (%s), got %s=%r; dropping key",
                context,
                key,
                expected,
                spec.description or "well-known key contract",
                type(value).__name__,
                value,
            )
            continue
        sanitized[key] = value
    return sanitized
