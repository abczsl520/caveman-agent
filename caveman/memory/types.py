"""Memory type definitions.

Two classification dimensions (PRD §4.1 + §5.2):
  - MemoryType: cognitive type (how the memory is structured)
  - MemorySource: PRD source category (where the memory came from)

Storage uses MemoryType. Nudge extraction uses MemorySource for labeling,
then maps to MemoryType for storage. Both are stored in metadata for
full traceability.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


class MemoryType(Enum):
    """Cognitive memory types (storage dimension)."""
    EPISODIC = "episodic"       # What happened (events, outcomes)
    SEMANTIC = "semantic"       # Facts (project info, API details)
    PROCEDURAL = "procedural"   # How-to (steps, patterns)
    WORKING = "working"         # Temporary (current task context)


class MemorySource(Enum):
    """PRD §5.2 source categories (labeling dimension).

    These map to MemoryType for storage:
      user      → SEMANTIC  (user preferences are facts about the user)
      feedback  → EPISODIC  (feedback is an event record)
      project   → SEMANTIC  (project knowledge is factual)
      reference → PROCEDURAL (reference material is how-to)
    """
    USER = "user"           # User preferences, working style
    FEEDBACK = "feedback"   # Task feedback, corrections
    PROJECT = "project"     # Project structure, dependencies
    REFERENCE = "reference" # Reference material, docs, patterns


# PRD source → cognitive type mapping
SOURCE_TO_TYPE: dict[MemorySource, MemoryType] = {
    MemorySource.USER: MemoryType.SEMANTIC,
    MemorySource.FEEDBACK: MemoryType.EPISODIC,
    MemorySource.PROJECT: MemoryType.SEMANTIC,
    MemorySource.REFERENCE: MemoryType.PROCEDURAL,
}


@dataclass
class MemoryEntry:
    """A single memory record with content, type, and metadata."""
    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Well-Known Metadata Keys (PRD §8.9.2) ---
# Keys with defined semantics that engines depend on.
# Keys starting with '_' are temporary/internal (not persisted across versions).

class MetadataKeys:
    """Well-known metadata_json keys — PRD §8.9.2 contract.

    Engines MUST use these constants instead of raw strings to prevent
    typos and enable refactoring. Other keys are allowed but not guaranteed
    to be read by any engine.
    """
    # Trust & retrieval (written by store/recall)
    TRUST_SCORE = "trust_score"           # float — redundant copy for MemoryEntry transport
    RETRIEVAL_COUNT = "retrieval_count"    # int — redundant copy for MemoryEntry transport
    LAST_ACCESSED = "last_accessed"        # str ISO 8601 — HybridScorer temporal_decay
    # Source tracking (written by Nudge)
    SOURCE = "source"                      # str — MemorySource value: user/feedback/project/reference
    NUDGE_TRIGGER = "nudge_trigger"        # str — shield_update/tool_error/loop_end
    # Knowledge network (written by Ripple)
    RELATED = "related"                    # list[str] — related memory IDs (graph edges)
    SUPERSEDES = "supersedes"              # str — ID of memory this one replaces
    CONFLICT_WITH = "conflict_with"        # str — ID of contradicting memory
    # Grounding (written by Grounding Gate)
    GROUNDING_STATUS = "grounding_status"  # str — verified/unverified/stale
    GROUNDING_CHECKED_AT = "grounding_checked_at"  # str ISO 8601
    # Legacy (deprecated, remove in v0.5)
    CONFIDENCE = "confidence"              # float — old ConfidenceTracker score
    # Internal/temporary (prefixed with _, not persisted across versions)
    _FTS_RANK = "_fts_rank"               # float — FTS5 rank score
    _VECTOR_SIM = "_vector_sim"           # float — vector cosine similarity


# --- Unified type resolution (used by Nudge + LLM extraction) ---

# Accepts both PRD source labels and cognitive type names
TYPE_MAP: dict[str, MemoryType] = {
    # PRD §5.2 source types
    "user": MemoryType.SEMANTIC,
    "feedback": MemoryType.EPISODIC,
    "project": MemoryType.SEMANTIC,
    "reference": MemoryType.PROCEDURAL,
    # Cognitive types (backward compat)
    "episodic": MemoryType.EPISODIC,
    "semantic": MemoryType.SEMANTIC,
    "procedural": MemoryType.PROCEDURAL,
    "working": MemoryType.WORKING,
}


def resolve_memory_type(type_str: str) -> MemoryType:
    """Resolve a type string to MemoryType, with fallback."""
    return TYPE_MAP.get(type_str.lower(), MemoryType.EPISODIC)


def get_turn_text(turn: dict) -> str:
    """Extract text content from a conversation turn dict."""
    content = turn.get("content", turn.get("value", ""))
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)
