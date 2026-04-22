"""Session Knowledge Extractor — distill high-value knowledge from conversations.

Three-stage pipeline:
  1. **Filter**: Score each turn for knowledge density, discard noise
  2. **Cluster**: Group related turns into coherent knowledge units
  3. **Extract**: Use LLM to produce structured knowledge entries

Design principles:
  - User controls whether to enable extraction (opt-in per session)
  - Quality over quantity — better to miss some knowledge than import noise
  - Preserve provenance — every extracted entry traces back to source session
  - Idempotent — re-extracting the same session produces the same results
  - LLM-optional — can do rule-based extraction without LLM, LLM enhances quality
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from caveman.memory.types import MemoryType

from .session_parser import (
    ConversationTurn,
    ParsedSession,
)

__all__ = [
    "KnowledgeCategory",
    "TurnScore",
    "score_turn",
    "KnowledgeCluster",
    "cluster_turns",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge classification
# ---------------------------------------------------------------------------

class KnowledgeCategory(Enum):
    """What kind of knowledge was extracted."""
    RESEARCH = "research"           # Analysis, investigation, study results
    DIAGNOSIS = "diagnosis"         # Root cause analysis, debugging findings
    ARCHITECTURE = "architecture"   # Design decisions, system architecture
    SOLUTION = "solution"           # Working solutions, fixes, implementations
    LESSON = "lesson"               # Lessons learned, gotchas, pitfalls
    DECISION = "decision"           # Product/technical decisions made
    STATUS = "status"               # Project status updates, milestones
    REFERENCE = "reference"         # API docs, config references, how-tos


# Category → MemoryType mapping
_CATEGORY_TO_MEMORY_TYPE: dict[KnowledgeCategory, MemoryType] = {
    KnowledgeCategory.RESEARCH: MemoryType.SEMANTIC,
    KnowledgeCategory.DIAGNOSIS: MemoryType.EPISODIC,
    KnowledgeCategory.ARCHITECTURE: MemoryType.SEMANTIC,
    KnowledgeCategory.SOLUTION: MemoryType.PROCEDURAL,
    KnowledgeCategory.LESSON: MemoryType.PROCEDURAL,
    KnowledgeCategory.DECISION: MemoryType.SEMANTIC,
    KnowledgeCategory.STATUS: MemoryType.EPISODIC,
    KnowledgeCategory.REFERENCE: MemoryType.SEMANTIC,
}


# ---------------------------------------------------------------------------
# Knowledge scoring — rule-based signal detection
# ---------------------------------------------------------------------------

# Positive signals: patterns that indicate high-value knowledge
_POSITIVE_SIGNALS: list[tuple[str, float, str]] = [
    # (pattern, weight, description)
    (r"根因|root cause|根本原因", 2.0, "root_cause_analysis"),
    (r"架构|architecture|设计决策", 1.8, "architecture_discussion"),
    (r"研究|分析|调研|investigation|analysis", 1.5, "research_analysis"),
    (r"方案|solution|解决|修复|fix", 1.3, "solution_found"),
    (r"教训|lesson|踩坑|gotcha|坑", 1.5, "lesson_learned"),
    (r"决策|decision|选择|trade-?off", 1.4, "decision_made"),
    (r"发现|found|discovered|关键发现", 1.2, "discovery"),
    (r"原因|because|因为|所以|therefore", 0.8, "causal_reasoning"),
    (r"对比|比较|vs\.?|versus|优劣", 1.3, "comparison"),
    (r"总结|summary|结论|conclusion", 1.5, "conclusion"),
    (r"步骤|step|流程|procedure|how.to", 1.0, "procedural"),
    (r"配置|config|设置|setup", 0.8, "configuration"),
    (r"性能|performance|优化|optimize", 1.2, "performance"),
    (r"安全|security|漏洞|vulnerability", 1.3, "security"),
    (r"```", 0.5, "has_code_block"),
    (r"Phase \d|阶段|milestone", 1.0, "phased_plan"),
]

# Negative signals: patterns that indicate low-value content
_NEGATIVE_SIGNALS: list[tuple[str, float, str]] = [
    (r"^(ok|好的|收到|嗯|明白|了解)$", -3.0, "trivial_ack"),
    (r"我来看看|让我查|我来查", -1.0, "filler_text"),
    (r"HEARTBEAT_OK|heartbeat", -2.0, "heartbeat_noise"),
    (r"正在执行|executing|running", -0.5, "status_noise"),
    (r"stream_read_error|502|timeout", -0.3, "error_noise"),
]


@dataclass
class TurnScore:
    """Scoring result for a single conversation turn."""
    turn: ConversationTurn
    score: float = 0.0
    signals: list[str] = field(default_factory=list)
    category: KnowledgeCategory | None = None

    @property
    def is_extractable(self) -> bool:
        """Is this turn worth extracting knowledge from?"""
        return self.score >= 2.0 and self.category is not None


def score_turn(turn: ConversationTurn) -> TurnScore:
    """Score a conversation turn for knowledge density.

    Combines:
    - Text length (longer = more likely to contain knowledge)
    - Positive signal pattern matching
    - Negative signal pattern matching
    - Structural signals (code blocks, lists, headers)
    """
    result = TurnScore(turn=turn)

    # Collect all text to analyze
    all_text = turn.assistant_prose
    for msg in turn.messages_sent:
        all_text += "\n" + msg
    all_text_lower = all_text.lower()

    if not all_text.strip():
        return result

    # --- Length score ---
    text_len = len(all_text)
    if text_len < 50:
        result.score -= 2.0
        result.signals.append("too_short")
    elif text_len >= 500:
        result.score += 1.0
        result.signals.append("substantial_length")
    if text_len >= 1500:
        result.score += 0.5
        result.signals.append("long_form")

    # --- Positive signals ---
    for pattern, weight, name in _POSITIVE_SIGNALS:
        if re.search(pattern, all_text_lower):
            result.score += weight
            result.signals.append(f"+{name}")

    # --- Negative signals ---
    for pattern, weight, name in _NEGATIVE_SIGNALS:
        if re.search(pattern, all_text_lower):
            result.score += weight  # weight is already negative
            result.signals.append(f"-{name}")

    # --- Structural signals ---
    # Numbered lists suggest structured analysis
    if re.search(r"^\d+\.\s", all_text, re.MULTILINE):
        result.score += 0.5
        result.signals.append("+numbered_list")

    # Markdown headers suggest organized content
    if re.search(r"^#{1,3}\s", all_text, re.MULTILINE):
        result.score += 0.5
        result.signals.append("+has_headers")

    # Bold text suggests emphasis on key points
    if re.search(r"\*\*[^*]+\*\*", all_text):
        result.score += 0.3
        result.signals.append("+has_emphasis")

    # --- Message tool bonus ---
    # Messages sent to user are curated output, higher quality
    if turn.messages_sent:
        msg_total = sum(len(m) for m in turn.messages_sent)
        if msg_total >= 200:
            result.score += 1.0
            result.signals.append("+substantial_user_message")

    # --- Infer category ---
    result.category = _infer_category(all_text_lower, result.signals)

    return result


def _infer_category(text_lower: str, signals: list[str]) -> KnowledgeCategory | None:
    """Infer the knowledge category from text content and signals."""
    # Priority-ordered category detection
    if any(s.endswith("root_cause_analysis") for s in signals):
        return KnowledgeCategory.DIAGNOSIS
    if any(s.endswith("architecture_discussion") for s in signals):
        return KnowledgeCategory.ARCHITECTURE
    if any(s.endswith("research_analysis") for s in signals):
        return KnowledgeCategory.RESEARCH
    if any(s.endswith("lesson_learned") for s in signals):
        return KnowledgeCategory.LESSON
    if any(s.endswith("decision_made") for s in signals):
        return KnowledgeCategory.DECISION
    if any(s.endswith("solution_found") for s in signals):
        return KnowledgeCategory.SOLUTION
    if any(s.endswith("procedural") for s in signals):
        return KnowledgeCategory.REFERENCE
    if any(s.endswith("comparison") for s in signals):
        return KnowledgeCategory.RESEARCH

    # Fallback: if score is high enough but no specific category
    # Use text heuristics
    if "状态" in text_lower or "进度" in text_lower or "完成" in text_lower:
        return KnowledgeCategory.STATUS
    if "配置" in text_lower or "config" in text_lower:
        return KnowledgeCategory.REFERENCE

    return None


# ---------------------------------------------------------------------------
# Knowledge clustering — group related turns
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeCluster:
    """A group of related turns that form a coherent knowledge unit."""
    scored_turns: list[TurnScore]
    category: KnowledgeCategory
    session_id: str = ""
    session_date: str = ""
    topic_hint: str = ""

    @property
    def combined_text(self) -> str:
        """All extractable text from the cluster, ordered by turn index."""
        parts = []
        for st in sorted(self.scored_turns, key=lambda s: s.turn.turn_index):
            # Prefer messages sent to user (curated), then assistant prose
            msgs = st.turn.messages_sent
            if msgs:
                parts.extend(msgs)
            elif st.turn.assistant_prose:
                parts.append(st.turn.assistant_prose)
        return "\n\n---\n\n".join(parts)

    @property
    def content_hash(self) -> str:
        """Deterministic hash for deduplication."""
        text = self.combined_text
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @property
    def total_chars(self) -> int:
        return len(self.combined_text)

    @property
    def avg_score(self) -> float:
        if not self.scored_turns:
            return 0.0
        return sum(st.score for st in self.scored_turns) / len(self.scored_turns)


def cluster_turns(
    scored_turns: list[TurnScore],
    session: ParsedSession,
) -> list[KnowledgeCluster]:
    """Group extractable turns into coherent knowledge clusters.

    Strategy:
    - Adjacent turns with the same category → merge into one cluster
    - Non-adjacent turns with the same category → separate clusters
    - Single high-scoring turns → standalone clusters
    """
    extractable = [st for st in scored_turns if st.is_extractable]
    if not extractable:
        return []

    clusters: list[KnowledgeCluster] = []
    current_group: list[TurnScore] = []
    current_category: KnowledgeCategory | None = None

    for st in extractable:
        if current_category is None:
            # Start first group
            current_group = [st]
            current_category = st.category
        elif (
            st.category == current_category
            and st.turn.turn_index - current_group[-1].turn.turn_index <= 3
        ):
            # Same category and close together → merge
            current_group.append(st)
        else:
            # Different category or too far apart → flush and start new
            if current_group and current_category:
                clusters.append(KnowledgeCluster(
                    scored_turns=current_group,
                    category=current_category,
                    session_id=session.metadata.session_id,
                    session_date=session.metadata.date,
                    topic_hint=session.topic_hint,
                ))
            current_group = [st]
            current_category = st.category

    # Flush last group
    if current_group and current_category:
        clusters.append(KnowledgeCluster(
            scored_turns=current_group,
            category=current_category,
            session_id=session.metadata.session_id,
            session_date=session.metadata.date,
            topic_hint=session.topic_hint,
        ))

    return clusters


# ---------------------------------------------------------------------------
# Knowledge extraction — produce memory entries
# ---------------------------------------------------------------------------

