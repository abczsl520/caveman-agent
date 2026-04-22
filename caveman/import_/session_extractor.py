"""Session Knowledge Extractor — extraction stage of the knowledge pipeline.

Uses scoring and clustering from session_scoring.py, then applies
rule-based or LLM-based extraction to produce structured knowledge entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType
from .session_parser import ParsedSession
from .session_scoring import (
    KnowledgeCategory, KnowledgeCluster, score_turn, cluster_turns, _CATEGORY_TO_MEMORY_TYPE,
)

__all__ = [
    "ExtractedKnowledge",
    "extract_knowledge_rule_based",
    "EXTRACTION_PROMPT",
    "extract_knowledge_with_llm",
    "ExtractionResult",
    "extract_session",
]


logger = logging.getLogger(__name__)

@dataclass
class ExtractedKnowledge:
    """A single piece of extracted knowledge ready for memory storage."""
    content: str
    memory_type: MemoryType
    category: KnowledgeCategory
    source_session: str
    source_date: str
    topic: str
    content_hash: str
    score: float
    cluster_size: int  # How many turns contributed

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dict for memory storage."""
        return {
            "source": "import:openclaw-session",
            "session_id": self.source_session,
            "session_date": self.source_date,
            "topic": self.topic,
            "category": self.category.value,
            "extraction_score": round(self.score, 2),
            "cluster_turns": self.cluster_size,
        }


def extract_knowledge_rule_based(
    cluster: KnowledgeCluster,
    max_chars: int = 4000,
) -> ExtractedKnowledge | None:
    """Extract knowledge from a cluster using rule-based approach (no LLM).

    This is the fallback when LLM is not available. It:
    - Combines the cluster text
    - Adds a header with context
    - Truncates if too long (preserving head and tail)
    """
    text = cluster.combined_text
    if not text or len(text) < 50:
        return None

    # Build a contextual header
    header_parts = []
    if cluster.session_date:
        header_parts.append(f"[{cluster.session_date}]")
    if cluster.topic_hint:
        header_parts.append(f"Topic: {cluster.topic_hint}")
    header_parts.append(f"Category: {cluster.category.value}")
    header = " | ".join(header_parts)

    # Truncate if needed, preserving head and tail
    if len(text) > max_chars:
        head_size = int(max_chars * 0.7)
        tail_size = max_chars - head_size - 50  # 50 for separator
        text = (
            text[:head_size]
            + "\n\n[... truncated ...]\n\n"
            + text[-tail_size:]
        )

    content = f"{header}\n\n{text}"
    memory_type = _CATEGORY_TO_MEMORY_TYPE.get(
        cluster.category, MemoryType.SEMANTIC
    )

    return ExtractedKnowledge(
        content=content,
        memory_type=memory_type,
        category=cluster.category,
        source_session=cluster.session_id,
        source_date=cluster.session_date,
        topic=cluster.topic_hint,
        content_hash=cluster.content_hash,
        score=cluster.avg_score,
        cluster_size=len(cluster.scored_turns),
    )


# ---------------------------------------------------------------------------
# LLM-enhanced extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """你是一个知识提取专家。从以下 AI 对话片段中提取核心知识。

要求：
1. 提取关键结论、发现、决策、方案 — 不要提取过程性的废话
2. 保留技术细节（代码片段、配置、命令、数据）
3. 保留因果关系（为什么这样做、根因是什么）
4. 用简洁的中文输出，技术术语保留英文
5. 如果内容价值不高，直接回复 "SKIP"
6. 输出格式：直接输出提取后的知识文本，不要加额外的包装

对话上下文：
- 日期：{date}
- 主题：{topic}
- 类别：{category}

对话内容：
{text}"""


async def extract_knowledge_with_llm(
    cluster: KnowledgeCluster,
    llm_complete: Any,  # async callable(messages, system) -> str
    max_input_chars: int = 8000,
) -> ExtractedKnowledge | None:
    """Extract knowledge using LLM for higher quality distillation.

    Args:
        cluster: The knowledge cluster to extract from
        llm_complete: Async function that takes (messages, system) and returns text
        max_input_chars: Max chars to send to LLM
    """
    text = cluster.combined_text
    if not text or len(text) < 50:
        return None

    # Truncate input for LLM
    if len(text) > max_input_chars:
        text = text[:max_input_chars] + "\n\n[... truncated ...]"

    prompt = EXTRACTION_PROMPT.format(
        date=cluster.session_date or "unknown",
        topic=cluster.topic_hint or "unknown",
        category=cluster.category.value,
        text=text,
    )

    try:
        result = await llm_complete(
            messages=[{"role": "user", "content": prompt}],
            system="你是知识提取专家，从对话中提炼核心知识。简洁、准确、保留技术细节。",
        )
    except Exception as e:
        logger.warning("LLM extraction failed for cluster %s: %s", cluster.session_id, e)
        # Fallback to rule-based
        return extract_knowledge_rule_based(cluster)

    result_text = result.strip() if isinstance(result, str) else str(result).strip()

    # LLM said skip
    if result_text.upper() == "SKIP" or len(result_text) < 20:
        return None

    # Build header
    header_parts = []
    if cluster.session_date:
        header_parts.append(f"[{cluster.session_date}]")
    if cluster.topic_hint:
        header_parts.append(f"Topic: {cluster.topic_hint}")
    header = " | ".join(header_parts)

    content = f"{header}\n\n{result_text}" if header else result_text
    memory_type = _CATEGORY_TO_MEMORY_TYPE.get(
        cluster.category, MemoryType.SEMANTIC
    )

    return ExtractedKnowledge(
        content=content,
        memory_type=memory_type,
        category=cluster.category,
        source_session=cluster.session_id,
        source_date=cluster.session_date,
        topic=cluster.topic_hint,
        content_hash=cluster.content_hash,
        score=cluster.avg_score,
        cluster_size=len(cluster.scored_turns),
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result of extracting knowledge from one session."""
    session_path: Path
    session_id: str
    total_turns: int
    scored_turns: int
    extractable_turns: int
    clusters_found: int
    knowledge_extracted: list[ExtractedKnowledge] = field(default_factory=list)
    skipped_by_llm: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Session {self.session_id[:12]}… | "
            f"{self.total_turns} turns → {self.extractable_turns} extractable → "
            f"{self.clusters_found} clusters → {len(self.knowledge_extracted)} knowledge entries"
        )


async def extract_session(
    session: ParsedSession,
    llm_complete: Any | None = None,
    min_score: float = 2.0,
    max_entries_per_session: int = 20,
) -> ExtractionResult:
    """Full extraction pipeline for one parsed session.

    Args:
        session: Parsed session data
        llm_complete: Optional LLM function for enhanced extraction
        min_score: Minimum score threshold for extraction
        max_entries_per_session: Cap on entries per session to prevent flooding
    """
    result = ExtractionResult(
        session_path=session.source_path,
        session_id=session.metadata.session_id,
        total_turns=len(session.turns),
        scored_turns=0,
        extractable_turns=0,
        clusters_found=0,
    )

    # Stage 1: Score all turns
    scored = []
    for turn in session.turns:
        ts = score_turn(turn)
        scored.append(ts)
    result.scored_turns = len(scored)
    result.extractable_turns = sum(1 for s in scored if s.score >= min_score and s.category)

    # Stage 2: Cluster
    clusters = cluster_turns(scored, session)
    result.clusters_found = len(clusters)

    if not clusters:
        return result

    # Stage 3: Extract (cap at max_entries_per_session)
    # Sort clusters by average score, extract best first
    clusters.sort(key=lambda c: c.avg_score, reverse=True)
    clusters = clusters[:max_entries_per_session]

    for cluster in clusters:
        try:
            if llm_complete is not None:
                entry = await extract_knowledge_with_llm(cluster, llm_complete)
            else:
                entry = extract_knowledge_rule_based(cluster)

            if entry is not None:
                result.knowledge_extracted.append(entry)
            else:
                result.skipped_by_llm += 1
        except Exception as e:
            result.errors.append(f"Cluster extraction error: {e}")
            logger.warning("Extraction error in session %s: %s", session.metadata.session_id, e)

    return result
