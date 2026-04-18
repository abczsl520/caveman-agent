"""Hybrid retrieval — FTS5 + Jaccard + vector + trust + temporal decay.

Ported from Hermes FactRetriever (MIT, Nous Research) with Caveman adaptations.
Replaces simple FTS5/vector search with multi-signal scoring pipeline.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from caveman.memory.types import MemoryEntry


# --- Jaccard similarity ---

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "this", "that", "these", "those", "it", "its", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "if", "then", "than", "very", "just", "also",
    "的", "了", "是", "在", "和", "有", "我", "他", "她", "它", "们",
    "这", "那", "不", "也", "就", "都", "要", "会", "能", "可以",
})


_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(_CJK_RANGE.search(text))


def _tokenize_cjk(text: str) -> list[str]:
    """Tokenize text with jieba for CJK, regex for Latin."""
    try:
        import jieba
        jieba.setLogLevel(20)  # suppress loading messages
        if not getattr(jieba, '_caveman_initialized', False):
            jieba.initialize()  # Pre-load dictionary (290ms cold, 0ms warm)
            jieba._caveman_initialized = True
        return list(jieba.cut(text))
    except ImportError:
        # Fallback: character bigrams for CJK + word split for Latin
        tokens = []
        cjk_buf = []
        for ch in text:
            if _CJK_RANGE.match(ch):
                cjk_buf.append(ch)
            else:
                if cjk_buf:
                    s = ''.join(cjk_buf)
                    tokens.extend(s[i:i+2] for i in range(len(s)-1))
                    cjk_buf = []
        if cjk_buf:
            s = ''.join(cjk_buf)
            tokens.extend(s[i:i+2] for i in range(len(s)-1))
        tokens.extend(re.findall(r'\b\w+\b', text.lower()))
        return tokens


def tokenize(text: str) -> set[str]:
    """Tokenize text into a set of meaningful words (supports CJK + Latin)."""
    if _has_cjk(text):
        words = _tokenize_cjk(text.lower())
    else:
        words = re.findall(r'\b\w+\b', text.lower())
    return {w.strip() for w in words if w.strip() not in _STOP_WORDS and len(w.strip()) > 1}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# --- Entity extraction ---

_RE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_QUOTED = re.compile(r'["\']([^"\']+)["\']')
_RE_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_RE_URL = re.compile(r'https?://\S+')
_RE_PATH = re.compile(r'(?:/[\w.-]+){2,}')


def extract_entities(text: str) -> list[str]:
    """Extract entities from text (names, IPs, URLs, paths, quoted terms)."""
    entities: list[str] = []

    # Capitalized multi-word names
    entities.extend(_RE_CAPITALIZED.findall(text))

    # Quoted terms
    entities.extend(_RE_QUOTED.findall(text))

    # IPs
    entities.extend(_RE_IP.findall(text))

    # URLs
    entities.extend(_RE_URL.findall(text))

    # File paths
    entities.extend(_RE_PATH.findall(text))

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        e_lower = e.lower().strip()
        if e_lower not in seen and len(e_lower) > 1:
            seen.add(e_lower)
            result.append(e.strip())
    return result


# --- Temporal decay ---

def temporal_decay(created_at: datetime, half_life_days: int = 30) -> float:
    """Exponential decay based on age. Returns multiplier in (0, 1]."""
    if half_life_days <= 0:
        return 1.0
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


# --- Trust scoring ---

_HELPFUL_DELTA = 0.08
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0


def adjust_trust(current: float, helpful: bool) -> float:
    """Adjust trust score based on feedback."""
    delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
    return max(_TRUST_MIN, min(_TRUST_MAX, current + delta))


# --- Hybrid scorer ---

class HybridScorer:
    """Multi-signal scoring for memory retrieval.

    Combines FTS5 rank, Jaccard similarity, vector similarity,
    trust score, and temporal decay into a single relevance score.

    Ported from Hermes FactRetriever (MIT, Nous Research).
    """

    def __init__(
        self,
        fts_weight: float = 0.30,
        jaccard_weight: float = 0.25,
        vector_weight: float = 0.20,
        trust_weight: float = 0.25,
        temporal_half_life_days: int = 90,
    ):
        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.vector_weight = vector_weight
        self.trust_weight = trust_weight
        self.temporal_half_life_days = temporal_half_life_days
        self._context_type: str = ""  # set by caller for context-aware ranking

    def set_context(self, query: str) -> None:
        """Detect query context for type-aware ranking.

        debug → procedural memories more valuable
        design → semantic memories more valuable
        coding → episodic memories (past similar tasks) more valuable
        """
        q = query.lower()
        if any(w in q for w in ("error", "bug", "fix", "debug", "crash", "fail",
                                 "错误", "修复", "报错", "崩溃")):
            self._context_type = "debug"
        elif any(w in q for w in ("design", "architect", "refactor", "structure",
                                   "设计", "架构", "重构")):
            self._context_type = "design"
        else:
            self._context_type = ""

    def _type_boost(self, memory_type_value: str) -> float:
        """Context-aware type boost."""
        if self._context_type == "debug":
            return {"procedural": 0.1, "episodic": 0.05}.get(memory_type_value, 0.0)
        if self._context_type == "design":
            return {"semantic": 0.1, "procedural": 0.05}.get(memory_type_value, 0.0)
        return 0.0

    def score(
        self,
        query_tokens: set[str],
        entry: MemoryEntry,
        fts_rank: float = 0.0,
        vector_sim: float = 0.0,
    ) -> float:
        """Compute hybrid relevance score for a memory entry.

        Signals: FTS5 rank + Jaccard + vector + trust + popularity.
        Popularity (retrieval_count) is the compound interest of the flywheel:
        memories that get used more are more likely to be useful again.
        """
        content_tokens = tokenize(entry.content)
        jaccard = jaccard_similarity(query_tokens, content_tokens)

        # Normalize FTS rank (SQLite FTS5 rank is negative, lower = better)
        fts_score = 1.0 / (1.0 + abs(fts_rank)) if fts_rank != 0 else 0.0

        # Trust from metadata
        trust = entry.metadata.get("trust_score", 0.5)

        # Popularity boost: log(1 + retrieval_count) / 10, capped at 0.3
        # This is the compound interest of the flywheel:
        # memories used more → rank higher → get used more
        retrieval_count = entry.metadata.get("retrieval_count", 0)
        if not isinstance(retrieval_count, (int, float)):
            retrieval_count = 0
        popularity = min(0.3, math.log1p(retrieval_count) / 10)

        # Combine signals
        relevance = (
            self.fts_weight * fts_score
            + self.jaccard_weight * jaccard
            + self.vector_weight * max(0.0, vector_sim)
            + self.trust_weight * trust
        )

        # Popularity is a meaningful signal, not noise — scale it properly
        # rc=10 → +0.023, rc=50 → +0.039, rc=100 → +0.046, rc=300 → +0.057
        relevance += popularity * 0.2

        # Context-aware type boost (debug → procedural, design → semantic)
        relevance += self._type_boost(entry.memory_type.value)

        # Adaptive temporal decay: based on max(created_at, last_accessed)
        # This implements "memory rejuvenation" — frequently used memories stay young.
        # A memory accessed yesterday has decay ≈ 1.0 regardless of when it was created.
        effective_age_start = entry.created_at
        last_accessed = entry.metadata.get("last_accessed")
        if last_accessed:
            try:
                la = datetime.fromisoformat(last_accessed)
                if la.tzinfo is None:
                    la = la.replace(tzinfo=timezone.utc)
                if la > effective_age_start.replace(tzinfo=timezone.utc) if effective_age_start.tzinfo is None else la > effective_age_start:
                    effective_age_start = la
            except (ValueError, TypeError):
                pass

        # High-trust memories decay slower
        trust_factor = 0.6 + trust * 1.4  # Range: 0.6 (trust=0) to 2.0 (trust=1)
        effective_half_life = int(self.temporal_half_life_days * trust_factor)
        decay = temporal_decay(effective_age_start, effective_half_life)
        return relevance * decay

    def rerank(
        self,
        query: str,
        entries: list[MemoryEntry],
        fts_ranks: dict[str, float] | None = None,
        vector_sims: dict[str, float] | None = None,
        limit: int = 10,
    ) -> list[tuple[float, MemoryEntry]]:
        """Rerank entries by hybrid score. Returns (score, entry) pairs."""
        self.set_context(query)
        query_tokens = tokenize(query)
        fts_ranks = fts_ranks or {}
        vector_sims = vector_sims or {}

        scored = [
            (
                self.score(
                    query_tokens, entry,
                    fts_rank=fts_ranks.get(entry.id, 0.0),
                    vector_sim=vector_sims.get(entry.id, 0.0),
                ),
                entry,
            )
            for entry in entries
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]


# ── Cross-language query expansion (CJK ↔ Latin) ──

# Common dev terms: Chinese → English mapping
# This is the minimum viable set for the Day 3 flywheel to work
# for Chinese developers working with English codebases.
_ZH_EN_DEV_TERMS: dict[str, list[str]] = {
    "登录": ["login", "auth", "signin"],
    "注册": ["register", "signup"],
    "用户": ["user", "account"],
    "密码": ["password", "passwd"],
    "数据库": ["database", "db", "sql"],
    "服务器": ["server", "host"],
    "部署": ["deploy", "deployment"],
    "上线": ["deploy", "release", "production"],
    "发布": ["release", "publish", "deploy"],
    "测试": ["test", "testing"],
    "配置": ["config", "configuration", "settings"],
    "接口": ["api", "endpoint", "interface"],
    "文件": ["file", "document"],
    "项目": ["project", "repo"],
    "错误": ["error", "bug", "issue"],
    "修复": ["fix", "resolve", "patch"],
    "安装": ["install", "setup"],
    "依赖": ["dependency", "dependencies", "package"],
    "权限": ["permission", "auth", "access"],
    "缓存": ["cache", "caching"],
    "日志": ["log", "logging"],
    "构建": ["build", "compile"],
    "分支": ["branch", "git"],
    "提交": ["commit", "push"],
    "合并": ["merge", "pull request"],
    "容器": ["container", "docker"],
    "网络": ["network", "http", "request"],
    "前端": ["frontend", "ui", "page"],
    "后端": ["backend", "server", "api"],
    "页面": ["page", "view", "template"],
    "组件": ["component", "widget"],
    "路由": ["route", "router", "routing"],
    "模型": ["model", "schema"],
    "迁移": ["migration", "migrate"],
    "环境": ["environment", "env"],
    "端口": ["port"],
    "版本": ["version"],
}

# Reverse mapping: English → Chinese
_EN_ZH_DEV_TERMS: dict[str, list[str]] = {}
for _zh, _en_list in _ZH_EN_DEV_TERMS.items():
    for _en in _en_list:
        _EN_ZH_DEV_TERMS.setdefault(_en, []).append(_zh)


def expand_query_cross_lang(query: str) -> str:
    """Expand a query with cross-language dev terms.

    '修复登录页面的bug' → '修复登录页面的bug login auth signin bug'

    This bridges the FTS5 gap for Chinese developers working with
    English codebases — the core Day 3 flywheel moment.
    """
    extra_terms: list[str] = []

    # CJK → English expansion
    for zh_term, en_terms in _ZH_EN_DEV_TERMS.items():
        if zh_term in query:
            extra_terms.extend(en_terms)

    # English → CJK expansion
    query_lower = query.lower()
    for en_term, zh_terms in _EN_ZH_DEV_TERMS.items():
        if en_term in query_lower.split():
            extra_terms.extend(zh_terms)

    if extra_terms:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = []
        for t in extra_terms:
            if t not in seen and t not in query:
                seen.add(t)
                unique.append(t)
        return query + " " + " ".join(unique[:8])  # Cap expansion

    return query
