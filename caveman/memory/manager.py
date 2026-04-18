"""Memory manager v3 — pluggable backend with SQLite default.

Breaking change from v2: MemoryManager now delegates to a MemoryBackend.
Default backend: SQLiteMemoryStore (FTS5 + hybrid retrieval).
Fallback: in-memory JSON (legacy, for tests or when SQLite unavailable).
"""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING

from .types import MemoryType, MemoryEntry
from .retrieval import HybridScorer, tokenize
from .recall_cache import RecallCache
from caveman.utils import cosine_similarity as _cosine_similarity

if TYPE_CHECKING:
    from .backend import MemoryBackend

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages agent memory with pluggable backend."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        embedding_fn=None,
        retrieval_log=None,
        ripple_engine=None,
        backend: "MemoryBackend | None" = None,
    ):
        from caveman.paths import MEMORY_DIR
        self.base_dir = Path(base_dir).expanduser() if base_dir else MEMORY_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = embedding_fn
        self._retrieval_log = retrieval_log
        self._lock = asyncio.Lock()
        self._recall_cache = RecallCache()
        self._ripple = ripple_engine

        self._backend = backend
        self._use_backend = backend is not None

        if not self._use_backend:
            import warnings
            warnings.warn(
                "MemoryManager without SQLite backend is deprecated. "
                "Use MemoryManager.with_sqlite() for production. "
                "JSON-only mode will be removed in v0.5.",
                DeprecationWarning, stacklevel=2,
            )

        # Legacy JSON storage (only used when no backend)
        self._memories: dict[MemoryType, list[MemoryEntry]] = {t: [] for t in MemoryType}
        self._embeddings: dict[str, list[float]] = {}

    @classmethod
    def with_sqlite(
        cls, base_dir: Path | str | None = None, db_path: Path | str | None = None,
        embedding_fn=None, retrieval_log=None, ripple_engine=None,
        scorer_config: dict | None = None,
    ) -> "MemoryManager":
        """Create a MemoryManager backed by SQLite + FTS5 (recommended)."""
        from .sqlite_store import SQLiteMemoryStore
        store = SQLiteMemoryStore(
            db_path=db_path, embedding_fn=embedding_fn,
            scorer_config=scorer_config,
        )
        return cls(
            base_dir=base_dir, embedding_fn=embedding_fn,
            retrieval_log=retrieval_log, ripple_engine=ripple_engine, backend=store,
        )

    def set_ripple(self, engine) -> None:
        self._ripple = engine

    async def store(self, content: str, memory_type: MemoryType, metadata: dict | None = None, trusted: bool = False) -> str:
        if self._use_backend:
            mid = await self._backend.store(content, memory_type, metadata, trusted=trusted)
            if not mid:  # Quality gate rejected
                return ""
            self._recall_cache.invalidate()
            if self._ripple:
                try:
                    entry = MemoryEntry(
                        id=mid, content=content, memory_type=memory_type,
                        created_at=datetime.now(), metadata=metadata or {},
                    )
                    await self._ripple.propagate(entry)
                except Exception as e:
                    logger.warning("Ripple propagation failed for %s: %s", mid, e)
            return mid
        return await self._store_json(content, memory_type, metadata, trusted=trusted)

    async def _store_json(self, content: str, memory_type: MemoryType, metadata: dict | None = None, trusted: bool = False) -> str:
        if not trusted:
            from .security import scan_memory_content
            threat = scan_memory_content(content)
            if threat:
                raise ValueError(threat)

        async with self._lock:
            mid = str(uuid.uuid4())
            entry = MemoryEntry(
                id=mid, content=content, memory_type=memory_type,
                created_at=datetime.now(), metadata=metadata or {},
            )
            self._memories[memory_type].append(entry)
            if self._embedding_fn:
                try:
                    self._embeddings[mid] = await self._embedding_fn(content)
                except Exception as e:
                    logger.debug("Embedding failed for %s: %s", mid, e)
            await self._save_unlocked()

        self._recall_cache.invalidate()
        if self._ripple:
            try:
                await self._ripple.propagate(entry)
            except Exception as e:
                logger.warning("Ripple propagation failed for %s: %s", mid, e)
        return mid

    async def recall(
        self, query: str, memory_type: MemoryType | None = None, top_k: int = 5
    ) -> List[MemoryEntry]:
        cached = self._recall_cache.get(query, top_k, memory_type)
        if cached is not None:
            return cached

        if self._use_backend:
            results = await self._backend.recall(query, memory_type, top_k)
            self._recall_cache.put(query, top_k, memory_type, results)
            if self._retrieval_log and results:
                try:
                    self._retrieval_log.log_search(
                        query=query, results=[(1.0, e) for e in results], source="memory_search",
                    )
                except Exception as e:
                    logger.debug("Retrieval log write failed: %s", e)
            return results

        return await self._recall_json(query, memory_type, top_k)

    async def _recall_json(
        self, query: str, memory_type: MemoryType | None = None, top_k: int = 5
    ) -> List[MemoryEntry]:
        top_results = await self._recall_json_scored(query, memory_type, top_k)
        results = [e for _, e in top_results]
        self._recall_cache.put(query, top_k, memory_type, results)
        return results

    async def _recall_json_scored(
        self, query: str, memory_type: MemoryType | None = None, top_k: int = 5
    ) -> List[tuple[float, MemoryEntry]]:
        async with self._lock:
            if not any(self._memories.values()):
                await self.load()
            search_types = [memory_type] if memory_type else list(MemoryType)
            all_entries = [e for mt in search_types for e in self._memories.get(mt, [])]

        if not all_entries:
            return []

        vector_sims: dict[str, float] = {}
        if self._embedding_fn and self._embeddings:
            try:
                query_emb = await self._embedding_fn(query)
                for entry in all_entries:
                    if entry.id in self._embeddings:
                        vector_sims[entry.id] = _cosine_similarity(query_emb, self._embeddings[entry.id])
            except Exception:
                vector_sims = {}

        # Keyword search → FTS-like ranks
        fts_ranks: dict[str, float] = {}
        keywords = list(tokenize(query))
        if keywords:
            for mt in search_types:
                for entry in self._memories.get(mt, []):
                    hits = sum(1 for kw in keywords if kw in entry.content.lower())
                    if hits > 0:
                        score = hits / len(keywords)
                        fts_ranks[entry.id] = -1.0 / (score + 1e-9)

        scorer = HybridScorer()
        top_results = scorer.rerank(
            query=query, entries=all_entries,
            fts_ranks=fts_ranks, vector_sims=vector_sims, limit=top_k,
        )
        if self._retrieval_log and top_results:
            try:
                self._retrieval_log.log_search(query=query, results=top_results, source="memory_search")
            except Exception as e:
                logger.debug("Retrieval log write failed: %s", e)
        return top_results

    async def recall_scored(
        self, query: str, memory_type: MemoryType | None = None, top_k: int = 5
    ) -> List[tuple[float, MemoryEntry]]:
        if self._use_backend:
            # Use HybridScorer for real scores instead of fake decreasing scores.
            # This matters for confidence feedback — fake scores mean fake learning.
            from .retrieval import HybridScorer, tokenize
            results = await self._backend.recall(query, memory_type, top_k)
            if not results:
                return []
            scorer = HybridScorer()
            query_tokens = tokenize(query)
            return [(scorer.score(query_tokens, e), e) for e in results]
        return await self._recall_json_scored(query, memory_type, top_k)

    async def forget(self, memory_id: str) -> bool:
        if self._use_backend:
            result = await self._backend.forget(memory_id)
            if result:
                self._recall_cache.invalidate()
            return result
        async with self._lock:
            for mt in MemoryType:
                for i, entry in enumerate(self._memories.get(mt, [])):
                    if entry.id == memory_id:
                        self._memories[mt].pop(i)
                        self._embeddings.pop(memory_id, None)
                        await self._save_unlocked()
                        self._recall_cache.invalidate()
                        return True
            return False

    async def save(self) -> None:
        if self._use_backend:
            return
        async with self._lock:
            await self._save_unlocked()

    async def update_metadata(self, memory_id: str, metadata: dict) -> bool:
        if self._use_backend:
            return await self._backend.update_metadata(memory_id, metadata)
        async with self._lock:
            for mt in MemoryType:
                for entry in self._memories.get(mt, []):
                    if entry.id == memory_id:
                        entry.metadata.update(metadata)
                        await self._save_unlocked()
                        self._recall_cache.invalidate()
                        return True
            return False

    async def _save_unlocked(self) -> None:
        for mt in MemoryType:
            path = self.base_dir / f"{mt.value}.json"
            entries = [
                {"id": e.id, "content": e.content, "type": e.memory_type.value,
                 "created_at": e.created_at.isoformat(), "metadata": e.metadata}
                for e in self._memories.get(mt, [])
            ]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

        if self._embeddings:
            emb_path = self.base_dir / "_embeddings.json"
            emb_tmp = emb_path.with_suffix(".tmp")
            emb_tmp.write_text(json.dumps(self._embeddings, ensure_ascii=False), encoding="utf-8")
            emb_tmp.replace(emb_path)

    async def load(self) -> None:
        if self._use_backend:
            return
        for mt in MemoryType:
            path = self.base_dir / f"{mt.value}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._memories[mt] = [
                    MemoryEntry(
                        id=e["id"], content=e["content"],
                        memory_type=MemoryType(e["type"]),
                        created_at=datetime.fromisoformat(e["created_at"]),
                        metadata=e.get("metadata", {}),
                    )
                    for e in data
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load memories from %s: %s", path, e)

        emb_path = self.base_dir / "_embeddings.json"
        if emb_path.exists():
            try:
                self._embeddings = json.loads(emb_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Failed to load embeddings: %s", e)

    @property
    def total_count(self) -> int:
        if self._use_backend:
            return self._backend.total_count
        return sum(len(entries) for entries in self._memories.values())

    def all_entries(self) -> list[MemoryEntry]:
        if self._use_backend:
            return self._backend.all_entries()
        return [e for entries in self._memories.values() for e in entries]

    def search_sync(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        if self._use_backend:
            return self._backend.search_sync(query, limit)
        query_lower = query.lower()
        scored = []
        for entry in self.all_entries():
            hits = sum(1 for w in query_lower.split() if w in entry.content.lower())
            if hits > 0:
                scored.append((hits, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def recent(self, limit: int = 20) -> list[MemoryEntry]:
        if self._use_backend:
            return self._backend.recent(limit)
        all_mem = self.all_entries()
        all_mem.sort(key=lambda e: e.created_at, reverse=True)
        return all_mem[:limit]

    async def nudge(self) -> None:
        """Background memory consolidation — runs when memory count exceeds threshold."""
        pass

    async def store_batch(self, items: list[dict]) -> list[str]:
        ids: list[str] = []
        for item in items:
            mid = await self.store(
                content=item["content"],
                memory_type=MemoryType(item.get("memory_type", "semantic")),
                metadata=item.get("metadata"),
            )
            ids.append(mid)
        return ids

    async def recall_batch(self, queries: list[str], limit: int = 5) -> list[list[MemoryEntry]]:
        return [await self.recall(q, top_k=limit) for q in queries]
