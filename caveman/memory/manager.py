"""Memory manager v3 — pluggable backend with SQLite default.

Breaking change from v2: MemoryManager now delegates to a MemoryBackend.
Default backend: SQLiteMemoryStore (FTS5 + hybrid retrieval).
Fallback: in-memory JSON (legacy, for tests or when SQLite unavailable).
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING, cast

from caveman.training.retrieval_log import RetrievalLog

from .types import MemoryType, MemoryEntry
from .metadata import validate_metadata
from .retrieval import HybridScorer, tokenize
from .recall_cache import RecallCache
from caveman.utils import cosine_similarity as _cosine_similarity
from caveman.aio import aio_exists, aio_read_text, aio_write_text

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
        bus=None,
    ):
        from caveman.paths import MEMORY_DIR
        self.base_dir = Path(base_dir).expanduser() if base_dir else MEMORY_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = embedding_fn
        self._retrieval_log = retrieval_log
        self._lock = asyncio.Lock()
        self._recall_cache = RecallCache()
        self._ripple = ripple_engine
        self._bus = bus

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

    @property
    def backend(self):
        """Return the active memory backend (read-only public accessor)."""
        return self._backend

    @classmethod
    def with_sqlite(
        cls, base_dir: Path | str | None = None, db_path: Path | str | None = None,
        embedding_fn=None, retrieval_log=None, ripple_engine=None,
        scorer_config: dict | None = None,
        quality_llm_fn=None,
        use_llm_quality_gate: bool = False,
    ) -> "MemoryManager":
        """Create a MemoryManager backed by SQLite + FTS5 (recommended)."""
        from .sqlite_store import SQLiteMemoryStore
        if db_path is None and base_dir is not None:
            db_path = Path(base_dir).expanduser() / "caveman.db"
        if retrieval_log is None:
            retrieval_log = cls._default_retrieval_log(base_dir=base_dir, db_path=db_path)
        store = SQLiteMemoryStore(
            db_path=db_path, embedding_fn=embedding_fn,
            scorer_config=scorer_config,
            quality_llm_fn=quality_llm_fn,
            use_llm_quality_gate=use_llm_quality_gate,
        )
        return cls(
            base_dir=base_dir, embedding_fn=embedding_fn,
            retrieval_log=retrieval_log, ripple_engine=ripple_engine, backend=store,
        )

    @staticmethod
    def _default_retrieval_log(
        base_dir: Path | str | None = None,
        db_path: Path | str | None = None,
    ) -> RetrievalLog:
        """Build the default retrieval log without breaking isolated memory stores.

        Production managers with no explicit store path feed the global training
        flywheel. Managers created with a custom base_dir/db_path (tests,
        doctor/selftests, imports, sandboxes) keep telemetry next to that custom
        store instead of silently polluting the user's global training database.
        """
        if base_dir is not None:
            return RetrievalLog(Path(base_dir).expanduser() / "training" / "retrieval_log.sqlite")
        if db_path is not None:
            db_dir = Path(db_path).expanduser().parent
            return RetrievalLog(db_dir / "training" / "retrieval_log.sqlite")
        return RetrievalLog()

    def set_ripple(self, engine) -> None:
        self._ripple = engine

    async def store(self, content: str, memory_type: MemoryType, metadata: dict | None = None, trusted: bool = False) -> str:
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            mid = await backend.store(content, memory_type, metadata, trusted=trusted)
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
            # Emit MEMORY_STORE event for flywheel Chain 4 (Lint) and metrics
            if self._bus:
                try:
                    from caveman.events import EventType
                    await self._bus.emit(EventType.MEMORY_STORE, {
                        "memory_id": mid, "content": content,
                        "type": memory_type.value,
                        "source": (metadata or {}).get("source", ""),
                        "metadata": metadata or {},
                    }, source="memory")
                except Exception as e:
                    logger.debug("MEMORY_STORE emit failed: %s", e)
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
            meta = validate_metadata(metadata, context="json memory store")
            entry = MemoryEntry(
                id=mid, content=content, memory_type=memory_type,
                created_at=datetime.now(), metadata=meta,
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
            self._log_cached_recall(query, cached)
            return cached

        if self._use_backend:
            start = time.perf_counter()
            backend = cast("MemoryBackend", self._backend)
            results = await backend.recall(query, memory_type, top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            self._recall_cache.put(query, top_k, memory_type, results)
            if self._retrieval_log and results:
                try:
                    self._retrieval_log.log_search(
                        query=query, results=[(1.0, e) for e in results],
                        source="memory_search", latency_ms=latency_ms,
                    )
                except Exception as e:
                    logger.debug("Retrieval log write failed: %s", e)
            return results

        return await self._recall_json(query, memory_type, top_k)

    def _log_cached_recall(self, query: str, results: List[MemoryEntry]) -> None:
        """Log cache-hit recalls so retrieval analytics reflect actual usage."""
        if not self._retrieval_log or not results:
            return
        try:
            self._retrieval_log.log_search(
                query=query, results=[(1.0, e) for e in results],
                source="memory_search_cache", latency_ms=0.0,
            )
        except Exception as e:
            logger.debug("Retrieval log cache write failed: %s", e)

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
            start = time.perf_counter()
            backend = cast("MemoryBackend", self._backend)
            results = await backend.recall(query, memory_type, top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            if not results:
                return []
            scorer = HybridScorer()
            query_tokens = tokenize(query)
            scored_results = [(scorer.score(query_tokens, e), e) for e in results]
            if self._retrieval_log:
                try:
                    self._retrieval_log.log_search(
                        query=query, results=scored_results,
                        source="memory_search_scored", latency_ms=latency_ms,
                    )
                except Exception as e:
                    logger.debug("Retrieval log scored write failed: %s", e)
            return scored_results
        return await self._recall_json_scored(query, memory_type, top_k)

    async def forget(self, memory_id: str) -> bool:
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            result = await backend.forget(memory_id)
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
            backend = cast("MemoryBackend", self._backend)
            return await backend.update_metadata(memory_id, metadata)
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
            await aio_write_text(tmp, json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

        if self._embeddings:
            emb_path = self.base_dir / "_embeddings.json"
            emb_tmp = emb_path.with_suffix(".tmp")
            await aio_write_text(emb_tmp, json.dumps(self._embeddings, ensure_ascii=False), encoding="utf-8")
            emb_tmp.replace(emb_path)

    async def load(self) -> None:
        if self._use_backend:
            return
        for mt in MemoryType:
            path = self.base_dir / f"{mt.value}.json"
            if not await aio_exists(path):
                continue
            try:
                data = json.loads(await aio_read_text(path, encoding="utf-8"))
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
        if await aio_exists(emb_path):
            try:
                self._embeddings = json.loads(await aio_read_text(emb_path, encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Failed to load embeddings: %s", e)

    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a memory by ID, or None if it does not exist."""
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            return await backend.get_by_id(memory_id)
        async with self._lock:
            if not any(self._memories.values()):
                await self.load()
            for entries in self._memories.values():
                for entry in entries:
                    if entry.id == memory_id:
                        return entry
        return None

    @property
    def total_count(self) -> int:
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            return backend.total_count
        return sum(len(entries) for entries in self._memories.values())

    def all_entries(self) -> list[MemoryEntry]:
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            return backend.all_entries()
        return [e for entries in self._memories.values() for e in entries]

    def search_sync(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        if self._use_backend:
            backend = cast("MemoryBackend", self._backend)
            return backend.search_sync(query, limit)
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
            backend = cast("MemoryBackend", self._backend)
            return backend.recent(limit)
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
        return list(await asyncio.gather(*(self.recall(q, top_k=limit) for q in queries)))
