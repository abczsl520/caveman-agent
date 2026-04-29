"""SQLite + FTS5 memory backend."""
from __future__ import annotations
import asyncio
import json
import logging
import sqlite3

from caveman.db import connect as db_connect
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .types import MemoryType, MemoryEntry
from .metadata import validate_metadata
from .store_helpers import row_to_entry, migrate_schema
from caveman.utils import cosine_similarity as _cosine_similarity
from caveman.memory.sqlite_schema import SCHEMA

logger = logging.getLogger(__name__)


class SQLiteMemoryStore:
    """SQLite + FTS5 backed memory store."""

    def __init__(self, db_path: Path | str | None = None, embedding_fn=None,
                 scorer_config: dict | None = None, quality_llm_fn=None,
                 use_llm_quality_gate: bool = False):
        from caveman.paths import MEMORY_DB_PATH
        self.db_path = Path(db_path).expanduser() if db_path else MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_fn = embedding_fn
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = asyncio.Lock()
        self._conn_lock = threading.Lock()
        self._scorer_config = scorer_config or {}
        self._quality_llm_fn = quality_llm_fn
        self._use_llm_quality_gate = use_llm_quality_gate

    def _get_conn(self) -> sqlite3.Connection:
        with self._conn_lock:
            if self._conn is None:
                self._conn = db_connect(self.db_path)
                self._conn.executescript(SCHEMA)
                migrate_schema(self._conn)
            return self._conn

    async def store(self, content: str, memory_type: MemoryType, metadata: dict | None = None, trusted: bool = False) -> str:
        from caveman.memory.retrieval import extract_entities
        from caveman.memory.security import scan_memory_content
        from caveman.memory.quality_gate import check_quality_async, truncate_if_needed

        if not trusted:
            threat = scan_memory_content(content)
            if threat:
                raise ValueError(threat)

        # Quality gate: reject garbage before it pollutes the flywheel
        rejection = await check_quality_async(
            content,
            trusted=trusted,
            llm_fn=self._quality_llm_fn,
            use_llm=self._use_llm_quality_gate,
        )
        if rejection:
            logger.debug("Quality gate rejected: %s — %s", content[:60], rejection)
            return ""  # Empty ID signals rejection

        content = truncate_if_needed(content)

        async with self._write_lock:
            conn = self._get_conn()
            mid = str(uuid.uuid4())
            now = datetime.now().isoformat()
            meta = validate_metadata(metadata, context="sqlite store")
            trust = meta.get("trust_score", 0.5)
            entities = extract_entities(content)

            conn.execute(
                "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, entities_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mid, content, memory_type.value, now,
                 json.dumps(meta, ensure_ascii=False), trust,
                 json.dumps(entities, ensure_ascii=False)),
            )

            if self._embedding_fn:
                try:
                    vector = await self._embedding_fn(content)
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (memory_id, vector_json) VALUES (?, ?)",
                        (mid, json.dumps(vector)),
                    )
                except Exception as e:
                    logger.debug("Embedding generation failed for %s: %s", mid, e)

            conn.commit()
        return mid

    async def recall(
        self, query: str, memory_type: MemoryType | None = None, top_k: int = 5
    ) -> List[MemoryEntry]:
        """Hybrid recall: FTS5 candidates → Jaccard + trust + decay reranking."""
        from caveman.memory.retrieval import HybridScorer, expand_query_cross_lang

        if not query.strip():
            return []

        conn = self._get_conn()
        scorer = HybridScorer(**self._scorer_config)

        expanded_query = expand_query_cross_lang(query)

        fts_entries = self._fts_search(expanded_query, memory_type, top_k * 3)
        fts_ranks = {e.id: e.metadata.get("_fts_rank", 0.0) for e in fts_entries}

        vector_sims: dict[str, float] = {}
        vec_entries: list[MemoryEntry] = []
        if self._embedding_fn:
            vec_entries = await self._vector_search(expanded_query, memory_type, top_k * 3)
            vector_sims = {e.id: e.metadata.get("_vector_sim", 0.0) for e in vec_entries}

        all_entries: dict[str, MemoryEntry] = {e.id: e for e in fts_entries}
        for e in vec_entries:
            all_entries.setdefault(e.id, e)

        if not all_entries:
            # Fallback: no FTS/vector match — return recent high-trust memories
            # Better than empty: gives LLM some context to work with
            fallback_rows = conn.execute(
                "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
                "FROM memories WHERE trust_score >= 0.5 "
                "ORDER BY trust_score DESC, COALESCE(last_accessed, created_at) DESC LIMIT ?",
                (top_k,),
            ).fetchall()
            if fallback_rows:
                return [row_to_entry(r) for r in fallback_rows]
            return []

        # Graph expansion: include cross-referenced memories (Ripple's knowledge network)
        related_ids: set[str] = set()
        for entry in all_entries.values():
            for rid in entry.metadata.get("related", []):
                if rid not in all_entries:
                    related_ids.add(rid)
        if related_ids:
            for rid in list(related_ids)[:10]:
                row = conn.execute(
                    "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
                    "FROM memories WHERE id = ?", (rid,)
                ).fetchone()
                if row:
                    entry = row_to_entry(row)
                    all_entries[entry.id] = entry

        results = scorer.rerank(
            query, list(all_entries.values()),
            fts_ranks=fts_ranks, vector_sims=vector_sims, limit=top_k,
        )

        returned_ids = [entry.id for _, entry in results]
        if returned_ids:
            async with self._write_lock:
                now = datetime.now().isoformat()
                ph = ",".join("?" * len(returned_ids))
                # Batch update: retrieval count, last_accessed, and small trust boost
                conn.execute(
                    f"UPDATE memories SET retrieval_count = retrieval_count + 1, "
                    f"last_accessed = ?, "
                    f"trust_score = MIN(1.0, CASE "
                    f"  WHEN trust_score < 0.3 THEN trust_score + 0.05 "
                    f"  ELSE trust_score + 0.01 "
                    f"END) WHERE id IN ({ph})",
                    [now, *returned_ids],
                )
                # Batch update last_accessed in metadata
                for mid in returned_ids:
                    row = conn.execute(
                        "SELECT metadata_json FROM memories WHERE id = ?", (mid,)
                    ).fetchone()
                    if row:
                        meta = json.loads(row[0]) if row[0] else {}
                        meta["last_accessed"] = now
                        conn.execute(
                            "UPDATE memories SET metadata_json = ? WHERE id = ?",
                            (json.dumps(meta, ensure_ascii=False), mid),
                        )
                conn.commit()

        return [entry for _, entry in results]

    def _fts_search(self, query: str, memory_type: MemoryType | None, top_k: int) -> List[MemoryEntry]:
        conn = self._get_conn()
        words = query.split()
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words[:10])
        try:
            base = ("SELECT m.id, m.content, m.type, m.created_at, m.metadata_json, "
                    "rank, m.trust_score, m.retrieval_count, m.last_accessed FROM memories m "
                    "JOIN memories_fts ON m.rowid = memories_fts.rowid WHERE memories_fts MATCH ?")
            if memory_type:
                rows = conn.execute(
                    base + " AND m.type = ? ORDER BY rank LIMIT ?",
                    (fts_query, memory_type.value, top_k),
                ).fetchall()
            else:
                rows = conn.execute(base + " ORDER BY rank LIMIT ?", (fts_query, top_k)).fetchall()
        except sqlite3.OperationalError:
            return self._like_search(query, memory_type, top_k)
        return [row_to_entry(row, fts_rank=row[5], trust=row[6],
                             retrieval_count=row[7] if len(row) > 7 else 0,
                             last_accessed=row[8] if len(row) > 8 else None)
                for row in rows]

    def _like_search(self, query: str, memory_type: MemoryType | None, top_k: int) -> List[MemoryEntry]:
        conn = self._get_conn()
        words = query.split()[:5]
        conditions = ["content LIKE ?"] * len(words)
        params: list = [f"%{w}%" for w in words]
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type.value)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
            f"FROM memories WHERE {where} LIMIT ?",
            params + [top_k],
        ).fetchall()
        return [row_to_entry(row) for row in rows]

    async def _vector_search(self, query: str, memory_type: MemoryType | None, top_k: int) -> List[MemoryEntry]:
        conn = self._get_conn()
        try:
            query_vec = await self._embedding_fn(query)
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return []

        cap = max(top_k * 10, 200)
        candidate_ids = self._fts_candidate_ids(query, cap)

        base = ("SELECT m.id, m.content, m.type, m.created_at, m.metadata_json, "
                "m.trust_score, m.retrieval_count, m.last_accessed, e.vector_json "
                "FROM memories m JOIN embeddings e ON m.id = e.memory_id")
        if candidate_ids:
            ph = ",".join("?" * len(candidate_ids))
            where = f" WHERE e.memory_id IN ({ph})"
            params: list = list(candidate_ids)
            if memory_type:
                where += " AND m.type = ?"
                params.append(memory_type.value)
            rows = conn.execute(base + where, params).fetchall()
        elif memory_type:
            rows = conn.execute(base + " WHERE m.type = ? LIMIT ?", (memory_type.value, cap)).fetchall()
        else:
            rows = conn.execute(base + " LIMIT ?", (cap,)).fetchall()

        if not rows:
            return []
        scored = []
        for row in rows:
            sim = _cosine_similarity(query_vec, json.loads(row[8]))
            entry = row_to_entry(row[:8], trust=row[5], retrieval_count=row[6], last_accessed=row[7])
            entry.metadata["_vector_sim"] = sim
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _fts_candidate_ids(self, query: str, cap: int) -> list[str] | None:
        conn = self._get_conn()
        words = query.split()[:10]
        if not words:
            return None
        fts_q = " OR ".join(f'"{w}"' for w in words)
        try:
            rows = conn.execute(
                "SELECT m.id FROM memories m "
                "JOIN memories_fts ON m.rowid = memories_fts.rowid "
                "WHERE memories_fts MATCH ? LIMIT ?", (fts_q, cap),
            ).fetchall()
            return [r[0] for r in rows] if rows else None
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return None

    async def mark_helpful(self, memory_id: str, helpful: bool = True, metadata: dict | None = None) -> None:
        from caveman.memory.retrieval import adjust_trust
        async with self._write_lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT trust_score, helpful_count, metadata_json FROM memories WHERE id = ?", (memory_id,),
            ).fetchone()
            if not row:
                return
            new_trust = adjust_trust(row[0], helpful)
            delta = 1 if helpful else 0
            existing_meta = json.loads(row[2]) if row[2] else {}
            if metadata:
                existing_meta.update(validate_metadata(metadata, context="mark_helpful"))
            conn.execute(
                "UPDATE memories SET trust_score = ?, helpful_count = helpful_count + ?, metadata_json = ? WHERE id = ?",
                (new_trust, delta, json.dumps(existing_meta, ensure_ascii=False), memory_id),
            )
            conn.commit()

    def search_by_entity(self, entity: str, top_k: int = 10) -> List[MemoryEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed FROM memories "
            'WHERE entities_json LIKE ? ORDER BY trust_score DESC LIMIT ?',
            (f'%"{entity}"%', top_k),
        ).fetchall()
        return [row_to_entry(row) for row in rows]

    async def update_metadata(self, memory_id: str, metadata: dict) -> bool:
        async with self._write_lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT metadata_json FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return False
            existing = json.loads(row[0]) if row[0] else {}
            existing.update(validate_metadata(metadata, context="update_metadata"))
            conn.execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                (json.dumps(existing, ensure_ascii=False), memory_id),
            )
            conn.commit()
        return True

    async def forget(self, memory_id: str) -> bool:
        async with self._write_lock:
            conn = self._get_conn()
            # Clean up cross-refs pointing to this memory (prevent dangling refs)
            rows = conn.execute(
                "SELECT id, metadata_json FROM memories WHERE metadata_json LIKE ?",
                (f'%{memory_id}%',),
            ).fetchall()
            for row in rows:
                try:
                    meta = json.loads(row[1]) if row[1] else {}
                    related = meta.get("related", [])
                    if memory_id in related:
                        related.remove(memory_id)
                        meta["related"] = related
                        conn.execute(
                            "UPDATE memories SET metadata_json = ? WHERE id = ?",
                            (json.dumps(meta, ensure_ascii=False), row[0]),
                        )
                except Exception as exc:
                    logger.debug("forget: suppressed %s", exc)
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a memory by ID, or None if it does not exist."""
        row = self._get_conn().execute(
            "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        return row_to_entry(
            row,
            trust=row[5],
            retrieval_count=row[6] if len(row) > 6 else 0,
            last_accessed=row[7] if len(row) > 7 else None,
        )

    @property
    def total_count(self) -> int:
        row = self._get_conn().execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0]) if row is not None else 0

    def type_counts(self) -> dict[str, int]:
        rows = self._get_conn().execute("SELECT type, COUNT(*) FROM memories GROUP BY type").fetchall()
        return {row[0]: row[1] for row in rows}

    def all_entries(self) -> List[MemoryEntry]:
        rows = self._get_conn().execute(
            "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count, last_accessed "
            "FROM memories ORDER BY created_at"
        ).fetchall()
        entries = []
        for row in rows:
            entry = row_to_entry(
                row,
                trust=row[5],
                retrieval_count=row[6],
                last_accessed=row[8],
            )
            entry.metadata["helpful_count"] = int(row[7] or 0)
            entries.append(entry)
        return entries

    def search_sync(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Synchronous search — FTS5 first, LIKE fallback."""
        results = self._fts_search(query, memory_type=None, top_k=limit)
        if results:
            return results
        words = query.lower().split()[:5]
        if not words:
            return []
        conditions = ["LOWER(content) LIKE ?" for _ in words]
        params = [f"%{w}%" for w in words]
        rows = self._get_conn().execute(
            f"SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
            f"FROM memories WHERE {' AND '.join(conditions)} LIMIT ?",
            params + [limit],
        ).fetchall()
        return [row_to_entry(row) for row in rows]

    def recent(self, limit: int = 20) -> List[MemoryEntry]:
        rows = self._get_conn().execute(
            "SELECT id, content, type, created_at, metadata_json, trust_score, retrieval_count, last_accessed "
            "FROM memories ORDER BY COALESCE(last_accessed, created_at) DESC LIMIT ?", (limit,),
        ).fetchall()
        return [row_to_entry(row) for row in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown or in a different
            # thread than the sqlite connection creator. Never surface cleanup
            # failures as unraisable warnings; explicit close() remains strict.
            pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()
