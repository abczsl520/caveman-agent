"""Memory tools — search, store, and retrieve agent memories."""
from __future__ import annotations

import logging

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="memory_search",
    description="Search memories by query",
    params={
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results", "default": 5},
    },
    required=["query"],
)
async def memory_search(query: str, limit: int = 5, _context: dict | None = None) -> list[dict]:
    """Search memories using hybrid recall."""
    mgr = (_context or {}).get("memory_manager")
    if not mgr:
        return [{"error": "memory_manager not available"}]
    entries = await mgr.recall(query, top_k=limit)
    return [
        {
            "id": e.id,
            "content": e.content,
            "type": e.memory_type.value,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ]


@tool(
    name="memory_store",
    description="Store a new memory",
    params={
        "content": {"type": "string", "description": "Memory content"},
        "memory_type": {"type": "string", "description": "Type: episodic, semantic, procedural, working", "default": "semantic"},
    },
    required=["content"],
)
async def memory_store(content: str, memory_type: str = "semantic", _context: dict | None = None) -> dict:
    """Store a memory and return its ID."""
    from caveman.memory.types import MemoryType
    mgr = (_context or {}).get("memory_manager")
    if not mgr:
        return {"error": "memory_manager not available"}
    try:
        mt = MemoryType(memory_type)
    except ValueError:
        return {"error": f"Invalid memory_type: {memory_type}. Use: episodic, semantic, procedural, working"}
    mid = await mgr.store(content, mt, trusted=True)
    # User-initiated stores deserve higher initial trust (0.7 vs default 0.5)
    backend = getattr(mgr, '_backend', None)
    if backend and mid:
        try:
            conn = backend._get_conn()
            conn.execute(
                "UPDATE memories SET trust_score = 0.7 WHERE id = ?", (mid,)
            )
        except Exception as e:
            logger.warning("Trust score update failed for %s: %s", mid, e)
    return {"ok": True, "memory_id": mid}


@tool(
    name="memory_recent",
    description="Get recent memories",
    params={
        "limit": {"type": "integer", "description": "Max results", "default": 10},
    },
    required=[],
)
async def memory_recent(limit: int = 10, _context: dict | None = None) -> list[dict]:
    """Return most recent memories."""
    mgr = (_context or {}).get("memory_manager")
    if not mgr:
        return [{"error": "memory_manager not available"}]
    entries = mgr.recent(limit=limit)
    return [
        {
            "id": e.id,
            "content": e.content,
            "type": e.memory_type.value,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ]
