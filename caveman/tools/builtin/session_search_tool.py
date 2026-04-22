"""Session search tool — search across conversation history.

Enables the agent to search its own past conversations for context,
decisions, and patterns. Uses FTS5 for fast full-text search.
"""
from __future__ import annotations
from caveman.tools.registry import tool
from caveman.paths import CAVEMAN_HOME


@tool(
    name="session_search",
    description="Search across past conversation transcripts for context, decisions, or patterns",
    params={
        "query": {"type": "string", "description": "Search query (supports FTS5 syntax: AND, OR, NOT, phrases)"},
        "session_id": {"type": "string", "description": "Optional: limit search to a specific session"},
        "limit": {"type": "integer", "description": "Max results (default 10)"},
    },
    required=["query"],
)
async def session_search(query: str, session_id: str = "", limit: int = 10, **_kw) -> dict:
    from caveman.agent.session_db import SessionDB
    db = SessionDB(CAVEMAN_HOME / "sessions.db")
    try:
        results = db.search_transcripts(
            query, limit=limit,
            session_id=session_id or None,
        )
        if not results:
            return {"matches": 0, "results": [], "hint": "No matches. Try simpler terms or OR syntax."}
        formatted = []
        for r in results:
            formatted.append({
                "session": r["session_id"],
                "role": r["role"],
                "snippet": r.get("snippet", r["content"][:200]),
                "timestamp": r["ts"],
            })
        return {"matches": len(formatted), "results": formatted}
    finally:
        db.close()
