"""Caveman MCP Server — expose the 6-engine flywheel as MCP tools.

Any MCP-compatible agent (Claude Code, Codex, Gemini CLI) can use Caveman's
memory, recall, shield, and reflect capabilities without switching tools.

Usage:
    caveman mcp serve              # stdio transport (for Claude Code)
    caveman mcp serve --http 8765  # HTTP transport (for remote agents)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from caveman.paths import MEMORY_DIR, SESSIONS_DIR, SKILLS_DIR
from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryType
from caveman.engines.recall import RecallEngine
from caveman.engines.shield import CompactionShield
from caveman.engines.reflect import ReflectEngine, Reflection
from caveman.skills.manager import SkillManager
from caveman.wiki import WikiStore
from caveman.wiki.compiler import WikiCompiler
from caveman.engines.flags import EngineFlags, ENGINES

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP(
    "Caveman",
    json_response=True,
)

# --- Memory Tools ---

@mcp.tool()
async def memory_store(
    content: str,
    tags: str = "",
    source: str = "mcp",
    importance: float = 0.5,
) -> dict[str, Any]:
    """Store a memory in Caveman's memory system.

    Args:
        content: The memory content to store.
        tags: Comma-separated tags for categorization.
        source: Where this memory came from (e.g., 'claude-code', 'user').
        importance: How important this memory is (0.0-1.0).
    """

    mgr = MemoryManager()
    await mgr.load()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Determine memory type from importance
    if importance >= 0.8:
        mem_type = MemoryType.SEMANTIC
    elif importance >= 0.5:
        mem_type = MemoryType.EPISODIC
    else:
        mem_type = MemoryType.WORKING

    memory_id = await mgr.store(
        content=content,
        memory_type=mem_type,
        metadata={"tags": tag_list, "source": source, "importance": importance},
    )

    return {
        "stored": True,
        "memory_id": memory_id,
        "tags": tag_list,
    }

@mcp.tool()
async def memory_search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.3,
) -> dict[str, Any]:
    """Search Caveman's memory for relevant information.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
        min_score: Minimum relevance score (0.0-1.0).
    """

    mgr = MemoryManager()
    await mgr.load()
    scored_results = await mgr.recall_scored(query, top_k=top_k)

    filtered = [
        {"content": r.content, "type": r.memory_type.value, "id": r.id, "score": round(score, 4)}
        for score, r in scored_results
        if score >= min_score
    ][:top_k]

    return {
        "query": query,
        "results": filtered,
        "count": len(filtered),
    }

@mcp.tool()
async def memory_recall(
    task: str = "",
    max_essences: int = 3,
) -> dict[str, Any]:
    """Recall context from previous Caveman sessions.

    Loads session essences (decisions, progress, todos) and relevant memories.
    Use at the start of a new task to get continuity.

    Args:
        task: Current task description (helps prioritize relevant context).
        max_essences: Maximum number of previous sessions to recall.
    """

    recall = RecallEngine(
        sessions_dir=SESSIONS_DIR,
        max_essences=max_essences,
    )
    result = await recall.restore_structured(task)

    return {
        "has_context": result.has_context,
        "essences_loaded": result.essences_loaded,
        "memories_loaded": result.memories_loaded,
        "context": result.as_prompt_text() if result.has_context else "",
    }

# --- Shield Tools ---

@mcp.tool()
async def shield_save(
    session_id: str,
    task: str = "",
    decisions: str = "",
    progress: str = "",
    todos: str = "",
) -> dict[str, Any]:
    """Save current session state as a Shield essence.

    Call this at the end of a work session to preserve state for next time.

    Args:
        session_id: Unique identifier for this session.
        task: What task was being worked on.
        decisions: Key decisions made (newline-separated).
        progress: What was accomplished (newline-separated).
        todos: What still needs to be done (newline-separated).
    """

    shield = CompactionShield(
        session_id=session_id,
        store_dir=SESSIONS_DIR,
    )

    # Set essence fields directly
    if decisions:
        shield._essence.decisions = [d.strip() for d in decisions.split("\n") if d.strip()]
    if progress:
        shield._essence.progress = [p.strip() for p in progress.split("\n") if p.strip()]
    if todos:
        shield._essence.open_todos = [t.strip() for t in todos.split("\n") if t.strip()]
    if task:
        shield._essence.task = task

    path = await shield.save()

    return {
        "saved": True,
        "session_id": session_id,
        "path": str(path),
        "decisions": len(shield._essence.decisions),
        "progress": len(shield._essence.progress),
        "todos": len(shield._essence.open_todos),
    }

@mcp.tool()
async def shield_load(session_id: str = "") -> dict[str, Any]:
    """Load a session essence by ID, or the most recent one.

    Args:
        session_id: Session ID to load. Empty = most recent.
    """

    if session_id:
        essence = await CompactionShield.load(session_id, store_dir=SESSIONS_DIR)
        if not essence:
            return {"error": f"Session not found: {session_id}"}
    else:
        essence = await CompactionShield.load_latest(store_dir=SESSIONS_DIR)
        if not essence:
            return {"error": "No sessions found"}
        session_id = essence.session_id

    return {
        "session_id": session_id,
        "task": essence.task,
        "decisions": essence.decisions,
        "progress": essence.progress,
        "todos": essence.open_todos,
        "turn_count": essence.turn_count,
        "updated_at": essence.updated_at.isoformat() if hasattr(essence.updated_at, 'isoformat') else str(essence.updated_at),
    }

# --- Reflect Tool ---

@mcp.tool()
async def reflect(
    task: str,
    outcome: str,
    what_worked: str = "",
    what_failed: str = "",
    lessons: str = "",
) -> dict[str, Any]:
    """Reflect on a completed task and extract learnings.

    Call after finishing a task to build Caveman's skill library.

    Args:
        task: What task was completed.
        outcome: 'success', 'partial', or 'failure'.
        what_worked: What approaches worked well (newline-separated).
        what_failed: What didn't work (newline-separated).
        lessons: Key takeaways for next time (newline-separated).
    """

    reflection = Reflection(
        task=task,
        outcome=outcome,
        effective_patterns=[w.strip() for w in what_worked.split("\n") if w.strip()],
        anti_patterns=[f.strip() for f in what_failed.split("\n") if f.strip()],
        lessons=[l.strip() for l in lessons.split("\n") if l.strip()],
        confidence=0.8 if outcome == "success" else 0.4,
    )
    # Persist reflection via ReflectEngine
    engine = ReflectEngine(skill_manager=SkillManager(skills_dir=SKILLS_DIR))
    engine._reflections.append(reflection)
    if reflection.skill_updates:
        engine._apply_skill_updates(reflection)
    try:
        mgr = MemoryManager()
        await mgr.load()
        summary = "; ".join(reflection.lessons) if reflection.lessons else outcome
        await mgr.store(
            content=f"Reflection on '{task}' ({outcome}): {summary}",
            memory_type=MemoryType.EPISODIC,
            metadata={"source": "mcp-reflect", "task": task, "outcome": outcome,
                       "patterns": reflection.effective_patterns,
                       "anti_patterns": reflection.anti_patterns},
        )
    except Exception as e:
        logger.warning("Failed to persist reflection to memory: %s", e)

    return {
        "reflected": True,
        "task": task,
        "outcome": outcome,
        "patterns_recorded": len(reflection.effective_patterns),
        "anti_patterns_recorded": len(reflection.anti_patterns),
        "lessons_recorded": len(reflection.lessons),
    }

# --- Skill Tools ---

@mcp.tool()
async def skill_list() -> dict[str, Any]:
    """List all available Caveman skills."""

    mgr = SkillManager(skills_dir=SKILLS_DIR)
    skills = mgr.list_all()

    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "trigger": s.trigger.value if hasattr(s.trigger, "value") else str(s.trigger),
                "success_count": s.success_count,
                "version": s.version,
            }
            for s in skills
        ],
        "count": len(skills),
    }

@mcp.tool()
async def skill_get(name: str) -> dict[str, Any]:
    """Get a specific skill's content.

    Args:
        name: Skill name to retrieve.
    """

    mgr = SkillManager(skills_dir=SKILLS_DIR)
    skill = mgr.get(name)

    if not skill:
        return {"error": f"Skill not found: {name}"}

    return {
        "name": skill.name,
        "description": skill.description,
        "content": skill.content,
        "version": skill.version,
        "success_count": skill.success_count,
        "fail_count": skill.fail_count,
    }

# --- Wiki Tools ---

@mcp.tool()
async def wiki_compile(max_tokens: int = 4000) -> dict[str, Any]:
    """Compile the wiki and return the compiled context.

    Runs the full compilation loop (promote + expire + enforce limits)
    and returns the compiled context suitable for system prompt injection.

    Args:
        max_tokens: Maximum tokens for the compiled context.
    """

    compiler = WikiCompiler(WikiStore())
    result = compiler.compile()
    context = compiler.get_compiled_context(max_tokens=max_tokens)
    stats = compiler.store.stats()

    return {
        "compiled": True,
        "entries_processed": result.entries_processed,
        "entries_promoted": result.entries_promoted,
        "entries_expired": result.entries_expired,
        "duration_ms": result.duration_ms,
        "stats": stats,
        "context": context,
    }

@mcp.tool()
async def wiki_ingest(
    content: str,
    title: str = "",
    tags: str = "",
    source: str = "mcp",
) -> dict[str, Any]:
    """Ingest a knowledge observation into the wiki's working tier.

    Args:
        content: The knowledge to store.
        title: Optional title (auto-generated from content if empty).
        tags: Comma-separated tags.
        source: Where this knowledge came from.
    """

    compiler = WikiCompiler(WikiStore())
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    entry = compiler.ingest(content, title=title, tags=tag_list, source=source)

    return {
        "ingested": True,
        "id": entry.id,
        "tier": entry.tier,
        "confidence": entry.confidence,
        "reinforced": entry.reinforcement_count > 0,
    }


@mcp.resource("caveman://status")
async def get_status() -> str:
    """Get Caveman system status."""
    flags = EngineFlags()
    engines = {name: flags.is_enabled(name) for name in ENGINES}
    sessions = list(SESSIONS_DIR.glob("*.yaml")) if SESSIONS_DIR.exists() else []
    return json.dumps({
        "version": "0.1.0", "engines": engines,
        "sessions": len(sessions),
        "memory_dir": str(MEMORY_DIR), "skills_dir": str(SKILLS_DIR),
    }, indent=2)


def run_stdio() -> None:
    """Run MCP server with stdio transport."""
    mcp.run(transport="stdio")


def run_http(port: int = 8765) -> None:
    """Run MCP server with HTTP transport."""
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port)

if __name__ == "__main__":
    run_stdio()
