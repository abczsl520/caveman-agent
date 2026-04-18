"""Agent phases — decomposed pipeline stages for the agent loop.

Extracted from loop.py to keep it under 400 lines.
Each phase is a standalone async function that takes explicit dependencies.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any

from caveman.agent.context import AgentContext
from caveman.agent.prompt import build_system_prompt
from caveman.compression.pipeline import CompressionPipeline
from caveman.events import EventBus, EventType
from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryType
from caveman.skills.manager import SkillManager
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.engines.flags import EngineFlags
from caveman.engines.recall import RecallEngine

logger = logging.getLogger(__name__)

# Pre-compiled patterns for episode filtering (Fix #16)
import re as _re
_LOW_VALUE_PATTERNS = (
    _re.compile(r"^(?:what is|explain|define|convert|write a (?:python|javascript|bash))\s", _re.I),
    _re.compile(r"^(?:list|name|describe)\s+\d+\s+(?:common|popular)", _re.I),
    _re.compile(r"^(?:create a file|read a file|hello|hi|test)\s*$", _re.I),
)


async def phase_prepare(
    task: str,
    system_prompt: str | None,
    provider,
    skill_manager: SkillManager,
    memory_manager: MemoryManager,
    trajectory_recorder: TrajectoryRecorder,
    recall_engine: RecallEngine,
    engine_flags: EngineFlags,
    bus: EventBus,
    tool_registry,
    surface: str = "cli",
) -> tuple[AgentContext, str, list]:
    """Load context, skills, memories. Return (context, system_prompt, matched_skills)."""
    # Cap task length to prevent FTS5/recall waste on extremely long inputs
    if len(task) > 2000:
        task = task[:2000]
    context = AgentContext(max_tokens=provider.context_length)

    skill_manager.load_all()
    matched_skills = skill_manager.match(task)
    if matched_skills:
        await bus.emit(EventType.SKILL_MATCH, {
            "skills": [s.name for s in matched_skills],
        }, source="skill")

    memories = await memory_manager.recall(task, top_k=5)
    await bus.emit(EventType.MEMORY_RECALL, {
        "query": task, "results": len(memories),
        # Track recalled memory IDs for confidence feedback in phase_finalize
        "recalled_ids": [m.id for m in memories],
        # Flywheel metrics
        "recall_hit": len(memories) > 0,
    }, source="memory")

    recall_context = ""
    if engine_flags.is_enabled("recall"):
        try:
            recall_context = await recall_engine.restore(task)
            if recall_context:
                await bus.emit(EventType.MEMORY_RECALL, {
                    "source": "recall_engine", "has_context": True,
                }, source="recall")
        except Exception as e:
            logger.warning("Recall engine failed: %s", e)

    # Build wiki context if available
    wiki_context = ""
    try:
        from caveman.wiki.compiler import WikiCompiler
        wiki = WikiCompiler()
        wiki_ctx = wiki.get_compiled_context(max_tokens=2000)
        if wiki_ctx:
            wiki_context = wiki_ctx
    except Exception as e:
        logger.debug("Wiki context unavailable: %s", e)

    system = system_prompt or build_system_prompt(
        memories=[{
            "content": m.content,
            "type": m.memory_type.value,
            "age_days": (datetime.now() - m.created_at).days if hasattr(m, 'created_at') else 0,
        } for m in memories] if memories else None,
        skills=matched_skills or None,
        tool_schemas=tool_registry.get_schemas(),
        recall_context=recall_context,
        wiki_context=wiki_context,
        surface=surface,
    )

    # Append format reminder for recency bias (LLM pays attention to last instruction)
    from caveman.agent.response_style import get_format_reminder
    reminder = get_format_reminder(surface)
    task_with_reminder = f"{task}\n{reminder}" if reminder else task

    context.add_message("user", task_with_reminder)
    await trajectory_recorder.record_turn("human", task)

    return context, system, matched_skills


async def phase_compress(
    context: AgentContext,
    compressor: CompressionPipeline,
    bus: EventBus,
) -> tuple[AgentContext, Any]:
    """Compress context if utilization is high. Best-effort — falls back on error."""
    try:
        usage = context.total_tokens / context.max_tokens if context.max_tokens else 0

        # Convert Message objects to dicts for the compressor.
        # Preserve opaque message metadata (e.g. tool_calls/tool_call_id) if present
        # so compression does not corrupt API message structure.
        msg_dicts = []
        for m in context.messages:
            d = {"role": m.role, "content": m.content, "tokens": m.tokens}
            for key, value in getattr(m, "__dict__", {}).items():
                if key not in d:
                    d[key] = value
            msg_dicts.append(d)
        compressed_dicts, stats = await compressor.compress(msg_dicts, usage)

        await bus.emit(EventType.CONTEXT_COMPRESS, {
            "usage_before": usage,
            "messages_before": len(context.messages),
            "messages_after": len(compressed_dicts),
            "layer": getattr(stats, 'layer_applied', 'unknown'),
        }, source="compress")

        # Convert back to Message objects
        from caveman.agent.context import Message
        context.messages = [
            Message(
                role=d.get("role", "user"),
                content=d.get("content", ""),
                tokens=d.get("tokens", 0),
            )
            for d in compressed_dicts
        ]
        return context, stats
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Compression failed, keeping original context: %s", e)
        return context, None


async def phase_llm_call(
    context: AgentContext,
    system: str,
    provider,
    tool_registry,
    bus: EventBus,
) -> tuple[str, list, str]:
    """Call LLM, stream response. Return (text, tool_calls, stop_reason)."""
    tool_schemas = tool_registry.get_schemas()
    await bus.emit(EventType.LLM_REQUEST, {
        "messages": len(context.messages), "tools": len(tool_schemas),
    }, source="llm")

    text_buf: list[str] = []
    tool_calls: list[dict] = []
    stop = "end_turn"

    try:
        async for ev in provider.safe_complete(
            messages=context.to_api_format(),
            tools=tool_schemas or None,
            stream=True, system=system,
        ):
            if ev["type"] == "delta":
                text_buf.append(ev["text"])
                # Only print to stdout in CLI mode (not gateway)
                if sys.stdout.isatty():
                    print(ev["text"], end="", flush=True)
            elif ev["type"] == "tool_call":
                tool_calls.append(ev)
            elif ev["type"] == "done":
                stop = ev.get("stop_reason", "end_turn")
            elif ev["type"] == "error":
                action = ev.get("action", "abort")
                error_msg = ev.get("error", "Unknown provider error")
                await bus.emit(EventType.LLM_ERROR, {
                    "error": error_msg, "action": action,
                }, source="llm")
                if action == "abort":
                    raise RuntimeError(f"Provider error (abort): {error_msg}")
                elif action == "compress":
                    raise RuntimeError(f"Context too long: {error_msg}")
    except Exception as e:
        await bus.emit(EventType.LLM_ERROR, {"error": str(e)}, source="llm")
        raise

    text = "".join(text_buf)
    if text and sys.stdout.isatty():
        print()

    await bus.emit(EventType.LLM_RESPONSE, {
        "text_len": len(text), "tool_calls": len(tool_calls), "stop": stop,
    }, source="llm")

    return text, tool_calls, stop


def record_assistant_turn(context: AgentContext, text: str, tool_calls: list) -> None:
    """Add assistant message to context."""
    assistant_content: list[dict[str, Any]] = []
    if text:
        assistant_content.append({"type": "text", "text": text})
    for tc in tool_calls:
        assistant_content.append({
            "type": "tool_use", "id": tc["id"],
            "name": tc["name"], "input": tc["input"],
        })
    if assistant_content:
        context.add_message("assistant", assistant_content)


async def phase_finalize(
    task: str,
    final: str,
    matched_skills: list,
    memory_manager: MemoryManager,
    skill_manager: SkillManager,
    trajectory_recorder: TrajectoryRecorder,
    bus: EventBus,
    llm_fn=None,
    recalled_ids: list[str] | None = None,
) -> str:
    """Store episode memory, record outcomes, save trajectory.

    Closes the confidence feedback loop: memories recalled in phase_prepare
    get their trust scores adjusted based on task outcome.
    """
    from caveman.utils import detect_success
    success = detect_success(final)

    # Store structured episodic memory — but only if the task has project value
    episode_content = _build_episode_content(task, final, success)
    if episode_content:  # None means task was filtered as low-value
        await memory_manager.store(episode_content, MemoryType.EPISODIC)
    await bus.emit(EventType.MEMORY_STORE, {
        "type": "episodic", "task": task[:100], "success": success,
    }, source="memory")

    # Confidence feedback loop: adjust trust for recalled memories
    # This is the core flywheel — memories that help succeed get boosted,
    # memories that don't help get demoted. Over time, the best memories
    # surface first. Uses SQLite trust_score column directly.
    #
    # Negative feedback: if a recalled memory wasn't referenced in the
    # final response, it was probably irrelevant — light demotion (-0.02).
    if recalled_ids:
        backend = getattr(memory_manager, '_backend', None)
        if backend and hasattr(backend, 'mark_helpful'):
            final_lower = final.lower() if final else ""
            for mid in recalled_ids:
                try:
                    mem = await memory_manager.get_by_id(mid) if hasattr(memory_manager, 'get_by_id') else None
                    mem_text = (mem.content[:50].lower() if mem else "").split()
                    # Check if any significant words from memory appear in response
                    was_used = False
                    if final_lower and mem_text:
                        was_used = any(w in final_lower for w in mem_text if len(w) > 4)
                    elif success:
                        was_used = True  # Fallback: can't verify, assume helpful if success
                    await backend.mark_helpful(mid, helpful=was_used)
                except Exception as e:
                    logger.debug("Confidence feedback failed for %s: %s", mid, e)

    for skill in matched_skills:
        skill_manager.record_outcome(skill.name, success)
        await bus.emit(EventType.SKILL_OUTCOME, {
            "skill": skill.name, "success": success,
        }, source="skill")

    try:
        traj = trajectory_recorder.to_sharegpt()
        # Only create skills from substantial trajectories with tool usage
        tool_turns = sum(1 for t in traj if t.get("from") == "function_call")
        if len(traj) >= 6 and tool_turns >= 2:
            await skill_manager.auto_create(traj, task=task, llm_fn=llm_fn)
    except Exception as e:
        logger.debug("Skill auto-create failed: %s", e)

    await trajectory_recorder.save()
    await bus.emit(EventType.TRAJECTORY_SAVE, {
        "turns": len(trajectory_recorder.to_sharegpt()),
    }, source="trajectory")

    return final



def _build_episode_content(task: str, result: str, success: bool) -> str | None:
    """Build high-quality episodic memory from task result.

    PRD §6 Iron Law #1: Write quality >> Retrieve sophistication.
    Returns None for low-value tasks (generic QA, trivial operations).
    """
    import re
    # Filter out generic QA that LLM already knows — zero project value
    task_lower = task.strip()
    for pat in _LOW_VALUE_PATTERNS:
        if pat.search(task_lower):
            return None

    # Skip trivially short tasks
    if len(task.strip()) < 10:
        return None

    outcome = "✅ succeeded" if success else "❌ failed"
    summary = result.strip()[:300]
    for end in ('.', '。', '!', '\n'):
        idx = summary.rfind(end)
        if idx > 50:
            summary = summary[:idx + 1]
            break

    return f"[{outcome}] {task[:150]}\nSummary: {summary}"
