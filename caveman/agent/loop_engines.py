"""Loop engine helpers — post-task engine orchestration extracted from loop.py."""
from __future__ import annotations
import logging
from caveman.events import EventType
from caveman.agent.display import show_error

__all__ = [
    "prepare_multi_turn",
    "post_task_engines",
    "record_turn_metrics",
    "check_termination",
    "update_shield",
    "run_preemptive_compaction",
    "build_user_content",
]


logger = logging.getLogger(__name__)


def _download_image_as_data_uri(url: str, content_type: str = "image/jpeg") -> str | None:
    """Download image and return as data: URI. Returns None on failure."""
    try:
        import urllib.request, base64
        req = urllib.request.Request(url, headers={"User-Agent": "caveman/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", content_type).split(";")[0].strip()
            b64 = base64.b64encode(data).decode()
            return f"data:{ct};base64,{b64}"
    except Exception as e:
        logger.warning("Image download failed (%s): %s", url[:80], e)
        return None


def build_user_content(task: str, attachments: list[dict[str, str]] | None = None) -> str | list:
    """Build user message content with optional vision image blocks.

    Returns plain string if no image attachments, or a list of content blocks
    (text + image_url) for multimodal messages.
    Downloads images as base64 data URIs for reliability (Discord CDN URLs expire).
    """
    if not attachments:
        return task
    image_atts = [a for a in attachments if a.get("content_type", "").startswith("image/")]
    if not image_atts:
        return task
    blocks: list[dict] = [{"type": "text", "text": task}]
    for att in image_atts:
        url = att["url"]
        ct = att.get("content_type", "image/jpeg")
        # Pre-download to base64 for reliability (CDN URLs may expire)
        data_uri = _download_image_as_data_uri(url, ct)
        if data_uri:
            blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
        else:
            # Fallback to direct URL if download fails
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks



async def prepare_multi_turn(loop, task: str, recalled_ids: list[str], attachments: list[dict[str, str]] | None = None) -> tuple:
    """Reuse context, re-recall memories, rebuild prompt if needed."""
    from caveman.agent.response_style import get_format_reminder
    from caveman.agent.prompt import build_system_prompt
    context = loop._persistent_context
    user_content = build_user_content(task, attachments)
    context.add_message("user", user_content)
    reminder = get_format_reminder(loop.surface)
    if reminder:
        context.add_message("system", reminder, ephemeral=True)
    await loop.trajectory_recorder.record_turn("human", task)
    if not loop._system_prompt_cache:
        loop._system_prompt_cache = build_system_prompt(
            tool_schemas=loop.tool_registry.get_schemas(), surface=loop.surface,
            conversation_state=loop._conversation_state)
        logger.info("Rebuilt system prompt for restored session (surface=%s)", loop.surface)
    matched_skills = loop.skill_manager.match(task)
    try:
        new_memories = await loop.memory_manager.recall(task, top_k=3)
        if new_memories:
            recalled_ids.extend(m.id for m in new_memories)
            await loop.bus.emit(EventType.MEMORY_RECALL, {
                "query": task, "results": len(new_memories),
                "recalled_ids": [m.id for m in new_memories], "recall_hit": True,
            }, source="memory")
    except Exception as e:
        logger.warning("Multi-turn recall failed: %s", e)
    return context, loop._system_prompt_cache, matched_skills


async def post_task_engines(loop, context, task, result, matched_skills) -> None:
    """Run post-task engines: shield, reflect, nudge, skill save, lint."""
    await update_shield(loop, context, task)
    if loop.engine_flags.is_enabled("reflect"):
        try:
            await loop._reflect.reflect(task, loop.trajectory_recorder.to_sharegpt(), result)
        except Exception as e:
            logger.debug("Reflect failed: %s", e)
    loop._safe_bg(loop._end_nudge(task))
    loop._safe_bg(loop._check_save_skill(task))
    if loop._lint and loop.engine_flags.is_enabled("lint"):
        loop._safe_bg(loop._run_lint())


def record_turn_metrics(loop, turn_start, recalled_ids, matched_skills, result) -> None:
    """Record turn-level metrics."""
    import time as _t
    loop.metrics.record_timing("total_turn_duration", _t.monotonic() - turn_start)
    loop.metrics.increment("turns_completed")
    for cond, key in [(recalled_ids, "recall_hits"), (matched_skills, "skill_match_hits")]:
        if cond:
            loop.metrics.increment(key)
    loop.metrics.increment("recall_attempts")
    loop.metrics.increment("skill_match_attempts")
    from caveman.utils import detect_success
    if detect_success(result):
        loop.metrics.increment("task_successes")


async def check_termination(stop: str, tool_calls: list, task: str) -> bool:
    """Check if the loop should terminate after this iteration."""
    if tool_calls:
        return False
    if stop == "end_turn":
        return True
    if stop == "max_tokens":
        show_error("Max tokens reached")
    elif stop != "tool_use":
        logger.warning("Unknown stop_reason '%s' — terminating", stop)
    else:
        logger.warning("stop_reason='tool_use' but no tool_calls — terminating")
    return True


async def update_shield(loop, context, task: str) -> None:
    """Update compaction shield with current context."""
    if not loop.engine_flags.is_enabled("shield"):
        return
    try:
        msgs = [m if isinstance(m, dict) else {"role": getattr(m, "role", "unknown"),
                 "content": getattr(m, "content", str(m))} for m in context.messages]
        await loop._shield.update(msgs, task)
        await loop._shield.save()
        await loop.bus.emit(EventType.SHIELD_UPDATE, {
            "session_id": loop._shield.essence.session_id,
            "turn_count": loop._shield.essence.turn_count,
        }, source="shield")
    except Exception as e:
        logger.warning("Shield update failed: %s", e)


async def run_preemptive_compaction(context, compressor, shield, bus, metrics) -> bool:
    """Run 3-tier preemptive compaction + fallback threshold compression.

    Returns updated context (always handles compression if needed).
    """
    import time as _time
    from caveman.compression.preemptive import (
        should_preemptively_compact, CompactionRoute,
        apply_tool_result_truncation, apply_image_pruning,
    )
    from caveman.agent.phases import phase_compress

    async def _do_llm_compress():
        nonlocal context
        if shield:
            try:
                await shield.update(context.to_api_format())
            except Exception as e:
                logger.warning("Shield pre-compression update failed: %s", e)
        _comp_start = _time.monotonic()
        context, _ = await phase_compress(context, compressor, bus)
        if metrics:
            metrics.record_timing("compression_duration", _time.monotonic() - _comp_start)

    precheck = should_preemptively_compact(context)
    if precheck.route == CompactionRoute.FITS:
        # Fallback: original threshold-based compression
        if context.should_compress():
            await _do_llm_compress()
        return context

    logger.info(
        "Preemptive compaction: route=%s overflow=%d truncatable=%d images=%d",
        precheck.route.value, precheck.overflow_tokens,
        precheck.truncatable_chars, precheck.prunable_images,
    )

    # Tier 1: Truncate oversized tool results (free)
    if precheck.route in (
        CompactionRoute.TRUNCATE_TOOL_RESULTS,
        CompactionRoute.TRUNCATE_THEN_COMPRESS,
    ):
        n = apply_tool_result_truncation(context)
        if n:
            logger.info("Preemptive: truncated %d tool result(s)", n)

    # Tier 2: Prune old images (free)
    if precheck.route == CompactionRoute.PRUNE_IMAGES:
        n = apply_image_pruning(context)
        if n:
            logger.info("Preemptive: pruned %d image(s)", n)

    # Tier 3: Full LLM compression (expensive, last resort)
    if precheck.route in (
        CompactionRoute.COMPRESS,
        CompactionRoute.TRUNCATE_THEN_COMPRESS,
    ):
        await _do_llm_compress()

    return context
