"""LLM streaming and response repair helpers for AgentLoop."""
from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Any

from caveman.agent.context import AgentContext
from caveman.agent.output_validator import final_text_looks_truncated
from caveman.agent.stream import StreamEvent
from caveman.events import EventType
from caveman.paths import DEFAULT_LLM_IDLE_TIMEOUT as _DEFAULT_LLM_IDLE_TIMEOUT
from caveman.providers.anthropic_adapter import CACHE_BOUNDARY

logger = logging.getLogger(__name__)
_PROVIDER_FINISH = {"message_stop", "done"}
_DONE = "_llm_done"


async def stream_llm_turn(loop: Any, context: AgentContext, system: str):
    """Yield stream events, then a private _llm_done event with turn data."""
    text_parts: list[str] = []
    tool_calls: list = []
    stop = "end_turn"
    messages = context.to_api_format()
    rules = loop._get_phase_rules()
    effective_system = system
    if rules:
        effective_system = (system or "") + CACHE_BOUNDARY + f"## Active Work State\n{rules}"
        logger.debug(
            "Phase rules injected (surface=%s, complexity=%s): %s",
            loop.surface, loop._conversation_state.complexity.value, rules[:80],
        )
    tool_defs = loop.tool_registry.get_schemas() if loop.tool_registry else []
    stream = loop.provider.safe_complete(
        messages=messages, system=effective_system, tools=tool_defs or None, stream=True,
    ).__aiter__()
    import caveman.agent.loop as loop_module
    idle_timeout = getattr(loop_module, "DEFAULT_LLM_IDLE_TIMEOUT", _DEFAULT_LLM_IDLE_TIMEOUT)
    while True:
        try:
            ev = await asyncio.wait_for(stream.__anext__(), timeout=idle_timeout)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            logger.warning("LLM idle timeout: %ds without token", idle_timeout)
            yield StreamEvent(
                type="error",
                data=f"LLM 无响应超时 ({idle_timeout}s)。任务未完成，请重试或切换模型。",
            )
            yield StreamEvent(type=_DONE, data=("", [], stop, False))
            return
        etype = ev.get("type")
        if etype == "delta":
            text_parts.append(ev["text"])
            await loop.bus.emit(EventType.LLM_STREAM_DELTA, {"text": ev["text"]})
            yield StreamEvent(type="token", data=ev["text"])
        elif etype == "thinking":
            yield StreamEvent(type="thinking", data=ev.get("text", ""))
        elif etype == "tool_call":
            tool_calls.append(ev)
            yield StreamEvent(type="tool_call", data=ev)
        elif etype in _PROVIDER_FINISH:
            stop = ev.get("stop_reason", "end_turn")
        elif etype == "error" and ev.get("action") == "abort":
            if loop._fallback_chain and loop._fallback_chain.has_fallbacks:
                new_provider = loop._fallback_chain.try_activate_next()
                if new_provider:
                    loop.provider = new_provider
                    yield StreamEvent(type="token", data="\n⚠️ 主模型失败，切换到备用模型...")
                    yield StreamEvent(type=_DONE, data=("", [], stop, True))
                    return
            yield StreamEvent(type="error", data=ev.get("error", "Unknown error"))
            yield StreamEvent(type=_DONE, data=("", [], stop, False))
            return
    from caveman.providers.message_sanitizer import strip_reasoning_tags
    yield StreamEvent(type=_DONE, data=(strip_reasoning_tags("".join(text_parts)), tool_calls, stop, False))


def request_continuation_if_needed(loop: Any, context: AgentContext, text: str, tool_calls: list, stop: str) -> bool:
    """Append a continuation prompt when final text was cut off."""
    if not getattr(loop, "allow_continuation_repair", True) or not text:
        return False
    if stop == "max_tokens":
        prompt = "你的回复被截断了，请继续从断点续写，不要重新开始。"
        reason = "Final response hit max_tokens"
    elif stop == "end_turn" and not tool_calls and final_text_looks_truncated(text):
        prompt = "你的最终回复看起来被截断了，请从断点续写并完整收尾，不要重新开始。"
        reason = "Final response looks truncated"
    else:
        return False
    logger.warning("%s; requesting continuation instead of finalizing", reason)
    context.add_message("assistant", text)
    context.add_message("user", prompt)
    return True


async def execute_tool_phase(loop: Any, context: AgentContext, tool_calls: list):
    """Execute tool calls and yield result events."""
    from caveman.agent.tools_exec import phase_tool_execution
    start = _time.monotonic()
    tool_names = [tc.get("name", "?") for tc in tool_calls]
    loop._last_activity_ts = _time.time()
    loop._last_activity_desc = f"Tools: {', '.join(tool_names)}"
    loop._current_tool = tool_names[0] if tool_names else ""
    loop._tool_call_count = await phase_tool_execution(
        context, tool_calls, loop.tool_registry, loop.permission_manager,
        loop.trajectory_recorder, loop.bus, loop._tool_call_count, loop._bg_skill_nudge,
    )
    loop.metrics.record_timing("tool_dispatch_duration", _time.monotonic() - start)
    await loop._offer_matching_skill(loop._nudge_task_ref)
    if loop._tool_call_count % 10 == 0:
        loop._safe_bg(loop._bg_skill_nudge())
    for tc in tool_calls:
        yield StreamEvent(type="tool_result", data={"name": tc.get("name", "?")})
