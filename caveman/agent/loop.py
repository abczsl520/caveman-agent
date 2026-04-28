"""Core agent loop v3 — thin orchestrator over decomposed phases."""
from __future__ import annotations
import logging, time as _time
from collections.abc import AsyncIterator
from typing import Any
logger = logging.getLogger(__name__)

from caveman.agent.loop_state import LoopState
from caveman.agent.stream import StreamEvent, is_result_event_type
from caveman.agent.context import AgentContext
from caveman.agent.bg_tasks import BackgroundTaskMixin
from caveman.agent.conversation_lifecycle import get_phase_rules
from caveman.providers.llm import LLMProvider
from caveman.compression.pipeline import CompressionPipeline
from caveman.tools.registry import ToolRegistry
from caveman.memory.manager import MemoryManager
from caveman.skills.manager import SkillManager
from caveman.skills.executor import SkillExecutor
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.security.permissions import PermissionManager
from caveman.events import EventBus, EventType, create_default_bus
from caveman.engines.flags import EngineFlags
from caveman.engines.manager import EngineManager
from caveman.agent.output_validator import CLOSING_LINE
from caveman.agent.display import show_error
from caveman.agent.metrics import AgentMetrics
from caveman.agent.iteration_budget import IterationBudget
from caveman.agent.phases import (
    phase_prepare, record_assistant_turn, phase_finalize,
)
from caveman.agent.llm_turn import (
    execute_tool_phase, request_continuation_if_needed, stream_llm_turn,
)
from caveman.agent.loop_init import (
    build_fallback_chain, ensure_memory_manager, ensure_provider, set_default_permissions,
)
from caveman.paths import DEFAULT_LLM_IDLE_TIMEOUT
class AgentLoop(BackgroundTaskMixin):
    """Core agent execution loop — manages tool calls, LLM turns, and phase transitions."""
    def __init__(
        self, model: str | None = None, max_iterations: int | None = None,
        provider: LLMProvider | None = None, tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None, skill_manager: SkillManager | None = None,
        trajectory_recorder: TrajectoryRecorder | None = None,
        permission_manager: PermissionManager | None = None,
        event_bus: EventBus | None = None, engine_flags: EngineFlags | None = None,
        llm_fn=None, lint_engine=None,
        shield=None, recall_engine=None, nudge_engine=None, reflect_engine=None,
        surface: str = "cli", allow_continuation_repair: bool = True,
    ):
        from caveman.paths import DEFAULT_MODEL, DEFAULT_MAX_ITERATIONS
        self.model = model or DEFAULT_MODEL
        self.max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        self.budget = IterationBudget(self.max_iterations)
        self._fallback_chain = build_fallback_chain()
        self._state = LoopState(surface=surface)
        self.provider = ensure_provider(provider, self.model)
        self.tool_registry = tool_registry or ToolRegistry()
        if tool_registry is None:
            self.tool_registry._register_builtins()
        self.memory_manager = ensure_memory_manager(memory_manager)
        self.skill_manager = skill_manager or SkillManager()
        self.trajectory_recorder = trajectory_recorder or TrajectoryRecorder()
        from caveman.mcp.manager import MCPManager
        from caveman.agent.checkpoint import CheckpointManager
        from caveman.gateway.router import GatewayRouter
        self.mcp_manager = MCPManager()
        self.checkpoint_manager = CheckpointManager()
        self.gateway_router = GatewayRouter()
        self.metrics = AgentMetrics()
        for k, v in [("memory_manager", self.memory_manager),
                      ("trajectory_recorder", self.trajectory_recorder),
                      ("mcp_manager", self.mcp_manager),
                      ("checkpoint_manager", self.checkpoint_manager),
                      ("gateway_router", self.gateway_router),
                      ("metrics", self.metrics)]:
            self.tool_registry.set_context(k, v)
        self.permission_manager = set_default_permissions(permission_manager)
        if event_bus is None:
            event_bus, self._metrics = create_default_bus()
        else:
            self._metrics = None
        self.bus = event_bus
        for _c in [self.trajectory_recorder, self.permission_manager]:
            if hasattr(_c, '_bus') and _c._bus is None:
                _c._bus = self.bus
        # Wire bus into memory_manager for MEMORY_STORE events (flywheel Chain 4)
        if hasattr(self.memory_manager, '_bus') and self.memory_manager._bus is None:
            self.memory_manager._bus = self.bus
        self.engine_flags = engine_flags or EngineFlags()
        self._llm_fn = llm_fn
        _em = EngineManager(
            flags=self.engine_flags, memory_manager=self.memory_manager,
            skill_manager=self.skill_manager, llm_fn=llm_fn, bus=self.bus,
        )
        _es = _em.create_all()
        self._shield = shield or _es.shield
        self._recall = recall_engine or _es.recall
        self._nudge = nudge_engine or _es.nudge
        self._reflect = reflect_engine or _es.reflect
        self._lint = lint_engine or _es.lint
        self._ripple = _es.ripple  # was None — now wired!
        self._outcome = _es.outcome
        self._skill_executor = SkillExecutor(tool_dispatch_fn=self._dispatch_skill_tool)
        self._nudge_task_ref = ""
        self._bg_tasks: set[Any] = set()
        self._last_activity_ts = 0.0
        self._last_activity_desc = ""
        self._current_tool = ""
        self._persistent_context: AgentContext | None = None
        self._system_prompt_cache: str | None = None
        self.allow_continuation_repair = allow_continuation_repair
        self._wire_flywheel()
    def _wire_flywheel(self):
        from caveman.engines.event_chain import wire_inner_flywheel
        from caveman.engines.manager import EngineSet
        es = EngineSet(shield=self._shield, nudge=self._nudge, reflect=self._reflect,
                       ripple=self._ripple, lint=self._lint, recall=self._recall,
                       outcome=self._outcome)
        self._flywheel_handlers = wire_inner_flywheel(
            self.bus, es, get_turns=lambda: self.trajectory_recorder.to_sharegpt(),
            get_task=lambda: self._nudge_task_ref, memory_manager=self.memory_manager)
    def set_lint(self, engine) -> None: self._lint = engine; engine and setattr(engine, "_bus", self.bus)
    def set_ripple(self, engine) -> None: self._ripple = engine; engine and setattr(engine, "_bus", self.bus)

    # --- State proxy: delegate LoopState fields transparently ---
    _STATE_FIELDS = {"surface", "_turn_number", "_turn_count", "_tool_call_count", "_iteration_count"}
    _STATE_MAP = {"surface": "surface", "_turn_number": "turn_number", "_turn_count": "turn_count",
                  "_tool_call_count": "tool_call_count", "_iteration_count": "iteration_count"}
    def __getattr__(self, name):
        if name in AgentLoop._STATE_MAP:
            if "_state" not in self.__dict__:
                self.__dict__["_state"] = LoopState()
            return getattr(self._state, AgentLoop._STATE_MAP[name])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        if name in AgentLoop._STATE_MAP:
            if "_state" not in self.__dict__:
                self.__dict__["_state"] = LoopState()
            setattr(self._state, AgentLoop._STATE_MAP[name], value)
        else:
            super().__setattr__(name, value)

    @property
    def shield(self) -> Any: return self._shield
    @property
    def nudge(self) -> Any: return self._nudge
    @property
    def nudge_task_ref(self) -> str: return self._nudge_task_ref or ""
    @property
    def system_prompt_len(self) -> int: return len(self._system_prompt_cache or "")
    async def close(self) -> None:
        await self.drain_background()
        if hasattr(self.provider, 'close'): await self.provider.close()
    def invalidate_system_prompt(self) -> None:
        """Invalidate + eagerly rebuild prompt cache (prevents 0-char prompt on mid-turn config reload)."""
        from caveman.agent.prompt import build_system_prompt
        self._system_prompt_cache = build_system_prompt(
            tool_schemas=self.tool_registry.get_schemas(),
            surface=self.surface,
            conversation_state=self._conversation_state,
        )
        logger.info("System prompt invalidated and rebuilt (%d chars)", len(self._system_prompt_cache))
    def switch_model(self, new_provider) -> None:
        old = getattr(self.provider, 'model', '?')
        self.provider = new_provider
        self._system_prompt_cache = None
        logger.info("Model switched: %s → %s", old, getattr(new_provider, 'model', '?'))
    def reset_session(self) -> None:
        self._state = LoopState(surface=self.surface)
        self._persistent_context = None
        self._system_prompt_cache = None
        self.metrics = type(self.metrics)()
        self.budget.reset()
    def get_activity_summary(self) -> dict:
        import time as _t
        elapsed = _t.time() - self._last_activity_ts if self._last_activity_ts else 0
        return {"last_activity_desc": self._last_activity_desc,
                "seconds_since_activity": round(elapsed, 1), "current_tool": self._current_tool,
                "budget_used": self.budget.used, "budget_max": self.budget.max_total,
                "turn_count": self._turn_count, "tool_call_count": self._tool_call_count}
    def snapshot(self) -> dict:
        import hashlib
        if "_state" not in self.__dict__:
            self.__dict__["_state"] = LoopState()
        prompt = self._system_prompt_cache or ""
        state = self._state.snapshot()
        state.update({"system_prompt_len": len(prompt), "system_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                      "system_prompt": prompt, "budget_used": self.budget.used, "budget_max": self.budget.max_total})
        return state
    @property
    def _conversation_state(self):
        from caveman.agent.conversation_lifecycle import ConversationState
        return ConversationState(
            turn_count=self._turn_count, tool_call_count=self._tool_call_count,
            has_progress_calls=self._tool_call_count > 0,
            iteration_count=self._iteration_count)
    def restore(self, state: dict, context=None) -> None:
        self._state = LoopState.from_snapshot(state)
        if context is not None:
            self._persistent_context = context
        persisted = state.get("system_prompt", "")
        if persisted and len(persisted) > 100:
            self._system_prompt_cache = persisted
        else:
            from caveman.agent.prompt import build_system_prompt
            self._system_prompt_cache = build_system_prompt(
                tool_schemas=self.tool_registry.get_schemas(), surface=self.surface,
                conversation_state=self._conversation_state)
        logger.info("Restore: prompt=%d chars, persisted=%s", len(self._system_prompt_cache), bool(persisted))
    async def _prepare_multi_turn(self, task: str, recalled_ids: list[str], attachments=None):
        from caveman.agent.loop_engines import prepare_multi_turn
        return await prepare_multi_turn(self, task, recalled_ids, attachments=attachments)
    async def _post_task_engines(self, context, task, result, matched_skills):
        from caveman.agent.loop_engines import post_task_engines
        await post_task_engines(self, context, task, result, matched_skills)
    def _get_phase_rules(self) -> str:
        return get_phase_rules(self.surface, self._conversation_state)
    def _record_turn_metrics(self, turn_start, recalled_ids, matched_skills, result):
        from caveman.agent.loop_engines import record_turn_metrics
        record_turn_metrics(self, turn_start, recalled_ids, matched_skills, result)
    async def run(self, task: str, system_prompt: str | None = None) -> str:
        result = ""
        async for ev in self.run_stream(task, system_prompt):
            if is_result_event_type(ev.type): result = str(ev.data) if ev.data else ""
            elif ev.type == "error": raise RuntimeError(str(ev.data))
        return result
    async def run_stream(self, task: str, system_prompt: str | None = None, attachments: list[dict[str, str]] | None = None) -> AsyncIterator[StreamEvent]:
        _turn_start = _time.monotonic()
        self._nudge_task_ref = task
        self._turn_number += 1
        self._turn_count += 1
        await self.bus.emit(EventType.LOOP_START, {"task": task}, source="loop")
        _recalled_ids: list[str] = []
        def _capture_recalled(event):
            if event.source == "memory" and event.data.get("recalled_ids"):
                _recalled_ids.extend(event.data["recalled_ids"])
        self.bus.on(EventType.MEMORY_RECALL, _capture_recalled)
        if self._persistent_context is not None and self._turn_number > 1:
            context, system, matched_skills = await self._prepare_multi_turn(task, _recalled_ids, attachments=attachments)
        else:
            context, system, matched_skills = await phase_prepare(
                task, system_prompt, self.provider, self.skill_manager,
                self.memory_manager, self.trajectory_recorder,
                self._recall, self.engine_flags, self.bus, self.tool_registry,
                surface=self.surface,
                conversation_state=self._conversation_state,
                attachments=attachments,
            )
            self._system_prompt_cache = system
        self.bus.off(EventType.MEMORY_RECALL, _capture_recalled)
        self._persistent_context = context
        final = ""
        compressor = CompressionPipeline(provider=self.provider)
        iteration = 0
        self._iteration_count = 0
        while self.budget.consume():
            await self.bus.emit(EventType.ITERATION_START, {"iteration": iteration}, source="loop")
            yield StreamEvent(type="iteration_start", data={"iteration": iteration, "max": self.max_iterations, "remaining": self.budget.remaining})
            try:
                utilization = float(context.utilization)
            except (TypeError, ValueError):
                utilization = 0.0
            if utilization >= 0.7:
                await self.bus.emit(EventType.CONTEXT_UTILIZATION, {
                    "utilization": utilization, "total_tokens": context.total_tokens,
                    "max_tokens": context.max_tokens, "messages": len(context.messages),
                }, source="loop")
                yield StreamEvent(type="context_pressure", data={
                    "utilization": utilization, "total_tokens": context.total_tokens,
                    "max_tokens": context.max_tokens,
                })
            from caveman.agent.loop_engines import run_preemptive_compaction
            context = await run_preemptive_compaction(
                context, compressor, self._shield, self.bus, self.metrics,
            )
            _llm_start = _time.monotonic()
            self._last_activity_ts = _time.time()
            self._last_activity_desc = "LLM call"
            self._current_tool = ""
            text = ""
            tool_calls: list = []
            stop = "end_turn"
            try:
                async for ev in stream_llm_turn(self, context, system):
                    if ev.type == "_llm_done":
                        text, tool_calls, stop, retry_iteration = ev.data
                    else:
                        yield ev
                if retry_iteration:
                    continue
                if request_continuation_if_needed(self, context, text, tool_calls, stop):
                    final = text
                    iteration += 1
                    self._iteration_count = iteration
                    continue
            except Exception as e:
                yield StreamEvent(type="error", data=str(e))
                return
            if not text and not tool_calls and stop == "end_turn":
                return
            self.metrics.record_timing("llm_call_duration", _time.monotonic() - _llm_start)
            if text: final = text
            record_assistant_turn(context, text, tool_calls)
            if text: await self.trajectory_recorder.record_turn("gpt", text)
            # Tool calls are executable intent. A premature terminal marker in
            # streamed text must not cancel real work; otherwise the assistant
            # can claim success and silently skip the actions it already emitted.
            if text and CLOSING_LINE in text and tool_calls:
                logger.info(
                    "Terminal marker found with %d tool_calls — stripping marker "
                    "and executing tools instead of treating the task as verified",
                    len(tool_calls),
                )
                text = text.replace(CLOSING_LINE, "").rstrip()
                final = text or final
            if tool_calls:
                async for ev in execute_tool_phase(self, context, tool_calls):
                    yield ev
            should_break = await self._check_termination(stop, tool_calls, task, text=text)
            await self.bus.emit(EventType.ITERATION_END, {"iteration": iteration, "stop": stop, "tool_calls": len(tool_calls), "text_len": len(text)}, source="loop")
            exhausted = False
            if should_break: break
            iteration += 1
            self._iteration_count = iteration
        else:
            exhausted = True
            show_error(f"Max iterations ({self.max_iterations}) reached — budget exhausted")
        result = await phase_finalize(
            task, final, matched_skills, self.memory_manager,
            self.skill_manager, self.trajectory_recorder, self.bus,
            llm_fn=self._llm_fn, recalled_ids=_recalled_ids or None,
        )
        await self._post_task_engines(context, task, result, matched_skills)
        try:
            usage = self.provider.usage_stats
            if isinstance(usage, dict):
                await self.bus.emit(EventType.TURN_USAGE, usage, source="provider")
        except Exception as exc:
            logger.debug("unknown: suppressed %s", exc)
        await self.bus.emit(EventType.LOOP_END, {
            "task": task, "result": result, "result_len": len(result),
            "iterations": iteration + 1, "tool_calls": self._tool_call_count,
            "recalled_ids": _recalled_ids, "matched_skills": matched_skills,
        }, source="loop")
        self._record_turn_metrics(_turn_start, _recalled_ids, matched_skills, result)
        # Enforce closing format only when lifecycle policy says this turn needs
        # a structural completion signal. Simple replies and open questions stay natural.
        from caveman.agent.output_validator import enforce_closing_format, should_use_closing_marker
        _surface = getattr(getattr(self, 'session', None), 'surface', 'cli')
        _state = self._conversation_state
        _should_close = should_use_closing_marker(state=_state, final_text=result, surface=_surface)
        result = enforce_closing_format(result, _should_close, surface=_surface, task=task)
        if exhausted:
            result = (
                f"⚠️ 已达到本轮迭代上限（{self.max_iterations}），任务没有被验证为完成。\n\n"
                f"{result}" if result else f"⚠️ 已达到本轮迭代上限（{self.max_iterations}），任务没有被验证为完成。"
            )
        yield StreamEvent(type="result", data=result)
    async def _dispatch_skill_tool(self, name: str, args: dict) -> str:
        r = await self.tool_registry.dispatch(name, args)
        return r if isinstance(r, str) else str(r)
    async def _offer_matching_skill(self, task: str) -> None:
        try:
            skills = self.skill_manager.match(task)
            if skills:
                await self.bus.emit(EventType.SKILL_MATCH, {
                    "skills": [s.name for s in skills], "offered": True}, source="skill")
        except Exception as exc:
            logger.debug("_offer_matching_skill: suppressed %s", exc)
    async def _check_termination(self, stop: str, tool_calls: list, task: str, text: str = "") -> bool:
        from caveman.agent.loop_engines import check_termination; return await check_termination(stop, tool_calls, task, text=text)
    async def _update_shield(self, context, task: str) -> None:
        from caveman.agent.loop_engines import update_shield
        await update_shield(self, context, task)
