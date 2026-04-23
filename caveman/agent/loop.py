"""Core agent loop v3 — thin orchestrator over decomposed phases."""
from __future__ import annotations
import asyncio, logging, os, time as _time
from collections.abc import AsyncIterator
logger = logging.getLogger(__name__)
from caveman.agent.stream import StreamEvent
from caveman.agent.context import AgentContext
from caveman.agent.bg_tasks import BackgroundTaskMixin
from caveman.agent.conversation_lifecycle import get_phase_rules
from caveman.providers.llm import LLMProvider
from caveman.providers.anthropic_adapter import CACHE_BOUNDARY
from caveman.compression.pipeline import CompressionPipeline
from caveman.tools.registry import ToolRegistry
from caveman.memory.manager import MemoryManager
from caveman.skills.manager import SkillManager
from caveman.skills.executor import SkillExecutor
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.security.permissions import PermissionManager, PermissionLevel
from caveman.events import EventBus, EventType, create_default_bus
from caveman.paths import DEFAULT_LLM_IDLE_TIMEOUT
from caveman.engines.flags import EngineFlags
from caveman.engines.manager import EngineManager
from caveman.agent.display import show_error
from caveman.agent.metrics import AgentMetrics
from caveman.agent.iteration_budget import IterationBudget
from caveman.agent.phases import (
    phase_prepare, record_assistant_turn, phase_finalize,
)
from caveman.agent.tools_exec import phase_tool_execution
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
        surface: str = "cli",
    ):
        from caveman.paths import DEFAULT_MODEL, DEFAULT_MAX_ITERATIONS
        self.model = model or DEFAULT_MODEL
        self.max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        self.budget = IterationBudget(self.max_iterations)
        self._fallback_chain = None
        try:  # Wire auxiliary client as fallback provider
            from caveman.agent.auxiliary_client import AuxiliaryConfig, _FallbackChain
            cfg = AuxiliaryConfig.from_env()
            self._fallback_chain = _FallbackChain(cfg) if cfg.provider else None
        except Exception as exc:
            logger.debug("__init__: suppressed %s", exc)
        self.surface = surface
        if provider is None:
            from caveman.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key=os.environ.get("ANTHROPIC_API_KEY", ""), model=self.model)
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()
        if tool_registry is None:
            self.tool_registry._register_builtins()
        self.memory_manager = memory_manager or MemoryManager()
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
        if permission_manager is None:
            permission_manager = PermissionManager()
            for k in list(permission_manager._permissions):
                permission_manager._permissions[k] = PermissionLevel.AUTO
        self.permission_manager = permission_manager
        if event_bus is None:
            event_bus, self._metrics = create_default_bus()
        else:
            self._metrics = None
        self.bus = event_bus
        for _c in [self.trajectory_recorder, self.permission_manager]:
            if hasattr(_c, '_bus') and _c._bus is None:
                _c._bus = self.bus
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
        self._turn_count = 0
        self._tool_call_count = 0
        self._nudge_task_ref = ""
        self._persistent_context: AgentContext | None = None
        self._system_prompt_cache: str | None = None
        self._turn_number = 0
        self._bg_tasks: set[asyncio.Task] = set()
        self._last_activity_ts = 0.0
        self._last_activity_desc = ""
        self._current_tool = ""
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
        self._persistent_context = None
        self._system_prompt_cache = None
        self._turn_number = 0
        self._tool_call_count = 0
        self._conversation_state = None
        self.metrics = type(self.metrics)()
        self.budget.reset()
        logger.info("Session state reset")
    def get_activity_summary(self) -> dict:
        import time as _t
        elapsed = _t.time() - self._last_activity_ts if self._last_activity_ts else 0
        return {
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "budget_used": self.budget.used,
            "budget_max": self.budget.max_total,
            "turn_count": self._turn_count,
            "tool_call_count": self._tool_call_count,
        }
    def snapshot(self) -> dict:
        import hashlib
        prompt = self._system_prompt_cache or ""
        return {
            "turn_number": self._turn_number,
            "turn_count": self._turn_count,
            "tool_call_count": self._tool_call_count,
            "surface": self.surface,
            "system_prompt_len": len(prompt),
            "system_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "system_prompt": prompt,  # Persisted for cache stability (Hermes pattern)
            "budget_used": self.budget.used,
            "budget_max": self.budget.max_total,
        }
    @property
    def _conversation_state(self):
        from caveman.agent.conversation_lifecycle import ConversationState
        return ConversationState(
            turn_count=self._turn_count, tool_call_count=self._tool_call_count,
            has_progress_calls=self._tool_call_count > 0)
    def restore(self, state: dict, context=None) -> None:
        self._turn_number = state.get("turn_number", 0)
        self._turn_count = state.get("turn_count", 0)
        self._tool_call_count = state.get("tool_call_count", 0)
        self.surface = state.get("surface", getattr(self, "surface", "cli"))
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
    def _record_turn_metrics(self, turn_start, recalled_ids, matched_skills, result):
        from caveman.agent.loop_engines import record_turn_metrics
        record_turn_metrics(self, turn_start, recalled_ids, matched_skills, result)
    async def run(self, task: str, system_prompt: str | None = None) -> str:
        """Execute task — delegates to run_stream()."""
        result = ""
        async for ev in self.run_stream(task, system_prompt):
            if ev.type == "done": result = str(ev.data) if ev.data else ""
            elif ev.type == "error": raise RuntimeError(str(ev.data))
        return result
    async def run_stream(self, task: str, system_prompt: str | None = None, attachments: list[dict[str, str]] | None = None) -> AsyncIterator[StreamEvent]:
        """Streaming execution — the SINGLE implementation. run() delegates here."""
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
            text_parts: list[str] = []
            tool_calls: list = []
            stop = "end_turn"
            try:
                messages = context.to_api_format()
                _phase_rules = get_phase_rules(self.surface, self._conversation_state)
                _effective_system = system
                if _phase_rules:
                    _effective_system = (
                        (system or "") +
                        CACHE_BOUNDARY +
                        f"## Conversation Phase\n{_phase_rules}"
                    )
                tool_defs = self.tool_registry.get_schemas() if self.tool_registry else []
                _last_token_ts = _time.monotonic()
                async for ev in self.provider.safe_complete(
                    messages=messages, system=_effective_system, tools=tool_defs or None, stream=True,
                ):
                    now = _time.monotonic()
                    if now - _last_token_ts > DEFAULT_LLM_IDLE_TIMEOUT:
                        logger.warning("LLM idle timeout: %ds without token", DEFAULT_LLM_IDLE_TIMEOUT)
                        yield StreamEvent(type="token", data=f"\n\n⚠️ LLM 无响应超时 ({DEFAULT_LLM_IDLE_TIMEOUT}s)")
                        stop = "idle_timeout"
                        break
                    _last_token_ts = now
                    etype = ev.get("type")
                    if etype == "delta":
                        text_parts.append(ev["text"])
                        await self.bus.emit(EventType.LLM_STREAM_DELTA, {"text": ev["text"]})
                        yield StreamEvent(type="token", data=ev["text"])
                    elif etype == "tool_call":
                        tool_calls.append(ev)
                        yield StreamEvent(type="tool_call", data=ev)
                    elif etype == "done":
                        stop = ev.get("stop_reason", "end_turn")
                        if stop == "max_tokens" and text_parts:
                            context.add_message("assistant", "".join(text_parts))
                            context.add_message("user", "你的回复被截断了，请继续。")
                            text_parts.clear(); tool_calls.clear()
                            continue
                    elif etype == "error" and ev.get("action") == "abort":
                        if self._fallback_chain and self._fallback_chain.has_fallbacks:
                            new_provider = self._fallback_chain.try_activate_next()
                            if new_provider:
                                self.provider = new_provider
                                yield StreamEvent(type="token", data=f"\n⚠️ 主模型失败，切换到备用模型...")
                                continue  # retry this iteration
                        yield StreamEvent(type="error", data=ev.get("error", "Unknown error"))
                        return
                text = "".join(text_parts)
                from caveman.providers.message_sanitizer import strip_reasoning_tags
                text = strip_reasoning_tags(text)
                if stop == "idle_timeout":
                    tool_calls = []  # discard incomplete tool calls
                    if not text:
                        text = "(LLM 无响应)"
            except Exception as e:
                yield StreamEvent(type="error", data=str(e))
                return
            self.metrics.record_timing("llm_call_duration", _time.monotonic() - _llm_start)
            if text: final = text
            record_assistant_turn(context, text, tool_calls)
            if text: await self.trajectory_recorder.record_turn("gpt", text)
            if tool_calls:
                _tool_start = _time.monotonic()
                tool_names = [tc.get("name", "?") for tc in tool_calls]
                self._last_activity_ts = _time.time()
                self._last_activity_desc = f"Tools: {', '.join(tool_names)}"
                self._current_tool = tool_names[0] if tool_names else ""
                self._tool_call_count = await phase_tool_execution(
                    context, tool_calls, self.tool_registry,
                    self.permission_manager, self.trajectory_recorder,
                    self.bus, self._tool_call_count, self._bg_skill_nudge,
                )
                self.metrics.record_timing("tool_dispatch_duration", _time.monotonic() - _tool_start)
                await self._offer_matching_skill(task)
                if self._tool_call_count % 10 == 0:
                    self._safe_bg(self._bg_skill_nudge())
                for tc in tool_calls:
                    yield StreamEvent(type="tool_result", data={"name": tc.get("name", "?")})
            should_break = await self._check_termination(stop, tool_calls, task)
            await self.bus.emit(EventType.ITERATION_END, {"iteration": iteration, "stop": stop, "tool_calls": len(tool_calls), "text_len": len(text)}, source="loop")
            if should_break: break
            iteration += 1
        else:
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
        yield StreamEvent(type="done", data=result)
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
    async def _check_termination(self, stop: str, tool_calls: list, task: str) -> bool:
        from caveman.agent.loop_engines import check_termination; return await check_termination(stop, tool_calls, task)
    async def _update_shield(self, context, task: str) -> None:
        from caveman.agent.loop_engines import update_shield
        await update_shield(self, context, task)
