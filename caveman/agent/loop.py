"""Core agent loop v3 — thin orchestrator over decomposed phases."""
from __future__ import annotations
import asyncio, logging, os, uuid
from collections.abc import AsyncIterator
logger = logging.getLogger(__name__)

from caveman.agent.stream import StreamEvent, StreamBuffer
from caveman.agent.context import AgentContext
from caveman.agent.bg_tasks import BackgroundTaskMixin
from caveman.providers.llm import LLMProvider
from caveman.compression.pipeline import CompressionPipeline
from caveman.providers.anthropic_provider import AnthropicProvider
from caveman.tools.registry import ToolRegistry
from caveman.memory.manager import MemoryManager
from caveman.memory.nudge import MemoryNudge
from caveman.skills.manager import SkillManager
from caveman.skills.executor import SkillExecutor
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.security.permissions import PermissionManager, PermissionLevel
from caveman.events import EventBus, EventType, create_default_bus
from caveman.engines.reflect import ReflectEngine
from caveman.engines.flags import EngineFlags
from caveman.engines.shield import CompactionShield
from caveman.engines.recall import RecallEngine
from caveman.agent.display import show_error
from caveman.agent.metrics import AgentMetrics

# Phase functions
from caveman.agent.phases import (
    phase_prepare, phase_compress,
    record_assistant_turn, phase_finalize,
)
from caveman.agent.tools_exec import phase_tool_execution

DEFAULT_SYSTEM = (
    "You are Caveman, an AI agent that learns, executes, and evolves.\n"
    "You have tools for bash, file ops, web search, and more.\n"
    "Think step-by-step. Be concise but thorough.\n"
    "Memory is automatic — Shield saves session state after each task."
)

class AgentLoop(BackgroundTaskMixin):
    """Main agent loop: orchestrates phases via event-driven pipeline."""

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
        self.surface = surface

        if provider is None:
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
        for ctx_key, ctx_val in [("checkpoint_manager", self.checkpoint_manager),
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
        self.engine_flags = engine_flags or EngineFlags()

        self._llm_fn = llm_fn
        self._shield = shield or CompactionShield(session_id=uuid.uuid4().hex[:12], llm_fn=llm_fn)
        self._recall = recall_engine or RecallEngine(memory_manager=self.memory_manager)
        self._nudge = nudge_engine or MemoryNudge(
            memory_manager=self.memory_manager, llm_fn=llm_fn, interval=10, first_nudge=3)
        self._reflect = reflect_engine or ReflectEngine(skill_manager=self.skill_manager, llm_fn=llm_fn)
        self._skill_executor = SkillExecutor(tool_dispatch_fn=self._dispatch_skill_tool)
        self._lint = lint_engine
        self._ripple = None
        self._turn_count = 0
        self._tool_call_count = 0
        self._nudge_task_ref = ""
        self._persistent_context: AgentContext | None = None
        self._system_prompt_cache: str | None = None
        self._turn_number = 0
        self._ripple = None
        self._bg_tasks: set[asyncio.Task] = set()

        # Wire inner flywheel event chain
        self._wire_flywheel()
    def _wire_flywheel(self):
        from caveman.engines.event_chain import wire_inner_flywheel
        from caveman.engines.manager import EngineSet
        engines = EngineSet(shield=self._shield, nudge=self._nudge,
                            reflect=self._reflect, ripple=self._ripple,
                            lint=self._lint, recall=self._recall)
        self._flywheel_handlers = wire_inner_flywheel(
            self.bus, engines,
            get_turns=lambda: self.trajectory_recorder.to_sharegpt(),
            get_task=lambda: self._nudge_task_ref)

    def set_lint(self, engine) -> None:
        self._lint = engine
    def set_ripple(self, engine) -> None:
        self._ripple = engine

    def snapshot(self) -> dict:
        """Capture all restorable state. Runner calls this before persisting."""
        return {
            "turn_number": self._turn_number,
            "turn_count": self._turn_count,
            "tool_call_count": self._tool_call_count,
            "surface": self.surface,
            "system_prompt_len": len(self._system_prompt_cache or ""),
        }

    def restore(self, state: dict, context=None) -> None:
        """Restore state from snapshot. Runner calls this after loading transcript."""
        self._turn_number = state.get("turn_number", 0)
        self._turn_count = state.get("turn_count", 0)
        self._tool_call_count = state.get("tool_call_count", 0)
        self.surface = state.get("surface", getattr(self, "surface", "cli"))
        if context is not None:
            self._persistent_context = context
            # Inject format reminder into restored context to override old style
            from caveman.agent.response_style import get_format_reminder
            reminder = get_format_reminder(self.surface)
            if reminder and context.messages:
                context.add_message("system", f"[Style reset] {reminder}")
        # Always rebuild system prompt (not persisted — depends on workspace files)
        from caveman.agent.prompt import build_system_prompt
        self._system_prompt_cache = build_system_prompt(
            tool_schemas=self.tool_registry.get_schemas(), surface=self.surface)

    async def _prepare_multi_turn(self, task: str, recalled_ids: list[str]):
        """Reuse context, re-recall memories, rebuild prompt if needed."""
        context = self._persistent_context
        from caveman.agent.response_style import get_format_reminder
        reminder = get_format_reminder(self.surface)
        context.add_message("user", f"{task}\n{reminder}" if reminder else task)
        await self.trajectory_recorder.record_turn("human", task)
        if not self._system_prompt_cache:
            from caveman.agent.prompt import build_system_prompt
            self._system_prompt_cache = build_system_prompt(
                tool_schemas=self.tool_registry.get_schemas(), surface=self.surface)
            logger.info("Rebuilt system prompt for restored session (surface=%s)", self.surface)
        matched_skills = self.skill_manager.match(task)
        try:
            new_memories = await self.memory_manager.recall(task, top_k=3)
            if new_memories:
                recalled_ids.extend(m.id for m in new_memories)
                await self.bus.emit(EventType.MEMORY_RECALL, {
                    "query": task, "results": len(new_memories),
                    "recalled_ids": [m.id for m in new_memories], "recall_hit": True,
                }, source="memory")
        except Exception as e:
            logger.debug("Multi-turn recall failed: %s", e)
        return context, self._system_prompt_cache, matched_skills
    async def _post_task_engines(self, context, task, result, matched_skills):
        await self._update_shield(context, task)
        if self.engine_flags.is_enabled("reflect"):
            try:
                await self._reflect.reflect(task, self.trajectory_recorder.to_sharegpt(), result)
            except Exception as e:
                logger.debug("Reflect failed: %s", e)
        self._safe_bg(self._end_nudge(task))
        self._safe_bg(self._check_save_skill(task))
        if self._lint and self.engine_flags.is_enabled("lint"):
            self._safe_bg(self._run_lint())

    def _record_turn_metrics(self, turn_start, recalled_ids, matched_skills, result):
        import time as _t
        self.metrics.record_timing("total_turn_duration", _t.monotonic() - turn_start)
        self.metrics.increment("turns_completed")
        for cond, key in [(recalled_ids, "recall_hits"), (matched_skills, "skill_match_hits")]:
            if cond:
                self.metrics.increment(key)
        self.metrics.increment("recall_attempts")
        self.metrics.increment("skill_match_attempts")
        from caveman.utils import detect_success
        if detect_success(result):
            self.metrics.increment("task_successes")

    async def run(self, task: str, system_prompt: str | None = None) -> str:
        """Execute task — delegates to run_stream() (single implementation)."""
        result = ""
        async for event in self.run_stream(task, system_prompt):
            if event.type == "done":
                result = str(event.data) if event.data else ""
            elif event.type == "error":
                raise RuntimeError(str(event.data))
        return result

    async def run_stream(self, task: str, system_prompt: str | None = None) -> AsyncIterator[StreamEvent]:
        """Streaming execution — the SINGLE implementation. run() delegates here."""
        import time as _time
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

        # Phase 1: Prepare
        if self._persistent_context is not None and self._turn_number > 1:
            context, system, matched_skills = await self._prepare_multi_turn(task, _recalled_ids)
        else:
            context, system, matched_skills = await phase_prepare(
                task, system_prompt, self.provider, self.skill_manager,
                self.memory_manager, self.trajectory_recorder,
                self._recall, self.engine_flags, self.bus, self.tool_registry,
                surface=self.surface,
            )
            self._system_prompt_cache = system

        self.bus.off(EventType.MEMORY_RECALL, _capture_recalled)
        self._persistent_context = context

        final = ""
        compressor = CompressionPipeline(provider=self.provider)
        iteration = 0

        for iteration in range(self.max_iterations):
            await self.bus.emit(EventType.ITERATION_START, {"iteration": iteration}, source="loop")

            # Phase 2: Compress
            if context.should_compress():
                if self._shield:
                    try:
                        msg_dicts = [{"role": m.role, "content": m.content} for m in context.messages]
                        await self._shield.update(msg_dicts)
                    except Exception as e:
                        logger.warning("Shield pre-compression update failed: %s", e)
                _comp_start = _time.monotonic()
                context, _ = await phase_compress(context, compressor, self.bus)
                self.metrics.record_timing("compression_duration", _time.monotonic() - _comp_start)

            # Phase 3: LLM call (streaming)
            _llm_start = _time.monotonic()
            text_parts: list[str] = []
            tool_calls: list = []
            stop = "end_turn"

            try:
                messages = [{"role": m.role, "content": m.content} for m in context.messages]
                tool_defs = self.tool_registry.get_schemas() if self.tool_registry else []
                async for ev in self.provider.safe_complete(
                    messages=messages, system=system, tools=tool_defs or None, stream=True,
                ):
                    etype = ev.get("type")
                    if etype == "delta":
                        text_parts.append(ev["text"])
                        yield StreamEvent(type="token", data=ev["text"])
                    elif etype == "tool_call":
                        tool_calls.append(ev)
                        yield StreamEvent(type="tool_call", data=ev)
                    elif etype == "done":
                        stop = ev.get("stop_reason", "end_turn")
                        if stop == "max_tokens":
                            yield StreamEvent(type="token", data="\n\n⚠️ (达到 token 上限，回复被截断)")
                    elif etype == "error" and ev.get("action") == "abort":
                        yield StreamEvent(type="error", data=ev.get("error", "Unknown error"))
                        return
                text = "".join(text_parts)
            except Exception as e:
                yield StreamEvent(type="error", data=str(e))
                return

            self.metrics.record_timing("llm_call_duration", _time.monotonic() - _llm_start)
            if text:
                final = text

            # Phase 4: Record
            record_assistant_turn(context, text, tool_calls)
            if text:
                await self.trajectory_recorder.record_turn("gpt", text)

            # Phase 5: Tools
            if tool_calls:
                _tool_start = _time.monotonic()
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

            # Phase 6: Termination
            should_break = await self._check_termination(stop, tool_calls, task)
            await self.bus.emit(EventType.ITERATION_END, {
                "iteration": iteration, "stop": stop,
                "tool_calls": len(tool_calls), "text_len": len(text),
            }, source="loop")

            if should_break:
                break
        else:
            show_error(f"Max iterations ({self.max_iterations}) reached")

        # Phase 7: Finalize
        result = await phase_finalize(
            task, final, matched_skills, self.memory_manager,
            self.skill_manager, self.trajectory_recorder, self.bus,
            llm_fn=self._llm_fn, recalled_ids=_recalled_ids or None,
        )

        # Phase 8: Post-task engines (shield, reflect, nudge, skill save, lint)
        await self._post_task_engines(context, task, result, matched_skills)

        await self.bus.emit(EventType.LOOP_END, {
            "task": task, "result_len": len(result),
            "iterations": iteration + 1, "tool_calls": self._tool_call_count,
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
        except Exception:
            pass

    async def _check_termination(self, stop: str, tool_calls: list, task: str) -> bool:
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
    async def _update_shield(self, context, task: str) -> None:
        if not self.engine_flags.is_enabled("shield"):
            return
        try:
            msgs = [m if isinstance(m, dict) else {"role": getattr(m, "role", "unknown"),
                     "content": getattr(m, "content", str(m))} for m in context.messages]
            await self._shield.update(msgs, task)
            await self._shield.save()
            await self.bus.emit(EventType.SHIELD_UPDATE, {
                "session_id": self._shield.essence.session_id,
                "turn_count": self._shield.essence.turn_count,
            }, source="shield")
        except Exception as e:
            logger.warning("Shield update failed: %s", e)
