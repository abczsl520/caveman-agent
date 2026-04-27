"""Gateway runner — streaming session management for Discord/Telegram."""
from __future__ import annotations
import asyncio, logging
import re, time
from typing import Any, NamedTuple

from caveman.agent.factory import create_loop
from caveman.agent.session_store import SessionMeta
from caveman.agent.session_db import SessionDB
from caveman.config.loader import load_config
from caveman.timeouts import TASK_SHORT
from caveman.gateway.router import GatewayRouter
from caveman.gateway.infra import GatewayInfra
from caveman.gateway.session_context import set_session_context, clear_session_context
from caveman.gateway.message_pipeline import (
    MessageContext, MessageAction, normalize_message, dedupe_message, DedupeCache,
)
from caveman.gateway.reply_queue import QueueManager
from caveman.gateway.routing import resolve_route
from caveman.paths import CAVEMAN_HOME
from caveman.gateway.transcript_cleaner import clean_transcript_message as _clean_transcript_message
from caveman.gateway.task_runner import AgentTaskError, run_single_task
__all__ = ["SESSION_TTL", "GatewayServer", "run_gateway", "run_gateway_forever"]

logger = logging.getLogger("caveman.gateway")


def _looks_like_iteration_budget_exhaustion(text: str) -> bool:
    """Return True for per-round iteration budget exhaustion, not task success.

    Automatic flywheel rounds frequently hit the agent-loop iteration ceiling.
    That is a continuable per-round budget signal; treating it like a terminal
    AgentTaskError made the 20-round chain look stopped.
    """
    return any(
        marker in text
        for marker in ("迭代上限", "budget exhausted", "Max iterations")
    )


_AUTO_MAX_ROUNDS = 20
_AUTO_PATTERNS = re.compile(
    r'继续\s*飞轮|自动第\s*\d+\s*/\s*\d+\s*轮|不要停|不间断|持续|一直|keep\s*going|don.t\s*stop|autonomous|auto.?continue',
    re.I,
)
_RUNNING_NUDGE_PATTERNS = re.compile(
    r'^(?:\s|[。！？!?,，、])*('
    r'继续(?:\s*(?:飞轮|下一步|不要停|别停|下去|推进|排查))?'
    r'|不要停|别停|接着(?:来|做|查|跑)?|继续跑|继续查|继续做'
    r'|怎么.*停(?:了)?|咋.*停(?:了)?|又停(?:了)?|没反应(?:了)?|是不是.*失败(?:了)?'
    r'|keep\s*going|don.t\s*stop'
    r')(?:\s|[。！？!?,，、])*$',
    re.I,
)
SESSION_TTL = 30 * 60


def _is_non_interrupting_running_task_nudge(task: str) -> bool:
    """Return True when an incoming message is a status/continue nudge.

    While a long flywheel is already running, impatient follow-ups like
    "继续不要停" or "怎么又停了" must not flip the current task's shutdown flag;
    doing so made the system stop exactly when the user asked it to continue.
    """
    return bool(_RUNNING_NUDGE_PATTERNS.search((task or "").strip()))

class _SendResult(NamedTuple):
    id: str

class _AdapterBridge:
    """Bridge: BasePlatformAdapter → legacy Gateway interface for router.send()."""
    def __init__(self, adapter):
        self._a = adapter
    @property
    def name(self) -> str:
        return self._a.name.lower()
    async def send_message(self, channel_id: str, content: str) -> None:
        r = await self._a.send(channel_id, content); return _SendResult(id=r.message_id) if r.success else None
    async def send_reply(self, channel_id: str, content: str, reply_to: int) -> None:
        r = await self._a.send(channel_id, content, reply_to=str(reply_to)); return _SendResult(id=r.message_id) if r.success else None

class GatewayServer:
    """Owns all gateway state: sessions, store, router, locks."""

    def __init__(self, db_path=None, config_path: str | None = None):
        self.router = GatewayRouter()
        self.sessions: dict[str, dict] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}
        self.store = SessionDB(db_path or (CAVEMAN_HOME / "sessions.db"))
        self.config_path = config_path
        from caveman.providers.usage_pricing import UsageTracker
        self.usage_tracker = UsageTracker()
        self._dedupe_cache = DedupeCache()
        self._queue_manager = QueueManager()
        self._infra = GatewayInfra()
        self._auto_patterns = _AUTO_PATTERNS
        self._cached_config = load_config(self.config_path)
        self._migrate_json_store()

    def _migrate_json_store(self):
        json_dir = CAVEMAN_HOME / "gateway_sessions"
        if json_dir.exists():
            try:
                migrated = self.store.migrate_from_json(json_dir)
                if migrated:
                    logger.info("Migrated %d sessions from JSON to SQLite", migrated)
            except Exception as e:
                logger.warning("JSON→SQLite migration failed: %s", e)
    def _cfg(self) -> dict: return self._cached_config

    def _session_key(self, ctx: dict[str, Any]) -> str:
        route = resolve_route(
            platform=ctx.get("gateway_name", "discord"),
            chat_id=str(ctx.get("channel_id", "?")),
            sender_id=str(ctx.get("user_id", "")),
            thread_id=str(ctx.get("thread_id", "")),
        )
        return route.session_key

    def _get_lock(self, key: str) -> asyncio.Lock:
        return self.session_locks.setdefault(key, asyncio.Lock())

    async def _cleanup_session(self, key: str, session: dict) -> None:
        try:
            loop = session.get("loop")
            if loop:
                from caveman.agent.session_hooks import on_session_end
                result = await on_session_end(
                    shield=loop.shield, nudge=loop.nudge,
                    trajectory=loop.trajectory_recorder,
                    task=loop.nudge_task_ref,
                )
                logger.info("Session %s cleanup: %s", key, result)
                await loop.close()
        except Exception as e:
            logger.warning("Session %s cleanup failed: %s", key, e)

    async def _get_or_create_session(self, key: str) -> dict:
        now = time.monotonic()
        if key in self.sessions:
            s = self.sessions[key]
            if now - s["last_active"] < SESSION_TTL:
                s["last_active"] = now
                return s
            try: await self._cleanup_session(key, s)
            except RuntimeError: pass  # intentional: cleanup best-effort
            del self.sessions[key]
            self.session_locks.pop(key, None)

        surface = key.split(":")[0] if ":" in key else "cli"
        loop = create_loop(config_path=self.config_path, surface=surface)
        loop.gateway_router = self.router
        loop.tool_registry.set_context("gateway_router", self.router)
        sid = key.replace(":", "_")
        meta = self.store.load_meta(sid)
        if meta:
            self._restore_session(key, sid, meta, loop, surface)
        else:
            meta = SessionMeta(session_id=sid, model=getattr(loop.provider, 'model_name', ''),
                               started_at=time.time(), surface=surface)
            self.store.save_meta(meta)
        session = {"loop": loop, "meta": meta, "last_active": now, "task_count": 0}
        self.sessions[key] = session
        return session

    def _restore_session(self, key, session_id, meta, loop, surface):
        transcript = self.store.load_transcript(session_id)
        if not transcript:
            return
        from caveman.agent.context import AgentContext
        from caveman.utils import estimate_tokens as _est
        ctx = AgentContext(max_tokens=loop.provider.context_length)
        restored, tok = 0, 0
        budget = loop.provider.context_length * 0.6 - getattr(loop, 'system_prompt_len', 0)
        for turn in transcript[-40:]:
            t = _est(turn.get("content", ""))
            if tok + t > budget:
                break
            c = _clean_transcript_message(turn["role"], turn.get("content", ""))
            if c is None:
                continue
            ctx.add_message(turn["role"], c)
            tok += t
            restored += 1
        snap = self.store.load_snapshot(session_id)
        for k, v in [("turn_number", meta.turn_count), ("turn_count", meta.turn_count), ("surface", meta.surface or surface)]:
            snap.setdefault(k, v)
        loop.restore(snap, context=ctx)
        logger.info("Restored %s: %d turns, %s, prompt=%d", key, restored, loop.surface, loop.system_prompt_len)

    async def handle_task(self, task: str, context: dict[str, Any]) -> str:
        # --- Message Pipeline: Normalize + Dedupe ---
        msg_ctx = MessageContext(
            body=task,
            message_id=str(context.get("message_id", "")),
            platform=context.get("gateway_name", "discord"),
            chat_id=str(context.get("channel_id", "")),
            sender_id=str(context.get("user_id", "")),
            sender_name=context.get("user_name", ""),
            thread_id=str(context.get("thread_id", "")),
            chat_type=context.get("chat_type", "dm"),
            is_mention=context.get("is_mention", False),
            is_reply_to_bot=context.get("is_reply_to_bot", False),
        )
        msg_ctx = normalize_message(msg_ctx)
        msg_ctx = dedupe_message(msg_ctx, self._dedupe_cache)
        if msg_ctx.action == MessageAction.REJECTED:
            logger.debug("Message deduplicated: %s", msg_ctx.message_id)
            return ""

        key = self._session_key(context)
        set_session_context(
            platform=msg_ctx.platform,
            chat_id=msg_ctx.chat_id,
            thread_id=msg_ctx.thread_id,
            sender_id=msg_ctx.sender_id,
            session_key=key,
        )
        lock = self._get_lock(key)

        task = msg_ctx.body

        if not msg_ctx.body.strip():
            clear_session_context()
            return ""

        # --- Command dispatch: /commands bypass agent loop ---
        if task.startswith("/"):
            from caveman.commands.dispatcher import dispatch as cmd_dispatch
            session = await self._get_or_create_session(key)
            gw = context.get("gateway_name", "discord")
            ch = str(context.get("channel_id", ""))
            respond = lambda msg, _g=gw, _c=ch: asyncio.ensure_future(self.router.send(_g, _c, msg))
            if await cmd_dispatch(task, session["loop"], surface=gw, session_key=key, respond_fn=respond):
                clear_session_context()
                return ""

        reply_to = context.get("reply_to")
        if reply_to and reply_to.get("content"):
            task = f'[回复 {reply_to.get("author", "?")} 的消息: "{reply_to["content"]}"]\n{task}'

        if lock.locked():
            session = self.sessions.get(key)
            if session:
                gw = context.get("gateway_name", "discord")
                ch = str(context.get("channel_id", ""))
                if _is_non_interrupting_running_task_nudge(task):
                    logger.info("Running-task nudge received without interrupting: %s", key)
                    await self.router.send(
                        gw,
                        ch,
                        "🔄 当前任务/自动飞轮还在运行；已收到催促，但不会中断正在执行的链路。请看后续 progress/自动续轮信号。",
                    )
                    clear_session_context()
                    return ""
                session["_interrupt"] = True
                ctx = session.get("_task_ctx")
                if ctx: ctx.shutdown_flag = True
                logger.info("Interrupting running task for new message: %s", key)
                await self.router.send(
                    gw,
                    ch,
                    "⏹️ 收到新消息，正在停止当前任务...",
                )

        from caveman.gateway.runner_task import run_locked_gateway_task
        try:
            return await run_locked_gateway_task(self, task, context, key, msg_ctx)
        finally:
            clear_session_context()

    async def _auto_continue(self, result, session, gw_name, channel_id, source_channel):
        config = self._cfg()
        for rnd in range(1, _AUTO_MAX_ROUNDS + 1):
            session["auto_rounds"] = rnd
            cont_task = (f"继续飞轮 (自动第 {rnd}/{_AUTO_MAX_ROUNDS} 轮)。"
                         f"上一轮结果摘要：{(result or '')[:200]}。"
                         "继续下一个最高复利的改进；如果还在排查或修复中，"
                         "只汇报进展和证据，不要输出终止性收尾。")
            logger.info("Auto-continue round %d/%d", rnd, _AUTO_MAX_ROUNDS)
            await self.router.send(gw_name, channel_id, f"🔄 飞轮自动继续 ({rnd}/{_AUTO_MAX_ROUNDS})...")
            source_channel["_progress_sent"] = 0; source_channel["_auto_continue"] = True
            try:
                result = await run_single_task(
                    cont_task, session, gw_name, channel_id, source_channel, self.router, self.store, config)
                result_text = str(result or "")
                if _looks_like_iteration_budget_exhaustion(result_text): await self.router.send(gw_name, channel_id, f"⚠️ 飞轮第 {rnd} 轮触达迭代预算上限；这不是完成或停止信号，继续下一轮排查。")
            except AgentTaskError as e:
                err_text = str(e)
                logger.warning("Auto-continue round %d reported agent error: %s", rnd, e)
                result = f"第 {rnd} 轮 agent 错误：{err_text[:200]}。这不是完成信号，继续下一轮排查。"
                if _looks_like_iteration_budget_exhaustion(err_text):
                    await self.router.send(gw_name, channel_id, f"⚠️ 飞轮第 {rnd} 轮触达迭代预算上限；这不是完成或停止信号，继续下一轮排查。")
                else:
                    await self.router.send(gw_name, channel_id, f"⚠️ 飞轮第 {rnd} 轮 agent 错误，已记录；不把它当完成或停止，继续下一轮排查。")
                continue
            except asyncio.TimeoutError:
                result = f"第 {rnd} 轮超时。这不是完成信号，继续下一轮排查。"
                await self.router.send(gw_name, channel_id, f"⏰ 飞轮第 {rnd} 轮超时；这不是完成或停止信号，继续下一轮排查。")
                continue
            except Exception as e:
                logger.exception("Auto-continue round %d failed", rnd)
                result = f"第 {rnd} 轮运行异常：{type(e).__name__}: {str(e)[:200]}。已记录日志，继续下一轮排查。"
                await self.router.send(gw_name, channel_id, f"⚠️ 飞轮第 {rnd} 轮运行异常，已记录日志；继续下一轮排查，不把它当完成。")
                continue
            finally:
                source_channel.pop("_auto_continue", None)
        else:
            await self.router.send(gw_name, channel_id, f"🔄 飞轮自动模式已跑满 {_AUTO_MAX_ROUNDS} 轮；这只是轮次上限，不代表所有问题已完成。")
        return result

    async def start(self) -> None:
        """Start all configured gateways."""
        self._cached_config = load_config(self.config_path)
        config = self._cached_config
        gw_config = config.get("gateway", {})
        if not gw_config:
            logger.error("No gateway config found.")
            return

        # Wire all subsystems
        from caveman.gateway.wiring import wire_gateway
        from caveman.wiring import wire_all
        self._wired = wire_gateway(self)
        wire_all()

        self._infra.load_hooks()
        self._infra.load_task_registry()
        await self._infra.emit_hook("gateway:startup", {"config_keys": list(gw_config.keys())})

        # Try new platform adapters first, fall back to legacy
        gateways = await self._start_platform_adapters(gw_config, config)
        if not gateways:
            gateways = await self._start_legacy_gateways(gw_config, config)
        if not gateways:
            logger.error("No gateways enabled.")
            return

        logger.info("Starting %d gateway(s): %s", len(gateways), ", ".join(n for n, _ in gateways))
        tasks = [asyncio.create_task(gw.connect() if hasattr(gw, 'connect') else gw.start())
                 for _, gw in gateways
                 if (hasattr(gw, 'connect') or hasattr(gw, 'start')) and not getattr(gw, '_running', False)]
        reaper = asyncio.create_task(self._session_reaper())
        cron_task = asyncio.create_task(self._start_cron(config))
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass  # intentional: KeyboardInterrupt suppressed
        finally:
            reaper.cancel()
            cron_task.cancel()
            for _, gw in gateways:
                try:
                    await (gw.disconnect() if hasattr(gw, 'disconnect') else gw.stop())
                except Exception as e:
                    logger.debug("Suppressed: %s", e)

    async def _start_platform_adapters(self, gw_config: dict, config: dict) -> list:
        """Start gateways using the new BasePlatformAdapter system."""
        from caveman.gateway.platform_types import PlatformConfig
        from caveman.gateway.platform_registry import get_adapter
        gateways = []
        for name in ("discord", "telegram", "slack", "whatsapp", "signal", "matrix", "feishu"):
            pcfg = gw_config.get(name, {})
            if not pcfg.get("enabled"):
                continue
            adapter = get_adapter(name, PlatformConfig.from_dict(pcfg))
            if adapter:
                adapter.set_message_handler(self._adapter_message_handler)
                # Register adapter as a legacy gateway bridge so router.send() works
                self.router.register(_AdapterBridge(adapter))
                gateways.append((adapter.name, adapter))
        return gateways

    async def _adapter_message_handler(self, event) -> str | None:
        """Bridge: BasePlatformAdapter → existing handle_task."""
        src = event.source
        context = {
            "channel_id": src.chat_id if src else "",
            "user_id": src.user_id if src else "",
            "user_name": src.user_name if src else "",
            "username": src.user_name if src else "",
            "message_id": event.message_id,
            "gateway_name": src.platform.value if src else "unknown",
            "is_thread": src.chat_type == "thread" if src else False,
            "thread_id": src.thread_id or "" if src else "",
            "chat_type": src.chat_type if src else "dm",
            "is_mention": getattr(event, "is_mention", False),
            "is_reply_to_bot": getattr(event, "is_reply_to_bot", False),
        }
        if event.media_urls:
            context["attachments"] = [{"url": u, "content_type": t}
                                      for u, t in zip(event.media_urls, event.media_types)]
        if event.reply_to_text:
            context["reply_to"] = {"content": event.reply_to_text}
        return await self.handle_task(event.text, context)

    async def _start_legacy_gateways(self, gw_config: dict, config: dict) -> list:
        """Start gateways using the legacy Gateway ABC system."""
        from caveman.gateway.legacy_startup import start_legacy_gateways
        return await start_legacy_gateways(gw_config, config, self.handle_task, self.router)

    async def _session_reaper(self):
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(TASK_SHORT)  # 5 minutes
            now = time.monotonic()
            expired = [
                k for k, s in self.sessions.items()
                if now - s.get("last_active", 0) >= SESSION_TTL
            ]
            for key in expired:
                session = self.sessions.get(key)
                if not session: continue
                try:
                    await self._cleanup_session(key, session)
                    self.sessions.pop(key, None)  # pop AFTER successful cleanup
                    self.session_locks.pop(key, None)
                    logger.info("Reaped expired session: %s", key)
                except Exception as e:
                    logger.warning("Reaper cleanup failed for %s, will retry: %s", key, e)

    async def _start_cron(self, config: dict):
        """Start the cron scheduler."""
        from caveman.gateway.cron_integration import start_cron_scheduler
        await start_cron_scheduler(
            config=config,
            get_or_create_session=self._get_or_create_session,
            router=self.router,
            store=self.store,
        )
# --- Backward-compatible module-level API (use GatewayServer directly) ---
_server: GatewayServer | None = None
def _get_server() -> GatewayServer:
    global _server
    if _server is None: _server = GatewayServer()
    return _server
async def run_gateway(config_path: str | None = None) -> None:
    """Backward-compatible entry point. Creates the singleton GatewayServer."""
    global _server
    _server = GatewayServer(config_path=config_path)
    await _server.start()
async def _drain_active_sessions(timeout: float) -> tuple[int, bool]:
    from caveman.gateway.gateway_lifecycle import drain_active_sessions
    srv = _get_server()
    return await drain_active_sessions(srv.sessions, srv.session_locks, timeout)
async def run_gateway_forever(config_path: str | None = None, max_restarts: int = 10) -> None:
    """Run the gateway server indefinitely with auto-restart on failure."""
    from caveman.gateway.gateway_lifecycle import run_gateway_forever as _run
    await _run(config_path=config_path, max_restarts=max_restarts)
