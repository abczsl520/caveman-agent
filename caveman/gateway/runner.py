"""Gateway runner — streaming session management for Discord/Telegram.

Architecture (Hermes-inspired three-layer):
- Layer 1: Conversation history keeps ALL text (including tool-call messages)
- Layer 2: Interim messages — text before tool_calls is sent to user as commentary
           (stripped of think blocks, deduplicated)
- Layer 3: Final response — only the last no-tool-call text is the "real" reply

- _SmartBuffer: time-aware flush (chars OR silence timeout OR boundary)
- Tool heartbeat: "⏳ Running {tool}..." during long tools
- progress_tool is the ONLY way agent explicitly sends mid-task updates
- Autonomous mode: detects "继续不要停"/"keep going" and auto-continues
- SessionStore persists full transcript to disk
"""
from __future__ import annotations
import asyncio
import logging
import re
import time
from typing import Any

from caveman.agent.factory import create_loop
from caveman.agent.session_store import SessionStore, SessionMeta
from caveman.config.loader import load_config
from caveman.gateway.router import GatewayRouter
from caveman.gateway.smart_buffer import _SmartBuffer
from caveman.gateway.task_runner import run_single_task
from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger("caveman.gateway")

_shared_router = GatewayRouter()
_sessions: dict[str, dict] = {}
_session_locks: dict[str, asyncio.Lock] = {}  # Per-session locks (not global)
SESSION_TTL = 30 * 60
_store = SessionStore(CAVEMAN_HOME / "gateway_sessions")

# Activity-based idle detection (replaces hard timeout)

# Stuck-loop detection: same tool+args repeated N times

# Autonomous mode: max consecutive auto-continues before requiring user input
_AUTO_MAX_ROUNDS = 20

# Patterns that activate autonomous mode
_AUTO_PATTERNS = re.compile(
    r'不要停|不间断|持续|一直|keep\s*going|don.t\s*stop|autonomous|auto.?continue',
    re.IGNORECASE,
)


def _session_key(ctx: dict[str, Any]) -> str:
    return f"{ctx.get('gateway_name','discord')}:{ctx.get('channel_id','?')}:{ctx.get('user_id','?')}"


def _get_session_lock(key: str) -> asyncio.Lock:
    """Get or create a per-session lock (allows parallel sessions)."""
    if key not in _session_locks:
        _session_locks[key] = asyncio.Lock()
    return _session_locks[key]


async def _cleanup_session(key: str, session: dict) -> None:
    """Run end-of-session hooks before evicting a session."""
    try:
        loop = session.get("loop")
        if loop:
            from caveman.agent.session_hooks import on_session_end
            result = await on_session_end(
                shield=loop._shield, nudge=loop._nudge,
                trajectory=loop.trajectory_recorder,
                task=loop._nudge_task_ref or "",
            )
            logger.info("Session %s cleanup: %s", key, result)
    except Exception as e:
        logger.warning("Session %s cleanup failed: %s", key, e)


async def _get_or_create_session(key: str, config_path: str | None) -> dict:
    now = time.monotonic()
    if key in _sessions:
        s = _sessions[key]
        if now - s["last_active"] < SESSION_TTL:
            s["last_active"] = now
            return s
        # Session expired — run cleanup synchronously before evicting
        try:
            loop = asyncio.get_running_loop()
            await _cleanup_session(key, s)
        except RuntimeError:
            pass  # No running loop — skip async cleanup
        del _sessions[key]
        _session_locks.pop(key, None)  # Fix #8: clean up lock too

    # Extract surface from session key (e.g. "discord:123:456" → "discord")
    surface = key.split(":")[0] if ":" in key else "cli"
    loop = create_loop(config_path=config_path, surface=surface)
    loop.gateway_router = _shared_router
    loop.tool_registry.set_context("gateway_router", _shared_router)

    session_id = key.replace(":", "_")
    meta = _store.load_meta(session_id)
    if meta:
        transcript = _store.load_transcript(session_id)
        if transcript:
            logger.info("Restoring session %s (%d turns)", key, len(transcript))
            from caveman.agent.context import AgentContext
            ctx = AgentContext(max_tokens=loop.provider.context_length)
            restored = 0
            budget = loop.provider.context_length * 0.6
            est_tokens = 0
            for turn in transcript[-40:]:
                turn_tokens = len(turn.get("content", "")) / 3.5
                if est_tokens + turn_tokens > budget:
                    break
                ctx.add_message(turn["role"], turn["content"])
                est_tokens += turn_tokens
                restored += 1
            # Load persisted snapshot if available, fallback to meta
            import json
            snap_path = _store._session_dir(session_id) / "loop_snapshot.json"
            if snap_path.exists():
                try:
                    snap = json.loads(snap_path.read_text())
                except Exception:
                    snap = {}
            else:
                snap = {}
            snap.setdefault("turn_number", meta.turn_count)
            snap.setdefault("turn_count", meta.turn_count)
            snap.setdefault("surface", meta.surface or surface)
            loop.restore(snap, context=ctx)
            logger.info("Restored session %s: %d turns, surface=%s, prompt=%d chars",
                         key, restored, loop.surface, len(loop._system_prompt_cache or ""))
            if restored < len(transcript[-40:]):
                logger.info("Restored %d/%d turns (budget limit)", restored, min(len(transcript), 40))
    else:
        meta = SessionMeta(
            session_id=session_id,
            model=getattr(loop.provider, 'model_name', ''),
            started_at=time.time(),
            surface=surface,
        )
        _store.save_meta(meta)
        logger.info("New session: %s", key)

    session = {"loop": loop, "meta": meta, "last_active": now, "task_count": 0}
    _sessions[key] = session
    return session

async def run_gateway(config_path: str | None = None):
    config = load_config(config_path)
    gw_config = config.get("gateway", {})
    if not gw_config:
        logger.error("No gateway config found.")
        return

    async def handle_task(task: str, context: dict[str, Any]) -> str:
        key = _session_key(context)
        lock = _get_session_lock(key)

        # Enrich task with reply context if present
        reply_to = context.get("reply_to")
        if reply_to and reply_to.get("content"):
            task = f'[回复 {reply_to.get("author", "?")} 的消息: "{reply_to["content"]}"]\n{task}'

        async with lock:
            session = await _get_or_create_session(key, config_path)
            gw_name = context.get("gateway_name", "discord")
            channel_id = str(context.get("channel_id", ""))

            source_channel = {
                "gateway": gw_name, "channel_id": channel_id,
                "user_id": context.get("user_id"), "_progress_sent": 0,
            }

            session["task_count"] += 1
            logger.info("Task #%d [%s]: %s", session["task_count"], key, task[:100])

            # Detect autonomous mode
            auto_mode = bool(_AUTO_PATTERNS.search(task))
            if auto_mode:
                session.setdefault("auto_rounds", 0)

            try:
                result = await run_single_task(
                    task, session, gw_name, channel_id, source_channel,
                    _shared_router, _store, config,
                )

                # Autonomous continuation loop
                if auto_mode:
                    auto_round = 0
                    while auto_round < _AUTO_MAX_ROUNDS:
                        auto_round += 1
                        session["auto_rounds"] = auto_round

                        # Generate continuation prompt
                        cont_task = (
                            f"继续飞轮 (自动第 {auto_round}/{_AUTO_MAX_ROUNDS} 轮)。"
                            f"上一轮结果摘要：{(result or '')[:200]}。"
                            f"继续下一个最高复利的改进。完成后报告。"
                        )

                        logger.info("Auto-continue round %d/%d [%s]", auto_round, _AUTO_MAX_ROUNDS, key)
                        await _shared_router.send(
                            gw_name, channel_id,
                            f"🔄 飞轮自动继续 ({auto_round}/{_AUTO_MAX_ROUNDS})..."
                        )

                        # Reset progress counter for new round
                        source_channel["_progress_sent"] = 0

                        try:
                            result = await asyncio.wait_for(
                                run_single_task(
                                    cont_task, session, gw_name, channel_id,
                                    source_channel, _shared_router, _store, config,
                                ),
                                timeout=600.0,  # 10 min per auto-round
                            )
                        except asyncio.TimeoutError:
                            await _shared_router.send(
                                gw_name, channel_id,
                                f"⏰ 飞轮第 {auto_round} 轮超时，暂停自动模式。发消息可继续。"
                            )
                            break
                        except Exception as e:
                            logger.warning("Auto-continue round %d failed: %s", auto_round, e)
                            await _shared_router.send(
                                gw_name, channel_id,
                                f"⚠️ 飞轮第 {auto_round} 轮出错：{str(e)[:200]}。暂停自动模式。"
                            )
                            break

                    if auto_round >= _AUTO_MAX_ROUNDS:
                        await _shared_router.send(
                            gw_name, channel_id,
                            f"✅ 飞轮自动模式完成 {_AUTO_MAX_ROUNDS} 轮。发消息可继续。"
                        )

                return "" if result else "Done."

            except Exception as e:
                logger.exception("Task failed: %s", e)
                # Don't delete session — transcript is persisted, context can be restored
                return f"⚠️ Something went wrong. Please try again."

    gateways = []
    discord_cfg = gw_config.get("discord", {})
    if discord_cfg.get("enabled") and discord_cfg.get("token"):
        from caveman.gateway.discord_gw import DiscordGateway
        dg = DiscordGateway(
            token=discord_cfg["token"],
            prefix=discord_cfg.get("prefix", "!cave"),
            trigger=discord_cfg.get("trigger", "all"),
            allowed_channels=discord_cfg.get("allowed_channels"),
            allowed_users=discord_cfg.get("allowed_users"),
            locale=config.get("locale", "en"),
        )
        dg.on_task(handle_task)
        _shared_router.register(dg)
        gateways.append(("Discord", dg))

    telegram_cfg = gw_config.get("telegram", {})
    if telegram_cfg.get("enabled") and telegram_cfg.get("token"):
        from caveman.gateway.telegram_gw import TelegramGateway
        tg = TelegramGateway(
            token=telegram_cfg["token"],
            allowed_users=telegram_cfg.get("allowed_users"),
        )
        tg.on_task(handle_task)
        _shared_router.register(tg)
        gateways.append(("Telegram", tg))

    if not gateways:
        logger.error("No gateways enabled.")
        return

    logger.info("Starting %d gateway(s): %s", len(gateways), ", ".join(n for n, _ in gateways))
    tasks = [asyncio.create_task(gw.start()) for _, gw in gateways]
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for _, gw in gateways:
            try:
                await gw.stop()
            except Exception as e:
                logger.debug("Suppressed in runner: %s", e)


async def run_gateway_forever(config_path: str | None = None, max_restarts: int = 10):
    """Run gateway with auto-restart on crash. Backs off exponentially."""
    restarts = 0
    while restarts < max_restarts:
        try:
            logger.info("Gateway starting (attempt %d/%d)", restarts + 1, max_restarts)
            await run_gateway(config_path)
            logger.info("Gateway exited cleanly")
            break
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Gateway stopped by user")
            break
        except Exception as e:
            restarts += 1
            delay = min(5 * (2 ** (restarts - 1)), 120)  # 5s, 10s, 20s, ... 120s max
            logger.error("Gateway crashed (attempt %d): %s. Restarting in %ds...", restarts, e, delay)
            await asyncio.sleep(delay)
    else:
        logger.error("Gateway exceeded max restarts (%d). Giving up.", max_restarts)
