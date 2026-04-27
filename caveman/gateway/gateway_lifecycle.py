"""Gateway process lifecycle — startup, SIGUSR1 restart, drain, and shutdown.

Orchestrates the full gateway process lifecycle:
1. Environment sanitization
2. Restart sentinel check (was this a restart?)
3. PID file management
4. SIGUSR1 handler for graceful restart
5. Drain active sessions before restart
6. Crash recovery with exponential backoff

Separated from runner.py (session management) for single-responsibility.
"""
from __future__ import annotations

import asyncio
import os
import sys
import logging

logger = logging.getLogger("caveman.gateway")


async def drain_active_sessions(
    sessions: dict[str, dict],
    session_locks: dict[str, asyncio.Lock],
    timeout: float,
    force_stop: callable | None = None,
) -> tuple[int, bool]:
    """Wait for active sessions to finish before restart.

    Args:
        sessions: The runner's session dict.
        session_locks: The runner's per-session lock dict.
        timeout: Max seconds to wait.
        force_stop: Optional predicate checked while draining. When it returns
            true, drain exits immediately so a second restart signal can force
            re-exec even if the active session is the one requesting restart.

    Returns:
        (active_count_at_start, timed_out)
    """
    active = [k for k, s in sessions.items() if s.get("loop") and
              session_locks.get(k) and session_locks[k].locked()]
    if not active:
        return 0, False

    logger.info("Draining %d active session(s), timeout=%.0fs", len(active), timeout)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if force_stop and force_stop():
            logger.warning("Drain interrupted by force restart request")
            return len(active), True
        still_active = [k for k in active if
                        session_locks.get(k) and session_locks[k].locked()]
        if not still_active:
            logger.info("All sessions drained")
            return len(active), False
        await asyncio.sleep(0.5)

    still = [k for k in active if session_locks.get(k) and session_locks[k].locked()]
    logger.warning("Drain timed out, %d session(s) still active", len(still))
    return len(active), True


# Module-level restart state (accessed by signal handler)
_restart_requested = False
_gateway_stopping = False


async def run_gateway_forever(config_path: str | None = None, max_restarts: int = 10) -> None:
    """Run gateway with auto-restart on crash. Backs off exponentially.

    Lifecycle:
    1. Sanitize environment
    2. Check for restart sentinel (notify user if restarted)
    3. Write PID file
    4. Install SIGUSR1 handler for graceful restart
    5. Run gateway
    6. On SIGUSR1: drain → write sentinel → exit 75
    7. On crash: exponential backoff retry
    8. Cleanup PID file on exit
    """
    global _restart_requested, _gateway_stopping

    from caveman.runtime_identity import sanitize_environment
    sanitize_environment()

    from caveman.safe_stdio import install_safe_stdio
    install_safe_stdio()

    from caveman.gateway.status import (
        write_pid_file, remove_pid_file, write_runtime_state,
        get_running_pid,
    )
    from caveman.gateway.restart import (
        RESTART_EXIT_CODE, DEFAULT_DRAIN_TIMEOUT,
        consume_restart_sentinel, write_restart_sentinel,
    )
    from caveman.gateway.runner import run_gateway, _get_server

    # Check if we were restarted (sentinel from previous process)
    sentinel = consume_restart_sentinel()
    if sentinel:
        logger.info("Restarted: kind=%s reason=%s", sentinel.get("kind"), sentinel.get("reason"))

    # Cleanup old tool result files (>24h)
    try:
        from caveman.tools.result_storage import cleanup_old_results
        removed = cleanup_old_results(max_age_hours=24)
        if removed:
            logger.info("Cleaned up %d old tool result files", removed)
    except Exception as e:
        logger.debug("Tool result cleanup failed: %s", e)

    # Guard: kill stale gateway if still running
    existing_pid = get_running_pid()
    if existing_pid and existing_pid != os.getpid():
        logger.warning("Existing gateway PID %d found, sending SIGTERM", existing_pid)
        try:
            os.kill(existing_pid, 15)  # SIGTERM
            import time
            time.sleep(2)
            if get_running_pid() == existing_pid:
                logger.warning("PID %d still alive, sending SIGKILL", existing_pid)
                os.kill(existing_pid, 9)
                time.sleep(0.5)
        except ProcessLookupError:
            pass
        remove_pid_file()

    # Write PID file
    write_pid_file()
    write_runtime_state(state="starting")

    # Register our PID with bash tool's self-kill protection so the
    # agent cannot kill its own gateway via `kill <PID>`.
    try:
        from caveman.tools.builtin.bash import register_gateway_pid
        register_gateway_pid()
        logger.info("Self-kill protection registered for PID %d", os.getpid())
    except Exception as e:
        logger.error("Failed to register self-kill protection: %s", e)

    # Install SIGUSR1 handler for graceful restart
    import signal as _signal

    _sigusr1_force = False

    def _sigusr1_handler():
        nonlocal _sigusr1_force
        global _restart_requested
        if _restart_requested:
            logger.warning("Second SIGUSR1 — force restart (skip drain)")
            _sigusr1_force = True
            return
        _restart_requested = True
        logger.info("SIGUSR1 received — initiating graceful restart")

    def _sigusr2_handler():
        """Treat SIGUSR2 as a full restart request, not in-process hot reload.

        Broad importlib.reload of caveman.* is unsafe in this gateway: long-lived
        sessions, registries, callbacks, and Enum instances survive while module
        classes are replaced, creating half-old/half-new process state.  A full
        re-exec is the stable boundary for loading code changes.
        """
        _sigusr1_handler()
        logger.info("SIGUSR2 received — requesting full restart; in-process hot-reload disabled")

    loop = asyncio.get_running_loop()
    if hasattr(_signal, "SIGUSR1"):
        try:
            loop.add_signal_handler(_signal.SIGUSR1, _sigusr1_handler)
            logger.info("SIGUSR1 handler installed for graceful restart")
        except NotImplementedError as exc:
            logger.debug("_sigusr1_handler: suppressed %s", exc)
    if hasattr(_signal, "SIGUSR2"):
        try:
            loop.add_signal_handler(_signal.SIGUSR2, _sigusr2_handler)
            logger.info("SIGUSR2 handler installed for full restart")
        except NotImplementedError as exc:
            logger.debug("_sigusr2_handler: suppressed %s", exc)

    def _shutdown_handler():
        global _gateway_stopping
        _gateway_stopping = True
        logger.info("Shutdown signal received")

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            pass  # intentional: NotImplementedError suppressed

    restarts = 0
    exit_code = 0

    # Start config watcher for hot reload
    from caveman.config.watcher import ConfigWatcher
    config_watcher = ConfigWatcher(config_path)

    async def _on_config_change(new_config):
        srv = _get_server()
        locale = new_config.get("locale", "en")
        for gw in srv.router._gateways.values():
            if hasattr(gw, 'locale'):
                gw.locale = locale
        # Invalidate system prompt cache on all active sessions
        for session in srv.sessions.values():
            loop = session.get("loop")
            if loop and hasattr(loop, 'invalidate_system_prompt'):
                loop.invalidate_system_prompt()
        logger.info("Hot config reload applied (locale=%s, %d sessions invalidated)",
                     locale, len(srv.sessions))

    config_watcher.on_change(_on_config_change)
    config_watcher.start()

    # Start health check server
    from caveman.gateway.health import HealthServer
    health_server = HealthServer(
        status_fn=lambda: {
            "gateway": "running",
            "sessions": len(_get_server().sessions),
            "restarts": restarts,
            "usage": _get_server().usage_tracker.summary(),
        }
    )
    await health_server.start()

    try:
        while restarts < max_restarts:
            _restart_requested = False
            _gateway_stopping = False

            try:
                logger.info("Gateway starting (attempt %d/%d)", restarts + 1, max_restarts)
                write_runtime_state(state="connecting")

                gateway_task = asyncio.create_task(run_gateway(config_path))

                while not gateway_task.done():
                    if _restart_requested:
                        if _sigusr1_force:
                            logger.info("Force restart: skipping drain")
                            write_runtime_state(state="force_restart", restart_requested=True)
                            active, timed_out = 0, False
                        else:
                            logger.info("Graceful restart: draining active sessions...")
                            write_runtime_state(state="draining", restart_requested=True)

                            srv = _get_server()
                            active, timed_out = await drain_active_sessions(
                                srv.sessions, srv.session_locks, DEFAULT_DRAIN_TIMEOUT,
                                force_stop=lambda: _sigusr1_force,
                            )
                            if timed_out:
                                logger.warning("Drain timed out with active sessions")

                        write_restart_sentinel(kind="restart", reason="SIGUSR1 graceful restart")

                        gateway_task.cancel()
                        try:
                            await gateway_task
                        except (asyncio.CancelledError, Exception):
                            pass  # intentional: Exception suppressed

                        logger.info("Graceful restart complete, exec-replacing process")
                        # Clean up before exec
                        config_watcher.stop()
                        await health_server.stop()
                        remove_pid_file()
                        # Re-exec ourselves — loads fresh code from disk
                        os.execv(sys.executable, [sys.executable, "-m", "caveman", "serve"])

                    if _gateway_stopping:
                        gateway_task.cancel()
                        try:
                            await gateway_task
                        except (asyncio.CancelledError, Exception):
                            pass  # intentional: Exception suppressed
                        logger.info("Gateway stopped by user")
                        return

                    await asyncio.sleep(0.5)

                await gateway_task
                logger.info("Gateway exited cleanly")
                break

            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Gateway stopped by user")
                break
            except Exception as e:
                restarts += 1
                delay = min(5 * (2 ** (restarts - 1)), 120)
                logger.error("Gateway crashed (attempt %d): %s. Restarting in %ds...",
                             restarts, e, delay)
                write_runtime_state(state="crashed", exit_reason=str(e))
                await asyncio.sleep(delay)
        else:
            logger.error("Gateway exceeded max restarts (%d). Giving up.", max_restarts)
            exit_code = 1
    finally:
        config_watcher.stop()
        await health_server.stop()
        write_runtime_state(state="stopped")
        remove_pid_file()
        logger.info("PID file removed, gateway fully stopped")

        if exit_code == RESTART_EXIT_CODE:
            import sys as _sys
            _sys.exit(RESTART_EXIT_CODE)
