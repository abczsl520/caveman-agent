"""Gateway wiring — connects all gateway subsystems into the live server.

Called once during GatewayServer.start() to wire orphan modules.
Each subsystem is imported in a try/except so failures are non-fatal.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from caveman.gateway.runner import GatewayServer

logger = logging.getLogger(__name__)


def wire_gateway(server: GatewayServer) -> dict[str, Any]:
    """Wire all gateway subsystems. Returns dict of wired components."""
    wired: dict[str, Any] = {}
    failed: list[str] = []

    def _wire(name: str, fn):
        try:
            result = fn()
            if result is not None:
                wired[name] = result
            logger.debug("Wired: %s", name)
        except Exception as e:
            failed.append(f"{name}: {e}")
            logger.debug("Skip %s: %s", name, e)

    # --- Message processing pipeline ---

    _wire("processor", lambda: _wire_processor(server))
    _wire("stream_consumer", lambda: _wire_stream_consumer(server))
    _wire("outbound", lambda: _wire_outbound(server))
    _wire("debounce", lambda: _wire_debounce(server))

    # --- Session management ---

    _wire("session_manager", lambda: _wire_session_manager(server))
    _wire("session_commands", lambda: _wire_session_commands(server))
    _wire("threading", lambda: _wire_threading(server))

    # --- Security & access ---

    _wire("access_control", lambda: _wire_access_control(server))
    _wire("allowlist_commands", lambda: _wire_allowlist_commands(server))
    _wire("rate_limiter", lambda: _wire_rate_limiter(server))

    # --- Context & memory ---

    _wire("context_pruning", lambda: _wire_context_pruning(server))
    _wire("agent_memory", lambda: _wire_agent_memory(server))

    # --- Model & dispatch ---

    _wire("model_selector", lambda: _wire_model_selector(server))
    _wire("dispatch", lambda: _wire_dispatch(server))
    _wire("command_registry", lambda: _wire_command_registry(server))
    _wire("directives", lambda: _wire_directives(server))

    # --- Infrastructure ---

    _wire("execution", lambda: _wire_execution(server))
    _wire("agent_runner", lambda: _wire_agent_runner(server))
    _wire("event_router", lambda: _wire_event_router(server))
    _wire("status_panel", lambda: _wire_status_panel(server))
    _wire("webhook", lambda: _wire_webhook(server))

    # --- Media & content ---

    _wire("attachments", lambda: _wire_attachments(server))
    _wire("media_cache", lambda: _wire_media_cache(server))
    _wire("interactive", lambda: _wire_interactive(server))
    _wire("message_splitting", lambda: _wire_message_splitting(server))

    # --- Platform ---

    _wire("pairing", lambda: _wire_pairing(server))
    _wire("mirror", lambda: _wire_mirror(server))
    _wire("channel_directory", lambda: _wire_channel_directory(server))
    _wire("acp_lifecycle", lambda: _wire_acp_lifecycle(server))

    # --- Legacy base ---
    _wire("base", lambda: _wire_base(server))

    if failed:
        logger.info("Gateway wiring: %d/%d ok, %d skipped",
                     len(wired), len(wired) + len(failed), len(failed))
    else:
        logger.info("Gateway wiring: all %d subsystems ok", len(wired))

    return wired


# ── Individual wiring functions ──────────────────────────────────────────

def _wire_processor(server) -> Any:
    from caveman.gateway.processor import MessageProcessor
    proc = MessageProcessor()
    server._processor = proc
    return proc


def _wire_stream_consumer(server) -> Any:
    from caveman.gateway.stream_consumer import StreamConsumer
    consumer = StreamConsumer()
    server._stream_consumer = consumer
    return consumer


def _wire_outbound(server) -> Any:
    from caveman.gateway.outbound import OutboundDelivery
    queue = OutboundDelivery(router=server.router)
    server._outbound = queue
    return queue


def _wire_debounce(server) -> Any:
    from caveman.gateway.debounce import MessageDebouncer
    pool = MessageDebouncer()
    server._debounce = pool
    return pool


def _wire_session_manager(server) -> Any:
    from caveman.gateway.session_manager import GatewaySessionManager
    mgr = GatewaySessionManager()
    server._session_manager = mgr
    return mgr


def _wire_session_commands(server) -> Any:
    from caveman.gateway.session_commands import SessionCommandHandler
    handler = SessionCommandHandler()
    server._session_commands = handler
    return handler


def _wire_threading(server) -> Any:
    from caveman.gateway.threading import ThreadManager
    mgr = ThreadManager()
    server._thread_manager = mgr
    return mgr


def _wire_access_control(server) -> Any:
    from caveman.gateway.access_control import AccessController
    ctrl = AccessController()
    server._access_control = ctrl
    return ctrl


def _wire_allowlist_commands(server) -> Any:
    from caveman.gateway.allowlist_commands import AllowlistManager
    handler = AllowlistManager()
    server._allowlist_commands = handler
    return handler


def _wire_rate_limiter(server) -> Any:
    from caveman.gateway.rate_limiter import RateLimiter
    limiter = RateLimiter()
    server._rate_limiter = limiter
    return limiter


def _wire_context_pruning(server) -> Any:
    from caveman.gateway.context_pruning import ContextPruner
    pruner = ContextPruner()
    server._context_pruner = pruner
    return pruner


def _wire_agent_memory(server) -> Any:
    from caveman.gateway.agent_memory import AgentMemoryManager
    mgr = AgentMemoryManager()
    server._agent_memory = mgr
    return mgr


def _wire_model_selector(server) -> Any:
    from caveman.gateway.model_selector import ModelSelector
    sel = ModelSelector()
    server._model_selector = sel
    return sel


def _wire_dispatch(server) -> Any:
    from caveman.gateway.dispatch import DispatchContext
    router = DispatchContext()
    server._dispatch_router = router
    return router


def _wire_command_registry(server) -> Any:
    from caveman.gateway.command_registry import CommandArg
    reg = CommandArg
    server._gw_command_registry = reg
    return reg


def _wire_directives(server) -> Any:
    from caveman.gateway.directives import ParsedDirectives
    proc = ParsedDirectives()
    server._directives = proc
    return proc


def _wire_execution(server) -> Any:
    from caveman.gateway.execution import ExecutionConfig
    mgr = ExecutionConfig()
    server._execution = mgr
    return mgr


def _wire_agent_runner(server) -> Any:
    from caveman.gateway.agent_runner import AgentRunner
    runner = AgentRunner()
    server._agent_runner = runner
    return runner


def _wire_event_router(server) -> Any:
    from caveman.gateway.event_router import EventRouter
    router = EventRouter()
    server._event_router = router
    return router


def _wire_status_panel(server) -> Any:
    from caveman.gateway.status_panel import SessionStatus
    panel = SessionStatus()
    server._status_panel = panel
    return panel


def _wire_webhook(server) -> Any:
    from caveman.gateway.webhook import WebhookManager
    mgr = WebhookManager()
    server._webhook = mgr
    return mgr


def _wire_attachments(server) -> Any:
    from caveman.gateway.attachments import AttachmentHandler
    proc = AttachmentHandler()
    server._attachment_processor = proc
    return proc


def _wire_media_cache(server) -> Any:
    from caveman.gateway.media_cache import MediaCache
    cache = MediaCache()
    server._media_cache = cache
    return cache


def _wire_interactive(server) -> Any:
    from caveman.gateway.interactive import InteractiveMessage
    server._interactive_cls = InteractiveMessage
    return InteractiveMessage


def _wire_message_splitting(server) -> Any:
    from caveman.gateway.message_splitting import split_message
    server._split_message = split_message
    return split_message


def _wire_pairing(server) -> Any:
    from caveman.gateway.pairing import PairingManager
    mgr = PairingManager()
    server._pairing = mgr
    return mgr


def _wire_mirror(server) -> Any:
    from caveman.gateway.mirror import mirror_to_session
    server._mirror_fn = mirror_to_session
    return mirror_to_session


def _wire_channel_directory(server) -> Any:
    from caveman.gateway.channel_directory import ChannelDirectory
    directory = ChannelDirectory()
    server._channel_directory = directory
    return directory


def _wire_acp_lifecycle(server) -> Any:
    from caveman.gateway.acp_lifecycle import ACPLifecycleManager
    mgr = ACPLifecycleManager()
    server._acp_lifecycle = mgr
    return mgr


def _wire_base(server) -> Any:
    from caveman.gateway.base import Gateway
    server._gateway_base_cls = Gateway
    return Gateway
