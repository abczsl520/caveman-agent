"""LLM Scheduler — priority-based request queuing for shared LLM access.

Problem: Shield, Nudge, SmartCompression, Verification all need LLM calls.
Without coordination they race, waste tokens, and can exceed rate limits.

Design:
  - Priority queue: P0 (Shield) > P1 (Verification) > P2 (Nudge/Lint)
  - Rate limiting: configurable calls/minute (default 30)
  - Token budget: optional per-minute token cap
  - Async: callers await their turn, never block the event loop
  - Metrics: track calls, tokens, wait times per caller
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """LLM request priority. Lower number = higher priority."""
    CRITICAL = 0   # Shield (compaction protection)
    HIGH = 1       # Verification (anti-rationalization)
    NORMAL = 2     # SmartCompression
    LOW = 3        # Nudge, Lint (background tasks)
    BACKGROUND = 4 # Trajectory scoring, analytics


@dataclass(order=True)
class LLMRequest:
    """A queued LLM request."""
    priority: int
    submitted_at: float = field(compare=False)
    caller: str = field(compare=False)
    prompt: str = field(compare=False, repr=False)
    future: asyncio.Future = field(compare=False, repr=False)
    request_id: int = field(compare=False, default=0)
    _seq: int = field(default=0, compare=True)  # FIFO tie-breaker


@dataclass
class CallerStats:
    """Per-caller usage statistics."""
    calls: int = 0
    tokens_est: int = 0
    total_wait_ms: float = 0
    errors: int = 0
    last_call: float = 0


class LLMScheduler:
    """Priority-based LLM request scheduler with rate limiting.

    Usage:
        scheduler = LLMScheduler(llm_fn, max_rpm=30)
        result = await scheduler.request("shield", Priority.CRITICAL, prompt)
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
        max_rpm: int = 30,
        max_tokens_per_min: int = 0,  # 0 = unlimited
        max_queue_size: int = 100,
    ):
        self._llm_fn = llm_fn
        self._max_rpm = max_rpm
        self._max_tpm = max_tokens_per_min
        self._max_queue = max_queue_size

        self._queue: asyncio.PriorityQueue[LLMRequest] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._seq = 0
        self._stats: dict[str, CallerStats] = {}
        self._call_timestamps: list[float] = []
        self._token_timestamps: list[tuple[float, int]] = []
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._total_calls = 0
        self._total_errors = 0

    async def start(self) -> None:
        """Start the scheduler worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("LLM Scheduler started (max_rpm=%d)", self._max_rpm)

    async def stop(self) -> None:
        """Stop the scheduler, cancelling pending requests."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if not req.future.done():
                    req.future.cancel()
            except asyncio.QueueEmpty:
                break

    async def request(
        self,
        caller: str,
        priority: Priority,
        prompt: str,
        timeout: float = 60.0,
    ) -> str:
        """Submit an LLM request and wait for the result."""
        if not self._running:
            await self.start()

        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()

        self._seq += 1
        req = LLMRequest(
            priority=priority,
            submitted_at=time.monotonic(),
            caller=caller,
            prompt=prompt,
            future=future,
            request_id=self._seq,
            _seq=self._seq,
        )

        try:
            self._queue.put_nowait(req)
        except asyncio.QueueFull:
            raise RuntimeError(
                f"LLM scheduler queue full ({self._max_queue}). "
                f"Caller: {caller}, priority: {priority.name}"
            )

        logger.debug(
            "Queued LLM request #%d from %s (priority=%s, queue_size=%d)",
            req.request_id, caller, priority.name, self._queue.qsize(),
        )

        return await asyncio.wait_for(future, timeout=timeout)

    async def _worker(self) -> None:
        """Background worker that processes the priority queue."""
        while self._running:
            try:
                req = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if req.future.done():
                continue

            await self._wait_for_rate_limit()

            wait_ms = (time.monotonic() - req.submitted_at) * 1000
            try:
                result = await self._llm_fn(req.prompt)
                if not req.future.done():
                    req.future.set_result(result)
                self._record_call(req.caller, len(req.prompt) // 4, wait_ms)
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)
                self._record_error(req.caller)
                logger.warning(
                    "LLM request #%d failed (caller=%s): %s",
                    req.request_id, req.caller, e,
                )

    async def _wait_for_rate_limit(self) -> None:
        """Wait until we're within rate limits."""
        now = time.monotonic()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60
        ]
        while len(self._call_timestamps) >= self._max_rpm:
            oldest = self._call_timestamps[0]
            wait = 60 - (now - oldest) + 0.1
            if wait > 0:
                await asyncio.sleep(wait)
            now = time.monotonic()
            self._call_timestamps = [
                t for t in self._call_timestamps if now - t < 60
            ]
        if self._max_tpm > 0:
            self._token_timestamps = [
                (t, n) for t, n in self._token_timestamps if now - t < 60
            ]
            tokens_used = sum(n for _, n in self._token_timestamps)
            while tokens_used >= self._max_tpm:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                self._token_timestamps = [
                    (t, n) for t, n in self._token_timestamps if now - t < 60
                ]
                tokens_used = sum(n for _, n in self._token_timestamps)
        self._call_timestamps.append(time.monotonic())

    def _record_call(self, caller: str, tokens_est: int, wait_ms: float) -> None:
        stats = self._stats.setdefault(caller, CallerStats())
        stats.calls += 1
        stats.tokens_est += tokens_est
        stats.total_wait_ms += wait_ms
        stats.last_call = time.monotonic()
        self._total_calls += 1
        if self._max_tpm > 0:
            self._token_timestamps.append((time.monotonic(), tokens_est))

    def _record_error(self, caller: str) -> None:
        stats = self._stats.setdefault(caller, CallerStats())
        stats.errors += 1
        self._total_errors += 1

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def get_stats(self, caller: str | None = None) -> dict:
        """Get usage statistics."""
        if caller:
            s = self._stats.get(caller, CallerStats())
            return {
                "caller": caller,
                "calls": s.calls,
                "tokens_est": s.tokens_est,
                "avg_wait_ms": s.total_wait_ms / max(s.calls, 1),
                "errors": s.errors,
            }
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "queue_size": self.queue_size,
            "max_rpm": self._max_rpm,
            "callers": {
                name: {
                    "calls": s.calls,
                    "tokens_est": s.tokens_est,
                    "avg_wait_ms": s.total_wait_ms / max(s.calls, 1),
                    "errors": s.errors,
                }
                for name, s in self._stats.items()
            },
        }
