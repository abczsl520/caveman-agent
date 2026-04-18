"""Retry engine with jittered exponential backoff.

Ported from Hermes retry_utils.py (MIT, Nous Research).
Integrates with ErrorClassifier for smart recovery decisions.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import threading
from typing import Any, Awaitable, Callable, Optional, TypeVar

from caveman.providers.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_error,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute jittered exponential backoff delay.

    Decorrelates concurrent retries to prevent thundering herd.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)
    return delay + jitter


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    on_retry: Optional[Callable[[ClassifiedError, int, float], None]] = None,
    on_compress: Optional[Callable[[], Awaitable[None]]] = None,
    on_rotate: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    **kwargs: Any,
) -> T:
    """Retry an async function with smart error classification.

    Args:
        fn: Async function to call.
        max_retries: Maximum retry attempts.
        base_delay: Base delay for backoff.
        provider: Current provider name.
        model: Current model name.
        approx_tokens: Approximate token count.
        on_retry: Callback(classified_error, attempt, delay) before each retry.
        on_compress: Async callback to compress context when needed.
        on_rotate: Async callback to rotate credential, returns new key or None.

    Raises:
        The last exception if all retries exhausted.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):  # +1 for initial + retries
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            classified = classify_error(
                e, provider=provider, model=model,
                approx_tokens=approx_tokens,
            )

            # Non-retryable errors
            if not classified.retryable and not classified.should_rotate:
                logger.warning(
                    "Non-retryable error: %s (%s)",
                    classified.reason.value, classified.message[:200],
                )
                raise

            # Max retries exhausted
            if attempt > max_retries:
                logger.error(
                    "Max retries (%d) exhausted: %s",
                    max_retries, classified.reason.value,
                )
                raise

            # Recovery actions
            if classified.should_compress and on_compress:
                logger.info("Compressing context due to %s", classified.reason.value)
                await on_compress()

            if classified.should_rotate and on_rotate:
                new_key = await on_rotate()
                if new_key:
                    logger.info("Rotated credential for %s", provider)

            # Backoff
            delay = jittered_backoff(
                attempt, base_delay=base_delay, max_delay=max_delay,
            )

            if on_retry:
                on_retry(classified, attempt, delay)

            logger.info(
                "Retry %d/%d in %.1fs: %s (%s)",
                attempt, max_retries, delay,
                classified.reason.value, classified.message[:100],
            )
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    assert last_error is not None
    raise last_error
