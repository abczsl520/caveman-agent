"""Anthropic provider — production-grade Claude API with streaming + tool use.

Integrates: anthropic_adapter (message conversion), error_classifier (smart retry),
retry (jittered backoff), rate_limit (header tracking), prompt_cache (cost reduction).

Ported from Hermes run_agent.py + auxiliary_client.py (MIT, Nous Research).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from caveman.providers.llm import LLMProvider, normalize_stop_reason
from caveman.providers.anthropic_adapter import (
    build_api_kwargs,
    get_max_output,
)
from caveman.providers.error_classifier import classify_error, ClassifiedError, FailoverReason
from caveman.providers.retry import jittered_backoff
from caveman.providers.rate_limit import (
    RateLimitState, parse_rate_limit_headers, format_compact,
)

logger = logging.getLogger(__name__)

# Default context windows by model family
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-3-5": 200_000,
}
_DEFAULT_CONTEXT = 200_000


class AnthropicProvider(LLMProvider):
    """Production Anthropic Claude provider.

    Features:
    - Streaming with content_block_delta events
    - Tool use (tool_call events from content_block_stop)
    - Thinking/reasoning support (adaptive + manual)
    - Smart error classification and retry
    - Rate limit header tracking
    - Message format conversion (OpenAI → Anthropic)
    - Orphan tool_use/tool_result cleanup
    - Role alternation enforcement
    - Thinking block signature management
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
        thinking: dict | None = None,
        credential_pool: Any | None = None,
    ) -> None:
        from caveman.paths import DEFAULT_MODEL
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens or get_max_output(self.model)
        self.base_url = base_url
        self.thinking = thinking
        self._client = None
        self._credential_pool = credential_pool
        self._last_usage: dict[str, int] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._rate_limit_state: RateLimitState | None = None

    @property
    def context_length(self) -> int:
        for key, val in _CONTEXT_WINDOWS.items():
            if key in self.model:
                return val
        return _DEFAULT_CONTEXT

    @property
    def usage_stats(self) -> dict[str, Any]:
        """Cumulative usage statistics."""
        stats = {
            "calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "last_usage": self._last_usage,
        }
        if self._rate_limit_state and self._rate_limit_state.has_data:
            stats["rate_limits"] = format_compact(self._rate_limit_state)
        return stats

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
            import httpx
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": 300.0,  # 5 min max (default is 10 min)
                "http_client": httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                    timeout=300.0,
                ),
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _build_params(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build API params using the adapter."""
        return build_api_kwargs(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=self.max_tokens,
            system=system,
            thinking=self.thinking,
            tool_choice=kwargs.get("tool_choice"),
            context_length=self.context_length,
        )

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        """Generate completion with smart retry on failures.

        Yields normalized events:
        - {"type": "delta", "text": "..."}
        - {"type": "thinking", "text": "..."}
        - {"type": "tool_call", "id": "...", "name": "...", "input": {...}}
        - {"type": "done", "stop_reason": "...", "usage": {...}}
        - {"type": "error", "error": "...", "action": "..."}
        """
        client = self._get_client()
        params = self._build_params(messages, system, tools, **kwargs)
        max_retries = 3

        for attempt in range(max_retries + 1):
            buffer: list[dict] = []
            success = False
            try:
                if stream:
                    async for event in self._stream(client, params):
                        buffer.append(event)
                else:
                    async for event in self._non_stream(client, params):
                        buffer.append(event)
                success = True
            except Exception as e:
                classification = classify_error(e)
                logger.warning(
                    "Anthropic API error (attempt %d/%d): %s → %s",
                    attempt + 1, max_retries + 1,
                    type(e).__name__, classification.reason.value,
                )

                if not classification.retryable or attempt >= max_retries:
                    yield {
                        "type": "error",
                        "error": str(e),
                        "action": "abort",
                        "category": classification.reason.value,
                    }
                    return

                if classification.should_compress:
                    yield {
                        "type": "error",
                        "error": "context_too_long",
                        "action": "compress",
                        "category": classification.reason.value,
                    }
                    return

                # Credential rotation on 429/401/402
                if classification.should_rotate and self._credential_pool:
                    next_cred = self._credential_pool.mark_exhausted(
                        "anthropic", self.api_key,
                        code=getattr(e, "status_code", None),
                        message=str(e)[:100],
                    )
                    if next_cred:
                        logger.info("Rotating to credential: %s", next_cred.label or next_cred.key[:8])
                        self.api_key = next_cred.key
                        if next_cred.base_url:
                            self.base_url = next_cred.base_url
                        self._client = None  # force new client with new key

                # Retry with backoff
                delay = jittered_backoff(attempt, base_delay=2.0, max_delay=60.0)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
                continue  # discard buffer, retry from scratch

            # Success — flush buffered events to caller
            if success:
                for event in buffer:
                    yield event
                return

    async def _stream(self, client: Any, params: dict) -> AsyncIterator[dict]:
        """Stream completion with event normalization."""
        async with client.messages.stream(**params) as stream:
            async for event in stream:
                etype = event.type

                if etype == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        yield {"type": "delta", "text": delta.text}
                    elif hasattr(delta, "thinking"):
                        yield {"type": "thinking", "text": delta.thinking}
                    elif hasattr(delta, "partial_json"):
                        pass  # Tool input accumulating, handled at block_stop

                elif etype == "content_block_stop":
                    msg = stream.current_message_snapshot
                    if event.index < len(msg.content):
                        block = msg.content[event.index]
                        if block.type == "tool_use":
                            yield {
                                "type": "tool_call",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }

                elif etype == "message_stop":
                    msg = stream.current_message_snapshot
                    usage = {
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens,
                    }
                    if hasattr(msg.usage, "cache_creation_input_tokens"):
                        usage["cache_creation"] = msg.usage.cache_creation_input_tokens
                    if hasattr(msg.usage, "cache_read_input_tokens"):
                        usage["cache_read"] = msg.usage.cache_read_input_tokens

                    self._record_usage(usage)

                    # Capture rate limits from stream's response
                    raw_resp = getattr(stream, "response", None)
                    if raw_resp and hasattr(raw_resp, "headers"):
                        state = parse_rate_limit_headers(
                            dict(raw_resp.headers), provider="anthropic",
                        )
                        if state:
                            self._rate_limit_state = state

                    yield {
                        "type": "done",
                        "stop_reason": normalize_stop_reason(msg.stop_reason),
                        "usage": usage,
                    }

    async def _non_stream(self, client: Any, params: dict) -> AsyncIterator[dict]:
        """Non-streaming completion."""
        response = await client.messages.create(**params)

        # Capture rate limit headers from raw response
        self._capture_rate_limits(response)

        for block in response.content:
            if block.type == "text":
                yield {"type": "delta", "text": block.text}
            elif block.type == "thinking":
                yield {"type": "thinking", "text": block.thinking}
            elif block.type == "tool_use":
                yield {
                    "type": "tool_call",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        self._record_usage(usage)
        yield {
            "type": "done",
            "stop_reason": normalize_stop_reason(response.stop_reason),
            "usage": usage,
        }

    def _record_usage(self, usage: dict) -> None:
        """Track cumulative token usage."""
        self._last_usage = usage
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)
        self._call_count += 1

    def _capture_rate_limits(self, response: Any) -> None:
        """Extract rate limit headers from an API response."""
        try:
            # Anthropic SDK: response has _response (httpx.Response) or
            # response.http_response for newer versions
            raw = getattr(response, "_response", None) or getattr(response, "http_response", None)
            if raw and hasattr(raw, "headers"):
                state = parse_rate_limit_headers(dict(raw.headers), provider="anthropic")
                if state:
                    self._rate_limit_state = state
                    logger.debug("Rate limits: %s", format_compact(state))
        except Exception:
            pass  # Rate limit tracking is best-effort

    @property
    def rate_limit_state(self) -> RateLimitState | None:
        """Current rate limit state from last API response."""
        return self._rate_limit_state
