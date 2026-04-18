"""OpenAI provider — GPT/compatible API with streaming + tool use.

Supports OpenAI, Azure, and any OpenAI-compatible endpoint.
Integrates error_classifier for smart retry decisions.
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import AsyncIterator, Any
from .llm import LLMProvider, normalize_stop_reason
from caveman.providers.error_classifier import classify_error, ClassifiedError
from caveman.providers.retry import jittered_backoff

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with streaming tool use."""

    def __init__(self, api_key: str, model: str | None = None, max_tokens: int | None = None, base_url: str | None = None):
        from caveman.paths import DEFAULT_OPENAI_MODEL, DEFAULT_MAX_TOKENS_OPENAI
        self.api_key = api_key
        self.model = model or DEFAULT_OPENAI_MODEL
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS_OPENAI
        self.base_url = base_url
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    @property
    def context_length(self) -> int:
        from caveman.paths import OPENAI_CONTEXT_WINDOW
        return OPENAI_CONTEXT_WINDOW

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                import httpx
                kwargs: dict[str, Any] = {
                    "api_key": self.api_key,
                    "timeout": 300.0,  # 5 min max
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
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
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
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build API params dict. Shared by stream/non-stream."""
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        openai_tools = None
        if tools:
            openai_tools = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("input_schema", {}),
                },
            } for t in tools]

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"
        return params

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        system: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Generate completion with smart retry on failures."""
        client = self._get_client()
        params = self._build_params(messages, system, tools, stream)
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
                    "OpenAI API error (attempt %d/%d): %s → %s",
                    attempt + 1, max_retries + 1,
                    type(e).__name__, classification.reason.value,
                )
                if not classification.retryable or attempt >= max_retries:
                    yield {"type": "error", "error": str(e), "action": "abort"}
                    return
                if classification.should_compress:
                    yield {"type": "error", "error": "context_too_long", "action": "compress"}
                    return
                delay = jittered_backoff(attempt)
                await asyncio.sleep(delay)
                continue  # discard buffer, retry

            if success:
                for event in buffer:
                    yield event
                return

    async def _stream(self, client, params) -> AsyncIterator[dict]:
        tc_buf: dict[int, dict] = {}
        async with client.chat.completions.create(**params) as stream:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta
                if delta.content:
                    yield {"type": "delta", "text": delta.content}
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_buf:
                            tc_buf[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tc_buf[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tc_buf[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tc_buf[idx]["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    for tc in tc_buf.values():
                        try:
                            inp = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            inp = {"raw": tc["arguments"]}
                        yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "input": inp}
                    yield {"type": "done", "stop_reason": normalize_stop_reason(choice.finish_reason), "usage": {}}

    async def _non_stream(self, client, params) -> AsyncIterator[dict]:
        """Non-streaming: single API call."""
        p = {**params, "stream": False}
        resp = await client.chat.completions.create(**p)

        choice = resp.choices[0]
        if choice.message.content:
            yield {"type": "delta", "text": choice.message.content}
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inp = {"raw": tc.function.arguments}
                yield {"type": "tool_call", "id": tc.id, "name": tc.function.name, "input": inp}
        yield {
            "type": "done",
            "stop_reason": normalize_stop_reason(choice.finish_reason),
            "usage": {
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        }
