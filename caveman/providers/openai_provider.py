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
from caveman.providers.error_classifier import classify_error
from caveman.providers.retry import jittered_backoff
from caveman.timeouts import HTTP_LLM

logger = logging.getLogger(__name__)

# Models that use max_completion_tokens instead of max_tokens
_NEW_TOKEN_PARAM_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _repair_json(raw: str) -> dict:
    """Attempt to repair malformed JSON from tool call arguments.

    Common issues: trailing commas, unquoted keys, truncated output.
    Falls back to {"raw": ...} if repair fails.
    """
    import re
    if not raw or not raw.strip():
        return {}
    # Try: strip trailing comma before }
    cleaned = re.sub(r',\s*}', '}', raw)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # intentional: Exception suppressed
    # Try: wrap in braces if it looks like key-value pairs
    if not raw.strip().startswith('{'):
        try:
            return json.loads('{' + raw + '}')
        except json.JSONDecodeError:
            pass  # intentional: Exception suppressed
    return {"raw": raw}


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with streaming tool use."""

    def __init__(self, api_key: str, model: str | None = None, max_tokens: int | None = None,
                 base_url: str | None = None, credential_pool: Any | None = None):
        from caveman.paths import DEFAULT_OPENAI_MODEL, DEFAULT_MAX_TOKENS_OPENAI
        self.api_key = api_key
        self.model = model or DEFAULT_OPENAI_MODEL
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS_OPENAI
        self.base_url = base_url
        self._client = None
        self._credential_pool = credential_pool
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._rate_limit_state = None

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
                        timeout=HTTP_LLM,
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

    def _capture_rate_limits(self, response) -> None:
        """Parse rate limit headers from HTTP response."""
        try:
            headers = getattr(response, 'headers', None)
            if not headers:
                return
            from caveman.providers.rate_limit import parse_rate_limit_headers
            state = parse_rate_limit_headers(dict(headers), provider="openai")
            if state:
                self._rate_limit_state = state
        except Exception as exc:
            logger.debug("_capture_rate_limits: suppressed %s", exc)

    @property
    def rate_limit_state(self) -> dict:
        return self._rate_limit_state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _record_usage(self, usage: dict) -> None:
        """Track cumulative token usage."""
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)
        self._call_count += 1

    def _uses_new_token_param(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(m) for m in _NEW_TOKEN_PARAM_MODELS)

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

        # Sanitize orphaned tool calls/results before API call
        from caveman.providers.message_sanitizer import sanitize_messages
        api_messages = sanitize_messages(api_messages)

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
            "stream": stream,
        }
        # o1/o3/o4 models use max_completion_tokens; others use max_tokens
        if self._uses_new_token_param():
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens

        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"

        # Request usage in streaming mode
        if stream:
            params["stream_options"] = {"include_usage": True}

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

                # Credential rotation on 429/401
                if classification.should_rotate and self._credential_pool:
                    next_cred = self._credential_pool.mark_exhausted(
                        "openai", self.api_key,
                        code=getattr(e, "status_code", None),
                        message=str(e)[:100],
                    )
                    if next_cred:
                        logger.info("Rotating to credential: %s", next_cred.label or next_cred.key[:8])
                        self.api_key = next_cred.key
                        if next_cred.base_url:
                            self.base_url = next_cred.base_url
                        self._client = None

                delay = jittered_backoff(attempt)
                await asyncio.sleep(delay)
                continue  # discard buffer, retry

            if success:
                for event in buffer:
                    yield event
                return

    async def _stream(self, client, params) -> AsyncIterator[dict]:
        tc_buf: dict[int, dict] = {}
        stream = await client.chat.completions.create(**params)
        try:
            # Capture rate limit headers from the HTTP response
            raw_response = getattr(stream, 'response', None)
            if raw_response:
                self._capture_rate_limits(raw_response)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    # Final chunk with usage only (no choices) when stream_options is set
                    if chunk.usage:
                        usage = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }
                        self._record_usage(usage)
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
                            inp = _repair_json(tc["arguments"])
                        yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "input": inp}
                    # Usage may come in this chunk or a separate final chunk
                    usage = {}
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage:
                        usage = {
                            "input_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(chunk_usage, "completion_tokens", 0) or 0,
                        }
                        self._record_usage(usage)
                    yield {"type": "done", "stop_reason": normalize_stop_reason(choice.finish_reason), "usage": usage}
        finally:
            # Ensure stream is closed even on generator exit / exception
            if hasattr(stream, 'close'):
                await stream.close()

    async def _non_stream(self, client, params) -> AsyncIterator[dict]:
        """Non-streaming: single API call."""
        p = {**params, "stream": False}
        p.pop("stream_options", None)  # not valid for non-stream
        resp = await client.chat.completions.create(**p)

        choice = resp.choices[0]
        if choice.message.content:
            yield {"type": "delta", "text": choice.message.content}
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inp = _repair_json(tc.function.arguments)
                yield {"type": "tool_call", "id": tc.id, "name": tc.function.name, "input": inp}
        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        self._record_usage(usage)
        yield {
            "type": "done",
            "stop_reason": normalize_stop_reason(choice.finish_reason),
            "usage": usage,
        }
