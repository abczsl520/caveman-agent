"""Gemini provider — Google Generative AI REST API with streaming + tool use.

Uses httpx to call the Gemini REST API directly (no SDK dependency).
Integrates error_classifier for smart retry decisions.
"""
from __future__ import annotations

__all__ = ["GeminiProvider"]

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

from caveman.providers.llm import LLMProvider, normalize_stop_reason
from caveman.providers.error_classifier import classify_error
from caveman.providers.retry import jittered_backoff

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
}
_MAX_OUTPUT: dict[str, int] = {
    "gemini-2.5-pro": 65_536,
    "gemini-2.5-flash": 65_536,
    "gemini-2.0-flash": 8_192,
}
_DEFAULT_CONTEXT = 1_048_576
_DEFAULT_MAX_OUTPUT = 8_192


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the REST API.

    Features:
    - Streaming via streamGenerateContent
    - Tool use (function calling)
    - Smart error classification and retry
    - No SDK dependency — uses httpx directly
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._base_url = (base_url or _BASE_URL).rstrip("/")
        self.max_tokens = max_tokens or _MAX_OUTPUT.get(model, _DEFAULT_MAX_OUTPUT)
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    @property
    def context_length(self) -> int:
        for key, val in _CONTEXT_WINDOWS.items():
            if key in self.model:
                return val
        return _DEFAULT_CONTEXT

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=300.0,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
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
        """Convert OpenAI-format messages/tools to Gemini API format."""
        params: dict[str, Any] = {
            "contents": _convert_messages(messages),
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system:
            params["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            params["tools"] = _convert_tools(tools)
        return params

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        """Generate completion. Yields normalized events (delta/tool_call/done)."""
        if not self.api_key:
            yield {"type": "error", "error": "Gemini API key not configured. Set GEMINI_API_KEY.", "action": "abort"}
            return

        params = self._build_params(messages, system, tools, **kwargs)
        max_retries = 3

        for attempt in range(max_retries + 1):
            buffer: list[dict] = []
            success = False
            try:
                if stream:
                    async for event in self._stream(params):
                        buffer.append(event)
                else:
                    async for event in self._non_stream(params):
                        buffer.append(event)
                success = True
            except Exception as e:
                classification = classify_error(e)
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s → %s",
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
                continue

            if success:
                for event in buffer:
                    yield event
                return

    async def _stream(self, params: dict) -> AsyncIterator[dict]:
        """Stream via Gemini streamGenerateContent."""
        client = self._get_client()
        url = f"{self._base_url}/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"

        async with client.stream("POST", url, json=params) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    events = _parse_response(data)
                    for event in events:
                        yield event
                except json.JSONDecodeError:
                    continue

    async def _non_stream(self, params: dict) -> AsyncIterator[dict]:
        """Non-streaming: single API call."""
        client = self._get_client()
        url = f"{self._base_url}/models/{self.model}:generateContent?key={self.api_key}"
        resp = await client.post(url, json=params)
        resp.raise_for_status()
        data = resp.json()

        events = _parse_response(data)
        for event in events:
            yield event

        # Record usage
        usage_meta = data.get("usageMetadata", {})
        self._call_count += 1
        self._total_input_tokens += usage_meta.get("promptTokenCount", 0)
        self._total_output_tokens += usage_meta.get("candidatesTokenCount", 0)


# --- Conversion helpers ---

def _convert_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Gemini contents format."""
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        # Gemini uses "user" and "model" roles
        gemini_role = "model" if role == "assistant" else "user"

        content = msg.get("content", "")
        parts = []

        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            # Multi-part content (e.g., text + images)
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        # Gemini inline image format
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:image/png;base64,xxx
                            mime_end = url.index(";")
                            mime = url[5:mime_end]
                            b64 = url[url.index(",") + 1:]
                            parts.append({
                                "inlineData": {"mimeType": mime, "data": b64}
                            })

        # Handle tool results
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            parts = [{"functionResponse": {
                "name": tool_call_id,
                "response": {"result": content},
            }}]

        # Handle assistant tool calls
        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if content:
                parts.append({"text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args_dict = {"raw": args}
                parts.append({"functionCall": {
                    "name": fn.get("name", ""),
                    "args": args_dict,
                }})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Caveman tool definitions to Gemini function declarations."""
    declarations = []
    for t in tools:
        decl: dict[str, Any] = {
            "name": t["name"],
            "description": t.get("description", ""),
        }
        schema = t.get("input_schema", {})
        if schema:
            # Gemini expects "parameters" as an OpenAPI-style schema
            decl["parameters"] = schema
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


def _parse_response(data: dict) -> list[dict]:
    """Parse Gemini response into normalized events."""
    events = []
    candidates = data.get("candidates", [])
    if not candidates:
        return events

    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])

    for part in parts:
        if "text" in part:
            events.append({"type": "delta", "text": part["text"]})
        elif "functionCall" in part:
            fc = part["functionCall"]
            events.append({
                "type": "tool_call",
                "id": fc.get("name", ""),  # Gemini doesn't have separate IDs
                "name": fc.get("name", ""),
                "input": fc.get("args", {}),
            })

    # Check for finish reason
    finish_reason = candidate.get("finishReason", "")
    if finish_reason:
        usage_meta = data.get("usageMetadata", {})
        events.append({
            "type": "done",
            "stop_reason": normalize_stop_reason(finish_reason),
            "usage": {
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
            },
        })

    return events


def _normalize_finish_reason(reason: str) -> str:
    """Normalize Gemini finish reasons to our standard format."""
    mapping = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "safety",
        "RECITATION": "recitation",
        "OTHER": "end_turn",
    }
    return mapping.get(reason, reason.lower())
