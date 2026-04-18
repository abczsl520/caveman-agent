"""Ollama LLM Provider — local model inference.

Connects to a local Ollama instance for:
  1. Running fine-tuned embedding models (training loop closure)
  2. Running local LLMs for cost-free inference
  3. Offline operation

Uses the OpenAI-compatible API that Ollama exposes at /v1/chat/completions.
"""
from __future__ import annotations

__all__ = ["OllamaProvider"]

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from caveman.providers.llm import LLMProvider, normalize_stop_reason
from caveman.providers.error_classifier import classify_error
from caveman.providers.retry import jittered_backoff

logger = logging.getLogger(__name__)

from caveman.paths import DEFAULT_OLLAMA_URL
DEFAULT_MODEL = "llama3.2"


class OllamaProvider(LLMProvider):
    """Ollama LLM provider using the OpenAI-compatible API.

    Inherits from LLMProvider and implements the abstract interface.
    Backward-compatible: still supports the old prompt-based complete()/stream() API.

    Usage:
        provider = OllamaProvider(model="llama3.2")

        # New unified API (yields normalized events)
        async for event in provider.complete(messages=[{"role": "user", "content": "hi"}]):
            print(event)

        # Legacy API still works
        text = await provider.complete_text("What is Python?")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    @property
    def context_length(self) -> int:
        # Most Ollama models default to 128K; specific overrides can be added
        return 128_000

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)
        return self._client

    def _build_params(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
        }

        if tools:
            params["tools"] = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("input_schema", {}),
                },
            } for t in tools]
            params["tool_choice"] = "auto"

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
                    "Ollama API error (attempt %d/%d): %s → %s",
                    attempt + 1, max_retries + 1,
                    type(e).__name__, classification.reason.value,
                )
                if not classification.retryable or attempt >= max_retries:
                    yield {"type": "error", "error": str(e), "action": "abort"}
                    return
                delay = jittered_backoff(attempt)
                await asyncio.sleep(delay)
                continue

            if success:
                for event in buffer:
                    yield event
                return

    async def _stream(self, params: dict) -> AsyncIterator[dict]:
        """Stream via OpenAI-compatible endpoint."""
        client = self._get_client()
        stream_params = {**params, "stream": True}
        tc_buf: dict[int, dict] = {}

        async with client.stream(
            "POST", "/v1/chat/completions", json=stream_params,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        yield {"type": "delta", "text": delta["content"]}
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tc_buf:
                                tc_buf[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.get("id"):
                                tc_buf[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tc_buf[idx]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tc_buf[idx]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        for tc in tc_buf.values():
                            try:
                                inp = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                inp = {"raw": tc["arguments"]}
                            yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "input": inp}
                        usage = data.get("usage", {})
                        yield {
                            "type": "done",
                            "stop_reason": normalize_stop_reason(choice["finish_reason"]),
                            "usage": {
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                            },
                        }
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def _non_stream(self, params: dict) -> AsyncIterator[dict]:
        """Non-streaming completion."""
        client = self._get_client()
        non_stream_params = {**params, "stream": False}
        resp = await client.post("/v1/chat/completions", json=non_stream_params)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        if msg.get("content"):
            yield {"type": "delta", "text": msg["content"]}
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    inp = {"raw": fn.get("arguments", "")}
                yield {"type": "tool_call", "id": tc["id"], "name": fn["name"], "input": inp}

        usage = data.get("usage", {})
        self._call_count += 1
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)
        yield {
            "type": "done",
            "stop_reason": normalize_stop_reason(choice.get("finish_reason", "stop")),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    # --- Legacy API (backward-compatible) ---

    async def complete_text(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Legacy: send a prompt, get a string back."""
        messages = [{"role": "user", "content": prompt}]
        result = []
        async for event in self.complete(
            messages=messages, system=system, stream=False,
            temperature=temperature, max_tokens=max_tokens,
        ):
            if event.get("type") == "delta":
                result.append(event.get("text", ""))
        return "".join(result)

    async def stream_text(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Legacy: stream text chunks."""
        messages = [{"role": "user", "content": prompt}]
        async for event in self.complete(
            messages=messages, system=system, stream=True,
            temperature=temperature, max_tokens=max_tokens,
        ):
            if event.get("type") == "delta":
                yield event.get("text", "")

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models on the Ollama instance."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []
