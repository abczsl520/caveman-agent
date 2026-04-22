"""LLM Router — unified interface for calling different LLM providers.

Provides a single call_llm() function that routes to the appropriate
provider based on model name. Extracted from Hermes LLM routing.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "LLMResponse",
    "LLMConfig",
    "resolve_provider",
    "call_llm",
]


logger = logging.getLogger("caveman.tools.llm_router")


@dataclass
class LLMResponse:
    """Unified LLM response."""
    text: str = ""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0
    finish_reason: str = ""
    raw: Optional[Any] = None


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str
    model: str
    api_key_env: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) if self.api_key_env else True


# ── Provider Registry ──

PROVIDER_CONFIGS: Dict[str, LLMConfig] = {
    "claude": LLMConfig("anthropic", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
    "gpt": LLMConfig("openai", "gpt-4o", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "gemini": LLMConfig("google", "gemini-2.5-pro", "GOOGLE_API_KEY"),
    "deepseek": LLMConfig("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
}


def resolve_provider(model: str) -> Optional[LLMConfig]:
    """Resolve a model name to its provider config."""
    model_lower = model.lower()
    if "claude" in model_lower or "anthropic" in model_lower:
        config = PROVIDER_CONFIGS["claude"]
        config.model = model
        return config
    if "gpt" in model_lower or "o4" in model_lower or "o3" in model_lower:
        config = PROVIDER_CONFIGS["gpt"]
        config.model = model
        return config
    if "gemini" in model_lower:
        config = PROVIDER_CONFIGS["gemini"]
        config.model = model
        return config
    if "deepseek" in model_lower:
        config = PROVIDER_CONFIGS["deepseek"]
        config.model = model
        return config
    return None


def call_llm(
    messages: List[Dict[str, Any]],
    model: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs,
) -> LLMResponse:
    """Call an LLM with unified interface.

    Routes to the appropriate provider based on model name.
    """
    config = resolve_provider(model) if model else None
    if not config:
        # Default to first available
        for cfg in PROVIDER_CONFIGS.values():
            if cfg.is_available:
                config = cfg
                break

    if not config:
        return LLMResponse(text="No LLM provider available")

    start = time.monotonic()

    if config.provider == "anthropic":
        response = _call_anthropic(messages, config, max_tokens, temperature, **kwargs)
    elif config.provider == "openai":
        response = _call_openai(messages, config, max_tokens, temperature, **kwargs)
    else:
        response = _call_openai_compatible(messages, config, max_tokens, temperature, **kwargs)

    response.latency_ms = (time.monotonic() - start) * 1000
    return response


def _call_anthropic(
    messages: List[Dict[str, Any]],
    config: LLMConfig,
    max_tokens: int,
    temperature: float,
    **kwargs,
) -> LLMResponse:
    """Call Anthropic API."""
    import urllib.request

    # Separate system message
    system = ""
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        else:
            non_system.append(msg)

    payload = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": non_system,
    }
    if system:
        payload["system"] = system

    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        req = urllib.request.Request(
            f"{config.base_url}/v1/messages",
            data=body, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", config.model),
            provider="anthropic",
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", ""),
            raw=data,
        )
    except Exception as e:
        return LLMResponse(text=f"Anthropic API error: {e}", provider="anthropic")


def _call_openai(
    messages: List[Dict[str, Any]],
    config: LLMConfig,
    max_tokens: int,
    temperature: float,
    **kwargs,
) -> LLMResponse:
    """Call OpenAI API."""
    return _call_openai_compatible(messages, config, max_tokens, temperature, **kwargs)


def _call_openai_compatible(
    messages: List[Dict[str, Any]],
    config: LLMConfig,
    max_tokens: int,
    temperature: float,
    **kwargs,
) -> LLMResponse:
    """Call OpenAI-compatible API."""
    import urllib.request

    payload = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    body = json.dumps(payload).encode()
    base_url = config.base_url or "https://api.openai.com/v1"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return LLMResponse(
            text=text,
            model=data.get("model", config.model),
            provider=config.provider,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )
    except Exception as e:
        return LLMResponse(text=f"API error: {e}", provider=config.provider)
