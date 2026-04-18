"""Model metadata — context lengths, pricing, and provider detection.

Ported from Hermes agent/model_metadata.py (1085 lines → ~350 lines).
Keeps the essential: model database, provider prefix stripping, context
length detection, pricing, and local server detection.
Drops: OpenRouter live API fetching, models.dev fetching, YAML config.
"""
from __future__ import annotations

__all__ = [
    "ModelInfo", "get_model_info", "strip_provider_prefix",
    "is_local_endpoint", "detect_local_server_type",
    "estimate_tokens", "MINIMUM_CONTEXT_LENGTH",
]

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MINIMUM_CONTEXT_LENGTH = 64_000

# Provider prefixes that can appear before model IDs
_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "anthropic", "openai", "openrouter", "google", "gemini",
    "deepseek", "ollama", "local", "custom", "github",
    "qwen", "alibaba", "dashscope", "minimax", "kimi",
    "moonshot", "glm", "zhipu", "xai", "fireworks",
    "copilot", "nous", "xiaomi",
})

_OLLAMA_TAG_RE = re.compile(
    r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)",
    re.IGNORECASE,
)


def strip_provider_prefix(model: str) -> str:
    """Strip provider prefix: 'anthropic/claude-opus-4' → 'claude-opus-4'.

    Preserves Ollama-style tags: 'qwen3.5:27b' stays unchanged.
    """
    if ":" not in model and "/" not in model:
        return model
    if model.startswith("http"):
        return model
    # Handle slash-separated: anthropic/claude-opus-4
    if "/" in model:
        prefix, suffix = model.split("/", 1)
        if prefix.strip().lower() in _PROVIDER_PREFIXES:
            return suffix
        return model
    # Handle colon-separated: local:my-model
    prefix, suffix = model.split(":", 1)
    if prefix.strip().lower() in _PROVIDER_PREFIXES:
        if _OLLAMA_TAG_RE.match(suffix.strip()):
            return model  # Ollama tag, keep as-is
        return suffix
    return model


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a model."""

    id: str
    context_length: int = 128_000
    max_output_tokens: int = 8_192
    input_cost_per_mtok: float = 0.0  # $ per million input tokens
    output_cost_per_mtok: float = 0.0  # $ per million output tokens
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    provider: str = ""

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_mtok / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_mtok / 1_000_000

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in dollars."""
        return (
            input_tokens * self.input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )


# Built-in model database — covers major providers as of 2026-04
_MODEL_DB: dict[str, ModelInfo] = {}


def _register(
    id: str, ctx: int, out: int = 8192,
    inp_cost: float = 0.0, out_cost: float = 0.0,
    vision: bool = False, provider: str = "",
) -> None:
    _MODEL_DB[id] = ModelInfo(
        id=id, context_length=ctx, max_output_tokens=out,
        input_cost_per_mtok=inp_cost, output_cost_per_mtok=out_cost,
        supports_vision=vision, provider=provider,
    )


# Anthropic
_register("claude-opus-4-6", 1_000_000, 32_000, 15.0, 75.0, True, "anthropic")
_register("claude-sonnet-4-6", 1_000_000, 16_000, 3.0, 15.0, True, "anthropic")
_register("claude-opus-4-5", 200_000, 32_000, 15.0, 75.0, True, "anthropic")
_register("claude-sonnet-4-5", 200_000, 8_192, 3.0, 15.0, True, "anthropic")
_register("claude-haiku-3-5", 200_000, 8_192, 0.25, 1.25, True, "anthropic")

# OpenAI
_register("gpt-4.1", 1_047_576, 32_768, 2.0, 8.0, True, "openai")
_register("gpt-4.1-mini", 1_047_576, 16_384, 0.4, 1.6, True, "openai")
_register("gpt-4.1-nano", 1_047_576, 16_384, 0.1, 0.4, False, "openai")
_register("gpt-4o", 128_000, 16_384, 2.5, 10.0, True, "openai")
_register("gpt-4o-mini", 128_000, 16_384, 0.15, 0.6, True, "openai")
_register("o3", 200_000, 100_000, 10.0, 40.0, True, "openai")
_register("o3-mini", 200_000, 100_000, 1.1, 4.4, False, "openai")
_register("o4-mini", 200_000, 100_000, 1.1, 4.4, True, "openai")

# Google
_register("gemini-2.5-pro", 1_048_576, 65_536, 1.25, 10.0, True, "google")
_register("gemini-2.5-flash", 1_048_576, 65_536, 0.15, 0.6, True, "google")
_register("gemini-2.0-flash", 1_048_576, 8_192, 0.1, 0.4, True, "google")

# DeepSeek
_register("deepseek-chat", 128_000, 8_192, 0.27, 1.1, False, "deepseek")
_register("deepseek-reasoner", 128_000, 8_192, 0.55, 2.19, False, "deepseek")

# Qwen
_register("qwen3-coder-plus", 1_000_000, 16_384, 0.0, 0.0, False, "alibaba")
_register("qwen3-coder", 262_144, 16_384, 0.0, 0.0, False, "alibaba")
_register("qwen-max", 131_072, 16_384, 0.0, 0.0, False, "alibaba")

# Meta Llama
_register("llama-4-maverick", 1_048_576, 16_384, 0.0, 0.0, True, "meta")
_register("llama-4-scout", 524_288, 16_384, 0.0, 0.0, True, "meta")

# Groq (hosted models)
_register("llama-3.3-70b-versatile", 128_000, 32_768, 0.59, 0.79, False, "groq")
_register("llama-3.1-8b-instant", 131_072, 8_192, 0.05, 0.08, False, "groq")
_register("gemma2-9b-it", 8_192, 8_192, 0.2, 0.2, False, "groq")

# Mistral
_register("mistral-large-latest", 128_000, 8_192, 2.0, 6.0, False, "mistral")
_register("mistral-small-latest", 128_000, 8_192, 0.2, 0.6, False, "mistral")
_register("codestral-latest", 256_000, 8_192, 0.3, 0.9, False, "mistral")

# Together AI (hosted models)
_register("meta-llama/Llama-3.3-70B-Instruct-Turbo", 131_072, 16_384, 0.88, 0.88, False, "together")
_register("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", 131_072, 16_384, 3.5, 3.5, False, "together")

# xAI Grok
_register("grok-4", 256_000, 16_384, 0.0, 0.0, True, "xai")
_register("grok-3", 131_072, 16_384, 0.0, 0.0, False, "xai")


# Fuzzy match patterns (substring → context length)
_FUZZY_CONTEXT: list[tuple[str, int]] = [
    ("claude-opus-4-6", 1_000_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude", 200_000),
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("gemini", 1_048_576),
    ("deepseek", 128_000),
    ("mistral", 128_000),
    ("codestral", 256_000),
    ("qwen", 131_072),
    ("llama", 131_072),
    ("grok", 131_072),
]


def get_model_info(model: str) -> ModelInfo:
    """Look up model metadata. Falls back to fuzzy match, then defaults."""
    bare = strip_provider_prefix(model)

    # Exact match
    if bare in _MODEL_DB:
        return _MODEL_DB[bare]

    # Fuzzy match (longest substring first — list is pre-sorted)
    bare_lower = bare.lower()
    for pattern, ctx in _FUZZY_CONTEXT:
        if pattern in bare_lower:
            return ModelInfo(id=bare, context_length=ctx)

    # Default
    logger.debug("Unknown model '%s', using 128K default", model)
    return ModelInfo(id=bare)


# --- Token estimation ---

# ~4 chars per token for English, ~2 for CJK
_CHARS_PER_TOKEN_EN = 4.0
_CHARS_PER_TOKEN_CJK = 2.0
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """Rough token estimate without a tokenizer."""
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return int(cjk_chars / _CHARS_PER_TOKEN_CJK + other_chars / _CHARS_PER_TOKEN_EN)


# --- Local server detection ---

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_CONTAINER_SUFFIXES = (".docker.internal", ".containers.internal", ".lima.internal")

_URL_TO_PROVIDER: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.deepseek.com": "deepseek",
    "generativelanguage.googleapis.com": "gemini",
    "openrouter.ai": "openrouter",
    "dashscope.aliyuncs.com": "alibaba",
    "api.minimax": "minimax",
    "api.moonshot.ai": "kimi",
    "api.x.ai": "xai",
    "api.groq.com": "groq",
    "api.mistral.ai": "mistral",
    "api.together.xyz": "together",
}


def infer_provider_from_url(base_url: str) -> str | None:
    """Infer provider name from a base URL."""
    normalized = (base_url or "").strip().rstrip("/").lower()
    if not normalized:
        return None
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.netloc or parsed.path
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def is_local_endpoint(base_url: str) -> bool:
    """Check if base_url points to a local machine."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return False
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    if host in _LOCAL_HOSTS:
        return True
    if any(host.endswith(s) for s in _CONTAINER_SUFFIXES):
        return True
    # RFC-1918 check
    parts = host.split(".")
    if len(parts) == 4:
        try:
            a, b = int(parts[0]), int(parts[1])
            if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
                return True
        except ValueError:
            pass
    return False


def detect_local_server_type(base_url: str) -> str | None:
    """Detect local server type: 'ollama', 'lm-studio', 'vllm', 'llamacpp'."""
    import httpx

    normalized = (base_url or "").strip().rstrip("/")
    server = normalized[:-3] if normalized.endswith("/v1") else normalized

    try:
        with httpx.Client(timeout=2.0) as client:
            # LM Studio
            try:
                r = client.get(f"{server}/api/v1/models")
                if r.status_code == 200:
                    return "lm-studio"
            except Exception as e:
                logger.debug("Suppressed in model_metadata: %s", e)
            # Ollama
            try:
                r = client.get(f"{server}/api/tags")
                if r.status_code == 200 and "models" in r.json():
                    return "ollama"
            except Exception as e:
                logger.debug("Suppressed in model_metadata: %s", e)
            # llama.cpp
            try:
                r = client.get(f"{server}/v1/props")
                if r.status_code == 200 and "default_generation_settings" in r.text:
                    return "llamacpp"
            except Exception as e:
                logger.debug("Suppressed in model_metadata: %s", e)
            # vLLM
            try:
                r = client.get(f"{server}/version")
                if r.status_code == 200 and "version" in r.json():
                    return "vllm"
            except Exception as e:
                logger.debug("Suppressed in model_metadata: %s", e)
    except Exception as e:
        logger.debug("Suppressed in model_metadata: %s", e)
    return None
