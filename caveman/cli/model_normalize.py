"""Model Normalize — model name normalization and alias resolution.

Normalizes model names across different formats and resolves
aliases. Extracted from Hermes hermes_cli/model_normalize.py.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "resolve_alias",
    "detect_provider",
    "normalize_model",
    "list_known_models",
]


# ── Model Aliases ──

MODEL_ALIASES: Dict[str, str] = {
    # Anthropic
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
    "claude": "claude-sonnet-4-20250514",
    # OpenAI
    "gpt4": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt4mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "o4": "o4-mini",
    "o4mini": "o4-mini",
    "o3": "o3-mini",
    # Google
    "gemini": "gemini-2.5-pro",
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "flash": "gemini-2.5-flash",
    # DeepSeek
    "deepseek": "deepseek-chat",
    "ds": "deepseek-chat",
    # Meta
    "llama": "llama-3.3-70b",
}

# ── Provider Detection ──

PROVIDER_PREFIXES: Dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o4": "openai",
    "o3": "openai",
    "gemini": "google",
    "deepseek": "deepseek",
    "llama": "meta",
    "mistral": "mistral",
    "qwen": "alibaba",
    "command": "cohere",
}


def resolve_alias(name: str) -> str:
    """Resolve a model alias to its full name."""
    normalized = name.strip().lower()
    return MODEL_ALIASES.get(normalized, name)


def detect_provider(model: str) -> str:
    """Detect the provider from a model name."""
    model_lower = model.lower()

    # Check if provider is explicitly specified (provider/model)
    if "/" in model:
        return model.split("/")[0]

    for prefix, provider in PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider

    return "unknown"


def normalize_model(name: str) -> Tuple[str, str]:
    """Normalize a model name. Returns (model, provider)."""
    # Strip provider prefix if present
    if "/" in name:
        provider, model = name.split("/", 1)
        model = resolve_alias(model)
        return model, provider

    model = resolve_alias(name)
    provider = detect_provider(model)
    return model, provider


def list_known_models() -> Dict[str, list]:
    """List all known models by provider."""
    models: Dict[str, list] = {}
    seen = set()
    for alias, full in MODEL_ALIASES.items():
        if full not in seen:
            provider = detect_provider(full)
            models.setdefault(provider, []).append(full)
            seen.add(full)
    return models
