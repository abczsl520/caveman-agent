"""Provider Registry — declarative provider resolution.

Replaces the 80-line if-elif chain in factory.py with a data-driven
registry. Each provider declares its model patterns and env var.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from caveman.providers.llm import LLMProvider


@dataclass
class ProviderSpec:
    """Declarative provider specification."""
    name: str
    cls_path: str  # e.g. "caveman.providers.anthropic_provider.AnthropicProvider"
    model_patterns: list[str]  # regex patterns matched against model name
    env_key: str  # e.g. "ANTHROPIC_API_KEY"
    config_section: str = ""  # key in providers config dict
    env_base_url: str = ""  # optional base_url env var
    priority: int = 100  # lower = matched first


# Registry ordered by priority (lower = checked first)
_REGISTRY: list[ProviderSpec] = [
    ProviderSpec(
        name="gemini",
        cls_path="caveman.providers.gemini_provider.GeminiProvider",
        model_patterns=[r"gemini"],
        env_key="GEMINI_API_KEY",
        config_section="gemini",
        env_base_url="GEMINI_BASE_URL",
        priority=10,
    ),
    ProviderSpec(
        name="groq",
        cls_path="caveman.providers.groq_provider.GroqProvider",
        model_patterns=[r"^groq[/-]"],
        env_key="GROQ_API_KEY",
        config_section="groq",
        priority=20,
    ),
    ProviderSpec(
        name="deepseek",
        cls_path="caveman.providers.deepseek_provider.DeepSeekProvider",
        model_patterns=[r"deepseek"],
        env_key="DEEPSEEK_API_KEY",
        config_section="deepseek",
        priority=30,
    ),
    ProviderSpec(
        name="mistral",
        cls_path="caveman.providers.mistral_provider.MistralProvider",
        model_patterns=[r"mistral"],
        env_key="MISTRAL_API_KEY",
        config_section="mistral",
        priority=40,
    ),
    ProviderSpec(
        name="together",
        cls_path="caveman.providers.together_provider.TogetherProvider",
        model_patterns=[r"meta-llama/"],
        env_key="TOGETHER_API_KEY",
        config_section="together",
        priority=50,
    ),
    ProviderSpec(
        name="ollama",
        cls_path="caveman.providers.ollama_provider.OllamaProvider",
        model_patterns=[r"^llama", r"^qwen"],
        env_key="",  # no API key needed
        config_section="ollama",
        env_base_url="OLLAMA_BASE_URL",
        priority=60,
    ),
    ProviderSpec(
        name="anthropic",
        cls_path="caveman.providers.anthropic_provider.AnthropicProvider",
        model_patterns=[r"claude", r"anthropic"],
        env_key="ANTHROPIC_API_KEY",
        config_section="anthropic",
        env_base_url="ANTHROPIC_BASE_URL",
        priority=70,
    ),
    ProviderSpec(
        name="openai",
        cls_path="caveman.providers.openai_provider.OpenAIProvider",
        model_patterns=[r"gpt", r"openai", r"o1", r"o3", r"o4"],
        env_key="OPENAI_API_KEY",
        config_section="openai",
        priority=80,
    ),
]


def _import_class(cls_path: str) -> type:
    """Import a class from a dotted path."""
    module_path, cls_name = cls_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def resolve_provider(
    model: str,
    providers_cfg: dict[str, Any],
    default_max_tokens: dict[str, int] | None = None,
) -> LLMProvider:
    """Resolve a provider from model name and config.

    Resolution order:
    1. Explicit `providers.default` in config
    2. Model name pattern matching against registry
    3. Fallback to Anthropic

    Raises ValueError if no API key is available.
    """
    default_max_tokens = default_max_tokens or {}
    explicit = providers_cfg.get("default", "")

    # Sort registry by priority
    sorted_specs = sorted(_REGISTRY, key=lambda s: s.priority)

    # Step 1: explicit provider override
    if explicit:
        for spec in sorted_specs:
            if spec.name == explicit:
                return _create_provider(spec, model, providers_cfg, default_max_tokens)

    # Step 2: model name pattern matching
    for spec in sorted_specs:
        for pattern in spec.model_patterns:
            if re.search(pattern, model, re.IGNORECASE):
                return _create_provider(spec, model, providers_cfg, default_max_tokens)

    # Step 3: fallback to anthropic
    anthropic_spec = next(s for s in sorted_specs if s.name == "anthropic")
    return _create_provider(anthropic_spec, model, providers_cfg, default_max_tokens)


def _create_provider(
    spec: ProviderSpec,
    model: str,
    providers_cfg: dict[str, Any],
    default_max_tokens: dict[str, int],
) -> LLMProvider:
    """Instantiate a provider from its spec."""
    section = providers_cfg.get(spec.config_section or spec.name, {})
    api_key = section.get("api_key") or os.environ.get(spec.env_key, "") if spec.env_key else ""
    base_url = None
    if spec.env_base_url:
        base_url = section.get("base_url") or os.environ.get(spec.env_base_url)

    # Validate API key for providers that need one
    if spec.env_key and not api_key and spec.name != "ollama":
        raise ValueError(
            f"{spec.name.title()} API key not configured. "
            f"Set {spec.env_key} env var or providers.{spec.name}.api_key in config.yaml"
        )

    cls = _import_class(spec.cls_path)
    max_tokens = section.get("max_tokens") or default_max_tokens.get(spec.name)

    kwargs: dict[str, Any] = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if base_url:
        kwargs["base_url"] = base_url

    # Ollama special case: base_url defaults
    if spec.name == "ollama" and not base_url:
        from caveman.paths import DEFAULT_OLLAMA_URL
        kwargs["base_url"] = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)

    return cls(**kwargs)
