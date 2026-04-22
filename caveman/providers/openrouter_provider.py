"""OpenRouter provider — access 200+ models through a single API.

OpenRouter provides a unified API compatible with OpenAI's format,
giving access to models from Anthropic, Google, Meta, Mistral, etc.
"""
from __future__ import annotations
from caveman.providers.openai_provider import OpenAIProvider

__all__ = ["OpenRouterProvider"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — OpenAI-compatible API with 200+ models."""

    def __init__(self, api_key: str = "", model: str = "anthropic/claude-3.5-sonnet", **kwargs):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=kwargs.pop("base_url", OPENROUTER_BASE_URL),
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"
