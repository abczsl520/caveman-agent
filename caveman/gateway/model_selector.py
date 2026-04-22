"""Model Selector — per-channel, per-user, fallback model resolution.

Extracted from OpenClaw model-picker.ts (946 lines) and
Hermes model selection patterns.

Features:
- Per-channel model override
- Per-user model preference
- Role-based model access
- Fallback chain resolution
- Model alias support
- Budget-aware selection
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("caveman.gateway.models")


@dataclass
class ModelEntry:
    """A model available for selection."""
    provider: str
    model: str
    alias: str = ""
    display_name: str = ""
    context_window: int = 200000
    max_output: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_vision: bool = False
    supports_tools: bool = True
    tier: str = "standard"  # "budget" | "standard" | "premium"

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def label(self) -> str:
        return self.display_name or self.alias or self.model


# Common model aliases
DEFAULT_ALIASES: Dict[str, tuple] = {
    "opus": ("anthropic", "claude-opus-4-6"),
    "sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "haiku": ("anthropic", "claude-haiku-4-20250414"),
    "gpt4o": ("openai", "gpt-4o"),
    "gpt4": ("openai", "gpt-4-turbo"),
    "gemini": ("google", "gemini-2.5-pro"),
    "deepseek": ("deepseek", "deepseek-chat"),
}


class ModelSelector:
    """Resolves which model to use for a given context."""

    def __init__(
        self,
        default_model: str = "",
        default_provider: str = "",
        models: Optional[List[ModelEntry]] = None,
        aliases: Optional[Dict[str, tuple]] = None,
        channel_overrides: Optional[Dict[str, str]] = None,
        user_preferences: Optional[Dict[str, str]] = None,
        tier_access: Optional[Dict[str, Set[str]]] = None,
    ):
        self._default_model = default_model
        self._default_provider = default_provider
        self._models = {m.full_name: m for m in (models or [])}
        self._aliases = aliases or dict(DEFAULT_ALIASES)
        self._channel_overrides = channel_overrides or {}
        self._user_preferences = user_preferences or {}
        self._tier_access = tier_access or {}  # tier → set of user_ids

    # ── Resolution ──

    def resolve(
        self,
        channel_id: str = "",
        user_id: str = "",
        session_override: str = "",
        require_vision: bool = False,
        require_tools: bool = False,
    ) -> tuple:
        """Resolve (provider, model) for the given context.

        Priority: session_override → user_preference → channel_override → default
        """
        # 1. Session override (from /model command)
        if session_override:
            resolved = self._resolve_alias(session_override)
            if resolved:
                return resolved

        # 2. User preference
        if user_id and user_id in self._user_preferences:
            resolved = self._resolve_alias(self._user_preferences[user_id])
            if resolved:
                entry = self._get_entry(*resolved)
                if entry and self._meets_requirements(entry, require_vision, require_tools):
                    return resolved

        # 3. Channel override
        if channel_id and channel_id in self._channel_overrides:
            resolved = self._resolve_alias(self._channel_overrides[channel_id])
            if resolved:
                return resolved

        # 4. Default
        if self._default_model:
            resolved = self._resolve_alias(self._default_model)
            if resolved:
                return resolved

        return (self._default_provider, self._default_model)

    def resolve_fallback_chain(
        self, primary_provider: str, primary_model: str,
    ) -> List[tuple]:
        """Build a fallback chain for the given primary model."""
        chain = [(primary_provider, primary_model)]
        primary_entry = self._get_entry(primary_provider, primary_model)
        if not primary_entry:
            return chain

        # Add same-tier alternatives from different providers
        for entry in self._models.values():
            if entry.full_name == primary_entry.full_name:
                continue
            if entry.tier == primary_entry.tier:
                chain.append((entry.provider, entry.model))
                if len(chain) >= 3:
                    break

        return chain

    # ── User Preferences ──

    def set_user_preference(self, user_id: str, model: str) -> bool:
        resolved = self._resolve_alias(model)
        if resolved:
            self._user_preferences[user_id] = model
            return True
        return False

    def clear_user_preference(self, user_id: str) -> None:
        self._user_preferences.pop(user_id, None)

    def get_user_preference(self, user_id: str) -> Optional[str]:
        return self._user_preferences.get(user_id)

    # ── Channel Overrides ──

    def set_channel_override(self, channel_id: str, model: str) -> bool:
        resolved = self._resolve_alias(model)
        if resolved:
            self._channel_overrides[channel_id] = model
            return True
        return False

    def clear_channel_override(self, channel_id: str) -> None:
        self._channel_overrides.pop(channel_id, None)

    # ── Model Registry ──

    def register_model(self, entry: ModelEntry) -> None:
        self._models[entry.full_name] = entry
        if entry.alias:
            self._aliases[entry.alias] = (entry.provider, entry.model)

    def list_models(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        models = self._models.values()
        if tier:
            models = [m for m in models if m.tier == tier]
        return [
            {
                "provider": m.provider,
                "model": m.model,
                "alias": m.alias,
                "display_name": m.label,
                "tier": m.tier,
                "context_window": m.context_window,
                "vision": m.supports_vision,
                "tools": m.supports_tools,
            }
            for m in sorted(models, key=lambda m: (m.tier, m.provider, m.model))
        ]

    # ── Internal ──

    def _resolve_alias(self, name: str) -> Optional[tuple]:
        """Resolve a model name or alias to (provider, model)."""
        # Direct alias
        if name.lower() in self._aliases:
            return self._aliases[name.lower()]
        # Full name (provider/model)
        if "/" in name:
            parts = name.split("/", 1)
            return (parts[0], parts[1])
        # Search by model name
        for entry in self._models.values():
            if entry.model == name or entry.alias == name:
                return (entry.provider, entry.model)
        return None

    def _get_entry(self, provider: str, model: str) -> Optional[ModelEntry]:
        return self._models.get(f"{provider}/{model}")

    @staticmethod
    def _meets_requirements(
        entry: ModelEntry, require_vision: bool, require_tools: bool,
    ) -> bool:
        if require_vision and not entry.supports_vision:
            return False
        if require_tools and not entry.supports_tools:
            return False
        return True
