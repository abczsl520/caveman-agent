"""Access Control — user/channel/role permission management.

Extracted from OpenClaw allow-list.ts (620 lines) and
Hermes config-based access patterns.

Features:
- Multi-level access: owner → admin → user → guest
- Channel allowlist with glob patterns
- User allowlist with role-based escalation
- Per-channel configuration overrides
- Pairing-based DM access
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "AccessLevel",
    "AccessRule",
    "ChannelConfig",
    "AccessController",
]


logger = logging.getLogger("caveman.gateway.access")


class AccessLevel(IntEnum):
    """Access levels, higher = more permissions."""
    BLOCKED = 0
    GUEST = 10
    USER = 20
    ADMIN = 50
    OWNER = 100


@dataclass
class AccessRule:
    """A single access rule."""
    pattern: str  # user ID, role ID, or glob pattern
    level: AccessLevel = AccessLevel.USER
    channels: Optional[Set[str]] = None  # Restrict to specific channels
    note: str = ""


@dataclass
class ChannelConfig:
    """Per-channel configuration override."""
    channel_id: str
    enabled: bool = True
    model: str = ""
    require_mention: bool = False
    max_tokens: int = 0
    system_prompt: str = ""
    allowed_tools: Optional[Set[str]] = None


class AccessController:
    """Manages access control for the gateway."""

    def __init__(
        self,
        owner_ids: Optional[Set[str]] = None,
        admin_ids: Optional[Set[str]] = None,
        rules: Optional[List[AccessRule]] = None,
        channel_configs: Optional[Dict[str, ChannelConfig]] = None,
        default_level: AccessLevel = AccessLevel.USER,
        dm_policy: str = "pairing",  # "open" | "pairing" | "allowlist" | "disabled"
    ):
        self._owner_ids = owner_ids or set()
        self._admin_ids = admin_ids or set()
        self._rules = rules or []
        self._channel_configs = channel_configs or {}
        self._default_level = default_level
        self._dm_policy = dm_policy
        self._paired_users: Set[str] = set()

    # ── Access Resolution ──

    def resolve_access(
        self, user_id: str, channel_id: str = "",
        role_ids: Optional[Set[str]] = None,
    ) -> AccessLevel:
        """Resolve the effective access level for a user."""
        # Owner check
        if user_id in self._owner_ids:
            return AccessLevel.OWNER

        # Admin check
        if user_id in self._admin_ids:
            return AccessLevel.ADMIN

        # Rule-based check (highest matching rule wins)
        best = self._default_level
        for rule in self._rules:
            if self._rule_matches(rule, user_id, channel_id, role_ids):
                if rule.level > best:
                    best = rule.level
                if rule.level == AccessLevel.BLOCKED:
                    return AccessLevel.BLOCKED  # Blocked overrides all

        return best

    def is_allowed(
        self, user_id: str, channel_id: str = "",
        min_level: AccessLevel = AccessLevel.USER,
        role_ids: Optional[Set[str]] = None,
    ) -> bool:
        """Check if a user has at least the required access level."""
        level = self.resolve_access(user_id, channel_id, role_ids)
        return level >= min_level

    def is_owner(self, user_id: str) -> bool:
        return user_id in self._owner_ids

    def is_admin(self, user_id: str) -> bool:
        return user_id in self._admin_ids or user_id in self._owner_ids

    # ── DM Policy ──

    def is_dm_allowed(self, user_id: str) -> bool:
        """Check if DM access is allowed for a user."""
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "open":
            return True
        if self._dm_policy == "pairing":
            return user_id in self._paired_users or self.is_admin(user_id)
        if self._dm_policy == "allowlist":
            return self.is_allowed(user_id)
        return False

    def pair_user(self, user_id: str) -> None:
        self._paired_users.add(user_id)

    def unpair_user(self, user_id: str) -> None:
        self._paired_users.discard(user_id)

    # ── Channel Config ──

    def get_channel_config(self, channel_id: str) -> Optional[ChannelConfig]:
        return self._channel_configs.get(channel_id)

    def is_channel_enabled(self, channel_id: str) -> bool:
        config = self._channel_configs.get(channel_id)
        if config is not None:
            return config.enabled
        return True  # Default: all channels enabled

    def set_channel_config(self, config: ChannelConfig) -> None:
        self._channel_configs[config.channel_id] = config

    # ── Rule Management ──

    def add_rule(self, rule: AccessRule) -> None:
        self._rules.append(rule)

    def remove_rules_for(self, pattern: str) -> int:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.pattern != pattern]
        return before - len(self._rules)

    def list_rules(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r.pattern, "level": r.level.name, "note": r.note}
            for r in self._rules
        ]

    # ── Internal ──

    @staticmethod
    def _rule_matches(
        rule: AccessRule, user_id: str, channel_id: str,
        role_ids: Optional[Set[str]] = None,
    ) -> bool:
        """Check if a rule matches the given context."""
        # Channel restriction
        if rule.channels and channel_id and channel_id not in rule.channels:
            return False

        # Direct user match
        if rule.pattern == user_id:
            return True

        # Role match
        if role_ids and rule.pattern in role_ids:
            return True

        # Glob pattern match
        if "*" in rule.pattern or "?" in rule.pattern:
            return fnmatch.fnmatch(user_id, rule.pattern)

        return False
