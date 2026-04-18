"""Multi-credential pool with rotation strategies.

Inspired by Hermes credential_pool.py (MIT, Nous Research).
Simplified for Caveman — manages multiple API keys per provider
with automatic cooldown on exhaustion.
"""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 3600  # 1 hour default cooldown


class RotationStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    RANDOM = "random"


@dataclass
class Credential:
    """A single API credential."""
    provider: str
    key: str
    label: str = ""
    base_url: Optional[str] = None
    priority: int = 0
    request_count: int = 0
    status: str = "ok"  # ok | exhausted | error
    exhausted_at: Optional[float] = None
    error_code: Optional[int] = None
    error_message: str = ""

    @property
    def is_available(self) -> bool:
        if self.status == "ok":
            return True
        if self.status == "exhausted" and self.exhausted_at:
            elapsed = time.time() - self.exhausted_at
            if elapsed >= COOLDOWN_SECONDS:
                self.status = "ok"
                self.exhausted_at = None
                self.error_code = None
                return True
        return False

    def mark_exhausted(self, code: Optional[int] = None, message: str = "") -> None:
        self.status = "exhausted"
        self.exhausted_at = time.time()
        self.error_code = code
        self.error_message = message
        logger.info(
            "Credential %s/%s exhausted (code=%s): %s",
            self.provider, self.label or self.key[:8], code, message[:100],
        )

    def mark_ok(self) -> None:
        self.status = "ok"
        self.exhausted_at = None
        self.error_code = None
        self.error_message = ""

    def record_use(self) -> None:
        self.request_count += 1


class CredentialPool:
    """Manages multiple credentials per provider with rotation."""

    def __init__(self, strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN):
        self._pools: dict[str, list[Credential]] = {}
        self._indices: dict[str, int] = {}
        self._lock = threading.Lock()
        self.strategy = strategy

    def add(self, cred: Credential) -> None:
        with self._lock:
            pool = self._pools.setdefault(cred.provider, [])
            pool.append(cred)
            pool.sort(key=lambda c: c.priority)

    def add_key(
        self, provider: str, key: str,
        label: str = "", base_url: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        self.add(Credential(
            provider=provider, key=key, label=label,
            base_url=base_url, priority=priority,
        ))

    def get(self, provider: str) -> Optional[Credential]:
        """Get next available credential for provider."""
        with self._lock:
            pool = self._pools.get(provider, [])
            available = [c for c in pool if c.is_available]
            if not available:
                return None

            if self.strategy == RotationStrategy.ROUND_ROBIN:
                idx = self._indices.get(provider, 0) % len(available)
                self._indices[provider] = idx + 1
                cred = available[idx]
            elif self.strategy == RotationStrategy.LEAST_USED:
                cred = min(available, key=lambda c: c.request_count)
            else:  # RANDOM
                import random
                cred = random.choice(available)

            cred.record_use()
            return cred

    def mark_exhausted(
        self, provider: str, key: str,
        code: Optional[int] = None, message: str = "",
    ) -> Optional[Credential]:
        """Mark a credential as exhausted and return next available."""
        with self._lock:
            pool = self._pools.get(provider, [])
            for cred in pool:
                if cred.key == key:
                    cred.mark_exhausted(code, message)
                    break

        # Return next available (outside lock to avoid deadlock)
        return self.get(provider)

    def available_count(self, provider: str) -> int:
        with self._lock:
            pool = self._pools.get(provider, [])
            return sum(1 for c in pool if c.is_available)

    def total_count(self, provider: str) -> int:
        with self._lock:
            return len(self._pools.get(provider, []))

    def providers(self) -> list[str]:
        with self._lock:
            return list(self._pools.keys())

    def status_summary(self) -> dict[str, dict[str, int]]:
        """Return {provider: {ok: N, exhausted: N, total: N}}."""
        with self._lock:
            result = {}
            for provider, pool in self._pools.items():
                ok = sum(1 for c in pool if c.is_available)
                result[provider] = {
                    "ok": ok,
                    "exhausted": len(pool) - ok,
                    "total": len(pool),
                }
            return result

    @classmethod
    def from_config(cls, config: dict) -> "CredentialPool":
        """Build pool from Caveman config.yaml providers section."""
        strategy_name = config.get("credential_strategy", "round_robin")
        try:
            strategy = RotationStrategy(strategy_name)
        except ValueError:
            strategy = RotationStrategy.ROUND_ROBIN

        pool = cls(strategy=strategy)

        providers = config.get("providers", {})
        for provider_name, provider_conf in providers.items():
            if not isinstance(provider_conf, dict):
                continue

            # Single key
            api_key = provider_conf.get("api_key", "")
            if api_key and not api_key.startswith("${"):
                pool.add_key(
                    provider_name, api_key,
                    label="primary",
                    base_url=provider_conf.get("base_url"),
                )

            # Extra keys
            extra_keys = provider_conf.get("extra_keys", [])
            for i, entry in enumerate(extra_keys):
                if isinstance(entry, str):
                    pool.add_key(provider_name, entry, label=f"extra-{i}")
                elif isinstance(entry, dict):
                    pool.add_key(
                        provider_name,
                        entry.get("key", ""),
                        label=entry.get("label", f"extra-{i}"),
                        base_url=entry.get("base_url"),
                        priority=entry.get("priority", i + 1),
                    )

        return pool
