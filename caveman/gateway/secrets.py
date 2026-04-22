"""Secrets Manager — credential storage, rotation, and resolution.

Provides secure credential management with multiple backends,
rotation support, and environment variable resolution.
Core patterns from OpenClaw src/secrets/ (20K LOC — extracted essentials).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.secrets")

_SECRETS_DIR = Path.home() / ".caveman" / "secrets"
_CREDENTIAL_FILE = _SECRETS_DIR / "credentials.json"


@dataclass
class Credential:
    """A stored credential."""
    key: str
    value: str = ""
    provider: str = ""
    label: str = ""
    created_at: float = 0
    rotated_at: float = 0
    expires_at: float = 0  # 0 = never
    source: str = "manual"  # manual | env | file | oauth

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def masked_value(self) -> str:
        if len(self.value) <= 8:
            return "***"
        return self.value[:4] + "..." + self.value[-4:]


class SecretsManager:
    """Manages credentials with multiple resolution strategies."""

    def __init__(self, secrets_dir: Optional[Path] = None):
        self._dir = secrets_dir or _SECRETS_DIR
        self._credentials: Dict[str, Credential] = {}
        self._load()

    def get(self, key: str) -> Optional[str]:
        """Get a credential value. Resolution order: store → env → None."""
        # 1. Check stored credentials
        cred = self._credentials.get(key)
        if cred and not cred.is_expired:
            return cred.value

        # 2. Check environment variables
        env_val = os.environ.get(key)
        if env_val:
            return env_val

        # 3. Check common env var patterns
        for prefix in ("CAVEMAN_", ""):
            env_val = os.environ.get(f"{prefix}{key}")
            if env_val:
                return env_val

        return None

    def set(
        self,
        key: str,
        value: str,
        provider: str = "",
        label: str = "",
        expires_at: float = 0,
    ) -> None:
        """Store a credential."""
        self._credentials[key] = Credential(
            key=key,
            value=value,
            provider=provider,
            label=label,
            created_at=time.time(),
            expires_at=expires_at,
            source="manual",
        )
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._credentials:
            del self._credentials[key]
            self._save()
            return True
        return False

    def rotate(self, key: str, new_value: str) -> bool:
        """Rotate a credential to a new value."""
        cred = self._credentials.get(key)
        if not cred:
            return False
        cred.value = new_value
        cred.rotated_at = time.time()
        self._save()
        return True

    def list_credentials(self, include_values: bool = False) -> List[Dict[str, Any]]:
        """List all credentials (masked by default)."""
        return [
            {
                "key": c.key,
                "provider": c.provider,
                "label": c.label,
                "value": c.value if include_values else c.masked_value,
                "source": c.source,
                "expired": c.is_expired,
            }
            for c in self._credentials.values()
        ]

    def resolve_for_provider(self, provider: str) -> Dict[str, str]:
        """Resolve all credentials for a provider."""
        result = {}
        for cred in self._credentials.values():
            if cred.provider == provider and not cred.is_expired:
                result[cred.key] = cred.value
        return result

    def audit(self) -> Dict[str, Any]:
        """Audit credential health."""
        total = len(self._credentials)
        expired = sum(1 for c in self._credentials.values() if c.is_expired)
        never_rotated = sum(1 for c in self._credentials.values() if c.rotated_at == 0)
        return {
            "total": total,
            "expired": expired,
            "never_rotated": never_rotated,
            "healthy": total - expired,
        }

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, cred in self._credentials.items():
            data[key] = {
                "key": cred.key, "value": cred.value,
                "provider": cred.provider, "label": cred.label,
                "created_at": cred.created_at, "rotated_at": cred.rotated_at,
                "expires_at": cred.expires_at, "source": cred.source,
            }
        try:
            _CREDENTIAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CREDENTIAL_FILE.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8",
            )
            # Restrict permissions
            _CREDENTIAL_FILE.chmod(0o600)
        except Exception as e:
            logger.error("Failed to save credentials: %s", e)

    def _load(self) -> None:
        if not _CREDENTIAL_FILE.exists():
            return
        try:
            data = json.loads(_CREDENTIAL_FILE.read_text(encoding="utf-8"))
            for key, d in data.items():
                self._credentials[key] = Credential(**d)
        except Exception as e:
            logger.error("Failed to load credentials: %s", e)
