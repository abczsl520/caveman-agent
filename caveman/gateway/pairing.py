"""Gateway device pairing — QR code / token based pairing for mobile apps.

Generates a pairing token, displays it (or QR code), and waits for a
device to claim it. Once paired, the device can send/receive messages
through the gateway.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from caveman.paths import CAVEMAN_HOME

__all__ = [
    "PairedDevice",
    "PairingToken",
    "PairingManager",
]


logger = logging.getLogger(__name__)

_PAIRING_DIR = CAVEMAN_HOME / "pairing"
_PAIRED_DEVICES_FILE = _PAIRING_DIR / "devices.json"
_TOKEN_TTL = 300  # 5 minutes


@dataclass
class PairedDevice:
    """A device paired to this agent instance via token exchange."""
    device_id: str
    name: str
    platform: str  # "android", "ios", "web"
    paired_at: float
    last_seen: float = 0.0
    capabilities: list[str] = field(default_factory=list)


@dataclass
class PairingToken:
    """Time-limited token for device pairing handshake."""
    token: str
    created_at: float
    expires_at: float
    claimed_by: str | None = None


class PairingManager:
    """Manages device pairing lifecycle."""

    def __init__(self, pairing_dir: Path | None = None) -> None:
        self._dir = pairing_dir or _PAIRING_DIR
        self._devices_file = self._dir / "devices.json"
        self._active_tokens: dict[str, PairingToken] = {}

    def generate_token(self, ttl: int = _TOKEN_TTL) -> PairingToken:
        """Generate a new pairing token."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        pt = PairingToken(token=token, created_at=now, expires_at=now + ttl)
        self._active_tokens[token] = pt
        self._cleanup_expired()
        logger.info("Generated pairing token (expires in %ds)", ttl)
        return pt

    def claim_token(
        self,
        token: str,
        device_id: str,
        name: str,
        platform: str,
        capabilities: list[str] | None = None,
    ) -> PairedDevice | None:
        """Claim a pairing token and register the device.

        Returns PairedDevice if successful, None if token invalid/expired.
        """
        self._cleanup_expired()

        pt = self._active_tokens.get(token)
        if not pt:
            logger.warning("Pairing: invalid or expired token")
            return None

        if pt.claimed_by:
            logger.warning("Pairing: token already claimed by %s", pt.claimed_by)
            return None

        now = time.time()
        device = PairedDevice(
            device_id=device_id,
            name=name,
            platform=platform,
            paired_at=now,
            last_seen=now,
            capabilities=capabilities or [],
        )

        pt.claimed_by = device_id
        del self._active_tokens[token]

        self._save_device(device)
        logger.info("Paired device: %s (%s/%s)", name, platform, device_id[:8])
        return device

    def get_devices(self) -> list[PairedDevice]:
        """List all paired devices."""
        return self._load_devices()

    def remove_device(self, device_id: str) -> bool:
        """Unpair a device."""
        devices = self._load_devices()
        filtered = [d for d in devices if d.device_id != device_id]
        if len(filtered) == len(devices):
            return False
        self._save_devices(filtered)
        logger.info("Unpaired device: %s", device_id[:8])
        return True

    def update_last_seen(self, device_id: str) -> None:
        """Update last_seen timestamp for a device."""
        devices = self._load_devices()
        for d in devices:
            if d.device_id == device_id:
                d.last_seen = time.time()
                break
        self._save_devices(devices)

    def is_paired(self, device_id: str) -> bool:
        """Check if a device is paired."""
        return any(d.device_id == device_id for d in self._load_devices())

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [t for t, pt in self._active_tokens.items() if pt.expires_at < now]
        for t in expired:
            del self._active_tokens[t]

    def _load_devices(self) -> list[PairedDevice]:
        if not self._devices_file.exists():
            return []
        try:
            data = json.loads(self._devices_file.read_text(encoding="utf-8"))
            return [PairedDevice(**d) for d in data]
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return []

    def _save_device(self, device: PairedDevice) -> None:
        devices = self._load_devices()
        # Replace if exists
        devices = [d for d in devices if d.device_id != device.device_id]
        devices.append(device)
        self._save_devices(devices)

    def _save_devices(self, devices: list[PairedDevice]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "device_id": d.device_id,
                "name": d.name,
                "platform": d.platform,
                "paired_at": d.paired_at,
                "last_seen": d.last_seen,
                "capabilities": d.capabilities,
            }
            for d in devices
        ]
        self._devices_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
