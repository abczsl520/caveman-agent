"""Webhook Server — HTTP webhook endpoint for external integrations.

Provides a lightweight webhook server that can receive events from
external services and route them to the agent. Extracted from
Hermes hermes_cli/webhook.py (259 lines).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.webhook")

_SUBSCRIPTIONS_DIR = Path.home() / ".caveman" / "webhooks"


@dataclass
class WebhookSubscription:
    """A webhook subscription."""
    id: str
    url: str
    events: List[str] = field(default_factory=lambda: ["*"])
    secret: str = ""
    created_at: float = 0
    last_triggered: float = 0
    trigger_count: int = 0
    active: bool = True

    def matches_event(self, event_type: str) -> bool:
        return "*" in self.events or event_type in self.events


class WebhookManager:
    """Manages webhook subscriptions and delivery."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self._subs: Dict[str, WebhookSubscription] = {}
        self._persist_dir = persist_dir or _SUBSCRIPTIONS_DIR
        self._handlers: Dict[str, Callable] = {}
        self._load()

    def subscribe(
        self,
        url: str,
        events: Optional[List[str]] = None,
        secret: str = "",
    ) -> WebhookSubscription:
        """Create a new webhook subscription."""
        sub_id = hashlib.sha256(f"{url}:{time.time()}".encode()).hexdigest()[:12]
        sub = WebhookSubscription(
            id=sub_id,
            url=url,
            events=events or ["*"],
            secret=secret,
            created_at=time.time(),
        )
        self._subs[sub_id] = sub
        self._save()
        return sub

    def unsubscribe(self, sub_id: str) -> bool:
        if sub_id in self._subs:
            del self._subs[sub_id]
            self._save()
            return True
        return False

    def list_subscriptions(self) -> List[WebhookSubscription]:
        return list(self._subs.values())

    def trigger(self, event_type: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trigger webhooks for an event. Returns delivery results."""
        results = []
        for sub in self._subs.values():
            if not sub.active or not sub.matches_event(event_type):
                continue
            result = self._deliver(sub, event_type, payload)
            sub.last_triggered = time.time()
            sub.trigger_count += 1
            results.append(result)
        self._save()
        return results

    def _deliver(
        self, sub: WebhookSubscription, event_type: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deliver a webhook payload."""
        import urllib.request
        body = json.dumps({
            "event": event_type,
            "timestamp": time.time(),
            "subscription_id": sub.id,
            "data": payload,
        }).encode()

        headers = {"Content-Type": "application/json"}
        if sub.secret:
            signature = hmac.new(
                sub.secret.encode(), body, hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        try:
            req = urllib.request.Request(sub.url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return {
                    "subscription_id": sub.id,
                    "status": resp.status,
                    "success": 200 <= resp.status < 300,
                }
        except Exception as e:
            return {
                "subscription_id": sub.id,
                "status": 0,
                "success": False,
                "error": str(e),
            }

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a local handler for incoming webhooks."""
        self._handlers[event_type] = handler

    def handle_incoming(self, event_type: str, payload: Dict[str, Any]) -> Any:
        """Handle an incoming webhook event."""
        handler = self._handlers.get(event_type) or self._handlers.get("*")
        if handler:
            return handler(event_type, payload)
        return None

    def verify_signature(self, body: bytes, signature: str, secret: str) -> bool:
        """Verify webhook signature."""
        if not signature.startswith("sha256="):
            return False
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature[7:], expected)

    def _save(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / "subscriptions.json"
        try:
            data = {
                sid: {
                    "id": s.id, "url": s.url, "events": s.events,
                    "secret": s.secret, "created_at": s.created_at,
                    "last_triggered": s.last_triggered,
                    "trigger_count": s.trigger_count, "active": s.active,
                }
                for sid, s in self._subs.items()
            }
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to save subscriptions: %s", e)

    def _load(self) -> None:
        path = self._persist_dir / "subscriptions.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for sid, d in data.items():
                self._subs[sid] = WebhookSubscription(**d)
        except Exception as e:
            logger.debug("Failed to load subscriptions: %s", e)
