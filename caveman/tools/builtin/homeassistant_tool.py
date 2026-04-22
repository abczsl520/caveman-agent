"""Home Assistant integration — control smart home devices via REST API.

Tools:
- ha_list_entities: List/filter entities by domain or area
- ha_get_state: Get detailed state of a single entity
- ha_call_service: Call a HA service (turn_on, turn_off, etc.)

Auth: Long-Lived Access Token via HASS_TOKEN env var.
URL: HASS_URL env var (default: http://homeassistant.local:8123).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from caveman.tools.registry import tool

__all__ = [
    "ha_list_entities",
    "ha_get_state",
    "ha_call_service",
]


logger = logging.getLogger(__name__)

_ENTITY_ID_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")

# Blocked domains for security
_BLOCKED_DOMAINS = frozenset({
    "shell_command", "command_line", "python_script",
    "pyscript", "hassio", "rest_command",
})


def _get_config() -> tuple[str, str]:
    """Return (hass_url, hass_token) from env vars."""
    return (
        os.getenv("HASS_URL", "http://homeassistant.local:8123").rstrip("/"),
        os.getenv("HASS_TOKEN", ""),
    )


def _headers(token: str = "") -> dict[str, str]:
    if not token:
        _, token = _get_config()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _ha_get(path: str) -> Any:
    """GET request to Home Assistant API."""
    import httpx
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{url}/api/{path}", headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def _ha_post(path: str, data: dict[str, Any] | None = None) -> Any:
    """POST request to Home Assistant API."""
    import httpx
    url, _ = _get_config()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{url}/api/{path}", headers=_headers(), json=data or {})
        resp.raise_for_status()
        return resp.json() if resp.content else {}


@tool(
    name="ha_list_entities",
    description="List Home Assistant entities, optionally filtered by domain (light, switch, sensor, etc.)",
    params={
        "domain": {"type": "string", "description": "Entity domain filter (e.g., 'light', 'switch')"},
    },
    required=[],
)
async def ha_list_entities(domain: str = "") -> str:
    """List HA entities."""
    _, token = _get_config()
    if not token:
        return "Error: HASS_TOKEN not set"

    try:
        states = await _ha_get("states")
        entities = []
        for s in states:
            eid = s.get("entity_id", "")
            if domain and not eid.startswith(f"{domain}."):
                continue
            state = s.get("state", "unknown")
            name = s.get("attributes", {}).get("friendly_name", eid)
            entities.append(f"  {eid}: {state} ({name})")

        if not entities:
            return f"No entities found" + (f" for domain '{domain}'" if domain else "")
        return f"Found {len(entities)} entities:\n" + "\n".join(entities[:50])
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ha_get_state",
    description="Get detailed state of a Home Assistant entity",
    params={
        "entity_id": {"type": "string", "description": "Entity ID (e.g., 'light.living_room')"},
    },
    required=["entity_id"],
)
async def ha_get_state(entity_id: str) -> str:
    """Get entity state."""
    if not _ENTITY_ID_RE.match(entity_id):
        return f"Invalid entity_id format: {entity_id}"

    try:
        state = await _ha_get(f"states/{entity_id}")
        attrs = state.get("attributes", {})
        lines = [
            f"Entity: {entity_id}",
            f"State: {state.get('state', 'unknown')}",
            f"Name: {attrs.get('friendly_name', 'N/A')}",
            f"Last changed: {state.get('last_changed', 'N/A')}",
        ]
        # Add key attributes
        for key in ("brightness", "color_temp", "temperature", "humidity", "unit_of_measurement"):
            if key in attrs:
                lines.append(f"{key}: {attrs[key]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="ha_call_service",
    description="Call a Home Assistant service (turn_on, turn_off, toggle, set_temperature, etc.)",
    params={
        "domain": {"type": "string", "description": "Service domain (e.g., 'light', 'switch', 'climate')"},
        "service": {"type": "string", "description": "Service name (e.g., 'turn_on', 'turn_off')"},
        "entity_id": {"type": "string", "description": "Target entity ID"},
        "data": {"type": "string", "description": "Additional service data as JSON string"},
    },
    required=["domain", "service", "entity_id"],
)
async def ha_call_service(
    domain: str,
    service: str,
    entity_id: str,
    data: str = "{}",
) -> str:
    """Call a HA service."""
    if domain in _BLOCKED_DOMAINS:
        return f"Error: domain '{domain}' is blocked for security"

    if not _ENTITY_ID_RE.match(entity_id):
        return f"Invalid entity_id: {entity_id}"

    import json
    try:
        extra_data = json.loads(data) if data and data != "{}" else {}
    except json.JSONDecodeError:
        return f"Invalid JSON in data: {data}"

    payload = {"entity_id": entity_id, **extra_data}

    try:
        result = await _ha_post(f"services/{domain}/{service}", payload)
        return f"✅ Called {domain}.{service} on {entity_id}"
    except Exception as e:
        return f"Error: {e}"
