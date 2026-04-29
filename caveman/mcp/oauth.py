"""MCP OAuth authentication for MCP server connections.

Handles OAuth 2.0 flows for MCP servers that require authentication.
Supports authorization code flow with PKCE.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode

from caveman.paths import CAVEMAN_HOME
from caveman.timeouts import HTTP_DEFAULT

__all__ = [
    "generate_pkce",
    "build_auth_url",
    "refresh_token",
    "save_tokens",
    "load_tokens",
    "is_token_expired",
]


logger = logging.getLogger(__name__)

_TOKEN_DIR = CAVEMAN_HOME / "mcp" / "tokens"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(64)
    challenge = urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(
    auth_endpoint: str,
    client_id: str,
    redirect_uri: str,
    scope: str = "",
    state: str | None = None,
    code_challenge: str | None = None,
) -> str:
    """Build OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    if scope:
        params["scope"] = scope
    if state:
        params["state"] = state
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    return f"{auth_endpoint}?{urlencode(params)}"


async def refresh_token(
    token_endpoint: str,
    refresh_tok: str,
    client_id: str,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Refresh an access token."""
    import httpx

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_tok,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=HTTP_DEFAULT) as client:
        resp = await client.post(token_endpoint, data=data)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return {"error": "token endpoint response must be a JSON object"}


def save_tokens(server_name: str, tokens: dict[str, Any], token_dir: Path | None = None) -> None:
    """Save tokens to disk."""
    tdir = token_dir or _TOKEN_DIR
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{server_name}.json"
    tokens["saved_at"] = time.time()
    path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")


def load_tokens(server_name: str, token_dir: Path | None = None) -> dict[str, Any] | None:
    """Load saved tokens from disk."""
    tdir = token_dir or _TOKEN_DIR
    path = tdir / f"{server_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return None
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None


def is_token_expired(tokens: dict[str, Any]) -> bool:
    """Check if access token is expired (with 60s buffer)."""
    saved_at_raw = tokens.get("saved_at", 0)
    expires_in_raw = tokens.get("expires_in", 3600)
    try:
        saved_at = float(saved_at_raw)
        expires_in = float(expires_in_raw)
    except (TypeError, ValueError):
        return True
    return time.time() > (saved_at + expires_in - 60)
