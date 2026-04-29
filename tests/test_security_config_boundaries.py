"""Regression coverage for security/config boundary helpers."""
from __future__ import annotations

import json

import pytest

from caveman.commands.handlers import _helpers
from caveman.config import loader
from caveman.security.osv_check import _is_malware, _query_osv
from caveman.security.website_policy import check_url


def test_osv_query_ignores_wrong_shaped_vulns(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps({"vulns": [{"id": "MAL-1"}, "bad", {"id": "CVE-1"}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())

    assert _query_osv("pkg", "npm") == [{"id": "MAL-1"}, {"id": "CVE-1"}]


def test_is_malware_requires_string_id() -> None:
    assert _is_malware({"id": "MAL-123"}) is True
    assert _is_malware({"id": None}) is False


@pytest.mark.parametrize("bad_value", ["example.com", "", 0, False, [123]])
def test_website_policy_fails_closed_for_malformed_allowlist(bad_value) -> None:
    allowed, reason = check_url(
        "https://example.com",
        {"enabled": True, "allowed_domains": bad_value, "blocked_domains": [], "shared_files": []},
    )

    assert allowed is False
    assert reason == "malformed allowed_domains"


@pytest.mark.parametrize("field, reason", [("blocked_domains", "malformed blocked_domains"), ("shared_files", "malformed shared_files")])
@pytest.mark.parametrize("bad_value", ["example.com", "", 0, False, [123]])
def test_website_policy_fails_closed_for_malformed_blocklist_fields(field, reason, bad_value) -> None:
    policy = {"enabled": True, "allowed_domains": [], "blocked_domains": [], "shared_files": []}
    policy[field] = bad_value

    allowed, actual_reason = check_url("https://example.com", policy)

    assert allowed is False
    assert actual_reason == reason


def test_read_json_returns_only_json_containers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_helpers, "CAVEMAN_HOME", tmp_path)
    (tmp_path / "scalar.json").write_text('"not-container"', encoding="utf-8")
    (tmp_path / "object.json").write_text('{"ok": true}', encoding="utf-8")

    assert _helpers.read_json("scalar.json") is None
    assert _helpers.read_json("object.json") == {"ok": True}


@pytest.mark.parametrize("content", ["- not\n- a\n- mapping\n", "[]\n", "false\n", "0\n", "''\n"])
def test_load_config_rejects_non_mapping_user_config(tmp_path, monkeypatch, content) -> None:
    user_config = tmp_path / "config.yaml"
    user_config.write_text(content, encoding="utf-8")
    monkeypatch.setattr(loader, "BUNDLED_DEFAULT", tmp_path / "missing-default.yaml")
    loader._cache.clear()

    with pytest.raises(TypeError, match="config file must be a mapping"):
        loader.load_config(user_config, validate=False)
