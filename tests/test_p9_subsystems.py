"""Tests for P9: webhook, browser providers, website policy, web research."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ── Webhook ──

class TestWebhook:
    def test_subscribe_unsubscribe(self, tmp_path):
        from caveman.gateway.webhook import WebhookManager
        mgr = WebhookManager(persist_dir=tmp_path / "webhooks")
        sub = mgr.subscribe("https://example.com/hook", events=["message"])
        assert sub.id
        assert sub.matches_event("message")
        assert not sub.matches_event("other")
        assert mgr.unsubscribe(sub.id)
        assert not mgr.unsubscribe(sub.id)

    def test_wildcard_subscription(self, tmp_path):
        from caveman.gateway.webhook import WebhookManager
        mgr = WebhookManager(persist_dir=tmp_path / "webhooks")
        sub = mgr.subscribe("https://example.com/hook")
        assert sub.matches_event("anything")

    def test_verify_signature(self, tmp_path):
        from caveman.gateway.webhook import WebhookManager
        mgr = WebhookManager(persist_dir=tmp_path / "webhooks")
        body = b'{"test": true}'
        import hmac, hashlib
        sig = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        assert mgr.verify_signature(body, sig, "secret")
        assert not mgr.verify_signature(body, "sha256=wrong", "secret")

    def test_handler_registration(self, tmp_path):
        from caveman.gateway.webhook import WebhookManager
        mgr = WebhookManager(persist_dir=tmp_path / "webhooks")
        received = []
        mgr.register_handler("test", lambda t, p: received.append(p))
        mgr.handle_incoming("test", {"data": 1})
        assert len(received) == 1

    def test_persistence(self, tmp_path):
        from caveman.gateway.webhook import WebhookManager
        mgr1 = WebhookManager(persist_dir=tmp_path / "webhooks")
        mgr1.subscribe("https://example.com/hook", events=["msg"])
        # Load from disk
        mgr2 = WebhookManager(persist_dir=tmp_path / "webhooks")
        assert len(mgr2.list_subscriptions()) == 1


# ── Browser Providers ──

class TestBrowserProviders:
    def test_list_providers(self):
        from caveman.tools.builtin.browser_providers import list_providers
        providers = list_providers()
        assert len(providers) >= 2
        names = [p["name"] for p in providers]
        assert "Browserbase" in names
        assert "Browser-Use" in names

    def test_get_provider_none(self):
        from caveman.tools.builtin.browser_providers import get_provider
        # Without env vars, no provider should be configured
        import os
        saved = {k: os.environ.pop(k, None) for k in ("BROWSERBASE_API_KEY", "BROWSER_USE_API_KEY")}
        try:
            assert get_provider() is None
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_browserbase_not_configured(self):
        from caveman.tools.builtin.browser_providers import BrowserbaseProvider
        import os
        saved = os.environ.pop("BROWSERBASE_API_KEY", None)
        try:
            p = BrowserbaseProvider()
            assert not p.is_configured()
        finally:
            if saved:
                os.environ["BROWSERBASE_API_KEY"] = saved

    def test_browser_use_not_configured(self):
        from caveman.tools.builtin.browser_providers import BrowserUseProvider
        import os
        saved = os.environ.pop("BROWSER_USE_API_KEY", None)
        try:
            p = BrowserUseProvider()
            assert not p.is_configured()
        finally:
            if saved:
                os.environ["BROWSER_USE_API_KEY"] = saved


# ── Website Policy ──

class TestWebsitePolicy:
    def test_blocked_domains(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        result = mgr.check_access("http://169.254.169.254/latest/meta-data")
        assert not result["allowed"]

    def test_private_ip(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        result = mgr.check_access("http://192.168.1.1/admin")
        assert not result["allowed"]

    def test_normal_domain(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        result = mgr.check_access("https://example.com/page")
        assert result["allowed"]

    def test_rate_limiting(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        mgr.set_policy("test.com", rate_limit_rpm=2)
        mgr.record_request("https://test.com/1")
        mgr.record_request("https://test.com/2")
        result = mgr.check_access("https://test.com/3")
        assert not result["allowed"]
        assert result["reason"] == "rate_limited"

    def test_custom_policy(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        mgr.set_policy("blocked.com", allowed=False, notes="manual block")
        result = mgr.check_access("https://blocked.com/page")
        assert not result["allowed"]

    def test_reset(self):
        from caveman.tools.builtin.website_policy import WebsitePolicyManager
        mgr = WebsitePolicyManager()
        mgr.set_policy("test.com", allowed=False)
        mgr.reset()
        assert len(mgr.list_policies()) == 0


# ── Web Research ──

class TestWebResearch:
    def test_research_session(self):
        from caveman.tools.builtin.web_research import ResearchSession, ResearchStep
        session = ResearchSession(question="What is Python?", started_at=time.time())
        session.add_step(ResearchStep(
            action="search", query="Python", urls=["https://python.org"],
        ))
        assert len(session.steps) == 1
        assert "python.org" in session.domains_used

    def test_score_computation(self):
        from caveman.tools.builtin.web_research import ResearchSession, ResearchStep
        session = ResearchSession(question="test", max_steps=10)
        session.add_step(ResearchStep(action="search", urls=["https://a.com", "https://b.com"]))
        session.add_step(ResearchStep(action="extract", urls=["https://c.com"]))
        session.answer = "Python is a programming language"
        score = session.compute_score("Python is a popular programming language")
        assert score.total > 0
        assert score.source_diversity > 0
        assert score.tool_usage > 0

    @pytest.mark.asyncio
    async def test_research_runner(self):
        from caveman.tools.builtin.web_research import WebResearchRunner

        def fake_search(query):
            return {"data": {"web": [{"url": "https://example.com", "title": "Test"}]}}

        def fake_extract(urls):
            return [{"url": urls[0], "content": "Test content"}]

        runner = WebResearchRunner(search_fn=fake_search, extract_fn=fake_extract)
        session = await runner.research("test question", "test answer")
        assert len(session.steps) == 2
        assert session.steps[0].action == "search"
        assert session.steps[1].action == "extract"

    def test_sample_questions(self):
        from caveman.tools.builtin.web_research import SAMPLE_QUESTIONS
        assert len(SAMPLE_QUESTIONS) >= 3
        for q in SAMPLE_QUESTIONS:
            assert "question" in q
            assert "reference" in q
