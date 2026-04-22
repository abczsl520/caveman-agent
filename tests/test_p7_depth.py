"""Tests for P7 depth: skills hub depth, dispatch depth, agent runner depth, status depth."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Skills Hub Depth Tests ──

class TestSkillsHubDepth:
    def test_validate_skill_name(self):
        from caveman.tools.builtin.skills_hub_depth import validate_skill_name
        assert validate_skill_name("my-skill") == "my-skill"
        assert validate_skill_name("  My_Skill  ") == "my_skill"
        with pytest.raises(ValueError):
            validate_skill_name("")
        with pytest.raises(ValueError):
            validate_skill_name("../evil")
        with pytest.raises(ValueError):
            validate_skill_name("a" * 100)

    def test_validate_bundle_path(self):
        from caveman.tools.builtin.skills_hub_depth import validate_bundle_path
        assert validate_bundle_path("SKILL.md") == "SKILL.md"
        with pytest.raises(ValueError):
            validate_bundle_path("../escape")
        with pytest.raises(ValueError):
            validate_bundle_path("nested/path", allow_nested=False)
        assert validate_bundle_path("nested/path", allow_nested=True) == "nested/path"

    def test_parse_frontmatter(self):
        from caveman.tools.builtin.skills_hub_depth import parse_frontmatter
        content = "---\nname: test\ntags: [a, b, c]\n---\n# Content"
        fm = parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["tags"] == ["a", "b", "c"]

    def test_parse_frontmatter_no_frontmatter(self):
        from caveman.tools.builtin.skills_hub_depth import parse_frontmatter
        assert parse_frontmatter("# Just content") == {}

    def test_audit_bundle_security(self):
        from caveman.tools.builtin.skills_hub_depth import audit_bundle_security
        files = {
            "run.py": "import os\nos.system('rm -rf /')\nprint('hello')",
            "safe.md": "# Safe content",
        }
        findings = audit_bundle_security(files)
        assert len(findings) >= 1
        assert findings[0]["file"] == "run.py"

    def test_audit_clean_bundle(self):
        from caveman.tools.builtin.skills_hub_depth import audit_bundle_security
        files = {"SKILL.md": "# My Skill\nA safe skill"}
        assert audit_bundle_security(files) == []

    @pytest.mark.asyncio
    async def test_parallel_search(self):
        from caveman.tools.builtin.skills_hub_depth import parallel_search
        from caveman.tools.builtin.skills_hub import SkillMeta

        class FakeSource:
            source_id = "fake"
            def search(self, query, limit=10):
                return [SkillMeta(name=f"skill-{self.source_id}")]

        sources = [FakeSource(), FakeSource()]
        sources[1].source_id = "fake2"
        results = await parallel_search("test", sources)
        assert len(results) == 2

    def test_github_auth_no_token(self):
        from caveman.tools.builtin.skills_hub_depth import GitHubAuth
        import os
        # Save and clear
        saved = {k: os.environ.pop(k, None) for k in ("GITHUB_TOKEN", "GH_TOKEN")}
        try:
            auth = GitHubAuth()
            headers = auth.get_headers()
            assert "Accept" in headers
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


# ── Dispatch Depth Tests ──

class TestDispatchDepth:
    @pytest.mark.asyncio
    async def test_streaming_dispatcher_text(self):
        from caveman.gateway.dispatch_depth import StreamingDispatcher, StreamChunk
        sent = []
        async def send(text):
            sent.append(text)
            return {"message_id": "m1"}
        dispatcher = StreamingDispatcher(send_fn=send)
        await dispatcher.handle_chunk(StreamChunk(text="hello ", is_final=False))
        await dispatcher.handle_chunk(StreamChunk(text="world", is_final=True))
        assert len(sent) == 1
        assert "hello world" in sent[0]

    @pytest.mark.asyncio
    async def test_streaming_dispatcher_tool_status(self):
        from caveman.gateway.dispatch_depth import StreamingDispatcher, StreamChunk
        dispatcher = StreamingDispatcher()
        await dispatcher.handle_chunk(StreamChunk(
            tool_name="bash", tool_status="start",
        ))
        stats = dispatcher.get_stats()
        assert stats["tool_statuses"]["bash"] == "start"

    def test_block_reply_config(self):
        from caveman.gateway.dispatch_depth import BlockReplyConfig
        config = BlockReplyConfig(min_chars=100, max_chars=500)
        assert config.break_preference == "paragraph"


# ── Agent Runner Depth Tests ──

class TestAgentRunnerDepth:
    def test_tool_progress(self):
        from caveman.gateway.agent_runner_depth import ToolProgress
        import time
        tp = ToolProgress(tool_name="bash", started_at=time.monotonic() - 1)
        assert tp.duration_ms > 900

    def test_stream_event(self):
        from caveman.gateway.agent_runner_depth import StreamEvent
        event = StreamEvent(type="text", text="hello")
        assert event.type == "text"

    @pytest.mark.asyncio
    async def test_streaming_runner_no_fn(self):
        from caveman.gateway.agent_runner_depth import StreamingAgentRunner
        runner = StreamingAgentRunner()
        events = []
        async for event in runner.run_streaming("s1", "hello"):
            events.append(event)
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_streaming_runner_with_fn(self):
        from caveman.gateway.agent_runner_depth import StreamingAgentRunner

        async def fake_stream(**kwargs):
            yield {"type": "text", "text": "hello "}
            yield {"type": "text", "text": "world"}
            yield {"type": "usage", "total_tokens": 100}

        runner = StreamingAgentRunner(agent_stream_fn=fake_stream)
        events = []
        async for event in runner.run_streaming("s1", "hi"):
            events.append(event)
        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 2

    @pytest.mark.asyncio
    async def test_cancel(self):
        from caveman.gateway.agent_runner_depth import StreamingAgentRunner
        runner = StreamingAgentRunner()
        await runner.cancel()  # Should not raise
        assert True  # Cancel on idle runner is safe


# ── Status Depth Tests ──

class TestStatusDepth:
    def test_token_stats_update(self):
        from caveman.gateway.status_depth import TokenStats
        stats = TokenStats()
        stats.update(1000, 500, 200)
        assert stats.total_tokens == 1500
        assert stats.api_calls == 1
        assert stats.avg_latency_ms == 200

    def test_token_stats_cost(self):
        from caveman.gateway.status_depth import TokenStats
        stats = TokenStats(prompt_tokens=1000000, completion_tokens=100000)
        cost = stats.estimate_cost("claude-opus-4-6")
        assert cost > 0
        assert stats.cost_usd == cost

    def test_model_info(self):
        from caveman.gateway.status_depth import get_model_info
        info = get_model_info("claude-opus-4-6")
        assert info.context_window == 200000
        assert info.supports_vision

    def test_model_info_unknown(self):
        from caveman.gateway.status_depth import get_model_info
        info = get_model_info("unknown-model")
        assert info.model == "unknown-model"

    def test_format_session_list(self):
        from caveman.gateway.status_depth import format_session_list, SessionListEntry
        import time
        sessions = [
            SessionListEntry(
                session_key="discord:channel:123",
                model="claude-opus-4-6",
                total_tokens=5000,
                messages=10,
                last_activity=time.time(),
                is_active=True,
            ),
        ]
        text = format_session_list(sessions)
        assert "discord" in text
        assert "5,000" in text

    def test_format_session_list_empty(self):
        from caveman.gateway.status_depth import format_session_list
        assert "No active" in format_session_list([])

    def test_format_token_stats(self):
        from caveman.gateway.status_depth import format_token_stats, TokenStats
        stats = TokenStats(prompt_tokens=10000, completion_tokens=5000, total_tokens=15000, api_calls=3)
        text = format_token_stats(stats, "claude-opus-4-6")
        assert "15,000" in text
        assert "$" in text

    def test_format_model_info(self):
        from caveman.gateway.status_depth import format_model_info, get_model_info
        info = get_model_info("gpt-4o")
        text = format_model_info(info)
        assert "128,000" in text
        assert "vision" in text
