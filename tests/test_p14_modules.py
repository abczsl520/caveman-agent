"""Tests for P14: memory_provider, memory_manager, file_tools, interactive,
skills_sync, mixture_of_agents, copilot_client, subdirectory_hints, flows."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── Memory Provider ──

class TestMemoryProvider:
    def test_abstract_interface(self):
        from caveman.agent.memory_provider import MemoryProvider
        # Can't instantiate abstract class
        with pytest.raises(TypeError):
            MemoryProvider()

    def test_concrete_provider(self):
        from caveman.agent.memory_provider import MemoryProvider

        class TestProvider(MemoryProvider):
            @property
            def name(self) -> str:
                return "test"
            def is_available(self) -> bool:
                return True
            def initialize(self, session_id: str, **kwargs) -> None:
                pass
            def get_tool_schemas(self) -> list:
                return [{"name": "test_recall", "description": "test"}]

        p = TestProvider()
        assert p.name == "test"
        assert p.is_available()
        assert p.prefetch("query") == ""
        assert len(p.get_tool_schemas()) == 1


# ── Memory Manager ──

class TestMemoryManager:
    def _make_provider(self, name="test", available=True):
        from caveman.agent.memory_provider import MemoryProvider

        class P(MemoryProvider):
            @property
            def name(self) -> str:
                return name
            def is_available(self) -> bool:
                return available
            def initialize(self, session_id, **kw):
                pass
            def get_tool_schemas(self):
                return []
            def prefetch(self, query, *, session_id=""):
                return f"recalled: {query}"
        return P()

    def test_add_provider(self):
        from caveman.agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        p = self._make_provider("builtin")
        assert mgr.add_provider(p)
        assert "builtin" in mgr.provider_names

    def test_reject_unavailable(self):
        from caveman.agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        p = self._make_provider("bad", available=False)
        assert not mgr.add_provider(p)

    def test_single_external_limit(self):
        from caveman.agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        mgr.add_provider(self._make_provider("ext1"))
        assert not mgr.add_provider(self._make_provider("ext2"))

    def test_prefetch_all(self):
        from caveman.agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        mgr.add_provider(self._make_provider("builtin"))
        result = mgr.prefetch_all("hello")
        assert "recalled: hello" in result

    def test_build_system_prompt(self):
        from caveman.agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        mgr.add_provider(self._make_provider("builtin"))
        # Default system_prompt_block returns ""
        assert mgr.build_system_prompt() == ""


# ── File Tools ──

class TestFileTools:
    def test_read_file(self, tmp_path):
        from caveman.tools.builtin.file_tools import read_file
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3")
        result = read_file(str(f))
        assert result["total_lines"] == 3
        assert "line1" in result["content"]

    def test_read_file_not_found(self):
        from caveman.tools.builtin.file_tools import read_file
        result = read_file("/nonexistent.txt")
        assert "error" in result

    def test_read_binary_blocked(self, tmp_path):
        from caveman.tools.builtin.file_tools import read_file
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG")
        result = read_file(str(f))
        assert "Binary" in result.get("error", "")

    def test_search_in_file(self, tmp_path):
        from caveman.tools.builtin.file_tools import search_in_file
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    pass\ndef world():\n    pass")
        result = search_in_file("def ", str(f))
        assert result["total_count"] == 2

    def test_replace_in_file(self, tmp_path):
        from caveman.tools.builtin.file_tools import replace_in_file
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = replace_in_file(str(f), "hello", "goodbye")
        assert result["replacements"] == 1
        assert "goodbye" in f.read_text()

    def test_create_file(self, tmp_path):
        from caveman.tools.builtin.file_tools import create_file
        path = str(tmp_path / "new" / "file.txt")
        result = create_file(path, "content")
        assert result["success"] is True
        assert Path(path).exists()

    def test_patch_file(self, tmp_path):
        from caveman.tools.builtin.file_tools import patch_file
        f = tmp_path / "test.txt"
        f.write_text("aaa bbb ccc")
        result = patch_file(str(f), [{"old": "aaa", "new": "xxx"}, {"old": "ccc", "new": "zzz"}])
        assert result["applied"] == 2
        assert f.read_text() == "xxx bbb zzz"

    def test_blocked_device(self):
        from caveman.tools.builtin.file_tools import is_blocked_device
        assert is_blocked_device("/dev/zero")
        assert not is_blocked_device("/tmp/safe.txt")


# ── Interactive ──

class TestInteractive:
    def test_button_row(self):
        from caveman.gateway.interactive import ButtonRow, Button
        row = ButtonRow()
        row.add("Yes", "yes").add("No", "no")
        assert len(row.buttons) == 2

    def test_interactive_message(self):
        from caveman.gateway.interactive import InteractiveMessage, Button
        msg = InteractiveMessage(content="Choose:")
        msg.add_buttons(Button("A", "a"), Button("B", "b"))
        assert len(msg.button_rows) == 1

    def test_text_fallback(self):
        from caveman.gateway.interactive import InteractiveMessage, Button, render_text_fallback
        msg = InteractiveMessage(content="Pick one:")
        msg.add_buttons(Button("Yes"), Button("No"))
        text = render_text_fallback(msg)
        assert "Yes" in text and "No" in text

    def test_discord_components(self):
        from caveman.gateway.interactive import InteractiveMessage, Button, render_discord_components
        msg = InteractiveMessage(content="Test")
        msg.add_buttons(Button("OK", "ok", style="primary"))
        components = render_discord_components(msg)
        assert len(components) == 1
        assert components[0]["components"][0]["style"] == 1


# ── Skills Sync ──

class TestSkillsSync:
    def test_scan_local_skills(self, tmp_path):
        from caveman.tools.builtin.skills_sync import scan_local_skills
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nSome description")
        manifests = scan_local_skills(tmp_path)
        assert len(manifests) == 1
        assert manifests[0].name == "My Skill"
        assert manifests[0].checksum  # non-empty

    def test_diff_skills(self):
        from caveman.tools.builtin.skills_sync import diff_skills, SkillManifest
        local = [SkillManifest(name="a", path="/a", checksum="111")]
        remote = [SkillManifest(name="a", path="/a", checksum="222"), SkillManifest(name="b", path="/b", checksum="333")]
        diff = diff_skills(local, remote)
        assert "b" in diff.added
        assert "a" in diff.updated

    def test_sync_state_persistence(self, tmp_path):
        from caveman.tools.builtin.skills_sync import SyncState, save_sync_state, load_sync_state
        state = SyncState(last_sync=time.time(), synced_skills={"test": "abc123"})
        state_file = tmp_path / "sync.json"
        save_sync_state(state, state_file)
        loaded = load_sync_state(state_file)
        assert "test" in loaded.synced_skills


# ── Mixture of Agents ──

class TestMixtureOfAgents:
    def test_moa_response_dataclass(self):
        from caveman.tools.builtin.mixture_of_agents import MoAResponse
        r = MoAResponse(model="gpt-4o", content="hello", success=True, latency_seconds=0.1)
        assert r.model == "gpt-4o"
        assert r.success

    def test_moa_result_dataclass(self):
        from caveman.tools.builtin.mixture_of_agents import MoAResult
        r = MoAResult(success=True, response="synthesized", aggregator_model="claude")
        assert r.response == "synthesized"


# ── Copilot ACP Client ──

class TestCopilotClient:
    def test_create_session(self):
        from caveman.acp.copilot_client import CopilotACPClient
        client = CopilotACPClient()
        session = client.create_session(session_id="test-1")
        assert session.session_id == "test-1"
        assert session.status == "active"

    def test_list_sessions(self):
        from caveman.acp.copilot_client import CopilotACPClient
        client = CopilotACPClient()
        client.create_session(session_id="s1")
        client.create_session(session_id="s2")
        assert len(client.list_sessions()) == 2

    def test_close_session(self):
        from caveman.acp.copilot_client import CopilotACPClient
        client = CopilotACPClient()
        s = client.create_session(session_id="s1")
        assert client.close_session(s.session_id)
        assert client.get_session(s.session_id).status == "closed"


# ── Subdirectory Hints ──

class TestSubdirectoryHints:
    def test_generate_hints(self, tmp_path):
        from caveman.agent.subdirectory_hints import generate_hints
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("pass")
        hints = generate_hints(str(tmp_path))
        assert len(hints) >= 2

    def test_format_hints(self, tmp_path):
        from caveman.agent.subdirectory_hints import generate_hints, format_hints_for_prompt
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("")
        hints = generate_hints(str(tmp_path))
        text = format_hints_for_prompt(hints)
        assert "Project structure" in text

    def test_skip_dirs(self, tmp_path):
        from caveman.agent.subdirectory_hints import generate_hints
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg").mkdir()
        (tmp_path / "src").mkdir()
        hints = generate_hints(str(tmp_path))
        paths = [h.path for h in hints]
        assert not any("node_modules" in p for p in paths)


# ── Flows ──

class TestFlows:
    def test_create_flow(self):
        from caveman.gateway.flows import Flow
        flow = Flow(name="test")
        flow.add_step("step1", "echo")
        flow.add_step("step2", "echo", depends_on=["step_0"])
        assert len(flow.steps) == 2

    def test_execute_simple_flow(self):
        from caveman.gateway.flows import Flow, FlowEngine, StepStatus
        engine = FlowEngine()
        engine.register_handler("echo", lambda args, ctx: "done")

        flow = Flow(name="simple")
        flow.add_step("greet", "echo")

        result = asyncio.run(engine.execute(flow))
        assert result.status == StepStatus.COMPLETED
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_execute_with_deps(self):
        from caveman.gateway.flows import Flow, FlowEngine, StepStatus
        order = []
        def step_a(args, ctx):
            order.append("a")
            return "a_done"
        def step_b(args, ctx):
            order.append("b")
            return "b_done"

        engine = FlowEngine()
        engine.register_handler("a", step_a)
        engine.register_handler("b", step_b)

        flow = Flow(name="deps")
        flow.add_step("first", "a")
        flow.add_step("second", "b", depends_on=["step_0"])

        result = asyncio.run(engine.execute(flow))
        assert result.status == StepStatus.COMPLETED
        assert order == ["a", "b"]

    def test_condition_skip(self):
        from caveman.gateway.flows import Flow, FlowEngine, StepStatus
        engine = FlowEngine()
        engine.register_handler("noop", lambda a, c: None)

        flow = Flow(name="cond")
        flow.context["skip"] = True
        flow.add_step("maybe", "noop", condition="not skip")

        result = asyncio.run(engine.execute(flow))
        assert result.steps[0].status == StepStatus.SKIPPED

    def test_handler_not_found(self):
        from caveman.gateway.flows import Flow, FlowEngine, StepStatus
        engine = FlowEngine()
        flow = Flow(name="missing")
        flow.add_step("bad", "nonexistent_handler")
        result = asyncio.run(engine.execute(flow))
        assert result.steps[0].status == StepStatus.FAILED
