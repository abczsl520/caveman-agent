"""Tests for P11: prompt_builder, context_compressor, smart_routing, prompt_caching, context_references, approval, code_execution, skills_guard, transcription, image_gen, checkpoint."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


# ── Prompt Builder ──

class TestPromptBuilder:
    def test_strip_yaml_frontmatter(self):
        from caveman.agent.prompt_builder import strip_yaml_frontmatter
        content = "---\ntitle: test\n---\n# Hello"
        assert strip_yaml_frontmatter(content) == "# Hello"

    def test_no_frontmatter(self):
        from caveman.agent.prompt_builder import strip_yaml_frontmatter
        assert strip_yaml_frontmatter("# Hello") == "# Hello"

    def test_truncate_content(self):
        from caveman.agent.prompt_builder import truncate_content
        short = "hello"
        assert truncate_content(short, "test.md") == "hello"
        long = "x" * 30000
        result = truncate_content(long, "test.md", max_chars=100)
        assert len(result) < 200
        assert "truncated" in result

    def test_build_environment_hints(self):
        from caveman.agent.prompt_builder import build_environment_hints
        hints = build_environment_hints()
        assert "OS:" in hints
        assert "Python:" in hints

    def test_build_skills_manifest_empty(self, tmp_path):
        from caveman.agent.prompt_builder import build_skills_manifest
        skills = build_skills_manifest(tmp_path / "nonexistent")
        assert skills == []

    def test_build_skills_manifest(self, tmp_path):
        from caveman.agent.prompt_builder import build_skills_manifest
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\ndescription: A test skill\n---\n# Test")
        skills = build_skills_manifest(tmp_path / "skills")
        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_find_git_root(self, tmp_path):
        from caveman.agent.prompt_builder import find_git_root
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert find_git_root(sub) == tmp_path


# ── Context Compressor ──

class TestContextCompressor:
    def test_should_compress_small(self):
        from caveman.agent.context_compressor import ContextCompressor
        comp = ContextCompressor(model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        assert not comp.should_compress(messages)

    def test_compress_with_fallback(self):
        from caveman.agent.context_compressor import ContextCompressor
        comp = ContextCompressor(model="gpt-4o", threshold_percent=0.001, protect_first_n=2, tail_token_budget=30)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Setup question"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 200}
            for i in range(10)
        ] + [
            {"role": "user", "content": "Latest"},
        ]
        result = comp.compress(messages)
        # Summary should be generated and message count reduced
        assert result.summary
        assert result.compacted_count < result.original_count

    def test_prune_tool_results(self):
        from caveman.agent.context_compressor import ContextCompressor
        comp = ContextCompressor(model="gpt-4o", tail_token_budget=20)
        messages = [
            {"role": "user", "content": "run something"},
            {"role": "tool", "content": "x" * 500},
            {"role": "assistant", "content": "done " * 50},
            {"role": "user", "content": "another thing"},
            {"role": "tool", "content": "y" * 500},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "latest"},
        ]
        pruned, count = comp._prune_tool_results(messages)
        assert count >= 1

    def test_reset(self):
        from caveman.agent.context_compressor import ContextCompressor
        comp = ContextCompressor(model="gpt-4o")
        comp._compression_count = 5
        comp.reset()
        assert comp._compression_count == 0


# ── Smart Model Routing ──

class TestSmartRouting:
    def test_simple_message(self):
        from caveman.agent.smart_model_routing import classify_message_complexity
        assert classify_message_complexity("hello") == "simple"
        assert classify_message_complexity("what time is it?") == "simple"

    def test_complex_message(self):
        from caveman.agent.smart_model_routing import classify_message_complexity
        assert classify_message_complexity("debug this error in the code") == "complex"
        assert classify_message_complexity("implement a new feature") == "complex"
        assert classify_message_complexity("x" * 200) == "complex"

    def test_code_indicators(self):
        from caveman.agent.smart_model_routing import classify_message_complexity
        assert classify_message_complexity("check `this`") == "complex"
        assert classify_message_complexity("```code```") == "complex"

    def test_choose_route_disabled(self):
        from caveman.agent.smart_model_routing import choose_route, RoutingConfig
        config = RoutingConfig(enabled=False)
        result = choose_route("hello", "claude-opus-4-6", config)
        assert result.model == "claude-opus-4-6"
        assert not result.is_cheap

    def test_choose_route_simple(self):
        from caveman.agent.smart_model_routing import choose_route, RoutingConfig
        config = RoutingConfig(enabled=True, cheap_model="gpt-4o-mini", cheap_provider="openai")
        result = choose_route("hello", "claude-opus-4-6", config)
        assert result.model == "gpt-4o-mini"
        assert result.is_cheap


# ── Prompt Caching ──

class TestPromptCaching:
    def test_cache_hit(self):
        from caveman.agent.prompt_caching import PromptCache
        cache = PromptCache()
        entry1 = cache.get_or_create("system prompt content")
        entry2 = cache.get_or_create("system prompt content")
        assert entry1.hash == entry2.hash
        assert entry2.hit_count == 1

    def test_cache_eviction(self):
        from caveman.agent.prompt_caching import PromptCache
        cache = PromptCache(max_entries=2)
        cache.get_or_create("a")
        cache.get_or_create("b")
        cache.get_or_create("c")
        assert cache.stats()["entries"] == 2

    def test_apply_cache_control(self):
        from caveman.agent.prompt_caching import PromptCache
        cache = PromptCache()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "x" * 2000},
        ]
        result = cache.apply_cache_control(messages)
        assert result[0]["content"][0]["cache_control"]["type"] == "ephemeral"


# ── Context References ──

class TestContextReferences:
    def test_detect_file_refs(self):
        from caveman.agent.context_references import detect_references
        refs = detect_references("Look at @main.py and @utils.py:10-20")
        file_refs = [r for r in refs if r.type == "file"]
        assert len(file_refs) == 2

    def test_detect_url_refs(self):
        from caveman.agent.context_references import detect_references
        refs = detect_references("Check https://example.com/page")
        url_refs = [r for r in refs if r.type == "url"]
        assert len(url_refs) == 1

    def test_resolve_file(self, tmp_path):
        from caveman.agent.context_references import Reference, resolve_file_reference
        (tmp_path / "test.py").write_text("print('hello')")
        ref = Reference(type="file", raw="test.py")
        resolved = resolve_file_reference(ref, str(tmp_path))
        assert resolved.content == "print('hello')"

    def test_resolve_missing_file(self, tmp_path):
        from caveman.agent.context_references import Reference, resolve_file_reference
        ref = Reference(type="file", raw="nonexistent.py")
        resolved = resolve_file_reference(ref, str(tmp_path))
        assert resolved.error


# ── Approval ──

class TestApproval:
    def test_detect_dangerous(self):
        from caveman.tools.builtin.approval import detect_dangerous_command
        result = detect_dangerous_command("rm -rf /")
        assert result.is_dangerous
        assert result.pattern_key == "rm_root"

    def test_safe_command(self):
        from caveman.tools.builtin.approval import detect_dangerous_command
        result = detect_dangerous_command("ls -la")
        assert not result.is_dangerous

    def test_approval_manager(self):
        from caveman.tools.builtin.approval import ApprovalManager
        mgr = ApprovalManager()
        assert not mgr.is_approved("s1", "rm_recursive")
        mgr.approve("s1", "rm_recursive", "allow-once")
        assert mgr.is_approved("s1", "rm_recursive")
        # Consumed
        assert not mgr.is_approved("s1", "rm_recursive")

    def test_yolo_mode(self):
        from caveman.tools.builtin.approval import ApprovalManager
        mgr = ApprovalManager()
        mgr.enable_yolo("s1")
        assert mgr.is_approved("s1", "anything")
        mgr.disable_yolo("s1")
        assert not mgr.is_approved("s1", "anything")

    def test_check_and_request(self):
        from caveman.tools.builtin.approval import ApprovalManager
        mgr = ApprovalManager()
        result = mgr.check_and_request("s1", "rm -rf /tmp/test")
        assert not result["approved"]
        assert result["pending"]


# ── Code Execution ──

class TestCodeExecution:
    def test_execute_python(self):
        from caveman.tools.builtin.code_execution import execute_code
        result = execute_code("print('hello')", "python")
        assert result.success
        assert "hello" in result.stdout

    def test_execute_bash(self):
        from caveman.tools.builtin.code_execution import execute_code
        result = execute_code("echo world", "bash")
        assert result.success
        assert "world" in result.stdout

    def test_unsupported_language(self):
        from caveman.tools.builtin.code_execution import execute_code
        result = execute_code("code", "cobol")
        assert not result.success
        assert "Unsupported" in result.error

    def test_timeout(self):
        from caveman.tools.builtin.code_execution import execute_code, ExecutionConfig
        config = ExecutionConfig(timeout=1)
        result = execute_code("import time; time.sleep(10)", "python", config)
        assert result.timed_out

    def test_safety_check(self):
        from caveman.tools.builtin.code_execution import execute_code
        result = execute_code("import ctypes", "python")
        assert not result.success
        assert "ctypes" in result.error


# ── Skills Guard ──

class TestSkillsGuard:
    def test_scan_safe_bundle(self):
        from caveman.tools.builtin.skills_guard import scan_bundle
        files = {"SKILL.md": "# Safe Skill\nA helpful skill."}
        result = scan_bundle(files)
        assert result.passed

    def test_scan_dangerous_bundle(self):
        from caveman.tools.builtin.skills_guard import scan_bundle
        files = {"run.py": "import os\nos.system('rm -rf /')"}
        result = scan_bundle(files)
        assert not result.passed
        assert result.critical_count >= 1

    def test_quarantine(self, tmp_path):
        from caveman.tools.builtin.skills_guard import quarantine_skill
        import caveman.tools.builtin.skills_guard as mod
        old_dir = mod._QUARANTINE_DIR
        mod._QUARANTINE_DIR = tmp_path / "quarantine"
        try:
            path = quarantine_skill("evil-skill", {"bad.py": "os.system('hack')"}, "dangerous")
            assert path.exists()
            assert (path / "_quarantine.json").exists()
        finally:
            mod._QUARANTINE_DIR = old_dir

    def test_guard_install_safe(self):
        from caveman.tools.builtin.skills_guard import guard_install
        result = guard_install("safe-skill", {"SKILL.md": "# Safe"}, auto_quarantine=False)
        assert result["allowed"]

    def test_guard_install_dangerous(self, tmp_path):
        from caveman.tools.builtin.skills_guard import guard_install
        import caveman.tools.builtin.skills_guard as mod
        old_dir = mod._QUARANTINE_DIR
        mod._QUARANTINE_DIR = tmp_path / "quarantine"
        try:
            result = guard_install("evil", {"x.py": "subprocess.run(['rm', '-rf', '/'])"})
            assert not result["allowed"]
        finally:
            mod._QUARANTINE_DIR = old_dir


# ── Checkpoint Manager ──

class TestCheckpointManager:
    def test_save_and_restore(self, tmp_path):
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(base_dir=tmp_path / "checkpoints")
        messages = [{"role": "user", "content": "hello"}]
        cp = mgr.save("test:session", messages, description="test checkpoint")
        assert cp.id
        restored = mgr.restore("test:session")
        assert restored is not None
        assert len(restored.messages) == 1

    def test_list_checkpoints(self, tmp_path):
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(base_dir=tmp_path / "checkpoints")
        mgr.save("s1", [{"role": "user", "content": "a"}])
        mgr.save("s1", [{"role": "user", "content": "b"}])
        listed = mgr.list_checkpoints("s1")
        assert len(listed) == 2

    def test_delete(self, tmp_path):
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(base_dir=tmp_path / "checkpoints")
        cp = mgr.save("s1", [{"role": "user", "content": "x"}])
        assert mgr.delete("s1", cp.id)
        assert mgr.restore("s1") is None

    def test_max_per_session(self, tmp_path):
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(base_dir=tmp_path / "checkpoints", max_per_session=2)
        mgr.save("s1", [{"role": "user", "content": "1"}])
        mgr.save("s1", [{"role": "user", "content": "2"}])
        mgr.save("s1", [{"role": "user", "content": "3"}])
        listed = mgr.list_checkpoints("s1")
        assert len(listed) == 2
