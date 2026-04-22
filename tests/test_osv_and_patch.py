"""Tests for OSV malware check and V4A patch parser."""
import pytest
from unittest.mock import patch, MagicMock
import json

from caveman.security.osv_check import (
    check_package_for_malware,
    _extract_package_info,
    _is_malware,
)
from caveman.tools.patch_parser import (
    parse_v4a_patch,
    apply_v4a_operations,
    PatchOp,
)


# ── OSV Check Tests ──

class TestExtractPackageInfo:
    def test_npx(self):
        name, eco = _extract_package_info("npx", ["@modelcontextprotocol/server-github"])
        assert name == "@modelcontextprotocol/server-github"
        assert eco == "npm"

    def test_pip(self):
        name, eco = _extract_package_info("pip", ["install", "requests>=2.0"])
        assert name == "requests"
        assert eco == "PyPI"

    def test_uvx(self):
        name, eco = _extract_package_info("uvx", ["mcp-server-fetch"])
        assert name == "mcp-server-fetch"
        assert eco == "PyPI"

    def test_unknown_command(self):
        name, eco = _extract_package_info("cargo", ["install", "something"])
        assert name is None

    def test_no_args(self):
        name, eco = _extract_package_info("npx", ["-y"])
        assert name is None


class TestIsMalware:
    def test_malware_id(self):
        assert _is_malware({"id": "MAL-2024-1234"}) is True

    def test_cve_id(self):
        assert _is_malware({"id": "CVE-2024-1234"}) is False

    def test_ghsa_id(self):
        assert _is_malware({"id": "GHSA-xxxx-yyyy"}) is False


class TestCheckPackage:
    def test_clean_package(self):
        with patch("caveman.security.osv_check._query_osv", return_value=[]):
            result = check_package_for_malware("npx", ["safe-package"])
        assert result is None

    def test_malware_detected(self):
        with patch("caveman.security.osv_check._query_osv", return_value=[
            {"id": "MAL-2024-001", "summary": "Malicious package"}
        ]):
            result = check_package_for_malware("npx", ["evil-package"])
        assert result is not None
        assert "MALWARE" in result

    def test_network_error_failopen(self):
        with patch("caveman.security.osv_check._query_osv", side_effect=Exception("timeout")):
            result = check_package_for_malware("npx", ["some-package"])
        assert result is None  # Fail-open


# ── V4A Patch Parser Tests ──

class TestParseV4A:
    def test_update_file(self):
        patch_text = """*** Begin Patch
*** Update File: src/main.py
 def hello():
-    print("old")
+    print("new")
*** End Patch"""
        ops, err = parse_v4a_patch(patch_text)
        assert err is None
        assert len(ops) == 1
        assert ops[0].op == PatchOp.UPDATE
        assert ops[0].path == "src/main.py"
        assert len(ops[0].hunks) == 1
        assert ops[0].hunks[0]["removes"] == ['    print("old")']
        assert ops[0].hunks[0]["adds"] == ['    print("new")']

    def test_add_file(self):
        patch_text = """*** Begin Patch
*** Add File: new_file.py
+print("hello")
+print("world")
*** End Patch"""
        ops, err = parse_v4a_patch(patch_text)
        assert err is None
        assert len(ops) == 1
        assert ops[0].op == PatchOp.ADD
        assert 'print("hello")' in ops[0].content

    def test_delete_file(self):
        patch_text = """*** Begin Patch
*** Delete File: old_file.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch_text)
        assert err is None
        assert len(ops) == 1
        assert ops[0].op == PatchOp.DELETE

    def test_move_file(self):
        patch_text = """*** Begin Patch
*** Move File: old/path.py -> new/path.py
*** End Patch"""
        ops, err = parse_v4a_patch(patch_text)
        assert err is None
        assert len(ops) == 1
        assert ops[0].op == PatchOp.MOVE
        assert ops[0].path == "old/path.py"
        assert ops[0].new_path == "new/path.py"

    def test_empty_patch(self):
        ops, err = parse_v4a_patch("nothing here")
        assert len(ops) == 0
        assert err is not None

    def test_multiple_operations(self):
        patch_text = """*** Begin Patch
*** Add File: a.py
+content
*** Delete File: b.py
*** Update File: c.py
-old
+new
*** End Patch"""
        ops, err = parse_v4a_patch(patch_text)
        assert err is None
        assert len(ops) == 3


class TestApplyV4A:
    def test_apply_add(self, tmp_path):
        from caveman.tools.patch_parser import PatchOperation
        ops = [PatchOperation(op=PatchOp.ADD, path="new.py", content="hello\n")]
        result = apply_v4a_operations(ops, tmp_path)
        assert "add: new.py" in result["applied"]
        assert (tmp_path / "new.py").read_text() == "hello\n"

    def test_apply_delete(self, tmp_path):
        (tmp_path / "old.py").write_text("bye")
        from caveman.tools.patch_parser import PatchOperation
        ops = [PatchOperation(op=PatchOp.DELETE, path="old.py")]
        result = apply_v4a_operations(ops, tmp_path)
        assert "delete: old.py" in result["applied"]
        assert not (tmp_path / "old.py").exists()

    def test_apply_move(self, tmp_path):
        (tmp_path / "old.py").write_text("content")
        from caveman.tools.patch_parser import PatchOperation
        ops = [PatchOperation(op=PatchOp.MOVE, path="old.py", new_path="new.py")]
        result = apply_v4a_operations(ops, tmp_path)
        assert not (tmp_path / "old.py").exists()
        assert (tmp_path / "new.py").read_text() == "content"

    def test_dry_run(self, tmp_path):
        from caveman.tools.patch_parser import PatchOperation
        ops = [PatchOperation(op=PatchOp.ADD, path="new.py", content="hello")]
        result = apply_v4a_operations(ops, tmp_path, dry_run=True)
        assert "add: new.py" in result["applied"]
        assert not (tmp_path / "new.py").exists()
