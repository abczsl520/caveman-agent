"""Tests for fuzzy match, interrupt, budget config, credential files, channel directory, display config."""
import threading
import pytest

from caveman.tools.fuzzy_match import fuzzy_find_and_replace
from caveman.tools.interrupt import set_interrupt, is_interrupted, clear_all
from caveman.tools.budget_config import BudgetConfig, PINNED_THRESHOLDS
from caveman.tools.credential_files import register_credential_file, get_credential_file_mounts, clear_registered
from caveman.gateway.channel_directory import ChannelDirectory, ChannelEntry
from caveman.gateway.display_config import get_display_config, split_message, Platform


# ── Fuzzy Match ──

class TestFuzzyMatch:
    def test_exact_match(self):
        content = "def hello():\n    print('hi')"
        new, count, strategy, err = fuzzy_find_and_replace(content, "print('hi')", "print('hello')")
        assert err is None
        assert count == 1
        assert strategy == "exact"
        assert "print('hello')" in new

    def test_line_trimmed(self):
        content = "  def foo():\n      pass"
        new, count, strategy, err = fuzzy_find_and_replace(content, "def foo():\n    pass", "def bar():\n    pass")
        assert err is None
        assert count >= 1

    def test_whitespace_normalized(self):
        content = "x  =   1"
        new, count, strategy, err = fuzzy_find_and_replace(content, "x = 1", "x = 2")
        assert err is None

    def test_no_match(self):
        content = "completely different"
        new, count, strategy, err = fuzzy_find_and_replace(content, "not here at all xyz", "replacement")
        assert count == 0
        assert err is not None

    def test_empty_search(self):
        _, _, _, err = fuzzy_find_and_replace("content", "", "new")
        assert err is not None

    def test_unicode_normalization(self):
        content = 'print(\u201chello\u201d)'
        new, count, _, err = fuzzy_find_and_replace(content, 'print("hello")', 'print("world")')
        assert err is None

    def test_escape_normalized(self):
        content = "line1\nline2"
        new, count, strategy, err = fuzzy_find_and_replace(content, "line1\\nline2", "replaced")
        assert err is None


# ── Interrupt ──

class TestInterrupt:
    def setup_method(self):
        clear_all()

    def test_not_interrupted_by_default(self):
        assert is_interrupted() is False

    def test_set_and_check(self):
        set_interrupt(True)
        assert is_interrupted() is True

    def test_clear(self):
        set_interrupt(True)
        set_interrupt(False)
        assert is_interrupted() is False

    def test_thread_isolation(self):
        set_interrupt(True)
        result = [None]

        def check():
            result[0] = is_interrupted()

        t = threading.Thread(target=check)
        t.start()
        t.join()
        assert result[0] is False  # Other thread not interrupted


# ── Budget Config ──

class TestBudgetConfig:
    def test_defaults(self):
        cfg = BudgetConfig()
        assert cfg.default_result_size == 100_000
        assert cfg.turn_budget == 200_000

    def test_pinned_threshold(self):
        cfg = BudgetConfig()
        assert cfg.resolve_threshold("read_file") == float("inf")

    def test_tool_override(self):
        cfg = BudgetConfig(tool_overrides={"bash": 50_000})
        assert cfg.resolve_threshold("bash") == 50_000

    def test_should_persist(self):
        cfg = BudgetConfig(default_result_size=1000)
        assert cfg.should_persist("bash", 2000) is True
        assert cfg.should_persist("bash", 500) is False
        assert cfg.should_persist("read_file", 999999) is False  # Pinned


# ── Credential Files ──

class TestCredentialFiles:
    def setup_method(self):
        clear_registered()

    def test_register_and_get(self):
        register_credential_file("/home/user/.ssh/id_rsa", "/root/.ssh/id_rsa")
        mounts = get_credential_file_mounts()
        assert "/home/user/.ssh/id_rsa" in mounts


# ── Channel Directory ──

class TestChannelDirectory:
    def test_register_and_lookup(self, tmp_path):
        cd = ChannelDirectory(data_dir=tmp_path)
        entry = ChannelEntry(platform="discord", chat_id="123", session_id="s1", display_name="test")
        cd.register(entry)
        found = cd.lookup("discord", "123")
        assert found is not None
        assert found.session_id == "s1"

    def test_lookup_miss(self, tmp_path):
        cd = ChannelDirectory(data_dir=tmp_path)
        assert cd.lookup("discord", "999") is None

    def test_find_by_platform(self, tmp_path):
        cd = ChannelDirectory(data_dir=tmp_path)
        cd.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        cd.register(ChannelEntry(platform="telegram", chat_id="2", session_id="s2"))
        assert len(cd.find_by_platform("discord")) == 1

    def test_find_by_session(self, tmp_path):
        cd = ChannelDirectory(data_dir=tmp_path)
        cd.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        cd.register(ChannelEntry(platform="telegram", chat_id="2", session_id="s1"))
        assert len(cd.find_by_session("s1")) == 2

    def test_remove(self, tmp_path):
        cd = ChannelDirectory(data_dir=tmp_path)
        cd.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        assert cd.remove("discord", "1") is True
        assert cd.lookup("discord", "1") is None

    def test_persistence(self, tmp_path):
        cd1 = ChannelDirectory(data_dir=tmp_path)
        cd1.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        cd2 = ChannelDirectory(data_dir=tmp_path)
        assert cd2.lookup("discord", "1") is not None


# ── Display Config ──

class TestDisplayConfig:
    def test_discord(self):
        cfg = get_display_config("discord")
        assert cfg.max_message_length == 2000
        assert cfg.supports_embeds is True

    def test_telegram(self):
        cfg = get_display_config("telegram")
        assert cfg.max_message_length == 4096

    def test_unknown_platform(self):
        cfg = get_display_config("unknown")
        assert cfg.platform == Platform.CLI

    def test_split_message_short(self):
        chunks = split_message("short", 100)
        assert chunks == ["short"]

    def test_split_message_long(self):
        text = "line1\nline2\nline3\nline4"
        chunks = split_message(text, 12)
        assert len(chunks) >= 2
        assert all(len(c) <= 12 for c in chunks)

    def test_split_no_limit(self):
        text = "x" * 10000
        chunks = split_message(text, 0)
        assert chunks == [text]
