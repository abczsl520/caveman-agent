"""Tests for gateway channel directory."""
import json
import pytest
from pathlib import Path

from caveman.gateway.channel_directory import ChannelDirectory, ChannelEntry


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "gateway"


@pytest.fixture
def directory(tmp_dir):
    return ChannelDirectory(data_dir=tmp_dir)


class TestChannelDirectory:
    def test_register_and_lookup(self, directory):
        entry = ChannelEntry(
            platform="discord", chat_id="123", session_id="s1",
            display_name="test-channel",
        )
        directory.register(entry)
        found = directory.lookup("discord", "123")
        assert found is not None
        assert found.session_id == "s1"
        assert found.display_name == "test-channel"

    def test_lookup_missing_returns_none(self, directory):
        assert directory.lookup("discord", "nonexistent") is None

    def test_find_by_session(self, directory):
        for i in range(3):
            directory.register(ChannelEntry(
                platform="discord", chat_id=str(i), session_id="shared",
            ))
        directory.register(ChannelEntry(
            platform="telegram", chat_id="99", session_id="other",
        ))
        results = directory.find_by_session("shared")
        assert len(results) == 3

    def test_find_by_platform(self, directory):
        directory.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        directory.register(ChannelEntry(platform="discord", chat_id="2", session_id="s2"))
        directory.register(ChannelEntry(platform="telegram", chat_id="3", session_id="s3"))
        assert len(directory.find_by_platform("discord")) == 2
        assert len(directory.find_by_platform("telegram")) == 1

    def test_remove(self, directory):
        directory.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))
        assert directory.remove("discord", "1")
        assert directory.lookup("discord", "1") is None

    def test_remove_nonexistent_returns_false(self, directory):
        assert not directory.remove("discord", "nonexistent")

    def test_persistence(self, tmp_dir):
        d1 = ChannelDirectory(data_dir=tmp_dir)
        d1.register(ChannelEntry(platform="discord", chat_id="1", session_id="s1"))

        d2 = ChannelDirectory(data_dir=tmp_dir)
        assert d2.lookup("discord", "1") is not None

    def test_thread_id_separation(self, directory):
        directory.register(ChannelEntry(
            platform="discord", chat_id="1", session_id="s1", thread_id="",
        ))
        directory.register(ChannelEntry(
            platform="discord", chat_id="1", session_id="s2", thread_id="t1",
        ))
        assert directory.lookup("discord", "1", "").session_id == "s1"
        assert directory.lookup("discord", "1", "t1").session_id == "s2"

    def test_all_entries(self, directory):
        directory.register(ChannelEntry(platform="a", chat_id="1", session_id="s1"))
        directory.register(ChannelEntry(platform="b", chat_id="2", session_id="s2"))
        assert len(directory.all_entries()) == 2

    def test_key_generation(self):
        assert ChannelDirectory._key("Discord", "123") == "discord:123"
        assert ChannelDirectory._key("Discord", "123", "t1") == "discord:123:t1"

    def test_update_existing(self, directory):
        directory.register(ChannelEntry(
            platform="discord", chat_id="1", session_id="old",
        ))
        directory.register(ChannelEntry(
            platform="discord", chat_id="1", session_id="new",
        ))
        assert directory.lookup("discord", "1").session_id == "new"
        assert len(directory.all_entries()) == 1
