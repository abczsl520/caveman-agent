"""Tests for gateway display config."""
import pytest

from caveman.gateway.display_config import (
    DisplayConfig,
    Platform,
    get_display_config,
    split_message,
    DISCORD_CONFIG,
    TELEGRAM_CONFIG,
    SLACK_CONFIG,
    CLI_CONFIG,
)


class TestPlatformEnum:
    def test_values(self):
        assert Platform.DISCORD == "discord"
        assert Platform.TELEGRAM == "telegram"
        assert Platform.SLACK == "slack"

    def test_string_enum(self):
        assert isinstance(Platform.DISCORD, str)


class TestDisplayConfig:
    def test_discord_config(self):
        assert DISCORD_CONFIG.max_message_length == 2000
        assert DISCORD_CONFIG.supports_embeds
        assert DISCORD_CONFIG.supports_buttons
        assert DISCORD_CONFIG.supports_threads

    def test_telegram_config(self):
        assert TELEGRAM_CONFIG.max_message_length == 4096
        assert TELEGRAM_CONFIG.supports_buttons

    def test_slack_config(self):
        assert SLACK_CONFIG.max_message_length == 40000
        assert SLACK_CONFIG.supports_threads

    def test_cli_no_limit(self):
        assert CLI_CONFIG.max_message_length == 0

    def test_frozen(self):
        with pytest.raises(AttributeError):
            DISCORD_CONFIG.max_message_length = 9999


class TestGetDisplayConfig:
    def test_by_string(self):
        cfg = get_display_config("discord")
        assert cfg.platform == Platform.DISCORD

    def test_by_enum(self):
        cfg = get_display_config(Platform.TELEGRAM)
        assert cfg.platform == Platform.TELEGRAM

    def test_case_insensitive(self):
        cfg = get_display_config("DISCORD")
        assert cfg.platform == Platform.DISCORD

    def test_unknown_falls_back_to_cli(self):
        cfg = get_display_config("unknown_platform")
        assert cfg.platform == Platform.CLI


class TestSplitMessage:
    def test_short_message_no_split(self):
        result = split_message("hello", 100)
        assert result == ["hello"]

    def test_no_limit_no_split(self):
        long_text = "x" * 10000
        result = split_message(long_text, 0)
        assert result == [long_text]

    def test_long_message_splits(self):
        text = "a" * 100
        result = split_message(text, 30)
        assert len(result) > 1
        assert all(len(chunk) <= 30 for chunk in result)
