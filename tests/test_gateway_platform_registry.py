"""Tests for gateway platform registry."""
from unittest.mock import MagicMock, patch

import pytest

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import PlatformConfig


class FakeAdapter(BasePlatformAdapter):
    """Minimal adapter for testing."""
    def __init__(self, config):
        self.config = config

    async def connect(self): return True
    async def disconnect(self): pass
    async def send(self, chat_id, content, reply_to=None, metadata=None): pass


class TestPlatformRegistry:
    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Reset the global registry before each test."""
        from caveman.gateway import platform_registry as pr
        original = pr._ADAPTERS.copy()
        yield
        pr._ADAPTERS.clear()
        pr._ADAPTERS.update(original)

    def test_register_and_get(self):
        from caveman.gateway.platform_registry import register_adapter, get_adapter
        config = MagicMock(spec=PlatformConfig)
        register_adapter("fake", FakeAdapter)
        adapter = get_adapter("fake", config)
        assert isinstance(adapter, FakeAdapter)
        assert adapter.config is config

    def test_get_unknown_returns_none(self):
        from caveman.gateway.platform_registry import get_adapter
        config = MagicMock(spec=PlatformConfig)
        result = get_adapter("nonexistent_platform_xyz", config)
        assert result is None

    def test_case_insensitive(self):
        from caveman.gateway.platform_registry import register_adapter, get_adapter
        config = MagicMock(spec=PlatformConfig)
        register_adapter("FaKe", FakeAdapter)
        assert get_adapter("fake", config) is not None
        assert get_adapter("FAKE", config) is not None

    def test_list_platforms(self):
        from caveman.gateway.platform_registry import list_platforms
        platforms = list_platforms()
        assert isinstance(platforms, list)
        # Built-in adapters should be registered
        assert "discord" in platforms

    def test_register_overwrites(self):
        from caveman.gateway.platform_registry import register_adapter, get_adapter, _ADAPTERS
        config = MagicMock(spec=PlatformConfig)

        class AnotherAdapter(FakeAdapter):
            pass

        register_adapter("fake", FakeAdapter)
        register_adapter("fake", AnotherAdapter)
        adapter = get_adapter("fake", config)
        assert isinstance(adapter, AnotherAdapter)
