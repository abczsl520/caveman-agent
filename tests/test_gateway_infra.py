"""Tests for gateway infrastructure module."""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from caveman.gateway.infra import GatewayInfra


class TestGatewayInfra:
    def test_initial_state(self):
        infra = GatewayInfra()
        assert infra.hooks is None
        assert infra.task_registry is None

    @patch("caveman.gateway.infra.GatewayInfra.load_hooks")
    def test_load_hooks_called(self, mock_load):
        infra = GatewayInfra()
        infra.load_hooks()
        mock_load.assert_called_once()

    def test_load_hooks_graceful_on_missing_module(self):
        """load_hooks should not crash if HookRegistry is unavailable."""
        infra = GatewayInfra()
        with patch.dict("sys.modules", {"caveman.gateway.hooks": None}):
            infra.load_hooks()  # Should not raise
        assert infra.hooks is None

    def test_load_task_registry_graceful_on_missing(self):
        """load_task_registry should not crash if TaskRegistry is unavailable."""
        infra = GatewayInfra()
        with patch.dict("sys.modules", {"caveman.gateway.task_registry": None}):
            infra.load_task_registry()  # Should not raise
        assert infra.task_registry is None

    @pytest.mark.asyncio
    async def test_emit_hook_noop_when_no_hooks(self):
        """emit_hook should be a no-op when hooks not loaded."""
        infra = GatewayInfra()
        await infra.emit_hook("test_event", {"key": "value"})  # Should not raise
        assert True  # No exception = hooks gracefully handle missing registry

    @pytest.mark.asyncio
    async def test_emit_hook_calls_registry(self):
        """emit_hook should delegate to HookRegistry.emit."""
        infra = GatewayInfra()
        mock_registry = MagicMock()
        mock_registry.emit = AsyncMock()
        infra._hooks_registry = mock_registry

        await infra.emit_hook("task_start", {"task_id": "123"})
        mock_registry.emit.assert_called_once_with("task_start", {"task_id": "123"})

    @pytest.mark.asyncio
    async def test_emit_hook_catches_errors(self):
        """emit_hook should catch and log errors from hooks."""
        infra = GatewayInfra()
        mock_registry = MagicMock()
        mock_registry.emit = AsyncMock(side_effect=RuntimeError("hook failed"))
        infra._hooks_registry = mock_registry

        await infra.emit_hook("test_event")  # Should not raise
        assert mock_registry.emit.called  # Hook was attempted
