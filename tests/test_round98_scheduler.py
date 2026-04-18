"""Tests for LLMScheduler integration with EngineManager."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from caveman.engines.manager import EngineManager, EngineSet, _make_scheduled_llm_fn
from caveman.engines.flags import EngineFlags
from caveman.engines.scheduler import LLMScheduler, Priority


class TestSchedulerIntegration:
    """Test that EngineManager wires scheduler into engines."""

    def test_scheduler_created_when_llm_fn_provided(self):
        """EngineManager should create scheduler when llm_fn is available."""
        mm = MagicMock()
        mm.set_ripple = MagicMock()
        flags = EngineFlags({"engines": {
            "nudge": {"enabled": True},
            "reflect": {"enabled": True},
            "ripple": {"enabled": True},
            "lint": {"enabled": True},
        }})

        async def fake_llm(prompt):
            return "ok"

        mgr = EngineManager(flags, mm, llm_fn=fake_llm)
        engines = mgr.create_all()

        assert engines.scheduler is not None
        assert isinstance(engines.scheduler, LLMScheduler)

    def test_no_scheduler_without_llm_fn(self):
        """No scheduler when llm_fn is None."""
        mm = MagicMock()
        mm.set_ripple = MagicMock()
        flags = EngineFlags()

        mgr = EngineManager(flags, mm, llm_fn=None)
        engines = mgr.create_all()

        assert engines.scheduler is None

    def test_scheduler_disabled_explicitly(self):
        """Scheduler can be disabled even with llm_fn."""
        mm = MagicMock()
        mm.set_ripple = MagicMock()

        async def fake_llm(prompt):
            return "ok"

        mgr = EngineManager(
            EngineFlags(), mm, llm_fn=fake_llm,
            enable_scheduler=False,
        )
        engines = mgr.create_all()

        assert engines.scheduler is None

    def test_engine_set_has_scheduler_field(self):
        """EngineSet should have scheduler field."""
        es = EngineSet()
        assert es.scheduler is None
        es.scheduler = "test"
        assert es.scheduler == "test"
        # scheduler should not appear in active_names
        assert "scheduler" not in es.active_names()


class TestScheduledLLMFn:
    """Test the priority-wrapped llm_fn."""

    @pytest.mark.asyncio
    async def test_scheduled_fn_calls_scheduler(self):
        """Scheduled fn should route through scheduler.request()."""
        mock_scheduler = AsyncMock(spec=LLMScheduler)
        mock_scheduler.request = AsyncMock(return_value="scheduled response")

        fn = _make_scheduled_llm_fn(mock_scheduler, "shield", Priority.CRITICAL)
        result = await fn("test prompt")

        assert result == "scheduled response"
        mock_scheduler.request.assert_called_once_with(
            "shield", Priority.CRITICAL, "test prompt",
        )

    @pytest.mark.asyncio
    async def test_different_priorities_for_engines(self):
        """Each engine should get its own priority level."""
        mock_scheduler = AsyncMock(spec=LLMScheduler)
        mock_scheduler.request = AsyncMock(return_value="ok")

        shield_fn = _make_scheduled_llm_fn(mock_scheduler, "shield", Priority.CRITICAL)
        lint_fn = _make_scheduled_llm_fn(mock_scheduler, "lint", Priority.LOW)

        await shield_fn("shield prompt")
        await lint_fn("lint prompt")

        calls = mock_scheduler.request.call_args_list
        assert calls[0].args == ("shield", Priority.CRITICAL, "shield prompt")
        assert calls[1].args == ("lint", Priority.LOW, "lint prompt")
