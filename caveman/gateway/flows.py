"""Flows — multi-step workflow engine with conditions and parallel execution.

Provides a DAG-based workflow engine for orchestrating complex multi-step
agent tasks with branching, loops, and parallel execution.

Inspired by OpenClaw src/flows/ but redesigned as a general-purpose engine.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "StepStatus",
    "StepResult",
    "FlowStep",
    "Flow",
    "FlowEngine",
    "FlowStatus",
    "ProviderSetupFlow",
    "create_flow",
    "list_flows",
]


logger = logging.getLogger("caveman.gateway.flows")


class StepStatus(Enum):
    """Execution status of a single flow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a workflow step."""
    status: StepStatus = StepStatus.COMPLETED
    output: Any = None
    error: str = ""
    duration_ms: float = 0


@dataclass
class FlowStep:
    """A single step in a workflow."""
    id: str
    name: str
    handler: str = ""  # Tool name or callable reference
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: str = ""  # Python expression evaluated against context
    retry_count: int = 0
    timeout: float = 60.0
    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None


@dataclass
class Flow:
    """A workflow definition."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    steps: List[FlowStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0

    def add_step(
        self,
        name: str,
        handler: str = "",
        *,
        depends_on: Optional[List[str]] = None,
        condition: str = "",
        **args,
    ) -> FlowStep:
        """Add a step to the flow."""
        step = FlowStep(
            id=f"step_{len(self.steps)}",
            name=name,
            handler=handler,
            args=args,
            depends_on=depends_on or [],
            condition=condition,
        )
        self.steps.append(step)
        return step


class FlowEngine:
    """Executes workflows with dependency resolution and parallel steps."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._active_flows: Dict[str, Flow] = {}

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a step handler."""
        self._handlers[name] = handler

    async def execute(self, flow: Flow) -> Flow:
        """Execute a workflow, respecting dependencies and conditions."""
        flow.status = StepStatus.RUNNING
        self._active_flows[flow.id] = flow

        try:
            while True:
                # Find ready steps (all deps completed, not yet run)
                ready = self._get_ready_steps(flow)
                if not ready:
                    # Check if all done or stuck
                    if all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
                           for s in flow.steps):
                        break
                    # Deadlock or all remaining have unmet deps
                    failed = [s for s in flow.steps if s.status == StepStatus.FAILED]
                    if failed:
                        break
                    # Shouldn't happen — safety break
                    logger.error("Flow %s stuck — no ready steps", flow.id)
                    break

                # Execute ready steps in parallel
                tasks = [self._execute_step(step, flow) for step in ready]
                await asyncio.gather(*tasks)

            # Determine final status
            failed_steps = [s for s in flow.steps if s.status == StepStatus.FAILED]
            flow.status = StepStatus.FAILED if failed_steps else StepStatus.COMPLETED
            flow.completed_at = time.time()

        except Exception as e:
            flow.status = StepStatus.FAILED
            logger.error("Flow %s execution error: %s", flow.id, e)

        finally:
            self._active_flows.pop(flow.id, None)

        return flow

    def _get_ready_steps(self, flow: Flow) -> List[FlowStep]:
        """Find steps whose dependencies are all completed."""
        ready = []
        completed_ids = {s.id for s in flow.steps if s.status == StepStatus.COMPLETED}

        for step in flow.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.depends_on):
                # Check condition
                if step.condition:
                    if not self._evaluate_condition(step.condition, flow.context):
                        step.status = StepStatus.SKIPPED
                        continue
                ready.append(step)

        return ready

    async def _execute_step(self, step: FlowStep, flow: Flow) -> None:
        """Execute a single step."""
        step.status = StepStatus.RUNNING
        start = time.monotonic()

        handler = self._handlers.get(step.handler)
        if not handler:
            step.result = StepResult(
                status=StepStatus.FAILED,
                error=f"Handler not found: {step.handler}",
            )
            step.status = StepStatus.FAILED
            return

        # Retry loop
        attempts = step.retry_count + 1
        for attempt in range(attempts):
            try:
                if asyncio.iscoroutinefunction(handler):
                    output = await asyncio.wait_for(
                        handler(step.args, flow.context),
                        timeout=step.timeout,
                    )
                else:
                    output = handler(step.args, flow.context)

                duration = (time.monotonic() - start) * 1000
                step.result = StepResult(
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration_ms=duration,
                )
                step.status = StepStatus.COMPLETED

                # Store output in flow context
                flow.context[f"{step.id}_output"] = output
                return

            except asyncio.TimeoutError:
                if attempt == attempts - 1:
                    step.result = StepResult(
                        status=StepStatus.FAILED,
                        error=f"Timeout after {step.timeout}s",
                        duration_ms=(time.monotonic() - start) * 1000,
                    )
                    step.status = StepStatus.FAILED
            except Exception as e:
                if attempt == attempts - 1:
                    step.result = StepResult(
                        status=StepStatus.FAILED,
                        error=str(e),
                        duration_ms=(time.monotonic() - start) * 1000,
                    )
                    step.status = StepStatus.FAILED
                else:
                    await asyncio.sleep(1.0 * (attempt + 1))

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition expression using AST.

        Only allows: comparisons, boolean ops, attribute access, literals,
        and name lookups into the provided context dict.
        """
        import ast as _ast
        import operator

        _SAFE_OPS = {
            _ast.Eq: operator.eq, _ast.NotEq: operator.ne,
            _ast.Lt: operator.lt, _ast.LtE: operator.le,
            _ast.Gt: operator.gt, _ast.GtE: operator.ge,
            _ast.Is: operator.is_, _ast.IsNot: operator.is_not,
            _ast.In: lambda a, b: a in b,
            _ast.NotIn: lambda a, b: a not in b,
        }

        def _eval_node(node):
            if isinstance(node, _ast.Expression):
                return _eval_node(node.body)
            if isinstance(node, _ast.Constant):
                return node.value
            if isinstance(node, _ast.Name):
                if node.id not in context:
                    raise NameError(node.id)
                return context[node.id]
            if isinstance(node, _ast.Attribute):
                obj = _eval_node(node.value)
                return getattr(obj, node.attr)
            if isinstance(node, _ast.Compare):
                left = _eval_node(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    op_fn = _SAFE_OPS.get(type(op))
                    if op_fn is None:
                        raise ValueError(f"unsupported op: {type(op).__name__}")
                    right = _eval_node(comparator)
                    if not op_fn(left, right):
                        return False
                    left = right
                return True
            if isinstance(node, _ast.BoolOp):
                if isinstance(node.op, _ast.And):
                    return all(_eval_node(v) for v in node.values)
                if isinstance(node.op, _ast.Or):
                    return any(_eval_node(v) for v in node.values)
            if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
                return not _eval_node(node.operand)
            raise ValueError(f"unsupported node: {type(node).__name__}")

        try:
            tree = _ast.parse(condition, mode="eval")
            return bool(_eval_node(tree))
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return False

    def get_flow_status(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active flow."""
        flow = self._active_flows.get(flow_id)
        if not flow:
            return None
        return {
            "id": flow.id,
            "name": flow.name,
            "status": flow.status.value,
            "steps": [
                {"id": s.id, "name": s.name, "status": s.status.value}
                for s in flow.steps
            ],
        }


# ── Setup Wizard Flows (backward compat) ──

class FlowStatus:
    """Overall execution status of a multi-step flow."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class _FlowState:
    status: str = FlowStatus.PENDING
    step: int = 0
    data: Dict[str, Any] = field(default_factory=dict)


class ProviderSetupFlow:
    """Interactive provider setup wizard."""

    _STEPS = ["provider", "api_key", "model", "proxy"]

    def __init__(self):
        self.state = _FlowState()

    def start(self) -> str:
        self.state.status = FlowStatus.ACTIVE
        self.state.step = 0
        return "Which provider would you like to configure? (anthropic/openai/google/deepseek)"

    def submit(self, value: str) -> tuple:
        """Submit a value for the current step. Returns (message, done)."""
        step_name = self._STEPS[self.state.step]

        if step_name == "provider":
            self.state.data["provider"] = value
            self.state.step += 1
            return "Enter your API key:", False

        elif step_name == "api_key":
            if not value:
                return "API key cannot be empty. Please enter your key:", False
            self.state.data["api_key"] = value
            self.state.step += 1
            return "Enter model name (or press enter for default):", False

        elif step_name == "model":
            self.state.data["model"] = value or "default"
            self.state.step += 1
            return "Enter proxy URL (or press enter to skip):", False

        elif step_name == "proxy":
            self.state.data["proxy"] = value
            self.state.status = FlowStatus.COMPLETED
            provider = self.state.data.get("provider", "unknown")
            return f"Provider {provider} configured successfully!", True

        return "Unknown step", False

    def cancel(self) -> str:
        self.state.status = FlowStatus.CANCELLED
        return "Flow cancelled."


_FLOW_REGISTRY = {
    "provider": ProviderSetupFlow,
    "channel": ProviderSetupFlow,  # Placeholder
}


def create_flow(flow_type: str) -> Flow | None:
    """Create a flow by type."""
    cls = _FLOW_REGISTRY.get(flow_type)
    if cls:
        return cls()
    return None


def list_flows() -> List[str]:
    """List available flow types."""
    return list(_FLOW_REGISTRY.keys())
