"""Coordinator — orchestrates 4 phases: Explore → Plan → Execute → Verify."""
from __future__ import annotations
from enum import Enum
from .loop import AgentLoop


class Phase(Enum):
    """Execution phases in a multi-step agent plan."""
    EXPLORE = "explore"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"


EXPLORE_SYSTEM = """You are in the EXPLORE phase. Understand the task and gather context.
1. Read relevant files and understand the environment
2. Search for relevant information
3. Identify what you know and what you need
4. End with a summary of requirements and available resources
DO NOT write code or make changes yet."""

PLAN_SYSTEM = """You are in the PLAN phase. Create a detailed plan.
1. List exact files to modify or create
2. Step-by-step implementation plan
3. Identify risks or blockers
4. End with: "PLAN READY"
DO NOT execute yet."""

EXECUTE_SYSTEM = """You are in the EXECUTE phase. Implement the plan.
- Follow the plan from PLAN phase
- Use tools to make actual changes
- Check your work after each step
- End with: "EXECUTION COMPLETE" """

VERIFY_SYSTEM = """You are in the VERIFY phase.
1. Run tests if applicable
2. Check all planned changes were made
3. Verify no regressions
4. Anti-rationalization: did you solve the problem or work around it?
5. End with: PASS or FAIL with findings"""


class PhasedCoordinator:
    """Orchestrates agent phases: Explore → Plan → Execute → Verify."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_iterations_per_phase: int = 15,
        skip_verify: bool = False,
    ):
        self.model = model
        self.max_iterations_per_phase = max_iterations_per_phase
        self.skip_verify = skip_verify
        self.phase_results: dict[Phase, str] = {}
        self.current_phase = Phase.EXPLORE

    async def run(self, task: str) -> str:
        """Execute task through all phases."""
        context = f"## Original Task\n{task}"
        systems = {
            Phase.EXPLORE: EXPLORE_SYSTEM,
            Phase.PLAN: PLAN_SYSTEM,
            Phase.EXECUTE: EXECUTE_SYSTEM,
            Phase.VERIFY: VERIFY_SYSTEM,
        }

        phases = [Phase.EXPLORE, Phase.PLAN, Phase.EXECUTE]
        if not self.skip_verify:
            phases.append(Phase.VERIFY)

        for phase in phases:
            self.current_phase = phase
            phase_task = f"{context}\n\n## Your Phase\n{phase.value.upper()}"
            loop = AgentLoop(model=self.model, max_iterations=self.max_iterations_per_phase)
            result = await loop.run(phase_task, system_prompt=systems[phase])
            self.phase_results[phase] = result
            context += f"\n\n## {phase.value.upper()} Result\n{result}"

        return self.phase_results.get(phases[-1], "")
