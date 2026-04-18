"""Tests for Round 19 Phase 2 — LLM Judge."""
from __future__ import annotations

import pytest
from caveman.trajectory.judge import LLMJudge, JudgeResult


class TestJudgeResult:
    def test_defaults(self):
        r = JudgeResult()
        assert r.overall == 0.5
        assert r.safety == 1.0
        assert r.mode == "heuristic"

    def test_to_dict(self):
        r = JudgeResult(correctness=0.8, efficiency=0.9, safety=1.0, helpfulness=0.7, overall=0.85)
        d = r.to_dict()
        assert d["correctness"] == 0.8
        assert "mode" in d


class TestHeuristicJudge:
    def setup_method(self):
        self.judge = LLMJudge()

    def test_successful_task(self):
        trajectory = [
            {"role": "user", "content": "Create a REST API"},
            {"role": "assistant", "content": "Done! Created FastAPI project. All 15 tests pass. ✅ Deployed successfully."},
        ]
        result = self.judge.evaluate_heuristic(trajectory, task="Create API")
        assert result.correctness > 0.6
        assert result.overall > 0.5
        assert "success" in result.reasoning

    def test_failed_task(self):
        trajectory = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Error: cannot connect. Failed to resolve. Exception traceback. ❌ Unable to fix."},
        ]
        result = self.judge.evaluate_heuristic(trajectory, task="Fix bug")
        assert result.correctness < 0.4
        assert "failure" in result.reasoning

    def test_inefficient_task(self):
        trajectory = [
            {"role": "user", "content": "Deploy the app"},
            {"role": "assistant", "content": "Let me try again. Sorry, that didn't work. Actually, let me reconsider. Trying another approach. Hmm, let me try once more."},
        ]
        result = self.judge.evaluate_heuristic(trajectory, task="Deploy")
        assert result.efficiency < 0.7
        assert "retries" in result.reasoning

    def test_safety_concerns(self):
        trajectory = [
            {"role": "user", "content": "Clean up the server"},
            {"role": "assistant", "content": "Running rm -rf /tmp/old and chmod 777 on the deploy dir. Also dropping unused tables: drop table logs."},
        ]
        result = self.judge.evaluate_heuristic(trajectory, task="Cleanup")
        assert result.safety < 0.5
        assert "safety" in result.reasoning

    def test_helpful_structured_response(self):
        trajectory = [
            {"role": "user", "content": "How do I set up Docker?"},
            {"role": "assistant", "content": "Here's how to set up Docker:\n1. Install Docker Desktop\n2. Create a Dockerfile:\n```dockerfile\nFROM python:3.12\nCOPY . /app\n```\n3. Build: `docker build -t myapp .`\n4. Run: `docker run -p 8000:8000 myapp`"},
        ]
        result = self.judge.evaluate_heuristic(trajectory, task="Docker setup")
        assert result.helpfulness > 0.5

    def test_empty_trajectory(self):
        result = self.judge.evaluate_heuristic([], task="Nothing")
        assert result.overall >= 0.0
        assert result.mode == "heuristic"

    def test_custom_weights(self):
        judge = LLMJudge(weights={"correctness": 1.0, "efficiency": 0.0, "safety": 0.0, "helpfulness": 0.0})
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Done! ✅ All tests pass."},
        ]
        result = judge.evaluate_heuristic(trajectory)
        # Overall should be dominated by correctness
        assert result.overall == pytest.approx(result.correctness, abs=0.01)


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_fallback_to_heuristic(self):
        judge = LLMJudge()  # No LLM
        trajectory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = await judge.evaluate(trajectory)
        assert result.mode == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        judge = LLMJudge(llm_fn=failing_llm)
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = await judge.evaluate(trajectory)
        assert result.mode == "heuristic"  # Fell back

    @pytest.mark.asyncio
    async def test_llm_evaluation(self):
        async def mock_llm(prompt: str) -> str:
            return '{"correctness": 0.9, "efficiency": 0.8, "safety": 1.0, "helpfulness": 0.85, "reasoning": "Good work"}'

        judge = LLMJudge(llm_fn=mock_llm)
        trajectory = [
            {"role": "user", "content": "Build API"},
            {"role": "assistant", "content": "Created FastAPI project"},
        ]
        result = await judge.evaluate(trajectory, task="Build API")
        assert result.mode == "llm"
        assert result.correctness == 0.9
        assert result.reasoning == "Good work"
