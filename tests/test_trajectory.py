"""Tests for trajectory recorder v2 and compressor."""
import pytest
import json
import tempfile
from pathlib import Path
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.trajectory.compressor import compress_trajectory, merge_trajectories


@pytest.mark.asyncio
async def test_trajectory_recording():
    with tempfile.TemporaryDirectory() as td:
        rec = TrajectoryRecorder(base_dir=td)
        rec.set_task("test task")
        await rec.record_turn("human", "do something")
        await rec.record_turn("gpt", "I'll use bash")
        await rec.record_turn("function_call", json.dumps({"name": "bash", "arguments": {"command": "echo hi"}}))
        await rec.record_turn("function_response", "hi", metadata={"tool": "bash"})
        await rec.record_turn("gpt", "Done! The command output 'hi'.")

        assert rec._tool_calls == 1
        assert rec._errors == 0
        assert len(rec._turns) == 5


@pytest.mark.asyncio
async def test_trajectory_quality_score():
    rec = TrajectoryRecorder()
    rec.set_task("test")
    await rec.record_turn("human", "do a complex task")
    await rec.record_turn("function_call", '{"name":"bash"}')
    await rec.record_turn("function_response", "output")
    await rec.record_turn("gpt", "Here is the result of the complex task with detailed explanation.")

    score = rec.score_quality()
    assert score > 0.5  # Should score well: has tool use, multi-turn, good completion


@pytest.mark.asyncio
async def test_trajectory_quality_low():
    rec = TrajectoryRecorder()
    await rec.record_turn("human", "hi")
    await rec.record_turn("gpt", "ok")

    score = rec.score_quality()
    assert score < 0.5  # Short, no tools, trivial


@pytest.mark.asyncio
async def test_trajectory_quality_errors():
    rec = TrajectoryRecorder()
    await rec.record_turn("human", "do it")
    await rec.record_turn("function_call", '{"name":"bash"}')
    await rec.record_turn("function_response", "error", metadata={"error": True})
    await rec.record_turn("gpt", "Failed.")

    score = rec.score_quality()
    assert score < 0.7  # Penalty for errors (but still has tool use + multi-turn)


@pytest.mark.asyncio
async def test_trajectory_save_load():
    with tempfile.TemporaryDirectory() as td:
        rec = TrajectoryRecorder(base_dir=td)
        rec.set_task("save test")
        await rec.record_turn("human", "hello")
        await rec.record_turn("gpt", "world")

        path = await rec.save()
        assert path.exists() or Path(td).exists()

        # Find the saved file
        files = list(Path(td).glob("*.json"))
        assert len(files) == 1

        loaded = TrajectoryRecorder.load(files[0])
        assert loaded["task"] == "save test"
        assert len(loaded["conversations"]) == 2
        assert loaded["metadata"]["quality_score"] is not None


@pytest.mark.asyncio
async def test_batch_export():
    with tempfile.TemporaryDirectory() as td:
        # Create some trajectories
        for i in range(3):
            rec = TrajectoryRecorder(base_dir=td)
            rec.set_task(f"task {i}")
            await rec.record_turn("human", f"task {i}")
            if i > 0:  # Make some higher quality
                await rec.record_turn("function_call", '{"name":"bash"}')
                await rec.record_turn("function_response", "output")
                await rec.record_turn("gpt", f"Completed task {i} with detailed results here.")
            else:
                await rec.record_turn("gpt", "ok")
            await rec.save()

        out = TrajectoryRecorder.batch_export(td, min_quality=0.0)
        assert out.exists()

        # Count exported lines
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # All exported


def test_compress_trajectory():
    turns = [
        {"from": "human", "value": "do it"},
        {"from": "gpt", "value": "ok"},
        {"from": "gpt", "value": "ok"},  # duplicate
        {"from": "function_response", "value": "x" * 1000},  # long
        {"from": "gpt", "value": "done"},
    ]
    result = compress_trajectory(turns)
    assert len(result) == 4  # duplicate removed
    # Long tool response should be truncated
    long_turn = [t for t in result if t["from"] == "function_response"][0]
    assert len(long_turn["value"]) < 600


def test_compress_trajectory_max_turns():
    turns = [{"from": "human", "value": f"msg {i}"} for i in range(100)]
    result = compress_trajectory(turns, max_turns=20)
    assert len(result) == 20


def test_merge_trajectories():
    t1 = [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]
    t2 = [{"from": "human", "value": "c"}, {"from": "gpt", "value": "d"}]
    merged = merge_trajectories([t1, t2])
    assert len(merged) == 4
    assert merged[0]["value"] == "a"
    assert merged[-1]["value"] == "d"
