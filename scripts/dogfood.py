"""Dogfood runner — run Caveman on real tasks to accumulate trajectories.

Usage:
    python scripts/dogfood.py                    # run all task sets
    python scripts/dogfood.py --set math         # run one set
    python scripts/dogfood.py --count 10         # run 10 random tasks
    python scripts/dogfood.py --stats            # show trajectory stats
"""
from __future__ import annotations
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Task sets — diverse tasks to exercise different capabilities
TASK_SETS = {
    "math": [
        "Calculate 17 * 34 step by step.",
        "What is the square root of 144?",
        "If I have 3 apples and buy 5 more, then give away 2, how many do I have?",
        "Convert 72 degrees Fahrenheit to Celsius.",
        "What is 15% of 240?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "I have a 3-gallon jug and a 5-gallon jug. How do I measure exactly 4 gallons?",
        "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25, so he gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9 (total $27) plus the bellboy has $2 = $29. Where is the missing dollar?",
    ],
    "coding": [
        "Write a Python function that checks if a string is a palindrome.",
        "Write a bash one-liner to find all .py files larger than 10KB in the current directory.",
        "Explain the difference between a stack and a queue with a simple example.",
        "What does this regex match: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    ],
    "tools": [
        "List all Python files in the caveman/ directory and count them.",
        "Read the first 5 lines of caveman/paths.py and explain what they do.",
        "Create a file /tmp/caveman_dogfood_test.txt with today's date, then read it back.",
        "Search for all files containing 'EngineFlags' in the caveman/ directory.",
    ],
    "knowledge": [
        "What are the SOLID principles in software engineering? List them briefly.",
        "Explain the CAP theorem in distributed systems in 3 sentences.",
        "What is the difference between concurrency and parallelism?",
        "Name 3 common design patterns and when to use each.",
    ],
}


async def run_task(loop, task: str, task_id: int) -> dict:
    """Run a single task and return results."""
    start = time.monotonic()
    try:
        result = await loop.run(task)
        elapsed = time.monotonic() - start
        quality = loop.trajectory_recorder.score_quality()

        # Save trajectory
        traj_path = await loop.trajectory_recorder.save()

        return {
            "task_id": task_id,
            "task": task,
            "success": True,
            "result_length": len(result),
            "result_preview": result[:150],
            "quality": quality,
            "elapsed": round(elapsed, 2),
            "tool_calls": loop._tool_call_count,
            "memories": loop.memory_manager.total_count,
            "trajectory": str(traj_path),
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "task_id": task_id,
            "task": task,
            "success": False,
            "error": str(e)[:200],
            "elapsed": round(elapsed, 2),
        }


async def run_dogfood(task_set: str | None = None, count: int | None = None):
    """Run dogfood tasks."""
    from caveman.agent.factory import create_loop

    # Select tasks
    if task_set and task_set in TASK_SETS:
        tasks = [(task_set, t) for t in TASK_SETS[task_set]]
    elif count:
        all_tasks = [(s, t) for s, ts in TASK_SETS.items() for t in ts]
        tasks = random.sample(all_tasks, min(count, len(all_tasks)))
    else:
        tasks = [(s, t) for s, ts in TASK_SETS.items() for t in ts]

    print(f"🦴 Caveman Dogfood — {len(tasks)} tasks")
    print("=" * 60)

    results = []
    for i, (set_name, task) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] ({set_name}) {task[:60]}...")

        # Create fresh loop for each task
        loop = create_loop()
        r = await run_task(loop, task, i + 1)
        results.append(r)

        if r["success"]:
            print(f"  ✅ quality={r['quality']:.2f} tools={r['tool_calls']} time={r['elapsed']}s")
            print(f"  → {r['result_preview'][:80]}...")
        else:
            print(f"  ❌ {r['error'][:80]}")

    # Summary
    print("\n" + "=" * 60)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    avg_quality = sum(r["quality"] for r in successes) / len(successes) if successes else 0
    avg_time = sum(r["elapsed"] for r in successes) / len(successes) if successes else 0

    print(f"📊 Results: {len(successes)}/{len(results)} success")
    print(f"   Avg quality: {avg_quality:.2f}")
    print(f"   Avg time: {avg_time:.1f}s")
    print(f"   Total tool calls: {sum(r.get('tool_calls', 0) for r in successes)}")
    if failures:
        print(f"   ❌ Failures: {len(failures)}")
        for f in failures:
            print(f"      - {f['task'][:50]}: {f['error'][:60]}")

    # Save summary
    summary_path = Path("~/.caveman/dogfood_results.json").expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n💾 Results saved: {summary_path}")


def show_stats():
    """Show current trajectory stats."""
    from caveman.training.stats import show_training_stats
    print(show_training_stats(None, 0.5))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Caveman Dogfood Runner")
    parser.add_argument("--set", choices=list(TASK_SETS.keys()), help="Run specific task set")
    parser.add_argument("--count", type=int, help="Run N random tasks")
    parser.add_argument("--stats", action="store_true", help="Show trajectory stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        asyncio.run(run_dogfood(task_set=args.set, count=args.count))
