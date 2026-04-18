#!/usr/bin/env python3
"""Real API end-to-end test — runs Caveman with actual Anthropic API.

Usage:
    python scripts/test_real_api.py "List files in the current directory"
"""
import asyncio
import sys
import os

# Use the OpenClaw proxy
os.environ.setdefault("ANTHROPIC_API_KEY", "${ANTHROPIC_API_KEY}")
os.environ.setdefault("ANTHROPIC_BASE_URL", "http://localhost:4200")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caveman.agent.loop import AgentLoop
from caveman.providers.anthropic_provider import AnthropicProvider
from caveman.cli.tui import show_banner


async def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2? Just answer the number."

    show_banner()
    print(f"\n🎯 Task: {task}")
    print(f"🔗 API: {os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')}")
    print("-" * 50)

    provider = AnthropicProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-opus-4-6",
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=4096,
    )

    loop = AgentLoop(
        model="claude-opus-4-6",
        provider=provider,
        max_iterations=10,
    )

    print(f"🔧 Tools loaded: {len(loop.tool_registry.get_schemas())}")
    print("-" * 50)

    try:
        result = await loop.run(task)
        print("\n" + "=" * 50)
        print("✅ RESULT:")
        print(result)
        print("=" * 50)
        print(f"📊 Turns: {loop._turn_count} | Tool calls: {loop._tool_call_count}")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
