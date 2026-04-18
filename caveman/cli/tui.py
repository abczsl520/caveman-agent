"""Caveman TUI — rich terminal UI components + interactive REPL.

Two modes:
  1. Components: show_banner(), show_tool_call(), etc. (used by AgentLoop)
  2. Interactive REPL: `caveman` with no args → multi-turn conversation
"""
from __future__ import annotations
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── UI Components (used by AgentLoop) ──

CAVE_BANNER = r"""
   [dim]__/\___/\___/\__[/dim]
  [dim]/                \\[/dim]
  [dim]|[/dim] [bold cyan]C A V E M A N[/bold cyan]  [dim]|[/dim]
  [dim]|[/dim] [dim]self-evolving[/dim]  [dim]|[/dim]
  [dim]|[/dim] [dim]ai agent[/dim] 🔥    [dim]|[/dim]
  [dim]\\________________/[/dim]
"""


def show_banner() -> None:
    console.print(CAVE_BANNER)


def show_status(model: str, tools: int = 0, memories: int = 0) -> None:
    console.print(f"[dim]Model:[/dim] {model} | [dim]Tools:[/dim] {tools} | [dim]Memories:[/dim] {memories}")


def show_tool_call(tool_name: str, args: dict) -> None:
    console.print(f"  [yellow]⚡[/yellow] [bold]{tool_name}[/bold] {_truncate_args(args)}")


def show_tool_result(tool_name: str, result: str, success: bool = True) -> None:
    icon = "[green]✅[/green]" if success else "[red]❌[/red]"
    console.print(f"  {icon} {tool_name}: {result[:200]}")


def show_memory_nudge() -> None:
    console.print("  [blue]📝 Memory nudge running...[/blue]")


def show_skill_nudge() -> None:
    console.print("  [green]⚡ Skill check triggered...[/green]")


def show_error(message: str) -> None:
    console.print(f"[red]❌ Error:[/red] {message}")


def show_thinking() -> None:
    console.print("[dim]🤔 Thinking...[/dim]")


def create_skills_table(skills: list[dict]) -> Table:
    table = Table(title="Installed Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Success", justify="right")
    table.add_column("Version", justify="right")
    for skill in skills:
        rate = skill.get("success_rate", 0)
        table.add_row(
            skill.get("name", ""),
            skill.get("description", "")[:60],
            f"{rate:.0%}" if rate else "—",
            str(skill.get("version", 1)),
        )
    return table


# ── Interactive REPL ──

async def interactive_loop(
    model: str = "claude-opus-4-6",
    max_iterations: int = 50,
) -> None:
    """Run interactive multi-turn REPL.

    Features:
      - Multi-turn conversation with context carry-over
      - Session persistence (auto-save/restore)
      - Multi-line input (paste or \\ continuation)
      - Streaming output
      - /commands for quick actions
      - History file (~/.caveman/repl_history)
      - Engine status display
    """
    from caveman.agent.factory import create_loop
    from caveman.paths import CAVEMAN_HOME

    show_banner()
    agent = create_loop(model=model, max_iterations=max_iterations)
    tool_count = len(agent.tool_registry.get_schemas())
    show_status(model, tools=tool_count)

    # Engine status
    engines = []
    for name in ["shield", "nudge", "recall", "ripple", "lint"]:
        status = "✅" if agent.engine_flags.is_enabled(name) else "❌"
        engines.append(f"{name}:{status}")
    console.print(f"[dim]Engines:[/dim] {' '.join(engines)}")
    console.print("[dim]Type your task. /help for commands. Ctrl+C to exit.[/dim]\n")

    # Enable readline with history + command completion
    history_file = CAVEMAN_HOME / "repl_history"
    try:
        import readline
        from caveman.commands.completer import CommandCompleter
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if history_file.exists():
            readline.read_history_file(str(history_file))
        readline.set_history_length(1000)
        completer = CommandCompleter()
        readline.set_completer(completer.readline_completer)
        readline.parse_and_bind("tab: complete")
    except (ImportError, OSError):
        pass

    turn = 0
    while True:
        try:
            user_input = _get_input(turn)
            if not user_input:
                continue

            # Handle /commands via dispatcher
            if user_input.startswith("/"):
                from caveman.commands.dispatcher import dispatch
                result = await dispatch(
                    user_input, agent, surface="cli",
                    respond_fn=lambda msg: _dispatch_respond(msg, console),
                )
                if result == "exit":
                    break
                if result:
                    continue

            turn += 1
            console.print()

            # Run agent
            result = await agent.run(user_input)

            console.print()
            mem_count = agent.memory_manager.total_count
            console.print(f"[dim]───── Turn {turn} | Memories: {mem_count} ─────[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C — type /exit to quit, or continue chatting[/yellow]")
        except EOFError:
            break
        except Exception as e:
            show_error(f"{type(e).__name__}: {e}")

    # Save history
    try:
        import readline
        readline.write_history_file(str(history_file))
    except (ImportError, OSError, NameError):
        pass

    # Drain background tasks before exit
    try:
        await agent.drain_background(timeout=5.0)
    except Exception:
        pass  # intentional: non-critical

    console.print("\n[cyan]👋 Bye![/cyan]")


def _get_input(turn: int) -> str:
    """Get user input with prompt. Supports multi-line with \\ continuation."""
    try:
        prompt = f"[bold green]caveman[/bold green] [{turn}]> " if turn > 0 else "[bold green]caveman[/bold green]> "
        console.print(prompt, end="")
        line = input().strip()
        # Multi-line: if line ends with \, keep reading
        lines = [line.rstrip("\\")]
        while line.endswith("\\"):
            console.print("[dim]...[/dim] ", end="")
            line = input()
            lines.append(line.rstrip("\\"))
        return "\n".join(lines).strip()
    except (KeyboardInterrupt, EOFError):
        raise


def _dispatch_respond(msg: str, console: Console) -> None:
    """Respond function for CLI dispatch — renders panels or plain text."""
    if msg.startswith("__panel__:"):
        # format_panel returns "__panel__:Title\ncontent" for CLI
        header, _, body = msg.partition("\n")
        title = header[len("__panel__:"):]
        console.print(Panel(body, title=title, border_style="cyan"))
    else:
        console.print(msg)


def _truncate_args(args: dict, max_len: int = 80) -> str:
    """Truncate tool args for display."""
    s = str(args)
    return s[:max_len] + "..." if len(s) > max_len else s
