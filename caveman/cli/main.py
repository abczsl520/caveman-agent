"""Caveman CLI entry point."""
import typer
from typing import Optional

app = typer.Typer(name="caveman", help="An agent that learns, executes, and evolves.")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Ensure home directories exist on every CLI invocation."""
    from caveman.paths import ensure_home
    ensure_home()
    if ctx.invoked_subcommand is None:
        # No subcommand → show help
        typer.echo(ctx.get_help())


# Register wiki + mcp commands
from caveman.cli.wiki_mcp import register_wiki_commands, register_mcp_commands
register_wiki_commands(app)
register_mcp_commands(app)

@app.command()
def run(
    task: Optional[str] = typer.Argument(None, help="Task to execute"),
    model: str = typer.Option("claude-opus-4-6", "--model", "-m", help="LLM model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    max_iter: int = typer.Option(50, "--max-iter", help="Max loop iterations"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive REPL mode"),
):
    """Run Caveman agent with a given task, or interactively."""
    import asyncio
    from caveman.cli.tui import show_banner, show_error

    if interactive or task is None:
        # Interactive REPL mode
        from caveman.cli.tui import interactive_loop
        try:
            asyncio.run(interactive_loop(model=model, max_iterations=max_iter))
        except KeyboardInterrupt:
            typer.echo("\n🛑 Bye!")
        return

    show_banner()
    from caveman.agent.factory import create_loop

    try:
        loop = create_loop(model=model, max_iterations=max_iter)
        typer.echo(f"[Model: {model} | Tools: {len(loop.tool_registry.get_schemas())} loaded]")
        result = asyncio.run(loop.run(task))
    except KeyboardInterrupt:
        typer.echo("\n🛑 Interrupted.")
        raise typer.Exit(0)
    except Exception as e:
        show_error(f"{type(e).__name__}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)

@app.command()
def skills():
    """List installed skills."""
    from caveman.skills.manager import SkillManager

    mgr = SkillManager()
    mgr.load_all()
    if not mgr._skills:
        typer.echo("No skills installed yet.")
        return
    for name, skill in mgr._skills.items():
        typer.echo(f"  {name}: {skill.description} (v{skill.version})")

@app.command()
def version():
    """Show Caveman version."""
    from caveman import __version__
    typer.echo(f"Caveman v{__version__}")

@app.command()
def tools():
    """List available tools."""
    from caveman.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg._register_builtins()
    for schema in reg.get_schemas():
        typer.echo(f"  🔧 {schema['name']}: {schema['description']}")

@app.command()
def setup():
    """Interactive setup — create ~/.caveman/config.yaml with auto-detection."""
    from caveman.paths import CAVEMAN_HOME, CONFIG_PATH, DEFAULT_MODEL
    CAVEMAN_HOME.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        overwrite = typer.confirm(f"{CONFIG_PATH} exists. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Setup cancelled.")
            return

    # FR-301: Auto-detect existing configs
    detected = _detect_external_configs()
    api_key = ""
    model = DEFAULT_MODEL

    if detected:
        typer.echo(f"\n\U0001f50d Detected existing configs:")
        for source, info in detected.items():
            typer.echo(f"  \u2022 {source}: {info['path']}")
            if info.get("api_key"):
                typer.echo(f"    API key: {info['api_key'][:8]}...{info['api_key'][-4:]}")
            if info.get("model"):
                typer.echo(f"    Model: {info['model']}")

        # Offer to import
        best = next(iter(detected.values()))
        if best.get("api_key"):
            import_key = typer.confirm(
                f"Import API key from {next(iter(detected))}?", default=True
            )
            if import_key:
                api_key = best["api_key"]
        if best.get("model"):
            model = best["model"]

    if not api_key:
        api_key = typer.prompt("Anthropic API key (or press Enter to skip)", default="", show_default=False)
    model = typer.prompt("Default model", default=model)

    from caveman.paths import MEMORY_DIR, SKILLS_DIR, TRAJECTORIES_DIR
    import yaml as _yaml
    config_data = {
        "agent": {"default_model": model, "max_iterations": 50},
        "providers": {"anthropic": {"api_key": api_key, "model": model}},
        "security": {"secret_scanning": True},
        "memory": {"backend": "local", "local_dir": str(MEMORY_DIR)},
        "skills": {"local_dir": str(SKILLS_DIR), "auto_create": True},
        "trajectory": {"enabled": True, "local_dir": str(TRAJECTORIES_DIR)},
    }
    CONFIG_PATH.write_text(
        "# Caveman config\n" + _yaml.safe_dump(config_data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    typer.echo(f"\u2705 Config saved to {CONFIG_PATH}")
    typer.echo("Run: caveman run 'your task here'")

def _detect_external_configs() -> dict[str, dict]:
    """Detect OpenClaw, Hermes, Claude Code configs."""
    from pathlib import Path
    import yaml

    detected = {}
    candidates = [
        ("OpenClaw", Path("~/.openclaw/config.yaml").expanduser()),
        ("Hermes", Path("~/.hermes/config.yaml").expanduser()),
        ("Claude Code", Path("~/.claude/settings.json").expanduser()),
    ]

    for name, path in candidates:
        if not path.exists():
            continue
        info: dict = {"path": str(path)}
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                import json
                data = json.loads(text)
                # Claude Code stores API key in env
            else:
                data = yaml.safe_load(text) or {}
                # OpenClaw/Hermes: look for API keys
                providers = data.get("providers", data.get("llm", {}))
                if isinstance(providers, dict):
                    for prov in providers.values():
                        if isinstance(prov, dict):
                            key = prov.get("api_key", prov.get("apiKey", ""))
                            if key and not key.startswith("$"):
                                info["api_key"] = key
                            m = prov.get("model", "")
                            if m:
                                info["model"] = m
            detected[name] = info
        except Exception:
            detected[name] = info

    return detected

@app.command()
def serve(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Start Caveman as a Discord/Telegram bot."""
    import asyncio
    from caveman.cli.tui import show_banner
    from caveman.gateway.runner import run_gateway

    show_banner()
    typer.echo("🌐 Starting gateway service...")
    try:
        asyncio.run(run_gateway(config_path=config))
    except KeyboardInterrupt:
        typer.echo("\n🛑 Gateway stopped.")

@app.command()
def export(
    min_quality: float = typer.Option(0.5, "--min-quality", help="Minimum quality score"),
    trajectory_dir: str = typer.Option(None, "--dir"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Export trajectories as JSONL for training."""
    from caveman.trajectory.recorder import TrajectoryRecorder
    from caveman.paths import TRAJECTORIES_DIR

    traj_dir = trajectory_dir or str(TRAJECTORIES_DIR)
    out = TrajectoryRecorder.batch_export(traj_dir, min_quality, output)
    typer.echo(f"✅ Exported to {out}")

@app.command(name="import")
def import_cmd(
    source: Optional[str] = typer.Argument(None, help="Source: openclaw, hermes, codex, claude-code, directory"),
    from_source: Optional[str] = typer.Option(None, "--from", help="Source to import from"),
    directory: Optional[str] = typer.Option(None, "--dir", "-d", help="Custom directory (for source=directory)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview without writing (default: dry-run)"),
    detect: bool = typer.Option(False, "--detect", help="Detect available sources"),
    all_sources: bool = typer.Option(False, "--all", help="Import from all detected sources"),
    only: Optional[str] = typer.Option(None, "--only", help="Filter: memory, config, workspace"),
    include_secrets: bool = typer.Option(False, "--include-secrets", help="Also import entries containing secrets/keys"),
):
    """Import memories from OpenClaw, Hermes, Codex, or Claude Code."""
    import asyncio
    from caveman.cli.importer import import_memories, detect_sources
    from caveman.import_.report import format_detect_report, format_result_report
    from caveman.memory.manager import MemoryManager

    if detect:
        detected = detect_sources()
        typer.echo(format_detect_report(detected))
        return

    effective_source = from_source or source

    if all_sources:
        detected = detect_sources()
        mm = MemoryManager.with_sqlite()
        for src, found in detected.items():
            if found:
                typer.echo(f"\n→ Importing from {src}...")
                result = asyncio.run(import_memories(src, mm, dry_run=dry_run, only=only, include_secrets=include_secrets))
                typer.echo(format_result_report(result))
        return

    if not effective_source:
        typer.echo("Specify a source: caveman import --from openclaw")
        typer.echo("Or use: caveman import --detect / --all")
        return

    mm = MemoryManager.with_sqlite()
    result = asyncio.run(import_memories(effective_source, mm, directory, dry_run, only=only, include_secrets=include_secrets))
    typer.echo(format_result_report(result))

@app.command()
def doctor():
    """Check flywheel health — memory, skills, trajectories, system."""
    import asyncio
    from caveman.cli.doctor import run_doctor

    report = asyncio.run(run_doctor())
    typer.echo(report.to_text())

@app.command()
def train(
    target: str = typer.Option("embedding", "--target", "-t", help="Training target: embedding, sft, dpo, ppo, grpo"),
    method: str = typer.Option("", "--method", help="Alias for --target (backward compat)"),
    model: str = typer.Option("", "--model", "-m", help="Base model name"),
    trajectory_dir: str = typer.Option(None, "--data"),
    output_dir: str = typer.Option(None, "--output", "-o"),
    min_quality: float = typer.Option(0.5, "--min-quality"),
    epochs: int = typer.Option(3, "--epochs"),
    format: str = typer.Option("sharegpt", "--format", help="Dataset format: sharegpt, chatml, openai"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build dataset only, skip training"),
    stats: bool = typer.Option(False, "--stats", help="Show trajectory stats without building"),
):
    """Train embedding model or export data for researchers."""
    from caveman.cli.tui import show_banner
    show_banner()

    effective_target = method if method else target

    if stats:
        from caveman.training.stats import show_training_stats
        typer.echo(show_training_stats(trajectory_dir, min_quality))
        return

    from caveman.training.cli_handler import run_train
    typer.echo(f"🎯 Target: {effective_target}")
    result = run_train(
        target=effective_target, model=model,
        trajectory_dir=trajectory_dir, output_dir=output_dir,
        min_quality=min_quality, epochs=epochs,
        format=format, dry_run=dry_run,
    )
    typer.echo(result)

@app.command()
def hub(
    query: str = typer.Argument("", help="Search query"),
    action: str = typer.Option("search", "--action", "-a", help="search, install, publish, stats"),
    name: str = typer.Option("", "--name", "-n", help="Skill/plugin name for install/publish"),
):
    """Caveman Hub — discover and share skills & plugins."""
    import asyncio
    from caveman.hub.client import HubClient

    client = HubClient()

    if action == "search":
        results = asyncio.run(client.search_skills(query))
        if results:
            for skill in results:
                typer.echo(f"  📦 {skill.get('name')}: {skill.get('description', '')}")
        else:
            typer.echo("No results (hub may be offline)")

    elif action == "install":
        if not name:
            typer.echo("❌ --name required for install")
            return
        ok = asyncio.run(client.install_skill(name))
        typer.echo(f"✅ Installed {name}" if ok else f"❌ Failed to install {name}")

    elif action == "stats":
        stats = asyncio.run(client.hub_stats())
        typer.echo(f"Hub: {stats}")

    else:
        typer.echo(f"Unknown action: {action}")

@app.command()
def plugins(
    action: str = typer.Option("list", "--action", "-a", help="list, load"),
):
    """Manage Caveman plugins."""
    from caveman.plugins.manager import PluginManager

    mgr = PluginManager()

    if action == "list":
        found = mgr.discover()
        if found:
            for p in found:
                status = "✅" if p.enabled else "❌"
                typer.echo(f"  {status} {p.name} v{p.version} ({p.plugin_type}): {p.description}")
        else:
            from caveman.paths import PLUGINS_DIR
            typer.echo(f"No plugins found in {PLUGINS_DIR}/")

    elif action == "load":
        count = mgr.load_all()
        typer.echo(f"Loaded {count} plugins")

@app.command()
def obsidian(
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Export memories as Obsidian-compatible markdown vault."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.obsidian import export_to_obsidian
    from caveman.paths import CAVEMAN_HOME

    out_dir = output or str(CAVEMAN_HOME / "obsidian_vault")
    mm = MemoryManager.with_sqlite()
    result = export_to_obsidian(mm, out_dir)
    typer.echo(f"\u2705 Exported {result['exported']} memories to {result['output_dir']}")

@app.command()
def status():
    """Show Caveman status dashboard + project stats."""
    from caveman.cli.status import status_text
    from caveman.cli.stats import get_stats
    typer.echo(status_text())
    typer.echo(get_stats())

@app.command()
def flywheel(
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target subsystem to audit"),
    all_: bool = typer.Option(False, "--all", help="Audit all discovered subsystems"),
    parallel: Optional[list[str]] = typer.Option(None, "--parallel", "-p", help="Audit multiple subsystems in parallel"),
    rounds: int = typer.Option(5, help="Number of flywheel rounds"),
    max_iter: int = typer.Option(15, "--max-iter", help="Max LLM iterations per round"),
    stats: bool = typer.Option(False, "--stats", help="Show flywheel statistics"),
):
    """Run the meta-flywheel: Caveman audits and fixes itself."""
    from caveman.cli.flywheel import flywheel_cli
    flywheel_cli(
        target=target, all_=all_, parallel=parallel,
        rounds=rounds, max_iter=max_iter, stats=stats,
    )

@app.command()
def audit():
    """Run static code quality checks (no LLM needed)."""
    from caveman.cli.audit import run_audit
    typer.echo(run_audit())

@app.command()
def bench(rounds: int = typer.Option(1, help="Number of benchmark rounds")):
    """Run memory system performance benchmarks."""
    from caveman.cli.bench import run_bench_sync
    run_bench_sync(rounds=rounds)

@app.command(name="self-test")
def self_test():
    """Run full lifecycle self-test (store→recall→shield→wiki→skills)."""
    import asyncio
    from caveman.cli.selftest import run_self_test
    typer.echo(asyncio.run(run_self_test()))

@app.command()
def changelog(n: int = typer.Option(20, help="Number of recent commits")):
    """Auto-generate changelog from git log."""
    from caveman.cli.changelog import generate_changelog
    typer.echo(generate_changelog(n=n))

# Register ACP commands from separate module
import caveman.cli.acp_cli  # noqa: F401, E402

if __name__ == "__main__":
    app()
