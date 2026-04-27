"""CLI commands: service, obsidian, status, flywheel, audit, bench, etc."""
from __future__ import annotations

from typing import Optional

import typer


def register_utility_commands(app: typer.Typer) -> None:
    """Register utility CLI commands on the app."""

    @app.command()
    def obsidian(
        output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    ) -> None:
        """Export memories as Obsidian-compatible markdown vault."""
        from caveman.memory.manager import MemoryManager
        from caveman.memory.obsidian import export_to_obsidian
        from caveman.paths import CAVEMAN_HOME

        out_dir = output or str(CAVEMAN_HOME / "obsidian_vault")
        mm = MemoryManager.with_sqlite()
        result = export_to_obsidian(mm, out_dir)
        typer.echo(f"\u2705 Exported {result['exported']} memories to {result['output_dir']}")

    @app.command()
    def status() -> None:
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
        max_iter: int = typer.Option(50, "--max-iter", help="Max LLM iterations per round (raise freely for long-compounding audits)"),
        round_timeout: float = typer.Option(900.0, "--round-timeout", help="Wall-clock timeout in seconds per flywheel round"),
        stats: bool = typer.Option(False, "--stats", help="Show flywheel statistics"),
    ) -> None:
        """Run the meta-flywheel: Caveman audits and fixes itself."""
        from caveman.cli.flywheel import flywheel_cli
        flywheel_cli(
            target=target, all_=all_, parallel=parallel,
            rounds=rounds, max_iter=max_iter, round_timeout_s=round_timeout, stats=stats,
        )

    @app.command()
    def migrate(
        db: Optional[str] = typer.Option(None, "--db", help="Memory DB path (default: CAVEMAN_HOME/memory/caveman.db)"),
        dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview migrations by default; use --apply to mutate"),
    ) -> None:
        """Preview or apply Caveman database migrations."""
        from caveman.cli.migrate import run_migrate
        typer.echo(run_migrate(db_path=db, dry_run=dry_run))

    @app.command()
    def audit() -> None:
        """Run static code quality checks (no LLM needed)."""
        from caveman.cli.audit import run_audit
        typer.echo(run_audit())

    @app.command()
    def bench(rounds: int = typer.Option(1, help="Number of benchmark rounds")) -> None:
        """Run memory system performance benchmarks."""
        from caveman.cli.bench import run_bench_sync
        run_bench_sync(rounds=rounds)

    @app.command(name="self-test")
    def self_test() -> None:
        """Run full lifecycle self-test (store→recall→shield→wiki→skills)."""
        import asyncio
        from caveman.cli.selftest import run_self_test
        typer.echo(asyncio.run(run_self_test()))

    @app.command()
    def changelog(n: int = typer.Option(20, help="Number of recent commits")) -> None:
        """Auto-generate changelog from git log."""
        from caveman.cli.changelog import generate_changelog
        typer.echo(generate_changelog(n=n))

    @app.command()
    def install() -> None:
        """Install Caveman gateway as a system service (launchd/systemd)."""
        from caveman.gateway.service import install_service
        result = install_service()
        if result["ok"]:
            typer.echo(f"✅ Service installed! {result.get('plist') or result.get('unit', '')}")
        else:
            typer.echo(f"❌ Install failed: {result.get('detail', '?')}")
            raise typer.Exit(1)

    @app.command()
    def uninstall() -> None:
        """Uninstall Caveman gateway system service."""
        from caveman.gateway.service import uninstall_service
        result = uninstall_service()
        if not result["ok"]:
            typer.echo(f"❌ Uninstall failed: {result.get('detail', '?')}")
            raise typer.Exit(1)
        typer.echo("✅ Service uninstalled.")

    @app.command(name="daemon-status")
    def daemon_status() -> None:
        """Show daemon process status."""
        from caveman.gateway.daemon import get_status
        status = get_status()
        if status.running:
            typer.echo(f"✅ Daemon running (PID {status.pid}, uptime {status.uptime_seconds:.0f}s, {status.memory_mb:.1f}MB)")
        else:
            typer.echo("❌ Daemon not running")

    @app.command(name="daemon-start")
    def daemon_start() -> None:
        """Start the daemon process."""
        from caveman.gateway.daemon import start
        result = start()
        if result["success"]:
            typer.echo(f"✅ Daemon started (PID {result['pid']})")
        else:
            typer.echo(f"❌ Start failed: {result.get('error', '?')}")
            raise typer.Exit(1)

    @app.command(name="daemon-stop")
    def daemon_stop() -> None:
        """Stop the daemon process."""
        from caveman.gateway.daemon import stop
        result = stop()
        if result["success"]:
            typer.echo("✅ Daemon stopped")
        else:
            typer.echo(f"❌ Stop failed: {result.get('error', '?')}")
            raise typer.Exit(1)
