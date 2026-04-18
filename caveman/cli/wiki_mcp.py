"""CLI handlers for wiki and mcp commands."""
from __future__ import annotations

__all__ = ["register_wiki_commands", "register_mcp_commands"]

import typer
from typing import Optional


def register_wiki_commands(app: typer.Typer) -> None:
    """Register wiki subcommands on the main app."""

    @app.command()
    def wiki(
        action: str = typer.Argument("status", help="Action: status|compile|search|export|index"),
        query: Optional[str] = typer.Argument(None, help="Search query (for search action)"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory (for export)"),
        max_tokens: int = typer.Option(4000, "--max-tokens", help="Max tokens for compiled context"),
    ) -> None:
        """Wiki compiler — compile, don't retrieve.

        Actions:
          status   Show wiki stats per tier
          compile  Run compilation (promote + expire + enforce limits)
          search   Search wiki entries
          export   Export as Obsidian-compatible markdown
          index    Generate index.md
          context  Show compiled context for system prompt
        """
        from caveman.wiki import WikiStore
        from caveman.wiki.compiler import WikiCompiler

        compiler = WikiCompiler(WikiStore())

        if action == "status":
            stats = compiler.store.stats()
            total = sum(stats.values())
            typer.echo(f"📚 Wiki: {total} entries")
            for tier, count in stats.items():
                bar = "█" * min(count, 40)
                typer.echo(f"  {tier:12s} {count:4d} {bar}")

        elif action == "compile":
            result = compiler.compile()
            typer.echo(f"✅ Compiled in {result.duration_ms:.1f}ms")
            typer.echo(f"   Processed: {result.entries_processed}")
            typer.echo(f"   Promoted:  {result.entries_promoted}")
            typer.echo(f"   Expired:   {result.entries_expired}")
            if result.errors:
                for err in result.errors:
                    typer.echo(f"   ⚠️  {err}")

        elif action == "search":
            if not query:
                typer.echo("Usage: caveman wiki search <query>", err=True)
                raise typer.Exit(1)
            results = compiler.store.search(query)
            if not results:
                typer.echo("No results found.")
                return
            for entry in results:
                typer.echo(f"  [{entry.tier}] [{entry.confidence:.2f}] {entry.title}")
                typer.echo(f"    {entry.content[:120]}")
                typer.echo()

        elif action == "export":
            from pathlib import Path
            out = Path(output) if output else None
            count = compiler.export_markdown(out)
            typer.echo(f"📄 Exported {count} entries as markdown")

        elif action == "index":
            index = compiler.generate_index()
            typer.echo(index)

        elif action == "context":
            ctx = compiler.get_compiled_context(max_tokens=max_tokens)
            if ctx:
                typer.echo(ctx)
            else:
                typer.echo("Wiki is empty. Ingest some knowledge first.")

        else:
            typer.echo(f"Unknown action: {action}. Use: status|compile|search|export|index|context", err=True)
            raise typer.Exit(1)


def register_mcp_commands(app: typer.Typer) -> None:
    """Register mcp subcommands on the main app."""

    @app.command()
    def mcp(
        action: str = typer.Argument("serve", help="Action: serve|tools|status"),
        port: int = typer.Option(0, "--port", "-p", help="HTTP port (0 = stdio transport)"),
    ) -> None:
        """MCP server — expose Caveman as MCP tools.

        Actions:
          serve   Start MCP server (stdio by default, --port for HTTP)
          tools   List available MCP tools
          status  Show MCP server status
        """
        if action == "serve":
            from caveman.mcp.server import run_stdio, run_http
            if port > 0:
                typer.echo(f"🔌 Starting MCP HTTP server on port {port}...")
                run_http(port=port)
            else:
                # stdio mode — no banner, just start
                run_stdio()

        elif action == "tools":
            from caveman.mcp.server import mcp as mcp_server
            tools = mcp_server._tool_manager.list_tools()
            typer.echo(f"🔧 {len(tools)} MCP tools available:")
            for tool in tools:
                desc = tool.description[:60] if tool.description else ""
                typer.echo(f"  {tool.name:20s} {desc}")

        elif action == "status":
            typer.echo("MCP server is not running (use 'caveman mcp serve' to start)")

        else:
            typer.echo(f"Unknown action: {action}. Use: serve|tools|status", err=True)
            raise typer.Exit(1)
