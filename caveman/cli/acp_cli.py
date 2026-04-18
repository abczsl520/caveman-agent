"""CLI handlers for ACP commands."""
from __future__ import annotations

import typer

from caveman.cli.main import app


@app.command(name="acp-serve")
def acp_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8766, "--port", "-p", help="Bind port"),
    model: str = typer.Option("claude-opus-4-6", "--model", "-m", help="LLM model"),
):
    """Start ACP server — expose Caveman as an ACP-compatible agent."""
    import asyncio
    from caveman.cli.tui import show_banner

    show_banner()
    typer.echo(f"🌐 Starting ACP server on {host}:{port}...")

    async def _run():
        from caveman.acp.server import ACPServer
        from caveman.agent.factory import create_loop

        loop = create_loop(model=model)

        async def agent_fn(message: str) -> str:
            return await loop.run(message)

        server = ACPServer(host=host, port=port, agent_fn=agent_fn)
        await server.start()
        typer.echo(f"✅ ACP server ready at http://{host}:{port}/acp/v1/tasks")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await server.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo("\n🛑 ACP server stopped.")


@app.command(name="acp-send")
def acp_send(
    url: str = typer.Argument(..., help="ACP agent URL (e.g. http://localhost:8766)"),
    message: str = typer.Argument(..., help="Message to send"),
):
    """Send a task to a remote ACP agent."""
    import asyncio

    async def _run():
        from caveman.acp.client import ACPClient
        client = ACPClient(url)
        try:
            typer.echo(f"📤 Sending to {url}...")
            result = await client.send_task(message)
            typer.echo(f"Status: {result.get('status')}")
            r = result.get("result")
            if r:
                for part in r.get("parts", []):
                    if part.get("type") == "text":
                        typer.echo(part["text"])
        finally:
            await client.close()

    asyncio.run(_run())
