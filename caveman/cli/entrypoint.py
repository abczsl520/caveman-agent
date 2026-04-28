"""Console-script entrypoint with portable stdio setup."""
from __future__ import annotations

from caveman.safe_stdio import install_safe_stdio


def main() -> None:
    """Run the Typer app after making stdout/stderr Unicode-safe."""
    install_safe_stdio()
    from caveman.cli.main import app

    app()


__all__ = ["main"]