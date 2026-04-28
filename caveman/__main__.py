"""Caveman — The Self-Evolving AI Agent Framework."""
from caveman.safe_stdio import install_safe_stdio

install_safe_stdio()

from caveman.cli.main import app

app()
