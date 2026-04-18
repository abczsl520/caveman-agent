# Contributing to Caveman

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/abczsl520/caveman-agent.git
cd caveman
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,anthropic]"
```

## Running Tests

```bash
pytest tests/ -q
```

All 1500+ tests should pass before submitting a PR.

## Code Standards

- Files must stay under 400 lines (enforced by tests)
- Functions should have type annotations and docstrings
- No hardcoded secrets — use environment variables or config
- Run `pytest` before every commit

## Pull Requests

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Submit a PR with a clear description

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
