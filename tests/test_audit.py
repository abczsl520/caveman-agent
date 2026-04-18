"""Test: caveman audit static checks pass in CI."""
from caveman.cli.audit import (
    _find_python_files,
    check_encoding,
    check_file_size,
    check_swallowed_exceptions,
    check_uuid_truncation,
)


def test_no_open_without_encoding():
    files = _find_python_files()
    issues = check_encoding(files)
    assert issues == [], f"open() without encoding:\n" + "\n".join(issues)


def test_no_uuid_truncation():
    files = _find_python_files()
    issues = check_uuid_truncation(files)
    assert issues == [], f"UUID truncation:\n" + "\n".join(issues)


def test_no_bare_except():
    files = _find_python_files()
    issues = check_swallowed_exceptions(files)
    assert issues == [], f"Bare except:\n" + "\n".join(issues)


def test_no_god_files():
    files = _find_python_files()
    issues = check_file_size(files, max_lines=450)
    assert issues == [], f"Files over 450 lines:\n" + "\n".join(issues)


def test_all_modules_importable():
    """Every caveman module must import without error.

    Catches: missing imports after refactoring (e.g. moving classes to new files),
    broken cross-file references, typos in import paths.
    """
    import importlib
    import pkgutil
    import os

    os.environ['CAVEMAN_NO_CLI'] = '1'
    skip = {'caveman.__main__', 'caveman.cli.main', 'caveman.mcp.__main__'}
    # Optional deps that may not be installed
    optional_deps = {'discord', 'telegram', 'anthropic', 'google', 'openai', 'groq', 'mistralai', 'together'}

    errors = []
    for _, modname, _ in pkgutil.walk_packages(['caveman'], prefix='caveman.'):
        if modname in skip:
            continue
        try:
            importlib.import_module(modname)
        except ImportError as e:
            # Skip if it's an optional dependency
            msg = str(e)
            if any(dep in msg for dep in optional_deps):
                continue
            errors.append(f"{modname}: {e}")
        except Exception as e:
            errors.append(f"{modname}: {type(e).__name__}: {e}")

    assert errors == [], "Module import failures:\n" + "\n".join(errors)


def test_gateway_mock_agent_importable():
    """Gateway mock agent must be importable — slash commands depend on it."""
    from caveman.gateway.mock_agent import GatewayMockAgent
    agent = GatewayMockAgent()
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'tool_registry')
    assert hasattr(agent, 'memory_manager')


def test_all_lazy_imports_resolve():
    """Every lazy import (inside functions) must resolve to a real name.

    Catches: moved/renamed classes, deleted functions, typos in import paths.
    This is the #1 source of runtime NameErrors that unit tests miss.
    """
    import ast
    import glob
    import importlib

    lazy_imports = []
    for f in sorted(glob.glob('caveman/**/*.py', recursive=True)):
        if '__pycache__' in f:
            continue
        try:
            with open(f) as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(node):
                        if isinstance(child, ast.ImportFrom) and child.module:
                            if child.module.startswith('caveman.'):
                                for alias in child.names:
                                    lazy_imports.append((f, child.lineno, child.module, alias.name))
        except SyntaxError:
            continue

    optional_deps = {'discord', 'telegram', 'anthropic', 'google', 'openai',
                     'groq', 'mistralai', 'together'}
    errors = []
    for f, line, module, name in lazy_imports:
        try:
            mod = importlib.import_module(module)
            if not hasattr(mod, name):
                errors.append(f"{f}:{line}  from {module} import {name} → not found")
        except ImportError as e:
            if any(dep in str(e) for dep in optional_deps):
                continue
            errors.append(f"{f}:{line}  from {module} import {name} → {e}")

    assert errors == [], f"Broken lazy imports:\n" + "\n".join(errors)
