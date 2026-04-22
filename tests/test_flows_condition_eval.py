"""Tests for flows condition evaluator — security and correctness."""
import pytest

from caveman.gateway.flows import FlowEngine


@pytest.fixture
def engine():
    """Minimal FlowEngine for testing _evaluate_condition."""
    return FlowEngine.__new__(FlowEngine)


class TestEvaluateCondition:
    """Verify the AST-based safe evaluator."""

    def test_simple_equality(self, engine):
        assert engine._evaluate_condition("status == 'ok'", {"status": "ok"})
        assert not engine._evaluate_condition("status == 'ok'", {"status": "fail"})

    def test_numeric_comparison(self, engine):
        assert engine._evaluate_condition("count > 5", {"count": 10})
        assert not engine._evaluate_condition("count > 5", {"count": 3})

    def test_boolean_and(self, engine):
        ctx = {"a": True, "b": True}
        assert engine._evaluate_condition("a and b", ctx)
        ctx["b"] = False
        assert not engine._evaluate_condition("a and b", ctx)

    def test_boolean_or(self, engine):
        assert engine._evaluate_condition("a or b", {"a": False, "b": True})

    def test_not(self, engine):
        assert engine._evaluate_condition("not failed", {"failed": False})

    def test_in_operator(self, engine):
        assert engine._evaluate_condition("'x' in tags", {"tags": ["x", "y"]})
        assert not engine._evaluate_condition("'z' in tags", {"tags": ["x", "y"]})

    def test_not_in(self, engine):
        assert engine._evaluate_condition("'z' not in tags", {"tags": ["x"]})

    def test_chained_comparison(self, engine):
        assert engine._evaluate_condition("1 < x < 10", {"x": 5})
        assert not engine._evaluate_condition("1 < x < 10", {"x": 15})

    def test_attribute_access(self, engine):
        class Obj:
            name = "test"
        assert engine._evaluate_condition("obj.name == 'test'", {"obj": Obj()})

    def test_constant_only(self, engine):
        assert engine._evaluate_condition("True", {})
        assert not engine._evaluate_condition("False", {})

    def test_missing_name_returns_false(self, engine):
        assert not engine._evaluate_condition("nonexistent > 0", {})

    def test_syntax_error_returns_false(self, engine):
        assert not engine._evaluate_condition("if True:", {})

    # ── Security: these MUST all return False ──

    def test_blocks_function_call(self, engine):
        """eval() would execute this; AST evaluator must reject it."""
        assert not engine._evaluate_condition("__import__('os').system('echo pwned')", {})

    def test_blocks_class_escape(self, engine):
        """Classic sandbox escape via __class__.__bases__."""
        assert not engine._evaluate_condition(
            "().__class__.__bases__[0].__subclasses__()", {}
        )

    def test_blocks_lambda(self, engine):
        assert not engine._evaluate_condition("(lambda: 1)()", {})

    def test_blocks_list_comprehension(self, engine):
        assert not engine._evaluate_condition("[x for x in range(10)]", {})

    def test_blocks_exec_in_string(self, engine):
        assert not engine._evaluate_condition("exec('import os')", {})

    def test_blocks_getattr_builtin(self, engine):
        assert not engine._evaluate_condition("getattr(__builtins__, 'eval')", {})

    def test_blocks_subscript(self, engine):
        """Subscript access should be rejected (not in allowed nodes)."""
        assert not engine._evaluate_condition("d['key']", {"d": {"key": "val"}})
