"""Tests for the six symbol-manipulation tools (ENGINEERING_PLAN.md §7).

§22's acceptance scenarios run through these tools in Phase 4, so the expectations here
are the reference answers: ``x^2 - 5*x + 6 = 0`` must come back as ``["2", "3"]``, and
anything SymPy cannot do must say so instead of inventing a result.
"""

from __future__ import annotations

import pytest

from tutor.tools.registry import ToolErrorKind, ToolRegistry, ToolResult

SOLUTIONS = ["2", "3"]


class TestSolveEquation:
    def test_solves_the_acceptance_equation(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x**2 - 5*x + 6", "variable": "x"})
        assert result.ok, result.error
        assert result.output == {
            "variable": "x",
            "solutions": SOLUTIONS,
            "count": 2,
        }

    def test_accepts_an_explicit_equals_zero(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x^2 - 5*x + 6 = 0"})
        assert result.output_or_raise()["solutions"] == SOLUTIONS

    def test_accepts_a_non_zero_right_hand_side(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "2*x + 1 = 7"})
        assert result.output_or_raise()["solutions"] == ["3"]

    def test_solves_for_the_requested_variable(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "y - 3 = 0", "variable": "y"})
        assert result.output == {"variable": "y", "solutions": ["3"], "count": 1}

    def test_reports_an_empty_solution_set_explicitly(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x + 1 = x + 2"})
        assert result.ok, result.error
        assert result.output == {
            "variable": "x",
            "solutions": [],
            "count": 0,
            "note": "no solution found",
        }

    def test_complex_solutions_are_reported_not_hidden(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x^2 + 1 = 0"})
        assert result.output_or_raise()["solutions"] == ["-I", "I"]

    def test_missing_equation_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"variable": "x"})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS

    def test_bad_variable_name_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x = 1", "variable": "__class__"})
        assert result.error_kind is ToolErrorKind.REJECTED


class TestSimplifyExpression:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("(x^2 - 1)/(x - 1)", "x + 1"),
            ("sin(x)^2 + cos(x)^2", "1"),
            ("2*x + 3*x", "5*x"),
            ("(x + 1)^2", "(x + 1)**2"),
        ],
    )
    def test_simplifies(self, tools: ToolRegistry, expression: str, expected: str) -> None:
        result = tools.call("simplify_expression", {"expression": expression})
        assert result.ok, result.error
        assert result.output == {"simplified": expected}

    def test_rejects_malicious_input(self, tools: ToolRegistry) -> None:
        result = tools.call("simplify_expression", {"expression": "__import__('os')"})
        assert result.error_kind is ToolErrorKind.REJECTED


class TestFactorExpression:
    def test_factors_a_difference_of_squares(self, tools: ToolRegistry) -> None:
        result = tools.call("factor_expression", {"expression": "x^2 - 1"})
        assert result.output == {"factored": "(x - 1)*(x + 1)"}

    def test_factors_the_acceptance_quadratic(self, tools: ToolRegistry) -> None:
        result = tools.call("factor_expression", {"expression": "x^2 - 5*x + 6"})
        # SymPy does not promise a factor order, so accept either arrangement.
        assert result.output == {"factored": "(x - 2)*(x - 3)"} or result.output == {
            "factored": "(x - 3)*(x - 2)"
        }

    def test_an_irreducible_expression_comes_back_unchanged(self, tools: ToolRegistry) -> None:
        assert tools.call("factor_expression", {"expression": "x + 1"}).output == {
            "factored": "x + 1"
        }


class TestDifferentiate:
    def test_first_derivative(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "x^3"})
        assert result.output == {"variable": "x", "order": 1, "derivative": "3*x**2"}

    def test_higher_order(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "x^3", "order": 3})
        assert result.output_or_raise()["derivative"] == "6"

    def test_respects_the_variable(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "y^2 + x", "variable": "y"})
        assert result.output_or_raise()["derivative"] == "2*y"

    def test_order_out_of_range_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "x^3", "order": 9})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS

    def test_order_zero_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "x^3", "order": 0})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS


class TestIntegrate:
    def test_indefinite_integral_mentions_the_constant(self, tools: ToolRegistry) -> None:
        result = tools.call("integrate", {"expression": "x"})
        assert result.output == {
            "definite": False,
            "variable": "x",
            "antiderivative": "x**2/2",
            "constant": "C",
        }

    def test_definite_integral_gives_an_exact_value(self, tools: ToolRegistry) -> None:
        result = tools.call("integrate", {"expression": "x", "lower": "0", "upper": "1"})
        assert result.output == {
            "definite": True,
            "variable": "x",
            "lower": "0",
            "upper": "1",
            "value": "1/2",
            "decimal": "0.5",
        }

    def test_definite_integral_over_a_symbolic_bound(self, tools: ToolRegistry) -> None:
        result = tools.call("integrate", {"expression": "x", "lower": "0", "upper": "a"})
        assert result.ok, result.error
        assert result.output_or_raise()["value"] == "a**2/2"
        assert "decimal" not in result.output_or_raise()

    def test_half_a_bound_is_unsupported(self, tools: ToolRegistry) -> None:
        result = tools.call("integrate", {"expression": "x", "lower": "0"})
        assert result.error_kind is ToolErrorKind.UNSUPPORTED
        assert "both" in (result.error or "")

    def test_unsolvable_antiderivative_is_reported_as_unsupported(
        self, tools: ToolRegistry
    ) -> None:
        result = tools.call("integrate", {"expression": "exp(sin(x))"})
        assert result.error_kind is ToolErrorKind.UNSUPPORTED
        assert "antiderivative" in (result.error or "")


class TestCheckEquivalence:
    @pytest.mark.parametrize(
        ("left", "right", "equivalent"),
        [
            ("(x + 1)^2", "x^2 + 2*x + 1", True),
            ("(x - 2)*(x - 3)", "x^2 - 5*x + 6", True),
            ("sin(x)^2 + cos(x)^2", "1", True),
            ("x + 1", "x + 2", False),
            ("2*x", "x", False),
        ],
    )
    def test_compares_two_expressions(
        self, tools: ToolRegistry, left: str, right: str, equivalent: bool
    ) -> None:
        result = tools.call("check_equivalence", {"expression_a": left, "expression_b": right})
        assert result.ok, result.error
        assert result.output is not None
        assert result.output_or_raise()["equivalent"] is equivalent

    def test_shows_the_difference(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "check_equivalence", {"expression_a": "x^2 + 2*x + 1", "expression_b": "(x+1)^2"}
        )
        assert result.output == {"equivalent": True, "difference": "0"}

    def test_the_difference_is_simplified(self, tools: ToolRegistry) -> None:
        result = tools.call("check_equivalence", {"expression_a": "x + 2", "expression_b": "x + 1"})
        assert result.output == {"equivalent": False, "difference": "1"}

    def test_missing_side_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("check_equivalence", {"expression_a": "x"})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS


class TestToolContractConsistency:
    """§7 DoD: all eight tools share one argument-validation and error style."""

    @pytest.mark.parametrize(
        "tool", ["calculator", "evaluate_expression", "solve_equation", "simplify_expression"]
    )
    def test_empty_arguments_are_a_validation_error_not_a_crash(
        self, tools: ToolRegistry, tool: str
    ) -> None:
        result: ToolResult = tools.call(tool, {})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert result.output is None

    @pytest.mark.parametrize(
        "tool",
        [
            "calculator",
            "evaluate_expression",
            "solve_equation",
            "simplify_expression",
            "factor_expression",
            "differentiate",
            "integrate",
            "check_equivalence",
        ],
    )
    def test_every_tool_rejects_absolute_paths_and_imports(
        self, tools: ToolRegistry, tool: str
    ) -> None:
        payloads = [
            {"expression": "open('/etc/passwd')"},
            {"equation": "__import__('os')"},
            {"expression_a": "eval('1')", "expression_b": "1"},
        ]
        for payload in payloads:
            result = tools.call(tool, payload)
            assert result.error_kind in {
                ToolErrorKind.REJECTED,
                ToolErrorKind.INVALID_ARGUMENTS,
            }
