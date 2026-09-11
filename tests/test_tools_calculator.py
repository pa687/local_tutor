"""Tests for ``calculator`` and ``evaluate_expression`` (ENGINEERING_PLAN.md §7).

Arithmetic tools are always called through the registry, because that is the path the
rest of the system uses: validation, timeout, trace and error typing included.
"""

from __future__ import annotations

import pytest
import sympy

from tutor.tools.calculator import decimal_string
from tutor.tools.registry import ToolErrorKind, ToolRegistry, ToolResult


def call(tools: ToolRegistry, expression: str, **extra: object) -> ToolResult:
    return tools.call("calculator", {"expression": expression, **extra})


class TestCalculator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2+3*4", "14"),
            ("(3+5)*7/2", "28"),
            ("2**10", "1024"),
            ("1/3", "1/3"),
            ("sqrt(8)", "2*sqrt(2)"),
            ("-5 + 5", "0"),
            ("pi", "pi"),
        ],
    )
    def test_evaluates_exactly(self, tools: ToolRegistry, expression: str, expected: str) -> None:
        result = call(tools, expression)
        assert result.ok, result.error
        assert result.output is not None
        assert result.output_or_raise()["result"] == expected

    def test_float_input_keeps_a_sensible_decimal(self, tools: ToolRegistry) -> None:
        result = call(tools, "2^0.5")
        assert result.ok, result.error
        assert str(result.output_or_raise()["decimal"]).startswith("1.41421356")

    def test_keeps_exact_fractions_with_a_decimal_twin(self, tools: ToolRegistry) -> None:
        result = call(tools, "1/3")
        assert result.output == {"result": "1/3", "decimal": "0.3333333333"}

    def test_precision_changes_only_the_decimal(self, tools: ToolRegistry) -> None:
        result = call(tools, "1/3", precision=3)
        assert result.output == {"result": "1/3", "decimal": "0.333"}

    def test_caret_is_accepted_as_a_power(self, tools: ToolRegistry) -> None:
        assert call(tools, "10^3").output == {"result": "1000", "decimal": "1000"}

    def test_rejects_symbolic_input_with_a_useful_hint(self, tools: ToolRegistry) -> None:
        result = call(tools, "x + 1")
        assert result.error_kind is ToolErrorKind.UNSUPPORTED
        assert "evaluate_expression" in (result.error or "")

    @pytest.mark.parametrize("expression", ["1/0", "0/0"])
    def test_non_finite_results_are_not_reported_as_numbers(
        self, tools: ToolRegistry, expression: str
    ) -> None:
        result = call(tools, expression)
        assert result.error_kind is ToolErrorKind.UNSUPPORTED
        assert "finite" in (result.error or "")

    def test_missing_argument_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call("calculator", {})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert "expression" in (result.error or "")

    def test_unknown_argument_is_rejected(self, tools: ToolRegistry) -> None:
        result = call(tools, "1+1", trick="rm -rf /")
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert "trick" in (result.error or "")

    @pytest.mark.parametrize("precision", [0, -1, 31, "many"])
    def test_out_of_range_precision_is_rejected(
        self, tools: ToolRegistry, precision: object
    ) -> None:
        result = call(tools, "1/3", precision=precision)
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS

    def test_malicious_expression_is_rejected_not_executed(self, tools: ToolRegistry) -> None:
        result = call(tools, "__import__('os').system('id')")
        assert result.error_kind is ToolErrorKind.REJECTED


class TestEvaluateExpression:
    def test_substitutes_and_evaluates(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression",
            {"expression": "x^2 - 5*x", "substitutions": {"x": "3"}},
        )
        assert result.ok, result.error
        assert result.output == {
            "expression": "x**2 - 5*x",
            "result": "-6",
            "decimal": "-6",
        }

    def test_reports_the_parsed_expression(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression", {"expression": "1/x", "substitutions": {"x": "4"}}
        )
        assert result.output_or_raise()["expression"] == "1/x"
        assert result.output_or_raise()["result"] == "1/4"

    def test_numeric_substitution_from_an_expression(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression",
            {"expression": "x + 1", "substitutions": {"x": "1/2"}},
        )
        assert result.output_or_raise()["result"] == "3/2"
        assert result.output_or_raise()["decimal"] == "1.5"

    def test_missing_substitution_is_unsupported(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression", {"expression": "x + y", "substitutions": {"x": "1"}}
        )
        assert result.error_kind is ToolErrorKind.UNSUPPORTED
        assert "y" in (result.error or "")

    def test_empty_substitutions_is_unsupported(self, tools: ToolRegistry) -> None:
        result = tools.call("evaluate_expression", {"expression": "x + 1", "substitutions": {}})
        assert result.error_kind is ToolErrorKind.UNSUPPORTED

    def test_bad_substitution_name_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression", {"expression": "x", "substitutions": {"__class__": "1"}}
        )
        assert result.error_kind is ToolErrorKind.REJECTED

    def test_bad_substitution_value_is_rejected(self, tools: ToolRegistry) -> None:
        result = tools.call(
            "evaluate_expression",
            {"expression": "x", "substitutions": {"x": "__import__('os')"}},
        )
        assert result.error_kind is ToolErrorKind.REJECTED

    def test_substitutions_must_be_a_mapping(self, tools: ToolRegistry) -> None:
        result = tools.call("evaluate_expression", {"expression": "x", "substitutions": ["2"]})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS


class TestDecimalString:
    def test_formats_significant_digits(self) -> None:
        assert decimal_string(sympy.Rational(1, 3), 5) == "0.33333"

    def test_falls_back_when_a_float_would_be_infinite(self) -> None:
        """``float(Float('1e500'))`` is ``inf``, which must never reach the model."""
        rendered = decimal_string(sympy.Integer(10) ** 500, 10)
        assert rendered != "inf"
        assert rendered.startswith("1.0000000")
        assert rendered.endswith("e+500")

    def test_keeps_integers_short(self) -> None:
        assert decimal_string(sympy.Integer(1024)) == "1024"
