"""Tests for the expression sandbox (ENGINEERING_PLAN.md §7, §23.7).

This is the security boundary of the whole tool system: if something executes code
here, everything downstream is compromised. The rejection list below is therefore the
most important test data in Phase 3.
"""

from __future__ import annotations

import pytest
import sympy

from tutor.tools.registry import ToolInputRejected
from tutor.tools.sandbox import (
    ALLOWED_FUNCTIONS,
    ExpressionRejected,
    parse_expression,
    parse_relation,
    parse_variable_name,
)


class TestValidExpressions:
    @pytest.mark.parametrize(
        "text",
        [
            "2+3*4",
            "x**2 - 5*x + 6",
            "x^2",
            "1/3",
            "sqrt(2)",
            "pi",
            "E",
            "sin(x)",
            "abs(-3)",
            "Abs(-3)",
            "-x",
            "2*x**3",
            "7 % 3",
            "7 // 2",
            "log(x)",
            "Max(1, 2)",
            "2**64",
            "x*y + z_1",
        ],
    )
    def test_accepts_ordinary_mathematical_text(self, text: str) -> None:
        assert isinstance(parse_expression(text), sympy.Expr)

    def test_caret_means_power(self) -> None:
        assert parse_expression("x^2") == sympy.Symbol("x") ** 2

    def test_pi_is_a_constant_not_a_symbol(self) -> None:
        assert parse_expression("pi").free_symbols == set()

    def test_collects_the_symbols_used(self) -> None:
        assert parse_expression("a*b + c").free_symbols == {
            sympy.Symbol("a"),
            sympy.Symbol("b"),
            sympy.Symbol("c"),
        }

    def test_trims_surrounding_whitespace(self) -> None:
        assert parse_expression("  1 + 1  ") == 2

    def test_large_but_reasonable_power_is_allowed(self) -> None:
        assert parse_expression("2**64") == 18446744073709551616

    def test_rejection_is_a_tool_input_rejection(self) -> None:
        """The registry maps this without importing the sandbox (§7)."""
        assert issubclass(ExpressionRejected, ToolInputRejected)


class TestSecurityRejections:
    @pytest.mark.parametrize(
        "text",
        [
            # the obvious escapes
            "__import__('os').system('id')",
            "().__class__.__bases__[0].__subclasses__()",
            "x.__class__",
            "x.real",
            "getattr(x, 'real')",
            "open('/etc/passwd')",
            "eval('1+1')",
            "exec('x=1')",
            "globals()",
            "print(x)",
            "sympy.sqrt(2)",
            "factorial(100)",
            # syntax that has no place in a maths expression
            "lambda: 1",
            "[x for x in ()]",
            "{1: 2}",
            "{1, 2}",
            "f'{x}'",
            "'abc'",
            "b'abc'",
            "None",
            "True",
            "x if y else z",
            "x := 1",
            "x and y",
            "x or y",
            "not x",
            "x < 1",
            "x == 1",
            "x | y",
            "x @ y",
            "import os",
            "x; y",
        ],
    )
    def test_rejects_anything_outside_the_allow_list(self, text: str) -> None:
        with pytest.raises(ExpressionRejected):
            parse_expression(text)

    @pytest.mark.parametrize("text", ["9**9**9", "10**100000", "2**1000000000"])
    def test_rejects_literal_powers_that_would_explode(self, text: str) -> None:
        with pytest.raises(ExpressionRejected, match="too large"):
            parse_expression(text)

    @pytest.mark.parametrize("text", ["", "   ", "\t\n"])
    def test_rejects_empty_input(self, text: str) -> None:
        with pytest.raises(ExpressionRejected, match="empty"):
            parse_expression(text)

    @pytest.mark.parametrize("value", [None, 42, ["x"], {"x": 1}])
    def test_rejects_non_strings(self, value: object) -> None:
        with pytest.raises(ExpressionRejected, match="string"):
            parse_expression(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("text", ["解方程", "x² + 1", "α + β"])
    def test_rejects_non_ascii(self, text: str) -> None:
        with pytest.raises(ExpressionRejected, match="ASCII"):
            parse_expression(text)

    def test_rejects_overlong_text(self) -> None:
        with pytest.raises(ExpressionRejected, match="longer than"):
            parse_expression("1+" * 300 + "1")

    def test_length_limit_is_configurable(self) -> None:
        with pytest.raises(ExpressionRejected, match="longer than 3"):
            parse_expression("1 + 1", max_length=3)

    def test_rejects_too_many_tokens(self) -> None:
        with pytest.raises(ExpressionRejected, match="tokens"):
            parse_expression("1+" * 150 + "1", max_length=10_000, max_nodes=50)

    def test_node_limit_is_configurable(self) -> None:
        assert parse_expression("1+1", max_nodes=5) == 2
        with pytest.raises(ExpressionRejected, match="tokens"):
            parse_expression("1+1", max_nodes=4)

    @pytest.mark.parametrize("text", ["1e400", "-1e400"])
    def test_rejects_non_finite_literals(self, text: str) -> None:
        with pytest.raises(ExpressionRejected, match="literal"):
            parse_expression(text)

    def test_rejects_invalid_syntax(self) -> None:
        with pytest.raises(ExpressionRejected, match="not a valid maths expression"):
            parse_expression("x**")

    def test_rejects_a_function_used_as_a_bare_name(self) -> None:
        with pytest.raises(ExpressionRejected, match="must be called"):
            parse_expression("sin + 1")

    def test_every_rejection_message_is_short_and_has_no_traceback(self) -> None:
        with pytest.raises(ExpressionRejected) as excinfo:
            parse_expression("__import__('os').system('id')")
        message = str(excinfo.value)
        assert len(message) < 200
        assert "Traceback" not in message
        assert "__import__" not in message


class TestVariableNames:
    @pytest.mark.parametrize("name", ["x", "y", "n", "abc123", "x_1", "  x  "])
    def test_accepts_plain_identifiers(self, name: str) -> None:
        assert isinstance(parse_variable_name(name), sympy.Symbol)

    @pytest.mark.parametrize(
        "name",
        ["", "_x", "__class__", "abcdefghij", "1x", "x y", "解", "sin", "pi", "E", "x-y"],
    )
    def test_rejects_names_outside_the_policy(self, name: str) -> None:
        with pytest.raises(ExpressionRejected):
            parse_variable_name(name)

    def test_rejects_non_strings(self) -> None:
        with pytest.raises(ExpressionRejected):
            parse_variable_name(1)  # type: ignore[arg-type]

    def test_function_names_are_not_symbols(self) -> None:
        assert "sin" in ALLOWED_FUNCTIONS


class TestParseRelation:
    def test_accepts_a_two_sided_equation(self) -> None:
        left, right = parse_relation("x^2 - 5*x + 6 = 0")
        assert left == sympy.Symbol("x") ** 2 - 5 * sympy.Symbol("x") + 6
        assert right == 0

    def test_a_bare_expression_is_read_as_equal_to_zero(self) -> None:
        left, right = parse_relation("x^2 - 5*x + 6")
        assert right == 0
        assert not left.free_symbols - {sympy.Symbol("x")}

    def test_reads_the_right_hand_side(self) -> None:
        left, right = parse_relation("2*x = 4")
        assert (left, right) == (2 * sympy.Symbol("x"), 4)

    @pytest.mark.parametrize("text", ["x == 1", "x >= 1", "x <= 1", "x != 1", "x < 1", "x = 1 = 2"])
    def test_rejects_multi_way_relations(self, text: str) -> None:
        with pytest.raises(ExpressionRejected):
            parse_relation(text)

    def test_empty_side_is_rejected(self) -> None:
        with pytest.raises(ExpressionRejected):
            parse_relation("x =")

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(ExpressionRejected, match="string"):
            parse_relation(None)  # type: ignore[arg-type]
