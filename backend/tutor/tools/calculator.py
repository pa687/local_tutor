"""Arithmetic tools (§7): ``calculator`` and ``evaluate_expression``.

Both are numeric: the answer is a *number*, exact where SymPy can keep it exact
(``1/3`` stays ``1/3``) with a decimal rendering alongside. Symbolic work belongs to
``simplify_expression`` / ``solve_equation`` etc., so a symbolic input here is an
``unsupported`` result rather than a guess.

Nothing in this module touches the filesystem, the network, subprocesses or Python's
``eval``: input goes through :mod:`tutor.tools.sandbox`, everything else is SymPy.
"""

from __future__ import annotations

import math
from typing import Any, Final

import sympy
from pydantic import BaseModel, ConfigDict, Field

from tutor.tools.registry import ToolSpec, ToolUnsupported
from tutor.tools.sandbox import parse_expression, parse_variable_name

MIN_PRECISION: Final = 1
MAX_PRECISION: Final = 30
DEFAULT_PRECISION: Final = 10
#: Python's shortest-repr formatting stops being useful past 15 significant digits.
_MAX_FLOAT_DIGITS: Final = 15


class _ToolArguments(BaseModel):
    """Strict base: unknown arguments are an error, not something to ignore (§7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CalculatorArgs(_ToolArguments):
    expression: str = Field(
        description="Arithmetic expression, e.g. '(3+5)*7/2'. No variables.",
    )
    precision: int = Field(
        default=DEFAULT_PRECISION,
        ge=MIN_PRECISION,
        le=MAX_PRECISION,
        description="Significant digits for the decimal rendering.",
    )


class EvaluateArgs(_ToolArguments):
    expression: str = Field(description="Expression, possibly with variables, e.g. 'x^2 - 5*x'.")
    substitutions: dict[str, str] = Field(
        default_factory=dict,
        description="Values to substitute, e.g. {'x': '3'}. Every variable must be covered.",
    )
    precision: int = Field(
        default=DEFAULT_PRECISION,
        ge=MIN_PRECISION,
        le=MAX_PRECISION,
        description="Significant digits for the decimal rendering.",
    )


def calculator(arguments: CalculatorArgs) -> dict[str, Any]:
    """Evaluate a pure arithmetic expression exactly."""
    expression = parse_expression(arguments.expression)
    if expression.free_symbols:
        raise ToolUnsupported(
            "calculator only accepts numbers; use evaluate_expression for symbolic input"
        )
    return _numeric_result(expression, arguments.precision)


def evaluate_expression(arguments: EvaluateArgs) -> dict[str, Any]:
    """Substitute values for the variables, then evaluate the result numerically."""
    # Parse first: input validation must not depend on how complete the call looks.
    expression = parse_expression(arguments.expression)
    if not arguments.substitutions:
        raise ToolUnsupported("evaluate_expression needs at least one substitution")
    substituted = expression
    for raw_name, raw_value in arguments.substitutions.items():
        symbol = parse_variable_name(raw_name)
        substituted = substituted.subs(symbol, parse_expression(raw_value))
    missing = sorted(symbol.name for symbol in substituted.free_symbols)
    if missing:
        raise ToolUnsupported(f"no value given for: {', '.join(missing)}")
    result = _numeric_result(substituted, arguments.precision)
    return {"expression": sympy.sstr(expression), **result}


def _numeric_result(expression: sympy.Expr, precision: int) -> dict[str, str]:
    """Render an expression as an exact string plus a decimal approximation."""
    value = sympy.simplify(expression)
    if value.free_symbols:
        raise ToolUnsupported("the result still contains variables")
    if not value.is_number:  # pragma: no cover - guarded by the parsers above
        raise ToolUnsupported(f"the result is not a number: {sympy.sstr(value)}")
    if value.is_finite is not True:
        raise ToolUnsupported(f"the result is not a finite number: {sympy.sstr(value)}")
    return {"result": sympy.sstr(value), "decimal": decimal_string(value, precision)}


def decimal_string(value: sympy.Expr, precision: int = DEFAULT_PRECISION) -> str:
    """A short decimal rendering; falls back to SymPy when a float cannot hold it."""
    approximate = sympy.N(value, precision)
    try:
        as_float = float(approximate)
    except (OverflowError, ValueError):
        return str(sympy.sstr(approximate))
    if not math.isfinite(as_float):  # SymPy returns inf instead of raising
        return str(sympy.sstr(approximate))
    return f"{as_float:.{min(precision, _MAX_FLOAT_DIGITS)}g}"


CALCULATOR_TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="calculator",
        description="Evaluate a pure arithmetic expression exactly (no variables).",
        arguments=CalculatorArgs,
        handler=calculator,
    ),
    ToolSpec(
        name="evaluate_expression",
        description="Substitute values for variables and evaluate the expression numerically.",
        arguments=EvaluateArgs,
        handler=evaluate_expression,
    ),
)
