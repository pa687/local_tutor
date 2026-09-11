"""Symbol-manipulation tools (§7): the six non-arithmetic tools.

``solve_equation``, ``simplify_expression``, ``factor_expression``,
``differentiate``, ``integrate``, ``check_equivalence`` — every one of them a thin
wrapper that (a) parses its input through :mod:`tutor.tools.sandbox`, (b) asks SymPy
for exactly one thing, and (c) returns strings so the result is exact and reproducible.

Two conventions worth knowing:

* an equation may be written with or without ``= 0`` (``"x^2 - 5*x + 6"`` is read as
  ``= 0``), because both spellings are natural for the model and for students;
* when SymPy cannot finish — no closed-form antiderivative, ``solve`` giving up — the
  result is ``unsupported`` with an explicit message. It is never silently reported as
  "no solution", and it never becomes a fabricated answer (§12: don't hide conflicts).
"""

from __future__ import annotations

from typing import Any, Final

import sympy
from pydantic import BaseModel, ConfigDict, Field

from tutor.tools.calculator import decimal_string
from tutor.tools.registry import ToolSpec, ToolUnsupported
from tutor.tools.sandbox import parse_expression, parse_relation, parse_variable_name

MAX_ORDER: Final = 5


class _ToolArguments(BaseModel):
    """Strict base: unknown arguments are an error, not something to ignore (§7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EquationArgs(_ToolArguments):
    equation: str = Field(
        description=(
            "Equation to solve, e.g. 'x^2 - 5*x + 6 = 0' or 'x^2 - 5*x + 6' (read as '= 0')."
        ),
    )
    variable: str = Field(default="x", description="Variable to solve for.")


class ExpressionArgs(_ToolArguments):
    expression: str = Field(description="Expression to work on, e.g. '(x^2 - 1)/(x - 1)'.")


class DifferentiateArgs(ExpressionArgs):
    variable: str = Field(default="x", description="Variable to differentiate with respect to.")
    order: int = Field(
        default=1,
        ge=1,
        le=MAX_ORDER,
        description="Derivative order (1 = first derivative).",
    )


class IntegrateArgs(ExpressionArgs):
    variable: str = Field(default="x", description="Variable to integrate with respect to.")
    lower: str | None = Field(default=None, description="Lower bound for a definite integral.")
    upper: str | None = Field(default=None, description="Upper bound for a definite integral.")


class EquivalenceArgs(_ToolArguments):
    expression_a: str = Field(description="First expression.")
    expression_b: str = Field(description="Second expression.")


def solve_equation(arguments: EquationArgs) -> dict[str, Any]:
    """Solve one equation for one variable."""
    symbol = parse_variable_name(arguments.variable)
    left, right = parse_relation(arguments.equation)
    try:
        solutions = sympy.solve(sympy.Eq(left, right), symbol)
    except NotImplementedError as exc:
        raise ToolUnsupported(f"sympy cannot solve this equation: {exc}") from exc
    if not isinstance(solutions, list):
        raise ToolUnsupported("sympy returned an unusable solution shape")
    rendered = list(dict.fromkeys(sympy.sstr(solution) for solution in solutions))
    output: dict[str, Any] = {
        "variable": symbol.name,
        "solutions": rendered,
        "count": len(rendered),
    }
    if not rendered:
        output["note"] = "no solution found"
    return output


def simplify_expression(arguments: ExpressionArgs) -> dict[str, Any]:
    """Simplify an expression."""
    expression = parse_expression(arguments.expression)
    return {"simplified": sympy.sstr(sympy.simplify(expression))}


def factor_expression(arguments: ExpressionArgs) -> dict[str, Any]:
    """Factor an expression over the rationals."""
    expression = parse_expression(arguments.expression)
    return {"factored": sympy.sstr(sympy.factor(expression))}


def differentiate(arguments: DifferentiateArgs) -> dict[str, Any]:
    """Differentiate an expression with respect to one variable."""
    symbol = parse_variable_name(arguments.variable)
    derivative = sympy.diff(parse_expression(arguments.expression), symbol, arguments.order)
    return {
        "variable": symbol.name,
        "order": arguments.order,
        "derivative": sympy.sstr(derivative),
    }


def integrate(arguments: IntegrateArgs) -> dict[str, Any]:
    """Integrate an expression: indefinite, or definite when both bounds are given."""
    symbol = parse_variable_name(arguments.variable)
    expression = parse_expression(arguments.expression)
    if (arguments.lower is None) != (arguments.upper is None):
        raise ToolUnsupported("give both 'lower' and 'upper', or neither")
    if arguments.lower is None or arguments.upper is None:
        antiderivative = sympy.integrate(expression, symbol)
        if antiderivative.has(sympy.Integral):
            raise ToolUnsupported("sympy could not find an antiderivative")
        return {
            "definite": False,
            "variable": symbol.name,
            "antiderivative": sympy.sstr(antiderivative),
            "constant": "C",
        }

    lower = parse_expression(arguments.lower)
    upper = parse_expression(arguments.upper)
    value = sympy.simplify(sympy.integrate(expression, (symbol, lower, upper)))
    if value.has(sympy.Integral):
        raise ToolUnsupported("sympy could not evaluate this definite integral")
    output: dict[str, Any] = {
        "definite": True,
        "variable": symbol.name,
        "lower": sympy.sstr(lower),
        "upper": sympy.sstr(upper),
        "value": sympy.sstr(value),
    }
    if value.is_number and value.is_finite is not False:
        output["decimal"] = decimal_string(value)
    return output


def check_equivalence(arguments: EquivalenceArgs) -> dict[str, Any]:
    """Decide whether two expressions are equivalent, and show the difference."""
    left = parse_expression(arguments.expression_a)
    right = parse_expression(arguments.expression_b)
    difference = sympy.simplify(left - right)
    return {"equivalent": bool(difference == 0), "difference": sympy.sstr(difference)}


ALGEBRA_TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="solve_equation",
        description="Solve an equation for one variable and list the solutions.",
        arguments=EquationArgs,
        handler=solve_equation,
    ),
    ToolSpec(
        name="simplify_expression",
        description="Simplify an expression to its shortest equivalent form.",
        arguments=ExpressionArgs,
        handler=simplify_expression,
    ),
    ToolSpec(
        name="factor_expression",
        description="Factor an expression into a product of simpler factors.",
        arguments=ExpressionArgs,
        handler=factor_expression,
    ),
    ToolSpec(
        name="differentiate",
        description="Differentiate an expression with respect to a variable.",
        arguments=DifferentiateArgs,
        handler=differentiate,
    ),
    ToolSpec(
        name="integrate",
        description="Integrate an expression; definite when lower and upper bounds are given.",
        arguments=IntegrateArgs,
        handler=integrate,
    ),
    ToolSpec(
        name="check_equivalence",
        description="Check whether two expressions are mathematically equivalent.",
        arguments=EquivalenceArgs,
        handler=check_equivalence,
    ),
)
