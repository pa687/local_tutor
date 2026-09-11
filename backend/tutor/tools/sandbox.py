"""Sandboxed parsing of mathematical expressions (ENGINEERING_PLAN.md §7, §23.7).

§7 allows structured maths tools and forbids executing model-generated Python, so the
only way into the tool system is an expression *string* — and this module is the gate
every string has to pass before SymPy sees it:

1. **shape limits** — a length cap and a node-count cap, so the work is bounded;
2. **AST allow-list** — no attribute access, no subscription, no lambda, no
   comprehension, no f-string, no string/bytes/None literal, no keyword or star
   argument; calls only to whitelisted maths functions by bare name;
3. **identifier policy** — symbol names are plain ASCII identifiers of at most 8
   characters, ``_``-prefixed names and maths-function names are refused;
4. **magnitude guard** — literal powers are bounded, so ``9**9**9`` is rejected
   instead of allocating hundreds of megabytes.

The caret is normalised to ``**`` *before* the AST is inspected, because SymPy's
``standard_transformations`` do **not** convert it (checked on SymPy 1.14) and the
bitwise reading would be silently wrong: ``2^64`` evaluates to ``66``. Doing the
replacement first also means the guards see a real ``Pow`` node.

Only then is the text handed to ``sympy.parsing.sympy_parser.parse_expr``. SymPy's own
documentation warns that ``sympify``/``parse_expr`` evaluate their input, which is
precisely why step 2 happens first and why no raw string ever reaches them.

``ExpressionRejected`` subclasses :class:`~tutor.tools.registry.ToolInputRejected`, so
the registry reports these as ``rejected`` without importing this module.

Error messages are short English diagnostics: they end up in logs and, in Phase 4, in
the self-correction prompt.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from typing import Any, Final

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from tutor.tools.registry import ToolInputRejected

#: Guard rails, not precision settings.
DEFAULT_MAX_LENGTH: Final = 500
DEFAULT_MAX_NODES: Final = 200
DEFAULT_MAX_DIGITS: Final = 10_000

_SYMBOL_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,7}$")
_RELATION_MARKERS: Final = ("==", "!=", ">=", "<=", ">", "<")
#: ``^`` is spelled as a power in school maths and in most model output.
_CARET: Final = "^"
_POWER: Final = "**"

#: The complete function vocabulary. Everything else is rejected, including
#: ``factorial`` (unbounded output) and anything reaching for attributes.
ALLOWED_FUNCTIONS: Final[Mapping[str, Any]] = {
    "sqrt": sympy.sqrt,
    "root": sympy.root,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "asinh": sympy.asinh,
    "acosh": sympy.acosh,
    "atanh": sympy.atanh,
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "sign": sympy.sign,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "Min": sympy.Min,
    "Max": sympy.Max,
}

ALLOWED_CONSTANTS: Final[Mapping[str, Any]] = {"pi": sympy.pi, "E": sympy.E}

_ALLOWED_BINARY_OPS: Final = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY_OPS: Final = (ast.UAdd, ast.USub)


class ExpressionRejected(ToolInputRejected):
    """The text is not something we are willing to hand to SymPy."""


def parse_expression(
    text: str,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_digits: int = DEFAULT_MAX_DIGITS,
) -> sympy.Expr:
    """Parse ``text`` into a SymPy expression, or raise :class:`ExpressionRejected`."""
    stripped = _screen_text(text, max_length=max_length).replace(_CARET, _POWER)
    tree = _parse_ast(stripped)
    if sum(1 for _ in ast.walk(tree)) > max_nodes:
        raise ExpressionRejected(f"expression has more than {max_nodes} tokens")

    names: set[str] = set()
    _inspect(tree.body, names, max_digits=max_digits)

    local_dict: dict[str, Any] = {name: sympy.Symbol(name) for name in sorted(names)}
    local_dict.update(ALLOWED_CONSTANTS)
    local_dict.update(ALLOWED_FUNCTIONS)
    try:
        expression = parse_expr(
            stripped,
            local_dict=local_dict,
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:  # SymPy raises a wide variety of exception types
        raise ExpressionRejected(f"sympy could not parse the expression: {exc}") from exc
    if not isinstance(expression, sympy.Expr):
        raise ExpressionRejected("the text did not parse into a symbolic expression")
    return expression


def parse_relation(
    text: str,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_digits: int = DEFAULT_MAX_DIGITS,
) -> tuple[sympy.Expr, sympy.Expr]:
    """Parse ``"lhs = rhs"`` into both sides; a bare expression is read as ``= 0``."""
    limits: dict[str, int] = {
        "max_length": max_length,
        "max_nodes": max_nodes,
        "max_digits": max_digits,
    }
    if not isinstance(text, str):
        raise ExpressionRejected("expression must be a string")
    if any(marker in text for marker in _RELATION_MARKERS) or text.count("=") > 1:
        raise ExpressionRejected("only a single '=' is supported between two sides")
    if text.count("=") == 1:
        left, right = text.split("=")
        return parse_expression(left, **limits), parse_expression(right, **limits)
    return parse_expression(text, **limits), sympy.Integer(0)


def parse_variable_name(name: str) -> sympy.Symbol:
    """Validate and return the symbol a tool was asked to work with."""
    if not isinstance(name, str):
        raise ExpressionRejected("variable must be a string")
    candidate = name.strip()
    if not _SYMBOL_PATTERN.match(candidate):
        raise ExpressionRejected(
            f"invalid variable name {name!r}: use 1-8 ASCII letters/digits, not starting with '_'"
        )
    if candidate in ALLOWED_FUNCTIONS or candidate in ALLOWED_CONSTANTS:
        raise ExpressionRejected(f"{candidate!r} is a function or constant, not a variable")
    return sympy.Symbol(candidate)


def _screen_text(text: str, *, max_length: int) -> str:
    if not isinstance(text, str):
        raise ExpressionRejected("expression must be a string")
    stripped = text.strip()
    if not stripped:
        raise ExpressionRejected("expression is empty")
    if len(stripped) > max_length:
        raise ExpressionRejected(f"expression is longer than {max_length} characters")
    if not stripped.isascii():
        raise ExpressionRejected("expression must be ASCII")
    return stripped


def _parse_ast(text: str) -> ast.Expression:
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionRejected(f"not a valid maths expression: {exc.msg}") from exc
    if not isinstance(parsed, ast.Expression):
        raise ExpressionRejected("not a single expression")
    return parsed


def _inspect(node: ast.AST, names: set[str], *, max_digits: int) -> None:
    """Walk the allow-list. ``names`` collects the symbols actually used."""
    if isinstance(node, ast.Constant):
        _check_constant(node)
        return
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_FUNCTIONS:
            raise ExpressionRejected(f"{node.id!r} is a function and must be called")
        if not _SYMBOL_PATTERN.match(node.id):
            raise ExpressionRejected(f"invalid name {node.id!r}")
        names.add(node.id)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPS):
            raise ExpressionRejected(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow):
            _guard_power(node, max_digits=max_digits)
        _inspect(node.left, names, max_digits=max_digits)
        _inspect(node.right, names, max_digits=max_digits)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPS):
            raise ExpressionRejected(f"operator {type(node.op).__name__} is not allowed")
        _inspect(node.operand, names, max_digits=max_digits)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ExpressionRejected("only calls to known maths functions are allowed")
        if node.keywords:
            raise ExpressionRejected("keyword arguments are not allowed")
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                raise ExpressionRejected("star arguments are not allowed")
            _inspect(argument, names, max_digits=max_digits)
        return
    raise ExpressionRejected(f"syntax {type(node).__name__} is not allowed")


def _check_constant(node: ast.Constant) -> None:
    if isinstance(node.value, bool):
        raise ExpressionRejected("boolean literals are not allowed")
    if isinstance(node.value, int):
        return
    if isinstance(node.value, float) and math.isfinite(node.value):
        return
    raise ExpressionRejected(f"literal {node.value!r} is not allowed")


def _guard_power(node: ast.BinOp, *, max_digits: int) -> None:
    """Reject literal powers whose exact value cannot possibly be useful."""
    base = _literal_value(node.left, max_digits=max_digits)
    exponent = _literal_value(node.right, max_digits=max_digits)
    if base is None or exponent is None or exponent <= 0 or abs(base) <= 1:
        return
    digits = exponent * math.log10(abs(base))
    if digits > max_digits:
        raise ExpressionRejected(
            f"power too large: the result would have about {digits:.0f} digits"
        )


def _literal_value(node: ast.AST, *, max_digits: int) -> float | None:
    """Value of a symbol-free subtree, or ``None`` when a symbol is involved."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, (int, float)):
            return _finite(float(node.value))
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
        inner = _literal_value(node.operand, max_digits=max_digits)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left, max_digits=max_digits)
        right = _literal_value(node.right, max_digits=max_digits)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return _finite(left + right)
            if isinstance(node.op, ast.Sub):
                return _finite(left - right)
            if isinstance(node.op, ast.Mult):
                return _finite(left * right)
            if isinstance(node.op, ast.Div):
                return None if right == 0 else _finite(left / right)
            if isinstance(node.op, ast.FloorDiv):
                return None if right == 0 else _finite(left // right)
            if isinstance(node.op, ast.Mod):
                return None if right == 0 else _finite(left % right)
            if isinstance(node.op, ast.Pow):
                if abs(left) > 1 and right > 0 and right * math.log10(abs(left)) > max_digits:
                    raise ExpressionRejected("power too large to evaluate safely")
                return _finite(left**right)
        except OverflowError:
            raise ExpressionRejected("numeric literal is too large to evaluate safely") from None
    return None


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ExpressionRejected("numeric literal is out of range")
    return value
