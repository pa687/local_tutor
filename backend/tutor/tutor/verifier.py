"""Claim extraction and SymPy-backed verification (ENGINEERING_PLAN.md §7, §22).

§7 wants the answer *checked*, not merely generated:

```text
LLM 解题 → 抽取可验证 claim → SymPy 校验 → 若冲突让模型重检 → 再次验证 → 最终回答
```

This module does the two middle steps, deterministically:

* **extraction** — pull solution claims (``x = 2 或 x = 3``) out of a piece of text,
  either the model's answer or the student's own work (§22 B/E);
* **checking** — substitute every claimed value into the equation found in the
  question and ask ``evaluate_expression`` for the residual.

The judge is SymPy, never another model: a second LLM call would be free to agree with
the first one's mistake, and §7 says "结果由 SymPy 验证". Extraction is deliberately
conservative — only closed-form values for the single unknown of a reference equation
count as checkable, and anything else is reported as ``nothing_to_verify`` rather than
guessed at.

Status vocabulary (the exact mapping onto §7's "仍然无法验证"):

* ``verified`` — claims found, reference found, every substitution holds;
* ``conflict`` — claims found, reference found, at least one substitution fails;
* ``unverifiable`` — claims + reference found, but the tool could not decide
  (unsupported / rejected / timeout): §7's "我们试过但没成功" → ``confidence = low``
  plus a warning;
* ``nothing_to_verify`` — no claim, or no reference equation to check it against.
  A conceptual answer is not a failure, so nothing is downgraded.

The re-check itself (re-prompting the model after a conflict) is the engine's job: this
module reports, it does not talk to the model.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import sympy

from tutor.tools.registry import ToolRegistry, ToolTrace
from tutor.tools.sandbox import (
    ALLOWED_FUNCTIONS,
    ExpressionRejected,
    parse_expression,
    parse_relation,
)
from tutor.tutor.response import VerificationStatus

logger = logging.getLogger(__name__)

ANSWER_SOURCE: Final = "answer"
STUDENT_SOURCE: Final = "student"

#: Numeric slack for a residual that is zero only up to the model's rounding.
ZERO_TOLERANCE: Final = 1e-6

_MATH_RUN: Final = re.compile(r"[0-9A-Za-z_+\-*/^().=\s]{3,}")
#: The ``variable =`` half of a solution statement; the value list is whatever follows
#: it, up to the next assignment or the end of the clause. Scanning assignments first
#: (instead of one greedy value pattern) is what lets ``x = 2 或 x = 3`` and
#: ``当 a = 0 时，x = 2`` both come out right.
_ASSIGNMENT: Final = re.compile(r"([A-Za-z][A-Za-z0-9_]{0,7})\s*=")
#: Statements are scanned paragraph by paragraph and sentence by sentence, because a
#: corrected answer very often *quotes* the value it just rejected ("x = 3 不是解"),
#: usually in an earlier paragraph than the answer itself.
_PARAGRAPH: Final = re.compile(r"\n\s*\n")
_SENTENCE: Final = re.compile(r"[。！？!?；;\n]+")
_NEGATIONS: Final = ("不是", "不对", "错误", "错的", "排除", "并非", "不符合", "≠")
#: A value is a *claim* when the sentence concludes with it; when a cue sits right next
#: to the assignment the value is being tested ("当 x = 2 时", "x = 2 代入后").
_CUES: Final = ("当", "时", "代入", "令", "设", "取", "检验", "验证", "试", "假设")
_CUE_WINDOW: Final = 6
_MAX_TAIL: Final = 80
_VALUE_SEPARATORS: Final = re.compile(r"[,，、]|\s+或\s+|\s+and\s+|\s+和\s+")
_VALUE_TRIM: Final = "。.,，、；;:：!！?？\"' 　"
_FRACTION: Final = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT: Final = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_BRACED_POWER: Final = re.compile(r"\^\s*\{([^{}]*)\}")
_BRACED_SUBSCRIPT: Final = re.compile(r"_\s*\{([^{}]*)\}")
#: ``5x`` → ``5*x`` and ``3(x+1)`` → ``3*(x+1)``, as school maths writes it. A function
#: name is never preceded by a digit, so ``sin(x)`` is left alone.
_IMPLICIT_PRODUCT_LETTER: Final = re.compile(r"(?<=[0-9])(?=[A-Za-z_])")
_IMPLICIT_PRODUCT_PAREN: Final = re.compile(r"(?<=[0-9)])\s*(?=\()")
#: ``√2`` / ``√(x+1)`` → ``sqrt(2)`` / ``sqrt(x+1)``.
_RADICAL_PAREN: Final = re.compile(r"√\s*\(")
_RADICAL_OPERAND: Final = re.compile(r"√\s*([0-9A-Za-z_]+)")
#: Notation that Chinese textbooks, students and models use instead of ASCII. Without
#: this, ``x² - 5x + 6 = 0`` would not even be recognised as an equation.
_UNICODE_MATH: Final[Mapping[str, str]] = {
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "⋅": "*",
    "·": "*",
    "÷": "/",
    "π": "pi",
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
}
_LATEX_NOISE: Final = (
    ("\\left", ""),
    ("\\right", ""),
    ("\\times", "*"),
    ("\\cdot", "*"),
    ("\\div", "/"),
    ("\\,", ""),
    ("\\;", ""),
    ("\\!", ""),
    ("$", ""),
)


@dataclass(frozen=True)
class Claim:
    """One verifiable assertion extracted from a piece of text."""

    kind: str
    variable: str
    values: tuple[str, ...]
    snippet: str
    source: str
    sentence_index: int = 0
    paragraph_index: int = 0

    def as_event(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "values": list(self.values),
            "snippet": self.snippet,
            "source": self.source,
        }


@dataclass(frozen=True)
class ClaimCheck:
    """The tool verdict for one claimed value."""

    claim: Claim
    value: str
    residual: str
    satisfied: bool

    def as_event(self) -> dict[str, Any]:
        return {
            "variable": self.claim.variable,
            "value": self.value,
            "residual": self.residual,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class VerificationReport:
    """What verification could (not) establish about one piece of text."""

    status: VerificationStatus
    source: str
    claims: tuple[Claim, ...] = ()
    checks: tuple[ClaimCheck, ...] = ()
    detail: str | None = None
    trace: ToolTrace = ToolTrace()

    @property
    def tools_used(self) -> tuple[str, ...]:
        """Tools that actually produced a result (§6 ``tools_used``)."""
        return self.trace.tools_used

    @property
    def conflicts(self) -> tuple[ClaimCheck, ...]:
        return tuple(check for check in self.checks if not check.satisfied)

    @property
    def verified_values(self) -> tuple[str, ...]:
        return tuple(check.value for check in self.checks if check.satisfied)

    def instruction(self) -> str | None:
        """A factual tool verdict to put in the context, or ``None`` if there is none.

        Only ``verified`` / ``conflict`` carry usable evidence; the other statuses say
        nothing about the text, so the model is told nothing at all.
        """
        if self.status in {VerificationStatus.VERIFIED, VerificationStatus.CONFLICT}:
            return f"工具校验（SymPy）：{self.detail}"
        return None

    def as_event(self) -> dict[str, Any]:
        """JSON-serializable verdict plus the full tool audit trail (§17)."""
        return {
            "status": self.status.value,
            "source": self.source,
            "detail": self.detail,
            "claims": [claim.as_event() for claim in self.claims],
            "checks": [check.as_event() for check in self.checks],
            "tools": self.trace.as_events(),
        }


class Verifier:
    """Extracts solution claims and checks them with the §7 tool registry."""

    def __init__(self, tools: ToolRegistry, *, zero_tolerance: float = ZERO_TOLERANCE) -> None:
        self._tools = tools
        self._zero_tolerance = zero_tolerance

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    # ------------------------------------------------------------------ public
    def verify(
        self,
        *,
        question: str,
        text: str,
        source: str = ANSWER_SOURCE,
    ) -> VerificationReport:
        """Check the solution claims in ``text`` against the equation in ``question``."""
        relation = self.find_relation(question)
        if relation is None:
            return VerificationReport(
                status=VerificationStatus.NOTHING_TO_VERIFY,
                source=source,
                detail="题目里没有可识别的方程，无法自动校验",
            )
        variable = _single_symbol(relation)
        if variable is None:
            return VerificationReport(
                status=VerificationStatus.NOTHING_TO_VERIFY,
                source=source,
                detail="方程里有多个未知量，无法确定要校验哪一个",
            )

        claims = self.extract_claims(text, source=source, variable=variable)
        if not claims:
            return VerificationReport(
                status=VerificationStatus.NOTHING_TO_VERIFY,
                source=source,
                detail=f"没有找到关于 {variable} 的明确解",
            )

        left, right = relation.split("=")
        expression = f"({left}) - ({right})"
        trace = ToolTrace()
        checks: list[ClaimCheck] = []
        inconclusive: list[tuple[int, str]] = []
        for claim in claims:
            for value in claim.values:
                if any(
                    check.value == value and check.claim.sentence_index == claim.sentence_index
                    for check in checks
                ):
                    continue  # the same root twice in one sentence (x = 2, x₁ = 2)
                result = self._tools.call(
                    "evaluate_expression",
                    {"expression": expression, "substitutions": {variable: value}},
                )
                trace = trace.with_result(result)
                if not result.ok:
                    inconclusive.append(
                        (claim.paragraph_index, f"{variable} = {value}（{result.error}）")
                    )
                    continue
                output = result.output_or_raise()
                residual = str(output["result"])
                satisfied = _is_zero(residual, str(output["decimal"]), self._zero_tolerance)
                checks.append(
                    ClaimCheck(claim=claim, value=value, residual=residual, satisfied=satisfied)
                )

        # The written solution's *conclusion* decides the verdict: the last paragraph
        # that states answers is the answer. An answer that narrates the mistake it just
        # fixed ("我一开始算成了 x = 3") would otherwise be read as claiming that value and
        # the loop would chase a conflict it invented. Earlier statements stay in
        # ``checks`` — the audit trail keeps them (§17) — and a whole paragraph counts, so
        # a multi-line root list is judged in full rather than by its last line alone.
        conclusion = max(claim.paragraph_index for claim in claims)
        decisive = [check for check in checks if check.claim.paragraph_index == conclusion]
        undecided = [message for index, message in inconclusive if index == conclusion]

        if any(not check.satisfied for check in decisive):
            status = VerificationStatus.CONFLICT
            detail = _conflict_detail(variable, decisive)
        elif undecided:
            status = VerificationStatus.UNVERIFIABLE
            detail = "无法校验：" + "；".join(undecided)
        else:
            status = VerificationStatus.VERIFIED
            detail = _verified_detail(variable, decisive)

        return VerificationReport(
            status=status,
            source=source,
            claims=claims,
            checks=tuple(checks),
            detail=detail,
            trace=trace,
        )

    def find_relation(self, question: str) -> str | None:
        """The equation written inside ``question``, as text, or ``None``.

        Two things disqualify a candidate:

        * a statement like ``x = 2`` — that is exactly the kind of claim we want to
          check, not something to check it against;
        * an identity like ``(x-2)(x-3) = x^2 - 5x + 6`` — it holds for every ``x``, so
          substituting a *wrong* root into it would look perfectly fine.

        Among several valid candidates the **leftmost** one wins: when a student writes
        the problem and then their own rearrangement, the problem is the reference.
        """
        for candidate in _math_candidates(question):
            if "=" not in candidate:
                continue
            try:
                left, right = parse_relation(candidate)
            except ExpressionRejected:
                continue
            if not (left.free_symbols or right.free_symbols):
                continue
            if _is_solution_statement(left, right) or _is_identity(left, right):
                continue
            return candidate
        return None

    def extract_claims(
        self,
        text: str,
        *,
        source: str = ANSWER_SOURCE,
        variable: str | None = None,
    ) -> tuple[Claim, ...]:
        """Pull ``variable = value`` claims out of ``text``.

        Repeated statements about the same variable are merged into one claim, so
        ``x = 2 或 x = 3`` reads as the solution set ``(2, 3)``. When ``variable`` is
        given, only claims about it are returned: a case discussion such as
        ``当 a = 0 时`` must never be mistaken for a solution.
        """
        normalized = _normalize_math_text(text)
        claims: list[Claim] = []
        sentence_index = 0
        for paragraph_index, paragraph in enumerate(_PARAGRAPH.split(normalized)):
            for sentence in _SENTENCE.split(paragraph):
                sentence_index += 1
                if not sentence.strip() or _is_negated(sentence):
                    continue
                claims.extend(
                    self._claims_in(
                        sentence,
                        source=source,
                        variable=variable,
                        sentence_index=sentence_index,
                        paragraph_index=paragraph_index,
                    )
                )
        return tuple(claims)

    # ----------------------------------------------------------------- private
    def _claims_in(
        self,
        sentence: str,
        *,
        source: str,
        variable: str | None,
        sentence_index: int,
        paragraph_index: int,
    ) -> list[Claim]:
        """Claims stated inside one sentence, one per variable, in reading order."""
        matches = list(_ASSIGNMENT.finditer(sentence))
        order: list[str] = []
        snippets: dict[str, str] = {}
        values_by_name: dict[str, list[str]] = {}
        for index, match in enumerate(matches):
            name = match.group(1)
            if name in ALLOWED_FUNCTIONS:
                continue
            if variable is not None and not _describes(name, variable):
                continue
            if _is_hypothesis(sentence, match.start(), match.end()):
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(sentence)
            values = self._values_in(sentence[match.end() : end][:_MAX_TAIL])
            if not values:
                continue
            if name not in values_by_name:
                order.append(name)
                values_by_name[name] = []
                snippets[name] = f"{name} = {', '.join(values)}"
            values_by_name[name].extend(
                value for value in values if value not in values_by_name[name]
            )
        return [
            Claim(
                kind="solution_set",
                variable=name,
                values=tuple(values_by_name[name]),
                snippet=snippets[name],
                source=source,
                sentence_index=sentence_index,
                paragraph_index=paragraph_index,
            )
            for name in order
        ]

    def _values_in(self, text: str) -> tuple[str, ...]:
        values: list[str] = []
        for raw_token in _VALUE_SEPARATORS.split(text):
            token = raw_token.strip().strip(_VALUE_TRIM).strip()
            if not token or len(token) > 40:
                continue
            for candidate in _expand_plus_minus(token):
                parsed_value = _parse_value(candidate)
                if parsed_value is not None and parsed_value not in values:
                    values.append(parsed_value)
        return tuple(values)


def _is_zero(residual: str, decimal: str, tolerance: float) -> bool:
    """True when the residual is exactly zero, or zero up to the model's rounding."""
    if residual in {"0", "0.0", "-0", "0e0"}:
        return True
    try:
        return abs(float(decimal)) <= tolerance
    except (TypeError, ValueError):  # pragma: no cover - the tool always sends a float
        return False


def _describes(name: str, variable: str) -> bool:
    """``x`` and ``x_1`` both describe the unknown ``x`` (answers are often written ``x₁``)."""
    return name == variable or name.startswith(f"{variable}_")


def _is_negated(sentence: str) -> bool:
    """True when the sentence rejects the values it mentions.

    A corrected answer that says "x = 3 不是解" must not be read as claiming ``x = 3``,
    or the loop would keep re-reporting a conflict it just resolved.
    """
    return any(marker in sentence for marker in _NEGATIONS)


def _is_hypothesis(sentence: str, start: int, end: int) -> bool:
    """True when a cue marks the assignment as a test, not an answer.

    An answer that checks its own candidates ("当 x = 2 时，左边 = -1",
    "试 x = 1，x = 0，x = 2") would otherwise be read as claiming those values, and the
    loop would report a conflict it invented. The cue is looked for both *before* the
    assignment (which covers every item of a list) and just after it.
    """
    if any(cue in sentence[:start] for cue in _CUES):
        return True
    window = sentence[max(0, start - _CUE_WINDOW) : end + _CUE_WINDOW]
    return any(cue in window for cue in _CUES)


def _expand_plus_minus(token: str) -> tuple[str, ...]:
    """``1 ± √2`` is two values, not one unparsable string."""
    if "±" not in token and "+-" not in token:
        return (token,)
    separator = "±" if "±" in token else "+-"
    prefix, _, suffix = token.partition(separator)
    if not suffix.strip():
        return ()
    return (f"{prefix}+({suffix})", f"{prefix}-({suffix})")


def _is_solution_statement(left: sympy.Expr, right: sympy.Expr) -> bool:
    """``x = 2`` (a claim) rather than ``x**2 - 5*x + 6 = 0`` (an equation)."""
    return (left.is_Symbol and not right.free_symbols) or (
        right.is_Symbol and not left.free_symbols
    )


def _is_identity(left: sympy.Expr, right: sympy.Expr) -> bool:
    """True when the two sides are equal for every value of the unknown."""
    try:
        return bool(sympy.simplify(left - right) == 0)
    except Exception as exc:  # noqa: BLE001 - a failed proof must not break a turn
        logger.warning("could not test a candidate relation for being an identity: %s", exc)
        return False


def _single_symbol(relation_text: str) -> str | None:
    """The one unknown of a reference equation, or ``None`` when it is ambiguous."""
    try:
        left, right = parse_relation(relation_text)
    except ExpressionRejected:  # pragma: no cover - callers pass a validated relation
        return None
    symbols = sorted((symbol.name for symbol in (left.free_symbols | right.free_symbols)), key=len)
    return symbols[0] if len(symbols) == 1 else None


def _conflict_detail(variable: str, checks: list[ClaimCheck]) -> str:
    return "；".join(
        f"{variable} = {check.value} 代入后方程两边的差为 {check.residual}"
        for check in checks
        if not check.satisfied
    )


def _verified_detail(variable: str, checks: list[ClaimCheck]) -> str:
    values = "、".join(f"{variable} = {check.value}" for check in checks)
    return f"{values} 代入原方程均成立"


def _math_candidates(text: str) -> list[str]:
    """The whole text first, then every maths-looking run in reading order.

    Reading order matters: a student's message can hold both the given problem and
    their own rearranged equation, and the *given* one comes first. Checking a claim
    against the student's own (wrong) rearrangement would always look consistent.
    """
    normalized = _normalize_math_text(text)
    ordered = [normalized.strip()]
    for match in _MATH_RUN.finditer(normalized):
        candidate = match.group(0).strip()
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return [candidate for candidate in ordered if candidate]


def _normalize_math_text(text: str) -> str:
    """Make ``$x^{2}$`` / ``x²`` / ``5x`` / ``\\frac{1}{3}`` readable by the sandbox.

    The sandbox stays a strict expression parser (it must not guess); the tolerance for
    how humans and models *write* maths lives here instead.
    """
    normalized = text
    for old, new in _UNICODE_MATH.items():
        normalized = normalized.replace(old, new)
    for old, new in _LATEX_NOISE:
        normalized = normalized.replace(old, new)
    previous = None
    while previous != normalized:  # \frac and \sqrt can nest
        previous = normalized
        normalized = _FRACTION.sub(r"((\1)/(\2))", normalized)
        normalized = _SQRT.sub(r"sqrt(\1)", normalized)
    normalized = _BRACED_POWER.sub(r"^(\1)", normalized)
    normalized = _BRACED_SUBSCRIPT.sub(r"_\1", normalized)
    normalized = _RADICAL_PAREN.sub("sqrt(", normalized)
    normalized = _RADICAL_OPERAND.sub(r"sqrt(\1)", normalized)
    normalized = _IMPLICIT_PRODUCT_LETTER.sub("*", normalized)
    return _IMPLICIT_PRODUCT_PAREN.sub("*", normalized)


def _parse_value(token: str) -> str | None:
    """A closed-form value for ``token``, or ``None`` when it is not a number."""
    candidate = token
    for _ in range(3):  # prose sometimes leaves an unmatched closing bracket
        try:
            parsed = parse_expression(candidate)
        except ExpressionRejected:
            if candidate.endswith((")", "]")):
                candidate = candidate[:-1].strip()
                continue
            return None
        return None if parsed.free_symbols else str(sympy.sstr(parsed))
    return None
