"""Tests for claim extraction and tool verification (ENGINEERING_PLAN.md §7, §22).

Two properties matter most here:

* the verifier never invents a verdict — it either checks something with SymPy or says
  it could not, and adversarial text from the model must not become a tool call;
* the four statuses are distinguishable, because §7 attaches different consequences to
  them (a conflict must be visible, "could not verify" downgrades confidence, and a
  conceptual answer must not be punished for having nothing to verify).
"""

from __future__ import annotations

import json

import pytest

from tutor.tools.calculator import EvaluateArgs
from tutor.tools.registry import ToolRegistry, ToolSpec, ToolUnsupported, build_default_registry
from tutor.tutor.response import VerificationStatus
from tutor.tutor.verifier import (
    ANSWER_SOURCE,
    STUDENT_SOURCE,
    Verifier,
)

QUADRATIC = "解方程 x^2 - 5*x + 6 = 0"
ROOTS = "所以 x = 2 或 x = 3，代回原方程即可验证。"


@pytest.fixture
def verifier(tools: ToolRegistry) -> Verifier:
    return Verifier(tools)


def failing_registry() -> ToolRegistry:
    """A registry whose ``evaluate_expression`` always gives up (the §7 '无法验证' case)."""

    def give_up(_: EvaluateArgs) -> dict[str, object]:
        raise ToolUnsupported("sympy cannot evaluate this expression")

    registry = build_default_registry()
    registry.register(
        ToolSpec(
            name="evaluate_expression",
            description="always fails",
            arguments=EvaluateArgs,
            handler=give_up,
        ),
        replace=True,
    )
    return registry


class TestFindRelation:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("解方程 x^2 - 5*x + 6 = 0", "x^2 - 5*x + 6 = 0"),
            ("x^2 - 5*x + 6 = 0", "x^2 - 5*x + 6 = 0"),
            ("解方程 x^2 - 5*x + 6 = 0。", "x^2 - 5*x + 6 = 0"),
            ("2*x + 1 = 7", "2*x + 1 = 7"),
            ("\\frac{x}{2} = 3", "((x)/(2)) = 3"),
            ("解方程 5x + 6 = 0", "5*x + 6 = 0"),
            ("$x^{2} - 5x + 6 = 0$", "x^(2) - 5*x + 6 = 0"),
            ("解方程 x² - 5x + 6 = 0", "x^2 - 5*x + 6 = 0"),
            ("解方程 x²−5x+6=0", "x^2-5*x+6=0"),
            ("2×x + 1 = 7", "2*x + 1 = 7"),
        ],
    )
    def test_finds_the_equation_inside_a_sentence(
        self, verifier: Verifier, question: str, expected: str
    ) -> None:
        assert verifier.find_relation(question) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "我算出来 x = 2，3，对吗？",
            "x = 2",
            "2 + 2 = 4",
            "为什么移项之后符号变了？",
            # An identity is true for every x, so a wrong root would look verified.
            "(x-2)(x-3) = x^2 - 5x + 6",
            "2*(x + 1) = 2*x + 2",
            "",
        ],
    )
    def test_refuses_to_treat_a_solution_statement_as_the_question(
        self, verifier: Verifier, text: str
    ) -> None:
        """``x = 2`` is what we check, not what we check against (§22 B)."""
        assert verifier.find_relation(text) is None


class TestExtractClaims:
    @pytest.mark.parametrize(
        ("text", "values"),
        [
            ("所以 x = 2 或 x = 3", ("2", "3")),
            ("x = 2, 3", ("2", "3")),
            ("x = 2、3", ("2", "3")),
            ("x = 2", ("2",)),
            ("x=2", ("2",)),
            ("x = 2 和 x = 3", ("2", "3")),
            ("x = 2 或 x = 3 或 x = 5", ("2", "3", "5")),
            ("x = 1/3", ("1/3",)),
            ("$x = \\frac{1}{2}$", ("1/2",)),
            ("x = -3", ("-3",)),
            ("x = 2 或 x = -3 或 x = 5", ("2", "-3", "5")),
            ("x = 2 或 x = 3，代回原方程即可验证。", ("2", "3")),
            ("x = −3", ("-3",)),
            ("x = 3×2", ("6",)),
        ],
    )
    def test_reads_solution_statements(
        self, verifier: Verifier, text: str, values: tuple[str, ...]
    ) -> None:
        claims = verifier.extract_claims(text, variable="x")
        assert len(claims) == 1
        assert claims[0].values == values
        assert claims[0].source == ANSWER_SOURCE

    def test_subscripted_names_describe_the_same_unknown(self, verifier: Verifier) -> None:
        """``x₁`` / ``x₂`` in the answer are both about the ``x`` of the question."""
        claims = verifier.extract_claims("所以 x₁ = 2，x₂ = 3", variable="x")
        assert [claim.values for claim in claims] == [("2",), ("3",)]

        report = verifier.verify(question=QUADRATIC, text="所以 x₁ = 2，x₂ = 3")
        assert report.status is VerificationStatus.VERIFIED
        assert [check.value for check in report.checks] == ["2", "3"]

    def test_plus_minus_is_two_values(self, verifier: Verifier) -> None:
        claims = verifier.extract_claims("所以 x = 1 ± √2", variable="x")
        assert claims[0].values == ("1 + sqrt(2)", "1 - sqrt(2)")

    def test_radicals_are_understood(self, verifier: Verifier) -> None:
        assert verifier.extract_claims("x = √2", variable="x")[0].values == ("sqrt(2)",)
        assert verifier.extract_claims("x = √(2)", variable="x")[0].values == ("sqrt(2)",)

    def test_quoted_wrong_values_are_not_claims(self, verifier: Verifier) -> None:
        """A corrected answer mentions the value it just rejected."""
        text = "事实是：x = 3 和 x = -2 绝对不是这个方程的解。\n正确的解是 x = 3/2 或 x = -2/3。"
        claims = verifier.extract_claims(text, variable="x")
        assert [claim.values for claim in claims] == [("3/2", "-2/3")]
        assert verifier.verify(question="6*x^2 - 5*x - 6 = 0", text=text).status is (
            VerificationStatus.VERIFIED
        )

    def test_the_same_root_stated_twice_is_checked_once(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text="所以 x = 2，x = 3，即 x₁ = 2，x₂ = 3")
        assert report.status is VerificationStatus.VERIFIED
        assert [check.value for check in report.checks] == ["2", "3"]

    def test_does_not_invent_values(self, verifier: Verifier) -> None:
        assert verifier.extract_claims("这是一道一元二次方程，用因式分解即可", variable="x") == ()

    def test_ignores_claims_about_other_letters(self, verifier: Verifier) -> None:
        """A case discussion (``当 a = 0 时``) is not a solution claim."""
        claims = verifier.extract_claims("所以 x = 2，其中 a = 0", variable="x")
        assert [claim.variable for claim in claims] == ["x"]

    def test_case_discussions_are_not_claims(self, verifier: Verifier) -> None:
        assert verifier.extract_claims("当 a = 0 时，x = 2", variable="x") == ()
        assert verifier.extract_claims("当 x = 2 时，左边 = -1", variable="x") == ()
        assert verifier.extract_claims("x = 2 代入后差值为 -1", variable="x") == ()

    def test_a_conclusion_survives_its_own_explanation(self, verifier: Verifier) -> None:
        claims = verifier.extract_claims("所以 x = 2 或 x = 3", variable="x")
        assert claims[0].values == ("2", "3")

    def test_ignores_values_that_are_not_closed_form(self, verifier: Verifier) -> None:
        assert verifier.extract_claims("x = 2*y", variable="x") == ()

    def test_ignores_mathematical_functions_used_as_names(self, verifier: Verifier) -> None:
        assert verifier.extract_claims("sin = 1", variable="sin") == ()

    @pytest.mark.parametrize(
        "text",
        [
            "x = __import__('os').system('id')",
            "x = 9^9^9",
            "x = open('/etc/passwd').read()",
            "x = lambda: 1",
        ],
    )
    def test_adversarial_values_are_dropped_not_executed(
        self, verifier: Verifier, text: str
    ) -> None:
        assert verifier.extract_claims(text, variable="x") == ()

    def test_source_is_recorded(self, verifier: Verifier) -> None:
        claims = verifier.extract_claims("x = 2", variable="x", source=STUDENT_SOURCE)
        assert claims[0].source == STUDENT_SOURCE


class TestVerified:
    def test_a_correct_answer_is_verified_by_sympy(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text=ROOTS)
        assert report.status is VerificationStatus.VERIFIED
        assert report.claims[0].variable == "x"
        assert [check.value for check in report.checks] == ["2", "3"]
        assert all(check.satisfied for check in report.checks)
        assert report.conflicts == ()
        assert report.tools_used == ("evaluate_expression",)
        assert report.detail is not None and "均成立" in report.detail
        assert "x = 2" in report.detail and "x = 3" in report.detail

    def test_instruction_carries_the_verdict(self, verifier: Verifier) -> None:
        instruction = verifier.verify(question=QUADRATIC, text=ROOTS).instruction()
        assert instruction is not None
        assert instruction.startswith("工具校验（SymPy）：")
        assert "x = 2" in instruction

    def test_rounding_is_tolerated(self, verifier: Verifier) -> None:
        report = verifier.verify(question="3*x - 1 = 0", text="所以 x = 0.3333333")
        assert report.status is VerificationStatus.VERIFIED

    def test_the_question_may_be_written_with_unicode_maths(self, verifier: Verifier) -> None:
        report = verifier.verify(question="解方程 x² − 5x + 6 = 0", text="所以 x = 2 或 x = 3")
        assert report.status is VerificationStatus.VERIFIED

    def test_irrational_roots_are_verified_too(self, verifier: Verifier) -> None:
        report = verifier.verify(question="x^2 - 2*x - 1 = 0", text="所以 x = 1 ± √2")
        assert report.status is VerificationStatus.VERIFIED
        assert [check.value for check in report.checks] == ["1 + sqrt(2)", "1 - sqrt(2)"]

    def test_latex_roots_are_understood(self, verifier: Verifier) -> None:
        report = verifier.verify(
            question="x^2 - 2*x - 1 = 0", text="所以 $x = 1 + \\sqrt{2}$ 或 $x = 1 - \\sqrt{2}$"
        )
        assert report.status is VerificationStatus.VERIFIED, report.detail

    def test_every_tool_call_is_auditable(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text=ROOTS)
        events = report.as_event()["tools"]
        assert len(events) == 2
        assert events[0]["tool"] == "evaluate_expression"
        assert events[0]["ok"] is True
        assert json.dumps(report.as_event(), ensure_ascii=False)


class TestConflict:
    def test_a_wrong_root_is_a_conflict_with_residuals(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text="所以 x = 2 或 x = 4")
        assert report.status is VerificationStatus.CONFLICT
        assert [check.value for check in report.conflicts] == ["4"]
        assert report.detail is not None
        assert "x = 4" in report.detail
        assert "差为" in report.detail

    def test_a_completely_wrong_answer_is_a_conflict(self, verifier: Verifier) -> None:
        report = verifier.verify(question="2*x + 1 = 7", text="x = 4")
        assert report.status is VerificationStatus.CONFLICT

    def test_the_conclusion_decides_the_verdict(self, verifier: Verifier) -> None:
        """A model that narrates its own mistake must not be read as claiming it."""
        text = "我一开始把解对应成了 x = 3 和 x = -2。\n\n正确的解是 x = 3/2 或 x = -2/3。"
        report = verifier.verify(question="6*x^2 - 5*x - 6 = 0", text=text)
        assert report.status is VerificationStatus.VERIFIED
        assert report.detail is not None and "x = 3/2" in report.detail
        # The rejected statement is still audited, it just is not decisive (§17).
        assert {check.value for check in report.checks} >= {"3", "-2", "3/2", "-2/3"}

    def test_the_whole_conclusion_paragraph_counts(self, verifier: Verifier) -> None:
        """A multi-line root list must be judged in full, not by its last line."""
        text = "解题过程略。\n\n最终结论：\nx₁ = 2\nx₂ = 7\n"
        report = verifier.verify(question=QUADRATIC, text=text)
        assert report.status is VerificationStatus.CONFLICT
        assert [check.value for check in report.conflicts] == ["7"]

    def test_a_wrong_conclusion_is_still_a_conflict(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text="先算得 x = 2。\n最后答案：x = 4。")
        assert report.status is VerificationStatus.CONFLICT
        assert [check.value for check in report.conflicts] == ["4"]

    def test_the_students_own_wrong_answer_is_caught(self, verifier: Verifier) -> None:
        """§22 E: the student's claim is checkable without the model's help."""
        report = verifier.verify(
            question=QUADRATIC,
            text="我的过程：x^2 - 5x + 6 = (x-2)(x-3)，所以 x = 2，5，对了吗？",
            source=STUDENT_SOURCE,
        )
        assert report.status is VerificationStatus.CONFLICT
        assert [check.value for check in report.conflicts] == ["5"]

    def test_the_given_problem_wins_over_the_students_rearrangement(
        self, verifier: Verifier
    ) -> None:
        """§22 E for real: ``(x-2)(x+3) = 0`` is self-consistent, but the problem is not."""
        message = "题目是 x^2 - 5*x + 6 = 0，我的过程：(x - 2)(x + 3) = 0，所以 x = 2 或 x = -3"
        report = verifier.verify(question=message, text=message, source=STUDENT_SOURCE)
        assert report.status is VerificationStatus.CONFLICT
        assert [check.value for check in report.conflicts] == ["-3"]
        assert report.detail is not None and "-3" in report.detail

    def test_a_correct_student_answer_is_verified(self, verifier: Verifier) -> None:
        report = verifier.verify(
            question=QUADRATIC,
            text="我的过程：所以 x = 2，3，对了吗？",
            source=STUDENT_SOURCE,
        )
        assert report.status is VerificationStatus.VERIFIED
        assert report.source == STUDENT_SOURCE

    def test_instruction_for_a_conflict_is_factual(self, verifier: Verifier) -> None:
        instruction = verifier.verify(question=QUADRATIC, text="x = 4").instruction()
        assert instruction is not None
        assert "x = 4" in instruction
        assert "差为" in instruction


class TestUnverifiable:
    def test_tool_failure_becomes_unverifiable_not_a_conflict(self) -> None:
        verifier = Verifier(failing_registry())
        report = verifier.verify(question=QUADRATIC, text=ROOTS)
        assert report.status is VerificationStatus.UNVERIFIABLE
        assert report.claims  # the claim was found, the tool just could not judge it
        assert report.detail is not None and "无法校验" in report.detail
        assert report.trace.entries

    def test_no_instruction_is_given_when_nothing_could_be_checked(self) -> None:
        verifier = Verifier(failing_registry())
        assert verifier.verify(question=QUADRATIC, text=ROOTS).instruction() is None


class TestNothingToVerify:
    def test_a_conceptual_answer_is_not_a_failure(self, verifier: Verifier) -> None:
        report = verifier.verify(
            question="为什么移项之后符号变了？", text="因为等式两边同时加减同一个数，等式仍成立。"
        )
        assert report.status is VerificationStatus.NOTHING_TO_VERIFY
        assert report.instruction() is None
        assert report.tools_used == ()

    def test_an_answer_without_a_solution_is_not_checked(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text="我们先用十字相乘法观察常数项。")
        assert report.status is VerificationStatus.NOTHING_TO_VERIFY
        assert report.detail is not None and "没有找到关于 x 的明确解" in report.detail

    def test_two_unknowns_are_not_guessed_at(self, verifier: Verifier) -> None:
        report = verifier.verify(question="x + y = 3", text="x = 1")
        assert report.status is VerificationStatus.NOTHING_TO_VERIFY
        assert report.detail is not None and "多个未知量" in report.detail

    def test_an_identity_is_not_an_equation_to_solve(self, verifier: Verifier) -> None:
        """§22 E depends on this: substituting into an identity always looks fine."""
        report = verifier.verify(question="证明 (x-2)(x-3) = x^2 - 5x + 6", text="所以 x = 5 是解")
        assert report.status is VerificationStatus.NOTHING_TO_VERIFY

    def test_no_tool_is_called_when_there_is_nothing_to_check(self, verifier: Verifier) -> None:
        report = verifier.verify(question="为什么移项之后符号变了？", text="因为天平要保持平衡。")
        assert report.trace.entries == ()


class TestNoModelJudge:
    """§7 says SymPy verifies; a second model call would be free to agree with the first."""

    def test_verifier_never_touches_a_language_model(self) -> None:
        import inspect

        from tutor.tutor import verifier as verifier_module

        source = inspect.getsource(verifier_module)
        assert "LlamaClient" not in source
        assert "stream_chat" not in source
        assert "complete(" not in source

    def test_verifier_only_calls_the_registry(self, verifier: Verifier) -> None:
        report = verifier.verify(question=QUADRATIC, text=ROOTS)
        assert {entry.tool for entry in report.trace.entries} == {"evaluate_expression"}
