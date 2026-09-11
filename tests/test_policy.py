"""Tests for the complete policy layer (ENGINEERING_PLAN.md §12).

§12 wants the five tutor rules and the grade constraints to be code, so these tests
are the contract: every rule has a positive and a negative case, and the grade bands
are pinned explicitly.
"""

from __future__ import annotations

import pytest

from tutor.llm.models import ChatMessage
from tutor.tutor.policy import (
    FORBIDDEN_METHODS,
    GradeBand,
    PolicyDecision,
    PolicyRule,
    TutorPolicy,
    find_out_of_scope_methods,
    grade_band,
    prior_assistant_turns,
    requests_full_solution,
    student_attempted,
)
from tutor.tutor.response import Subject, TutorMode

POLICY = TutorPolicy()
USER = "解 x^2 - 5x + 6 = 0"


def decide(
    *,
    mode: TutorMode = TutorMode.TUTOR,
    subject: Subject = Subject.MATH,
    grade: int | None = 9,
    message: str = USER,
    history: tuple[ChatMessage, ...] = (),
) -> PolicyDecision:
    return POLICY.decide(mode=mode, subject=subject, grade=grade, message=message, history=history)


def instructions(decision: PolicyDecision) -> str:
    return "\n".join(decision.instructions)


class TestGradeBands:
    @pytest.mark.parametrize(
        ("grade", "expected"),
        [
            (1, GradeBand.MIDDLE),
            (9, GradeBand.MIDDLE),
            (10, GradeBand.HIGH_LOWER),
            (11, GradeBand.HIGH_LOWER),
            (12, GradeBand.HIGH_UPPER),
            (None, GradeBand.UNKNOWN),
        ],
    )
    def test_grade_band_boundaries(self, grade: int | None, expected: GradeBand) -> None:
        assert grade_band(grade) is expected

    @pytest.mark.parametrize("grade", [7, 9])
    def test_middle_school_bans_calculus_and_matrices(self, grade: int) -> None:
        decision = decide(grade=grade)
        assert decision.grade_band is GradeBand.MIDDLE
        assert {"微积分", "导数", "积分", "极限", "矩阵", "复数"} <= set(decision.forbidden_methods)
        assert PolicyRule.FORBIDDEN_METHODS in decision.rules
        assert "不要使用以下超出该年级的方法" in instructions(decision)

    def test_high_school_lower_keeps_derivatives_but_not_analysis(self) -> None:
        decision = decide(grade=10)
        assert decision.grade_band is GradeBand.HIGH_LOWER
        assert "导数" not in decision.forbidden_methods
        assert {"极限", "洛必达", "泰勒"} <= set(decision.forbidden_methods)

    def test_high_school_upper_is_the_least_restrictive_band(self) -> None:
        decision = decide(grade=12)
        assert decision.grade_band is GradeBand.HIGH_UPPER
        assert "导数" not in decision.forbidden_methods
        assert "微积分" in decision.forbidden_methods

    def test_unknown_grade_warns_instead_of_guessing(self) -> None:
        decision = decide(grade=None)
        assert decision.grade_band is GradeBand.UNKNOWN
        assert decision.forbidden_methods == ()
        assert "学生年级未知，年级约束未生效" in decision.warnings
        assert PolicyRule.FORBIDDEN_METHODS not in decision.rules

    def test_grade_scope_instruction_carries_the_grade(self) -> None:
        assert "学生年级：9（初中）" in instructions(decide(grade=9))

    def test_non_math_subject_is_not_graded_on_math_methods(self) -> None:
        decision = decide(subject=Subject.ENGLISH, grade=9)
        assert decision.forbidden_methods == ()
        assert PolicyRule.FORBIDDEN_METHODS not in decision.rules

    def test_every_band_has_a_forbidden_list_entry(self) -> None:
        assert set(FORBIDDEN_METHODS) == set(GradeBand)


class TestRule1StudentAttempt:
    @pytest.mark.parametrize(
        "message",
        [
            "我算出来 x=2，3，对吗？",
            "我觉得应该先配方",
            "我的答案是 2",
            "x=2 你帮我看看",
            "对吗",
        ],
    )
    def test_detects_an_attempt(self, message: str) -> None:
        assert student_attempted(message) is True

    @pytest.mark.parametrize(
        "message",
        ["解 x^2 - 5x + 6 = 0", "为什么移项之后符号变了？", "这题怎么做", "什么是二次函数"],
    )
    def test_does_not_mistake_a_question_for_an_attempt(self, message: str) -> None:
        assert student_attempted(message) is False

    def test_attempt_switches_rule_1_on_and_rule_2_off(self) -> None:
        decision = decide(message="我算出来 x=2，3")
        assert PolicyRule.STUDENT_ATTEMPTED in decision.rules
        assert PolicyRule.HINT_FIRST not in decision.rules


class TestRule2HintFirst:
    def test_untried_question_only_gets_the_first_step(self) -> None:
        decision = decide()
        assert PolicyRule.HINT_FIRST in decision.rules
        assert "只给第一步提示" in instructions(decision)
        assert "不要给出完整解答" in instructions(decision)

    def test_check_mode_asks_for_the_work_instead(self) -> None:
        decision = decide(mode=TutorMode.CHECK)
        assert PolicyRule.CHECK_NEEDS_WORK in decision.rules
        assert PolicyRule.HINT_FIRST not in decision.rules


class TestRule3EscalateHints:
    def test_second_request_escalates_the_hint(self) -> None:
        history = (
            ChatMessage(role="user", content="这题怎么做"),
            ChatMessage(role="assistant", content="先试着把它因式分解。"),
        )
        decision = decide(history=history)
        assert PolicyRule.ESCALATE_HINTS in decision.rules
        assert "把提示提高一个层级" in instructions(decision)
        assert "不要给出完整答案" in instructions(decision)

    def test_first_request_does_not_escalate(self) -> None:
        assert PolicyRule.ESCALATE_HINTS not in decide().rules

    def test_escalation_needs_the_student_to_still_be_stuck(self) -> None:
        history = (ChatMessage(role="assistant", content="先试试因式分解。"),)
        decision = decide(message="我算出来 x=2，3", history=history)
        assert PolicyRule.ESCALATE_HINTS not in decision.rules

    @pytest.mark.parametrize(
        ("history", "expected"),
        [
            ((), 0),
            ((ChatMessage(role="user", content="hi"),), 0),
            ((ChatMessage(role="assistant", content="hi"),), 1),
        ],
    )
    def test_prior_assistant_turns_counts_answers_only(
        self, history: tuple[ChatMessage, ...], expected: int
    ) -> None:
        assert prior_assistant_turns(history) == expected


class TestRule4NoFullAnswer:
    def test_full_answer_is_withheld_by_default(self) -> None:
        decision = decide()
        assert PolicyRule.NO_FULL_ANSWER in decision.rules
        assert PolicyRule.FULL_ANSWER_REQUESTED not in decision.rules
        assert "不要直接展示完整答案" in instructions(decision)

    @pytest.mark.parametrize(
        "message",
        ["请给我完整解答", "直接告诉我答案", "别提示了，把过程写出来", "我要完整过程"],
    )
    def test_explicit_request_unlocks_it(self, message: str) -> None:
        decision = decide(message=message)
        assert PolicyRule.FULL_ANSWER_REQUESTED in decision.rules
        assert PolicyRule.NO_FULL_ANSWER not in decision.rules
        assert "学生明确要求完整解答" in instructions(decision)

    def test_a_refusal_is_not_a_request(self) -> None:
        assert requests_full_solution("不要告诉我答案，给我提示就行") is False

    def test_explain_mode_is_a_full_explanation_by_definition(self) -> None:
        decision = decide(mode=TutorMode.EXPLAIN)
        assert PolicyRule.EXPLAIN_COMPLETE in decision.rules
        assert PolicyRule.NO_FULL_ANSWER not in decision.rules
        assert PolicyRule.HINT_FIRST not in decision.rules


class TestCheckMode:
    def test_check_mode_never_regenerates_the_answer(self) -> None:
        decision = decide(mode=TutorMode.CHECK, message="我算出来 x=2，3，对吗？")
        assert PolicyRule.CHECK_ONLY in decision.rules
        assert "不要重新给出完整标准答案" in instructions(decision)
        assert "最早" in instructions(decision)

    def test_all_rule_ids_are_unique(self) -> None:
        assert len(set(PolicyRule)) == len(list(PolicyRule))


class TestOutOfScopeDetection:
    def test_flags_an_advanced_method_for_a_middle_schooler(self) -> None:
        decision = decide(grade=9)
        answer = "对函数求导，导数等于 2x - 5，所以极小值在 x=2.5。"
        assert find_out_of_scope_methods(answer, decision) == ("导数", "求导")

    def test_a_legitimate_caveat_is_not_a_violation(self) -> None:
        decision = decide(grade=9)
        answer = "这题不用导数，用因式分解就够：\n导数属于高中内容，超出你的年级。"
        assert find_out_of_scope_methods(answer, decision) == ()

    def test_no_scan_when_the_grade_is_unknown(self) -> None:
        decision = decide(grade=None)
        assert find_out_of_scope_methods("对函数求导", decision) == ()

    def test_high_school_lower_may_use_derivatives(self) -> None:
        decision = decide(grade=10)
        assert find_out_of_scope_methods("求导得到 2x - 5", decision) == ()

    def test_empty_answer_yields_no_findings(self) -> None:
        assert find_out_of_scope_methods("", decide(grade=9)) == ()


class TestContextBlock:
    def test_renders_a_numbered_instruction_block(self) -> None:
        block = decide().as_context_block()
        lines = block.splitlines()
        assert lines[0].startswith("本轮教学要求")
        assert lines[1].startswith("1. ")
        assert len(lines) == len(decide().instructions) + 1
