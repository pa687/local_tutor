"""Tutor Policy layer — the complete Phase 2 rule set (ENGINEERING_PLAN.md §12).

§12 requires the five tutor rules and the grade constraints to live *here*, in code,
rather than inside a system prompt: they must stay testable, loggable and changeable
without editing prompt files. :class:`TutorPolicy` turns
``(mode, subject, grade, message, history)`` into a :class:`PolicyDecision` whose
instructions are injected as their own context layer, and
:func:`find_out_of_scope_methods` checks the finished answer against the very same
constraints (a Phase-2 preview of the Phase-4 verification loop).

The module never calls the model: a decision is a pure function of its inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tutor.llm.models import ChatMessage
from tutor.tutor.response import Subject, TutorMode


class GradeBand(StrEnum):
    """Coarse grade buckets used to scope the allowed methods (§12)."""

    MIDDLE = "middle"  # 1–9, 初中
    HIGH_LOWER = "high_lower"  # 10–11, 高中低年级
    HIGH_UPPER = "high_upper"  # 12
    UNKNOWN = "unknown"


#: Methods that must not be used silently, per band (§12). 高中低年级 keeps 导数 /
#: 复数 / 向量 (they are on the syllabus) but not the analysis toolkit.
FORBIDDEN_METHODS: Final[Mapping[GradeBand, tuple[str, ...]]] = {
    GradeBand.MIDDLE: (
        "微积分",
        "导数",
        "求导",
        "积分",
        "极限",
        "洛必达",
        "泰勒",
        "矩阵",
        "行列式",
        "复数",
        "虚数",
        "叉积",
    ),
    GradeBand.HIGH_LOWER: ("微积分", "极限", "洛必达", "泰勒", "矩阵", "行列式"),
    GradeBand.HIGH_UPPER: ("微积分", "洛必达", "泰勒", "矩阵", "行列式"),
    GradeBand.UNKNOWN: (),
}

#: Only math answers are scanned for out-of-scope *math* methods.
_SCOPED_SUBJECTS: Final = (Subject.MATH,)

_BAND_LABELS: Final[Mapping[GradeBand, str]] = {
    GradeBand.MIDDLE: "初中",
    GradeBand.HIGH_LOWER: "高中低年级",
    GradeBand.HIGH_UPPER: "高中高年级",
    GradeBand.UNKNOWN: "未知年级",
}

#: "The student already tried" signals (§12 rule 1). Deliberately language-based:
#: pasting an equation is not an attempt, "我算出来…" is.
_ATTEMPT_MARKERS: Final = (
    "我算",
    "我求",
    "我解",
    "我得",
    "我写",
    "我设",
    "我认",
    "我觉得",
    "我认为",
    "我的答案",
    "我的过程",
    "我的解法",
    "我这样做",
    "我是这样",
    "我用了",
    "我代入",
    "对吗",
    "对不对",
    "对不",
    "是不是这样",
    "你看我",
    "你帮我看看",
)

#: "Give me the whole solution" signals (§12 rule 4).
_FULL_SOLUTION_MARKERS: Final = (
    "完整答案",
    "完整解答",
    "完整过程",
    "完整的答案",
    "完整的解答",
    "标准答案",
    "全部答案",
    "直接给答案",
    "直接告诉我答案",
    "告诉我答案",
    "给我答案",
    "不要提示",
    "别提示",
    "不用提示",
    "把过程写出来",
    "直接算出来",
    "完整讲解",
)

#: If any of these appear, the student is explicitly *refusing* the full answer.
_FULL_SOLUTION_NEGATIONS: Final = (
    "不要告诉我答案",
    "不要给我答案",
    "不要直接给答案",
    "不要直接告诉我",
    "别告诉我答案",
    "别给答案",
    "不用给答案",
)

#: Contexts in which a banned term is mentioned as a warning, not used as a method.
_METHOD_NEGATIONS: Final = (
    "不用",
    "不需要",
    "不涉及",
    "未使用",
    "没有用到",
    "无需",
    "超出",
    "超纲",
)


class PolicyRule(StrEnum):
    """Stable identifiers for the rules a decision fired (test + log friendly)."""

    STUDENT_ATTEMPTED = "tutor.student_attempted"
    HINT_FIRST = "tutor.hint_first"
    ESCALATE_HINTS = "tutor.escalate_hints"
    NO_FULL_ANSWER = "tutor.no_full_answer"
    FULL_ANSWER_REQUESTED = "tutor.full_answer_requested"
    GRADE_SCOPE = "tutor.grade_scope"
    FORBIDDEN_METHODS = "tutor.forbidden_methods"
    EXPLAIN_COMPLETE = "explain.full_explanation"
    CHECK_ONLY = "check.check_only"
    CHECK_NEEDS_WORK = "check.needs_student_work"


@dataclass(frozen=True)
class PolicyDecision:
    """What the policy layer wants this turn to look like."""

    mode: TutorMode
    grade_band: GradeBand
    rules: tuple[PolicyRule, ...] = ()
    instructions: tuple[str, ...] = ()
    forbidden_methods: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_context_block(self) -> str:
        """Render the instructions as one context layer (§10 layer 1)."""
        lines = ["本轮教学要求（由策略层生成，优先级高于通用作答要求）："]
        lines.extend(f"{index}. {text}" for index, text in enumerate(self.instructions, start=1))
        return "\n".join(lines)


def grade_band(grade: int | None) -> GradeBand:
    """Map a numeric grade onto a :class:`GradeBand`."""
    if grade is None:
        return GradeBand.UNKNOWN
    if grade <= 9:
        return GradeBand.MIDDLE
    if grade <= 11:
        return GradeBand.HIGH_LOWER
    return GradeBand.HIGH_UPPER


def student_attempted(message: str) -> bool:
    """True when the message shows work or an answer of the student's own (§12.1)."""
    return any(marker in message for marker in _ATTEMPT_MARKERS)


def requests_full_solution(message: str) -> bool:
    """True when the student explicitly asks for the complete answer (§12.4)."""
    if any(negation in message for negation in _FULL_SOLUTION_NEGATIONS):
        return False
    return any(marker in message for marker in _FULL_SOLUTION_MARKERS)


def prior_assistant_turns(history: tuple[ChatMessage, ...] | list[ChatMessage]) -> int:
    """How many tutor answers the conversation already contains (§12.3)."""
    return sum(1 for message in history if message.role == "assistant")


class TutorPolicy:
    """Turns (mode, subject, grade, message, history) into instructions."""

    def decide(
        self,
        *,
        mode: TutorMode,
        subject: Subject,
        grade: int | None,
        message: str,
        history: tuple[ChatMessage, ...] | list[ChatMessage] = (),
    ) -> PolicyDecision:
        """Build the instructions for one turn.

        All five §12 tutor rules are applied in ``tutor`` mode; ``explain`` and
        ``check`` reuse the same grade constraints but their own framing.
        """
        band = grade_band(grade)
        scoped = subject in _SCOPED_SUBJECTS
        forbidden = FORBIDDEN_METHODS[band] if scoped else ()
        attempted = student_attempted(message)
        rules: list[PolicyRule] = []
        instructions: list[str] = []
        warnings: list[str] = []

        if mode is TutorMode.TUTOR:
            if attempted:
                rules.append(PolicyRule.STUDENT_ATTEMPTED)
                instructions.append(
                    "学生已经给出自己的思路或答案：先针对他这一步回应，指出对错和原因，"
                    "不要从头重讲一遍。"
                )
            else:
                rules.append(PolicyRule.HINT_FIRST)
                instructions.append(
                    "学生还没有给出自己的尝试：只给第一步提示，引导他自己往下走，不要给出完整解答。"
                )
                if prior_assistant_turns(history) >= 1:
                    rules.append(PolicyRule.ESCALATE_HINTS)
                    instructions.append(
                        "这已经是学生再次请求且仍未提供自己的尝试：把提示提高一个层级，"
                        "更具体一些，但仍然不要给出完整答案。"
                    )
            if requests_full_solution(message):
                rules.append(PolicyRule.FULL_ANSWER_REQUESTED)
                instructions.append(
                    "学生明确要求完整解答：可以给出完整过程，但每一步都要说明依据。"
                )
            else:
                rules.append(PolicyRule.NO_FULL_ANSWER)
                instructions.append("除非学生明确要求完整解答，否则不要直接展示完整答案。")
        elif mode is TutorMode.EXPLAIN:
            rules.append(PolicyRule.EXPLAIN_COMPLETE)
            instructions.append(
                "这是完整讲解模式：给出完整的推导过程、涉及的知识点和最终结论，每步说明依据。"
            )
            if attempted:
                rules.append(PolicyRule.STUDENT_ATTEMPTED)
                instructions.append(
                    "学生已经给出自己的答案：在讲解中顺带指出他的答案是否正确，错在哪一步。"
                )
        else:  # TutorMode.CHECK
            rules.append(PolicyRule.CHECK_ONLY)
            instructions.append("这是检查模式：不要重新给出完整标准答案，只检查学生的过程。")
            if attempted:
                rules.append(PolicyRule.STUDENT_ATTEMPTED)
                instructions.append(
                    "学生已给出过程：指出**最早**出现错误的那一步，说明错在哪里、为什么错，"
                    "而不是重新生成标准答案。"
                )
            else:
                rules.append(PolicyRule.CHECK_NEEDS_WORK)
                instructions.append(
                    "学生没有提供自己的解题过程：先请他给出过程，不要替他猜一个过程再检查。"
                )

        rules.append(PolicyRule.GRADE_SCOPE)
        if band is GradeBand.UNKNOWN:
            warnings.append("学生年级未知，年级约束未生效")
            instructions.append(
                "学生年级未知：按最保守的方式处理，遇到可能超纲的方法先说明它属于哪个年级。"
            )
        else:
            instructions.append(
                f"学生年级：{grade}（{_BAND_LABELS[band]}）。只使用这个范围内的知识和技巧。"
            )
        if forbidden:
            rules.append(PolicyRule.FORBIDDEN_METHODS)
            instructions.append(
                "不要使用以下超出该年级的方法：" + "、".join(forbidden) + "。"
                "如果确实必须使用，先说明它超出当前年级并解释它是什么。"
            )

        return PolicyDecision(
            mode=mode,
            grade_band=band,
            rules=tuple(rules),
            instructions=tuple(instructions),
            forbidden_methods=forbidden,
            warnings=tuple(warnings),
        )


def find_out_of_scope_methods(answer: str, decision: PolicyDecision) -> tuple[str, ...]:
    """Banned methods that the finished answer appears to use (§12 rule 5).

    A term is ignored when it appears on a line that also says the method is *not*
    needed ("不用", "超出", …), so a legitimate caveat is not reported as a violation.
    """
    if not decision.forbidden_methods:
        return ()
    found: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or any(marker in line for marker in _METHOD_NEGATIONS):
            continue
        for term in decision.forbidden_methods:
            if term in line and term not in found:
                found.append(term)
    return tuple(found)
