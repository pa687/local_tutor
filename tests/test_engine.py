"""Tests for ``TutorEngine`` (ENGINEERING_PLAN.md §6).

The DoD asks for an end-to-end structured ``TutorResponse`` out of every mode, so
these tests drive the real :class:`LlamaClient` against a fake llama-server that
serves both the classification call and the answer stream.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tutor.config import AppConfig, LLMConfig, MemoryConfig
from tutor.llm.client import LlamaClient, LlamaTimeoutError, LlamaUnavailableError
from tutor.llm.models import ChatMessage
from tutor.llm.prompts import PromptLibrary
from tutor.tools.calculator import EvaluateArgs
from tutor.tools.registry import ToolRegistry, ToolSpec, ToolUnsupported, build_default_registry
from tutor.tutor.engine import (
    EMPTY_ANSWER_WARNING,
    UNVERIFIABLE_WARNING,
    ConversationRef,
    StudentRef,
    TutorEngine,
)
from tutor.tutor.response import (
    Confidence,
    Subject,
    TutorDoneEvent,
    TutorEvent,
    TutorMode,
    TutorResponse,
    TutorRevisionEvent,
    TutorStartEvent,
    TutorTokenEvent,
    VerificationStatus,
)
from tutor.tutor.verifier import Verifier

Factory = Callable[..., LlamaClient]
QUESTION = "解 x^2 - 5*x + 6 = 0"


def engine_config(*, enable_thinking: bool = False) -> AppConfig:
    """The engine configuration used by these tests."""
    return AppConfig(
        llm=LLMConfig(
            base_url="http://llama.test",
            model="qwen3.5-9b",
            enable_thinking=enable_thinking,
        ),
        memory=MemoryConfig(),
    )


def failing_verifier() -> Verifier:
    """A verifier whose only tool always gives up — §7's "试过但无法验证"."""

    def give_up(_: EvaluateArgs) -> dict[str, object]:
        raise ToolUnsupported("sympy cannot evaluate this expression")

    registry: ToolRegistry = build_default_registry()
    registry.register(
        ToolSpec(
            name="evaluate_expression",
            description="always fails",
            arguments=EvaluateArgs,
            handler=give_up,
        ),
        replace=True,
    )
    return Verifier(registry)


def build_engine(
    llama: LlamaClient,
    prompts: PromptLibrary,
    *,
    enable_thinking: bool = False,
    verifier: Verifier | None = None,
) -> TutorEngine:
    return TutorEngine(
        llama, prompts, engine_config(enable_thinking=enable_thinking), verifier=verifier
    )


async def collect(
    engine: TutorEngine,
    message: str = QUESTION,
    *,
    mode: TutorMode = TutorMode.TUTOR,
    student: StudentRef | None = None,
    conversation: ConversationRef | None = None,
) -> list[TutorEvent]:
    return [
        event
        async for event in engine.stream(
            student if student is not None else StudentRef(id="student-001"),
            conversation if conversation is not None else ConversationRef(id="abc"),
            message,
            mode=mode,
        )
    ]


def done_of(events: list[TutorEvent]) -> TutorResponse:
    finals = [event.response for event in events if isinstance(event, TutorDoneEvent)]
    assert len(finals) == 1
    return finals[0]


def start_of(events: list[TutorEvent]) -> TutorResponse:
    starts = [event.response for event in events if isinstance(event, TutorStartEvent)]
    assert len(starts) == 1
    return starts[0]


class TestThreeModes:
    @pytest.mark.parametrize("mode", list(TutorMode))
    async def test_each_mode_returns_a_structured_response(
        self, mode: TutorMode, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(), prompts)
        events = await collect(engine, mode=mode)
        response = done_of(events)

        assert isinstance(response, TutorResponse)
        assert response.mode is mode
        assert response.answer == "Hello"
        assert response.subject is Subject.MATH
        assert response.estimated_grade == 9
        assert response.topic == "quadratic_equations"
        assert response.confidence is Confidence.HIGH
        assert response.tools_used == ()
        assert response.warnings == ()

    @pytest.mark.parametrize("mode", list(TutorMode))
    async def test_event_sequence_is_start_token_done(
        self, mode: TutorMode, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=("A", "B", "C")), prompts)
        events = await collect(engine, mode=mode)
        assert [type(event) for event in events] == [
            TutorStartEvent,
            TutorTokenEvent,
            TutorTokenEvent,
            TutorTokenEvent,
            TutorDoneEvent,
        ]
        assert (
            "".join(event.text for event in events if isinstance(event, TutorTokenEvent)) == "ABC"
        )


class TestMetadata:
    async def test_start_carries_metadata_without_the_answer(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(), prompts)
        start = start_of(await collect(engine))
        assert start.answer == ""
        assert start.subject is Subject.MATH
        assert start.confidence is Confidence.HIGH

    async def test_student_grade_overrides_the_estimate(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(), prompts)
        events = await collect(engine, student=StudentRef(id="s1", grade=11))
        assert done_of(events).estimated_grade == 11

    async def test_token_accounting_travels_with_the_done_event(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(prompt_tokens=42, completion_tokens=7), prompts)
        events = await collect(engine)
        done = next(event for event in events if isinstance(event, TutorDoneEvent))
        assert done.prompt_tokens == 42
        assert done.completion_tokens == 7


class TestDegradedClassification:
    async def test_unusable_classification_degrades_visibly(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(classification="这是一道二次方程题。"), prompts)
        response = done_of(await collect(engine))

        assert response.subject is Subject.UNKNOWN
        assert response.estimated_grade is None
        assert response.confidence is Confidence.LOW
        assert "题目结构化失败" in response.warnings[0]
        # The answer is still served, and the grade-unknown warning is included too.
        assert response.answer == "Hello"
        assert "学生年级未知" in " ".join(response.warnings)

    async def test_uncertain_classification_caps_confidence_at_medium(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        uncertain = json.dumps({"subject": "math", "grade": 9, "topic": None, "uncertain": True})
        engine = build_engine(tutor_llama(classification=uncertain), prompts)
        response = done_of(await collect(engine))
        assert response.confidence is Confidence.MEDIUM
        assert response.topic is None


class TestOutOfScopeAnswer:
    async def test_middle_school_answer_using_calculus_is_flagged(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(
            tutor_llama(tokens=("对函数求导，", "导数等于 2x - 5。")),
            prompts,
        )
        response = done_of(await collect(engine))

        assert response.answer == "对函数求导，导数等于 2x - 5。"
        assert response.confidence is Confidence.LOW
        assert any("超出年级范围" in warning for warning in response.warnings)

    async def test_empty_answer_is_reported(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=()), prompts)
        response = done_of(await collect(engine))
        assert response.answer == ""
        assert EMPTY_ANSWER_WARNING in response.warnings


class TestFailureHandling:
    async def test_backend_down_before_the_answer_raises(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        llama = tutor_llama(classify_error=httpx.ConnectError("refused"))
        engine = build_engine(llama, prompts)
        with pytest.raises(LlamaUnavailableError):
            await collect(engine)

    async def test_backend_dies_mid_answer_after_the_start_event(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        llama = tutor_llama(stream_error=httpx.ReadTimeout("too slow"))
        engine = build_engine(llama, prompts)
        seen: list[TutorEvent] = []
        with pytest.raises(LlamaTimeoutError):
            async for event in engine.stream(
                StudentRef(id="s1"), ConversationRef(id="c1"), QUESTION, mode=TutorMode.TUTOR
            ):
                seen.append(event)
        assert [type(event) for event in seen] == [TutorStartEvent]


class TestRespond:
    async def test_respond_returns_the_aggregated_answer(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=("x ", "= ", "2, 3")), prompts)
        response = await engine.respond(
            StudentRef(id="s1"), ConversationRef(id="c1"), QUESTION, mode=TutorMode.TUTOR
        )
        assert response.answer == "x = 2, 3"
        assert response.confidence is Confidence.HIGH


class TestVerificationLoop:
    """§7: the tools check the answer, a conflict forces a re-check, nothing is hidden."""

    QUADRATIC = "解方程 x^2 - 5*x + 6 = 0"
    GOOD = "所以 x = 2 或 x = 3"
    BAD = "所以 x = 4"

    async def test_a_correct_answer_is_verified_without_a_revision(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=(self.GOOD,)), prompts)
        events = await collect(engine, message=self.QUADRATIC)
        done = next(event for event in events if isinstance(event, TutorDoneEvent))

        assert isinstance(done.response, TutorResponse)
        assert [type(event) for event in events] == [
            TutorStartEvent,
            TutorTokenEvent,
            TutorDoneEvent,
        ]
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.VERIFIED
        assert done.verification.attempts == 0
        assert done.response.tools_used == ("evaluate_expression",)
        assert done.response.confidence is Confidence.HIGH
        assert done.response.warnings == ()

    async def test_a_conflicting_answer_triggers_one_correction(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(answers=[(self.BAD,), (self.GOOD,)]), prompts)
        events = await collect(engine, message=self.QUADRATIC)

        assert [type(event) for event in events] == [
            TutorStartEvent,
            TutorTokenEvent,
            TutorRevisionEvent,
            TutorTokenEvent,
            TutorDoneEvent,
        ]
        revision = next(event for event in events if isinstance(event, TutorRevisionEvent))
        assert revision.attempt == 1
        assert "x = 4" in revision.detail

        done = next(event for event in events if isinstance(event, TutorDoneEvent))
        assert done.response.answer == self.GOOD  # the corrected pass is the final answer
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.VERIFIED
        assert done.verification.attempts == 1
        assert done.response.warnings == ()
        assert done.response.confidence is Confidence.HIGH

    async def test_the_recheck_prompt_carries_the_conflict(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(
            tutor_llama(answers=[(self.BAD,), (self.GOOD,)], capture=payloads), prompts
        )
        await collect(engine, message=self.QUADRATIC)

        streams = [payload for payload in payloads if payload.get("stream")]
        assert len(streams) == 2
        recheck = streams[1]["messages"]
        assert [message["role"] for message in recheck] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert recheck[-2]["content"] == self.BAD  # the model sees its own answer
        assert prompts.get("verify.md") in recheck[-1]["content"]
        assert "x = 4" in recheck[-1]["content"]

    async def test_a_persistent_conflict_stops_after_the_configured_attempts(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        attempts_allowed = 2
        engine = build_engine(tutor_llama(tokens=(self.BAD,)), prompts)
        events = await collect(engine, message=self.QUADRATIC)
        done = next(event for event in events if isinstance(event, TutorDoneEvent))

        assert [type(event) for event in events].count(TutorRevisionEvent) == attempts_allowed
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.CONFLICT
        assert done.verification.attempts == attempts_allowed
        # §7: a conflict is never hidden, and it costs confidence.
        assert done.response.confidence is Confidence.LOW
        assert any("冲突" in warning for warning in done.response.warnings)

    async def test_the_conflict_warning_names_the_failing_value(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=(self.BAD,)), prompts)
        done = next(
            event
            for event in await collect(engine, message=self.QUADRATIC)
            if isinstance(event, TutorDoneEvent)
        )
        assert any("x = 4" in warning for warning in done.response.warnings)

    async def test_an_unverifiable_answer_says_so_and_downgrades_confidence(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(
            tutor_llama(tokens=(self.GOOD,)), prompts, verifier=failing_verifier()
        )
        done = next(
            event
            for event in await collect(engine, message=self.QUADRATIC)
            if isinstance(event, TutorDoneEvent)
        )
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.UNVERIFIABLE
        assert UNVERIFIABLE_WARNING in done.response.warnings
        assert done.response.confidence is Confidence.LOW

    async def test_an_answer_with_nothing_to_check_is_left_alone(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(tokens=("因为等式两边同时加减同一个数。",)), prompts)
        done = next(
            event
            for event in await collect(engine, message="为什么移项之后符号变了？")
            if isinstance(event, TutorDoneEvent)
        )
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.NOTHING_TO_VERIFY
        assert done.response.warnings == ()
        assert done.response.confidence is Confidence.HIGH
        assert done.response.tools_used == ()

    async def test_non_maths_subjects_are_not_sympy_verified(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        classification = json.dumps(
            {"subject": "english", "grade": 9, "topic": None, "uncertain": False}
        )
        engine = build_engine(tutor_llama(classification=classification), prompts)
        done = next(
            event
            for event in await collect(engine, message=self.QUADRATIC)
            if isinstance(event, TutorDoneEvent)
        )
        assert done.verification is not None
        assert done.verification.status is VerificationStatus.NOTHING_TO_VERIFY
        assert done.verification.detail is not None
        assert "非数学科目" in done.verification.detail

    async def test_token_accounting_covers_every_pass(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(
            tutor_llama(answers=[(self.BAD,), (self.GOOD,)], prompt_tokens=10, completion_tokens=5),
            prompts,
        )
        done = next(
            event
            for event in await collect(engine, message=self.QUADRATIC)
            if isinstance(event, TutorDoneEvent)
        )
        assert done.prompt_tokens == 20  # two passes, both counted for the log (§18)
        assert done.completion_tokens == 10


class TestStudentPreCheck:
    """§22 B/E: in check mode the student's own work goes through the tools first."""

    STUDENT_WORK = "题目是 x^2 - 5*x + 6 = 0，我的过程是 (x-2)(x-3)，所以 x = 2，5，对了吗？"

    async def test_the_tool_verdict_reaches_the_model(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        await collect(engine, message=self.STUDENT_WORK, mode=TutorMode.CHECK)

        system = next(payload for payload in payloads if payload.get("stream"))["messages"][0]
        assert "工具校验（SymPy）" in system["content"]
        assert "x = 5" in system["content"]

    async def test_the_verdict_is_reported_on_the_done_event(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        engine = build_engine(tutor_llama(), prompts)
        done = next(
            event
            for event in await collect(engine, message=self.STUDENT_WORK, mode=TutorMode.CHECK)
            if isinstance(event, TutorDoneEvent)
        )
        assert done.verification is not None
        assert done.verification.student_status is VerificationStatus.CONFLICT
        assert done.verification.student_detail is not None
        assert "x = 5" in done.verification.student_detail
        assert done.response.tools_used == ("evaluate_expression",)

    async def test_a_correct_student_answer_is_confirmed_by_the_tools(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        await collect(
            engine,
            message="题目是 x^2 - 5*x + 6 = 0，我的过程 (x-2)(x-3)，所以 x = 2，3",
            mode=TutorMode.CHECK,
        )
        system = next(payload for payload in payloads if payload.get("stream"))["messages"][0]
        assert "均成立" in system["content"]

    async def test_tutor_mode_does_not_pre_check_the_student(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        await collect(engine, message=self.STUDENT_WORK, mode=TutorMode.TUTOR)
        system = next(payload for payload in payloads if payload.get("stream"))["messages"][0]
        assert "工具校验" not in system["content"]


class TestPromptAssembly:
    async def test_policy_instructions_are_appended_to_the_system_layer(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        history = (
            ChatMessage(role="user", content="这题怎么做"),
            ChatMessage(role="assistant", content="先试试因式分解。"),
        )
        await collect(engine, conversation=ConversationRef(id="c1", history=history))

        streamed = next(payload for payload in payloads if payload.get("stream"))
        messages = streamed["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        system = messages[0]["content"]
        assert system.startswith(prompts.get("tutor_system.md"))
        assert messages[1]["content"] == "这题怎么做"
        assert messages[-1]["content"] == QUESTION
        assert "只给第一步提示" in system  # §12 rule 2
        assert "不要直接展示完整答案" in system  # §12 rule 4
        assert "把提示提高一个层级" in system  # §12 rule 3

    async def test_classification_runs_before_the_answer_stream(
        self, tutor_llama: Factory, prompts: PromptLibrary
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        await collect(engine)
        assert [payload.get("stream") for payload in payloads] == [False, True]

    async def test_answer_thinking_follows_the_configuration(
        self, prompts: PromptLibrary, tutor_llama: Factory
    ) -> None:
        """§1.5: inference parameters come from config, not from a hardcoded literal."""
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts, enable_thinking=True)
        await collect(engine)

        classification, answer = payloads
        assert classification["chat_template_kwargs"] == {"enable_thinking": False}
        assert answer["chat_template_kwargs"] == {"enable_thinking": True}

    async def test_thinking_is_off_by_default(
        self, prompts: PromptLibrary, tutor_llama: Factory
    ) -> None:
        payloads: list[dict[str, Any]] = []
        engine = build_engine(tutor_llama(capture=payloads), prompts)
        await collect(engine)
        assert payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
