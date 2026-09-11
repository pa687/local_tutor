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
from tutor.tutor.engine import (
    EMPTY_ANSWER_WARNING,
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
    TutorStartEvent,
    TutorTokenEvent,
)

Factory = Callable[..., LlamaClient]
QUESTION = "解 x^2 - 5x + 6 = 0"


def build_engine(
    llama: LlamaClient, prompts: PromptLibrary, *, enable_thinking: bool = False
) -> TutorEngine:
    config = AppConfig(
        llm=LLMConfig(
            base_url="http://llama.test",
            model="qwen3.5-9b",
            enable_thinking=enable_thinking,
        ),
        memory=MemoryConfig(),
    )
    return TutorEngine(llama, prompts, config)


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
