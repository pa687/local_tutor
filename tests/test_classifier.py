"""Tests for the classifier (ENGINEERING_PLAN.md §1, §6).

The classifier fills ``subject`` / ``estimated_grade`` / ``topic``. It must be
deterministic (temperature 0), prompt-file driven, and it must degrade *visibly* on
garbage while never swallowing a dead backend.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tutor.llm.client import LlamaClient, LlamaUnavailableError
from tutor.llm.prompts import PromptLibrary
from tutor.tutor.classifier import (
    CLASSIFY_MAX_TOKENS,
    CLASSIFY_PROMPT,
    FALLBACK_WARNING,
    TutorClassifier,
)
from tutor.tutor.response import Subject

Factory = Callable[..., LlamaClient]
QUESTION = "解 x^2 - 5x + 6 = 0"


def clean_reply(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "subject": "math",
        "grade": 9,
        "topic": "quadratic_equations",
        "uncertain": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def parse_only(offline_llama: LlamaClient, prompts: PromptLibrary) -> TutorClassifier:
    """A classifier used only for ``parse``; its client is never called."""
    return TutorClassifier(offline_llama, prompts)


class TestParse:
    def test_parses_a_clean_reply(self, parse_only: TutorClassifier) -> None:
        result = parse_only.parse(clean_reply())
        assert result.subject is Subject.MATH
        assert result.grade == 9
        assert result.topic == "quadratic_equations"
        assert result.uncertain is False
        assert result.warnings == ()
        assert result.is_fallback is False

    def test_tolerates_markdown_fences_and_prose(self, parse_only: TutorClassifier) -> None:
        raw = f"好的，这里是结果：\n```json\n{clean_reply()}\n```\n希望有帮助。"
        result = parse_only.parse(raw)
        assert result.subject is Subject.MATH
        assert result.grade == 9

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [
            ("Maths", Subject.MATH),
            ("数学", Subject.MATH),
            ("PHYSICS", Subject.PHYSICS),
            ("none", Subject.UNKNOWN),
        ],
    )
    def test_normalises_subject_spellings(
        self, parse_only: TutorClassifier, reported: str, expected: Subject
    ) -> None:
        assert parse_only.parse(clean_reply(subject=reported)).subject is expected

    def test_unusable_reply_degrades_visibly(
        self, parse_only: TutorClassifier, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="tutor.tutor.classifier")
        result = parse_only.parse("我觉得这是一道二次方程题。")
        assert result.is_fallback is True
        assert result.subject is Subject.UNKNOWN
        assert result.grade is None
        assert result.topic is None
        assert result.uncertain is True
        assert result.warnings == (FALLBACK_WARNING,)
        assert FALLBACK_WARNING in caplog.text
        assert QUESTION not in caplog.text  # §18: no student content in logs

    def test_unknown_subject_keeps_the_rest(self, parse_only: TutorClassifier) -> None:
        result = parse_only.parse(clean_reply(subject="astrophysics"))
        assert result.subject is Subject.UNKNOWN
        assert result.grade == 9
        assert "无法识别" in result.warnings[0]

    @pytest.mark.parametrize("grade", [0, 13, 99])
    def test_out_of_range_grade_is_dropped_with_a_warning(
        self, parse_only: TutorClassifier, grade: int
    ) -> None:
        result = parse_only.parse(clean_reply(grade=grade))
        assert result.grade is None
        assert result.subject is Subject.MATH
        assert "不在 1–12 范围内" in result.warnings[0]

    def test_missing_topic_becomes_none(self, parse_only: TutorClassifier) -> None:
        assert parse_only.parse(clean_reply(topic="")).topic is None

    def test_uncertain_flag_is_propagated(self, parse_only: TutorClassifier) -> None:
        assert parse_only.parse(clean_reply(uncertain=True)).uncertain is True

    def test_extra_keys_are_ignored(self, parse_only: TutorClassifier) -> None:
        result = parse_only.parse(clean_reply(reason="因为它可以因式分解", confidence=0.9))
        assert result.subject is Subject.MATH


class TestClassifyCall:
    async def test_uses_the_prompt_file_at_temperature_zero(
        self, llama_client_factory: Factory, prompts: PromptLibrary
    ) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return chat_response(clean_reply())

        classifier = TutorClassifier(llama_client_factory(handler), prompts)
        result = await classifier.classify(QUESTION)

        assert result.grade == 9
        assert captured["stream"] is False
        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == CLASSIFY_MAX_TOKENS
        assert captured["chat_template_kwargs"] == {"enable_thinking": False}
        assert prompts.get(CLASSIFY_PROMPT) in captured["messages"][0]["content"]
        assert captured["messages"][1]["content"] == QUESTION
        assert captured["messages"][0]["role"] == "system"

    async def test_backend_failure_is_not_swallowed(
        self, offline_llama: LlamaClient, prompts: PromptLibrary
    ) -> None:
        classifier = TutorClassifier(offline_llama, prompts)
        with pytest.raises(LlamaUnavailableError):
            await classifier.classify(QUESTION)
