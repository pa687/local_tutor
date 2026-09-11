"""Subject / grade / topic classifier (ENGINEERING_PLAN.md §1, §6).

The structured fields of :class:`~tutor.tutor.response.TutorResponse` — ``subject``,
``estimated_grade`` and ``topic`` — are filled by one short, deterministic,
non-streaming model call *before* the answer is streamed, so the ``start`` SSE event
already carries them.

Design rules:

* the prompt lives in ``prompts/classify.md`` (§13), never in Python;
* the answer is parsed as JSON and validated defensively — a malformed reply degrades
  **visibly** (``subject=unknown`` plus a warning), never silently;
* a dead backend is *not* degraded at all: :class:`LlamaClientError` propagates so the
  endpoint can answer 503 / 504 / 502 (§23.8).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from tutor.llm.client import LlamaClient
from tutor.llm.models import ChatMessage, GenerationOptions
from tutor.llm.prompts import PromptLibrary
from tutor.tutor.response import Subject

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = "classify.md"
CLASSIFY_MAX_TOKENS = 200
MIN_GRADE = 1
MAX_GRADE = 12

FALLBACK_WARNING = "题目结构化失败，科目/年级/知识点按未知处理"

#: Tolerated spellings the small model occasionally produces.
_SUBJECT_ALIASES: dict[str, Subject] = {
    "maths": Subject.MATH,
    "mathematics": Subject.MATH,
    "数学": Subject.MATH,
    "physics": Subject.PHYSICS,
    "物理": Subject.PHYSICS,
    "chemistry": Subject.CHEMISTRY,
    "化学": Subject.CHEMISTRY,
    "english": Subject.ENGLISH,
    "英语": Subject.ENGLISH,
    "other": Subject.OTHER,
    "unknown": Subject.UNKNOWN,
    "none": Subject.UNKNOWN,
    "null": Subject.UNKNOWN,
}


@dataclass(frozen=True)
class ClassifiedQuestion:
    """Structured view of one student message, plus what went wrong while building it."""

    subject: Subject
    grade: int | None
    topic: str | None
    uncertain: bool
    warnings: tuple[str, ...] = ()

    @property
    def is_fallback(self) -> bool:
        """True when nothing at all could be extracted from the model's reply."""
        return self.warnings == (FALLBACK_WARNING,)


class _RawClassification(BaseModel):
    """Lenient shape of the classifier's JSON; unknown keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    subject: str | None = None
    grade: int | None = None
    topic: str | None = None
    uncertain: bool = False


class TutorClassifier:
    """Fills ``subject`` / ``estimated_grade`` / ``topic`` for one turn."""

    def __init__(self, client: LlamaClient, prompts: PromptLibrary) -> None:
        self._client = client
        self._prompts = prompts

    async def classify(self, message: str) -> ClassifiedQuestion:
        """Ask the model to structure ``message`` and validate what comes back."""
        messages = [
            ChatMessage(role="system", content=self._prompts.get(CLASSIFY_PROMPT)),
            ChatMessage(role="user", content=message),
        ]
        completion = await self._client.complete(
            messages,
            GenerationOptions(
                temperature=0.0,
                max_tokens=CLASSIFY_MAX_TOKENS,
                # Thinking would burn the whole token budget before the JSON object
                # ever appears (verified against Qwen3.5: 200/200 tokens of reasoning,
                # empty content). Structured extraction does not need it.
                enable_thinking=False,
            ),
        )
        return self.parse(completion.content or "")

    def parse(self, raw: str) -> ClassifiedQuestion:
        """Validate one classifier reply. Never raises for model-side garbage."""
        try:
            payload = _extract_json_object(raw)
        except ValueError as exc:
            logger.warning("classifier reply was not usable JSON: %s", exc)
            return _fallback()
        try:
            parsed = _RawClassification.model_validate(payload)
        except ValidationError as exc:
            logger.warning("classifier reply did not match the expected shape: %s", exc)
            return _fallback()

        warnings: list[str] = []
        subject = _coerce_subject(parsed.subject)
        if subject is None:
            warnings.append(f"模型返回的科目 {parsed.subject!r} 无法识别，按 unknown 处理")
            subject = Subject.UNKNOWN

        grade = parsed.grade
        if grade is not None and not (MIN_GRADE <= grade <= MAX_GRADE):
            warnings.append(f"模型给出的年级 {grade} 不在 {MIN_GRADE}–{MAX_GRADE} 范围内，已忽略")
            grade = None

        topic = parsed.topic.strip() if isinstance(parsed.topic, str) else None
        topic = topic or None

        return ClassifiedQuestion(
            subject=subject,
            grade=grade,
            topic=topic,
            uncertain=bool(parsed.uncertain),
            warnings=tuple(warnings),
        )


def _fallback() -> ClassifiedQuestion:
    logger.warning(FALLBACK_WARNING)
    return ClassifiedQuestion(
        subject=Subject.UNKNOWN,
        grade=None,
        topic=None,
        uncertain=True,
        warnings=(FALLBACK_WARNING,),
    )


def _coerce_subject(value: Any) -> Subject | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if not token:
        return None
    if token in _SUBJECT_ALIASES:
        return _SUBJECT_ALIASES[token]
    try:
        return Subject(token)
    except ValueError:
        return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (fences and prose tolerated)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value is not an object")
    return payload
