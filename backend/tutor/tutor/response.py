"""Structured tutor output (ENGINEERING_PLAN.md §6, §12, §16).

The vocabulary shared by the whole tutor layer lives here:

* :class:`Confidence` — ``high`` / ``medium`` / ``low``, never a fake probability
  (§6);
* :class:`TutorMode` — the answering modes of §12 (``exam`` / ``practice`` are
  added in Phase 10);
* :class:`Subject` — the classifier's subject vocabulary (§1 题目结构化);
* :class:`TutorResponse` — the structured result of one tutor turn;
* the three stream events the engine emits towards the SSE endpoint.

Nothing here talks to the model or to the network.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InvalidTutorModeError(ValueError):
    """Raised when a caller asks for a mode the tutor does not implement."""


class Confidence(StrEnum):
    """Coarse answer confidence (§6: no pseudo-precise probabilities)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TutorMode(StrEnum):
    """Answering modes (§1, §12). ``exam`` / ``practice`` arrive in Phase 10."""

    TUTOR = "tutor"
    EXPLAIN = "explain"
    CHECK = "check"

    @classmethod
    def parse(cls, value: str) -> TutorMode:
        """Coerce a request string into a mode, or fail loudly."""
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in cls)
            raise InvalidTutorModeError(
                f"unknown tutor mode {value!r}; allowed: {allowed}"
            ) from exc


class Subject(StrEnum):
    """Subject vocabulary shared by the classifier and the response model."""

    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    ENGLISH = "english"
    OTHER = "other"
    UNKNOWN = "unknown"


class TutorResponse(BaseModel):
    """One structured tutor answer (§6).

    ``answer`` is empty in the ``start`` event, where only the metadata produced
    before generation is known, and filled in the ``done`` event.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = ""
    subject: Subject
    estimated_grade: int | None = Field(default=None, ge=1, le=12)
    topic: str | None = None
    confidence: Confidence
    tools_used: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    mode: TutorMode

    def with_answer(
        self, answer: str, *, confidence: Confidence, warnings: tuple[str, ...]
    ) -> TutorResponse:
        """Rebuild the response once the whole answer is known."""
        return self.model_copy(
            update={"answer": answer, "confidence": confidence, "warnings": warnings}
        )


class TutorStartEvent(BaseModel):
    """Emitted once the metadata is known and before the first token."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["start"] = "start"
    response: TutorResponse


class TutorTokenEvent(BaseModel):
    """One streamed piece of the answer."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["token"] = "token"
    text: str


class TutorDoneEvent(BaseModel):
    """Emitted after the answer is complete, carrying the final structure.

    Token accounting travels with this event because the engine is the layer that
    consumes the model stream; §18 needs the counts on every chat request.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["done"] = "done"
    response: TutorResponse
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


TutorEvent = TutorStartEvent | TutorTokenEvent | TutorDoneEvent
