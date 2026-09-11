"""``TutorEngine`` — the only path from a student message to an answer (§6).

Call chain (§6, with the policy layer of §12 made explicit):

```text
ChatRequest → TutorEngine → Classifier ┐
                        → Policy       ├→ ContextBuilder → LlamaClient → TutorResponse
                        └──────────────┘
```

Endpoints must never touch :class:`~tutor.llm.client.LlamaClient` themselves; they
translate engine events into SSE and engine exceptions into HTTP status codes.

The engine emits three kinds of event:

* ``start`` — metadata known before generation (subject, estimated grade, topic,
  baseline confidence);
* ``token`` — pieces of the answer, passed through untouched;
* ``done`` — the final structured :class:`~tutor.tutor.response.TutorResponse`.

Confidence semantics for this phase: it describes how well *this phase* could ground
the turn — a clean classification with no detected policy conflict is ``high``, a
fallback classification or an unknown grade is ``medium``, and a detected
out-of-grade method (or a failed classification) is ``low``. Phase 4 lowers it
further when tool verification contradicts the answer.

Phase-7/8 note: ``StudentRef`` / ``ConversationRef`` are deliberately thin stand-ins
for ``StudentProfile`` (§9) and ``Conversation`` (§10). They keep the §6 signature
(``respond(student, conversation, message)``) stable while profiles and conversation
memory do not exist yet; those phases widen the stand-ins, not the call chain.

Inference parameters come from :class:`~tutor.config.AppConfig` (§1.5): the engine
reads ``llm.enable_thinking`` (see :attr:`TutorEngine.answer_options`) and
``memory.recent_turns`` instead of hardcoding them.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from tutor.config import AppConfig
from tutor.llm.client import LlamaClient
from tutor.llm.context import ContextBuilder
from tutor.llm.models import ChatMessage, GenerationOptions
from tutor.llm.prompts import PromptLibrary
from tutor.tutor.classifier import ClassifiedQuestion, TutorClassifier
from tutor.tutor.policy import TutorPolicy, find_out_of_scope_methods
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

EMPTY_ANSWER_WARNING = "模型没有返回任何内容"
OUT_OF_SCOPE_WARNING = "答案中出现了可能超出年级范围的方法：{methods}，请确认是否适合该学生"

_CONFIDENCE_ORDER: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


class TutorEngineError(RuntimeError):
    """The engine could not produce a response at all."""


@dataclass(frozen=True)
class StudentRef:
    """Phase-2 stand-in for ``StudentProfile`` (§9 lands in Phase 7)."""

    id: str
    grade: int | None = None


@dataclass(frozen=True)
class ConversationRef:
    """Phase-2 stand-in for ``Conversation`` (§10 memory lands in Phase 8).

    ``history`` are the *previous* turns only; the current message is passed
    separately so the two can never be duplicated or lost.
    """

    id: str
    history: tuple[ChatMessage, ...] = ()


class TutorEngine:
    """Assembles context, streams the answer, and returns structured metadata."""

    def __init__(
        self,
        client: LlamaClient,
        prompts: PromptLibrary,
        config: AppConfig,
        *,
        policy: TutorPolicy | None = None,
        classifier: TutorClassifier | None = None,
        context: ContextBuilder | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._policy = policy if policy is not None else TutorPolicy()
        self._classifier = (
            classifier if classifier is not None else TutorClassifier(client, prompts)
        )
        self._context = context if context is not None else ContextBuilder(prompts, config.memory)

    @property
    def answer_options(self) -> GenerationOptions:
        """Generation defaults for the answer stream, taken from configuration.

        §1.5 forbids hardcoding inference parameters, so whether the model may think
        before answering is a config decision (``llm.enable_thinking``) rather than a
        literal buried in this module.
        """
        return GenerationOptions(enable_thinking=self._config.llm.enable_thinking)

    async def stream(
        self,
        student: StudentRef,
        conversation: ConversationRef,
        message: str,
        *,
        mode: TutorMode,
    ) -> AsyncGenerator[TutorEvent, None]:
        """Run one turn, yielding ``start`` → ``token``… → ``done`` (§6).

        Anything that fails before the answer starts (classification included) is
        raised, not turned into a token stream, so the caller can still answer with
        a real HTTP status code.
        """
        classification = await self._classifier.classify(message)
        grade = student.grade if student.grade is not None else classification.grade
        decision = self._policy.decide(
            mode=mode,
            subject=classification.subject,
            grade=grade,
            message=message,
            history=conversation.history,
        )
        baseline = _baseline_confidence(classification)
        warnings = [*classification.warnings, *decision.warnings]
        metadata = TutorResponse(
            subject=classification.subject,
            estimated_grade=grade,
            topic=classification.topic,
            confidence=baseline,
            warnings=tuple(warnings),
            mode=mode,
        )
        yield TutorStartEvent(response=metadata)

        messages = self._context.build(
            message=message,
            policy_block=decision.as_context_block(),
            history=conversation.history,
        )
        parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        async for delta in self._client.stream_chat(messages, self.answer_options):
            prompt_tokens = delta.prompt_tokens or prompt_tokens
            completion_tokens = delta.completion_tokens or completion_tokens
            if delta.content:
                parts.append(delta.content)
                yield TutorTokenEvent(text=delta.content)

        answer = "".join(parts)
        out_of_scope = find_out_of_scope_methods(answer, decision)
        if out_of_scope:
            warnings.append(OUT_OF_SCOPE_WARNING.format(methods="、".join(out_of_scope)))
        if not answer.strip():
            warnings.append(EMPTY_ANSWER_WARNING)
        yield TutorDoneEvent(
            response=metadata.with_answer(
                answer,
                confidence=_final_confidence(baseline, out_of_scope=out_of_scope),
                warnings=tuple(warnings),
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def respond(
        self,
        student: StudentRef,
        conversation: ConversationRef,
        message: str,
        *,
        mode: TutorMode,
    ) -> TutorResponse:
        """Non-streaming counterpart of :meth:`stream` (§6 core interface)."""
        final: TutorResponse | None = None
        async for event in self.stream(student, conversation, message, mode=mode):
            if isinstance(event, TutorDoneEvent):
                final = event.response
        if final is None:  # pragma: no cover - stream() always yields a done event
            raise TutorEngineError("tutor stream ended without a done event")
        return final


def _baseline_confidence(classification: ClassifiedQuestion) -> Confidence:
    """Confidence from the classification alone (known before the answer exists)."""
    if classification.is_fallback:
        return Confidence.LOW
    if classification.uncertain or classification.subject is Subject.UNKNOWN:
        return Confidence.MEDIUM
    return Confidence.HIGH


def _final_confidence(
    baseline: Confidence,
    *,
    out_of_scope: tuple[str, ...],
) -> Confidence:
    """Downgrade the baseline once the finished answer has been inspected."""
    ceiling = Confidence.LOW if out_of_scope else Confidence.HIGH
    return min((baseline, ceiling), key=lambda level: _CONFIDENCE_ORDER[level])
