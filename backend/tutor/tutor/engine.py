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
  baseline confidence, and the tools already used for the student's work);
* ``token`` — pieces of the answer, passed through untouched;
* ``revision`` — a corrected answer is coming: the previous pass conflicted with the
  tools, so the client must discard what it has streamed so far;
* ``done`` — the final structured :class:`~tutor.tutor.response.TutorResponse` plus the
  verification verdict and its tool trail.

Verification (§7) runs twice per turn:

* **before** generating, in ``check`` mode, on the *student's* own work — the tool
  verdict goes into the context so the model can point at the earliest broken step
  (§22 B/E) instead of re-solving;
* **after** generating, on the *model's* answer — a conflict triggers the
  self-correction loop (at most ``tutor.max_verification_attempts`` re-checks, §7).

Confidence semantics: a clean classification with no policy conflict is ``high``; a
fallback classification, an unknown grade or an unverifiable answer is ``medium`` /
``low``; a detected out-of-grade method or a conflict that survived the loop is
``low``. Everything that lowers confidence also adds a warning (§7 forbids hiding it).

Phase-7/8 note: ``StudentRef`` / ``ConversationRef`` are deliberately thin stand-ins
for ``StudentProfile`` (§9) and ``Conversation`` (§10). They keep the §6 signature
(``respond(student, conversation, message)``) stable while profiles and conversation
memory do not exist yet; those phases widen the stand-ins, not the call chain.

Inference parameters come from :class:`~tutor.config.AppConfig` (§1.5): the engine
reads ``llm.enable_thinking`` (see :attr:`TutorEngine.answer_options`) and
``memory.recent_turns`` instead of hardcoding them.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import dataclass

from tutor.config import AppConfig
from tutor.llm.client import LlamaClient
from tutor.llm.context import ContextBuilder
from tutor.llm.models import ChatMessage, GenerationOptions, StreamDelta
from tutor.llm.prompts import PromptLibrary
from tutor.tools.registry import build_default_registry
from tutor.tutor.classifier import ClassifiedQuestion, TutorClassifier
from tutor.tutor.policy import TutorPolicy, find_out_of_scope_methods
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
    VerificationSummary,
)
from tutor.tutor.verifier import ANSWER_SOURCE, STUDENT_SOURCE, VerificationReport, Verifier

EMPTY_ANSWER_WARNING = "模型没有返回任何内容"
OUT_OF_SCOPE_WARNING = "答案中出现了可能超出年级范围的方法：{methods}，请确认是否适合该学生"
CONFLICT_WARNING = "工具校验发现答案与题目冲突：{detail}（已尝试 {attempts} 次自我修正，仍未通过）"
UNVERIFIABLE_WARNING = "结果未能通过自动验证"

#: Prompt used when a conflict forces the model to re-check its own answer (§7).
RECHECK_PROMPT = "verify.md"

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


@dataclass
class PassAccounting:
    """Text and token counts of one model pass (a first answer or a revision)."""

    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def record(self, delta: StreamDelta) -> None:
        """Pick up the counts llama-server reports in the trailing usage chunk."""
        self.prompt_tokens = delta.prompt_tokens or self.prompt_tokens
        self.completion_tokens = delta.completion_tokens or self.completion_tokens


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
        verifier: Verifier | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._prompts = prompts
        self._policy = policy if policy is not None else TutorPolicy()
        self._classifier = (
            classifier if classifier is not None else TutorClassifier(client, prompts)
        )
        self._context = context if context is not None else ContextBuilder(prompts, config.memory)
        self._verifier = verifier if verifier is not None else Verifier(build_default_registry())

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

        # §22 B/E: in check mode the student's own work is verified first, so the answer
        # can cite the tool verdict instead of re-deriving everything.
        student_report = self._student_report(
            mode=mode, subject=classification.subject, message=message
        )
        metadata = TutorResponse(
            subject=classification.subject,
            estimated_grade=grade,
            topic=classification.topic,
            confidence=baseline,
            tools_used=student_report.tools_used if student_report is not None else (),
            warnings=tuple(warnings),
            mode=mode,
        )
        yield TutorStartEvent(response=metadata)

        messages = self._context.build(
            message=message,
            policy_block=decision.as_context_block(),
            history=conversation.history,
            verification_block=(
                student_report.instruction() or "" if student_report is not None else ""
            ),
        )

        passes: list[PassAccounting] = []
        first = PassAccounting()
        async for event in self._stream_pass(messages, first):
            yield event
        passes.append(first)
        answer = first.text

        verification = self._verify_answer(
            question=message, subject=classification.subject, answer=answer
        )
        attempts = 0
        while (
            verification.status is VerificationStatus.CONFLICT
            and attempts < self._config.tutor.max_verification_attempts
        ):
            attempts += 1
            yield TutorRevisionEvent(attempt=attempts, detail=verification.detail or "")
            revision = PassAccounting()
            async for event in self._stream_pass(
                self._recheck_messages(messages, answer=answer, report=verification), revision
            ):
                yield event
            passes.append(revision)
            answer = revision.text
            verification = self._verify_answer(
                question=message, subject=classification.subject, answer=answer
            )

        out_of_scope = find_out_of_scope_methods(answer, decision)
        if out_of_scope:
            warnings.append(OUT_OF_SCOPE_WARNING.format(methods="、".join(out_of_scope)))
        if not answer.strip():
            warnings.append(EMPTY_ANSWER_WARNING)
        if verification.status is VerificationStatus.CONFLICT:
            warnings.append(
                CONFLICT_WARNING.format(detail=verification.detail or "", attempts=attempts)
            )
        elif verification.status is VerificationStatus.UNVERIFIABLE:
            warnings.append(UNVERIFIABLE_WARNING)
        tools_used = tuple(
            dict.fromkeys(
                [
                    *(student_report.tools_used if student_report is not None else ()),
                    *verification.tools_used,
                ]
            )
        )
        yield TutorDoneEvent(
            response=metadata.with_answer(
                answer,
                confidence=_final_confidence(
                    baseline, out_of_scope=out_of_scope, verification=verification
                ),
                warnings=tuple(warnings),
                tools_used=tools_used,
            ),
            prompt_tokens=_sum_counts(p.prompt_tokens for p in passes),
            completion_tokens=_sum_counts(p.completion_tokens for p in passes),
            verification=_verification_summary(
                verification, attempts=attempts, student_report=student_report
            ),
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

    # --------------------------------------------------------------- internals
    async def _stream_pass(
        self,
        messages: Sequence[ChatMessage],
        accounting: PassAccounting,
    ) -> AsyncGenerator[TutorTokenEvent, None]:
        """Stream one model pass, recording its text and token counts in ``accounting``."""
        parts: list[str] = []
        async for delta in self._client.stream_chat(messages, self.answer_options):
            accounting.record(delta)
            if delta.content:
                parts.append(delta.content)
                yield TutorTokenEvent(text=delta.content)
        accounting.text = "".join(parts)

    def _student_report(
        self,
        *,
        mode: TutorMode,
        subject: Subject,
        message: str,
    ) -> VerificationReport | None:
        """Verify the student's own claims before answering (±22 B/E), maths only."""
        if mode is not TutorMode.CHECK or subject is not Subject.MATH:
            return None
        return self._verifier.verify(question=message, text=message, source=STUDENT_SOURCE)

    def _verify_answer(self, *, question: str, subject: Subject, answer: str) -> VerificationReport:
        """Verify the model's answer; non-maths subjects are simply not checked."""
        if subject is not Subject.MATH:
            return VerificationReport(
                status=VerificationStatus.NOTHING_TO_VERIFY,
                source=ANSWER_SOURCE,
                detail="非数学科目不做 SymPy 校验",
            )
        return self._verifier.verify(question=question, text=answer, source=ANSWER_SOURCE)

    def _recheck_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        answer: str,
        report: VerificationReport,
    ) -> list[ChatMessage]:
        """Ask the model to re-examine the answer the tools contradicted (§7)."""
        instruction = self._prompts.get(RECHECK_PROMPT)
        detail = report.detail or ""
        return [
            *messages,
            ChatMessage(role="assistant", content=answer),
            ChatMessage(role="user", content=f"{instruction}\n\n工具校验发现的冲突：{detail}"),
        ]


def _sum_counts(values: Iterable[int | None]) -> int | None:
    """Total a per-pass token count, staying ``None`` when llama-server reported none."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _verification_summary(
    report: VerificationReport,
    *,
    attempts: int,
    student_report: VerificationReport | None,
) -> VerificationSummary:
    """Shape a :class:`VerificationReport` for the API, keeping its tool trail (§17)."""
    return VerificationSummary.model_validate(
        {
            **report.as_event(),
            "attempts": attempts,
            "student_status": student_report.status if student_report is not None else None,
            "student_detail": student_report.detail if student_report is not None else None,
        }
    )


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
    verification: VerificationReport,
) -> Confidence:
    """Downgrade the baseline once the finished answer and the tools have spoken."""
    downgrades = out_of_scope or verification.status in {
        VerificationStatus.CONFLICT,
        VerificationStatus.UNVERIFIABLE,
    }
    ceiling = Confidence.LOW if downgrades else Confidence.HIGH
    return min((baseline, ceiling), key=lambda level: _CONFIDENCE_ORDER[level])
