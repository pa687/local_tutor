"""``POST /api/chat`` — SSE streaming chat (ENGINEERING_PLAN.md §6, §16).

Phase 2 routes every turn through :class:`~tutor.tutor.engine.TutorEngine`, so this
module owns only the transport: it validates the request, resolves the mode, maps
engine events onto SSE, and maps engine failures onto HTTP status codes. It contains
no prompt text, no policy rule and no direct model call — §6 forbids it and
``tests/test_api_boundaries.py`` enforces it.

SSE contract (also documented in README):

    event: start   data: {"request_id", "model", "mode", "subject", "estimated_grade",
                          "topic", "confidence", "tools_used", "warnings"}
    event: token   data: {"text": "..."}
    event: revision data: {"attempt": 1, "detail": "..."}   # discard the text so far
    event: done    data: {"request_id", "prompt_tokens", "completion_tokens",
                          "latency_ms", "mode", "subject", "estimated_grade", "topic",
                          "confidence", "tools_used", "warnings", "verification"}
    event: error   data: {"error", "detail"}

``start`` and ``done`` carry the same metadata shape; ``start`` reports it before the
first token (with the baseline confidence) and ``done`` after the answer has been
inspected (with the final confidence, any policy warnings, and the §7 verification
verdict). ``answer`` is not repeated in ``done`` — the client already has it from the
``token`` events. A ``revision`` event means the tools contradicted the answer and a
corrected pass follows: the client must drop the text it has buffered so far (§7).

Failures before the first token are reported as HTTP status codes (503 / 504 / 502 /
422) so callers do not have to parse a stream to learn the request cannot be served.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tutor.config import AppConfig
from tutor.llm.client import (
    LlamaClientError,
    LlamaResponseError,
    LlamaTimeoutError,
    LlamaUnavailableError,
)
from tutor.logging_config import RequestLogRecord, current_request_id, log_request
from tutor.tutor.engine import ConversationRef, StudentRef, TutorEngine
from tutor.tutor.response import (
    InvalidTutorModeError,
    TutorDoneEvent,
    TutorEvent,
    TutorMode,
    TutorResponse,
    TutorRevisionEvent,
    TutorStartEvent,
    TutorTokenEvent,
    VerificationSummary,
)

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body of ``POST /api/chat`` (§16).

    ``grade`` is optional: Phase 2 takes it from the request, Phase 7 replaces it
    with the student profile's grade. When absent, the classifier's estimate is used.
    """

    student_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    mode: str | None = None
    grade: int | None = Field(default=None, ge=1, le=12)
    message: str = Field(min_length=1)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _engine(request: Request) -> TutorEngine:
    engine: TutorEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - create_app always installs one
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "tutor engine not initialised")
    return engine


def _resolve_mode(raw: str | None, config: AppConfig) -> TutorMode:
    """Turn the request/config mode string into a :class:`TutorMode` (422 if unknown)."""
    try:
        return TutorMode.parse(raw if raw is not None else config.tutor.default_mode)
    except InvalidTutorModeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


def _metadata(response: TutorResponse) -> dict[str, Any]:
    return {
        "mode": response.mode.value,
        "subject": response.subject.value,
        "estimated_grade": response.estimated_grade,
        "topic": response.topic,
        "confidence": response.confidence.value,
        "tools_used": list(response.tools_used),
        "warnings": list(response.warnings),
    }


def _as_http_error(exc: LlamaClientError) -> HTTPException:
    if isinstance(exc, LlamaUnavailableError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"llama-server unavailable: {exc}"
        )
    if isinstance(exc, LlamaTimeoutError):
        return HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, f"llama-server timeout: {exc}")
    if isinstance(exc, LlamaResponseError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, f"llama-server error: {exc}")
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Stream an answer as SSE, one token event at a time."""
    config: AppConfig = request.app.state.config
    engine = _engine(request)
    mode = _resolve_mode(payload.mode, config)
    model = config.llm.model
    request_id = current_request_id() or "unknown"

    events = engine.stream(
        StudentRef(id=payload.student_id, grade=payload.grade),
        ConversationRef(id=payload.conversation_id),
        payload.message,
        mode=mode,
    )
    # Pull the first event eagerly: a dead backend or an unusable request must still
    # be reportable as an HTTP status code, not as a half-open stream.
    try:
        first: TutorEvent | None = await anext(events)
    except StopAsyncIteration:  # pragma: no cover - the engine always emits a start
        first = None
    except LlamaClientError as exc:
        await events.aclose()
        raise _as_http_error(exc) from exc

    started_at = time.perf_counter()

    async def event_stream() -> AsyncIterator[str]:
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        final: TutorResponse | None = None
        verification: VerificationSummary | None = None
        try:
            async for event in _chain(first, events):
                if isinstance(event, TutorStartEvent):
                    yield _sse(
                        "start",
                        {"request_id": request_id, "model": model, **_metadata(event.response)},
                    )
                elif isinstance(event, TutorTokenEvent):
                    yield _sse("token", {"text": event.text})
                elif isinstance(event, TutorRevisionEvent):
                    logger.info(
                        "chat revision %s after a tool conflict: %s", event.attempt, event.detail
                    )
                    yield _sse("revision", {"attempt": event.attempt, "detail": event.detail})
                elif isinstance(event, TutorDoneEvent):
                    final = event.response
                    verification = event.verification
                    prompt_tokens = event.prompt_tokens or prompt_tokens
                    completion_tokens = event.completion_tokens or completion_tokens
                    yield _sse(
                        "done",
                        {
                            "request_id": request_id,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
                            **_metadata(final),
                            "verification": (
                                verification.model_dump(mode="json")
                                if verification is not None
                                else None
                            ),
                        },
                    )
        except LlamaClientError as exc:
            logger.warning("chat stream aborted: %s", exc)
            yield _sse("error", {"error": type(exc).__name__, "detail": str(exc)})
        finally:
            await events.aclose()
            log_request(
                RequestLogRecord(
                    request_id=request_id,
                    student_id=payload.student_id,
                    conversation_id=payload.conversation_id,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    tools_used=final.tools_used if final is not None else (),
                    verification_status=(
                        verification.status.value if verification is not None else None
                    ),
                ),
                event="chat.completed",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _chain(
    first: TutorEvent | None,
    events: AsyncGenerator[TutorEvent, None],
) -> AsyncGenerator[TutorEvent, None]:
    """Re-attach the event pulled eagerly (for status codes) to the rest of the stream."""
    if first is not None:
        yield first
    async for event in events:
        yield event
