"""``POST /api/chat`` — SSE streaming chat (ENGINEERING_PLAN.md §5, §16).

Phase 1 sends the student's message straight to llama-server, so the transport is
provable before any policy exists. Phase 2 replaces the message assembly below with
the ``TutorEngine → Policy → ContextBuilder`` chain (§6); this module must not grow
prompt or policy logic in the meantime.

SSE contract (also documented in README):

    event: start   data: {"request_id", "model", "mode"}
    event: token   data: {"text": "..."}
    event: done    data: {"request_id", "prompt_tokens", "completion_tokens", "latency_ms"}
    event: error   data: {"error", "detail"}

Failures before the first token are reported as HTTP status codes (503 / 504 / 502)
so callers do not have to parse a stream to learn the backend is gone.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tutor.config import AppConfig
from tutor.llm.client import (
    LlamaClient,
    LlamaClientError,
    LlamaResponseError,
    LlamaTimeoutError,
    LlamaUnavailableError,
)
from tutor.llm.models import ChatMessage, StreamDelta
from tutor.logging_config import RequestLogRecord, current_request_id, log_request

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body of ``POST /api/chat`` (§16)."""

    student_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    mode: str | None = None
    message: str = Field(min_length=1)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _llama_client(request: Request) -> LlamaClient:
    client: LlamaClient | None = getattr(request.app.state, "llama", None)
    if client is None:  # pragma: no cover - create_app always installs one
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "llama client not initialised")
    return client


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
    client = _llama_client(request)
    mode = payload.mode or config.tutor.default_mode
    model = client.model
    request_id = current_request_id() or "unknown"

    # TODO(Phase 2): assemble messages through TutorEngine/Policy/ContextBuilder (§6).
    messages = [ChatMessage(role="user", content=payload.message)]

    stream = client.stream_chat(messages)
    try:
        first: StreamDelta | None = await anext(stream)
    except StopAsyncIteration:
        first = None
    except LlamaClientError as exc:
        await stream.aclose()
        raise _as_http_error(exc) from exc

    started_at = time.perf_counter()

    async def event_stream() -> AsyncIterator[str]:
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            yield _sse("start", {"request_id": request_id, "model": model, "mode": mode})
            async for delta in _chain(first, stream):
                prompt_tokens = delta.prompt_tokens or prompt_tokens
                completion_tokens = delta.completion_tokens or completion_tokens
                if delta.content:
                    yield _sse("token", {"text": delta.content})
            yield _sse(
                "done",
                {
                    "request_id": request_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            )
        except LlamaClientError as exc:
            logger.warning("chat stream aborted: %s", exc)
            yield _sse("error", {"error": type(exc).__name__, "detail": str(exc)})
        finally:
            await stream.aclose()
            log_request(
                RequestLogRecord(
                    request_id=request_id,
                    student_id=payload.student_id,
                    conversation_id=payload.conversation_id,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
                ),
                event="chat.completed",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _chain(
    first: StreamDelta | None,
    stream: AsyncIterator[StreamDelta],
) -> AsyncIterator[StreamDelta]:
    """Re-attach the delta pulled eagerly (for status codes) to the rest of the stream."""
    if first is not None:
        yield first
    async for delta in stream:
        yield delta
