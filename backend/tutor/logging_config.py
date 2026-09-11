"""Logging skeleton for Local Tutor (ENGINEERING_PLAN.md §18).

Every request carries a ``request_id`` (propagated via a :class:`ContextVar` and
echoed in the ``X-Request-Id`` response header). Per-request metadata is emitted
as a single JSON line with exactly the fields §18 requires:

    request_id, student_id, conversation_id, model, prompt_tokens,
    completion_tokens, latency, tools_used, verification_status

Student message content is never logged here — conversations belong in the
database, not in the log.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final

REQUEST_ID_HEADER: Final = "X-Request-Id"
REQUEST_LOGGER_NAME: Final = "tutor.request"
LOG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s %(message)s"

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


@dataclass(frozen=True)
class RequestLogRecord:
    """Metadata-only record for one request (§18).

    Values are ``None`` until the responsible phase fills them in; they are kept in
    the payload as explicit nulls so downstream log consumers see a stable schema.
    """

    request_id: str
    student_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    tools_used: tuple[str, ...] = ()
    verification_status: str | None = None

    def as_fields(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "student_id": self.student_id,
            "conversation_id": self.conversation_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency": self.latency_ms,
            "tools_used": list(self.tools_used),
            "verification_status": self.verification_status,
        }


def setup_logging(level: str | int = logging.INFO) -> None:
    """Configure root logging. Idempotent enough to be called per app factory run."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def current_request_id() -> str | None:
    """The ``request_id`` of the request currently being handled, else ``None``."""
    return request_id_var.get()


def log_request(
    record: RequestLogRecord,
    *,
    logger: logging.Logger | None = None,
    event: str = "request",
) -> None:
    """Emit one metadata-only JSON log line.

    Never pass message content to this function: §18 forbids logging full student
    messages outside the database.
    """
    target = logger if logger is not None else logging.getLogger(REQUEST_LOGGER_NAME)
    payload = {"event": event, **record.as_fields()}
    target.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
