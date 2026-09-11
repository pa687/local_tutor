"""Tests for the Phase 0 logging skeleton (ENGINEERING_PLAN.md §18)."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from tutor.logging_config import (
    RequestLogRecord,
    current_request_id,
    log_request,
    request_id_var,
)

REQUIRED_FIELDS = {
    "request_id",
    "student_id",
    "conversation_id",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "latency",
    "tools_used",
    "verification_status",
}


def _tutor_request_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Only the records this project emits; the HTTP client logs too."""
    return [record for record in caplog.records if record.name == "tutor.request"]


def test_record_exposes_every_required_field() -> None:
    record = RequestLogRecord(request_id="r1").as_fields()
    assert set(record) == REQUIRED_FIELDS


def test_record_serialises_tools_as_list() -> None:
    record = RequestLogRecord(request_id="r1", tools_used=("solve_equation",)).as_fields()
    assert record["tools_used"] == ["solve_equation"]


def test_log_request_emits_parseable_json(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="tutor.request")
    log_request(
        RequestLogRecord(
            request_id="r1",
            student_id="student-001",
            conversation_id="abc",
            model="qwen3.5-9b",
            prompt_tokens=120,
            completion_tokens=45,
            latency_ms=812.5,
            tools_used=("check_equivalence",),
            verification_status="verified",
        )
    )
    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "request"
    assert payload["request_id"] == "r1"
    assert payload["tools_used"] == ["check_equivalence"]
    assert payload["verification_status"] == "verified"


def test_request_id_context_is_empty_outside_a_request() -> None:
    assert request_id_var.get() is None
    assert current_request_id() is None


def test_middleware_logs_metadata_without_message_content(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="tutor.request")
    question_marker = "MARKER-STUDENT-QUESTION-42"
    response = client.get("/health", params={"message": question_marker})

    assert response.status_code == 200
    records = _tutor_request_records(caplog)
    logged = "\n".join(record.message for record in records)
    assert question_marker not in logged

    payload = json.loads(records[-1].message)
    assert payload["request_id"] == response.headers["X-Request-Id"]
    assert payload["model"] == "qwen3.5-9b"
    assert payload["latency"] is not None
    assert set(payload) == REQUIRED_FIELDS | {"event"}


def test_middleware_logs_metadata_even_when_the_route_is_missing(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed request is still logged as metadata, still without its content."""
    caplog.set_level(logging.INFO, logger="tutor.request")
    question_marker = "MARKER-STUDENT-QUESTION-43"
    response = client.post("/api/chat", json={"message": question_marker})

    assert response.status_code == 404
    records = _tutor_request_records(caplog)
    assert question_marker not in "\n".join(record.message for record in records)

    payload = json.loads(records[-1].message)
    assert payload["request_id"] == response.headers["X-Request-Id"]
