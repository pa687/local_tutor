"""Tests for ``POST /api/chat`` SSE streaming (ENGINEERING_PLAN.md §6, §16).

Phase 2 changed the transport contract: ``start`` and ``done`` now carry the
structured metadata of the ``TutorResponse`` (§6), and an unknown ``mode`` is a
validation error instead of free text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from tutor.config import AppConfig
from tutor.llm.client import LlamaClient, LlamaUnavailableError
from tutor.llm.models import ChatCompletion, ChatMessage, GenerationOptions, StreamDelta
from tutor.main import create_app

Factory = Callable[..., LlamaClient]
REQUEST: dict[str, object] = {
    "student_id": "student-001",
    "conversation_id": "abc",
    "mode": "tutor",
    "message": "why does the sign flip?",
}


def parse_sse(payload: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE payload into ``(event, data)`` pairs."""
    parsed: list[tuple[str, dict[str, object]]] = []
    for block in payload.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "{}")
        parsed.append((event, json.loads(data)))
    return parsed


def app_client(config: AppConfig, llama: LlamaClient) -> TestClient:
    return TestClient(create_app(config, llama_client=llama))


class DyingClient:
    """Classifies fine, streams one token, then the backend dies (OOM/crash case)."""

    model = "qwen3.5-9b"

    async def complete(
        self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None
    ) -> ChatCompletion:
        return ChatCompletion(
            content=json.dumps(
                {"subject": "math", "grade": 9, "topic": "quadratic_equations", "uncertain": False}
            )
        )

    async def stream_chat(
        self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[StreamDelta]:
        yield StreamDelta(content="Hel")
        raise LlamaUnavailableError("llama-server unreachable: connection reset")

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None


class TestHappyPath:
    def test_streams_start_token_done(self, test_config: AppConfig, tutor_llama: Factory) -> None:
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(response.text)
        assert [name for name, _ in events] == ["start", "token", "token", "done"]
        assert events[1][1]["text"] == "Hel"
        assert events[2][1]["text"] == "lo"
        assert events[0][1]["model"] == "qwen3.5-9b"
        assert events[0][1]["mode"] == "tutor"
        assert events[3][1]["prompt_tokens"] == 11
        assert events[3][1]["completion_tokens"] == 2
        assert float(cast(float, events[3][1]["latency_ms"])) >= 0

    def test_start_and_done_carry_the_tutor_metadata(
        self, test_config: AppConfig, tutor_llama: Factory
    ) -> None:
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json=REQUEST)

        events = dict(parse_sse(response.text))
        for name in ("start", "done"):
            payload = dict(events[name])
            assert payload["subject"] == "math"
            assert payload["estimated_grade"] == 9
            assert payload["topic"] == "quadratic_equations"
            assert payload["confidence"] == "high"
            assert payload["tools_used"] == []
            assert payload["warnings"] == []
        # §6: the answer is streamed once, never repeated in the done payload.
        assert "answer" not in events["done"]

    def test_mode_defaults_to_config(self, test_config: AppConfig, tutor_llama: Factory) -> None:
        request = {key: value for key, value in REQUEST.items() if key != "mode"}
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json=request)

        start = parse_sse(response.text)[0]
        assert start[1]["mode"] == test_config.tutor.default_mode

    @pytest.mark.parametrize("mode", ["tutor", "explain", "check"])
    def test_each_mode_is_accepted(
        self, test_config: AppConfig, tutor_llama: Factory, mode: str
    ) -> None:
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json={**REQUEST, "mode": mode})

        assert response.status_code == 200
        assert parse_sse(response.text)[0][1]["mode"] == mode

    def test_grade_from_the_request_reaches_the_response(
        self, test_config: AppConfig, tutor_llama: Factory
    ) -> None:
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json={**REQUEST, "grade": 11})

        assert parse_sse(response.text)[0][1]["estimated_grade"] == 11

    def test_degraded_classification_is_visible_in_both_events(
        self, test_config: AppConfig, tutor_llama: Factory
    ) -> None:
        with app_client(test_config, tutor_llama(classification="不是 JSON")) as client:
            response = client.post("/api/chat", json=REQUEST)

        events = dict(parse_sse(response.text))
        assert events["start"]["subject"] == "unknown"
        assert events["done"]["confidence"] == "low"
        warnings = cast(list[str], events["done"]["warnings"])
        assert any("结构化失败" in warning for warning in warnings)


class TestBackendFailures:
    def test_unreachable_llama_returns_503_not_a_crash(
        self, test_config: AppConfig, tutor_llama: Factory
    ) -> None:
        llama = tutor_llama(classify_error=httpx.ConnectError("connection refused"))
        with app_client(test_config, llama) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_timeout_returns_504(self, test_config: AppConfig, tutor_llama: Factory) -> None:
        llama = tutor_llama(classify_error=httpx.ReadTimeout("too slow"))
        with app_client(test_config, llama) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 504
        assert "timeout" in response.json()["detail"]

    def test_llama_error_status_returns_502(
        self, test_config: AppConfig, tutor_llama: Factory
    ) -> None:
        llama = tutor_llama(classify_error=httpx.Response(500, text='{"error":"oom"}'))
        with app_client(test_config, llama) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 502

    def test_failure_mid_stream_becomes_an_error_event(self, test_config: AppConfig) -> None:
        client_stub = cast(LlamaClient, DyingClient())
        with app_client(test_config, client_stub) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 200
        events = parse_sse(response.text)
        assert [name for name, _ in events] == ["start", "token", "error"]
        assert events[2][1]["error"] == "LlamaUnavailableError"


class TestValidation:
    @pytest.mark.parametrize("missing", ["student_id", "conversation_id", "message"])
    def test_required_fields(self, client: TestClient, missing: str) -> None:
        payload = {key: value for key, value in REQUEST.items() if key != missing}
        assert client.post("/api/chat", json=payload).status_code == 422

    def test_empty_message_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={**REQUEST, "message": ""}).status_code == 422

    def test_unknown_mode_is_rejected_with_the_allowed_values(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={**REQUEST, "mode": "socratic"})
        assert response.status_code == 422
        assert "allowed" in response.json()["detail"]

    @pytest.mark.parametrize("grade", [0, 13])
    def test_out_of_range_grade_is_rejected(self, client: TestClient, grade: int) -> None:
        assert client.post("/api/chat", json={**REQUEST, "grade": grade}).status_code == 422


class TestLogging:
    def test_metadata_is_logged_without_message_content(
        self,
        test_config: AppConfig,
        tutor_llama: Factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="tutor.request")
        with app_client(test_config, tutor_llama()) as client:
            response = client.post("/api/chat", json=REQUEST)

        records = [record for record in caplog.records if record.name == "tutor.request"]
        payload = json.loads(records[-1].message)
        assert payload["event"] == "chat.completed"
        assert payload["request_id"] == response.headers["X-Request-Id"]
        assert payload["student_id"] == "student-001"
        assert payload["conversation_id"] == "abc"
        assert payload["prompt_tokens"] == 11
        assert payload["completion_tokens"] == 2
        assert payload["tools_used"] == []
        assert "sign flip" not in caplog.text
