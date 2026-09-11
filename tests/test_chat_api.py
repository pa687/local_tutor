"""Tests for ``POST /api/chat`` SSE streaming (ENGINEERING_PLAN.md §5, §16)."""

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
from tutor.llm.models import ChatMessage, GenerationOptions, StreamDelta
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


def chunk(text: str, finish: str | None = None) -> str:
    payload = {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n"


def final_chunks(prompt_tokens: int, completion_tokens: int) -> str:
    """The two trailing chunks llama-server sends: finish reason, then usage."""
    finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return f"data: {json.dumps(finish)}\n\ndata: {json.dumps(usage)}\n\n"


def streaming_llama(
    factory: Factory,
    *,
    tokens: Sequence[str] = ("Hel", "lo"),
    prompt_tokens: int = 11,
    completion_tokens: int = 2,
) -> LlamaClient:
    body = "".join(chunk(token) for token in tokens)
    body += final_chunks(prompt_tokens, completion_tokens)
    body += "data: [DONE]\n\n"

    return factory(lambda request: httpx.Response(200, text=body))


def refusing_llama(factory: Factory) -> LlamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return factory(handler)


def timing_out_llama(factory: Factory) -> LlamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    return factory(handler)


def failing_llama(factory: Factory) -> LlamaClient:
    return factory(lambda request: httpx.Response(500, text='{"error":"out of memory"}'))


def app_client(config: AppConfig, llama: LlamaClient) -> TestClient:
    return TestClient(create_app(config, llama_client=llama))


class DyingClient:
    """Streams one token, then the backend dies mid-answer (OOM/crash case)."""

    model = "qwen3.5-9b"

    async def stream_chat(
        self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[StreamDelta]:
        yield StreamDelta(content="Hel")
        raise LlamaUnavailableError("llama-server unreachable: connection reset")

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None


class TestHappyPath:
    def test_streams_start_token_done(
        self, test_config: AppConfig, llama_client_factory: Factory
    ) -> None:
        llama = streaming_llama(llama_client_factory)
        with app_client(test_config, llama) as client:
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

    def test_mode_defaults_to_config(
        self, test_config: AppConfig, llama_client_factory: Factory
    ) -> None:
        request = {key: value for key, value in REQUEST.items() if key != "mode"}
        with app_client(test_config, streaming_llama(llama_client_factory)) as client:
            response = client.post("/api/chat", json=request)

        start = parse_sse(response.text)[0]
        assert start[1]["mode"] == test_config.tutor.default_mode


class TestBackendFailures:
    def test_unreachable_llama_returns_503_not_a_crash(
        self, test_config: AppConfig, llama_client_factory: Factory
    ) -> None:
        with app_client(test_config, refusing_llama(llama_client_factory)) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]

    def test_timeout_returns_504(
        self, test_config: AppConfig, llama_client_factory: Factory
    ) -> None:
        with app_client(test_config, timing_out_llama(llama_client_factory)) as client:
            response = client.post("/api/chat", json=REQUEST)

        assert response.status_code == 504
        assert "timeout" in response.json()["detail"]

    def test_llama_error_status_returns_502(
        self, test_config: AppConfig, llama_client_factory: Factory
    ) -> None:
        with app_client(test_config, failing_llama(llama_client_factory)) as client:
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


class TestLogging:
    def test_metadata_is_logged_without_message_content(
        self,
        test_config: AppConfig,
        llama_client_factory: Factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="tutor.request")
        with app_client(test_config, streaming_llama(llama_client_factory)) as client:
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
