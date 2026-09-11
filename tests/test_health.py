"""Tests for the Phase 0 ``/health`` placeholder (ENGINEERING_PLAN.md §16)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tutor.config import AppConfig, LLMConfig
from tutor.main import create_app


def test_health_reports_backend_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "backend": "ok",
        "llama": "unknown",
        "model": "qwen3.5-9b",
        "context_size": None,
    }


def test_health_reflects_injected_config() -> None:
    config = AppConfig(llm=LLMConfig(model="qwen3.6-27b"))
    with TestClient(create_app(config)) as test_client:
        assert test_client.get("/health").json()["model"] == "qwen3.6-27b"


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    response = client.get("/health")
    request_id = response.headers["X-Request-Id"]
    assert request_id
    assert len(request_id) == 36  # uuid4 string form


def test_incoming_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-Id": "abc-123"})
    assert response.headers["X-Request-Id"] == "abc-123"
