"""Tests for ``GET /health`` (ENGINEERING_PLAN.md §16, §5 DoD)."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from tutor.config import AppConfig, LLMConfig
from tutor.llm.client import LlamaClient
from tutor.main import create_app

Factory = Callable[..., LlamaClient]


def test_backend_ok_and_llama_down_when_unreachable(client: TestClient) -> None:
    """The API stays up and honest while the model backend is gone."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "backend": "ok",
        "llama": "down",
        "model": "qwen3.5-9b",
        "context_size": None,
    }


def test_llama_ok_reports_real_context_size(
    test_config: AppConfig, llama_client_factory: Factory
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={"model_alias": "qwen3.5-9b", "default_generation_settings": {"n_ctx": 65536}},
        )

    llama = llama_client_factory(handler)
    with TestClient(create_app(test_config, llama_client=llama)) as test_client:
        assert test_client.get("/health").json() == {
            "backend": "ok",
            "llama": "ok",
            "model": "qwen3.5-9b",
            "context_size": 65536,
        }


def test_health_reflects_injected_config(
    llama_client_factory: Factory,
    offline_llama: LlamaClient,
) -> None:
    config = AppConfig(llm=LLMConfig(base_url="http://llama.test", model="qwen3.6-27b"))
    with TestClient(create_app(config, llama_client=offline_llama)) as test_client:
        assert test_client.get("/health").json()["model"] == "qwen3.6-27b"


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    response = client.get("/health")
    request_id = response.headers["X-Request-Id"]
    assert request_id
    assert len(request_id) == 36  # uuid4 string form


def test_incoming_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-Id": "abc-123"})
    assert response.headers["X-Request-Id"] == "abc-123"
