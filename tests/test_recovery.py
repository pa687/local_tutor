"""Regression test for Phase 1 DoD: a backend must survive a backend outage.

The real restart drill (stop llama-server, watch /health go ``down`` and ``/api/chat``
return 503, restart it, watch both recover without restarting the app) is recorded in
``docs/phase1_verification.md``. This test pins the property that makes it work: the
client keeps no cached health or connection state, so the very next request after the
outage goes through.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from tutor.config import AppConfig
from tutor.llm.client import LlamaClient
from tutor.main import create_app

Factory = Callable[..., LlamaClient]
REQUEST = {
    "student_id": "student-001",
    "conversation_id": "abc",
    "mode": "tutor",
    "message": "ping",
}


def build_flaky_transport() -> tuple[httpx.MockTransport, dict[str, bool]]:
    """Refuse connections until the returned flag is set (a llama-server restart)."""
    state = {"up": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_alias": "qwen3.5-9b", "n_ctx": 65536})
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    return httpx.MockTransport(handler), state


def test_backend_recovers_when_llama_server_comes_back(
    test_config: AppConfig, llama_client_factory: Factory
) -> None:
    transport, state = build_flaky_transport()
    llama = LlamaClient(
        test_config.llm,
        client=httpx.AsyncClient(base_url=test_config.llm.base_url, transport=transport),
    )

    with TestClient(create_app(test_config, llama_client=llama)) as client:
        # 1. Down: the API stays up, /health is honest, chat fails cleanly.
        assert client.get("/health").json()["llama"] == "down"
        assert client.post("/api/chat", json=REQUEST).status_code == 503

        # 2. llama-server comes back (no backend restart, no cache invalidation).
        state["up"] = True

        assert client.get("/health").json()["llama"] == "ok"
        response = client.post("/api/chat", json=REQUEST)
        assert response.status_code == 200
        assert '"text": "pong"' in response.text
