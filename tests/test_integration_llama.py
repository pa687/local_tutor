"""Integration tests against a real llama-server (ENGINEERING_PLAN.md §5).

Skipped by default so the normal suite stays hermetic; the Phase 1 DoD is not
satisfied by mocks, so these are run for real before the phase is called done:

    scripts/start_llama.sh &
    LLAMA_INTEGRATION=1 uv run pytest -m integration -v

The "llama-server restarted and the backend recovered" DoD item is verified
externally (stop the server, watch ``/health`` report ``down`` and ``/api/chat``
return 503, restart the server, watch both recover without restarting the backend);
the recorded run lives in ``docs/phase1_verification.md``. The client-side half of
that guarantee — no sticky failure state — is covered deterministically in
``tests/test_recovery.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tutor.config import AppConfig, LLMConfig, get_config
from tutor.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("LLAMA_INTEGRATION") != "1",
        reason="set LLAMA_INTEGRATION=1 to run against a live llama-server",
    ),
]

CHAT_REQUEST = {
    "student_id": "integration-student",
    "conversation_id": "integration-conversation",
    "mode": "tutor",
    "message": "Reply with exactly: pong",
}


@pytest.fixture
def live_client() -> Iterator[TestClient]:
    with TestClient(create_app(get_config())) as client:
        yield client


def parse_events(payload: str) -> list[tuple[str, dict[str, object]]]:
    parsed: list[tuple[str, dict[str, object]]] = []
    for block in payload.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "{}")
        parsed.append((event, json.loads(data)))
    return parsed


def test_health_confirms_llama_is_online(live_client: TestClient) -> None:
    payload = live_client.get("/health").json()
    assert payload["backend"] == "ok"
    assert payload["llama"] == "ok"
    assert isinstance(payload["context_size"], int)
    assert payload["context_size"] > 0


def test_plain_chat_streams_real_tokens(live_client: TestClient) -> None:
    response = live_client.post("/api/chat", json=CHAT_REQUEST)
    assert response.status_code == 200
    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    text = "".join(str(data.get("text", "")) for name, data in events if name == "token")
    assert text.strip()
    done = events[-1][1]
    # §18: the trailing usage chunk must reach the client, not be dropped.
    assert isinstance(done["completion_tokens"], int)
    assert done["completion_tokens"] > 0
    assert isinstance(done["prompt_tokens"], int)
    assert done["prompt_tokens"] > 0


def test_answer_arrives_as_more_than_one_token_event(live_client: TestClient) -> None:
    request = {**CHAT_REQUEST, "message": "Count from 1 to 20, one number per line."}
    with live_client.stream("POST", "/api/chat", json=request) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    tokens = [name for name, _ in parse_events(body) if name == "token"]
    assert len(tokens) > 1


def test_unreachable_server_does_not_break_the_backend() -> None:
    """A dead llama-server must yield 503, not an exception, and the API stays up."""
    config = AppConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1", model="qwen3.5-9b", timeout=2.0)
    )
    with TestClient(create_app(config)) as client:
        assert client.post("/api/chat", json=CHAT_REQUEST).status_code == 503
        assert client.get("/health").json()["llama"] == "down"
