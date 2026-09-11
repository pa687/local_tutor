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
from tutor.llm.client import LlamaClient
from tutor.llm.prompts import get_prompt_library
from tutor.main import create_app
from tutor.tutor.engine import ConversationRef, StudentRef, TutorEngine
from tutor.tutor.response import Confidence, Subject, TutorMode, TutorResponse

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


# --------------------------------------------------------------------- Phase 2


MATH_MESSAGE = "解方程 x^2 - 5x + 6 = 0"
MATH_REQUEST = {
    "student_id": "integration-student",
    "conversation_id": "integration-conversation",
    "mode": "tutor",
    "grade": 9,
    "message": MATH_MESSAGE,
}


def test_chat_exposes_structured_metadata_against_a_live_model(
    live_client: TestClient,
) -> None:
    """Phase 2 DoD: the three modes return a structured TutorResponse, for real."""
    response = live_client.post("/api/chat", json=MATH_REQUEST)
    assert response.status_code == 200

    events = parse_events(response.text)
    start = dict(events[0][1])
    done = dict(events[-1][1])

    assert start["mode"] == "tutor"
    assert start["subject"] in {subject.value for subject in Subject}
    assert start["confidence"] in {level.value for level in Confidence}
    assert start["estimated_grade"] == 9
    assert done["confidence"] in {level.value for level in Confidence}
    assert isinstance(done["completion_tokens"], int)
    assert done["completion_tokens"] > 0
    # The classification call is real: a maths question must not come back unknown.
    assert start["subject"] != Subject.UNKNOWN.value


@pytest.mark.parametrize("mode", ["tutor", "explain", "check"])
def test_every_mode_answers_against_a_live_model(live_client: TestClient, mode: str) -> None:
    response = live_client.post("/api/chat", json={**MATH_REQUEST, "mode": mode})
    assert response.status_code == 200
    events = parse_events(response.text)
    assert events[0][1]["mode"] == mode
    assert [name for name, _ in events][-1] == "done"


async def test_tutor_engine_responds_with_a_structured_response() -> None:
    """The engine's §6 core interface, exercised without an HTTP layer."""
    config = get_config()
    client = LlamaClient(config.llm)
    try:
        engine = TutorEngine(client, get_prompt_library(), config)
        response = await engine.respond(
            StudentRef(id="integration-student", grade=9),
            ConversationRef(id="integration-conversation"),
            MATH_MESSAGE,
            mode=TutorMode.TUTOR,
        )
    finally:
        await client.aclose()

    assert isinstance(response, TutorResponse)
    assert response.answer.strip()
    assert response.subject in set(Subject)
    assert response.estimated_grade == 9
    assert "只给第一步提示" not in response.warnings  # rules go to the model, not the client
