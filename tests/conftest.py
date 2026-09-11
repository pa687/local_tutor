"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tutor.config import AppConfig, LLMConfig, load_config, reset_config_cache
from tutor.llm.client import LlamaClient
from tutor.llm.prompts import PromptLibrary
from tutor.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_BASE_URL = "http://llama.test"
Handler = Callable[[httpx.Request], httpx.Response]

#: A clean classifier reply, i.e. what a well-behaved model returns.
CLASSIFICATION_JSON = json.dumps(
    {
        "subject": "math",
        "grade": 9,
        "topic": "quadratic_equations",
        "uncertain": False,
    }
)


def sse_content_chunk(text: str, finish: str | None = None) -> str:
    """One streaming chunk carrying answer text."""
    payload = {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n"


def sse_final_chunks(prompt_tokens: int, completion_tokens: int) -> str:
    """The two trailing chunks llama-server sends: finish reason, then usage."""
    finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return f"data: {json.dumps(finish)}\n\ndata: {json.dumps(usage)}\n\n"


@pytest.fixture(autouse=True)
def _clear_config_cache() -> Iterator[None]:
    """Keep the process-wide config cache from leaking between tests."""
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def prompts() -> PromptLibrary:
    """The repository's real prompt files (§13)."""
    return PromptLibrary(REPO_ROOT / "prompts")


@pytest.fixture
def repo_config() -> AppConfig:
    """Configuration loaded from the repository's own ``config.yaml``."""
    return load_config(REPO_ROOT / "config.yaml", environ={})


@pytest.fixture
def test_config() -> AppConfig:
    """Deterministic configuration pointing at a fake llama-server."""
    return AppConfig(llm=LLMConfig(base_url=TEST_BASE_URL, model="qwen3.5-9b", timeout=5.0))


@pytest.fixture
def llama_client_factory() -> Callable[..., LlamaClient]:
    """Build a real :class:`LlamaClient` backed by an httpx mock transport."""

    def factory(handler: Handler, *, timeout: float = 5.0) -> LlamaClient:
        http_client = httpx.AsyncClient(
            base_url=TEST_BASE_URL,
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(timeout),
        )
        return LlamaClient(
            LLMConfig(base_url=TEST_BASE_URL, model="qwen3.5-9b", timeout=timeout),
            client=http_client,
        )

    return factory


@pytest.fixture
def tutor_llama(llama_client_factory: Callable[..., LlamaClient]) -> Callable[..., LlamaClient]:
    """A fake llama-server that serves both call shapes the engine uses.

    Phase 2 makes two calls per turn: a non-streaming classification call and a
    streaming answer. The handler branches on the ``stream`` flag of the payload, so
    tests exercise the real :class:`LlamaClient` on both paths.
    """

    def factory(
        *,
        classification: str | None = CLASSIFICATION_JSON,
        tokens: Sequence[str] = ("Hel", "lo"),
        prompt_tokens: int = 11,
        completion_tokens: int = 2,
        classify_error: Exception | None = None,
        stream_error: Exception | None = None,
        capture: list[dict[str, Any]] | None = None,
    ) -> LlamaClient:
        def handler(request: httpx.Request) -> httpx.Response:
            payload: dict[str, Any] = json.loads(request.content)
            if capture is not None:
                capture.append(payload)
            if payload.get("stream"):
                if isinstance(stream_error, httpx.Response):
                    return stream_error
                if stream_error is not None:
                    raise stream_error
                body = "".join(sse_content_chunk(token) for token in tokens)
                body += sse_final_chunks(prompt_tokens, completion_tokens)
                body += "data: [DONE]\n\n"
                return httpx.Response(200, text=body)
            if isinstance(classify_error, httpx.Response):
                return classify_error
            if classify_error is not None:
                raise classify_error
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": classification or ""}}]},
            )

        return llama_client_factory(handler)

    return factory


@pytest.fixture
def offline_llama(llama_client_factory: Callable[..., LlamaClient]) -> LlamaClient:
    """A llama-server that refuses every connection (the Phase 1 'crash' case)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return llama_client_factory(handler)


@pytest.fixture
def client(test_config: AppConfig, offline_llama: LlamaClient) -> Iterator[TestClient]:
    with TestClient(create_app(test_config, llama_client=offline_llama)) as test_client:
        yield test_client
