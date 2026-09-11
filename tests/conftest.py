"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tutor.config import AppConfig, LLMConfig, load_config, reset_config_cache
from tutor.llm.client import LlamaClient
from tutor.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_BASE_URL = "http://llama.test"
Handler = Callable[[httpx.Request], httpx.Response]


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
def offline_llama(llama_client_factory: Callable[..., LlamaClient]) -> LlamaClient:
    """A llama-server that refuses every connection (the Phase 1 'crash' case)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return llama_client_factory(handler)


@pytest.fixture
def client(test_config: AppConfig, offline_llama: LlamaClient) -> Iterator[TestClient]:
    with TestClient(create_app(test_config, llama_client=offline_llama)) as test_client:
        yield test_client
