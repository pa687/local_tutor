"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tutor.config import AppConfig, load_config, reset_config_cache
from tutor.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]


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
def client(repo_config: AppConfig) -> Iterator[TestClient]:
    with TestClient(create_app(repo_config)) as test_client:
        yield test_client
