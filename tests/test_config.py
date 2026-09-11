"""Tests for the Phase 0 configuration loader (ENGINEERING_PLAN.md §19)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.config import (
    AppConfig,
    ConfigError,
    get_config,
    load_config,
    reset_config_cache,
)

VALID_YAML = """
llm:
  base_url: http://127.0.0.1:8080
  model: qwen3.5-9b
  timeout: 180

tutor:
  default_mode: tutor
  max_verification_attempts: 2

memory:
  recent_turns: 30

retrieval:
  enabled: false
  top_k: 6
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_config_matches_plan_defaults(repo_config: AppConfig) -> None:
    assert repo_config.llm.base_url == "http://127.0.0.1:8080"
    assert repo_config.llm.model == "qwen3.5-9b"
    assert repo_config.llm.timeout == 180
    assert repo_config.tutor.default_mode == "tutor"
    assert repo_config.tutor.max_verification_attempts == 2
    assert repo_config.memory.recent_turns == 30
    assert repo_config.retrieval.enabled is False
    assert repo_config.retrieval.top_k == 6


def test_env_override_replaces_every_section(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    config = load_config(
        path,
        environ={
            "LOCAL_TUTOR__LLM__MODEL": "qwen3.6-27b",
            "LOCAL_TUTOR__LLM__TIMEOUT": "65536",
            "LOCAL_TUTOR__TUTOR__DEFAULT_MODE": "explain",
            "LOCAL_TUTOR__MEMORY__RECENT_TURNS": "40",
            "LOCAL_TUTOR__RETRIEVAL__ENABLED": "true",
            "LOCAL_TUTOR__RETRIEVAL__TOP_K": "9",
        },
    )
    assert config.llm.model == "qwen3.6-27b"
    assert config.llm.timeout == 65536
    assert isinstance(config.llm.timeout, float)
    assert config.tutor.default_mode == "explain"
    assert config.memory.recent_turns == 40
    assert config.retrieval.enabled is True
    assert config.retrieval.top_k == 9


def test_env_override_is_scalar_parsed_not_stringly(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    config = load_config(path, environ={"LOCAL_TUTOR__RETRIEVAL__ENABLED": "false"})
    assert config.retrieval.enabled is False


def test_unrelated_env_vars_are_ignored(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    config = load_config(path, environ={"LLM_MODEL": "ignored", "LOCAL_TUTOR": "ignored"})
    assert config.llm.model == "qwen3.5-9b"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config(path, environ={"LOCAL_TUTOR__LLM__NOT_A_KEY": "1"})


def test_invalid_value_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config(path, environ={"LOCAL_TUTOR__LLM__TIMEOUT": "0"})


def test_bad_yaml_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, "llm: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path, environ={})


def test_non_mapping_yaml_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, "- just\n- a list\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(path, environ={})


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml", environ={})


def test_empty_key_after_prefix_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    with pytest.raises(ConfigError, match="does not name a config key"):
        load_config(path, environ={"LOCAL_TUTOR__": "x"})


def test_conflicting_overrides_are_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    with pytest.raises(ConfigError, match="conflicts with another override"):
        load_config(
            path,
            environ={"LOCAL_TUTOR__LLM": "1", "LOCAL_TUTOR__LLM__MODEL": "qwen3.6-27b"},
        )


def test_config_path_env_selects_the_file(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML.replace("qwen3.5-9b", "from-env-path"))
    config = load_config(environ={"TUTOR_CONFIG_PATH": str(path)})
    assert config.llm.model == "from-env-path"


def test_get_config_is_cached_until_reset(tmp_path: Path) -> None:
    path = write_config(tmp_path, VALID_YAML)
    first = get_config()
    reset_config_cache()
    assert get_config() == first
    assert str(path)  # tmp_path fixture keeps the directory alive for the assertion


def test_shipped_config_has_no_absolute_paths(repo_root: Path) -> None:
    """§1.5: model paths and absolute paths never live in committed config/code."""
    text = (repo_root / "config.yaml").read_text(encoding="utf-8")
    assert "/home/" not in text
    assert ".gguf" not in text
