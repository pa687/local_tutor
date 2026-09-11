"""Configuration loading for Local Tutor (ENGINEERING_PLAN.md §19).

Single source of truth is ``config.yaml`` at the repository root, overridable by
environment variables shaped like ``LOCAL_TUTOR__<SECTION>__<KEY>``.

No model paths, checkpoint names or user-specific absolute paths are hardcoded
here (§1.5); the defaults below mirror the safe values shipped in ``config.yaml``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "LOCAL_TUTOR__"
CONFIG_PATH_ENV = "TUTOR_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be read, parsed, or validated."""


class SectionModel(BaseModel):
    """Base for config sections: unknown keys rejected, values immutable once loaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMConfig(SectionModel):
    """llama-server endpoint settings."""

    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen3.5-9b"
    timeout: float = Field(default=180.0, gt=0)


class TutorConfig(SectionModel):
    """Tutor policy settings."""

    default_mode: str = "tutor"
    max_verification_attempts: int = Field(default=2, ge=0, le=10)


class MemoryConfig(SectionModel):
    """Conversation memory settings."""

    recent_turns: int = Field(default=30, gt=0, le=200)


class RetrievalConfig(SectionModel):
    """Textbook retrieval settings; disabled by default (§11)."""

    enabled: bool = False
    top_k: int = Field(default=6, gt=0, le=50)


class AppConfig(SectionModel):
    """The whole application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tutor: TutorConfig = Field(default_factory=TutorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def _read_yaml_file(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - exercised via a mocked failure mode
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping at the top level")
    return {str(key): value for key, value in parsed.items()}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Translate ``LOCAL_TUTOR__A__B=value`` into ``{"a": {"b": value}}``."""
    overrides: dict[str, Any] = {}
    for name, raw_value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            raise ConfigError(f"environment variable {name} does not name a config key")
        try:
            value: Any = yaml.safe_load(raw_value) if raw_value.strip() else ""
        except yaml.YAMLError as exc:
            message = f"environment variable {name} has an unparsable value: {exc}"
            raise ConfigError(message) from exc
        cursor = overrides
        for part in path[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(
                    f"environment variable {name} conflicts with another override on '{part}'"
                )
            cursor = child
        cursor[path[-1]] = value
    return overrides


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Read ``config.yaml``, apply environment overrides, then validate.

    ``path`` wins over ``TUTOR_CONFIG_PATH``; both win over the repository default.
    Raises :class:`ConfigError` for a missing file, bad YAML, or invalid values.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    raw_path = str(path) if path is not None else env.get(CONFIG_PATH_ENV)
    config_path = Path(raw_path).expanduser() if raw_path else DEFAULT_CONFIG_PATH

    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    data = _deep_merge(_read_yaml_file(config_path), _env_overrides(env))
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration."""
    return load_config()


def reset_config_cache() -> None:
    """Drop the cache so the next :func:`get_config` re-reads files and environment."""
    get_config.cache_clear()
