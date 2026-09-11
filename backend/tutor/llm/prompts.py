"""Prompt loading (ENGINEERING_PLAN.md §4, §13).

Every prompt lives in ``prompts/*.md`` at the repository root and is read from disk
on each access, so editing a prompt file takes effect without restarting the process
and without touching Python code (Phase 2 DoD).

§13 forbids embedding long prompt strings in Python; this module therefore contains
no prompt text at all, only the machinery that finds, reads and validates files.

The directory resolves in this order:

1. the ``directory`` argument (used by tests);
2. ``$TUTOR_PROMPTS_DIR``;
3. ``<repo root>/prompts``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR_ENV = "TUTOR_PROMPTS_DIR"
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

#: Prompt files the code base expects to exist. Keeping the list explicit turns a
#: typo in a filename into a startup-time error instead of a silent empty prompt.
REQUIRED_PROMPTS: tuple[str, ...] = (
    "tutor_system.md",
    "solve.md",
    "classify.md",
    "verify.md",
    "summarize_student.md",
)


class PromptError(RuntimeError):
    """Base class for prompt-loading problems."""


class PromptNotFoundError(PromptError):
    """A required prompt file is missing or unreadable."""


class PromptLibrary:
    """Reads prompt files from a directory, one read per access."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        raw = directory if directory is not None else os.environ.get(PROMPTS_DIR_ENV)
        self._directory = Path(raw).expanduser() if raw else DEFAULT_PROMPTS_DIR

    @property
    def directory(self) -> Path:
        return self._directory

    def path(self, name: str) -> Path:
        """Absolute path of ``name`` inside the prompt directory."""
        return self._directory / name

    def get(self, name: str) -> str:
        """Return the contents of ``name``.

        Raises :class:`PromptNotFoundError` when the file is missing, empty, or
        unreadable — never fall back to a hardcoded default (§23.8).
        """
        path = self.path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptNotFoundError(f"cannot read prompt {name!r} from {path}: {exc}") from exc
        if not text.strip():
            raise PromptNotFoundError(f"prompt {name!r} at {path} is empty")
        return text.strip()

    def describe(self) -> str:
        """Human-readable summary used in startup logs and error reports."""
        present = [name for name in REQUIRED_PROMPTS if self.path(name).is_file()]
        return f"{len(present)}/{len(REQUIRED_PROMPTS)} required prompts found in {self._directory}"


@lru_cache(maxsize=1)
def get_prompt_library() -> PromptLibrary:
    """Process-wide prompt library (cached directory resolution only, no file cache)."""
    return PromptLibrary()


def reset_prompt_library_cache() -> None:
    """Drop the cached library so the next :func:`get_prompt_library` re-reads the env."""
    get_prompt_library.cache_clear()
