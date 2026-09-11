"""Tests for prompt loading (ENGINEERING_PLAN.md §4, §13).

§13 requires the prompts to live in ``prompts/*.md`` and to be editable without
touching Python. These tests pin both halves: the files exist and parse, and a
changed file is picked up without restarting anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tutor.llm import prompts as prompts_module
from tutor.llm.prompts import (
    DEFAULT_PROMPTS_DIR,
    PROMPTS_DIR_ENV,
    REQUIRED_PROMPTS,
    PromptLibrary,
    PromptNotFoundError,
    get_prompt_library,
    reset_prompt_library_cache,
)


@pytest.fixture(autouse=True)
def _reset_prompt_cache() -> Iterator[None]:
    reset_prompt_library_cache()
    yield
    reset_prompt_library_cache()


class TestRepositoryPrompts:
    @pytest.mark.parametrize("name", REQUIRED_PROMPTS)
    def test_required_prompt_exists_and_is_not_empty(
        self, prompts: PromptLibrary, name: str
    ) -> None:
        assert prompts.get(name).strip()

    def test_default_directory_is_the_repository_prompts_dir(self, repo_root: Path) -> None:
        assert DEFAULT_PROMPTS_DIR == repo_root / "prompts"

    def test_tutor_system_prompt_has_no_placeholder_left(self, prompts: PromptLibrary) -> None:
        assert "PLACEHOLDER" not in prompts.get("tutor_system.md").upper()

    def test_system_prompt_is_short_and_hard(self, prompts: PromptLibrary) -> None:
        """§13: no 'you are the smartest teacher' filler eating the context."""
        content = prompts.get("tutor_system.md")
        assert "smartest teacher" not in content.lower()
        assert len(content) < 1000
        for principle in ("准确性高于流畅性", "不知道就明确说不知道", "不要隐藏工具检测出的冲突"):
            assert principle in content

    def test_describe_reports_all_required_prompts(self, prompts: PromptLibrary) -> None:
        assert prompts.describe().startswith(f"{len(REQUIRED_PROMPTS)}/{len(REQUIRED_PROMPTS)}")


class TestLoading:
    def test_reads_from_an_explicit_directory(self, tmp_path: Path) -> None:
        (tmp_path / "solve.md").write_text("内容", encoding="utf-8")
        library = PromptLibrary(tmp_path)
        assert library.get("solve.md") == "内容"
        assert library.directory == tmp_path

    def test_missing_file_raises_instead_of_defaulting(self, tmp_path: Path) -> None:
        library = PromptLibrary(tmp_path)
        with pytest.raises(PromptNotFoundError, match="solve.md"):
            library.get("solve.md")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "solve.md").write_text("   \n", encoding="utf-8")
        with pytest.raises(PromptNotFoundError, match="empty"):
            library = PromptLibrary(tmp_path)
            library.get("solve.md")

    def test_edited_prompt_is_reread_without_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "solve.md"
        path.write_text("第一版", encoding="utf-8")
        library = PromptLibrary(tmp_path)
        assert library.get("solve.md") == "第一版"

        path.write_text("第二版", encoding="utf-8")
        assert library.get("solve.md") == "第二版"

    def test_environment_variable_selects_the_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "solve.md").write_text("来自环境变量", encoding="utf-8")
        monkeypatch.setenv(PROMPTS_DIR_ENV, str(tmp_path))
        assert get_prompt_library().get("solve.md") == "来自环境变量"

    def test_default_directory_used_when_environment_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(PROMPTS_DIR_ENV, raising=False)
        assert get_prompt_library().directory == prompts_module.DEFAULT_PROMPTS_DIR
