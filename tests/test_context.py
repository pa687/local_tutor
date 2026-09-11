"""Tests for context assembly (ENGINEERING_PLAN.md §10, §12, §13).

Two things are contract here: the layers appear in the §10 order, and the §12 rules
stay in the policy layer instead of leaking into the prompt files (Phase 2 DoD).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.config import MemoryConfig
from tutor.llm.context import ContextBuilder
from tutor.llm.models import ChatMessage
from tutor.llm.prompts import PromptLibrary
from tutor.tutor.policy import TutorPolicy
from tutor.tutor.response import Subject, TutorMode

POLICY_BLOCK = "本轮教学要求（由策略层生成，优先级高于通用作答要求）：\n1. 只给第一步提示。"


@pytest.fixture
def builder(prompts: PromptLibrary) -> ContextBuilder:
    return ContextBuilder(prompts, MemoryConfig(recent_turns=30))


def history_of(turns: int) -> tuple[ChatMessage, ...]:
    messages: list[ChatMessage] = []
    for index in range(turns):
        messages.append(ChatMessage(role="user", content=f"问{index}"))
        messages.append(ChatMessage(role="assistant", content=f"答{index}"))
    return tuple(messages)


class TestLayerOrder:
    def test_layers_appear_in_the_order_of_the_plan(
        self, builder: ContextBuilder, prompts: PromptLibrary
    ) -> None:
        messages = builder.build(
            message="当前问题",
            policy_block=POLICY_BLOCK,
            history=history_of(1),
        )
        assert [message.role for message in messages] == ["system", "user", "assistant", "user"]

        system = messages[0].content
        assert system.startswith(prompts.get("tutor_system.md"))
        assert system.index(prompts.get("tutor_system.md")) < system.index(prompts.get("solve.md"))
        assert system.index(prompts.get("solve.md")) < system.index("本轮教学要求")
        assert messages[-1].content == "当前问题"

    def test_the_system_layers_share_one_message(self, builder: ContextBuilder) -> None:
        """Qwen3.5's template aborts on a system message after a user/assistant one."""
        messages = builder.build(
            message="当前问题",
            policy_block=POLICY_BLOCK,
            history=history_of(1),
        )
        assert [message.role for message in messages].count("system") == 1
        assert messages[0].role == "system"

    def test_empty_policy_block_is_skipped(self, builder: ContextBuilder) -> None:
        messages = builder.build(message="问题", policy_block="   ")
        assert [message.role for message in messages] == ["system", "user"]
        assert messages[0].content.count("本轮教学要求") == 0

    def test_turn_without_history_still_layers_everything_into_the_system_message(
        self, builder: ContextBuilder
    ) -> None:
        messages = builder.build(message="问题", policy_block=POLICY_BLOCK)
        assert len(messages) == 2
        assert messages[-1].content == "问题"

    def test_recent_turns_keeps_only_the_latest_messages(self, prompts: PromptLibrary) -> None:
        builder = ContextBuilder(prompts, MemoryConfig(recent_turns=4))
        messages = builder.build(
            message="当前问题",
            policy_block=POLICY_BLOCK,
            history=history_of(10),
        )
        conversation = [message.content for message in messages[1:]]
        assert conversation == ["问8", "答8", "问9", "答9", "当前问题"]

    def test_prompt_edits_reach_the_context_without_code_changes(self, tmp_path: Path) -> None:
        (tmp_path / "tutor_system.md").write_text("第一版系统提示", encoding="utf-8")
        (tmp_path / "solve.md").write_text("第一版作答要求", encoding="utf-8")
        builder = ContextBuilder(PromptLibrary(tmp_path), MemoryConfig())

        first = builder.build(message="问题", policy_block="")
        assert first[0].content.startswith("第一版系统提示")

        (tmp_path / "tutor_system.md").write_text("第二版系统提示", encoding="utf-8")
        second = builder.build(message="问题", policy_block="")
        assert second[0].content.startswith("第二版系统提示")


class TestPolicyStaysOutOfPrompts:
    """Phase 2 DoD: the §12 rules must live in code, not in a prompt file."""

    @pytest.mark.parametrize("mode", list(TutorMode))
    def test_no_policy_instruction_is_copied_into_a_prompt_file(
        self, prompts: PromptLibrary, mode: TutorMode
    ) -> None:
        for message in ("这题怎么做", "我算出来 x=2，3，对吗？", "给我完整解答"):
            decision = TutorPolicy().decide(
                mode=mode, subject=Subject.MATH, grade=9, message=message
            )
            for filename in ("tutor_system.md", "solve.md"):
                content = prompts.get(filename)
                for instruction in decision.instructions:
                    assert instruction not in content, f"{instruction!r} leaked into {filename}"

    def test_distinctive_rule_phrases_are_absent_from_the_prompts(
        self, prompts: PromptLibrary
    ) -> None:
        system = prompts.get("tutor_system.md")
        for phrase in (
            "只给第一步提示",
            "把提示提高一个层级",
            "不要直接展示完整答案",
            "不要使用以下超出该年级的方法",
            "学生已经给出自己的思路",
            "不要重新给出完整标准答案",
        ):
            assert phrase not in system
