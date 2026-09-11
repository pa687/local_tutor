"""Endpoint boundaries (ENGINEERING_PLAN.md §6, §12, §13).

§6 is explicit: the endpoint must not assemble prompts and must not talk to
``LlamaClient``; §12 is explicit that the policy rules belong to the policy layer.
Those rules are only worth anything if they are checked, so this module walks the
``api`` package with AST and with plain text.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "backend" / "tutor" / "api"

#: Methods that answer with the model. Only the engine may call them (§6).
MODEL_METHODS = {"stream_chat", "complete"}

#: Attributes that would hand an endpoint the shared llama client.
CLIENT_ATTRIBUTES = {"llama"}

#: Distinctive §12 rule phrases. They must never appear in an endpoint.
POLICY_PHRASES = (
    "只给第一步提示",
    "把提示提高一个层级",
    "不要直接展示完整答案",
    "不要使用以下超出该年级的方法",
    "不要重新给出完整标准答案",
)


def api_modules() -> list[Path]:
    return sorted(API_DIR.glob("*.py"))


def test_api_package_is_scanned() -> None:
    assert [path.name for path in api_modules()] == [
        "__init__.py",
        "chat.py",
        "health.py",
        "students.py",
    ]


@pytest.mark.parametrize("path", api_modules(), ids=lambda path: path.name)
def test_no_endpoint_calls_the_model_directly(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        f"{path.name}:{node.lineno}:{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in MODEL_METHODS
    ]
    assert offenders == []


def test_chat_endpoint_never_touches_the_llama_client() -> None:
    """``/api/chat`` must go through the engine; ``/health`` keeps its probe."""
    tree = ast.parse((API_DIR / "chat.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in CLIENT_ATTRIBUTES
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("path", api_modules(), ids=lambda path: path.name)
def test_no_prompt_text_is_embedded_in_an_endpoint(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    prompt_dir = REPO_ROOT / "prompts"
    for prompt_file in sorted(prompt_dir.glob("*.md")):
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if len(candidate) >= 8 and not candidate.startswith(("<!--", "#", "-", "```", "{")):
                assert candidate not in source, f"{prompt_file.name} line leaked into {path.name}"


@pytest.mark.parametrize("path", api_modules(), ids=lambda path: path.name)
def test_no_policy_rule_is_restated_in_an_endpoint(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for phrase in POLICY_PHRASES:
        assert phrase not in source, f"policy rule leaked into {path.name}"
