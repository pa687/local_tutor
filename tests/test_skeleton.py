"""Structural guards for the repository skeleton.

These tests make the layout DoD executable:

* the repository layout matches ``ENGINEERING_PLAN.md`` §4;
* modules that belong to later phases are still pure placeholders (no logic leaked
  into an earlier phase);
* the §23.7 red line holds — no ``eval`` / ``exec`` / ``subprocess`` anywhere in the
  backend or the test suite (dev scripts under ``scripts/`` are exempt: they only
  call nvidia-smi);
* the §7 tool package imports maths and nothing else.
* the §7 tool package imports maths and nothing else.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend" / "tutor"

EXPECTED_PATHS = [
    "README.md",
    "ENGINEERING_PLAN.md",
    "pyproject.toml",
    "config.yaml",
    ".env.example",
    "backend/tutor/main.py",
    "backend/tutor/api/chat.py",
    "backend/tutor/api/students.py",
    "backend/tutor/api/health.py",
    "backend/tutor/llm/client.py",
    "backend/tutor/llm/models.py",
    "backend/tutor/llm/prompts.py",
    "backend/tutor/llm/context.py",
    "backend/tutor/tutor/engine.py",
    "backend/tutor/tutor/policy.py",
    "backend/tutor/tutor/classifier.py",
    "backend/tutor/tutor/verifier.py",
    "backend/tutor/tutor/response.py",
    "backend/tutor/tools/registry.py",
    "backend/tutor/tools/sandbox.py",
    "backend/tutor/tools/calculator.py",
    "backend/tutor/tools/algebra.py",
    "backend/tutor/tools/units.py",
    "backend/tutor/memory/student.py",
    "backend/tutor/memory/conversation.py",
    "backend/tutor/memory/summarizer.py",
    "backend/tutor/retrieval/ingest.py",
    "backend/tutor/retrieval/chunk.py",
    "backend/tutor/retrieval/search.py",
    "backend/tutor/retrieval/models.py",
    "backend/tutor/db/models.py",
    "backend/tutor/db/session.py",
    "prompts/tutor_system.md",
    "prompts/solve.md",
    "prompts/classify.md",
    "prompts/verify.md",
    "prompts/summarize_student.md",
    "eval/runner.py",
    "eval/graders.py",
    "scripts/bench_llama.py",
    "scripts/start_llama.sh",
    "scripts/benchmark_model.sh",
    "scripts/smoke_test.sh",
]


@pytest.mark.parametrize("relative_path", EXPECTED_PATHS)
def test_expected_layout_exists(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).exists(), f"missing {relative_path}"


@pytest.mark.parametrize("dir_path", ["frontend", "eval/datasets", "eval/reports", "tests"])
def test_expected_directories_exist(dir_path: str) -> None:
    assert (REPO_ROOT / dir_path).is_dir()


def python_modules() -> list[Path]:
    return sorted(BACKEND.rglob("*.py"))


def placeholder_modules() -> list[Path]:
    """Modules whose docstring declares them placeholders for a later phase."""
    found = []
    for path in python_modules():
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        if docstring.startswith(("Placeholder", "Reserved")):
            found.append(path)
    return found


def test_placeholders_exist_for_later_phases() -> None:
    # Phase 2 turned six of them into real modules, Phase 3 three more (registry,
    # calculator, algebra plus the new sandbox), Phase 4 the verifier; the rest belong
    # to Phases 6–9.
    assert len(placeholder_modules()) >= 11


@pytest.mark.parametrize("script", ["start_llama.sh", "benchmark_model.sh", "smoke_test.sh"])
def test_shell_scripts_are_executable(script: str) -> None:
    path = REPO_ROOT / "scripts" / script
    assert path.stat().st_mode & 0o111, f"{script} is not executable"


#: Modules a tool implementation may import. §7 allows structured maths and nothing
#: else: no shell, no network, no filesystem, no dynamic execution.
TOOL_ALLOWED_IMPORTS = {
    "__future__",
    "ast",
    "collections",
    "concurrent",
    "dataclasses",
    "enum",
    "logging",
    "math",
    "pydantic",
    "re",
    "sympy",
    "time",
    "typing",
}


def test_tool_modules_import_nothing_dangerous() -> None:
    """§7: the tool package does maths through SymPy and reaches for nothing else."""
    offenders: list[str] = []
    for path in sorted((BACKEND / "tools").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root and root not in TOOL_ALLOWED_IMPORTS and root != "tutor":
                    offenders.append(f"{path.name}:{root}")
    assert offenders == []


def test_placeholder_modules_contain_no_logic() -> None:
    """Phase 0 must not smuggle in business logic behind a placeholder docstring."""
    offenders = []
    for path in placeholder_modules():
        body = ast.parse(path.read_text(encoding="utf-8")).body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_no_eval_exec_or_subprocess_anywhere() -> None:
    """§23.7: model-generated content must never be executed.

    ``tests/test_integration_llama.py`` is the single exemption: restarting
    llama-server for the Phase 1 DoD genuinely needs a subprocess, and it never runs
    model output.
    """
    exempt = {"tests/test_integration_llama.py"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    offenders: list[str] = []
    scanned = [
        path
        for path in [*python_modules(), *(REPO_ROOT / "tests").rglob("*.py")]
        if str(path.relative_to(REPO_ROOT)) not in exempt
    ]
    for path in scanned:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden_calls:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.func.id}")
            if isinstance(node, ast.Import) and any(
                alias.name in {"subprocess", "os.system", "pty"} for alias in node.names
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"system", "popen"}:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.func.attr}"
                    )
    assert offenders == []


def test_engineering_plan_is_the_finalised_plan(repo_root: Path) -> None:
    """ENGINEERING_PLAN.md must be the finalised temp.md, not a paraphrase."""
    plan = (repo_root / "ENGINEERING_PLAN.md").read_text(encoding="utf-8")
    assert plan.startswith("# Local Tutor — 工程实施计划")
    assert "# 27. 项目成功标准" in plan


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Ids of the string constants that are docstrings, i.e. allowed to be long."""
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def test_no_long_prompt_literals_in_python() -> None:
    """§13: prompt text belongs in ``prompts/*.md``, not in a Python literal."""
    limit = 400
    offenders: list[str] = []
    for path in python_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in allowed
                and len(node.value) > limit
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []
