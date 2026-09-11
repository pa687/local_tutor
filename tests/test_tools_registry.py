"""Tests for the tool registry (ENGINEERING_PLAN.md §7).

The registry is the contract: the eight tool names §7 fixes, one validation style, one
result shape, a timeout, and an audit trail that a UI or a log can consume.
"""

from __future__ import annotations

import json
import time

import pytest
from pydantic import BaseModel, ConfigDict

from tutor.tools.registry import (
    DEFAULT_TOOL_TIMEOUT,
    ToolCall,
    ToolErrorKind,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolTrace,
    build_default_registry,
)


class _Args(BaseModel):
    """Minimal argument model for registry-level tests."""

    model_config = ConfigDict(extra="forbid")

    expression: str


def _slow_handler(_: _Args) -> dict[str, object]:
    """Deliberately slow handler: the timeout tests must not depend on luck."""
    time.sleep(0.5)
    return {"slept": True}


def _slow_registry(**kwargs: float) -> ToolRegistry:
    registry = ToolRegistry(**kwargs)
    registry.register(
        ToolSpec(name="slow", description="d", arguments=_Args, handler=_slow_handler)
    )
    return registry


#: The tool set ENGINEERING_PLAN.md §7 fixes (the §26 list of five is a reference
#: example, not the limit — work.md Phase 3 spells this out).
SECTION_7_TOOLS = (
    "calculator",
    "solve_equation",
    "simplify_expression",
    "factor_expression",
    "differentiate",
    "integrate",
    "evaluate_expression",
    "check_equivalence",
)


class TestDefaultRegistry:
    def test_exposes_exactly_the_eight_tools_of_the_plan(self, tools: ToolRegistry) -> None:
        assert set(tools.names()) == set(SECTION_7_TOOLS)
        assert len(tools.names()) == len(SECTION_7_TOOLS)

    def test_every_tool_has_a_description_and_a_schema(self, tools: ToolRegistry) -> None:
        catalog = {entry["tool"]: entry for entry in tools.catalog()}
        assert set(catalog) == set(SECTION_7_TOOLS)
        for name, entry in catalog.items():
            assert entry["description"], name
            schema = entry["arguments"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False, name
            assert schema["properties"], name

    def test_registry_is_fresh_on_each_build(self) -> None:
        first = build_default_registry()
        second = build_default_registry()
        assert first is not second
        assert first.names() == second.names()

    def test_duplicate_registration_is_refused(self, tools: ToolRegistry) -> None:
        spec = tools.specs()[0]
        with pytest.raises(ValueError, match="already registered"):
            tools.register(spec)

    def test_duplicate_registration_can_replace(self) -> None:
        registry = build_default_registry()
        spec = registry.specs()[0]
        registry.register(spec, replace=True)
        assert len(registry.names()) == len(SECTION_7_TOOLS)

    def test_invalid_tool_name_is_refused(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="identifier"):
            registry.register(
                ToolSpec(
                    name="not a name",
                    description="x",
                    arguments=_Args,
                    handler=lambda _: {},
                )
            )

    def test_non_positive_timeout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ToolRegistry(default_timeout=0)


class TestDispatch:
    def test_unknown_tool_lists_the_available_ones(self, tools: ToolRegistry) -> None:
        result = tools.call("solve", {"equation": "x = 1"})
        assert result.error_kind is ToolErrorKind.UNKNOWN_TOOL
        assert "calculator" in (result.error or "")
        assert result.ok is False

    def test_trace_marks_a_successful_call(self, tools: ToolRegistry) -> None:
        result = tools.call("calculator", {"expression": "1+1"})
        assert result.ok is True
        assert result.tool == "calculator"
        assert result.duration_ms >= 0
        summary = result.summary()
        assert summary.startswith("✓ calculator(")
        assert "result=2" in summary

    def test_trace_marks_a_failed_call(self, tools: ToolRegistry) -> None:
        result = tools.call("calculator", {"expression": "x"})
        assert result.summary().startswith("✗ calculator(")
        assert "→" in result.summary()

    def test_arguments_are_echoed_normalised(self, tools: ToolRegistry) -> None:
        result = tools.call("differentiate", {"expression": "x^2"})
        assert result.arguments == {"expression": "x^2", "variable": "x", "order": 1}

    def test_validation_errors_name_the_field_not_the_value(self, tools: ToolRegistry) -> None:
        result = tools.call("calculator", {"expression": 5, "precision": 3})
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert "expression" in (result.error or "")
        assert "5" not in (result.error or "")

    def test_result_events_are_json_serializable(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x^2 - 5*x + 6"})
        payload = json.dumps(result.as_event(), ensure_ascii=False)
        assert '"tool": "solve_equation"' in payload
        assert '"error_kind": null' in payload


class TestStructuredCallEnvelope:
    """§7 fixes the wire format: ``{"tool": ..., "arguments": {...}}``."""

    def test_runs_a_call_written_in_the_plan_format(self, tools: ToolRegistry) -> None:
        result = tools.call_request(
            {"tool": "solve_equation", "arguments": {"equation": "x**2 - 5*x + 6"}}
        )
        assert result.ok, result.error
        assert result.output_or_raise()["solutions"] == ["2", "3"]

    def test_accepts_the_typed_envelope(self, tools: ToolRegistry) -> None:
        result = tools.call_request(ToolCall(tool="calculator", arguments={"expression": "2+2"}))
        assert result.output_or_raise()["result"] == "4"

    @pytest.mark.parametrize(
        "payload",
        [
            {"tool": "calculator"},
            {"tool": "calculator", "arguments": {"expression": "1+1"}, "extra": True},
            {"arguments": {"expression": "1+1"}},
            {"tool": "", "arguments": {}},
            "solve_equation",
            [],
        ],
    )
    def test_malformed_envelopes_are_reported(self, tools: ToolRegistry, payload: object) -> None:
        result = tools.call_request(payload)  # type: ignore[arg-type]
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert result.output is None

    def test_a_valid_envelope_with_a_bad_tool_still_reports_the_tool(
        self, tools: ToolRegistry
    ) -> None:
        result = tools.call_request({"tool": "nope", "arguments": {}})
        assert result.error_kind is ToolErrorKind.UNKNOWN_TOOL
        assert result.tool == "nope"

    def test_call_request_results_are_traceable(self, tools: ToolRegistry) -> None:
        trace = ToolTrace().with_result(
            tools.call_request({"tool": "factor_expression", "arguments": {"expression": "x^2-1"}})
        )
        assert trace.tools_used == ("factor_expression",)
        assert json.dumps(trace.as_events(), ensure_ascii=False)


class TestNeverRaises:
    @pytest.mark.parametrize("arguments", ["a string", ["a", "list"], 42, object()])
    def test_non_mapping_arguments_are_reported(
        self, tools: ToolRegistry, arguments: object
    ) -> None:
        result = tools.call("calculator", arguments)  # type: ignore[arg-type]
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert "must be an object" in (result.error or "")

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("calculator", {"expression": None}),
            ("calculator", {"expression": "1+1", "precision": [1]}),
            ("solve_equation", {"equation": {"nested": "dict"}}),
            ("differentiate", {"expression": "x", "order": "first"}),
            ("integrate", {"expression": "x", "lower": 1, "upper": 2}),
            ("check_equivalence", {"expression_a": "x", "expression_b": None}),
            ("evaluate_expression", {"expression": "x", "substitutions": {"x": 1}}),
        ],
    )
    def test_garbage_arguments_never_raise(
        self, tools: ToolRegistry, tool: str, arguments: dict[str, object]
    ) -> None:
        result = tools.call(tool, arguments)
        assert result.error_kind is ToolErrorKind.INVALID_ARGUMENTS
        assert result.output is None

    def test_a_handler_bug_is_reported_as_internal(self) -> None:
        """A handler that explodes must not take the request down (§23.8)."""

        def explode(_: _Args) -> dict[str, object]:
            raise ZeroDivisionError("boom")

        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="explode", description="d", arguments=_Args, handler=explode)
        )
        result = registry.call("explode", {"expression": "x"})
        assert result.error_kind is ToolErrorKind.INTERNAL
        assert "ZeroDivisionError" in (result.error or "")


class TestTimeout:
    def test_slow_call_times_out_instead_of_blocking(self) -> None:
        registry = _slow_registry()
        result = registry.call("slow", {"expression": "x"}, timeout=0.01)
        assert result.error_kind is ToolErrorKind.TIMEOUT
        assert "did not finish" in (result.error or "")

    def test_timeout_is_configurable_per_registry(self) -> None:
        registry = _slow_registry(default_timeout=0.01)
        result = registry.call("slow", {"expression": "x"})
        assert result.error_kind is ToolErrorKind.TIMEOUT

    def test_default_timeout_is_positive(self) -> None:
        assert DEFAULT_TOOL_TIMEOUT > 0

    def test_normal_calls_finish_well_inside_the_default(self, tools: ToolRegistry) -> None:
        result = tools.call("solve_equation", {"equation": "x^2 - 5*x + 6"})
        assert result.ok, result.error
        assert result.duration_ms < DEFAULT_TOOL_TIMEOUT * 1000


class TestToolTrace:
    def test_records_every_call_in_order(self, tools: ToolRegistry) -> None:
        trace = ToolTrace()
        trace = trace.with_result(tools.call("calculator", {"expression": "1+1"}))
        trace = trace.with_result(tools.call("calculator", {"expression": "x"}))
        trace = trace.with_result(tools.call("solve_equation", {"equation": "x = 1"}))

        assert len(trace.entries) == 3
        assert trace.tools_used == ("calculator", "solve_equation")
        assert len(trace.failures) == 1
        assert trace.failures[0].error_kind is ToolErrorKind.UNSUPPORTED

    def test_trace_is_immutable(self, tools: ToolRegistry) -> None:
        trace = ToolTrace()
        extended = trace.with_result(tools.call("calculator", {"expression": "1+1"}))
        assert trace.entries == ()
        assert len(extended.entries) == 1

    def test_trace_events_are_json_serializable(self, tools: ToolRegistry) -> None:
        trace = ToolTrace().with_result(tools.call("calculator", {"expression": "1+1"}))
        payload = json.dumps(trace.as_events(), ensure_ascii=False)
        assert "calculator" in payload

    def test_tools_used_is_empty_for_a_conversation_without_tools(self) -> None:
        assert ToolTrace().tools_used == ()

    def test_failed_calls_are_audited_but_not_reported_as_used(self, tools: ToolRegistry) -> None:
        trace = ToolTrace()
        trace = trace.with_result(tools.call("calculator", {"expression": "x"}))
        trace = trace.with_result(tools.call("nope", {}))
        assert trace.tools_used == ()
        assert len(trace.entries) == 2
        assert len(trace.failures) == 2


class TestEveryToolWorks:
    """§7 DoD: all eight tools are usable, one call each, through the registry."""

    @pytest.mark.parametrize(
        ("tool", "arguments", "expected"),
        [
            ("calculator", {"expression": "2+2"}, {"result": "4"}),
            (
                "evaluate_expression",
                {"expression": "x^2", "substitutions": {"x": "5"}},
                {"result": "25"},
            ),
            ("solve_equation", {"equation": "x^2 - 5*x + 6"}, {"count": 2}),
            ("simplify_expression", {"expression": "(x^2 - 1)/(x - 1)"}, {"simplified": "x + 1"}),
            ("factor_expression", {"expression": "x^2 - 1"}, {"factored": "(x - 1)*(x + 1)"}),
            ("differentiate", {"expression": "x^2"}, {"derivative": "2*x"}),
            (
                "integrate",
                {"expression": "x", "lower": "0", "upper": "2"},
                {"value": "2"},
            ),
            (
                "check_equivalence",
                {"expression_a": "x + x", "expression_b": "2*x"},
                {"equivalent": True},
            ),
        ],
    )
    def test_tool_returns_structured_output(
        self,
        tools: ToolRegistry,
        tool: str,
        arguments: dict[str, object],
        expected: dict[str, object],
    ) -> None:
        result: ToolResult = tools.call(tool, arguments)
        assert result.ok, result.error
        assert result.output is not None
        for key, value in expected.items():
            assert result.output_or_raise()[key] == value, key
