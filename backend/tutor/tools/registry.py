"""Tool registry and dispatcher — the eight tools fixed by §7.

The registry is the single way into the tool system, and it is deliberately boring:

* a tool is a :class:`ToolSpec` — name, description, a **Pydantic** argument model and
  a pure handler ``validated_arguments -> dict``;
* :meth:`ToolRegistry.call` never raises for bad input. It returns a
  :class:`ToolResult` carrying either structured ``output`` or a typed
  ``error_kind`` + message, so callers (Phase 4's verifier, the API, the UI) all see
  one shape;
* every call is timed and can be audited: :class:`ToolTrace` collects the results and
  exposes ``tools_used`` (which feeds ``TutorResponse.tools_used``, §6/§18) and
  ``as_events()`` (which feeds the UI's tool trace, §17).

Error handling policy (§7, §23.8): a handler signals *"this input is not acceptable"*
by raising :class:`ToolInputRejected` (unsafe or malformed) or
:class:`ToolUnsupported` (legitimate maths we cannot do). Anything else is a bug and
is reported as ``internal`` after being logged — it is never swallowed.

Timeout policy: SymPy has no cancellation, so the timeout bounds **how long we wait**,
not the CPU a pathological call may keep burning. Expressions are size-limited up
front (``tutor.tools.sandbox``) to make that a rare case; a process pool is the
escalation path if Phase 4 ever needs a hard kill.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

#: §7 does not name a timeout value; this is the guard rail, not a precision setting.
DEFAULT_TOOL_TIMEOUT: Final = 5.0
_MAX_WORKERS: Final = 4

_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="tutor-tool")


class ToolErrorKind(StrEnum):
    """Why a tool call did not produce a result."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


class ToolError(RuntimeError):
    """Base class for tool-side problems."""


class ToolInputRejected(ToolError):
    """The input itself is unacceptable (unsafe, oversized, malformed)."""


class ToolUnsupported(ToolError):
    """The input is fine but this operation cannot produce an answer."""


class ToolCall(BaseModel):
    """The §7 wire format of one tool call.

    ```json
    {"tool": "solve_equation", "arguments": {"equation": "x**2 - 5*x + 6", "variable": "x"}}
    ```

    Turning free-form model output into this envelope is Phase 4's job; the envelope
    itself is part of the tool contract, so it lives here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool."""

    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """JSON schema of the arguments, for prompts (§7) and for the UI."""
        return self.arguments.model_json_schema()


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call — the auditable unit of §7/§17."""

    tool: str
    arguments: Mapping[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    error_kind: ToolErrorKind | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error_kind is None

    def output_or_raise(self) -> dict[str, Any]:
        """The output of a successful call.

        Callers that expect a result (Phase 4's verifier, the tests) get one dict
        without repeating the ``None`` check; a failed call raises :class:`ToolError`
        carrying the recorded reason instead of returning an empty result.
        """
        if self.output is None:
            raise ToolError(f"tool {self.tool!r} produced no output: {self.error}")
        return self.output

    def summary(self) -> str:
        """One-line, human-readable entry for the tool trace (§17)."""
        mark = "✓" if self.ok else "✗"
        arguments = ", ".join(f"{key}={value}" for key, value in self.arguments.items())
        if self.ok and self.output is not None:
            outcome = ", ".join(f"{key}={value}" for key, value in self.output.items())
        else:
            outcome = self.error or "failed"
        return _shorten(f"{mark} {self.tool}({arguments}) → {outcome}")

    def as_event(self) -> dict[str, Any]:
        """JSON-serializable trace record (log metadata, API payload, UI)."""
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_kind": self.error_kind.value if self.error_kind is not None else None,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(frozen=True)
class ToolTrace:
    """Ordered, immutable audit trail of the tool calls made during one turn."""

    entries: tuple[ToolResult, ...] = ()

    def with_result(self, result: ToolResult) -> ToolTrace:
        """Return a new trace with ``result`` appended."""
        return ToolTrace(entries=(*self.entries, result))

    @property
    def tools_used(self) -> tuple[str, ...]:
        """Distinct tools that produced a result, in first-use order (§6 ``tools_used``).

        Failed calls stay in :attr:`entries` — the trace is the full audit trail (§17) —
        but they did not inform the answer, so they are not reported as *used*.
        """
        return tuple(dict.fromkeys(entry.tool for entry in self.entries if entry.ok))

    @property
    def failures(self) -> tuple[ToolResult, ...]:
        return tuple(entry for entry in self.entries if not entry.ok)

    def as_events(self) -> list[dict[str, Any]]:
        return [entry.as_event() for entry in self.entries]


class ToolRegistry:
    """Registers tool specs and dispatches calls to them."""

    def __init__(self, *, default_timeout: float = DEFAULT_TOOL_TIMEOUT) -> None:
        if default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        self._specs: dict[str, ToolSpec] = {}
        self._default_timeout = default_timeout

    # ------------------------------------------------------------ registration
    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        """Add ``spec``; re-registering a name fails unless ``replace`` is set."""
        if spec.name in self._specs and not replace:
            raise ValueError(f"tool {spec.name!r} is already registered")
        if not spec.name.isidentifier():
            raise ValueError(f"tool name {spec.name!r} is not a valid identifier")
        self._specs[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        """Registered tool names, in registration order."""
        return tuple(self._specs)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def catalog(self) -> list[dict[str, Any]]:
        """Name + description + argument schema of every tool (§7's structured contract)."""
        return [
            {"tool": spec.name, "description": spec.description, "arguments": spec.schema()}
            for spec in self._specs.values()
        ]

    # --------------------------------------------------------------- dispatch
    def call_request(
        self,
        request: ToolCall | Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> ToolResult:
        """Run a call written in the §7 envelope; malformed envelopes are results too."""
        try:
            envelope = (
                request if isinstance(request, ToolCall) else ToolCall.model_validate(request)
            )
        except ValidationError as exc:
            return self._failure(
                "<malformed>",
                _json_safe(request),
                ToolErrorKind.INVALID_ARGUMENTS,
                f"tool call must be {{'tool': ..., 'arguments': {{...}}}}: "
                f"{_format_validation_error(exc)}",
                time.perf_counter(),
            )
        return self.call(envelope.tool, envelope.arguments, timeout=timeout)

    def call(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ToolResult:
        """Run ``tool`` with ``arguments`` and always return a :class:`ToolResult`.

        ``arguments`` are validated by the tool's Pydantic model *before* the handler
        runs, so a handler only ever sees a typed object.
        """
        started = time.perf_counter()
        if arguments is None:
            supplied: Mapping[str, Any] = {}
        elif isinstance(arguments, Mapping):
            supplied = arguments
        else:
            return self._failure(
                tool,
                {},
                ToolErrorKind.INVALID_ARGUMENTS,
                f"arguments must be an object, got {type(arguments).__name__}",
                started,
            )
        raw_arguments = _json_safe(dict(supplied))

        spec = self._specs.get(tool)
        if spec is None:
            return self._failure(
                tool,
                raw_arguments,
                ToolErrorKind.UNKNOWN_TOOL,
                f"unknown tool {tool!r}; available: {', '.join(self._specs) or 'none'}",
                started,
            )

        try:
            validated = spec.arguments.model_validate(raw_arguments)
        except ValidationError as exc:
            return self._failure(
                tool,
                raw_arguments,
                ToolErrorKind.INVALID_ARGUMENTS,
                _format_validation_error(exc),
                started,
            )

        deadline = self._default_timeout if timeout is None else timeout
        future = _EXECUTOR.submit(spec.handler, validated)
        try:
            output = future.result(timeout=deadline)
        except FutureTimeoutError:
            future.cancel()
            return self._failure(
                tool,
                validated.model_dump(),
                ToolErrorKind.TIMEOUT,
                f"tool {tool!r} did not finish within {deadline}s",
                started,
            )
        except ToolInputRejected as exc:
            return self._failure(
                tool, validated.model_dump(), ToolErrorKind.REJECTED, str(exc), started
            )
        except ToolUnsupported as exc:
            return self._failure(
                tool, validated.model_dump(), ToolErrorKind.UNSUPPORTED, str(exc), started
            )
        except Exception as exc:
            logger.warning("tool %s failed with %s: %s", tool, type(exc).__name__, exc)
            return self._failure(
                tool,
                validated.model_dump(),
                ToolErrorKind.INTERNAL,
                f"{type(exc).__name__}: {exc}",
                started,
            )

        return ToolResult(
            tool=tool,
            arguments=validated.model_dump(),
            output=_json_safe(output),
            duration_ms=_elapsed_ms(started),
        )

    def _failure(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        kind: ToolErrorKind,
        message: str,
        started: float,
    ) -> ToolResult:
        return ToolResult(
            tool=tool,
            arguments=arguments,
            error=message,
            error_kind=kind,
            duration_ms=_elapsed_ms(started),
        )


def build_default_registry(*, default_timeout: float = DEFAULT_TOOL_TIMEOUT) -> ToolRegistry:
    """Registry holding exactly the eight tools §7 fixes.

    The imports are local so the tool modules can import this module's error types
    without a circular import at load time.
    """
    from tutor.tools.algebra import ALGEBRA_TOOLS
    from tutor.tools.calculator import CALCULATOR_TOOLS

    registry = ToolRegistry(default_timeout=default_timeout)
    for spec in (*CALCULATOR_TOOLS, *ALGEBRA_TOOLS):
        registry.register(spec)
    return registry


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _format_validation_error(exc: ValidationError) -> str:
    """Compact, value-free rendering of a Pydantic failure."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "arguments"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _json_safe(value: Any) -> Any:
    """Coerce ``value`` into plain JSON types so traces can always be serialized."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _shorten(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}…"
