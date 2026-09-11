"""LLM request/response models (ENGINEERING_PLAN.md §4, §5).

These mirror the OpenAI-compatible payloads that llama-server speaks. They contain
no prompt text and no policy: prompt assembly belongs to Phase 2.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One message in an OpenAI-compatible chat request."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class GenerationOptions(BaseModel):
    """Optional sampling parameters; unset values are left to llama-server defaults.

    ``enable_thinking`` is passed through as ``chat_template_kwargs`` and only sent
    when set: Qwen-style templates honour it, other templates simply never see the
    key. Internal calls that need a deterministic, short answer (the classifier) turn
    it off; the answer itself leaves the model on its own default.
    """

    model_config = ConfigDict(frozen=True)

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    enable_thinking: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return payload


class ChatCompletion(BaseModel):
    """Aggregated result of a non-streaming completion."""

    model_config = ConfigDict(frozen=True)

    content: str
    model: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class StreamDelta(BaseModel):
    """One chunk of a streaming completion."""

    model_config = ConfigDict(frozen=True)

    content: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None

    @property
    def has_accounting(self) -> bool:
        return self.prompt_tokens is not None or self.completion_tokens is not None

    @property
    def is_empty(self) -> bool:
        """True when the chunk carries neither text, nor a finish reason, nor counts."""
        return not (self.content or self.is_final or self.has_accounting)


class LlamaStatus(BaseModel):
    """Result of a llama-server liveness probe; never raises."""

    model_config = ConfigDict(frozen=True)

    online: bool
    model: str | None = None
    context_size: int | None = None
    error: str | None = None
