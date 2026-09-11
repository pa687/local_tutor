"""``LlamaClient`` — the only place that talks HTTP to llama-server (§5).

Responsibilities:

* OpenAI-compatible ``/v1/chat/completions``, streaming and non-streaming;
* a liveness probe backing ``GET /health``;
* turning transport problems into explicit exceptions, never into silence:
  :class:`LlamaUnavailableError` (down / crashed / OOM-killed),
  :class:`LlamaTimeoutError` (no answer within ``llm.timeout``),
  :class:`LlamaResponseError` (error status or unusable payload).

The client keeps no connection state and no cached "is it up" flag, so a restarted
llama-server is picked up by the next request automatically (Phase 1 DoD).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any

import httpx

from tutor.config import LLMConfig
from tutor.llm.models import (
    ChatCompletion,
    ChatMessage,
    GenerationOptions,
    LlamaStatus,
    StreamDelta,
)

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
LLAMA_HEALTH_PATH = "/health"
PROPS_PATH = "/props"
SSE_DATA_PREFIX = "data:"
SSE_DONE = "[DONE]"


class LlamaClientError(RuntimeError):
    """Base class for every llama-server problem."""


class LlamaUnavailableError(LlamaClientError):
    """llama-server is not reachable (not started, crashed, or OOM-killed)."""


class LlamaTimeoutError(LlamaClientError):
    """llama-server did not answer within ``llm.timeout`` seconds."""


class LlamaResponseError(LlamaClientError):
    """llama-server returned an error status or a payload we cannot use."""


class LlamaClient:
    """Thin async client around the llama-server HTTP API."""

    def __init__(self, config: LLMConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        timeout = httpx.Timeout(config.timeout, connect=min(10.0, config.timeout))
        self._client = client or httpx.AsyncClient(base_url=config.base_url, timeout=timeout)

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if we created it)."""
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ probe
    async def probe(self) -> LlamaStatus:
        """Check whether llama-server answers, and report model + context size.

        Never raises: ``/health`` must stay usable while the model backend is down.
        """
        try:
            health = await self._client.get(LLAMA_HEALTH_PATH)
        except httpx.TimeoutException:
            return LlamaStatus(online=False, error="timeout while probing llama-server")
        except httpx.TransportError as exc:
            return LlamaStatus(online=False, error=f"llama-server unreachable: {exc}")
        if health.status_code >= 400:
            return LlamaStatus(
                online=False,
                error=f"llama-server not ready: HTTP {health.status_code}",
            )

        model: str | None = None
        context_size: int | None = None
        try:
            props = await self._client.get(PROPS_PATH)
            if props.status_code == 200:
                payload = props.json()
                if isinstance(payload, dict):
                    model = _extract_model(payload)
                    context_size = _extract_context_size(payload)
        except httpx.HTTPError as exc:
            logger.warning("could not read /props from llama-server: %s", exc)
        except ValueError as exc:
            logger.warning("llama-server /props returned invalid JSON: %s", exc)
        return LlamaStatus(online=True, model=model, context_size=context_size)

    # -------------------------------------------------------------- inference
    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """Yield deltas as the model produces them."""
        payload = self._chat_payload(messages, options, stream=True)
        try:
            async with self._client.stream("POST", CHAT_COMPLETIONS_PATH, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise LlamaResponseError(
                        f"llama-server returned HTTP {response.status_code}: {_shorten(body)}"
                    )
                async for line in response.aiter_lines():
                    delta = _parse_stream_line(line)
                    if delta is not None and not delta.is_empty:
                        yield delta
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LlamaUnavailableError(f"llama-server unreachable: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LlamaTimeoutError(
                f"llama-server did not answer within {self._config.timeout}s"
            ) from exc
        except httpx.TransportError as exc:
            raise LlamaUnavailableError(f"llama-server unreachable: {exc}") from exc

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatCompletion:
        """Non-streaming completion, used for short internal calls."""
        payload = self._chat_payload(messages, options, stream=False)
        try:
            response = await self._client.post(CHAT_COMPLETIONS_PATH, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LlamaUnavailableError(f"llama-server unreachable: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LlamaTimeoutError(
                f"llama-server did not answer within {self._config.timeout}s"
            ) from exc
        except httpx.TransportError as exc:
            raise LlamaUnavailableError(f"llama-server unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise LlamaResponseError(
                f"llama-server returned HTTP {response.status_code}: {_shorten(response.text)}"
            )
        try:
            payload_out = response.json()
        except ValueError as exc:
            raise LlamaResponseError("llama-server returned a non-JSON body") from exc
        if not isinstance(payload_out, dict):
            raise LlamaResponseError("llama-server returned an unexpected payload shape")

        choices = payload_out.get("choices") or []
        if not choices:
            raise LlamaResponseError("llama-server returned no choices")
        message = choices[0].get("message") or {}
        usage = payload_out.get("usage") or {}
        return ChatCompletion(
            content=str(message.get("content") or ""),
            model=payload_out.get("model"),
            finish_reason=choices[0].get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    # ---------------------------------------------------------------- helpers
    def _chat_payload(
        self,
        messages: Sequence[ChatMessage],
        options: GenerationOptions | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [message.model_dump() for message in messages],
            "stream": stream,
        }
        if stream:
            # §18 needs prompt/completion token counts, which OpenAI-compatible
            # servers only report in the final chunk when asked to.
            payload["stream_options"] = {"include_usage": True}
        if options is not None:
            payload.update(options.to_payload())
        return payload


def _shorten(text: str, limit: int = 300) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}…"


def _parse_stream_line(line: str) -> StreamDelta | None:
    """Turn one SSE line into a :class:`StreamDelta` (``None`` = ignore the line)."""
    stripped = line.strip()
    if not stripped or not stripped.startswith(SSE_DATA_PREFIX):
        return None
    data = stripped[len(SSE_DATA_PREFIX) :].strip()
    if not data or data == SSE_DONE:
        return None
    try:
        chunk = json.loads(data)
    except ValueError:
        logger.warning("ignoring malformed SSE chunk from llama-server")
        return None
    if not isinstance(chunk, dict):
        return None
    usage = chunk.get("usage") or {}
    timings = chunk.get("timings") or {}
    choices = chunk.get("choices") or []
    if not choices:
        # llama-server reports token usage in a trailing chunk with no choices.
        prompt_tokens = _count(usage, timings, "prompt_tokens", "prompt_n")
        completion_tokens = _count(usage, timings, "completion_tokens", "predicted_n")
        if prompt_tokens is None and completion_tokens is None:
            return None
        return StreamDelta(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choice = choices[0]
    delta = choice.get("delta") or {}
    return StreamDelta(
        content=str(delta.get("content") or ""),
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=_count(usage, timings, "prompt_tokens", "prompt_n"),
        completion_tokens=_count(usage, timings, "completion_tokens", "predicted_n"),
    )


def _count(
    usage: dict[str, Any],
    timings: dict[str, Any],
    usage_key: str,
    timings_key: str,
) -> int | None:
    """Prefer the OpenAI-style ``usage`` field, fall back to llama.cpp ``timings``."""
    value = usage.get(usage_key)
    if isinstance(value, int):
        return value
    fallback = timings.get(timings_key)
    return fallback if isinstance(fallback, int) else None


def _extract_model(props: dict[str, Any]) -> str | None:
    for key in ("model_alias", "model_path", "model"):
        value = props.get(key)
        if isinstance(value, str) and value:
            if key == "model_path":
                return Path(value).name.removesuffix(".gguf")
            return value
    return None


def _extract_context_size(props: dict[str, Any]) -> int | None:
    candidates: list[Any] = [props.get("n_ctx")]
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        candidates.append(settings.get("n_ctx"))
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None
