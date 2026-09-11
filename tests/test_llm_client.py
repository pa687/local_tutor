"""Tests for ``LlamaClient`` (ENGINEERING_PLAN.md §5).

Transport failures are simulated with an httpx mock transport, which covers the
client's own mapping of httpx errors onto the Phase 1 exception contract. A real
llama-server is exercised in ``test_integration_llama.py`` — mocks are never allowed
to stand in for an integration failure (§23.9).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tutor.llm.client import (
    LlamaClient,
    LlamaResponseError,
    LlamaTimeoutError,
    LlamaUnavailableError,
)
from tutor.llm.models import ChatMessage

MESSAGES = [ChatMessage(role="user", content="hi")]
Factory = Callable[..., LlamaClient]


def stream_body(chunks: list[str]) -> str:
    return "".join(f"data: {chunk}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def completion_chunk(content: str, finish_reason: str | None = None) -> str:
    finish = json.dumps(finish_reason)
    return (
        f'{{"choices":[{{"delta":{{"content":{json.dumps(content)}}},"finish_reason":{finish}}}]}}'
    )


def completion_body(content: str = "hello", *, choices: int = 1) -> str:
    payload = {
        "model": "qwen3.5-9b",
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ][:choices],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    return json.dumps(payload)


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def slow(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("too slow", request=request)


async def collect(client: LlamaClient) -> list[str]:
    return [delta.content async for delta in client.stream_chat(MESSAGES) if delta.content]


def sse_handler(body: str, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body)

    return handler


class TestStreaming:
    async def test_yields_token_text(self, llama_client_factory: Factory) -> None:
        body = stream_body([completion_chunk("Hel"), completion_chunk("lo", "stop")])
        assert await collect(llama_client_factory(sse_handler(body))) == ["Hel", "lo"]

    async def test_ignores_blank_and_non_data_lines(self, llama_client_factory: Factory) -> None:
        body = f": ping\n\nevent: message\n{stream_body([completion_chunk('ok')])}"
        assert await collect(llama_client_factory(sse_handler(body))) == ["ok"]

    async def test_ignores_malformed_json_chunks(self, llama_client_factory: Factory) -> None:
        body = "data: {not json\n\n" + stream_body([completion_chunk("ok")])
        assert await collect(llama_client_factory(sse_handler(body))) == ["ok"]

    async def test_error_status_raises_response_error(self, llama_client_factory: Factory) -> None:
        client = llama_client_factory(
            sse_handler('{"error":{"message":"out of memory"}}', status_code=500)
        )
        with pytest.raises(LlamaResponseError, match="500"):
            await collect(client)

    async def test_connection_refused_raises_unavailable(
        self, llama_client_factory: Factory
    ) -> None:
        with pytest.raises(LlamaUnavailableError, match="unreachable"):
            await collect(llama_client_factory(unreachable))

    async def test_read_timeout_raises_timeout(self, llama_client_factory: Factory) -> None:
        with pytest.raises(LlamaTimeoutError, match="did not answer"):
            await collect(llama_client_factory(slow))

    async def test_connect_timeout_counts_as_unavailable(
        self, llama_client_factory: Factory
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("no route", request=request)

        with pytest.raises(LlamaUnavailableError):
            await collect(llama_client_factory(handler))

    async def test_asks_for_usage_in_streaming_requests(
        self, llama_client_factory: Factory
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, text=stream_body([completion_chunk("ok")]))

        await collect(llama_client_factory(handler))
        assert seen["stream"] is True
        assert seen["stream_options"] == {"include_usage": True}

    async def test_falls_back_to_llama_cpp_timings(self, llama_client_factory: Factory) -> None:
        body = (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "timings": {"prompt_n": 5, "predicted_n": 9},
                }
            )
            + "\n\n"
        )
        deltas = [
            delta async for delta in llama_client_factory(sse_handler(body)).stream_chat(MESSAGES)
        ]
        assert deltas[-1].prompt_tokens == 5
        assert deltas[-1].completion_tokens == 9

    async def test_usage_only_trailing_chunk_is_kept(self, llama_client_factory: Factory) -> None:
        """llama-server sends usage in a trailing chunk with an empty choices list."""
        body = (
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            "data: "
            + json.dumps({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 34}})
            + "\n\ndata: [DONE]\n\n"
        )
        deltas = [
            delta async for delta in llama_client_factory(sse_handler(body)).stream_chat(MESSAGES)
        ]
        assert [delta.content for delta in deltas] == ["ok", ""]
        assert deltas[-1].prompt_tokens == 12
        assert deltas[-1].completion_tokens == 34


class TestComplete:
    async def test_parses_content_and_usage(self, llama_client_factory: Factory) -> None:
        client = llama_client_factory(sse_handler(completion_body("42")))
        result = await client.complete(MESSAGES)
        assert result.content == "42"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3
        assert result.finish_reason == "stop"

    async def test_no_choices_raises_response_error(self, llama_client_factory: Factory) -> None:
        client = llama_client_factory(sse_handler(completion_body(choices=0)))
        with pytest.raises(LlamaResponseError, match="no choices"):
            await client.complete(MESSAGES)

    async def test_non_json_raises_response_error(self, llama_client_factory: Factory) -> None:
        client = llama_client_factory(sse_handler("<html>oops</html>"))
        with pytest.raises(LlamaResponseError, match="non-JSON"):
            await client.complete(MESSAGES)

    async def test_connection_error_raises_unavailable(self, llama_client_factory: Factory) -> None:
        with pytest.raises(LlamaUnavailableError):
            await llama_client_factory(unreachable).complete(MESSAGES)


class TestProbe:
    async def test_online_reports_model_and_context(self, llama_client_factory: Factory) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(
                200,
                json={
                    "model_alias": "qwen3.5-9b",
                    "default_generation_settings": {"n_ctx": 65536},
                },
            )

        status = await llama_client_factory(handler).probe()
        assert status.online is True
        assert status.model == "qwen3.5-9b"
        assert status.context_size == 65536

    async def test_model_path_is_reduced_to_a_name(self, llama_client_factory: Factory) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, json={"model_path": "/opt/models/Qwen_Qwen3.5-9B.gguf"})

        status = await llama_client_factory(handler).probe()
        assert status.model == "Qwen_Qwen3.5-9B"
        assert status.context_size is None

    async def test_offline_reports_error_without_raising(
        self, llama_client_factory: Factory
    ) -> None:
        status = await llama_client_factory(unreachable).probe()
        assert status.online is False
        assert status.error is not None

    async def test_not_ready_is_offline(self, llama_client_factory: Factory) -> None:
        client = llama_client_factory(sse_handler('{"error":"loading model"}', status_code=503))
        status = await client.probe()
        assert status.online is False
        assert "503" in (status.error or "")

    async def test_props_failure_still_online(self, llama_client_factory: Factory) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404, text="not found")

        status = await llama_client_factory(handler).probe()
        assert status.online is True
        assert status.context_size is None
