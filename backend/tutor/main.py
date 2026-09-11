"""FastAPI application entry point (ENGINEERING_PLAN.md §4, §16).

Owns the application factory, the ``request_id`` middleware (§18), the lifetime of
the shared :class:`~tutor.llm.client.LlamaClient`, and the single
:class:`~tutor.tutor.engine.TutorEngine` every chat endpoint answers through (§6).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from tutor.api import chat, health
from tutor.config import AppConfig, get_config
from tutor.llm.client import LlamaClient
from tutor.llm.prompts import PromptLibrary, get_prompt_library
from tutor.logging_config import (
    REQUEST_ID_HEADER,
    RequestLogRecord,
    log_request,
    request_id_var,
    setup_logging,
)
from tutor.tutor.engine import TutorEngine


def create_app(
    config: AppConfig | None = None,
    llama_client: LlamaClient | None = None,
    engine: TutorEngine | None = None,
    prompts: PromptLibrary | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Every collaborator is injectable so tests can supply a fake llama-server, a fake
    engine or an alternative prompt directory; production passes nothing.
    """
    setup_logging()
    resolved = config if config is not None else get_config()
    client = llama_client if llama_client is not None else LlamaClient(resolved.llm)
    library = prompts if prompts is not None else get_prompt_library()
    tutor_engine = engine if engine is not None else TutorEngine(client, library, resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if llama_client is None:
                await client.aclose()

    app = FastAPI(title="Local Tutor", version="0.0.4", lifespan=lifespan)
    app.state.config = resolved
    app.state.llama = client
    app.state.prompts = library
    app.state.engine = tutor_engine

    @app.middleware("http")
    async def attach_request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
            log_request(
                RequestLogRecord(
                    request_id=request_id,
                    model=app.state.config.llm.model,
                    latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
                )
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
