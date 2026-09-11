"""FastAPI application entry point (ENGINEERING_PLAN.md §4, §16).

Phase 0 scope: application factory, ``request_id`` middleware, and the ``/health``
placeholder. No LLM calls and no business logic live here.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from tutor.api import health
from tutor.config import AppConfig, get_config
from tutor.logging_config import (
    REQUEST_ID_HEADER,
    RequestLogRecord,
    log_request,
    request_id_var,
    setup_logging,
)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI application. ``config`` overrides the process configuration."""
    setup_logging()
    app = FastAPI(title="Local Tutor", version="0.0.1")
    app.state.config = config if config is not None else get_config()

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
    return app


app = create_app()
