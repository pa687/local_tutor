"""``GET /health`` (ENGINEERING_PLAN.md §16, §5 DoD).

Reports backend liveness plus whether llama-server answers. A model backend that is
down must never take the API down with it: the endpoint always returns 200 and says
``llama: "down"`` instead of raising.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from tutor.config import AppConfig
from tutor.llm.client import LlamaClient

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def get_health(request: Request) -> dict[str, object]:
    """Return the §16 health payload."""
    config: AppConfig = request.app.state.config
    client: LlamaClient | None = getattr(request.app.state, "llama", None)

    status = await client.probe() if client is not None else None
    if status is not None and not status.online:
        logger.warning("llama-server not reachable: %s", status.error)

    return {
        "backend": "ok",
        "llama": "ok" if status is not None and status.online else "down",
        "model": config.llm.model,
        "context_size": status.context_size if status is not None else None,
    }
