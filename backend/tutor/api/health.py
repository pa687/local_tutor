"""Health endpoint (ENGINEERING_PLAN.md §16).

Phase 0 placeholder: reports that the backend itself is up. ``llama`` reachability
and the real ``context_size`` are wired to llama-server in Phase 1 (§5 DoD).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health(request: Request) -> dict[str, object]:
    """Return the §16 health payload, with llama fields still unknown in Phase 0."""
    config = request.app.state.config
    return {
        "backend": "ok",
        "llama": "unknown",
        "model": config.llm.model,
        "context_size": None,
    }
