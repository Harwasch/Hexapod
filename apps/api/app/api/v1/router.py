from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.v1 import (
    agent,
    artifacts,
    assets,
    captures,
    ion,
    jobs,
    layers,
    phone,
    plans,
    recipes,
    sites,
    storage,
    system,
)
from app.schemas.common import Problem

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    # 401 is documented API-wide rather than per route: it is the shared write token's
    # answer on every mutating endpoint, and a read never raises it.
    401: {"model": Problem, "description": "Missing or wrong write token"},
    404: {"model": Problem, "description": "Not found"},
    409: {"model": Problem, "description": "Conflict"},
    422: {"model": Problem, "description": "Validation error"},
}

api_v1 = APIRouter(prefix="/api/v1", responses=ERROR_RESPONSES)
api_v1.include_router(system.router)
api_v1.include_router(sites.router)
api_v1.include_router(assets.router)
api_v1.include_router(layers.router)
api_v1.include_router(ion.router)
api_v1.include_router(agent.router)
api_v1.include_router(plans.router)
api_v1.include_router(captures.router)
api_v1.include_router(phone.router)
api_v1.include_router(jobs.router)
api_v1.include_router(artifacts.router)
api_v1.include_router(recipes.router)
api_v1.include_router(storage.router)
