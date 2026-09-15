from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import assets, ion, layers, sites, system
from app.schemas.common import Problem

ERROR_RESPONSES = {
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
