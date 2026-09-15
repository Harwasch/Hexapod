from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import assets, ion, layers, sites, system

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(system.router)
api_v1.include_router(sites.router)
api_v1.include_router(assets.router)
api_v1.include_router(layers.router)
api_v1.include_router(ion.router)
