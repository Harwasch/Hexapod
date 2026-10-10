from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint


class BoundaryImportRead(CamelModel):
    status: Literal["ready", "choose-layer", "needs-crs", "needs-repair"]
    boundary: Footprint | None = None
    source_crs: str | None = None
    target_crs: str = "EPSG:4326"
    layers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
