"""A saved Land as ground to plan on: its boundary and inventory features as zones."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint, LineString


class LandGroundItem(CamelModel):
    """One plannable piece of a Land. Read model: every field explicit."""

    # "L-" and the first 8 hex digits of the land (boundary) or feature id.
    zone_id: str = Field(max_length=40)
    kind: Literal["boundary", "feature"]
    feature_id: uuid.UUID | None
    # The boundary's or the feature's own revision.
    revision: int
    name: str
    category: str | None
    status: str | None
    # What a plan covers: the outline, or for a line the corridor around it.
    footprint: Footprint
    # A mapped line (a fence, a ditch) itself; the corridor is `bufferM` wide around it.
    line: LineString | None
    area_m2: float
    length_m: float | None
    buffer_m: float | None
    # Where it came from, shown wherever the zone is drawn.
    source: str = Field(max_length=300)


class LandGround(CamelModel):
    land_id: uuid.UUID
    name: str
    boundary_revision: int
    items: list[LandGroundItem]
    # Features that cannot be planned on as ground (points), counted so the console can say so.
    skipped: int
