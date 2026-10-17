"""Mapped streams and ditches in and around a saved Land, from the USGS NHD."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import LineString
from app.schemas.place import PlaceFactSource


class HydrographyFeature(CamelModel):
    # Unique in this answer: the NHD permanent identifier, plus "/2", "/3" for further pieces
    # when the clip splits one flowline.
    id: str = Field(max_length=80)
    permanent_identifier: str = Field(max_length=60)
    name: str | None
    fcode: int
    # The NHD hydrographic category; "other" covers canals, connectors, pipelines and the
    # artificial paths drawn through lakes.
    kind: Literal["perennial", "intermittent", "ephemeral", "other"]
    # The feature code in words, e.g. "Stream/river, intermittent".
    description: str
    geometry: LineString
    length_m: float


class LandHydrography(CamelModel):
    land_id: uuid.UUID
    boundary_revision: int
    # available: flowlines found. empty: NHD maps none here. uncovered: the land is outside the
    # United States, which is all NHD covers. unavailable: the lookup failed; try again.
    status: Literal["available", "empty", "uncovered", "unavailable"]
    message: str
    # Flowlines are clipped to the boundary grown by this distance, so a creek just outside
    # the line still counts.
    buffer_m: float
    features: list[HydrographyFeature]
    # The service answered with its record limit: more flowlines may exist.
    truncated: bool
    source: PlaceFactSource
    retrieved_at: datetime
