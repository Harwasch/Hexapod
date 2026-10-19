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
    # Natural stream channels (NHD Stream/River) by permanence: perennial, intermittent,
    # ephemeral, or unclassified when NHD records none (FCode 46000). artificial-path carries a
    # stream through a pond, lake or wide river, so with the channels it keeps a creek's line
    # unbroken. other: canals and ditches, connectors, underground conduits, pipelines, coastline.
    kind: Literal[
        "perennial", "intermittent", "ephemeral", "unclassified", "artificial-path", "other"
    ]
    # The feature code in words, e.g. "Stream/river, intermittent".
    description: str
    geometry: LineString
    length_m: float


class LandHydrography(CamelModel):
    land_id: uuid.UUID
    boundary_revision: int
    # available: flowlines found. empty: NHD maps none here. uncovered: the land is outside the
    # United States, which is all NHD covers. unavailable: no flowlines could be read; the
    # message says whether trying again can help (the service did not answer) or not (more
    # mapped flowlines than one answer can carry).
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
