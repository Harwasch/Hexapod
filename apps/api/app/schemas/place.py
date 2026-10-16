"""What public data says about one point on the map, looked up live and never saved.

The inspector's "At this point" rows (apps/web/src/features/inspector/PlaceFacts.tsx). Each
fact names its source, how it was made (`basis`), its vintage and its limits, so a modelled
elevation or a survey map unit is never read as a measurement of the clicked spot.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import HttpUrl

from app.schemas.base import CamelModel

PlaceFactKind = Literal["elevation", "soil", "land-cover", "flood-zone", "ecoregion"]


class PlaceFactSource(CamelModel):
    provider: str  # The SourceSpec id, e.g. "usgs-elevation".
    name: str
    url: HttpUrl  # The exact public query that answered.
    # What a citation link opens: `url` itself when that query is a GET a browser can show,
    # otherwise the source's public viewer. Soil Data Access answers only a POST (its SQL is in
    # `data.query`), and land cover is read from a 10-20 MB image tile (`data.tileUrl`).
    link_url: HttpUrl
    documentation_url: HttpUrl
    license: str
    attribution: str
    resolution: str


class PlaceFact(CamelModel):
    kind: PlaceFactKind
    # available: an answer. empty: the source answered with nothing mapped here. uncovered:
    # the source does not cover this place. unavailable: the lookup failed; try again.
    status: Literal["available", "empty", "uncovered", "unavailable"]
    label: str
    detail: str | None = None
    basis: Literal[
        "modelled-surface",
        "mapped-survey",
        "classified-imagery",
        "regulatory-map",
        "regional-framework",
    ]
    vintage: str | None = None
    caveat: str
    longitude: float
    latitude: float
    retrieved_at: datetime
    cached: bool = False
    source: PlaceFactSource | None
    data: dict[str, Any] = {}
