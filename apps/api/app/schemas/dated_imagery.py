"""Dated public imagery for an area: what the Time Lantern can show under the cursor.

Each source is one acquisition a person can step to -- a NAIP year, a dry-season Sentinel-2
scene, one edition of a USGS topographic sheet -- with what its date means, its licence and how
far its positions may be off, in the source's own words or a published standard. Nothing here
was measured against today's 3D world; the web client says so on every label.

Listed by GET /api/v1/imagery/dated (app/api/v1/imagery.py) from
app/research/providers/dated_imagery.py; nothing is stored.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint

Finite = Annotated[float, Field(allow_inf_nan=False)]
#: West, south, east, north in degrees (WGS 84).
Bounds = Annotated[list[Finite], Field(min_length=4, max_length=4)]

DatedImageryKind = Literal["aerial-photo", "satellite", "historical-map"]
ProviderStatus = Literal["available", "empty", "uncovered", "unavailable"]


class StatedAccuracy(CamelModel):
    """How far positions may be off, and on whose word.

    measured-fit: control points measured against the image (a land's own alignment).
    stated-by-source: the record's own sentence, quoted in `basis`. standard: the published
    standard the source was made to, if the sheet states compliance. not-stated: the record says
    nothing, and nothing is borrowed from another year or product.
    """

    kind: Literal["measured-fit", "stated-by-source", "standard", "not-stated"]
    value_m: Finite | None = Field(default=None, ge=0)
    basis: str = Field(max_length=1200)
    citation_url: HttpUrl | None = None


class DatumShift(CamelModel):
    """How a map drawn on an older datum was moved onto WGS 84 to lie under today's world: the
    operation PROJ used, its stated accuracy, and whether a better one exists that this server
    does not have (the NADCON grids). The lens says it beside the map's own accuracy."""

    from_datum: Literal["NAD27"]
    operation: str = Field(max_length=300)
    #: The operation's stated accuracy (m), when it states one.
    accuracy_m: Finite | None = Field(default=None, ge=0)
    better_available: bool


class DatedImageryPart(CamelModel):
    """One record a source is made of (a NAIP quarter quad, a scene, a map sheet)."""

    observed_on: date | None = None
    observed_year: int = Field(ge=1800, le=2200)
    footprint: Footprint
    record_url: HttpUrl
    accuracy: StatedAccuracy


class DatedImageryTiles(CamelModel):
    """How a browser draws the source: an XYZ template on a tiling scheme."""

    scheme: Literal["web-mercator", "geographic"]
    #: Absolute (a public tiler) or API-relative (`/api/v1/...`, which the client prefixes).
    url_template: str = Field(max_length=2000)
    maximum_level: int = Field(ge=0, le=22)
    format: Literal["png"] = "png"


class DatedImagerySource(CamelModel):
    id: str = Field(max_length=200)
    #: The SourceSpec it came from (GET /research/sources lists them).
    provider: str
    kind: DatedImageryKind
    title: str = Field(max_length=300)
    date_label: str = Field(max_length=120)
    date_meaning: str = Field(max_length=600)
    #: Orders the rail, newest first.
    sort_date: date
    #: Ground size of a pixel (m); for a scanned map, of a scan pixel, not of what it shows.
    #: Null when the records do not say.
    resolution_m: Finite | None = Field(default=None, gt=0)
    #: The printed scale of a map (24000 for 1:24,000), else null.
    map_scale: int | None = Field(default=None, gt=0)
    bounds: Bounds
    tiles: DatedImageryTiles
    license: str = Field(max_length=300)
    license_url: HttpUrl | None = None
    attribution: str = Field(max_length=500)
    parts: list[DatedImageryPart] = Field(min_length=1, max_length=40)
    #: For a map moved from an older datum: how, and how well.
    datum_shift: DatumShift | None = None
    #: Caveats in words, each for the reader: the datum shift, a trimmed sheet edge, clouds.
    warnings: list[str] = []


class DatedImageryProviderStatus(CamelModel):
    provider: str
    name: str
    status: ProviderStatus
    summary: str


class DatedImageryCatalog(CamelModel):
    #: The area searched: the request's, rounded outward to 0.01°.
    bounds: Bounds
    retrieved_at: datetime
    #: Answered from this server's cache (6 h) rather than asked again.
    cached: bool = False
    sources: list[DatedImagerySource]
    providers: list[DatedImageryProviderStatus]
