"""Show once, everywhere: one tapped cell of a Land against every other, by seasonal signature.

A `sentinel-2-signature` raster (`app.analysis.signature`) holds each 10 m cell's reflectance
and NDVI in dated windows; `app.analysis.similarity` compares them. Scores travel as a
`LonLatGrid` the console drapes over the ground, and the alike cells come back as areas for a
plan. A similar signature is similar reflectance on these dates, never a species, cover or
habitat identification.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint
from app.schemas.grids import LonLatGrid
from app.schemas.land_rasters import VegetationPeriod

Finite = Annotated[float, Field(allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
#: west, south, east, north in degrees.
Rectangle = Annotated[list[Finite], Field(min_length=4, max_length=4)]


class SignatureScene(CamelModel):
    """One dated window of the signature: the scene it was read from."""

    window: int = Field(ge=0, le=5)
    period: VegetationPeriod
    scene_id: str | None
    acquired_at: datetime | None


class SignatureSeedWindow(CamelModel):
    """The tapped cell's own values in one window (reflectance 0..1, NDVI -1..1)."""

    window: int = Field(ge=0, le=5)
    acquired_at: datetime | None
    green: Finite
    red: Finite
    nir: Finite
    swir16: Finite
    ndvi: Finite


class SignatureSeed(CamelModel):
    #: Where the tap landed.
    longitude: Finite
    latitude: Finite
    #: The centre of the 10 m analysis cell it fell in, which is what was compared.
    cell_longitude: Finite
    cell_latitude: Finite
    windows: list[SignatureSeedWindow]


class SignatureSimilarityRead(CamelModel):
    raster_id: uuid.UUID
    land_id: uuid.UUID
    boundary_revision: int
    #: The land's boundary changed since the raster was made: the answer is for the old one.
    stale: bool
    seed: SignatureSeed
    #: How alike each cell is to the seed: u8, 0..250 = cosine x 250 (scale 0.004; a cosine
    #: below 0 is 0), 255 = outside the land or no full signature (a season not clear there).
    #: A lon/lat grid at about the analysis spacing, resampled nearest from the analysis grid.
    grid: LonLatGrid
    #: Ground area of one grid cell, for a tally of lit cells (approximate: the display grid).
    cell_area_m2: Finite
    native_resolution_m: Finite
    #: Cells with a full signature (clear in every window), and all cells inside the boundary.
    compared_cells: int = Field(ge=0)
    boundary_cells: int = Field(ge=0)
    scenes: list[SignatureScene]
    method: str
    interpretation: str
    warnings: list[str]
    attribution: str


class SignatureZonesRequest(CamelModel):
    """The seed again (the server recomputes on the analysis grid), the slider's threshold
    and the cells the operator protected."""

    longitude: Longitude
    latitude: Latitude
    threshold: Finite = Field(default=0.8, ge=0.3, le=0.99)
    #: Ground to leave out, as lon/lat rectangles: the console sends its protected display
    #: cells as runs along each row. A cell whose centre is in one is never part of the job.
    protected: list[Rectangle] = Field(default_factory=list, max_length=5000)
    #: Patches smaller than this are left out and counted in `dropped` (2000 m², half an acre).
    min_patch_m2: Finite = Field(default=2000, ge=0, le=1_000_000)
    #: The largest patches become areas of their own; the rest join one `remainder` area.
    max_zones: int = Field(default=8, ge=1, le=20)


class SignatureZone(CamelModel):
    footprint: Footprint
    area_m2: Finite = Field(ge=0)
    #: Alike analysis cells inside it.
    cells: int = Field(ge=0)
    mean_similarity: Finite
    #: Separate patches it is made of (1, except for the remainder).
    patches: int = Field(ge=1)


class SignatureLeftOut(CamelModel):
    patches: int = Field(ge=0)
    area_m2: Finite = Field(ge=0)


class SignatureZonesRead(CamelModel):
    raster_id: uuid.UUID
    land_id: uuid.UUID
    boundary_revision: int
    stale: bool
    threshold: Finite
    seed: SignatureSeed
    #: The largest patches, largest first.
    zones: list[SignatureZone]
    #: Every other patch at least `min_patch_m2`, as one area; None when there are none.
    remainder: SignatureZone | None
    #: Patches under `min_patch_m2`, left out of the job.
    dropped: SignatureLeftOut
    #: Analysis cells alike enough but protected, and so left out.
    protected_cells: int = Field(ge=0)
    #: Analysis cells alike enough and not protected, whatever patch they fell in.
    matched_cells: int = Field(ge=0)
    cell_area_m2: Finite
    scenes: list[SignatureScene]
    method: str
    interpretation: str
    attribution: str
