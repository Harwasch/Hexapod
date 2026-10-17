"""Workable ground on a saved Land: where an assumed machine class can work an outline.

The dress rehearsal raises hands where a plan's lanes go past a class's assumed limits; this
is the one-click answer to them (`app.analysis.workable`): the outline less the ground too
steep for the class on the Land's elevation raster, less the class's buffer around mapped
creeks, less slivers too small to work. Derived, never measured: the limits are assumptions,
and the slope is a model's.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.agent import MachineClassId
from app.schemas.base import CamelModel
from app.schemas.geojson import Footprint
from app.schemas.geojson import MultiPolygon as MultiPolygonModel

Area = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class WorkableGroundRequest(CamelModel):
    #: The outline to work, in lon/lat on the Land (a plan's zone).
    footprint: Footprint
    #: Whose assumed limits apply (machine_classes.json); a drone has none on the ground.
    machine_class: MachineClassId
    #: The elevation raster to read slope from; default the Land's freshest bare-earth one.
    dem_raster_id: uuid.UUID | None = None
    #: Count NHD ephemeral drainages as creeks too (off, as in the rehearsal: they run only
    #: after rain).
    include_ephemeral_creeks: bool = False


class WorkableRemoved(CamelModel):
    """Ground taken out of the outline, by reason. Each part is counted once, in the order
    no data, too steep, creek buffer, slivers, so the parts and `workableM2` add up to
    `footprintM2`."""

    steep_m2: Area
    creek_buffer_m2: Area
    slivers_m2: Area
    #: Pieces under 0.1 ac left after the cuts.
    slivers: int = Field(ge=0)
    #: Where the raster has no height (outside the Land's saved outline): not known to be
    #: workable, so left out.
    no_data_m2: Area


class WorkableLeftOut(CamelModel):
    """One reason's left-out ground, for the map to draw hatched."""

    reason: Literal["steep", "creek-buffer", "sliver", "no-data"]
    #: e.g. "Left out: too steep".
    label: str
    area_m2: Area
    #: None when the piece cannot be stored as an outline at the footprint's precision.
    geometry: MultiPolygonModel | None


class WorkableLimits(CamelModel):
    machine_class: MachineClassId
    #: The class's label, which says "(assumed)".
    label: str
    max_slope_pct: float
    creek_buffer_m: float


class WorkableElevation(CamelModel):
    raster_id: uuid.UUID
    dataset: Literal["usgs-3dep-13", "cop-dem-glo-30"]
    #: e.g. "USGS 3DEP 1/3″ bare-earth DEM, 10 m".
    label: str
    #: The source file's publication date (YYYY-MM-DD) where it has one: 3DEP tiles carry a
    #: file date, not an acquisition date.
    published: str | None
    resolution_m: float
    #: Made for an earlier outline of the Land.
    stale: bool
    #: The Copernicus surface model (canopy included) stood in for a missing 3DEP raster.
    fallback: bool


class WorkableCreeks(CamelModel):
    status: Literal["available", "empty", "uncovered", "unavailable"]
    #: e.g. "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000".
    label: str
    #: Flowlines counted as creeks, after the ephemeral rule.
    counted: int = Field(ge=0)
    ephemeral_included: bool
    message: str


class WorkableGroundRead(CamelModel):
    land_id: uuid.UUID
    boundary_revision: int
    #: The ground the class can work; None when nothing is left.
    workable: MultiPolygonModel | None
    footprint_m2: Area
    workable_m2: Area
    removed: WorkableRemoved
    left_out: list[WorkableLeftOut]
    limits: WorkableLimits
    elevation: WorkableElevation
    creeks: WorkableCreeks
    #: Where each number came from, in short lines ("Slope: …", "Creeks: …", "Limits: …").
    sources: list[str]
    #: Every assumption the split rests on, in sentences.
    assumptions: list[str]
    warnings: list[str]
