"""The Fog of Knowing over a saved Land (policy fog-v1, `app.analysis.knowledge`)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, HttpUrl

from app.schemas.base import CamelModel
from app.schemas.geojson import Polygon, Position
from app.schemas.grids import LonLatGrid

Finite = Annotated[float, Field(allow_inf_nan=False)]
KnowledgeClassId = Literal["centimetre", "field", "aerial", "satellite", "regional"]


class KnowledgeClassRead(CamelModel):
    id: KnowledgeClassId
    label: str
    quality: Finite
    half_life_days: Finite
    max_resolution_m: Finite | None
    example: str


class KnowledgePolicyRead(CamelModel):
    version: str
    classes: list[KnowledgeClassRead]
    undated_factor: Finite
    field_reach_m: dict[str, Finite]
    foggy_threshold: Finite
    formula: str


class KnowledgeProvenance(CamelModel):
    provider: str
    license: str
    attribution: str
    url: HttpUrl | None = None


class KnowledgeLink(CamelModel):
    """Where the record lives in the app: a scan site, an inventory feature, a land raster, an
    aligned archive image, a research evidence row or a field survey."""

    kind: Literal["site", "inventory", "raster", "registration", "evidence", "survey"]
    id: str
    detail: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRecordRead(CamelModel):
    key: str
    kind: Literal[
        "capture",
        "inspection",
        "raster-observation",
        "archive-alignment",
        "research-evidence",
        "survey",
    ]
    title: str
    #: The class the record earns on this grid (after any one-class demotion).
    class_id: KnowledgeClassId
    #: The class it counts as: regional for an unlocated record that would earn more.
    counted_class_id: KnowledgeClassId
    class_note: str | None
    resolution_m: Finite | None
    observed_at: datetime | None
    observed_label: str
    recorded_at: datetime
    age_days: Finite | None
    #: The credit it gives every cell it covers (constant per record under fog-v1).
    peak_credit: Finite
    arithmetic: str
    located: bool
    positional_uncertainty_m: Finite | None
    #: Inside cells the record covers on this grid.
    cells: int
    provenance: KnowledgeProvenance
    link: KnowledgeLink


class NotCountedRead(CamelModel):
    title: str
    reason: str


class KnowledgeSummary(CamelModel):
    inside_cells: int
    #: Share of inside cells by the class of the record that sets them; 'none' included.
    by_best_class: dict[str, Finite]
    foggy_fraction: Finite
    median_density: Finite | None


class KnowledgeGridRead(CamelModel):
    land_id: uuid.UUID
    boundary_revision: int
    as_of: datetime
    #: Density per cell: u8, 0..250 = density x 250 (scale 0.004), 255 = outside the land.
    density: LonLatGrid
    #: The record that sets each cell: u16 index into `records`, 65535 = no record.
    best: LonLatGrid
    records: list[KnowledgeRecordRead]
    summary: KnowledgeSummary
    policy: KnowledgePolicyRead
    not_counted: list[NotCountedRead]


class KnowledgeCellRecord(CamelModel):
    record: KnowledgeRecordRead
    credit: Finite
    best: bool
    arithmetic: str


class KnowledgeCellRead(CamelModel):
    longitude: Finite
    latitude: Finite
    column: int | None
    row: int | None
    inside: bool
    density: Finite | None
    cell_m: Finite
    #: The cell's corners, west/south/east/north, for the card to outline it.
    bounds: Annotated[list[Finite], Field(min_length=4, max_length=4)] | None
    records: list[KnowledgeCellRecord]
    not_counted: list[NotCountedRead]
    policy_version: str


class KnowledgeScoutRequest(CamelModel):
    threshold: Finite = Field(default=0.5, ge=0.05, le=1)
    budget_m2: Finite | None = Field(default=None, gt=0, le=5e7)
    #: Where the operator is looking (lon, lat): ties go to the cells nearest it.
    focus: Position | None = None
    max_areas: int = Field(default=3, ge=1, le=6)
    cell_m: Finite = Field(default=20, ge=10, le=100)


class KnowledgeScoutArea(CamelModel):
    footprint: Polygon
    area_m2: Finite
    cells: int
    mean_density: Finite
    best_class_counts: dict[str, int]
    reason: str


class KnowledgeScoutRead(CamelModel):
    areas: list[KnowledgeScoutArea]
    threshold: Finite
    budget_m2: Finite | None
    truncated_by_budget: bool
    forecast: str
    policy_version: str
