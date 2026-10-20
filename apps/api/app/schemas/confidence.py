"""Read models of an epoch's confidence map (M4, ADR 0016).

What the API says about how well each part of an epoch was seen: the summary of the grid
its current revision shipped (`EpochConfidenceRead`, on every epoch), the weak cells as
re-scan zones (`WeakAreasRead`), and the cells a re-scan left weak (`FillTargetsRead`).
The grid itself is a `hexapod.cellgrid` file beside the tiles; these are what the API
computes from it.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import Polygon, Position

Finite = Annotated[float, Field(allow_inf_nan=False)]


class AlignmentCapRead(CamelModel):
    """How the epoch's placement capped its scores (ADR 0016, C4): a cell may only claim
    evidence when the epoch's uncertainty from the site frame is under half a cell.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    #: The epoch's uncertainty relative to the site frame, metres (0 for the epoch that
    #: made the frame).
    uncertainty_m: Finite
    #: `min(1, cell / (2 * uncertaintyM))`: what every score was multiplied by.
    factor: Finite
    #: The site's published epochs before this one when it was measured.
    prior_epochs: int
    #: Which file the uncertainty was read from (`epoch_alignment.json`, `georef.json`, ...).
    source: str


class EpochConfidenceRead(CamelModel):
    """The confidence map an epoch's current revision shipped, summarised at its finest
    level. Null on an epoch means not measured: the run that made its current revision
    measured none (a splat upload carries no capture evidence).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    status: Literal["measured"]
    #: The revision (asset) the map was measured on.
    asset_id: uuid.UUID
    #: The `hexapod.cellgrid` file beside the tiles.
    url: str
    #: The finest level's cell, metres, and its lattice level (`0.25 m * 2**level`).
    cell_m: Finite
    level: int
    #: Which cells count as weak: `weak-v1` (score under 128, or unseen).
    policy: str
    #: Weak cells over the cells this epoch speaks for (scored, or expected and unseen).
    weak_share: Finite
    weak_cells: int
    scored_cells: int
    unseen_cells: int
    #: Per reason (`views`, `spread`, `gsd`, `accuracy`, `unseen`, `alignment`), how many
    #: weak cells carry it.
    reasons: dict[str, int]
    alignment_cap: AlignmentCapRead


#: The words every weak area's pattern carries: how to fly it again is a rule of thumb.
HEURISTIC_LABEL = (
    "How to fly each area again is a rule of thumb from photogrammetry practice (a UX "
    "heuristic), not a calibrated plan; which cells are weak is weak-v1 (ADR 0016)."
)


class WeakAreasRequest(CamelModel):
    """Which of an epoch's weak cells to turn into re-scan zones."""

    #: Square metres the zones may cover together (plus at most one cell); the weakest
    #: cells first, then those nearest `focus`.
    budget_m2: Finite | None = Field(default=None, gt=0, le=5e7)
    #: Where the operator is looking (lon, lat): ties go to the cells nearest it.
    focus: Position | None = None
    max_areas: int = Field(default=3, ge=1, le=8)


class WeakArea(CamelModel):
    """One re-scan zone: weak cells merged, smoothed and placed on the globe.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    footprint: Polygon
    area_m2: Finite
    #: Weak cells (at the grid's finest level) inside it.
    cells: int
    #: Mean score of its scored cells (1..255; 0 when every cell in it is unseen).
    mean_score: Finite
    #: Per reason, how many of its cells carry it.
    reasons: dict[str, int]
    #: The tallest column in it (its cells' relief) and how far across it is, metres.
    relief_m: Finite
    diameter_m: Finite
    #: How it might be flown again (a UX heuristic, see `heuristic`).
    pattern: Literal["nadir", "oblique", "orbit"]
    #: The ground sampling to fly for, metres a pixel: the capture's median, when it was
    #: the detail that fell short and the capture had a metric scale; else null.
    target_gsd_m: Finite | None
    #: The camera pitch the pattern suggests, degrees (-90 straight down).
    gimbal_pitch_deg: Finite
    hint: str


class WeakAreasRead(CamelModel):
    """An epoch's weak cells as re-scan zones, largest first.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    epoch_id: uuid.UUID
    #: The revision whose map this is (the epoch's current asset).
    asset_id: uuid.UUID
    policy: Literal["weak-v1"]
    cell_m: Finite
    level: int
    #: Weak cells, and the cells the epoch speaks for (scored, or expected and unseen).
    weak_cells: int
    total_cells: int
    truncated_by_budget: bool
    heuristic: str
    areas: list[WeakArea]
