"""Read models of an epoch's height surface and of the change between two dates (M5, ADR 0018).

What the API says about the heights an epoch's current revision measured
(`EpochSurfaceRead`, on every epoch), and the change between two epochs it derives from
them (`ChangeMapRead`, `GET /sites/{id}/change`). The surfaces themselves are
`hexapod.cellgrid` files beside the tiles; these are what the API computes from them.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.geojson import MultiPolygon

Finite = Annotated[float, Field(allow_inf_nan=False)]


class EpochSurfaceRead(CamelModel):
    """The height surface an epoch's current revision shipped: per cell of its confidence
    grid's finest plane, its top surface (`dsm`), bare ground and spread, in centimetres
    above the site frame's origin. Null on an epoch means not measured: the run that made
    its current revision measured none (a splat upload, or a run before the surface stage).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    status: Literal["measured"]
    #: The revision (asset) the surface was measured on.
    asset_id: uuid.UUID
    #: The `hexapod.cellgrid` file beside the tiles.
    url: str
    #: Its cell, metres, and lattice level (`0.25 m * 2**level`): its confidence grid's own.
    cell_m: Finite
    level: int
    #: How it was estimated (`surface-v1`), and whether that was ever calibrated on a site
    #: scanned twice: `uncalibrated` until one is.
    policy: str
    calibration: str
    #: Cells of its plane, and those with a surface (enough samples to estimate one).
    cells: int
    surfaced_cells: int
    #: The median spread of its surfaced cells, metres (null when none is surfaced).
    median_sigma_m: Finite | None
    #: The ellipsoidal height of the frame origin its heights are above, metres.
    origin_height_m: Finite
    #: The window its bare ground was taken the least over, metres (5 m from P5; null when
    #: the descriptor does not say): uncalibrated, like `policy`'s estimators.
    ground_window_m: Finite | None


#: What every change map says it is, in words a person reads.
CHANGE_LABEL = "Derived from height surfaces, not observed"
APPEARANCE_REASON = (
    "needs the new photos compared with the old epoch at fixed geometry; never splat pixels"
)
ChangeClass = Literal["unchanged", "raised", "lowered", "insufficient"]


class ChangeCountsRead(CamelModel):
    """Cells of each class at the compared level (`level`).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    unchanged: int
    raised: int
    lowered: int
    insufficient: int


class ChangeLowConfidenceRead(CamelModel):
    """Of each compared class, the cells either epoch's confidence does not vouch for (weak,
    unseen or no evidence in its grid at the compared level) or whose sigma saturates. They
    are classified all the same, and counted in `counts` too.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    unchanged: int
    raised: int
    lowered: int


class ChangeUncertaintyRead(CamelModel):
    """Each epoch's placement uncertainty from the site frame, metres, as its confidence
    grid states it (`alignmentCap.uncertaintyM`): 0 for the epoch that made the frame.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    from_m: Finite
    to_m: Finite


class ChangeAppearanceRead(CamelModel):
    """Change seen in appearance only (a repaint, leaf-off): not computed, and why.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    status: Literal["not-computed"]
    reason: str


class ChangeArea(CamelModel):
    """One 4-connected region of cells of one class (raised and lowered regions are also of
    one confidence), outlined on the cells' exact edges and placed on the globe.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    change_class: ChangeClass = Field(alias="class")
    footprint: MultiPolygon
    #: Its cells at `areaLevel` (the compared level unless the map was too large to outline).
    cells: int
    area_m2: Finite
    #: True when every cell of it is low confidence; and how many are.
    low_confidence: bool
    low_confidence_cells: int
    #: The mean and the largest change of its top surface (dsm), metres, positive up; null
    #: for an insufficient area (no child both epochs measured, so nothing to compare).
    mean_delta_m: Finite | None
    max_abs_delta_m: Finite | None
    #: The mean change of its bare ground, metres; null where `meanDeltaM` is.
    mean_ground_delta_m: Finite | None


class ChangeMapRead(CamelModel):
    """The change from epoch `fromEpochId` to `toEpochId` of one site (`change-v1`, ADR 0018):
    derived from their height surfaces, never observed and never from splat pixels.

    Per cell of the coarser of the two surfaces' levels, each side taken over only the
    children both epochs measured: `delta = dsm_to - dsm_from` against
    `t = thresholdSigma * sqrt(sigma_from^2 + sigma_to^2 + u_from^2 + u_to^2)`, with `u` each
    epoch's uncertainty from the site frame (`uncertainty`): `raised` above `t`, `lowered`
    below `-t`, else `unchanged`; `insufficient` where either measured something but they
    share no child. A cell compared over fewer children than either side measured is low
    confidence. The 3 sigma rule is a first-principles choice, `uncalibrated` until a site
    scanned twice measures it.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    policy: Literal["change-v1"]
    derived: Literal[True]
    label: str
    calibration: Literal["uncalibrated"]
    site_id: uuid.UUID
    from_epoch_id: uuid.UUID
    to_epoch_id: uuid.UUID
    #: The revisions (assets) whose surfaces were compared: each epoch's current one.
    from_asset_id: uuid.UUID
    to_asset_id: uuid.UUID
    frame: Literal["tileset"]
    #: The compared level and its cell (`0.25 m * 2**level`): the coarser of the two
    #: surfaces' own (`nativeLevel`), or coarser still when the two windows together were
    #: over 2^22 cells.
    level: int
    native_level: int
    cell_m: Finite
    threshold_sigma: Finite
    #: The least threshold over the compared cells, metres: no change smaller than this can
    #: be seen anywhere between these dates. Null when no cell was compared.
    min_detectable_m: Finite | None
    uncertainty: ChangeUncertaintyRead
    counts: ChangeCountsRead
    low_confidence: ChangeLowConfidenceRead
    appearance: ChangeAppearanceRead
    #: The level the areas are outlined at (`level` unless the outline would have been over
    #: 2^20 cells or 2^18 cell edges: then classified again, by the same rule, coarser).
    area_level: int
    area_cell_m: Finite
    #: Raised and lowered first, then insufficient, then unchanged; largest first in each.
    areas: list[ChangeArea]
