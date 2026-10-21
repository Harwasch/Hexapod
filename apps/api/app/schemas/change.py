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
    unseen or no evidence in its grid at the compared level). They are classified all the
    same, and counted in `counts` too.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    unchanged: int
    raised: int
    lowered: int


class ChangeUncertaintyRead(CamelModel):
    """Each epoch's placement uncertainty `u`, metres, as the threshold takes it: the larger
    of its uncertainty from the site frame (`fromFrameM`, its confidence grid's
    `alignmentCap.uncertaintyM`: 0 for the epoch that made the frame) and its alignment's
    own claim -- so a `coarse` epoch (a phone, a drone without RTK evidence) brings its
    metres, however well it was aligned visually. An `unaligned` epoch is refused (409).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    from_m: Finite
    to_m: Finite
    #: Each epoch's alignment class, as far as it may claim one (`survey` only for an RTK
    #: drone with RTK evidence in its frames).
    from_verdict: Literal["survey", "coarse"]
    to_verdict: Literal["survey", "coarse"]
    #: Each epoch's uncertainty from the site frame alone, metres.
    from_frame_m: Finite
    to_frame_m: Finite


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

    change_class: Literal["raised", "lowered", "insufficient"] = Field(alias="class")
    footprint: MultiPolygon
    #: Its cells at `areaLevel` (the compared level unless the map was too large to outline).
    cells: int
    area_m2: Finite
    #: True when every cell of it is low confidence; and how many are.
    low_confidence: bool
    low_confidence_cells: int
    #: The mean and the largest change of its layer's height (the ground, or the top
    #: surface), metres, positive up; null for an insufficient area (nothing compared).
    mean_delta_m: Finite | None
    max_abs_delta_m: Finite | None


#: What each layer is, in words a person reads.
GROUND_LABEL = "Ground: the bare earth"
VEGETATION_LABEL = (
    "Vegetation and objects: the top surface, at a higher threshold; never ground change"
)


class ChangeLayerRead(CamelModel):
    """One layer of the change map (ADR 0018 § API): `ground`, the bare earth (P5's `ground`
    and `groundSigma`), the primary product; or `vegetation`, the top surface (`dsm` and
    `sigma`: canopy, crops, and anything standing on the ground), with its own motion term
    so that wind in a crown is never a change and growth is never ground change.

    Per cell, `delta = h_to - h_from` of the layer's height against
    `t = thresholdSigma * sqrt(s_from^2 + s_to^2 + u_from^2 + u_to^2 + 2 motionM^2) + drift`
    (`drift` the ground envelope's on a slope, up to `grade * driftReachM`; 0 for
    vegetation), each cell against the other epoch's neighbourhood within
    `horizontalRadiusM`: `raised` above `t`, `lowered` below `-t`, else `unchanged`;
    `insufficient` where either measured the layer but they cannot be compared (no shared
    child, a partial child set, a saturated sigma, ground never observed).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    layer: Literal["ground", "vegetation"]
    label: str
    #: Which of the surface's heights the layer is made of.
    heights: Literal["ground", "dsm"]
    #: The least threshold over the layer's compared cells, metres: no change of this layer
    #: smaller than it can be seen anywhere between these dates. Null when none was compared.
    min_detectable_m: Finite | None
    #: The layer's own motion between visits, per epoch, metres (one standard deviation,
    #: vertical and horizontal): 0 for the ground, 0.25 for vegetation.
    motion_m: Finite
    #: The horizontal radius each cell was compared within, metres (whole cells).
    horizontal_radius_m: Finite
    #: How far from a cell its ground may have come (the half-diagonal of P5's ground
    #: window), metres: the reach of the drift bound. 0 for vegetation.
    drift_reach_m: Finite
    counts: ChangeCountsRead
    low_confidence: ChangeLowConfidenceRead
    #: The level the areas are outlined at (`level` unless the outline would have been over
    #: 2^20 cells or 2^18 cell edges: then classified again, by the same rule, coarser).
    area_level: int
    area_cell_m: Finite
    #: Raised and lowered first, then insufficient; largest first in each. Unchanged cells
    #: are counted (`counts`), never outlined: an area of them would read as confident
    #: wherever the comparison could not see.
    areas: list[ChangeArea]


class ChangeMapRead(CamelModel):
    """The change from epoch `fromEpochId` to `toEpochId` of one site (`change-v1`, ADR 0018):
    derived from their height surfaces, never observed and never from splat pixels. Two
    layers, each classified on its own: `ground` (the bare earth: the primary product) and
    `vegetation` (the top surface, at its own higher threshold). Per cell of the coarser of
    the two surfaces' levels, each side taken over the children both epochs measured (a
    partial child set is insufficient), with `u` each epoch's placement uncertainty
    (`uncertainty`). The 3 sigma rule is a first-principles choice, `uncalibrated` until a
    site scanned twice measures it.

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
    uncertainty: ChangeUncertaintyRead
    #: The bare earth's change: the primary product.
    ground: ChangeLayerRead
    #: The top surface's change: vegetation and anything standing, never read as ground.
    vegetation: ChangeLayerRead
    appearance: ChangeAppearanceRead
