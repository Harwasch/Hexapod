"""The change between two epochs of a site, as the API answers it (M5, ADR 0018 § API).

`change` reads each epoch's current revision's height surface and confidence grid
(`epoch_maps`), loads their planes from the public bucket through the grid cache that
weak areas use (`confidence_maps.load_grid`), and measures the change between them
(`app/analysis/change.py`). It refuses what it cannot compare honestly, each with a code:

* 404 when the site or either epoch is not there, or an epoch is another site's;
* 422 when both are the same epoch;
* 409 `epoch-not-published` when either epoch is rejected or superseded;
* 409 `no-site-frame` when the site has no frame to place cells by;
* 409 `surface-not-measured` when either epoch's current revision has no height surface
  (it shows no tiles, or its run measured none) -- one side missing everywhere is not a
  map of insufficient cells, it is no map;
* 409 `confidence-not-measured` when either has no confidence grid: its uncertainty from
  the site frame (`alignmentCap`) is part of the `u` of the threshold, and nothing vouches
  for its cells without one;
* 409 `epoch-unaligned` when either epoch was never aligned to the site (its alignment's
  verdict, as far as it may claim one -- `epochs.claimed_verdict` -- is `unaligned`, or
  none was recorded): placed by its georeference alone, its heights have no place in the
  frame another epoch's could be compared with (review B5a's ruling);
* 409 `frames-differ` when a surface's heights are above another origin height than the
  site frame's (its `vertical.originHeightM`; to the millimetre, a tenth of the heights'
  own unit) -- the two would differ by the offset everywhere;
* 409 `surface-unreadable` / `confidence-unreadable` when a grid cannot be read, or an
  epoch's confidence grid is coarser than its surface (they must be one plane).

**Each epoch's `u`** is the larger of its uncertainty from the site frame (its confidence
grid's `alignmentCap.uncertaintyM`: 0 for the epoch that made the frame) and its
alignment's own claim (`alignment.uncertaintyM`, `UNRECORDED_UNCERTAINTY_M` when it states
none). A `survey` epoch's claim is centimetres; a `coarse` one's -- a phone, a drone
without RTK evidence, anything aligned into a coarse reference -- is metres, so a pair
with a coarse epoch claims no change under metres however well it was aligned visually
(Harrison's rule: a phone epoch is never presented as survey grade). The answer states
each epoch's verdict and both figures.

Every answer says it is derived, not observed, and uncalibrated.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from sqlalchemy.orm import Session

from app.analysis import cell_grid
from app.analysis import change as analysis
from app.analysis.weak_areas import site_projection
from app.models import Epoch, EpochMap, Site, SiteFrame
from app.models.enums import EpochMapKind, EpochStatus
from app.schemas.change import (
    APPEARANCE_REASON,
    CHANGE_LABEL,
    ChangeAppearanceRead,
    ChangeArea,
    ChangeCountsRead,
    ChangeLowConfidenceRead,
    ChangeMapRead,
    ChangeUncertaintyRead,
)
from app.schemas.geojson import MultiPolygon
from app.services import epoch_maps, epochs
from app.services.confidence_maps import NO_FRAME, NOT_MEASURED, UNREADABLE, load_grid
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.storage import ObjectStorage

NOT_PUBLISHED = "epoch-not-published"
UNALIGNED = "epoch-unaligned"
SURFACE_NOT_MEASURED = "surface-not-measured"
SURFACE_UNREADABLE = "surface-unreadable"
FRAMES_DIFFER = "frames-differ"
#: A surface's origin height may differ from the frame's by this much, metres: a tenth of
#: the heights' own centimetre, so only the same number (as JSON carried it) passes.
ORIGIN_TOLERANCE_M = 0.001


def _epoch(db: Session, site_id: uuid.UUID, epoch_id: uuid.UUID) -> Epoch:
    epoch = db.get(Epoch, epoch_id)
    if epoch is None or epoch.site_id != site_id:
        raise NotFoundError("epoch", epoch_id)
    return epoch


def _published(epoch: Epoch) -> None:
    if epoch.status != EpochStatus.PUBLISHED.value:
        raise ConflictError(
            f"epoch {epoch.id} is {epoch.status}: only published epochs are compared",
            code=NOT_PUBLISHED,
        )


def _maps(db: Session, epoch: Epoch) -> tuple[uuid.UUID, EpochMap, EpochMap]:
    """The epoch's current revision and its surface and confidence maps, or a 409 saying
    which it lacks."""
    current = epochs.current_asset(db, epoch.id)
    if current is None:
        raise ConflictError(
            f"epoch {epoch.id} shows no tiles, so no height surface of it was measured",
            code=SURFACE_NOT_MEASURED,
        )
    rows = epoch_maps.maps_of_asset(db, current.id)
    surface = epoch_maps.map_of(rows, current.id, EpochMapKind.SURFACE)
    if surface is None:
        raise ConflictError(
            f"epoch {epoch.id} has no height surface: the run that made what it shows "
            "measured none (a splat upload, or a run from before the surface stage)",
            code=SURFACE_NOT_MEASURED,
        )
    graded = epoch_maps.map_of(rows, current.id, EpochMapKind.CONFIDENCE)
    if graded is None:
        raise ConflictError(
            f"epoch {epoch.id} has no confidence map, so neither its uncertainty from the "
            "site frame nor how well its cells were seen is known",
            code=NOT_MEASURED,
        )
    if graded.descriptor.get("policy") != "weak-v1":
        raise ConflictError(
            f"epoch {epoch.id}'s confidence map follows policy "
            f"{graded.descriptor.get('policy')!r}, which this API cannot read weak cells by",
            code=UNREADABLE,
        )
    return current.id, surface, graded


def _uncertainty(epoch: Epoch, graded: EpochMap) -> float:
    cap = graded.descriptor.get("alignmentCap") or {}
    value = cap.get("uncertaintyM")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConflictError(
            f"epoch {epoch.id}'s confidence map states no usable uncertainty ({value!r})",
            code=UNREADABLE,
        )
    return float(value)


@dataclass(frozen=True)
class _Placement:
    """How well an epoch is placed for a comparison: `u`, and what it was made of."""

    verdict: str
    frame_m: float
    claim_m: float

    @property
    def u_m(self) -> float:
        return max(self.frame_m, self.claim_m)


def _placement(epoch: Epoch, graded: EpochMap) -> _Placement:
    """The epoch's alignment class and `u` (module docstring), or a 409 when it was never
    aligned to the site."""
    verdict = epochs.claimed_verdict(epoch.alignment, epoch.platform)
    if verdict not in ("survey", "coarse"):
        label = (epoch.alignment or {}).get("label") if isinstance(epoch.alignment, dict) else None
        raise ConflictError(
            f"epoch {epoch.id} was never aligned to its site ({label or verdict or 'no alignment recorded'}): "
            "placed by its georeference alone, its heights cannot be compared with another "
            "epoch's",
            code=UNALIGNED,
        )
    alignment = epoch.alignment if isinstance(epoch.alignment, dict) else {}
    claim = alignment.get("uncertaintyM")
    claim_m = (
        float(claim)
        if not isinstance(claim, bool)
        and isinstance(claim, int | float)
        and math.isfinite(claim)
        and claim > 0
        else epochs.UNRECORDED_UNCERTAINTY_M
    )
    return _Placement(verdict, _uncertainty(epoch, graded), claim_m)


def _same_frame(epoch: Epoch, surface: EpochMap, frame: SiteFrame) -> None:
    origin = (surface.descriptor.get("vertical") or {}).get("originHeightM")
    if (
        isinstance(origin, bool)
        or not isinstance(origin, int | float)
        or abs(float(origin) - frame.height) > ORIGIN_TOLERANCE_M
    ):
        raise ConflictError(
            f"epoch {epoch.id}'s heights are above an origin at {origin!r} m, but site "
            f"{frame.site_id}'s frame is at {frame.height} m: they cannot be compared",
            code=FRAMES_DIFFER,
        )


def _footprint(geometry: BaseGeometry) -> MultiPolygon:
    mapped = mapping(geometry)
    return MultiPolygon.model_validate(
        {
            "type": "MultiPolygon",
            "coordinates": [
                [[[round(float(p[0]), 9), round(float(p[1]), 9)] for p in ring] for ring in polygon]
                for polygon in mapped["coordinates"]
            ],
        }
    )


def _metres(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def change(
    db: Session,
    storage: ObjectStorage,
    site_id: uuid.UUID,
    from_id: uuid.UUID,
    to_id: uuid.UUID,
) -> ChangeMapRead:
    """The change from epoch `from_id` to `to_id` of site `site_id` (module docstring)."""
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    if from_id == to_id:
        raise InvalidInputError("a change is between two different epochs")
    # Both found (404) before either is judged (409): a missing epoch is the answer.
    before, after = _epoch(db, site_id, from_id), _epoch(db, site_id, to_id)
    _published(before)
    _published(after)
    frame = db.get(SiteFrame, site_id)
    if frame is None:
        raise ConflictError(
            f"site {site_id} has no frame, so its cells cannot be placed on the globe",
            code=NO_FRAME,
        )
    asset_a, surface_a, graded_a = _maps(db, before)
    asset_b, surface_b, graded_b = _maps(db, after)
    for epoch, surface in ((before, surface_a), (after, surface_b)):
        _same_frame(epoch, surface, frame)
    placed_a, placed_b = _placement(before, graded_a), _placement(after, graded_b)
    ua, ub = placed_a.u_m, placed_b.u_m
    grids = [load_grid(storage, row) for row in (surface_a, surface_b, graded_a, graded_b)]
    for epoch, heights, graded in ((before, grids[0], grids[2]), (after, grids[1], grids[3])):
        # A surface is written on its confidence grid's finest plane; one finer than its
        # grid has cells nothing vouches for at its level, whatever its descriptor says.
        if graded.finest.level > heights.finest.level:
            raise ConflictError(
                f"epoch {epoch.id}'s confidence grid (level {graded.finest.level}) is coarser "
                f"than its height surface (level {heights.finest.level}): they are not one "
                "plane, so its cells cannot be vouched for",
                code=UNREADABLE,
            )
    try:
        found = analysis.change_map(
            analysis.Surface.of(grids[0].finest),
            analysis.Surface.of(grids[1].finest),
            ua,
            ub,
            site_projection(frame.lat, frame.lon, frame.height),
            confidence_a=grids[2],
            confidence_b=grids[3],
        )
    except cell_grid.CellGridError as error:
        raise ConflictError(
            f"the height surfaces cannot be compared: {error}", code=SURFACE_UNREADABLE
        ) from error
    planes, drawn = found.planes, found.drawn
    counts, lows = planes.counts(), planes.low_counts()
    area_cell = drawn.cell_m
    return ChangeMapRead(
        policy="change-v1",
        derived=True,
        label=CHANGE_LABEL,
        calibration="uncalibrated",
        site_id=site_id,
        from_epoch_id=before.id,
        to_epoch_id=after.id,
        from_asset_id=asset_a,
        to_asset_id=asset_b,
        frame="tileset",
        level=planes.level,
        native_level=planes.native_level,
        cell_m=planes.cell_m,
        threshold_sigma=analysis.THRESHOLD_SIGMA,
        min_detectable_m=_metres(planes.min_detectable_m),
        uncertainty=ChangeUncertaintyRead(
            from_m=ua,
            to_m=ub,
            from_verdict=placed_a.verdict,
            to_verdict=placed_b.verdict,
            from_frame_m=placed_a.frame_m,
            to_frame_m=placed_b.frame_m,
        ),
        counts=ChangeCountsRead(**counts),
        low_confidence=ChangeLowConfidenceRead(**lows),
        appearance=ChangeAppearanceRead(status="not-computed", reason=APPEARANCE_REASON),
        area_level=drawn.level,
        area_cell_m=area_cell,
        areas=[
            ChangeArea(
                change_class=area.change,
                footprint=_footprint(area.footprint),
                cells=area.cells,
                area_m2=round(area.cells * area_cell * area_cell, 3),
                low_confidence=area.low_confidence,
                low_confidence_cells=area.low_confidence_cells,
                mean_delta_m=_metres(area.mean_delta_m),
                max_abs_delta_m=_metres(area.max_abs_delta_m),
                mean_ground_delta_m=_metres(area.mean_ground_delta_m),
            )
            for area in found.areas
        ],
    )
