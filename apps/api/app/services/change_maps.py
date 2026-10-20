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
  the site frame (`alignmentCap`) is the `u` of the threshold, and nothing vouches for its
  cells without one;
* 409 `frames-differ` when a surface's heights are above another origin height than the
  site frame's (its `vertical.originHeightM`; to the millimetre, a tenth of the heights'
  own unit) -- the two would differ by the offset everywhere;
* 409 `surface-unreadable` / `confidence-unreadable` when a grid cannot be read.

Every answer says it is derived, not observed, and uncalibrated.
"""

from __future__ import annotations

import math
import uuid

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from sqlalchemy.orm import Session

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
SURFACE_NOT_MEASURED = "surface-not-measured"
FRAMES_DIFFER = "frames-differ"
#: A surface's origin height may differ from the frame's by this much, metres: a tenth of
#: the heights' own centimetre, so only the same number (as JSON carried it) passes.
ORIGIN_TOLERANCE_M = 0.001


def _epoch(db: Session, site_id: uuid.UUID, epoch_id: uuid.UUID) -> Epoch:
    epoch = db.get(Epoch, epoch_id)
    if epoch is None or epoch.site_id != site_id:
        raise NotFoundError("epoch", epoch_id)
    if epoch.status != EpochStatus.PUBLISHED.value:
        raise ConflictError(
            f"epoch {epoch.id} is {epoch.status}: only published epochs are compared",
            code=NOT_PUBLISHED,
        )
    return epoch


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
    before, after = _epoch(db, site_id, from_id), _epoch(db, site_id, to_id)
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
    ua, ub = _uncertainty(before, graded_a), _uncertainty(after, graded_b)
    grids = [load_grid(storage, row) for row in (surface_a, surface_b, graded_a, graded_b)]
    found = analysis.change_map(
        analysis.Surface.of(grids[0].finest),
        analysis.Surface.of(grids[1].finest),
        ua,
        ub,
        site_projection(frame.lat, frame.lon, frame.height),
        confidence_a=grids[2],
        confidence_b=grids[3],
    )
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
        uncertainty=ChangeUncertaintyRead(from_m=ua, to_m=ub),
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
