"""An epoch's confidence map as re-scan zones, and what a re-scan left weak (M4, ADR 0016).

`weak_areas` reads the map of the epoch's current revision (`epoch_maps`), loads its grid
from the bucket the tiles are published in, and turns the weak cells into zones on the
globe (`app/analysis/weak_areas.py`). `fill_targets` compares an epoch's grid with the
earlier measured epochs of its site and answers the cells that are still weak after the
re-scan (`app/analysis/rescan_gate.py`): all an inferred fill may fill (F4).

**Loading a grid.** The map's URL is the file beside the tiles in the generation the run
published (or the attach that last moved the asset cut); its key is that URL less the
bucket's public base, read from the public bucket -- the published copy, never the run's
own key in the private one, which a Refine rewrites in place. A grid is at most
`MAX_GRID_BYTES` on the wire (refused past it, before it is read), and decoded no further
than its descriptor declares (`app/analysis/cell_grid.py`). Decoded grids are kept in a
small LRU bounded by bytes and keyed by **URL and the sha256 of the whole descriptor**: a
published generation is immutable, so its URL alone would do, but a single bucket's run
keys are rewritten in place. The planes' own sha256 (which `decode` checks against the
bytes) is not enough: the same bytes under a descriptor with another origin or window are
another grid, so the key is a hash of everything that says how to read them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis import cell_grid, rescan_gate
from app.analysis.weak_areas import cell_patches, site_plane, site_projection
from app.models import Asset, Epoch, EpochMap, Site, SiteFrame
from app.models.enums import AssetSourceKind, EpochMapKind, EpochStatus
from app.schemas.confidence import (
    HEURISTIC_LABEL,
    FillTargetsRead,
    WeakArea,
    WeakAreasRead,
    WeakAreasRequest,
)
from app.schemas.geojson import Polygon
from app.services import epoch_maps, epochs
from app.services.errors import ConflictError, NotFoundError
from app.storage import ObjectStorage

log = logging.getLogger(__name__)

NOT_MEASURED = "confidence-not-measured"
UNREADABLE = "confidence-unreadable"
NO_FRAME = "no-site-frame"

#: A grid file on the wire. The pipeline's largest (2^22 cells a level, eight levels over
#: it, three bytes a cell) is ~17 MB before gzip; a file past this is not one it wrote.
MAX_GRID_BYTES = 32 * 1024 * 1024
#: Decoded grids kept, by their planes' bytes.
CACHE_BYTES = 96 * 1024 * 1024


class GridCache:
    """Decoded grids by `(url, descriptor sha256)`, least recently used out first, bounded
    by bytes."""

    def __init__(self, max_bytes: int = CACHE_BYTES) -> None:
        self.max_bytes = max_bytes
        self._grids: OrderedDict[tuple[str, str], tuple[cell_grid.Grid, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, str]) -> cell_grid.Grid | None:
        with self._lock:
            found = self._grids.get(key)
            if found is None:
                self.misses += 1
                return None
            self._grids.move_to_end(key)
            self.hits += 1
            return found[0]

    def put(self, key: tuple[str, str], grid: cell_grid.Grid) -> None:
        size = sum(int(array.nbytes) for plane in grid.planes for array in plane.channels.values())
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._grids:
                return
            self._grids[key] = (grid, size)
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, (_, gone) = self._grids.popitem(last=False)
                self._bytes -= gone

    def clear(self) -> None:
        with self._lock:
            self._grids.clear()
            self._bytes = 0


GRIDS = GridCache()


def storage_key(storage: ObjectStorage, url: str) -> str | None:
    """The key of `url` in `storage`, or None when `url` is not in it (another host)."""
    base = storage.public_url("")
    if not url.startswith(base):
        return None
    key = url[len(base) :]
    if not key or any(c in key for c in "?#") or ".." in key.split("/"):
        return None
    return key


#: What a map's grid is called in an answer, and its code when it cannot be read.
_NOUNS = {"confidence": "confidence grid", "surface": "height surface"}


def descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """The sha256 of the whole descriptor, canonical JSON: it holds the planes' own sha256
    and everything that says how to read them (levels, windows, channels, kind)."""
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_grid(storage: ObjectStorage, row: EpochMap) -> cell_grid.Grid:
    """The decoded grid of `row`, from the cache or the bucket; a 409 `<kind>-unreadable`
    (`confidence-unreadable`, `surface-unreadable`) when it is not in this deployment's
    bucket, is missing, too large or damaged."""
    cache_key = (row.url, descriptor_sha256(row.descriptor))
    cached = GRIDS.get(cache_key)
    if cached is not None:
        return cached
    kind = row.kind or str(row.descriptor.get("kind") or "confidence")
    noun = _NOUNS.get(kind, f"{kind} grid")
    code = UNREADABLE if kind == "confidence" else f"{kind}-unreadable"
    key = storage_key(storage, row.url)
    if key is None:
        raise ConflictError(f"the {noun} {row.url} is not in this deployment's bucket", code=code)
    head = storage.head_object(key)
    if head is None:
        raise ConflictError(f"the {noun} {row.url} is missing", code=code)
    if head.size > MAX_GRID_BYTES:
        raise ConflictError(
            f"the {noun} {row.url} is {head.size} bytes, over {MAX_GRID_BYTES}", code=code
        )
    data, size = storage.get_object_tail(key, MAX_GRID_BYTES + 1)
    if size > MAX_GRID_BYTES:
        raise ConflictError(f"the {noun} {row.url} grew past {MAX_GRID_BYTES} bytes", code=code)
    try:
        grid = cell_grid.decode(row.descriptor, data)
    except cell_grid.CellGridError as error:
        log.warning("%s: %s is unreadable: %s", kind, row.url, error)
        raise ConflictError(f"the {noun} {row.url} cannot be read: {error}", code=code) from error
    if grid.kind != kind:
        raise ConflictError(f"the {noun} {row.url} is a {grid.kind} grid", code=code)
    GRIDS.put(cache_key, grid)
    return grid


def flyable(site: Site) -> BaseGeometry:
    """The ground a re-scan of `site` may fly (lon/lat): weak-area zones are clipped to it,
    as the fog scout's are. Today that is the site's boundary, and its no-go areas are the
    boundary's holes (its exclusions) only. The hook for explicit no-go zones (ADR 0016
    § API 5): when a site can carry them, they are subtracted here, and every zone stays
    out of them with no other change."""
    return to_shape(site.boundary)


def epoch_of_site(db: Session, site_id: uuid.UUID, epoch_id: uuid.UUID) -> Epoch:
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    epoch = db.get(Epoch, epoch_id)
    if epoch is None or epoch.site_id != site_id:
        raise NotFoundError("epoch", epoch_id)
    return epoch


def measured_map(db: Session, epoch: Epoch) -> EpochMap:
    """The confidence map of the epoch's current revision, or a 409 saying it has none."""
    current = epochs.current_asset(db, epoch.id)
    row = (
        epoch_maps.map_of(
            epoch_maps.maps_of_asset(db, current.id), current.id, EpochMapKind.CONFIDENCE
        )
        if current is not None
        else None
    )
    if current is None:
        raise ConflictError(
            f"epoch {epoch.id} shows no tiles ({epoch.status}), so nothing of it was measured",
            code=NOT_MEASURED,
        )
    if row is None:
        raise ConflictError(
            f"epoch {epoch.id} has no confidence map: the run that made what it shows "
            "measured none (a splat upload carries no capture evidence to measure)",
            code=NOT_MEASURED,
        )
    return row


def _polygon(geometry: Any) -> Polygon:
    mapped = mapping(geometry)
    return Polygon.model_validate(
        {
            "type": "Polygon",
            "coordinates": [
                [[round(float(p[0]), 9), round(float(p[1]), 9)] for p in ring]
                for ring in mapped["coordinates"]
            ],
        }
    )


def weak_areas(
    db: Session,
    storage: ObjectStorage,
    site_id: uuid.UUID,
    epoch_id: uuid.UUID,
    request: WeakAreasRequest,
) -> WeakAreasRead:
    """The epoch's weak cells as at most `request.max_areas` re-scan zones (module doc)."""
    epoch = epoch_of_site(db, site_id, epoch_id)
    row = measured_map(db, epoch)
    frame = db.get(SiteFrame, site_id)
    if frame is None:
        raise ConflictError(
            f"site {site_id} has no frame, so its cells cannot be placed on the globe",
            code=NO_FRAME,
        )
    if row.descriptor.get("policy") != "weak-v1":
        raise ConflictError(
            f"epoch {epoch.id}'s map follows policy {row.descriptor.get('policy')!r}, which "
            "this API cannot read weak cells by",
            code=UNREADABLE,
        )
    grid = load_grid(storage, row)
    plane = grid.finest
    mask = cell_grid.weak_mask(plane)
    summary = row.summary
    relief_unit = next(
        (
            float(c.get("unitM") or 0.25)
            for c in row.descriptor.get("channels", [])
            if isinstance(c, dict) and c.get("name") == "relief"
        ),
        0.25,
    )
    gsd = summary.get("medianGsdMm")
    projection = site_projection(frame.lat, frame.lon, frame.height)
    site = db.get(Site, site_id)
    assert site is not None  # epoch_of_site found it
    found = cell_patches(
        plane,
        mask,
        projection,
        allowed=site_plane(flyable(site), projection),
        max_areas=request.max_areas,
        budget_m2=request.budget_m2,
        focus=(request.focus[0], request.focus[1]) if request.focus is not None else None,
        median_gsd_mm=float(gsd)
        if isinstance(gsd, int | float) and summary.get("metric")
        else None,
        relief_unit_m=relief_unit,
        total_cells=int(cell_grid.covered_mask(plane).sum()),
    )
    areas = [
        WeakArea(
            footprint=_polygon(patch.footprint),
            area_m2=round(patch.area_m2, 3),
            cells=patch.cells,
            mean_score=round(patch.mean_score, 2),
            reasons=patch.reasons,
            relief_m=round(patch.relief_max_m, 3),
            diameter_m=round(patch.diameter_m, 3),
            pattern=patch.pattern,
            target_gsd_m=patch.target_gsd_m,
            gimbal_pitch_deg=patch.gimbal_pitch_deg,
            hint=patch.hint,
        )
        for patch in found.patches
    ]
    return WeakAreasRead(
        epoch_id=epoch.id,
        asset_id=row.asset_id,
        policy="weak-v1",
        cell_m=plane.cell_m,
        level=plane.level,
        weak_cells=found.weak_cells,
        total_cells=found.total_cells,
        truncated_by_budget=found.truncated_by_budget,
        heuristic=HEURISTIC_LABEL,
        areas=areas,
    )


NO_RESCAN = "no-rescan-yet"
#: Earlier epochs a fill-targets answer compares with, newest first. Each may be a grid
#: of ~17 MB decoded; eight bound one call's memory on a small machine, and a site scanned
#: more often is compared with its eight most recent (and says so, `capped`).
MAX_EARLIER = 8


def _latest_published(db: Session, site_id: uuid.UUID) -> Epoch:
    epoch = db.scalars(
        select(Epoch)
        .where(Epoch.site_id == site_id, Epoch.status == EpochStatus.PUBLISHED.value)
        .order_by(Epoch.observed_at.desc(), Epoch.ordinal.desc())
        .limit(1)
    ).first()
    if epoch is None:
        raise ConflictError(f"site {site_id} has no published epoch", code=NOT_MEASURED)
    return epoch


def earlier_measured(db: Session, epoch: Epoch, limit: int) -> list[tuple[Epoch, EpochMap]]:
    """At most `limit` published epochs of `epoch`'s site observed before it whose current
    revision has a confidence map, newest first, with that map: one query, however many
    unmeasured epochs lie between (`measured_map`'s rule, in SQL)."""
    current = (
        select(Asset.id.label("asset_id"), Asset.epoch_id.label("epoch_id"))
        .join(Epoch, Epoch.id == Asset.epoch_id)
        .where(
            Epoch.site_id == epoch.site_id,
            Asset.valid_to.is_(None),
            Asset.source_kind == AssetSourceKind.OBSERVED.value,
        )
        .distinct(Asset.epoch_id)
        .order_by(Asset.epoch_id, Asset.created_at.desc(), Asset.id.desc())
        .subquery()
    )
    rows = db.execute(
        select(Epoch, EpochMap)
        .join(current, current.c.epoch_id == Epoch.id)
        .join(
            EpochMap,
            (EpochMap.asset_id == current.c.asset_id)
            & (EpochMap.kind == EpochMapKind.CONFIDENCE.value),
        )
        .where(
            Epoch.site_id == epoch.site_id,
            Epoch.status == EpochStatus.PUBLISHED.value,
            Epoch.observed_at < epoch.observed_at,
            Epoch.id != epoch.id,
        )
        .order_by(Epoch.observed_at.desc(), Epoch.ordinal.desc())
        .limit(limit)
    ).all()
    return [(found_epoch, found_map) for found_epoch, found_map in rows]


def fill_targets(
    db: Session, storage: ObjectStorage, site_id: uuid.UUID, epoch_id: uuid.UUID | None
) -> FillTargetsRead:
    """The cells epoch B (`epoch_id`, else the site's latest published) left weak after a
    re-scan (`app/analysis/rescan_gate.py`): 409 `confidence-not-measured` when B has no
    map, `no-rescan-yet` when no earlier epoch of the site was measured."""
    if db.get(Site, site_id) is None:
        raise NotFoundError("site", site_id)
    epoch = (
        epoch_of_site(db, site_id, epoch_id)
        if epoch_id is not None
        else _latest_published(db, site_id)
    )
    row = measured_map(db, epoch)
    found = earlier_measured(db, epoch, MAX_EARLIER + 1)
    earlier_rows = found[:MAX_EARLIER]
    capped = len(found) > MAX_EARLIER
    if not earlier_rows:
        raise ConflictError(
            f"epoch {epoch.id} has no earlier measured epoch on its site: nothing has been "
            "re-scanned yet, so no cell can be still weak after one",
            code=NO_RESCAN,
        )
    grid = load_grid(storage, row)
    earlier = [load_grid(storage, found) for _, found in earlier_rows]
    try:
        result = rescan_gate.still_weak(grid, earlier)
    except cell_grid.CellGridError as error:
        log.warning("confidence: epoch %s cannot be compared: %s", epoch.id, error)
        raise ConflictError(
            f"epoch {epoch.id}'s confidence grids cannot be compared: {error}", code=UNREADABLE
        ) from error
    return FillTargetsRead(
        policy="rescan-gate-v1",
        site_id=site_id,
        epoch_id=epoch.id,
        weak_in=[earlier_rows[index][0].id for index in result.contributed],
        compared_with=[candidate.id for candidate, _ in earlier_rows],
        capped=capped,
        frame="tileset",
        cell_m=math.ldexp(cell_grid.LATTICE_BASE_M, result.level),
        level=result.level,
        i0=result.i0,
        j0=result.j0,
        ni=result.ni,
        nj=result.nj,
        row_bytes=result.row_bytes,
        bit_order="little",
        mask=result.packed(),
        cells=result.cells,
        satisfied_by=result.satisfied_packed(),
        reason=(
            "weak in this epoch, and weak in the newest earlier measured epoch that covers "
            f"them: a re-scan that did not fix them ({rescan_gate.POLICY}, at this epoch's "
            f"own {math.ldexp(cell_grid.LATTICE_BASE_M, result.level):g} m cells)"
        ),
    )
