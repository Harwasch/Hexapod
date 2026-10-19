"""Show once, everywhere over a saved Land: a tapped cell against every other, and the alike
cells as areas for a plan (`app.analysis.similarity`).

Both answers are computed on request from a stored `sentinel-2-signature` raster and never
saved. The standardized vectors of a raster are kept in memory between taps (keyed by the
raster's content hash, like the map tiles' band cache), so a tap after the first is a matrix
product, not a decode.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict

import numpy as np
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis.lonlat_grid import encode, grid_for, inside_mask, reproject_band
from app.analysis.similarity import (
    METHOD,
    NO_SCORE,
    Patch,
    Stack,
    alike_patches,
    build_stack,
    centres_in,
    cosine,
    footprint_mapping,
    interpretation,
    polygon_parts,
    protected_geometry,
    quantize,
    raw_threshold,
    resolution_label,
    snapped_outline,
)
from app.models.land import LandArea, LandBoundaryRevision
from app.models.land_raster import LandRaster
from app.schemas.geojson import MultiPolygon as MultiPolygonModel
from app.schemas.geojson import Polygon as PolygonModel
from app.schemas.land_rasters import RasterMetadata, RasterRequest
from app.schemas.land_similarity import (
    SignatureLeftOut,
    SignatureScene,
    SignatureSeed,
    SignatureSeedWindow,
    SignatureSimilarityRead,
    SignatureZone,
    SignatureZonesRead,
    SignatureZonesRequest,
)
from app.services import land_rasters
from app.services.errors import ConflictError, InvalidInputError, NotFoundError

#: Standardized vectors kept between taps. Oak Bench Ranch's 54,000 cells of 20 features take
#: 4.6 MB and a 512 x 512 twenty-feature raster about 24 MB, so this keeps several lands'; a
#: larger stack is kept alone. Sized with the API's other in-process caches (decoded bands,
#: topo sheets, hydrography) and one cold build's peak to fit the 1 GB API VM.
STACK_CACHE_BYTES = 96 * 1024 * 1024
#: The display grid's cells may be at most this many on a side (the analysis grid's limit).
DISPLAY_MAX_CELLS = 1024

_stacks: OrderedDict[str, Stack] = OrderedDict()
_loading: dict[str, threading.Lock] = {}
_lock = threading.Lock()
#: One cold build at a time, across rasters: a 512 x 512 twenty-feature build peaks near
#: 200 MB on top of its blob and bands, and first taps on two lands at once must not double
#: that on the 1 GB VM. A tap on a built stack never waits for it.
_building = threading.Lock()


def _signature(
    db: Session, workspace_id: uuid.UUID, raster_id: uuid.UUID
) -> tuple[LandRaster, RasterMetadata, RasterRequest]:
    row = land_rasters.scoped(db, workspace_id, raster_id)
    request = RasterRequest.model_validate(row.request)
    metadata = RasterMetadata.model_validate(row.metadata_json)
    if request.dataset != "sentinel-2-signature" or metadata.signature is None:
        raise ConflictError(
            f"Similarity compares seasonal 10 m signatures, and this raster is {request.dataset}. "
            "Prepare 10 m signatures for the land and compare with that raster.",
            code="not_a_signature",
        )
    if not metadata.signature.common_cells:
        raise InvalidInputError(
            "No cell of this raster is clear in every season, so nothing can be compared. "
            "Prepare the signatures again with other dates."
        )
    return row, metadata, request


def _stack(db: Session, row: LandRaster, metadata: RasterMetadata) -> Stack:
    key = row.sha256
    with _lock:
        cached = _stacks.get(key)
        if cached is not None:
            _stacks.move_to_end(key)
            return cached
        loading = _loading.setdefault(key, threading.Lock())
    # Taps arriving together decode once: the first builds, the rest wait and reuse it.
    with loading:
        with _lock:
            cached = _stacks.get(key)
        if cached is not None:
            return cached
        try:
            signature = metadata.signature
            assert signature is not None  # checked by _signature
            with _building:
                data = land_rasters.blob(db, row)
                with MemoryFile(data) as memory, memory.open(driver="GTiff") as dataset:
                    stack = build_stack(
                        dataset.read(),
                        signature.features,
                        dataset.transform,
                        dataset.crs,
                        dataset.nodata,
                    )
            with _lock:
                _stacks[key] = stack
                size = sum(item.nbytes for item in _stacks.values())
                while size > STACK_CACHE_BYTES and len(_stacks) > 1:
                    size -= _stacks.popitem(last=False)[1].nbytes
        finally:
            with _lock:
                _loading.pop(key, None)
    return stack


def _boundary(db: Session, row: LandRaster) -> BaseGeometry:
    revision = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == row.land_id,
            LandBoundaryRevision.revision == row.boundary_revision,
        )
    )
    if revision is None:
        raise NotFoundError("land boundary revision", f"{row.land_id}@{row.boundary_revision}")
    return shape(revision.boundary)


def _stale(db: Session, row: LandRaster) -> bool:
    current = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return current != row.boundary_revision


def _scenes(metadata: RasterMetadata, request: RasterRequest) -> list[SignatureScene]:
    observations = metadata.vegetation.observations if metadata.vegetation else []
    scenes = []
    for window, period in enumerate(request.periods):
        observation = observations[window] if window < len(observations) else None
        scenes.append(
            SignatureScene(
                window=window,
                period=period,
                scene_id=observation.scene_id if observation else None,
                acquired_at=observation.acquired_at if observation else None,
            )
        )
    return scenes


def _attribution(metadata: RasterMetadata) -> str:
    attributions = dict.fromkeys(source.attribution for source in metadata.sources)
    return " · ".join(attributions) or "Contains modified Copernicus Sentinel data"


def _seed(
    stack: Stack,
    boundary: BaseGeometry,
    scenes: list[SignatureScene],
    resolution_m: float,
    longitude: float,
    latitude: float,
) -> tuple[int, SignatureSeed]:
    if not boundary.covers(Point(longitude, latitude)):
        raise InvalidInputError(
            "That point is outside the land's boundary. Choose a place inside it."
        )
    to_metric = Transformer.from_crs("EPSG:4326", stack.crs, always_xy=True)
    to_geographic = Transformer.from_crs(stack.crs, "EPSG:4326", always_xy=True)
    x, y = to_metric.transform(longitude, latitude)
    column_f, row_f = ~stack.transform @ (x, y)
    column, row = math.floor(column_f), math.floor(row_f)
    inside = 0 <= row < stack.shape[0] and 0 <= column < stack.shape[1]
    position = stack.index_of(row, column) if inside else None
    if position is None:
        raise InvalidInputError(
            f"That {resolution_label(resolution_m)} cell has no full signature: at least one "
            "season was cloudy, shadowed, snowy or water there. Choose another place."
        )
    center_x, center_y = stack.transform @ (column + 0.5, row + 0.5)
    cell_longitude, cell_latitude = to_geographic.transform(center_x, center_y)
    values = stack.values(position)
    windows: dict[int, dict[str, float]] = {}
    for feature, value in zip(stack.features, values, strict=True):
        # Reflectance is stored in steps of 1e-4: the vector's float error goes with rounding.
        windows.setdefault(feature.window, {})[feature.key] = round(float(value), 4)
    acquired = {scene.window: scene.acquired_at for scene in scenes}
    return position, SignatureSeed(
        longitude=longitude,
        latitude=latitude,
        cell_longitude=cell_longitude,
        cell_latitude=cell_latitude,
        windows=[
            SignatureSeedWindow(window=window, acquired_at=acquired.get(window), **items)
            for window, items in sorted(windows.items())
        ],
    )


def similar(
    db: Session, workspace_id: uuid.UUID, raster_id: uuid.UUID, longitude: float, latitude: float
) -> SignatureSimilarityRead:
    row, metadata, request = _signature(db, workspace_id, raster_id)
    stack = _stack(db, row, metadata)
    boundary = _boundary(db, row)
    scenes = _scenes(metadata, request)
    position, seed = _seed(stack, boundary, scenes, metadata.resolution_m, longitude, latitude)
    quantized = quantize(cosine(stack, position))
    try:
        spec = grid_for(boundary, metadata.resolution_m, max_cells=DISPLAY_MAX_CELLS)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    display = reproject_band(
        spec, quantized.astype(np.float32), stack.transform, stack.crs, float(NO_SCORE)
    )
    shown = np.isfinite(display) & inside_mask(spec, boundary)
    values = np.where(shown, display, NO_SCORE).astype(np.uint8)
    stale = _stale(db, row)
    warnings = list(metadata.warnings)
    if stale:
        warnings.insert(
            0,
            "The land's boundary changed after these signatures were made; they cover the "
            "earlier boundary. Prepare them again for this one.",
        )
    signature = metadata.signature
    assert signature is not None  # checked by _signature
    return SignatureSimilarityRead(
        raster_id=row.id,
        land_id=row.land_id,
        boundary_revision=row.boundary_revision,
        stale=stale,
        seed=seed,
        grid=encode(spec, values, "u8", scale=1 / 250, nodata=NO_SCORE),
        cell_area_m2=spec.cell_area_m2,
        native_resolution_m=metadata.resolution_m,
        compared_cells=int(stack.cells.size),
        boundary_cells=metadata.boundary_cells,
        scenes=scenes,
        method=METHOD,
        interpretation=interpretation(metadata.resolution_m),
        warnings=warnings,
        attribution=_attribution(metadata),
    )


def _stored(geometry: BaseGeometry) -> PolygonModel | MultiPolygonModel | None:
    parts = polygon_parts(geometry)
    if not parts:
        return None
    outline = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    model = PolygonModel if isinstance(outline, Polygon) else MultiPolygonModel
    try:
        return model.model_validate(footprint_mapping(outline))
    except ValueError:
        return None


def _footprint(geometry: BaseGeometry) -> PolygonModel | MultiPolygonModel | None:
    """A shapely outline as a stored footprint. Rounded to the footprint's precision, a thin
    patch can pinch or two parts can meet edge to edge; then the outline is snapped to that
    precision with a valid result and tried once more. None for what is still not an area."""
    stored = _stored(geometry)
    if stored is None:
        snapped = snapped_outline(geometry)
        stored = _stored(snapped) if snapped is not None else None
    return stored


def _zone(patches: list[Patch]) -> tuple[SignatureZone | None, list[Patch]]:
    """Patches as one area, and those left out of it: a patch no outline can be stored for (a
    sliver narrower than the rounding) is counted, never dropped without a word, and never
    counted in an area it is not drawn in."""
    stored = [(patch, _footprint(patch.footprint)) for patch in patches]
    outlined = [patch for patch, own in stored if own is not None]
    outline: PolygonModel | MultiPolygonModel | None = None
    if len(outlined) == 1:
        outline = next(own for _, own in stored if own is not None)
    elif outlined:
        outline = _footprint(MultiPolygon([patch.footprint for patch in outlined]))
    if outline is None:
        return None, patches
    kept = {id(patch) for patch in outlined}
    cells = sum(patch.cells for patch in outlined)
    weighted = sum(patch.mean_similarity * patch.cells for patch in outlined)
    zone = SignatureZone(
        footprint=outline,
        area_m2=sum(patch.area_m2 for patch in outlined),
        cells=cells,
        mean_similarity=weighted / cells if cells else 0.0,
        patches=len(outlined),
    )
    return zone, [patch for patch in patches if id(patch) not in kept]


def _left_out(patches: list[Patch]) -> SignatureLeftOut:
    return SignatureLeftOut(patches=len(patches), area_m2=sum(patch.area_m2 for patch in patches))


def zones(
    db: Session, workspace_id: uuid.UUID, raster_id: uuid.UUID, body: SignatureZonesRequest
) -> SignatureZonesRead:
    row, metadata, request = _signature(db, workspace_id, raster_id)
    stack = _stack(db, row, metadata)
    boundary = _boundary(db, row)
    scenes = _scenes(metadata, request)
    position, seed = _seed(
        stack, boundary, scenes, metadata.resolution_m, body.longitude, body.latitude
    )
    scores = cosine(stack, position)
    quantized = quantize(scores)
    alike = (quantized != NO_SCORE) & (quantized >= raw_threshold(body.threshold))
    to_metric = Transformer.from_crs("EPSG:4326", stack.crs, always_xy=True)
    to_geographic = Transformer.from_crs(stack.crs, "EPSG:4326", always_xy=True)
    protected = transform_geometry(to_metric.transform, protected_geometry(body.protected))
    protected_cells = centres_in(protected, stack.shape, stack.transform)
    held = int(np.count_nonzero(alike & protected_cells))
    alike &= ~protected_cells
    patches = alike_patches(
        alike,
        scores,
        stack.transform,
        metadata.resolution_m,
        transform_geometry(to_metric.transform, boundary),
        protected,
        to_geographic.transform,
    )
    kept = [patch for patch in patches if patch.area_m2 >= body.min_patch_m2]
    small = [patch for patch in patches if patch.area_m2 < body.min_patch_m2]
    own: list[SignatureZone] = []
    unoutlined: list[Patch] = []
    for patch in kept[: body.max_zones]:
        zone, left = _zone([patch])
        if zone is not None:
            own.append(zone)
        unoutlined.extend(left)
    rest = kept[body.max_zones :]
    remainder, left = _zone(rest) if rest else (None, [])
    unoutlined.extend(left)
    return SignatureZonesRead(
        raster_id=row.id,
        land_id=row.land_id,
        boundary_revision=row.boundary_revision,
        stale=_stale(db, row),
        threshold=body.threshold,
        seed=seed,
        zones=own,
        remainder=remainder,
        dropped=_left_out(small),
        unoutlined=_left_out(unoutlined),
        protected_cells=held,
        matched_cells=int(np.count_nonzero(alike)),
        cell_area_m2=metadata.resolution_m**2,
        native_resolution_m=metadata.resolution_m,
        scenes=scenes,
        method=METHOD,
        interpretation=interpretation(metadata.resolution_m),
        attribution=_attribution(metadata),
    )
