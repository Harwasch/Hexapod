"""Work only the ground it can: an outline on a saved Land less what its machine class cannot work.

`POST /land/{id}/workable-ground` answers the dress rehearsal's raised hands with ground a
plan can be redrafted on (`app.analysis.workable`). It reads the same things the rehearsal
reads: the Land's freshest elevation raster (USGS 3DEP bare earth; Copernicus GLO-30 when
that is all there is, and then says so), the USGS NHD flowlines around the Land
(`land_hydrography`, the rehearsal's own source, with its ephemeral rule), and the class's
assumed limits from machine_classes.json. Computed on request, never saved.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely import make_valid
from shapely.geometry import MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform as transform_geometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis.similarity import footprint_mapping, polygon_parts, snapped_outline
from app.analysis.workable import ACRE_M2, MIN_PART_M2, counts_as_creek, slope_percent, split_ground
from app.models.land_raster import LandRaster
from app.schemas.geojson import MultiPolygon as MultiPolygonModel
from app.schemas.land import LandRead
from app.schemas.land_hydrography import LandHydrography
from app.schemas.land_rasters import RasterMetadata, RasterRequest
from app.schemas.land_workable import (
    WorkableCreeks,
    WorkableElevation,
    WorkableGroundRead,
    WorkableGroundRequest,
    WorkableLeftOut,
    WorkableLimits,
    WorkableRemoved,
)
from app.services import land_hydrography, land_rasters
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.land import get_land
from app.services.machine_classes import MACHINE_CLASSES

#: Elevation datasets, best first for machine slope: bare earth, then a surface model.
DEM_DATASETS = ("usgs-3dep-13", "cop-dem-glo-30")
#: Cells read around the outline, so its edge cells have both neighbours.
MARGIN_CELLS = 2
LABELS = {
    "steep": "Left out: too steep",
    "steep-edge": "Left out: steep edge",
    "creek-buffer": "Left out: creek buffer",
    "sliver": "Left out: too small to work",
    "no-data": "Left out: no slope data",
}


def _dataset(row: LandRaster) -> str:
    return RasterRequest.model_validate(row.request).dataset


def _elevation_raster(
    db: Session, workspace_id: uuid.UUID, land: LandRead, raster_id: uuid.UUID | None
) -> LandRaster:
    """The raster asked for, or the Land's newest elevation raster for its current outline:
    3DEP before GLO-30."""
    if raster_id is not None:
        row = land_rasters.scoped(db, workspace_id, raster_id)
        if row.land_id != land.id:
            raise NotFoundError("land raster", raster_id)
        if _dataset(row) not in DEM_DATASETS:
            raise ConflictError(
                f"Workable ground reads slope from an elevation raster, and this one is "
                f"{_dataset(row)}. Choose a USGS 3DEP or Copernicus GLO-30 raster.",
                code="not_elevation",
            )
        return row
    rows = db.scalars(
        select(LandRaster)
        .where(LandRaster.land_id == land.id, LandRaster.boundary_revision == land.revision)
        .order_by(LandRaster.created_at.desc(), LandRaster.id)
    ).all()
    for dataset in DEM_DATASETS:
        found = next((row for row in rows if _dataset(row) == dataset), None)
        if found is not None:
            return found
    raise ConflictError(
        f"{land.name} has no elevation for its current outline, so its slope cannot be read. "
        "Analyze bare-earth terrain (USGS 3DEP) for it in Land, then try again.",
        code="no_elevation",
    )


def _published(metadata: RasterMetadata) -> str | None:
    """The newest source file date (YYYY-MM-DD): a 3DEP tile's publication date."""
    dates = []
    for source in metadata.sources:
        try:
            if source.last_modified:
                dates.append(parsedate_to_datetime(source.last_modified).date().isoformat())
            elif source.catalog_datetime:
                dates.append(source.catalog_datetime[:10])
        except (TypeError, ValueError):
            continue
    return max(dates) if dates else None


def _elevation_label(dataset: str, resolution_m: float) -> str:
    if dataset == "usgs-3dep-13":
        return f"USGS 3DEP 1/3″ bare-earth DEM, {resolution_m:.0f} m"
    return f"Copernicus GLO-30, a {resolution_m:.0f} m surface model incl. canopy"


def _window(
    outline: BaseGeometry, band: land_rasters.DecodedBand
) -> tuple[NDArray[np.float64], Affine] | None:
    """The elevation around the outline (a margin of cells), NaN for no data, and its
    transform; None when the outline misses the raster."""
    rows, columns = band.data.shape
    left, bottom, right, top = outline.bounds
    window = from_bounds(left, bottom, right, top, band.transform)
    row0 = max(0, int(np.floor(window.row_off)) - MARGIN_CELLS)
    col0 = max(0, int(np.floor(window.col_off)) - MARGIN_CELLS)
    row1 = min(rows, int(np.ceil(window.row_off + window.height)) + MARGIN_CELLS)
    col1 = min(columns, int(np.ceil(window.col_off + window.width)) + MARGIN_CELLS)
    if row1 <= row0 or col1 <= col0:
        return None
    values = band.data[row0:row1, col0:col1].astype(np.float64)
    if band.nodata is not None:
        values[values == band.nodata] = np.nan
    values[~np.isfinite(values)] = np.nan
    crop = Window(col0, row0, col1 - col0, row1 - row0)
    return values, window_transform(crop, band.transform)


def _multipolygon(
    geometry: BaseGeometry, to_geographic: Callable[..., Any]
) -> MultiPolygonModel | None:
    """Metric ground as a stored lon/lat MultiPolygon; None for no area (or one too thin to
    keep at the footprint's precision)."""
    if geometry.is_empty:
        return None
    lonlat = make_valid(transform_geometry(to_geographic, geometry))
    # Largest first: whatever reads a footprint's first polygon reads its main piece.
    parts = sorted(
        (orient(part) for part in polygon_parts(lonlat) if part.area > 0),
        key=lambda part: part.area,
        reverse=True,
    )
    if not parts:
        return None
    try:
        return MultiPolygonModel.model_validate(footprint_mapping(MultiPolygon(parts)))
    except ValueError:
        snapped = snapped_outline(MultiPolygon(parts))
    if snapped is None:
        return None
    pieces = polygon_parts(snapped)
    try:
        return MultiPolygonModel.model_validate(footprint_mapping(MultiPolygon(pieces)))
    except ValueError:
        return None


def _creek_lines(
    hydrography: LandHydrography, include_ephemeral: bool, to_metric: Transformer
) -> list[BaseGeometry]:
    return [
        transform_geometry(to_metric.transform, shape(feature.geometry.model_dump()))
        for feature in hydrography.features
        if counts_as_creek(feature.kind, include_ephemeral)
    ]


def _creek_label(hydrography: LandHydrography) -> str:
    # "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000"
    scale = hydrography.source.resolution.split(";")[0].split("(")[0].strip()
    return f"{hydrography.source.name}, {scale}" if scale else hydrography.source.name


def _acres(area_m2: float) -> str:
    acres = area_m2 / ACRE_M2
    return f"{acres:.1f}" if acres < 10 else f"{acres:,.0f}"


def workable_ground(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, body: WorkableGroundRequest
) -> WorkableGroundRead:
    land = get_land(db, workspace_id, land_id)
    if body.machine_class == "drone":
        raise InvalidInputError(
            "A drone flies over slope and creeks: workable ground is for ground machines."
        )
    machine = MACHINE_CLASSES.ground(body.machine_class)
    footprint = make_valid(shape(body.footprint.model_dump()))
    if not shape(land.boundary.model_dump()).intersects(footprint):
        raise InvalidInputError(
            f"This ground is outside {land.name}. Choose ground on the Land to read its terrain."
        )
    row = _elevation_raster(db, workspace_id, land, body.dem_raster_id)
    dataset = _dataset(row)
    metadata = RasterMetadata.model_validate(row.metadata_json)
    # Creeks first: the lookup may wait on NHD and releases the session's transaction.
    hydrography = land_hydrography.hydrography(db, workspace_id, land.id)
    band = land_rasters.band_array(db, row, 1)
    to_metric = Transformer.from_crs("EPSG:4326", band.crs, always_xy=True)
    to_geographic = Transformer.from_crs(band.crs, "EPSG:4326", always_xy=True)
    outline = make_valid(transform_geometry(to_metric.transform, footprint))
    read = _window(outline, band)
    if read is None:
        raise InvalidInputError(
            f"This ground is outside {land.name}'s elevation raster. Choose ground on the Land."
        )
    elevation, transform = read
    resolution = float(metadata.resolution_m)
    slope = slope_percent(elevation, resolution)
    creeks_applied = hydrography.status in ("available", "empty")
    lines = (
        _creek_lines(hydrography, body.include_ephemeral_creeks, to_metric)
        if creeks_applied
        else []
    )
    split = split_ground(
        outline,
        slope,
        transform,
        max_slope_pct=machine.max_slope_pct,
        creeks=lines,
        buffer_m=machine.creek_buffer_m,
        edge_m=resolution / 2,
    )
    if split.cells_read == 0:
        raise InvalidInputError(
            f"{land.name}'s elevation has no heights under this ground, so its slope cannot "
            "be read. Choose ground inside the Land's saved outline."
        )
    removed = WorkableRemoved(
        steep_m2=split.steep.area,
        steep_edge_m2=split.steep_edge.area,
        creek_buffer_m2=split.creek_buffer.area,
        slivers_m2=split.slivers.area,
        slivers=split.sliver_count,
        no_data_m2=split.no_data.area,
    )
    left_out = [
        WorkableLeftOut(
            reason=reason,
            label=LABELS[reason],
            area_m2=geometry.area,
            geometry=_multipolygon(geometry, to_geographic.transform),
        )
        for reason, geometry in (
            ("steep", split.steep),
            ("steep-edge", split.steep_edge),
            ("creek-buffer", split.creek_buffer),
            ("sliver", split.slivers),
            ("no-data", split.no_data),
        )
        if geometry.area > 0
    ]
    elevation_label = _elevation_label(dataset, resolution)
    published = _published(metadata)
    stale = row.boundary_revision != land.revision
    fallback = dataset == "cop-dem-glo-30"
    creek_label = _creek_label(hydrography)
    plain = machine.label.removesuffix(" (assumed)")
    assumptions = [
        f"{plain} (assumed): works slopes up to {machine.max_slope_pct:g} % and keeps "
        f"{machine.creek_buffer_m:g} m from creeks. Both limits are assumed per machine class "
        "(machine_classes.json), not measured on a machine.",
        f"Slope is each {resolution:g} m cell's steepest gradient on the {elevation_label}, so "
        "a cell within the limit can be driven across or up and down; a cell over it is left "
        "out whole.",
        f"Ground within {resolution / 2:g} m (half a cell) of a too-steep cell is left out too: "
        "slope is read from heights a cell either side, so beside a steep cell it takes in part "
        "of that cell's slope, and a lane run up to it would still go past the limit.",
    ]
    if creeks_applied:
        kinds = "perennial, intermittent and unclassified channels and artificial paths"
        ephemeral = (
            "and ephemeral drainages"
            if body.include_ephemeral_creeks
            else "not ephemeral drainages, which run only after rain"
        )
        assumptions.append(
            f"The {machine.creek_buffer_m:g} m buffer is drawn around {len(lines)} mapped "
            f"flowline{'s' if len(lines) != 1 else ''} ({creek_label}): {kinds}, {ephemeral}. "
            "Mapped channels, not surveyed banks."
        )
    assumptions.append("Ground both too steep and inside a creek buffer counts once, as too steep.")
    assumptions.append(
        f"Pieces of workable ground under {MIN_PART_M2 / ACRE_M2:.1f} ac are too small to work "
        f"as a pass and are left out ({split.sliver_count} "
        f"piece{'s' if split.sliver_count != 1 else ''}, {_acres(removed.slivers_m2)} ac)."
    )
    if removed.no_data_m2 > 0:
        assumptions.append(
            f"{_acres(removed.no_data_m2)} ac has no height in the raster (it is outside the "
            "Land's saved outline), so its slope is not known and it is left out."
        )
    warnings: list[str] = []
    if fallback:
        warnings.append(
            "Slope is from Copernicus GLO-30, a 30 m surface model that includes tree canopy "
            "and buildings, so canopy edges read as slope. Analyze bare-earth terrain (USGS "
            "3DEP) for this land for the ground's own slope."
        )
    if stale:
        warnings.append(
            f"This elevation was made for an earlier outline of {land.name} (revision "
            f"{row.boundary_revision}); analyze it again for the current one."
        )
    if not creeks_applied:
        warnings.append(f"Creek buffer not applied: {hydrography.message}")
    elif hydrography.truncated:
        warnings.append(
            f"{creek_label} stopped at its record limit here: more flowlines may exist than the "
            f"{len(hydrography.features)} read."
        )
    sources = [
        f"Slope: {elevation_label}, raster {str(row.id)[:8]}"
        + (f", source published {published}" if published else "")
    ]
    if creeks_applied:
        sources.append(f"Creeks: {creek_label}, read {hydrography.retrieved_at.date().isoformat()}")
    sources.append("Limits: assumed per machine class (machine_classes.json)")
    workable = _multipolygon(split.workable, to_geographic.transform)
    return WorkableGroundRead(
        land_id=land.id,
        boundary_revision=land.revision,
        workable=workable,
        footprint_m2=outline.area,
        workable_m2=split.workable.area if workable is not None else 0.0,
        removed=removed,
        left_out=left_out,
        limits=WorkableLimits(
            machine_class=body.machine_class,
            label=machine.label,
            max_slope_pct=machine.max_slope_pct,
            creek_buffer_m=machine.creek_buffer_m,
        ),
        elevation=WorkableElevation(
            raster_id=row.id,
            dataset=dataset,
            label=elevation_label,
            published=published,
            resolution_m=resolution,
            edge_m=resolution / 2,
            stale=stale,
            fallback=fallback,
        ),
        creeks=WorkableCreeks(
            status=hydrography.status,
            label=creek_label,
            counted=len(lines),
            ephemeral_included=body.include_ephemeral_creeks,
            message=hydrography.message,
        ),
        sources=sources,
        assumptions=assumptions,
        warnings=warnings,
    )
