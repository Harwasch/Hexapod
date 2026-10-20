"""The Fog of Knowing over a saved Land: the stored records behind it, read in a fixed number of
queries, and the fog, a cell's ledger and the foggiest patches computed from them.

What counts (policy fog-v1, `app.analysis.knowledge`), and only as stored and cited here:

- captures: a public scan site's assets whose footprint (or the site's outline) meets the Land,
  dated by the asset, then its capture, classed by its recorded ground resolution;
- field inspections of the Land's inventory features, on the geometry as inspected;
- the Land's raster analyses, per dated observation, on the cells the source actually saw
  (a cloud-masked Sentinel-2 cell that came back empty earns nothing);
- archive images aligned to the ground, on their fitted footprint;
- research evidence and field surveys, which carry no footprint: land-wide, capped at the
  regional class (GBIF withholds its point coordinates, and they are never drawn here).

Plans, the simulated fleet, Living Survey motion, inferred fill and the photorealistic basemap
never count; `NOT_COUNTED` says why, and the legend prints it. The fog therefore lifts only
when new evidence is stored, never by computing again.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import ValidationError
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.analysis import knowledge
from app.analysis.knowledge import (
    CLASSES,
    FIELD_REACH_M,
    FOGGY,
    FORMULA,
    NO_RECORD,
    OUTSIDE,
    POLICY_VERSION,
    UNDATED_FACTOR,
    ClassId,
    Composed,
    FogGrid,
    KnowledgeSource,
)
from app.analysis.lonlat_grid import encode, reproject_band
from app.models.asset import Asset
from app.models.capture import Capture
from app.models.land import LandArea
from app.models.land_archive_image import LandArchiveImage
from app.models.land_feature import FeatureInspection, LandFeature, LandFeatureRevision
from app.models.land_image_registration import LandImageRegistration
from app.models.land_raster import LandRaster, LandRasterBlob
from app.models.land_survey import LandSurvey
from app.models.research import Evidence, Investigation, ResearchRun
from app.models.site import Site
from app.schemas.geojson import Polygon
from app.schemas.land import LandRead
from app.schemas.land_knowledge import (
    KnowledgeCellRead,
    KnowledgeCellRecord,
    KnowledgeClassRead,
    KnowledgeGridRead,
    KnowledgeLink,
    KnowledgePolicyRead,
    KnowledgeProvenance,
    KnowledgeRecordRead,
    KnowledgeScoutArea,
    KnowledgeScoutRead,
    KnowledgeScoutRequest,
    KnowledgeSummary,
    NotCountedRead,
)
from app.services.errors import InvalidInputError
from app.services.land import get_land

#: The newest rasters weighed per request: each is a stored GeoTIFF warped onto the fog grid.
RASTER_LIMIT = 20
#: Most raster blob bytes read in one query on a cold mask cache (`_rasters`).
BLOB_BATCH_BYTES = 32 * 1024 * 1024
#: Newest rows read per kind; a Land with more keeps its newest evidence.
ROW_LIMIT = 2000
#: How the raster datasets count: class, and the native resolution of what the source saw.
RASTER_DATASETS: dict[str, tuple[ClassId, float]] = {
    "sentinel-2-ndvi": ("satellite", 10.0),
    "sentinel-2-signature": ("satellite", 10.0),
    "esa-worldcover-2021": ("satellite", 10.0),
    "cop-dem-glo-30": ("regional", 30.0),
    "usgs-3dep-13": ("regional", 10.0),
}
#: Research providers whose records describe the land at large: regional, land-wide.
REGIONAL_PROVIDERS = frozenset(
    {
        "usgs-elevation",
        "nasa-power",
        "usda-soils",
        "usda-ecological-sites",
        "epa-ecoregions",
        "fema-flood-zones",
    }
)
#: Counted directly (inspections, surveys) or as masks (rasters): their evidence rows are not
#: counted again land-wide.
COUNTED_ELSEWHERE = frozenset({"private-inventory", "field-survey", *RASTER_DATASETS})
#: Research outputs that are reasoning about the land, not a look at it.
DERIVED_PROVIDERS = {
    "saved-solar-assessment": "A modelled solar assessment, not an observation of the ground.",
    "saved-scenario": "A planning scenario, not an observation of the ground.",
    "col-taxonomy": "A name lookup in a species checklist, not an observation of this land.",
}
SCAN_REPRESENTATIONS = frozenset({"gaussian-splat", "mesh", "point-cloud"})
YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")

NOT_COUNTED: tuple[NotCountedRead, ...] = (
    NotCountedRead(
        title="Google Photorealistic 3D Tiles (the basemap)",
        reason="The provider publishes no capture date for these tiles, so they say nothing "
        "about how fresh the picture is.",
    ),
    NotCountedRead(
        title="Public aerial photos (USDA NAIP, about 0.6 m)",
        reason="Flown over each US state every two to three years, and viewable for most US "
        "land, but not stored here as evidence, so they do not lift the fog.",
    ),
    NotCountedRead(
        title="Sentinel-2 satellite scenes not yet fetched (10 m, about every five days)",
        reason="Only scenes stored in one of this land's analyses count. Fetch the latest "
        "satellite look to store one.",
    ),
    NotCountedRead(
        title="Mission plans, the simulated fleet and its tracks, Living Survey motion and "
        "inferred fill",
        reason="Simulated or inferred, never evidence.",
    ),
)
FORECAST = (
    "Forecast, not evidence: flown at 5 cm per pixel or finer and processed into a capture, "
    "these cells would read as centimetre evidence."
)

_MASK_CACHE: OrderedDict[tuple[Any, ...], NDArray[np.bool_]] = OrderedDict()
_MASK_LOCK = threading.Lock()
_MASK_CACHE_SIZE = 96


def policy() -> KnowledgePolicyRead:
    return KnowledgePolicyRead(
        version=POLICY_VERSION,
        classes=[
            KnowledgeClassRead(
                id=item.id,
                label=item.label,
                quality=item.quality,
                half_life_days=item.half_life_days,
                max_resolution_m=item.max_resolution_m,
                example=item.example,
            )
            for item in CLASSES
        ],
        undated_factor=UNDATED_FACTOR,
        field_reach_m=dict(FIELD_REACH_M),
        foggy_threshold=FOGGY,
        formula=FORMULA,
    )


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time(12), UTC)
    if isinstance(value, str) and value:
        try:
            return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _day(value: datetime) -> str:
    return value.astimezone(UTC).date().isoformat()


def _text(value: Any, fallback: str, limit: int = 500) -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return (text or fallback)[:limit]


def _url(value: Any) -> str | None:
    return value if isinstance(value, str) and value.startswith(("https://", "http://")) else None


def _field(mapping: Any, name: str) -> Any:
    """A stored JSON value by its snake_case name, or the camelCase one: rows written with
    `model_dump(mode="json")` and with `by_alias=True` both exist."""
    if not isinstance(mapping, dict):
        return None
    if name in mapping:
        return mapping[name]
    head, *rest = name.split("_")
    return mapping.get(head + "".join(part.title() for part in rest))


def _provenance(provider: str, license: Any, attribution: Any, url: Any = None) -> dict[str, Any]:
    return {
        "provider": _text(provider, "unknown"),
        "license": _text(license, "Not stated"),
        "attribution": _text(attribution, "Not stated", 2000),
        "url": _url(url),
    }


# ── Captures: public scan sites over the Land ────────────────────────────────────────────────


def _captures(
    db: Session, land_id: uuid.UUID, not_counted: list[NotCountedRead]
) -> list[KnowledgeSource]:
    land = select(LandArea.boundary).where(LandArea.id == land_id).scalar_subquery()
    stats = (
        select(
            Capture.site_id.label("site_id"),
            func.max(Capture.captured_at).label("captured_at"),
            func.min(Capture.uncertainty_m).label("uncertainty_m"),
        )
        .where(Capture.site_id.is_not(None))
        .group_by(Capture.site_id)
        .subquery()
    )
    footprint = func.coalesce(Asset.footprint, Site.boundary)
    rows = db.execute(
        select(
            Asset,
            Site.id,
            Site.name,
            Site.slug,
            func.ST_AsGeoJSON(footprint),
            stats.c.captured_at,
            stats.c.uncertainty_m,
        )
        .join(Site, Site.id == Asset.site_id)
        .outerjoin(stats, stats.c.site_id == Site.id)
        .where(func.ST_Intersects(footprint, land))
        # A superseded revision of an epoch is history, not knowledge of the land.
        .where(or_(Asset.epoch_id.is_(None), Asset.valid_to.is_(None)))
        .order_by(Asset.created_at.desc())
        .limit(ROW_LIMIT)
    ).all()
    sources: list[KnowledgeSource] = []
    for asset, site_id, site_name, slug, geojson, captured_at, uncertainty in rows:
        representation = str(getattr(asset.representation, "value", asset.representation))
        resolution = asset.resolution or {}
        metres = _field(resolution, "ground_sample_distance_m") or _field(
            resolution, "point_spacing_m"
        )
        note = None
        if isinstance(metres, int | float) and metres > 0:
            class_id = knowledge.class_for_resolution(float(metres))
            resolution_m: float | None = float(metres)
        elif representation in SCAN_REPRESENTATIONS:
            class_id, resolution_m = "centimetre", None
            note = "Class assumed from the representation (a scan); no resolution is recorded."
        else:
            not_counted.append(
                NotCountedRead(
                    title=f"{site_name}: {asset.name}",
                    reason="No ground resolution is recorded for this asset, so its detail "
                    "cannot be weighed.",
                )
            )
            continue
        observed: datetime | None
        if asset.observed_at is not None:
            observed, label = _utc(asset.observed_at), "observed"
        elif asset.valid_from is not None:
            observed, label = _utc(asset.valid_from), "valid from"
        elif captured_at is not None:
            observed, label = _utc(captured_at), "captured"
        else:
            observed, label = None, ""
        attribution = "; ".join(
            str(item.get("text")) for item in (asset.attribution or []) if item.get("text")
        )
        license = (asset.license or {}).get("name")
        sources.append(
            KnowledgeSource(
                key=f"asset:{asset.id}",
                kind="capture",
                title=f"{site_name}: {asset.name} ({representation})",
                class_id=class_id,
                resolution_m=resolution_m,
                observed_at=observed,
                observed_label=f"{label} {_day(observed)}"
                if observed
                else "Undated: no observation or capture date is recorded",
                recorded_at=_utc(asset.created_at) or datetime.now(UTC),
                located=True,
                positional_uncertainty_m=float(uncertainty) if uncertainty is not None else None,
                provenance=_provenance(asset.provider, license, attribution or site_name),
                link={"kind": "site", "id": str(site_id), "detail": {"slug": slug}},
                geometry=shape(json.loads(geojson)),
                class_note=note,
            )
        )
    return sources


# ── Field inspections of inventory features ──────────────────────────────────────────────────


def _inspections(db: Session, land_id: uuid.UUID) -> list[KnowledgeSource]:
    rows = db.execute(
        select(FeatureInspection, LandFeatureRevision.content)
        .join(
            LandFeatureRevision,
            and_(
                LandFeatureRevision.feature_id == FeatureInspection.feature_id,
                LandFeatureRevision.revision == FeatureInspection.feature_revision,
            ),
        )
        .join(LandFeature, LandFeature.id == FeatureInspection.feature_id)
        .where(LandFeature.land_id == land_id)
        .order_by(FeatureInspection.observed_at.desc())
        .limit(ROW_LIMIT)
    ).all()
    sources: list[KnowledgeSource] = []
    for inspection, content in rows:
        geometry = content.get("geometry")
        if not isinstance(geometry, dict):
            continue
        observed = _utc(inspection.observed_at)
        condition = str(inspection.content.get("condition", "unknown"))
        sources.append(
            KnowledgeSource(
                key=f"inspection:{inspection.id}",
                kind="inspection",
                title=f"Inspection: {content.get('name', 'feature')}, {condition}",
                class_id="field",
                resolution_m=None,
                observed_at=observed,
                observed_label=f"inspected {_day(observed)}" if observed else "Undated",
                recorded_at=_utc(inspection.created_at) or datetime.now(UTC),
                located=True,
                positional_uncertainty_m=None,
                provenance=_provenance(
                    "field inspection",
                    "Workspace record",
                    "Recorded in this workspace's inventory",
                ),
                link={
                    "kind": "inventory",
                    "id": str(inspection.feature_id),
                    "detail": {
                        "inspectionId": str(inspection.id),
                        "featureRevision": inspection.feature_revision,
                    },
                },
                geometry=shape(geometry),
            )
        )
    return sources


# ── Raster analyses: what each dated observation saw ────────────────────────────────────────


@dataclass(frozen=True)
class _Observation:
    band: int | None
    acquired_at: datetime | None
    scene_id: str | None


def _observations(metadata: dict[str, Any]) -> list[_Observation]:
    """Dated scenes a raster's metadata lists: the vegetation timeline's, or the same shape
    under `signature` or at the top level (sentinel-2-signature); empty when none."""
    found: list[Any] = []
    for holder in (metadata.get("vegetation"), metadata.get("signature"), metadata):
        if isinstance(holder, dict) and isinstance(holder.get("observations"), list):
            found = holder["observations"]
            break
    result: list[_Observation] = []
    for item in found:
        if not isinstance(item, dict):
            continue
        cells = _field(item, "valid_cells")
        if isinstance(cells, int) and cells <= 0:
            continue
        band = item.get("band")
        result.append(
            _Observation(
                band=band if isinstance(band, int) else None,
                acquired_at=_parse_datetime(_field(item, "acquired_at")),
                scene_id=_field(item, "scene_id"),
            )
        )
    return result


def _cache_get(key: tuple[Any, ...]) -> NDArray[np.bool_] | None:
    with _MASK_LOCK:
        found = _MASK_CACHE.get(key)
        if found is not None:
            _MASK_CACHE.move_to_end(key)
        return found


def _cache_put(key: tuple[Any, ...], value: NDArray[np.bool_]) -> None:
    with _MASK_LOCK:
        _MASK_CACHE[key] = value
        _MASK_CACHE.move_to_end(key)
        while len(_MASK_CACHE) > _MASK_CACHE_SIZE:
            _MASK_CACHE.popitem(last=False)


def _mask_key(grid: FogGrid, row: LandRaster, band: int | None) -> tuple[Any, ...]:
    spec = grid.spec
    return (row.id, row.sha256, spec.west, spec.north, spec.columns, spec.rows, spec.dlon, band)


def _bands_for(dataset: str, observations: list[_Observation]) -> list[int | None]:
    """The bands whose valid cells a raster counts on: each dated scene's, else band 1 of a
    known dataset (None, any band, for an unknown one)."""
    if observations:
        return sorted({item.band for item in observations}, key=lambda band: band or 0)
    return [1 if dataset in RASTER_DATASETS else None]


def _raster_masks(
    grid: FogGrid, row: LandRaster, bands: list[int | None], data: bytes | None
) -> dict[int | None, NDArray[np.bool_]]:
    """Valid cells per band on the fog grid (None: any band), each band warped once, then
    cached by raster and grid."""
    masks = {band: _cache_get(_mask_key(grid, row, band)) for band in bands}
    found = {band: mask for band, mask in masks.items() if mask is not None}
    missing = [band for band, mask in masks.items() if mask is None]
    if not missing or data is None:
        return found
    spec = grid.spec
    with (
        rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024),
        MemoryFile(data) as memory,
        memory.open(driver="GTiff") as dataset,
    ):
        warped: dict[int, NDArray[np.bool_]] = {}

        def valid(index: int) -> NDArray[np.bool_]:
            if index not in warped:
                values = reproject_band(
                    spec, dataset.read(index), dataset.transform, dataset.crs, dataset.nodata
                )
                warped[index] = np.isfinite(values)
            return warped[index]

        for band in missing:
            if band is None:
                mask = np.logical_or.reduce([valid(i) for i in range(1, dataset.count + 1)])
            elif 1 <= band <= dataset.count:
                mask = valid(band)
            else:
                mask = np.zeros(spec.shape, dtype=bool)
            found[band] = mask
            _cache_put(_mask_key(grid, row, band), mask)
    return found


def _raster_provenance(
    dataset: str, sources: list[Any], scene: str | None = None
) -> dict[str, Any]:
    """The cited source for a raster record: the scene's own file when named, else the first."""
    listed = [item for item in sources if isinstance(item, dict)] or [{}]
    chosen = next(
        (item for item in listed if scene and scene in str(item.get("id", ""))), listed[0]
    )
    return _provenance(
        dataset or "land raster",
        chosen.get("license"),
        chosen.get("attribution"),
        _field(chosen, "catalog_url") or chosen.get("url"),
    )


RASTER_TITLES = {
    "esa-worldcover-2021": "ESA WorldCover 2021 land cover",
    "cop-dem-glo-30": "Copernicus 30 m elevation model",
    "usgs-3dep-13": "USGS 3DEP 1/3 arc-second bare-earth elevation (aerial lidar)",
}


def _raster_sources(
    grid: FogGrid,
    row: LandRaster,
    observations: list[_Observation],
    masks: dict[int | None, NDArray[np.bool_]],
) -> list[KnowledgeSource]:
    dataset = str(row.request.get("dataset", ""))
    metadata = row.metadata_json
    listed = metadata.get("sources") or []
    class_id, native = RASTER_DATASETS.get(dataset, ("regional", 0.0))
    analysis_m = _float(_field(metadata, "resolution_m"))
    resolution_m = native or analysis_m
    grid_note = (
        f" sampled on a {analysis_m:.0f} m analysis grid"
        if analysis_m and native and analysis_m > native * 1.01
        else ""
    )

    def source(
        key: str,
        title: str,
        observed: datetime | None,
        label: str,
        band: int | None,
        scene: str | None = None,
    ) -> KnowledgeSource:
        return KnowledgeSource(
            key=key,
            kind="raster-observation",
            title=title,
            class_id=class_id,
            resolution_m=resolution_m,
            observed_at=observed,
            observed_label=label,
            recorded_at=_utc(row.created_at) or datetime.now(UTC),
            located=True,
            positional_uncertainty_m=None,
            provenance=_raster_provenance(dataset, listed, scene),
            link={
                "kind": "raster",
                "id": str(row.id),
                "detail": {"band": band, "dataset": dataset},
            },
            mask=masks.get(band, np.zeros(grid.shape, dtype=bool)),
        )

    if observations:
        suffix = "" if dataset == "sentinel-2-ndvi" else f" ({dataset})"
        return [
            source(
                f"raster:{row.id}:{item.band or 0}:{item.scene_id or ''}",
                f"Sentinel-2 scene {_day(item.acquired_at)}{suffix}"
                if item.acquired_at
                else f"Sentinel-2 scene{suffix}",
                item.acquired_at,
                f"acquired {_day(item.acquired_at)}, 10 m bands{grid_note}"
                if item.acquired_at
                else "Undated: the analysis did not record the scene date",
                item.band,
                item.scene_id,
            )
            for item in observations
        ]
    first = next((item for item in listed if isinstance(item, dict)), {})
    if dataset == "esa-worldcover-2021":
        observed: datetime | None = datetime(2021, 7, 1, tzinfo=UTC)
        label = "Reference year 2021 (dated mid-year)"
    elif dataset.startswith("sentinel-2-"):
        observed, label = None, "Undated: the analysis did not record its scene dates"
    else:
        period = _field(first, "observation_period")
        observed, label = None, f"Undated: {period}" if period else "Undated"
    title = RASTER_TITLES.get(dataset, f"Land raster ({dataset or 'unknown dataset'})")
    return [source(f"raster:{row.id}", title, observed, label, _bands_for(dataset, [])[0])]


def _rasters(db: Session, land_id: uuid.UUID, grid: FogGrid) -> list[KnowledgeSource]:
    found = db.execute(
        select(LandRaster, ResearchRun.investigation_id)
        .join(ResearchRun, ResearchRun.id == LandRaster.run_id)
        .where(LandRaster.land_id == land_id)
        .order_by(LandRaster.created_at.desc(), LandRaster.id)
        .limit(RASTER_LIMIT)
    ).all()
    investigations = {row.id: investigation for row, investigation in found}
    rows = [row for row, _ in found]
    planned: list[tuple[LandRaster, list[_Observation], list[int | None]]] = []
    for row in rows:
        dataset = str(row.request.get("dataset", ""))
        observations = _observations(row.metadata_json) if dataset.startswith("sentinel-2-") else []
        planned.append((row, observations, _bands_for(dataset, observations)))
    # The blobs a cached mask does not already answer for, a few at a time: one query per
    # batch of at most BLOB_BATCH_BYTES (one blob when it is bigger), each batch warped and
    # let go before the next is read. A cold cache over twenty 16 MiB rasters would otherwise
    # hold all of them at once; the usual land's few small rasters are still one query.
    bands_by_row = {row.id: bands for row, _, bands in planned}
    masks: dict[uuid.UUID, dict[int | None, NDArray[np.bool_]]] = {}
    batch: list[LandRaster] = []
    batch_bytes = 0

    def warp(rows: list[LandRaster]) -> None:
        by_id = {row.id: row for row in rows}
        query = select(LandRasterBlob).where(LandRasterBlob.raster_id.in_(list(by_id)))
        for blob in db.scalars(query):
            row = by_id[blob.raster_id]
            masks[row.id] = _raster_masks(grid, row, bands_by_row[row.id], blob.data)

    for row, _, bands in planned:
        cached = {band: _cache_get(_mask_key(grid, row, band)) for band in bands}
        hits = {band: mask for band, mask in cached.items() if mask is not None}
        if len(hits) == len(bands):
            masks[row.id] = hits
            continue
        if batch and batch_bytes + row.byte_size > BLOB_BATCH_BYTES:
            warp(batch)
            batch, batch_bytes = [], 0
        batch.append(row)
        batch_bytes += row.byte_size
    if batch:
        warp(batch)
    sources: list[KnowledgeSource] = []
    for row, observations, _ in planned:
        for source in _raster_sources(grid, row, observations, masks.get(row.id, {})):
            source.link["detail"]["investigationId"] = str(investigations[row.id])
            sources.append(source)
    return sources


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


# ── Archive images aligned to the ground ─────────────────────────────────────────────────────


def _evidence_date(content: dict[str, Any]) -> tuple[datetime | None, str]:
    observed = _parse_datetime(_field(content, "observed_at"))
    if observed:
        return observed, f"observed {_day(observed)}"
    published = _parse_datetime(_field(content, "published_at"))
    if published:
        return published, f"published {_day(published)}"
    media = content.get("media") or {}
    source_date = _field(media, "source_date")
    if isinstance(source_date, str):
        match = YEAR.search(source_date)
        if match:
            return (
                datetime(int(match.group(1)), 7, 1, tzinfo=UTC),
                f'catalog says "{source_date[:60]}" (year only, dated mid-year)',
            )
    retrieved = _parse_datetime(_field(content, "retrieved_at"))
    suffix = (
        f" (retrieved {_day(retrieved)}, which is not an observation date)" if retrieved else ""
    )
    return None, f"Undated: the source gives no observation or publication date{suffix}"


def _registrations(db: Session, land_id: uuid.UUID) -> tuple[list[KnowledgeSource], set[uuid.UUID]]:
    rows = db.execute(
        select(LandImageRegistration, Evidence.content, Investigation.id)
        .join(LandArchiveImage, LandArchiveImage.evidence_id == LandImageRegistration.evidence_id)
        .join(Evidence, Evidence.id == LandArchiveImage.evidence_id)
        .join(ResearchRun, ResearchRun.id == Evidence.run_id)
        .join(Investigation, Investigation.id == ResearchRun.investigation_id)
        .where(Investigation.land_id == land_id)
        .distinct(LandImageRegistration.evidence_id)
        .order_by(LandImageRegistration.evidence_id, LandImageRegistration.created_at.desc())
        .limit(ROW_LIMIT)
    ).all()
    sources: list[KnowledgeSource] = []
    aligned: set[uuid.UUID] = set()
    for registration, content, investigation_id in rows:
        result = registration.result or {}
        footprint = result.get("footprint")
        if not isinstance(footprint, dict):
            continue
        aligned.add(registration.evidence_id)
        observed, label = _evidence_date(content)
        loo = _field(result, "leave_one_out_rms_m")
        uncertainty = loo if isinstance(loo, int | float) else _field(result, "rms_error_m")
        resolution = _float(_field(result, "approximate_m_per_pixel"))
        media = content.get("media") or {}
        sources.append(
            KnowledgeSource(
                key=f"registration:{registration.id}",
                kind="archive-alignment",
                title=f"Aligned image: {_text(media.get('title') or content.get('title'), 'archive image', 200)}",
                class_id=knowledge.class_for_resolution(resolution),
                resolution_m=resolution,
                observed_at=observed,
                observed_label=label,
                recorded_at=_utc(registration.created_at) or datetime.now(UTC),
                located=True,
                positional_uncertainty_m=_float(uncertainty),
                provenance=_provenance(
                    str(content.get("provider", "archive")),
                    media.get("license") or content.get("license"),
                    content.get("attribution"),
                    _field(media, "source_url") or content.get("url"),
                ),
                link={
                    "kind": "registration",
                    "id": str(registration.id),
                    "detail": {
                        "evidenceId": str(registration.evidence_id),
                        "investigationId": str(investigation_id),
                    },
                },
                geometry=shape(footprint),
            )
        )
    return sources, aligned


# ── Research evidence and field surveys: land-wide ──────────────────────────────────────────


def _land_wide(
    db: Session,
    land_id: uuid.UUID,
    aligned: set[uuid.UUID],
    not_counted: list[NotCountedRead],
) -> list[KnowledgeSource]:
    rows = db.execute(
        select(Evidence.id, Evidence.content, Evidence.created_at, ResearchRun.id, Investigation.id)
        .join(ResearchRun, ResearchRun.id == Evidence.run_id)
        .join(Investigation, Investigation.id == ResearchRun.investigation_id)
        .where(Investigation.land_id == land_id)
        .order_by(Evidence.created_at.desc(), Evidence.id)
        .limit(ROW_LIMIT)
    ).all()
    sources: list[KnowledgeSource] = []
    seen: set[tuple[str, str, str]] = set()
    skipped: dict[str, list[str]] = {}
    for evidence_id, content, created_at, run_id, investigation_id in rows:
        provider = str(content.get("provider", ""))
        title = _text(content.get("title"), "Research record", 300)
        if (
            provider in COUNTED_ELSEWHERE
            or content.get("inventory")
            or content.get("survey")
            or evidence_id in aligned
        ):
            continue
        if provider in DERIVED_PROVIDERS:
            skipped.setdefault(DERIVED_PROVIDERS[provider], []).append(title)
            continue
        if _field(content, "spatial_relevance") == "unresolved":
            skipped.setdefault("Where it applies could not be resolved to this land.", []).append(
                title
            )
            continue
        observed, label = _evidence_date(content)
        identity = (
            provider,
            str(_field(content, "record_id") or content.get("url") or title),
            observed.isoformat() if observed else "",
        )
        if identity in seen:
            continue
        seen.add(identity)
        # GBIF is a person's sighting, but its coordinates are withheld by the provider and
        # never drawn: field class, no location, so it counts land-wide at the regional cap.
        class_id: ClassId = "field" if provider == "gbif-occurrences" else "regional"
        where = (
            "Regional source"
            if provider in REGIONAL_PROVIDERS
            else "No published location"
            if provider == "gbif-occurrences"
            else "A document or page about the land"
        )
        sources.append(
            KnowledgeSource(
                key=f"evidence:{evidence_id}",
                kind="research-evidence",
                title=title,
                class_id=class_id,
                resolution_m=None,
                observed_at=observed,
                observed_label=label,
                recorded_at=_utc(created_at) or datetime.now(UTC),
                located=False,
                positional_uncertainty_m=None,
                provenance=_provenance(
                    provider or "research",
                    content.get("license"),
                    content.get("attribution"),
                    content.get("url"),
                ),
                link={
                    "kind": "evidence",
                    "id": str(evidence_id),
                    "detail": {"runId": str(run_id), "investigationId": str(investigation_id)},
                },
                class_note=f"{where}: counts for the whole land.",
            )
        )
    for reason, titles in skipped.items():
        shown = "; ".join(titles[:2]) + (f"; and {len(titles) - 2} more" if len(titles) > 2 else "")
        not_counted.append(NotCountedRead(title=f"Research: {shown}"[:300], reason=reason))
    return sources


def _surveys(db: Session, land_id: uuid.UUID) -> list[KnowledgeSource]:
    rows = db.scalars(
        select(LandSurvey)
        .where(LandSurvey.land_id == land_id)
        .order_by(LandSurvey.created_at.desc())
        .limit(ROW_LIMIT)
    ).all()
    sources: list[KnowledgeSource] = []
    for survey in rows:
        observed = _parse_datetime(_field(survey.content, "observed_on"))
        sources.append(
            KnowledgeSource(
                key=f"survey:{survey.id}",
                kind="survey",
                title=f"Field survey: {_text(survey.content.get('name'), 'survey', 200)}",
                class_id="field",
                resolution_m=None,
                observed_at=observed,
                observed_label=f"observed {_day(observed)}" if observed else "Undated",
                recorded_at=_utc(survey.created_at) or datetime.now(UTC),
                located=False,
                positional_uncertainty_m=None,
                provenance=_provenance(
                    "field survey",
                    "Workspace record",
                    f"Observer: {_text(survey.content.get('observer'), 'not stated', 200)}",
                ),
                link={"kind": "survey", "id": str(survey.id), "detail": {}},
                class_note="Survey plots have no mapped location yet: counts for the whole land.",
            )
        )
    return sources


def collect(
    db: Session, land: LandRead, grid: FogGrid
) -> tuple[list[KnowledgeSource], list[NotCountedRead]]:
    """Every stored record about the Land, in a fixed number of queries."""
    not_counted: list[NotCountedRead] = list(NOT_COUNTED)
    registrations, aligned = _registrations(db, land.id)
    sources = [
        *_captures(db, land.id, not_counted),
        *_inspections(db, land.id),
        *_rasters(db, land.id, grid),
        *registrations,
        *_land_wide(db, land.id, aligned, not_counted),
        *_surveys(db, land.id),
    ]
    return sources, not_counted


# ── Public: the grid, a cell, the foggiest patches ──────────────────────────────────────────


def _as_of(value: datetime | None) -> datetime:
    now = datetime.now(UTC)
    if value is None:
        return now
    moment = _utc(value) or now
    if moment > now + timedelta(minutes=5):
        raise InvalidInputError("The fog can be computed as of now or an earlier time, not later.")
    return min(moment, now)


@dataclass
class _Fog:
    land: LandRead
    grid: FogGrid
    composed: Composed
    not_counted: list[NotCountedRead]
    as_of: datetime


def _fog(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    cell_m: float,
    as_of: datetime | None,
) -> _Fog:
    land = get_land(db, workspace_id, land_id)
    moment = _as_of(as_of)
    boundary = shape(land.boundary.model_dump())
    try:
        grid = knowledge.make_fog_grid(boundary, cell_m)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    sources, not_counted = collect(db, land, grid)
    composed = knowledge.compose(grid, sources, moment)
    return _Fog(land, grid, composed, not_counted, moment)


def _record(item: knowledge.Counted, as_of: datetime) -> KnowledgeRecordRead:
    source = item.source
    age = knowledge.age_days(source.observed_at, as_of)
    return KnowledgeRecordRead(
        key=source.key,
        kind=source.kind,
        title=source.title[:500],
        class_id=item.weighed.class_id,
        counted_class_id=item.weighed.counted_class_id,
        class_note=item.weighed.note,
        resolution_m=source.resolution_m,
        observed_at=source.observed_at,
        observed_label=source.observed_label[:300],
        recorded_at=source.recorded_at,
        age_days=age,
        peak_credit=item.weighed.credit,
        arithmetic=item.weighed.arithmetic,
        located=source.located,
        positional_uncertainty_m=source.positional_uncertainty_m,
        cells=int(np.count_nonzero(item.mask)),
        provenance=KnowledgeProvenance.model_validate(source.provenance),
        link=KnowledgeLink.model_validate(source.link),
    )


def knowledge_grid(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    cell_m: float = 20,
    as_of: datetime | None = None,
) -> KnowledgeGridRead:
    fog = _fog(db, workspace_id, land_id, cell_m, as_of)
    spec = fog.grid.spec
    return KnowledgeGridRead(
        land_id=fog.land.id,
        boundary_revision=fog.land.revision,
        as_of=fog.as_of,
        density=encode(spec, fog.composed.density, "u8", scale=1 / 250, nodata=OUTSIDE),
        best=encode(spec, fog.composed.best, "u16", nodata=NO_RECORD),
        records=[_record(item, fog.as_of) for item in fog.composed.counted],
        summary=KnowledgeSummary.model_validate(knowledge.summary(fog.grid, fog.composed)),
        policy=policy(),
        not_counted=fog.not_counted,
    )


def knowledge_cell(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    longitude: float,
    latitude: float,
    cell_m: float = 20,
    as_of: datetime | None = None,
) -> KnowledgeCellRead:
    fog = _fog(db, workspace_id, land_id, cell_m, as_of)
    spec = fog.grid.spec
    index = spec.cell_index(longitude, latitude)
    inside = index is not None and bool(fog.grid.inside[index[1], index[0]])
    records: list[KnowledgeCellRecord] = []
    density: float | None = None
    bounds: list[float] | None = None
    if index is not None:
        column, row = index
        west = spec.west + column * spec.dlon
        north = spec.north - row * spec.dlat
        bounds = [west, north - spec.dlat, west + spec.dlon, north]
    if index is not None and inside:
        column, row = index
        density = float(fog.composed.density[row, column]) / 250
        best = int(fog.composed.best[row, column])
        covering = [
            (position, item)
            for position, item in enumerate(fog.composed.counted)
            if item.mask[row, column]
        ]
        covering.sort(key=lambda pair: (-pair[1].weighed.credit, pair[0]))
        records = [
            KnowledgeCellRecord(
                record=_record(item, fog.as_of),
                credit=item.weighed.credit,
                best=position == best,
                arithmetic=item.weighed.arithmetic,
            )
            for position, item in covering
        ]
    return KnowledgeCellRead(
        longitude=longitude,
        latitude=latitude,
        column=index[0] if index else None,
        row=index[1] if index else None,
        inside=inside,
        density=density,
        cell_m=spec.cell_m,
        bounds=bounds,
        records=records,
        not_counted=fog.not_counted,
        policy_version=POLICY_VERSION,
    )


def _age_phrase(item: knowledge.Counted, as_of: datetime) -> str:
    age = knowledge.age_days(item.source.observed_at, as_of)
    if age is None:
        return "undated"
    if age < 1:
        return "today"
    return f"{age:.0f} day{'s' if round(age) != 1 else ''} old"


def _reason(
    patch: knowledge.ScoutPatch,
    fog: _Fog,
    truncated: bool,
    focused: bool,
    threshold: float,
    tied: bool,
) -> str:
    acres = patch.area_m2 / 4046.8564224
    if truncated:
        where = "nearest your view" if focused else "nearest the middle of the land"
        lead = f"Foggiest {acres:,.0f} ac within one battery, {where}"
        if tied:
            lead += " (the fog is even here, so the nearest cells came first)"
    else:
        lead = f"{acres:,.0f} ac at least {threshold:.2f} foggy"
    lead += f"; mean density {patch.mean_density:.2f}"
    if patch.dominant_record is None:
        return f"{lead}; no stored evidence covers it."
    best = fog.composed.counted[patch.dominant_record]
    return f"{lead}; best evidence here: {best.source.title}, {_age_phrase(best, fog.as_of)}."


def scout_areas(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, request: KnowledgeScoutRequest
) -> KnowledgeScoutRead:
    fog = _fog(db, workspace_id, land_id, request.cell_m, None)
    focus = (request.focus[0], request.focus[1]) if request.focus else None
    patches, truncated = knowledge.scout(
        fog.grid,
        fog.composed,
        threshold=request.threshold,
        budget_m2=request.budget_m2,
        focus=focus,
        max_areas=request.max_areas,
    )
    candidates = fog.composed.density[
        fog.grid.inside & (fog.composed.density >= request.threshold * 250 - 1e-6)
    ]
    tied = bool(candidates.size) and int(candidates.max()) - int(candidates.min()) < (
        knowledge.SCOUT_TIE_STEP * 250
    )
    areas: list[KnowledgeScoutArea] = []
    for patch in patches:
        try:
            footprint = Polygon.model_validate(_mapping(patch.footprint))
        except ValidationError:
            continue
        areas.append(
            KnowledgeScoutArea(
                footprint=footprint,
                area_m2=patch.area_m2,
                cells=patch.cells,
                mean_density=patch.mean_density,
                best_class_counts=patch.best_class_counts,
                reason=_reason(patch, fog, truncated, focus is not None, request.threshold, tied),
            )
        )
    return KnowledgeScoutRead(
        areas=areas,
        threshold=request.threshold,
        budget_m2=request.budget_m2,
        truncated_by_budget=truncated,
        forecast=FORECAST,
        policy_version=POLICY_VERSION,
    )


def _mapping(geometry: BaseGeometry) -> dict[str, Any]:
    mapped = mapping(geometry)
    return {
        "type": mapped["type"],
        "coordinates": [[list(point[:2]) for point in ring] for ring in mapped["coordinates"]],
    }
