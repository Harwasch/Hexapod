"""Seasonal Sentinel-2 signatures: four reflectance bands and NDVI per dated window, 10 m.

A cell's signature is its green, red, near-infrared and shortwave-infrared reflectance and its
NDVI in each window, sampled only where that window's own scene classification says clear
land. Comparing cells (from the stored bands, later) is relative to the variety on this land;
it is never a species identification, cover estimate or habitat assessment.
"""

from __future__ import annotations

from typing import Literal

import httpx
import numpy as np
import rasterio
from numpy.typing import NDArray

from app.analysis.raster_grid import make_grid
from app.analysis.raster_io import ReadBudget
from app.analysis.terrain import TerrainResult, band_statistics
from app.analysis.vegetation import SPEC as VEGETATION_SPEC
from app.analysis.vegetation import encode, ndvi, quality_counts, read_band, select_scene
from app.research.providers.base import SourceSpec
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import (
    RasterBand,
    RasterMetadata,
    RasterRequest,
    RasterSource,
    SignatureFeature,
    SignatureInfo,
    VegetationObservation,
    VegetationTimeline,
)

ALGORITHM = "sentinel-2-signature-v1"
SPEC = SourceSpec(
    id="sentinel-2-signature",
    name="Sentinel-2 seasonal signatures",
    domain="ecology",
    coverage=VEGETATION_SPEC.coverage,
    resolution="10 m green/red/NIR and 20 m SWIR reflectance with a 20 m scene classification, "
    "sampled nearest on a 10 m grid",
    license=VEGETATION_SPEC.license,
    attribution=VEGETATION_SPEC.attribution,
    documentation_url=VEGETATION_SPEC.documentation_url,
    endpoint=VEGETATION_SPEC.endpoint,
)
# Band order inside every window. NDVI repeats red/NIR information, deliberately: it is the
# seasonal greenness a person reads, and it weighs phenology in the similarity.
FeatureKey = Literal["green", "red", "nir", "swir16", "ndvi"]
FEATURES: tuple[tuple[FeatureKey, str, str], ...] = (
    ("green", "Green (B03)", "reflectance"),
    ("red", "Red (B04)", "reflectance"),
    ("nir", "NIR (B08)", "reflectance"),
    ("swir16", "SWIR (B11)", "reflectance"),
    ("ndvi", "NDVI", "NDVI"),
)
REFLECTANCE: tuple[FeatureKey, ...] = ("green", "red", "nir", "swir16")
# Two quality masks per window keep four windows inside the processor's time limit through
# a proxy (about a minute measured); the warning says so.
CANDIDATES_PER_WINDOW = 2
STANDARDIZATION = "z-score per feature over cells valid in every window inside the boundary"
SIMILARITY = "cosine of standardized vectors"
INTERPRETATION = (
    "Quality-masked seasonal reflectance and NDVI; a similar signature is not a species "
    "identification, cover estimate or habitat assessment."
)


def analyze(boundary: Footprint, request: RasterRequest, client: httpx.Client) -> TerrainResult:
    grid = make_grid(boundary, request)
    total = int(np.count_nonzero(grid.inside))
    budget = ReadBudget(max_bytes=128 * 1024 * 1024, max_requests=1024, max_seconds=180)
    sources: list[RasterSource] = []
    observations: list[VegetationObservation] = []
    arrays: list[NDArray[np.float32]] = []
    definitions: list[RasterBand] = []
    features: list[tuple[int, int, FeatureKey, str]] = []
    windows = len(request.periods)
    warnings = [
        (
            f"Similar spectral signature at {grid.resolution:.0f} m across {windows} dates; "
            "not a species identification, cover estimate or habitat assessment."
        ),
        (
            "Alike is relative to the variety on this land: each band is standardized over this "
            "land's common cells, so the same place can score differently inside another boundary."
        ),
        (
            "The scene classification (SCL) mask is 20 m and B11 is 20 m, resampled nearest to "
            "10 m: one 20 m source pixel spans four signature cells."
        ),
        (
            "Only SCL classes 4 (vegetation) and 5 (not vegetated) are sampled, and one mask per "
            "window applies to all five bands. Water, snow, shadows, unclassified, defective and "
            "cloudy cells are excluded; classification errors and thin clouds may remain."
        ),
        (
            f"Each window selects one scene with the most clear land samples among up to "
            f"{CANDIDATES_PER_WINDOW} candidates. Candidate ranking favors boundary overlap then "
            "catalog cloud fraction, from at most 25 catalog results. This is not an exhaustive "
            "best-image search or a temporal composite."
        ),
        (
            "Reflectance uses each file's explicit scale and offset. Season, acquisition date, "
            "weather, illumination, management and sensor/processing differences change "
            "reflectance and NDVI; a signature describes these dates, not a lasting condition."
        ),
    ]
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=64 * 1024 * 1024,
        AWS_NO_SIGN_REQUEST="YES",
    ):
        for window, period in enumerate(request.periods):
            choice = select_scene(client, budget, grid, period, limit=CANDIDATES_PER_WINDOW)
            sources.append(choice.catalog_source)
            sources.extend(choice.scl_sources)
            stack = {
                key: np.full((grid.height, grid.width), np.nan, dtype=np.float32)
                for key, _, _ in FEATURES
            }
            counts: dict[str, int] = {}
            explanation = "No matching satellite scene was returned for this window."
            if choice.item is not None and choice.quality is not None:
                counts = quality_counts(grid.inside, choice.quality)
                if choice.clear:
                    for key in REFLECTANCE:
                        values, source = read_band(client, budget, choice.item, key, grid)
                        stack[key] = values
                        sources.append(source)
                    stack["ndvi"] = ndvi(stack["red"], stack["nir"], choice.quality, grid.inside)
                    # A cell is a sample in this window only where all five bands are valid,
                    # so every window's bands share one footprint.
                    valid = np.logical_and.reduce([np.isfinite(stack[key]) for key in stack])
                    for key in stack:
                        stack[key] = np.where(valid, stack[key], np.nan).astype(np.float32)
                    explanation = (
                        "One selected acquisition; its five bands use the clear land cells of its "
                        "own quality mask, which may cover only part of the boundary."
                    )
                else:
                    explanation = (
                        "Scenes were found but the examined quality masks contain no clear land "
                        "samples within this boundary."
                    )
            acquired = choice.acquired
            dated = (
                acquired.date().isoformat()
                if acquired
                else f"{period.start_date} to {period.end_date}"
            )
            ndvi_band = 0
            for key, label, unit in FEATURES:
                band = len(arrays) + 1
                definition = band_statistics(stack[key], band, f"{dated} {label}", unit)
                definition.palette = "viridis"
                definition.display_minimum, definition.display_maximum = (
                    (-1, 1) if key == "ndvi" else (0, 0.6)
                )
                arrays.append(stack[key])
                definitions.append(definition)
                features.append((band, window, key, definition.name))
                if key == "ndvi":
                    ndvi_band = band
            valid_cells = int(np.count_nonzero(np.isfinite(stack["ndvi"])))
            observations.append(
                VegetationObservation(
                    period=period,
                    band=ndvi_band,
                    scene_id=choice.item["id"] if choice.item else None,
                    acquired_at=acquired,
                    candidate_count=choice.count,
                    catalog_truncated=choice.truncated,
                    quality_candidates_examined=choice.candidates,
                    valid_cells=valid_cells,
                    coverage_fraction=valid_cells / total if total else None,
                    mean_ndvi=definitions[ndvi_band - 1].mean,
                    common_mean_ndvi=None,
                    quality_counts=counts,
                    explanation=explanation,
                )
            )
            if choice.truncated:
                warnings.append(
                    f"Catalog results for {period.start_date} through {period.end_date} were "
                    "truncated to 25 candidates; only the stated candidates were examined."
                )
            if not valid_cells:
                warnings.append(
                    f"The {period.start_date} to {period.end_date} window has no clear land "
                    "samples, so no cell has a full signature. Run again without this window or "
                    "with other dates; gaps are never filled by interpolation."
                )
    common = grid.inside & np.logical_and.reduce([np.isfinite(values) for values in arrays])
    common_count = int(np.count_nonzero(common))
    for observation in observations:
        values = arrays[observation.band - 1]
        observation.common_mean_ndvi = (
            float(np.mean(values[common], dtype=np.float64)) if common_count else None
        )
    signature_features = []
    for band, window, key, name in features:
        values = arrays[band - 1][common].astype(np.float64)
        signature_features.append(
            SignatureFeature(
                band=band,
                window=window,
                key=key,
                name=name,
                common_mean=float(values.mean()) if common_count else None,
                common_standard_deviation=float(values.std()) if common_count else None,
            )
        )
    if not total:
        warnings.append(
            "No analysis cell centers fall inside this boundary; the source cannot resolve this "
            "land at the selected grid spacing."
        )
    elif not common_count:
        warnings.append(
            "No cell is clear in every window, so no signatures can be compared. Choose dates "
            "with clearer scenes or fewer windows."
        )
    if grid.resolution > request.resolution_m * 1.001:
        warnings.append(
            f"The grid was coarsened to {grid.resolution:.1f} m to fit the analysis dimensions."
        )
    valid_any = int(
        np.count_nonzero(np.logical_or.reduce([np.isfinite(values) for values in arrays]))
    )
    common_fraction = common_count / total if total else None
    metadata = RasterMetadata(
        algorithm=ALGORITHM,
        method=(
            "Sentinel-2 Collection 1 L2A; per window, bounded catalog search and local SCL scene "
            f"selection from at most {CANDIDATES_PER_WINDOW} quality masks; explicit reflectance "
            "scale/offset; green B03, red B04, NIR B08 and SWIR B11 reflectance and "
            "NDVI=(NIR-red)/(NIR+red), nearest samples on a shared local metric grid; one SCL 4/5 "
            "mask per window applied to all five bands; polygon/hole cell-center mask; common "
            "cells are valid in every band of every window."
        ),
        bounds=list(grid.bounds),
        crs=grid.crs.to_wkt(),
        resolution_m=grid.resolution,
        width=grid.width,
        height=grid.height,
        boundary_cells=total,
        valid_cells=valid_any,
        coverage_fraction=valid_any / total if total else None,
        sampled_area_m2=valid_any * grid.resolution**2,
        bands=definitions,
        sources=sources,
        warnings=warnings,
        downloaded_bytes=budget.downloaded_bytes,
        # The NDVI bands, dated by acquisition: the same timeline a vegetation run shows.
        vegetation=VegetationTimeline(
            observations=observations,
            common_cells=common_count,
            common_coverage_fraction=common_fraction,
            change_band=None,
            change_cells=0,
            mean_change=None,
        ),
        signature=SignatureInfo(
            features=signature_features,
            common_cells=common_count,
            common_coverage_fraction=common_fraction,
            standardization=STANDARDIZATION,
            similarity=SIMILARITY,
        ),
    )
    # Band interleaving reads one band without decoding the other nineteen (map tiles), and
    # compressed smaller than pixel interleaving with or without a float predictor (measured).
    data = encode(
        arrays,
        definitions,
        grid,
        metadata,
        algorithm=ALGORITHM,
        interpretation=INTERPRETATION,
        name="signature",
        interleave="BAND",
    )
    return TerrainResult(data, metadata)
