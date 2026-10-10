"""Dated, quality-masked Sentinel-2 NDVI samples and comparisons on common cells."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

import httpx
import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import HttpUrl
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as copy_raster
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

from app.analysis.raster_grid import AnalysisGrid, make_grid
from app.analysis.raster_io import RasterOpener, ReadBudget
from app.analysis.terrain import TerrainResult, band_statistics
from app.research.providers.base import SourceSpec, fetch_json
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import (
    RasterBand,
    RasterMetadata,
    RasterRequest,
    RasterSource,
    VegetationObservation,
    VegetationPeriod,
    VegetationTimeline,
)

COLLECTION = "sentinel-2-c1-l2a"
PREFIX = "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/"
LICENSE_URL = "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice"
SPEC = SourceSpec(
    id="sentinel-2-ndvi",
    name="Sentinel-2 vegetation observations",
    domain="ecology",
    coverage="Mapped land within Sentinel-2 acquisition coverage; clouds and missing scenes limit observations",
    resolution="10 m red/NIR reflectance, 20 m scene classification; analysis grid at least 20 m",
    license="Copernicus Sentinel Data legal notice (free, full and open use with attribution)",
    attribution="Contains modified Copernicus Sentinel data; Collection 1 COGs and catalog by Element 84 Earth Search",
    documentation_url="https://registry.opendata.aws/sentinel-2-l2a-cogs/",
    endpoint="https://earth-search.aws.element84.com/v1/search",
)
QUALITY = {
    0: "No data",
    1: "Saturated / defective",
    2: "Dark / cast shadow",
    3: "Cloud shadow",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud, medium probability",
    9: "Cloud, high probability",
    10: "Cirrus",
    11: "Snow / ice",
}
BANDS = {"red": "B04", "nir": "B08", "scl": "SCL"}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def catalog_candidates(
    client: httpx.Client, grid: AnalysisGrid, period: VegetationPeriod
) -> tuple[list[dict[str, Any]], RasterSource, bool, int]:
    catalog, query_url = fetch_json(
        client,
        SPEC,
        {
            "collections": COLLECTION,
            "bbox": ",".join(str(v) for v in grid.bounds),
            "datetime": f"{period.start_date}T00:00:00Z/{period.end_date}T23:59:59.999999Z",
            "limit": 25,
            "sortby": "+properties.eo:cloud_cover",
        },
    )
    items = catalog.get("features", [])
    if not isinstance(items, list) or len(items) > 25:
        raise ValueError("The vegetation catalog returned an unexpected result size.")
    matched = catalog.get("numberMatched", catalog.get("context", {}).get("matched"))
    truncated = (isinstance(matched, int) and matched > len(items)) or (
        any(link.get("rel") == "next" for link in catalog.get("links", []))
        and matched != len(items)
    )
    candidates = []
    for item in items:
        identifier = item.get("id", "")
        if item.get("collection") != COLLECTION or not re.fullmatch(
            r"S2[ABC]_T\d{2}[A-Z]{3}_\d{8}T\d{6}_L2A", identifier
        ):
            raise ValueError("The vegetation catalog returned an unsupported scene identifier.")
        properties = item.get("properties", {})
        observed = datetime.fromisoformat(properties.get("datetime", "").replace("Z", "+00:00"))
        if observed.tzinfo is None or not period.start_date <= observed.date() <= period.end_date:
            raise ValueError("The vegetation scene is outside its requested observation window.")
        cloud = properties.get("eo:cloud_cover")
        if not isinstance(cloud, (int, float)) or not np.isfinite(cloud) or not 0 <= cloud <= 100:
            raise ValueError("The vegetation scene has invalid cloud metadata.")
        coverage = shape(item["geometry"]).intersection(grid.geometry).area / grid.geometry.area
        if coverage > 0:
            candidates.append((coverage, cloud, identifier, item))
    # Prefer spatial coverage, then catalog cloud fraction; measure local SCL coverage for up to three.
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    source = RasterSource(
        id=f"catalog/{period.start_date}/{period.end_date}",
        url=HttpUrl(query_url),
        catalog_url=HttpUrl(query_url),
        etag=None,
        last_modified=None,
        catalog_datetime=None,
        observation_period=f"Candidate search {period.start_date} through {period.end_date}",
        license=SPEC.license,
        license_url=HttpUrl(LICENSE_URL),
        attribution=SPEC.attribution,
        catalog_sha256=digest(catalog),
        purpose="bounded scene selection",
    )
    return [value[3] for value in candidates[:3]], source, truncated, len(items)


def read_band(
    client: httpx.Client, budget: ReadBudget, item: dict[str, Any], key: str, grid: AnalysisGrid
) -> tuple[NDArray[np.float32], RasterSource]:
    asset = item.get("assets", {}).get(key, {})
    href = asset.get("href", "")
    if (
        not isinstance(href, str)
        or not href.startswith(PREFIX)
        or not href.endswith(f"/{item['id']}/{BANDS[key]}.tif")
    ):
        raise ValueError("The vegetation catalog returned an unsupported public band location.")
    bands = asset.get("raster:bands", [])
    if not isinstance(bands, list) or len(bands) != 1 or bands[0].get("nodata") != 0:
        raise ValueError("The vegetation band lacks its expected no-data metadata.")
    info = bands[0]
    expected_type = "uint8" if key == "scl" else "uint16"
    if info.get("data_type") != expected_type or info.get("spatial_resolution") != (
        20 if key == "scl" else 10
    ):
        raise ValueError("The vegetation band has unexpected type or source resolution.")
    scale, offset = (1.0, 0.0) if key == "scl" else (info.get("scale"), info.get("offset"))
    if (
        not isinstance(scale, (int, float))
        or not isinstance(offset, (int, float))
        or not np.isfinite([scale, offset]).all()
        or not 0 < scale <= 1
        or not -1 <= offset <= 1
    ):
        raise ValueError("The reflectance band lacks a supported explicit scale and offset.")
    opener = RasterOpener(client, href, budget)
    with rasterio.open(href, opener=opener, driver="GTiff") as source:
        if (
            source.count != 1
            or source.crs is None
            or source.dtypes != (expected_type,)
            or source.nodata != 0
        ):
            raise ValueError("The vegetation raster does not match its declared band metadata.")
        with WarpedVRT(
            source,
            crs=grid.crs.to_wkt(),
            transform=grid.affine,
            width=grid.width,
            height=grid.height,
            dtype="float32",
            src_nodata=0,
            nodata=float("nan"),
            resampling=Resampling.nearest,
        ) as warped:
            values = warped.read(1, masked=True).filled(np.nan).astype(np.float32)
    if key != "scl":
        values = values * np.float32(scale) + np.float32(offset)
    version = opener.files[-1]
    record = RasterSource(
        id=f"{item['id']}/{key}",
        url=HttpUrl(href),
        catalog_url=HttpUrl(
            f"https://earth-search.aws.element84.com/v1/collections/{COLLECTION}/items/{item['id']}"
        ),
        etag=version.etag,
        last_modified=version.last_modified,
        catalog_datetime=item["properties"]["datetime"],
        observation_period=f"Satellite acquisition {item['properties']['datetime']}",
        license=SPEC.license,
        license_url=HttpUrl(LICENSE_URL),
        attribution=SPEC.attribution,
        band=key,
        scale=float(scale),
        offset=float(offset),
        catalog_sha256=digest(item),
        purpose="candidate quality mask" if key == "scl" else "selected reflectance",
    )
    return values, record


def ndvi(
    red: NDArray[np.float32],
    nir: NDArray[np.float32],
    scl: NDArray[np.float32],
    inside: NDArray[np.bool_],
) -> NDArray[np.float32]:
    # Do not interpolate cloud-contaminated reflectance into a clear target sample.
    # SCL 4 and 5 include vegetated and nonvegetated land; water is not a land vegetation sample.
    denominator = nir + red
    valid = (
        inside
        & np.isin(scl, [4, 5])
        & np.isfinite(red)
        & np.isfinite(nir)
        & (red >= 0)
        & (nir >= 0)
        & (denominator > 1e-6)
    )
    values = np.full(red.shape, np.nan, dtype=np.float32)
    np.divide(nir - red, denominator, out=values, where=valid)
    return values


def encode(
    bands: list[NDArray[np.float32]],
    definitions: list[RasterBand],
    grid: AnalysisGrid,
    metadata: RasterMetadata,
) -> bytes:
    with MemoryFile() as source, MemoryFile() as output:
        with source.open(
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=len(bands),
            dtype="float32",
            crs=grid.crs.to_wkt(),
            transform=grid.affine,
            nodata=-9999.0,
        ) as dataset:
            for index, (values, definition) in enumerate(zip(bands, definitions, strict=True), 1):
                dataset.write(
                    np.where(np.isfinite(values), values, -9999).astype(np.float32), index
                )
                dataset.set_band_description(index, definition.name)
                dataset.set_band_unit(index, definition.unit)
            dataset.update_tags(
                algorithm="sentinel-2-ndvi-v1",
                interpretation=(
                    "Quality-masked sampled vegetation index; not species cover, biomass or restoration "
                    "success."
                ),
                attribution=SPEC.attribution,
                license=SPEC.license,
                license_url=LICENSE_URL,
                analysis_metadata=metadata.model_dump_json(),
            )
        copy_raster(
            source.name,
            output.name,
            driver="COG",
            compress="DEFLATE",
            blocksize=128,
            overview_resampling="AVERAGE",
        )
        data: bytes = output.read()
    if len(data) > 16 * 1024 * 1024:
        raise ValueError(
            "The vegetation raster exceeds its output limit. Use fewer dates or a smaller grid."
        )
    return data


def analyze(boundary: Footprint, request: RasterRequest, client: httpx.Client) -> TerrainResult:
    grid = make_grid(boundary, request)
    total = int(np.count_nonzero(grid.inside))
    budget = ReadBudget(max_bytes=128 * 1024 * 1024, max_requests=768, max_seconds=140)
    sources: list[RasterSource] = []
    observations: list[VegetationObservation] = []
    arrays: list[NDArray[np.float32]] = []
    definitions: list[RasterBand] = []
    warnings = [
        (
            "NDVI measures a red/near-infrared vegetation signal. It does not identify species, "
            "percent species cover, habitat quality, biomass, native status or restoration "
            "success."
        ),
        (
            "Only SCL classes 4 (vegetation) and 5 (not vegetated) are sampled. Water, snow, "
            "shadows, unclassified, defective and cloudy cells are excluded; classification errors "
            "and thin clouds may remain."
        ),
        (
            "Each window selects one scene with the most clear land samples among up to three "
            "candidates. Candidate ranking favors boundary overlap then catalog cloud fraction, "
            "from at most 25 catalog results. This is not an exhaustive best-image search or a "
            "temporal composite."
        ),
        (
            "Nearest-neighbor red/NIR reflectance samples use the explicit source scale and offset "
            "before NDVI. Scene classification is nominally 20 m; the analysis grid may be "
            "coarser. Small features and narrow exclusions may be unresolved."
        ),
        (
            "The trend uses the same cells that are valid in every observation. First-to-last "
            "change uses cells valid at both endpoints. Missing samples are not zero and different "
            "observation footprints must not be compared as whole-land changes."
        ),
        (
            "Season, acquisition date, weather, illumination, management and sensor/processing "
            "differences can affect NDVI. A difference does not establish a cause or ecological "
            "improvement; compare similar seasons and check field evidence."
        ),
    ]
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=64 * 1024 * 1024,
        AWS_NO_SIGN_REQUEST="YES",
    ):
        for index, period in enumerate(request.periods, 1):
            candidates, catalog_source, truncated, count = catalog_candidates(client, grid, period)
            sources.append(catalog_source)
            best: tuple[int, dict[str, Any], NDArray[np.float32]] | None = None
            for item in candidates:
                quality, source = read_band(client, budget, item, "scl", grid)
                sources.append(source)
                clear = int(np.count_nonzero(grid.inside & np.isin(quality, [4, 5])))
                if best is None or clear > best[0]:
                    best = (clear, item, quality)
            values = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
            quality_counts = {}
            selected_id, acquired = None, None
            explanation = "No matching satellite scene was returned for this window."
            if best is not None:
                _, selected, quality = best
                selected_id = selected["id"]
                acquired = datetime.fromisoformat(
                    selected["properties"]["datetime"].replace("Z", "+00:00")
                )
                quality_counts = {
                    label: int(np.count_nonzero(grid.inside & (quality == code)))
                    for code, label in QUALITY.items()
                    if code != 0
                }
                quality_counts["No data"] = int(
                    np.count_nonzero(grid.inside & (~np.isfinite(quality) | (quality == 0)))
                )
                if best[0]:
                    red, red_source = read_band(client, budget, selected, "red", grid)
                    nir, nir_source = read_band(client, budget, selected, "nir", grid)
                    sources.extend([red_source, nir_source])
                    values = ndvi(red, nir, quality, grid.inside)
                    explanation = (
                        "One selected acquisition; summary uses clear sampled land cells, which may cover only "
                        "part of the boundary."
                    )
                else:
                    explanation = (
                        "Scenes were found but the examined quality masks contain no clear land samples within "
                        "this boundary."
                    )
            definition = band_statistics(
                values, index, f"NDVI {period.start_date} to {period.end_date}", "NDVI"
            )
            definition.palette = "viridis"
            definition.display_minimum, definition.display_maximum = -1, 1
            arrays.append(values)
            definitions.append(definition)
            observations.append(
                VegetationObservation(
                    period=period,
                    band=index,
                    scene_id=selected_id,
                    acquired_at=acquired,
                    candidate_count=count,
                    catalog_truncated=truncated,
                    quality_candidates_examined=len(candidates),
                    valid_cells=definition.valid_cells,
                    coverage_fraction=definition.valid_cells / total if total else None,
                    mean_ndvi=definition.mean,
                    common_mean_ndvi=None,
                    quality_counts=quality_counts,
                    explanation=explanation,
                )
            )
            if truncated:
                warnings.append(
                    f"Catalog results for {period.start_date} through {period.end_date} were truncated "
                    "to 25 candidates; only the stated candidates were examined."
                )
    common = grid.inside & np.logical_and.reduce([np.isfinite(values) for values in arrays])
    common_count = int(np.count_nonzero(common))
    for observation, values in zip(observations, arrays, strict=True):
        observation.common_mean_ndvi = (
            float(np.mean(values[common], dtype=np.float64)) if common_count else None
        )
    change_band, change_cells, mean_change = None, 0, None
    if len(arrays) > 1:
        change = arrays[-1] - arrays[0]
        change_band = len(arrays) + 1
        definition = band_statistics(
            change, change_band, "NDVI change: last minus first observation", "NDVI difference"
        )
        definition.palette = "rdylgn"
        definition.display_minimum, definition.display_maximum = -2, 2
        change_cells, mean_change = definition.valid_cells, definition.mean
        arrays.append(change)
        definitions.append(definition)
    if not total:
        warnings.append(
            "No analysis cell centers fall inside this boundary; the source cannot resolve this "
            "land at the selected grid spacing."
        )
    if len(observations) > 1 and not common_count:
        warnings.append(
            "There are no common clear land cells across all requested observations. No comparable "
            "trend can be calculated."
        )
    if grid.resolution > request.resolution_m * 1.001:
        warnings.append(
            f"The grid was coarsened to {grid.resolution:.1f} m to fit the analysis dimensions."
        )
    valid_any = int(
        np.count_nonzero(
            np.logical_or.reduce([np.isfinite(values) for values in arrays[: len(observations)]])
        )
    )
    metadata = RasterMetadata(
        algorithm="sentinel-2-ndvi-v1",
        method=(
            "Sentinel-2 Collection 1 L2A; bounded catalog and local SCL scene selection; explicit "
            "reflectance scale/offset; NDVI=(NIR-red)/(NIR+red); nearest samples on a shared local "
            "metric grid; polygon/hole cell-center mask; common-cell temporal summaries."
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
        vegetation=VegetationTimeline(
            observations=observations,
            common_cells=common_count,
            common_coverage_fraction=common_count / total if total else None,
            change_band=change_band,
            change_cells=change_cells,
            mean_change=mean_change,
        ),
    )
    return TerrainResult(encode(arrays, definitions, grid, metadata), metadata)
