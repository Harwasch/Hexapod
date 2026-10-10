"""ESA WorldCover 2021 broad land-cover classes, never species/condition inference."""

from __future__ import annotations

import json
import re

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
from app.analysis.terrain import TerrainResult
from app.research.providers.base import SourceSpec, fetch_json
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import (
    RasterBand,
    RasterClass,
    RasterMetadata,
    RasterRequest,
    RasterSource,
)

GRID_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.geojson"
)
SPEC = SourceSpec(
    id="esa-worldcover-2021",
    name="ESA WorldCover 2021 land cover",
    domain="ecology",
    coverage="Global mapped land; source coverage and classification uncertainty remain",
    resolution="10 m broad land-cover classes; not species cover or current field conditions",
    license="CC BY 4.0",
    attribution="© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) "
    "processed by ESA WorldCover consortium",
    documentation_url="https://esa-worldcover.org/en/data-access",
    endpoint=GRID_URL,
)
CLASSES = (
    (10, "Tree cover", "#006400"),
    (20, "Shrubland", "#ffbb22"),
    (30, "Grassland", "#ffff4c"),
    (40, "Cropland", "#f096ff"),
    (50, "Built-up", "#fa0000"),
    (60, "Bare / sparse vegetation", "#b4b4b4"),
    (70, "Snow and ice", "#f0f0f0"),
    (80, "Permanent water bodies", "#0064c8"),
    (90, "Herbaceous wetland", "#0096a0"),
    (95, "Mangroves", "#00cf75"),
    (100, "Moss and lichen", "#fae6a0"),
)


def encode(values: NDArray[np.uint8], grid: AnalysisGrid) -> bytes:
    with MemoryFile() as source, MemoryFile() as output:
        with source.open(
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="uint8",
            crs=grid.crs.to_wkt(),
            transform=grid.affine,
            nodata=0,
        ) as dataset:
            dataset.write(values, 1)
            dataset.set_band_description(1, "WorldCover 2021 land-cover class")
            dataset.set_band_unit(1, "class code")
            dataset.write_colormap(
                1, {code: (*bytes.fromhex(color[1:]), 255) for code, _, color in CLASSES}
            )
            dataset.update_tags(
                algorithm="worldcover-v1",
                source="ESA WorldCover 2021 v200",
                reference_year="2021",
                license=SPEC.license,
                attribution=SPEC.attribution,
                classes=json.dumps({code: label for code, label, _ in CLASSES}),
                interpretation="Broad land-cover classification; not a species survey or current site condition.",
            )
        copy_raster(
            source.name,
            output.name,
            driver="COG",
            compress="DEFLATE",
            blocksize=128,
            overview_resampling="NEAREST",
        )
        result: bytes = output.read()
    if len(result) > 16 * 1024 * 1024:
        raise ValueError("The derived land-cover raster exceeds the output limit.")
    return result


def analyze(boundary: Footprint, request: RasterRequest, client: httpx.Client) -> TerrainResult:
    grid = make_grid(boundary, request)
    catalog, _ = fetch_json(client, SPEC, {})
    features = catalog.get("features", [])
    if not isinstance(features, list) or len(features) > 5000:
        raise ValueError("The WorldCover grid catalog has an unexpected structure.")
    tiles = []
    for feature in features:
        if shape(feature["geometry"]).intersection(grid.geometry).area > 0:
            identifier = feature.get("properties", {}).get("ll_tile", "")
            if not re.fullmatch(r"[NS]\d{2}[EW]\d{3}", identifier):
                raise ValueError("The WorldCover catalog returned an invalid tile identifier.")
            tiles.append(identifier)
    tiles = sorted(set(tiles))
    if not tiles:
        raise ValueError("WorldCover 2021 has no mapped tile covering this land.")
    if len(tiles) > 8:
        raise ValueError(
            "This area needs more than eight WorldCover tiles. Analyze smaller sections."
        )
    values = np.zeros((grid.height, grid.width), dtype=np.uint8)
    budget, sources = ReadBudget(), []
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=64 * 1024 * 1024,
        AWS_NO_SIGN_REQUEST="YES",
    ):
        for tile in tiles:
            identifier = f"ESA_WorldCover_10m_2021_v200_{tile}_Map"
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{identifier}.tif"
            opener = RasterOpener(client, url, budget)
            with rasterio.open(url, opener=opener, driver="GTiff") as source:
                if source.count != 1 or source.crs is None or source.dtypes != ("uint8",):
                    raise ValueError("The WorldCover source is not its expected categorical band.")
                with WarpedVRT(
                    source,
                    crs=grid.crs.to_wkt(),
                    transform=grid.affine,
                    width=grid.width,
                    height=grid.height,
                    dtype="uint8",
                    nodata=0,
                    resampling=Resampling.nearest,
                ) as warped:
                    samples = warped.read(1)
                np.copyto(values, samples, where=(values == 0) & (samples != 0))
            version = opener.files[-1]
            sources.append(
                RasterSource(
                    id=identifier,
                    url=HttpUrl(url),
                    catalog_url=HttpUrl(GRID_URL),
                    etag=version.etag,
                    last_modified=version.last_modified,
                    catalog_datetime=None,
                    observation_period="2021 reference-year classification, algorithm v200; not a current survey.",
                    license=SPEC.license,
                    license_url=HttpUrl("https://creativecommons.org/licenses/by/4.0/"),
                    attribution=SPEC.attribution,
                )
            )
    recognized = np.isin(values, [code for code, _, _ in CLASSES])
    unknown = int(np.count_nonzero(grid.inside & (values != 0) & ~recognized))
    values[~grid.inside | ~recognized] = 0
    total, valid = int(np.count_nonzero(grid.inside)), int(np.count_nonzero(values))
    classes = [
        RasterClass(
            code=code,
            label=label,
            color=color,
            cells=int(np.count_nonzero(values == code)),
            fraction=float(np.count_nonzero(values == code) / valid) if valid else None,
            sampled_area_m2=float(np.count_nonzero(values == code) * grid.resolution**2),
        )
        for code, label, color in CLASSES
    ]
    warnings = [
        "These are broad mapped land-cover classes for reference year 2021, not current conditions, "
        "habitat quality, native/invasive status, species identities or species percentage cover.",
        "Classification errors and mixed 10 m source pixels remain. Validate decisions with recent imagery "
        "and field observations; small areas and narrow exclusions may be unresolved.",
        "Nearest-neighbor sampling preserves class codes. Cell centers determine boundary inclusion. "
        "Percentages describe valid sampled cells; sampled areas are approximate, not cadastral measurements.",
        "WorldCover 2020 and 2021 use different algorithms. "
        "Differences between them are not proof of land-cover change.",
    ]
    if grid.resolution > 10.001:
        warnings.append(
            f"Sampled on a {grid.resolution:.1f} m grid. Coarser sampling can miss small classes; "
            "class percentages are sampled proportions, not exact source-pixel area totals."
        )
    if not total:
        warnings.append(
            "No analysis cell centers fall inside this boundary; no reliable cover summary is available."
        )
    elif valid < total:
        warnings.append(
            "Some boundary cells have no recognized source class. Percentages exclude those missing cells."
        )
    if unknown:
        warnings.append(f"Excluded {unknown} sampled cells with unrecognized source codes.")
    metadata = RasterMetadata(
        algorithm="worldcover-v1",
        method="ESA WorldCover 2021 v200 categorical nearest-neighbor sampling "
        "onto a bounded local metric grid; polygon/hole cell-center mask; "
        "counts and fractions of valid sampled classes.",
        bounds=list(grid.bounds),
        crs=grid.crs.to_wkt(),
        resolution_m=grid.resolution,
        width=grid.width,
        height=grid.height,
        boundary_cells=total,
        valid_cells=valid,
        coverage_fraction=valid / total if total else None,
        sampled_area_m2=valid * grid.resolution**2,
        bands=[
            RasterBand(
                index=1,
                name="Land cover (2021)",
                unit="class code",
                valid_cells=valid,
                minimum=None,
                maximum=None,
                mean=None,
                standard_deviation=None,
                percentiles={},
                histogram_edges=[],
                histogram_counts=[],
                palette="categorical",
                classes=classes,
            )
        ],
        sources=sources,
        warnings=warnings,
        downloaded_bytes=budget.downloaded_bytes,
    )
    return TerrainResult(encode(values, grid), metadata)
