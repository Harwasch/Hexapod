"""Reproducible, bounded surface-elevation and slope analysis over a land boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
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

from app.analysis.raster_grid import make_grid
from app.analysis.raster_io import RasterOpener, ReadBudget
from app.research.providers.base import SourceSpec, fetch_json
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterBand, RasterMetadata, RasterRequest, RasterSource

SPEC = SourceSpec(
    id="cop-dem-glo-30",
    name="Copernicus GLO-30 surface elevation",
    domain="physical",
    coverage="Global land; source voids and local uncertainty remain",
    resolution="Approximately 30 m source grid; not surveyed bare-earth terrain",
    license="Copernicus DEM license (free access and use, subject to attribution and license terms)",
    attribution="Copernicus DEM GLO-30, ESA / Airbus; catalog provided by Element 84 Earth Search",
    documentation_url="https://registry.opendata.aws/copernicus-dem/",
    endpoint="https://earth-search.aws.element84.com/v1/search",
)
LICENSE_URL = (
    "https://spacedata.copernicus.eu/documents/20126/0/CSCDA_ESA_Mission-specific+Annex.pdf"
)


@dataclass(frozen=True)
class TerrainResult:
    data: bytes
    metadata: RasterMetadata


def band_statistics(data: NDArray[np.float32], index: int, name: str, unit: str) -> RasterBand:
    valid = data[np.isfinite(data)].astype(np.float64)
    counts: list[int] = []
    edges: list[float] = []
    if valid.size:
        histogram, bins = np.histogram(valid, bins=20)
        counts = [int(value) for value in histogram]
        edges = [float(value) for value in bins]
    return RasterBand(
        index=index,
        name=name,
        unit=unit,
        valid_cells=int(valid.size),
        minimum=float(valid.min()) if valid.size else None,
        maximum=float(valid.max()) if valid.size else None,
        mean=float(valid.mean()) if valid.size else None,
        standard_deviation=float(valid.std()) if valid.size else None,
        percentiles={str(p): float(np.percentile(valid, p)) for p in (5, 25, 50, 75, 95)}
        if valid.size
        else {},
        histogram_edges=edges,
        histogram_counts=counts,
        palette="viridis" if index == 1 else "magma",
    )


def derive(
    surface: NDArray[np.float32], inside: NDArray[np.bool_], resolution: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate derivatives before applying the boundary, including holes, to avoid edge artifacts."""
    dy, dx = np.gradient(surface, resolution, resolution)
    slopes = np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)
    elevation = np.where(inside, surface, np.nan).astype(np.float32)
    slopes[~inside | ~np.isfinite(surface)] = np.nan
    return elevation, slopes


def encode_cog(bands: list[NDArray[np.float32]], crs: str, affine: Any) -> bytes:
    with MemoryFile() as source, MemoryFile() as output:
        with source.open(
            driver="GTiff",
            height=bands[0].shape[0],
            width=bands[0].shape[1],
            count=len(bands),
            dtype="float32",
            crs=crs,
            transform=affine,
            nodata=-9999.0,
        ) as dataset:
            for index, band in enumerate(bands, 1):
                dataset.write(np.where(np.isfinite(band), band, -9999).astype(np.float32), index)
                dataset.set_band_description(
                    index, "Surface elevation" if index == 1 else "Surface slope"
                )
                dataset.set_band_unit(index, "m" if index == 1 else "degrees")
            dataset.update_tags(
                algorithm="terrain-v1",
                source="Copernicus DEM GLO-30",
                vertical_reference="EGM2008 (elevation band)",
                interpretation="Digital surface model; not surveyed bare-earth terrain.",
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
        raise ValueError("The derived raster exceeds the 16 MiB output limit.")
    return data


def analyze(boundary: Footprint, request: RasterRequest, client: httpx.Client) -> TerrainResult:
    grid = make_grid(boundary, request)
    west, south, east, north = grid.bounds
    metric_crs, affine, resolution = grid.crs, grid.affine, grid.resolution
    width, height, inside = grid.width, grid.height, grid.inside
    # Search a padded region because slope needs neighboring samples outside the boundary.
    padding = max(0.002, resolution * 3 / 111000 / max(0.05, math.cos(math.radians(grid.latitude))))
    bbox = [
        max(-180, west - padding),
        max(-90, south - padding),
        min(180, east + padding),
        min(90, north + padding),
    ]
    catalog, _ = fetch_json(
        client,
        SPEC,
        {"collections": request.dataset, "bbox": ",".join(str(v) for v in bbox), "limit": 9},
    )
    items = catalog.get("features", [])
    if not isinstance(items, list) or not items:
        raise ValueError("The terrain catalog returned no coverage for this area.")
    matched = catalog.get("numberMatched", catalog.get("context", {}).get("matched"))
    if len(items) > 8 or (isinstance(matched, int) and matched > 8):
        raise ValueError(
            "This area needs more than eight source tiles. Analyze smaller sections to preserve full coverage."
        )
    # Earth Search advertises a next link even after returning every matching item.
    # Its explicit match count establishes completeness without following arbitrary URLs.
    if any(link.get("rel") == "next" for link in catalog.get("links", [])) and matched != len(
        items
    ):
        raise ValueError(
            "The catalog did not confirm complete terrain coverage. Try this analysis again."
        )
    surface = np.full((height, width), np.nan, dtype=np.float32)
    sources = []
    budget = ReadBudget(max_seconds=60)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=64 * 1024 * 1024,
        AWS_NO_SIGN_REQUEST="YES",
    ):
        for item in sorted(items, key=lambda value: value["id"]):
            href = item.get("assets", {}).get("data", {}).get("href", "")
            prefix = "s3://copernicus-dem-30m/"
            if not href.startswith(prefix):
                raise ValueError("The terrain catalog returned an unsupported asset location.")
            url = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/" + href[len(prefix) :]
            opener = RasterOpener(client, url, budget)
            with rasterio.open(url, opener=opener, driver="GTiff") as source:
                if source.count != 1 or source.crs is None:
                    raise ValueError(
                        "The terrain source does not have its expected band/coordinate system."
                    )
                with WarpedVRT(
                    source,
                    crs=metric_crs.to_wkt(),
                    transform=affine,
                    width=width,
                    height=height,
                    dtype="float32",
                    nodata=float("nan"),
                    resampling=Resampling.bilinear,
                ) as warped:
                    values = warped.read(1, masked=True).filled(np.nan).astype(np.float32)
                np.copyto(surface, values, where=~np.isfinite(surface) & np.isfinite(values))
            version = opener.files[-1]
            sources.append(
                RasterSource(
                    id=item["id"],
                    url=HttpUrl(url),
                    catalog_url=HttpUrl(
                        f"https://earth-search.aws.element84.com/v1/collections/{request.dataset}/items/{item['id']}"
                    ),
                    etag=version.etag,
                    last_modified=version.last_modified,
                    catalog_datetime=item.get("properties", {}).get("datetime"),
                    observation_period="Source acquisition dates vary; the catalog date is not an acquisition date.",
                    license=SPEC.license,
                    license_url=HttpUrl(LICENSE_URL),
                    attribution=SPEC.attribution,
                )
            )
    elevation, slopes = derive(surface, inside, resolution)
    count, valid = int(np.count_nonzero(inside)), int(np.count_nonzero(np.isfinite(elevation)))
    warnings = [
        "Elevation uses the source EGM2008 vertical reference, not local survey datums.",
        "This is a digital surface model. Buildings and vegetation may affect elevation and slope; "
        "it is not a ground survey or a geotechnical assessment.",
        "Values are sampled on a local metric grid using bilinear interpolation. Cell centers determine "
        "boundary inclusion; small features and narrow exclusions may be unresolved.",
        "Slope is the gradient at the analysis resolution. Coarser sampling can smooth steep features; "
        "neighboring no-data cells reduce slope coverage.",
    ]
    if resolution > request.resolution_m * 1.001:
        warnings.append(
            f"The analysis grid was coarsened to {resolution:.1f} m to fit the selected extent "
            f"within {request.max_dimension} pixels."
        )
    if not count:
        warnings.append(
            "No analysis cell centers fall inside this boundary. "
            "The source cannot resolve a reliable area summary at this scale."
        )
    elif valid < count:
        warnings.append(
            "Some boundary cells have no source data. Summary statistics describe valid sampled cells only."
        )
    method = (
        "Copernicus GLO-30 bilinear resampling onto a local azimuthal-equidistant metric grid; "
        "cell-center polygon/hole mask; central-difference surface slope; unweighted valid-cell statistics. "
        "Approximate sampled area is cell count multiplied by squared grid spacing, not a cadastral area measurement."
    )
    metadata = RasterMetadata(
        algorithm="terrain-v1",
        method=method,
        bounds=[west, south, east, north],
        crs=metric_crs.to_wkt(),
        resolution_m=resolution,
        width=width,
        height=height,
        boundary_cells=count,
        valid_cells=valid,
        coverage_fraction=valid / count if count else None,
        sampled_area_m2=valid * resolution * resolution,
        bands=[
            band_statistics(elevation, 1, "Surface elevation", "m"),
            band_statistics(slopes, 2, "Surface slope", "degrees"),
        ],
        sources=sources,
        warnings=warnings,
        downloaded_bytes=budget.downloaded_bytes,
    )
    return TerrainResult(encode_cog([elevation, slopes], metadata.crs, affine), metadata)
