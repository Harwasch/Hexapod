"""Show once, everywhere: how alike every cell of a Land is to one cell, by seasonal signature.

A stored `sentinel-2-signature` raster (`app.analysis.signature`) holds, per dated window,
green, red, NIR and SWIR reflectance and NDVI on a local metric grid. Only cells valid in
every band of every window are compared. Each band is standardized by the mean and spread its
metadata stores for those common cells (so a comparison can be reproduced from the stored
raster alone), each cell's vector is scaled to unit length, and two cells are as alike as the
cosine of their vectors: 1 the same signature, 0 unrelated, below 0 opposite. Alike is
relative to the variety on this land, and a similar signature is not the same species, cover
or habitat.

Scores are quantized once, `round(max(0, cosine) * 250)`, and every threshold compares the
quantized score (`raw_threshold`), so the console's slider, its tally and the areas made here
light the same cells.

`alike_patches` turns the alike cells into areas the way the Fog's scout patches are made
(`app.analysis.knowledge.mask_polygons`): cell squares merged, closed by half a cell (the
staircase rounded, single-cell gaps and diagonal steps bridged), simplified, clipped to the
boundary, with the protected ground cut out afterwards so it is always a hole.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import Geod
from rasterio.features import geometry_mask, rasterize, shapes
from shapely import set_precision
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.schemas.land_rasters import SignatureFeature

#: Quantized score of a cosine of 1; 255 is "no score here".
SCORE_SCALE = 250
NO_SCORE = 255
FEATURE_KEYS = ("green", "red", "nir", "swir16", "ndvi")
METHOD = (
    "Per band, z-score over the cells valid in every window (the raster's stored mean and "
    "standard deviation); unit-length vectors; cosine against the tapped cell; quantized as "
    "round(max(0, cosine) x 250)."
)
GEOD = Geod(ellps="WGS84")
#: Footprints are stored to 8 decimal degrees (about a millimetre).
FOOTPRINT_DECIMALS = 8


def resolution_label(resolution_m: float) -> str:
    """The grid's spacing as the console writes it: "10 m", or "11.9 m" for a grid coarsened
    to fit a large land (`make_grid` keeps at most 506 cells across)."""
    rounded = round(resolution_m, 1)
    return f"{rounded:.0f} m" if rounded.is_integer() else f"{rounded:.1f} m"


def interpretation(resolution_m: float) -> str:
    return (
        f"Similar seasonal satellite signature at {resolution_label(resolution_m)}. Not a "
        "species ID, cover estimate or habitat assessment: cells alike here reflected light "
        "alike on these dates."
    )


def raw_threshold(threshold: float) -> int:
    """The smallest quantized score at or above `threshold` (0..1): the console's rule too."""
    return max(0, min(SCORE_SCALE, math.ceil(threshold * SCORE_SCALE - 1e-9)))


@dataclass(frozen=True)
class Stack:
    """The comparable cells of one signature raster, standardized and unit length."""

    #: (cells, used features) unit vectors.
    unit: NDArray[np.float32]
    #: Each cell's vector length before scaling (to give its values back).
    norm: NDArray[np.float32]
    #: Flat index (row * width + column) of each compared cell on the analysis grid.
    cells: NDArray[np.int64]
    shape: tuple[int, int]
    transform: Affine
    crs: Any
    means: NDArray[np.float64]
    deviations: NDArray[np.float64]
    #: Features with any spread; a band that is the same everywhere cannot tell cells apart.
    used: NDArray[np.bool_]
    features: tuple[SignatureFeature, ...]

    @property
    def nbytes(self) -> int:
        return int(self.unit.nbytes + self.norm.nbytes + self.cells.nbytes)

    def index_of(self, row: int, column: int) -> int | None:
        """Position of an analysis cell among the compared cells; None when it has none."""
        flat = row * self.shape[1] + column
        position = int(np.searchsorted(self.cells, flat))
        if position < self.cells.size and int(self.cells[position]) == flat:
            return position
        return None

    def values(self, position: int) -> NDArray[np.float64]:
        """The compared cell's stored values, feature by feature, back from its vector."""
        values = self.means.copy()
        values[self.used] += (
            self.unit[position].astype(np.float64) * float(self.norm[position])
        ) * self.deviations[self.used]
        return values


def build_stack(
    data: NDArray[Any],
    features: Sequence[SignatureFeature],
    transform: Affine,
    crs: Any,
    nodata: float | None,
) -> Stack:
    """`data` is the raster's bands (bands, rows, columns); `features` names each band."""
    ordered = tuple(sorted(features, key=lambda feature: feature.band))
    stack = np.stack([data[feature.band - 1] for feature in ordered]).astype(np.float32)
    if nodata is not None and not math.isnan(nodata):
        stack[stack == np.float32(nodata)] = np.nan
    valid = np.logical_and.reduce(np.isfinite(stack), axis=0)
    cells = np.flatnonzero(valid).astype(np.int64)
    values = stack.reshape(len(ordered), -1)[:, cells].T.astype(np.float64)
    means = np.zeros(len(ordered), dtype=np.float64)
    deviations = np.zeros(len(ordered), dtype=np.float64)
    for index, feature in enumerate(ordered):
        mean, deviation = feature.common_mean, feature.common_standard_deviation
        # A raster stored without them (none of its cells common then) is measured here.
        if mean is None or deviation is None:
            column = values[:, index]
            mean = float(column.mean()) if column.size else 0.0
            deviation = float(column.std()) if column.size else 0.0
        means[index], deviations[index] = mean, deviation
    used = deviations > 1e-9
    standardized = (values[:, used] - means[used]) / deviations[used]
    norm = np.linalg.norm(standardized, axis=1)
    # A cell exactly at the land's mean in every band has no direction to compare.
    keep = norm > 1e-9
    unit = (standardized[keep] / norm[keep, None]).astype(np.float32)
    return Stack(
        unit=np.ascontiguousarray(unit),
        norm=norm[keep].astype(np.float32),
        cells=cells[keep],
        shape=(int(data.shape[1]), int(data.shape[2])),
        transform=transform,
        crs=crs,
        means=means,
        deviations=deviations,
        used=used,
        features=ordered,
    )


def cosine(stack: Stack, position: int) -> NDArray[np.float32]:
    """Every compared cell's cosine to the one at `position`, on the analysis grid; NaN for
    cells without a full signature."""
    scores = np.full(stack.shape[0] * stack.shape[1], np.nan, dtype=np.float32)
    scores[stack.cells] = stack.unit @ stack.unit[position]
    return scores.reshape(stack.shape)


def quantize(scores: NDArray[np.float32]) -> NDArray[np.uint8]:
    """`round(max(0, cosine) * 250)`, 255 where there is no score."""
    finite = np.isfinite(scores)
    quantized = np.full(scores.shape, NO_SCORE, dtype=np.uint8)
    quantized[finite] = np.rint(np.clip(scores[finite], 0.0, 1.0) * SCORE_SCALE).astype(np.uint8)
    return quantized


def polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """The polygons of any geometry (a collection's lines and points are not area)."""
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [polygon for part in geometry.geoms for polygon in polygon_parts(part)]
    return []


def geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Area on the ellipsoid of a lon/lat geometry."""
    return sum(
        abs(GEOD.geometry_area_perimeter(orient(part))[0]) for part in polygon_parts(geometry)
    )


def protected_geometry(rectangles: Sequence[Sequence[float]]) -> BaseGeometry:
    """The protected ground (lon/lat), from west/south/east/north rectangles."""
    boxes = [
        box(min(w, e), min(s, n), max(w, e), max(s, n))
        for w, s, e, n in rectangles
        if w != e and s != n
    ]
    return unary_union(boxes) if boxes else Polygon()


def centres_in(
    geometry: BaseGeometry, shape_: tuple[int, int], transform: Affine
) -> NDArray[np.bool_]:
    """Analysis cells whose centre lies in a geometry already in the grid's CRS."""
    if geometry.is_empty:
        return np.zeros(shape_, dtype=bool)
    mask: NDArray[np.bool_] = geometry_mask(
        [mapping(geometry)], out_shape=shape_, transform=transform, invert=True
    )
    return mask


@dataclass(frozen=True)
class Patch:
    #: Lon/lat outline, holes where protected ground or unlike cells are.
    footprint: Polygon
    area_m2: float
    cells: int
    mean_similarity: float


def alike_patches(
    alike: NDArray[np.bool_],
    scores: NDArray[np.float32],
    transform: Affine,
    resolution_m: float,
    boundary: BaseGeometry,
    protected: BaseGeometry,
    to_geographic: Any,
) -> list[Patch]:
    """The alike cells (already without protected ones) as smoothed patches, largest first.

    `boundary` and `protected` are in the grid's metric CRS; `to_geographic` maps it to lon/lat
    (a pyproj `Transformer.transform`). A cell counts in the patch its centre falls in.
    """
    if not alike.any():
        return []
    squares = [
        shape(geometry)
        for geometry, value in shapes(
            alike.astype(np.uint8), mask=alike, transform=transform, connectivity=4
        )
        if value == 1
    ]
    half = resolution_m / 2
    merged = unary_union(squares)
    smoothed = merged.buffer(half, quad_segs=4).buffer(-half, quad_segs=4).simplify(half / 2)
    cut = make_valid(smoothed)
    if not protected.is_empty:
        cut = make_valid(cut.difference(protected))
    cut = make_valid(cut.intersection(boundary))
    parts = [part for part in polygon_parts(cut) if part.area > 0]
    if not parts:
        return []
    labels = rasterize(
        [(mapping(part), index + 1) for index, part in enumerate(parts)],
        out_shape=alike.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    counted = labels[alike]
    cells = np.bincount(counted, minlength=len(parts) + 1)
    sums = np.bincount(counted, weights=scores[alike].astype(np.float64), minlength=len(parts) + 1)
    patches: list[Patch] = []
    for index, part in enumerate(parts):
        count = int(cells[index + 1])
        mean = float(sums[index + 1] / count) if count else 0.0
        # Reprojected, a part pinched to a point comes back as two polygons: its cells are
        # shared between them by area, never counted again for each.
        pieces = polygon_parts(make_valid(transform_geometry(to_geographic, part)))
        areas = [geodesic_area_m2(piece) for piece in pieces]
        for piece, area, share in zip(pieces, areas, shares(count, areas), strict=True):
            patches.append(
                Patch(footprint=orient(piece), area_m2=area, cells=share, mean_similarity=mean)
            )
    return sorted(patches, key=lambda patch: patch.area_m2, reverse=True)


def shares(total: int, weights: Sequence[float]) -> list[int]:
    """`total` split in proportion to `weights`, in whole numbers that add up to it (the
    largest remainders round up)."""
    if not weights:
        return []
    whole = sum(weights)
    if whole <= 0:
        return [total, *[0] * (len(weights) - 1)]
    exact = [total * weight / whole for weight in weights]
    split = [math.floor(value) for value in exact]
    by_remainder = sorted(range(len(split)), key=lambda i: exact[i] - split[i], reverse=True)
    for i in by_remainder[: total - sum(split)]:
        split[i] += 1
    return split


def footprint_mapping(geometry: Polygon | MultiPolygon) -> dict[str, Any]:
    """GeoJSON for a Footprint: two dimensions, rings closed, millimetre-scale rounding."""

    def ring(coordinates: Any) -> list[list[float]]:
        points = [
            [round(point[0], FOOTPRINT_DECIMALS), round(point[1], FOOTPRINT_DECIMALS)]
            for point in coordinates
        ]
        if points[0] != points[-1]:
            points.append(points[0])
        return points

    def polygon(item: Polygon) -> list[list[list[float]]]:
        return [ring(item.exterior.coords), *[ring(hole.coords) for hole in item.interiors]]

    if isinstance(geometry, Polygon):
        return {"type": "Polygon", "coordinates": polygon(geometry)}
    return {"type": "MultiPolygon", "coordinates": [polygon(part) for part in geometry.geoms]}


def snapped_outline(geometry: BaseGeometry) -> Polygon | MultiPolygon | None:
    """An outline snapped to the footprint's rounding with a valid result, for one that rounding
    alone breaks (a thin neck pinched shut, two parts laid edge to edge): parts that touch
    merge, and what is narrower than the rounding goes. None when no area is left."""
    try:
        snapped = make_valid(set_precision(make_valid(geometry), 10.0**-FOOTPRINT_DECIMALS))
    except GEOSException:
        return None
    parts = [part for part in polygon_parts(snapped) if part.area > 0]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)
