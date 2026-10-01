"""From where each part of a splat was seen, so a viewer can stop drawing it from elsewhere.

A splat is trained to match photos taken from somewhere. Seen from there it is right; seen
from a side no photo saw, it is whatever the optimiser left: the back of a tree that was only
ever seen from inside a clearing, sky-coloured gaussians placed in a canopy to explain the
sky between its branches, the dark undersides of foliage. A capture walked *inside* a site
(Fort Clatsop: 22.6M gaussians, the cameras at eye height among the cabins, 20-40 m conifers
all round) looks right from where it was walked and like fog from the globe's overview,
because the overview looks at all of that from above and outside.

This module gives every cell of a coarse grid over the scan a **cone of directions it was
seen from**: an axis (from the observers towards the cell) and a half-angle. A viewer fades
a splat out as its viewing direction leaves its cell's cone (`FADE_DEG` past the edge), and
draws it unchanged inside it -- so the view from where the capture was taken is exactly what
it always was, and the views it never had stop showing what was never measured.

**The observers, without cameras.** A trained splat that arrives as a file (`splat-ingest`)
carries no camera poses. It still says where they were: a gaussian is about as large as the
pixel it was fitted to, a pixel covers `depth / focal`, so detail is finest nearest the
cameras and coarsens with distance from them. On Fort Clatsop the 25th-percentile middle
axis is 0.4 cm on the ground among the cabins, 7-11 cm in the canopy 20-50 m overhead and
10-27 cm past 50 m out. So the **finest-detail region stands in for the observers**:

1. per cell, the 25th percentile of the log middle axis of the gaussians centred in it (the
   middle axis: the smallest is a flat gaussian's thickness and the largest a needle's
   length, neither the pixel it fitted). Cells with fewer than `MIN_CELL_GAUSSIANS` are not
   judged;
2. the scan's own finest detail: the gaussian-weighted `FINE_PERCENTILE`th percentile of
   those cell values; the **fine region** is every judged cell within `FINE_FACTOR` of it;
3. each cell's cone: the unit vectors from the fine region's cells (weighted by their
   gaussians) to it; the axis is their weighted mean, the half-angle the angle from the axis
   that holds `CONE_SHARE` of their weight. A cell within `OMNI_CELLS` of the fine region,
   or one the fine region surrounds (the mean vector shorter than `OMNI_RESULTANT` of the
   weight), is **seen from everywhere** and is never faded;
4. and so is any cell whose detail -- the finest within `ENVELOPE_CELLS` -- is within
   `CONTRAST` of the scan's finest, or unknown: only a part much coarser than the closest
   surfaces is far enough from the observers for the fine region to stand in for them, and
   nothing is faded without evidence.

Nothing in it is about forests, walks or heights. An orbit's subject is finest and the
background behind it coarse: the subject is seen from everywhere and the background from the
subject's side. A drone grid's detail is even across the site: nearly every cell is near the
fine region, and nothing fades. On Fort Clatsop: the fine region is the clearing, ground to
5.5 m; 86 % of the gaussians are seen from everywhere; the canopy is seen from below and the
far trees from inside the clearing. The overview then shows the fort in its clearing and the
inside views are unchanged (measured with an offline renderer over the published tiles).

**With cameras** the observers are known, and `observers` takes their centres in the same
frame in place of the fine region; nothing else changes.

**Format `hexapod.viewcones` v1.** `viewcones.bin` beside `tileset.json`, gzip (mtime 0):
one RGBA8 texel per cell of a `dims` grid, x fastest, then y, then z. A point p is in cell
`floor((p - origin) / cell)` per axis, in the tileset's local east/north/up metres (the
frame the SPZ positions are in). `r`, `g`: the axis, octahedrally encoded (`encode_axis`);
`b`: the half-angle, `round(deg * 254 / 180)`; `b == 255`: seen from everywhere. `a`: 255, reserved. A
point outside the grid takes the nearest edge cell (clamped indices). The root tile's
`extras.viewCones` carries `origin`, `cell`, `dims`, `fadeDeg` and what was measured.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "CONE_SHARE",
    "FADE_DEG",
    "FINE_FACTOR",
    "FINE_PERCENTILE",
    "FORMAT",
    "MIN_CELL_GAUSSIANS",
    "OMNI",
    "OMNI_CELLS",
    "OMNI_RESULTANT",
    "URI",
    "VERSION",
    "ConeGrid",
    "build_view_cones",
    "cone_grid",
    "cone_grid_from_chunks",
    "decode_axis",
    "decode_view_cones",
    "encode_axis",
    "lookup",
    "view_cones_extras",
    "visibility",
    "write_view_cones",
]

FORMAT = "hexapod.viewcones"
VERSION = 1
URI = "viewcones.bin"

#: The grid's longest axis, in cells. Fort Clatsop's ~150 m is 1.6 m cells: a cone changes
#: slowly with position (it is the observers' direction), so this is fine enough, and the
#: whole grid is a few hundred kilobytes of texels however large the scan.
MAX_CELLS = 96
#: ...and no finer than this, so a table-top scan does not get millimetre cells.
MIN_CELL_M = 0.02
#: The grid spans every gaussian the tiles keep (the packer's floater filter has already
#: bounded them): the far, coarse edges of a scan are exactly where cones matter most, so
#: no percentile trims them. A point outside the grid takes its nearest edge cell's cone.

#: A cell needs this many gaussians before its detail is judged.
MIN_CELL_GAUSSIANS = 8
#: The scan's finest detail: this percentile, gaussian-weighted, of the cells' detail.
FINE_PERCENTILE = 20.0
#: The fine region: cells whose detail is within this factor of the finest.
FINE_FACTOR = 2.5
#: Within this many cells of the fine region a cell is seen from everywhere.
OMNI_CELLS = 2.5
#: ...and so is a cell whose detail (the finest within `ENVELOPE_CELLS` of it) is within
#: this factor of the scan's finest: distance from the cameras coarsens detail by tens of
#: times (Fort Clatsop's canopy 20x, its far edge 25-50x), what a surface is made of by a
#: few. Only what is much coarser than the closest surfaces is far enough from the observers
#: for the finest-detail region to stand in for them -- the ground around an orbited tree
#: is as fine as the tree, and the orbit saw it from outside, not from the tree.
CONTRAST = 4.0
ENVELOPE_CELLS = 2
#: A mean observer direction shorter than this share of the weight: surrounded, everywhere.
OMNI_RESULTANT = 0.2
#: The half-angle holds this share of the observers' weight.
CONE_SHARE = 0.9
#: Past the cone's edge a viewer fades a splat out over this many degrees.
FADE_DEG = 20.0
#: The fine region is reduced to at most this many weighted points (coarser cells merged)
#: before every cell's cone is taken against it.
MAX_OBSERVERS = 2048

#: Detail is histogrammed per cell in log-metre bins this wide, from `LOG_LOW` up.
LOG_BIN = 0.125
LOG_LOW = -12.0
LOG_BINS = 128

OMNI = 255

RULE = (
    f"observers: cells within {FINE_FACTOR:g}x of the scan's finest detail (the "
    f"gaussian-weighted {FINE_PERCENTILE:g}th percentile of each cell's 25th-percentile "
    f"middle axis, cells of at least {MIN_CELL_GAUSSIANS}); a cell's cone holds "
    f"{CONE_SHARE:.0%} of their weight; seen from everywhere within {OMNI_CELLS:g} cells of "
    f"them or where they surround it"
)


@dataclass
class ConeGrid:
    """A cone per cell, and what it was measured from."""

    origin: tuple[float, float, float]
    cell: float
    dims: tuple[int, int, int]
    #: (nz * ny * nx, 4) uint8 texels, x fastest.
    texels: np.ndarray
    stats: dict[str, object] = field(default_factory=dict)
    #: Where the cones point from: the observer points and their weights (not written to
    #: the file; the fill teacher places its cameras there).
    observers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    observer_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))


Chunk = tuple[np.ndarray, np.ndarray]  # (xyz float64 (n, 3), log middle axis (n,))


def encode_axis(axis: np.ndarray) -> np.ndarray:
    """Unit vectors (n, 3) to octahedral (n, 2) uint8 -- 1.4 degrees worst case."""
    v = np.asarray(axis, np.float64)
    v = v / np.maximum(np.abs(v).sum(axis=1, keepdims=True), 1e-12)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    fx = np.where(z >= 0, x, (1 - np.abs(y)) * np.where(x >= 0, 1.0, -1.0))
    fy = np.where(z >= 0, y, (1 - np.abs(x)) * np.where(y >= 0, 1.0, -1.0))
    return np.stack(
        [np.round((fx * 0.5 + 0.5) * 255), np.round((fy * 0.5 + 0.5) * 255)], axis=1
    ).astype(np.uint8)


def decode_axis(rg: np.ndarray) -> np.ndarray:
    """The inverse of `encode_axis`, as the shader does it."""
    f = np.asarray(rg, np.float64) / 255.0 * 2.0 - 1.0
    x, y = f[:, 0], f[:, 1]
    z = 1.0 - np.abs(x) - np.abs(y)
    t = np.clip(-z, 0.0, None)
    x = x + np.where(x >= 0, -t, t)
    y = y + np.where(y >= 0, -t, t)
    v = np.stack([x, y, z], axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lookup(
    texels: np.ndarray,
    origin: tuple[float, float, float],
    cell: float,
    dims: tuple[int, int, int],
    positions: np.ndarray,
) -> np.ndarray:
    """Each position's cell's texel; a position outside the grid takes the nearest edge cell."""
    index = np.floor((np.asarray(positions, np.float64) - np.asarray(origin)) / cell)
    index = np.clip(index, 0, np.asarray(dims) - 1).astype(np.int64)
    linear = (index[:, 2] * dims[1] + index[:, 1]) * dims[0] + index[:, 0]
    return np.asarray(texels)[linear]


def visibility(texels: np.ndarray, view_dirs: np.ndarray, fade_deg: float = FADE_DEG) -> np.ndarray:
    """The weight a viewer gives a splat: 1 inside its cone, 0 past it by `fade_deg`.

    `view_dirs` are unit vectors from the viewer to each splat. The reference for the
    shader's arithmetic (apps/web src/cesium/splatViewCones.ts).
    """
    texels = np.asarray(texels)
    out = np.ones(texels.shape[0])
    cone = texels[:, 2] != OMNI
    if cone.any():
        axis = decode_axis(texels[cone, :2])
        half = texels[cone, 2].astype(np.float64) * 180.0 / 254.0
        cos = np.clip((axis * view_dirs[cone]).sum(axis=1), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        out[cone] = np.clip((half + fade_deg - angle) / fade_deg, 0.0, 1.0)
    return out


def _middle_axis(scales: np.ndarray) -> np.ndarray:
    """The log middle axis of log scales (n, 3)."""
    return np.sort(np.asarray(scales, np.float64), axis=1)[:, 1]


def cone_grid_from_chunks(
    first: Iterable[Chunk],
    second: Iterable[Chunk],
    count: int,
    observers: np.ndarray | None = None,
) -> ConeGrid:
    """The cone grid from two passes over the same `count` gaussians.

    `first` and `second` yield the same `(xyz, log middle axis)` windows in the same order
    (the first pass finds the extent, the second fills the grid), so a scan larger than
    memory is never held. `observers`, when known, replaces the fine region with those
    points (equal weight).
    """
    lo, hi = _bounds(first)
    if not np.all(np.isfinite(lo)):
        return ConeGrid((0.0, 0.0, 0.0), 1.0, (1, 1, 1), _omni(1), {"gaussians": 0})
    extent = float((hi - lo).max())
    cell = max(extent / MAX_CELLS, MIN_CELL_M)
    dims_arr = np.floor((hi - lo) / cell).astype(np.int64) + 1
    dims = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))
    origin = lo.astype(np.float64)

    cells, counts, detail = _histogram_pass(second, origin, cell, dims_arr)
    judged = counts >= MIN_CELL_GAUSSIANS
    total = int(counts.sum())
    centres = (np.stack(np.unravel_index(cells, dims[::-1]), axis=1)[:, ::-1] + 0.5) * cell + origin

    if observers is not None:
        points = np.asarray(observers, np.float64).reshape(-1, 3)
        weights = np.ones(points.shape[0])
        finest = float("nan")
        fine_cells = 0
        fine_share = float("nan")
        fine_mask = None
        # Known observers need no contrast guard: every cell gets its cone from them.
        finest_for_contrast = -np.inf
    elif judged.any():
        finest = _weighted_percentile(detail[judged], counts[judged], FINE_PERCENTILE)
        fine = judged & (detail <= finest + math.log(FINE_FACTOR))
        fine_mask = fine
        finest_for_contrast = finest
        points, weights = centres[fine], counts[fine].astype(np.float64)
        fine_cells = int(fine.sum())
        fine_share = float(counts[fine].sum() / max(total, 1))
    else:
        points = np.zeros((0, 3))
        weights = np.zeros(0)
        finest = float("nan")
        fine_cells = 0
        fine_share = 0.0
        fine_mask = None
        finest_for_contrast = np.inf
    points, weights = _reduce_observers(points, weights, cell)

    # Every cell, occupied or not: a merged parent's centre can land in an empty one, and a
    # viewer clamps a point outside the grid to the nearest edge cell. Cells near the fine
    # region, or not much coarser than it, are seen from everywhere; the rest get a cone.
    shape = (dims[2], dims[1], dims[0])
    detail_grid = np.full(int(np.prod(dims_arr)), np.inf)
    detail_grid[cells[judged]] = detail[judged]
    envelope = _min_filter(detail_grid.reshape(shape), ENVELOPE_CELLS).reshape(-1)
    # Only evidence fades: a cell with no judged detail within reach is left alone too.
    everywhere = ~np.isfinite(envelope) | (envelope < finest_for_contrast + math.log(CONTRAST))
    if observers is not None:
        # Known cameras are evidence for every cell, however sparse (an inferred layer is).
        everywhere[:] = False
    elif fine_mask is not None:
        near = np.zeros(int(np.prod(dims_arr)), bool)
        near[cells[fine_mask]] = True
        everywhere |= _dilate(near.reshape(shape), OMNI_CELLS).reshape(-1)
    candidates = np.nonzero(~everywhere)[0]
    texels = _omni(int(np.prod(dims_arr)))
    texels[candidates] = _cones(
        (np.stack(np.unravel_index(candidates, shape), axis=1)[:, ::-1] + 0.5) * cell + origin,
        points,
        weights,
    )
    # The share of gaussians a cone applies to, the number a reader wants to see.
    cone_of_cell = texels[cells, 2] != OMNI
    directional = float(counts[cone_of_cell].sum() / max(total, 1))
    stats: dict[str, object] = {
        "gaussians": total,
        "occupiedCells": int(cells.size),
        "observers": "cameras" if observers is not None else "finest-detail region",
        "observerPoints": int(points.shape[0]),
        "fineCells": fine_cells,
        "fineShare": round(fine_share, 4) if not math.isnan(fine_share) else None,
        "finestMiddleAxisM": round(math.exp(finest), 6) if not math.isnan(finest) else None,
        "directionalShare": round(directional, 4),
    }
    if points.shape[0]:
        stats["observerBox"] = {
            "min": [round(float(v), 3) for v in points.min(axis=0)],
            "max": [round(float(v), 3) for v in points.max(axis=0)],
        }
    return ConeGrid(
        (float(origin[0]), float(origin[1]), float(origin[2])),
        float(cell),
        dims,
        texels,
        stats,
        points,
        weights,
    )


def _omni(n: int) -> np.ndarray:
    texels = np.zeros((n, 4), np.uint8)
    texels[:, 0:2] = 128
    texels[:, 2] = OMNI
    texels[:, 3] = 255
    return texels


def _bounds(chunks: Iterable[Chunk]) -> tuple[np.ndarray, np.ndarray]:
    """The box around every position."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for xyz, _ in chunks:
        if xyz.shape[0]:
            lo = np.minimum(lo, xyz.min(axis=0))
            hi = np.maximum(hi, xyz.max(axis=0))
    return lo, hi


def _histogram_pass(
    chunks: Iterable[Chunk], origin: np.ndarray, cell: float, dims: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupied cells (sorted linear indices), each one's gaussian count and its detail.

    The detail is the 25th percentile of the cell's log middle axes, read off a histogram
    of `LOG_BIN`-wide bins held sparsely -- a (cell, bin) key and a count per non-empty
    bin, never a dense cells x bins array (118k cells would be 115 MB of it).
    """
    keys = np.zeros(0, np.int64)
    sums = np.zeros(0, np.int64)
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    waiting = 0
    for xyz, middle in chunks:
        index = np.floor((xyz - origin) / cell).astype(np.int64)
        inside = np.all((index >= 0) & (index < dims), axis=1)
        index = index[inside]
        if index.shape[0] == 0:
            continue
        linear = (index[:, 2] * dims[1] + index[:, 1]) * dims[0] + index[:, 0]
        bins = np.clip(
            np.floor((middle[inside] - LOG_LOW) / LOG_BIN).astype(np.int64), 0, LOG_BINS - 1
        )
        key, n = np.unique(linear * LOG_BINS + bins, return_counts=True)
        pending.append((key, n.astype(np.int64)))
        waiting += key.size
        if waiting > max(keys.size // 2, 1 << 20):
            keys, sums = _merge_keys([(keys, sums), *pending])
            pending, waiting = [], 0
    keys, sums = _merge_keys([(keys, sums), *pending])
    cell_of = keys // LOG_BINS
    starts = np.flatnonzero(np.r_[True, cell_of[1:] != cell_of[:-1]]) if keys.size else keys
    cells = cell_of[starts]
    counts = np.add.reduceat(sums, starts) if keys.size else sums
    # The first bin of each cell at which its running count reaches a quarter of the cell.
    running = np.cumsum(sums)
    before = np.r_[0, running[:-1]][starts]
    target = before + np.maximum(counts * 0.25, 1e-9)
    group = np.repeat(np.arange(cells.size), np.diff(np.r_[starts, keys.size]))
    reached = running >= target[group]
    first = np.full(cells.size, keys.size)
    np.minimum.at(first, group[reached], np.flatnonzero(reached))
    detail = LOG_LOW + ((keys[first] % LOG_BINS) + 0.5) * LOG_BIN
    return cells, counts, detail


def _merge_keys(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate([part[0] for part in parts])
    weights = np.concatenate([part[1] for part in parts])
    keys, inverse = np.unique(merged, return_inverse=True)
    return keys, np.bincount(inverse, weights=weights).astype(np.int64)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order].astype(np.float64))
    at = np.searchsorted(cumulative, cumulative[-1] * q / 100.0)
    return float(values[order][min(int(at), order.size - 1)])


def _reduce_observers(
    points: np.ndarray, weights: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """At most `MAX_OBSERVERS` weighted points: cells merged on coarser grids until they fit."""
    size = cell
    while points.shape[0] > MAX_OBSERVERS:
        size *= 2.0
        key = np.floor(points / size).astype(np.int64)
        _, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        total = np.bincount(inverse, weights=weights)
        merged = np.stack(
            [np.bincount(inverse, weights=points[:, k] * weights) for k in range(3)], axis=1
        )
        points, weights = merged / total[:, None], total
    return points, weights


def _offsets(radius: float) -> list[tuple[int, int, int]]:
    reach = math.floor(radius)
    return [
        (dz, dy, dx)
        for dz in range(-reach, reach + 1)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if dx * dx + dy * dy + dz * dz <= radius * radius
    ]


def _shifted(grid: np.ndarray, offset: tuple[int, int, int], fill: float | bool) -> np.ndarray:
    """`grid` moved by `offset` (z, y, x), `fill` where nothing moved in."""
    out = np.full_like(grid, fill)
    src = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(offset, grid.shape, strict=True))
    dst = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(offset, grid.shape, strict=True))
    out[dst] = grid[src]
    return out


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Every cell within `radius` cells of a set one."""
    out = mask.copy()
    for offset in _offsets(radius):
        out |= _shifted(mask, offset, False)
    return out


def _min_filter(grid: np.ndarray, radius: int) -> np.ndarray:
    """Each cell's minimum within a cube of `radius` cells (inf outside the grid)."""
    out = grid.copy()
    reach = range(-radius, radius + 1)
    for offset in [(z, y, x) for z in reach for y in reach for x in reach]:
        np.minimum(out, _shifted(grid, offset, np.inf), out=out)
    return out


def _cones(centres: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Texels for `centres` against the weighted observer `points`."""
    texels = _omni(centres.shape[0])
    if points.shape[0] == 0:
        return texels
    total = float(weights.sum())
    # (cells x observers) at a time: 2^18 pairs, ~25 MB of working arrays, whatever the scan.
    block = max(1, (1 << 18) // max(points.shape[0], 1))
    for start in range(0, centres.shape[0], block):
        c = centres[start : start + block]
        v = c[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(v, axis=2)
        unit = v / np.maximum(dist, 1e-12)[..., None]
        mean = (unit * weights[None, :, None]).sum(axis=1)
        length = np.linalg.norm(mean, axis=1)
        axis = mean / np.maximum(length, 1e-12)[:, None]
        angle = np.degrees(np.arccos(np.clip((unit * axis[:, None, :]).sum(axis=2), -1.0, 1.0)))
        order = np.argsort(angle, axis=1, kind="stable")
        sorted_angle = np.take_along_axis(angle, order, axis=1)
        share = np.cumsum(weights[order], axis=1) / total
        half = sorted_angle[np.arange(c.shape[0]), np.argmax(share >= CONE_SHARE, axis=1)]
        everywhere = length / total < OMNI_RESULTANT
        rows = texels[start : start + block]
        cone = ~everywhere
        rows[cone, 0:2] = encode_axis(axis[cone])
        rows[cone, 2] = np.clip(np.round(half[cone] * 254.0 / 180.0), 0, 254).astype(np.uint8)
    return texels


def cone_grid(layout: object, keep: np.ndarray, observers: np.ndarray | None = None) -> ConeGrid:
    """The cone grid of the rows `keep` selects in a 3DGS PLY (`splat_tiles.ply_layout`)."""
    from splat_tiles import iter_ply_rows

    columns = ("x", "y", "z", "scale_0", "scale_1", "scale_2")

    def chunks() -> Iterator[Chunk]:
        for start, chunk in iter_ply_rows(layout, columns):  # type: ignore[arg-type]
            kept = keep[start : start + chunk["x"].size]
            xyz = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1)[kept]
            scales = np.stack([chunk["scale_0"], chunk["scale_1"], chunk["scale_2"]], axis=1)
            yield xyz.astype(np.float64), _middle_axis(scales[kept])

    return cone_grid_from_chunks(chunks(), chunks(), int(keep.sum()), observers)


def cone_grid_from_tileset(tileset: Path, observers: np.ndarray | None = None) -> ConeGrid:
    """The cone grid of a packed tileset's leaves -- for a scan whose PLY is gone."""
    from rig_tiles import glb_spz
    from splat_tiles import unpack_spz

    document = json.loads(tileset.read_text(encoding="utf-8"))
    leaves: list[str] = []
    count = 0
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
            count += int(tile.get("extras", {}).get("gaussians", 0))
    leaves.sort()

    def chunks() -> Iterator[Chunk]:
        for uri in leaves:
            data = unpack_spz(glb_spz(tileset.parent / uri))
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
            scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
            yield xyz, _middle_axis(scales)

    return cone_grid_from_chunks(chunks(), chunks(), count, observers)


def encode_view_cones(grid: ConeGrid) -> bytes:
    """`viewcones.bin`: the texels, gzip (mtime 0, so the same grid is the same bytes)."""
    return gzip.compress(np.ascontiguousarray(grid.texels).tobytes(), compresslevel=9, mtime=0)


def decode_view_cones(blob: bytes, dims: tuple[int, int, int]) -> np.ndarray:
    """`viewcones.bin` back to (n, 4) texels, checked against `dims`."""
    raw = gzip.decompress(blob)
    expected = int(np.prod(dims)) * 4
    if len(raw) != expected:
        raise ValueError(f"view cones are {len(raw)} bytes; a {dims} grid is {expected}")
    return np.frombuffer(raw, np.uint8).reshape(-1, 4)


def view_cones_extras(grid: ConeGrid) -> dict[str, object]:
    """The root tile's `extras.viewCones`: how to read `viewcones.bin`."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "uri": URI,
        "origin": [round(v, 6) for v in grid.origin],
        "cell": round(grid.cell, 6),
        "dims": list(grid.dims),
        "fadeDeg": FADE_DEG,
        "rule": RULE,
        **grid.stats,
    }


def write_view_cones(grid: ConeGrid, out_dir: Path) -> dict[str, object]:
    """`viewcones.bin` in `out_dir`; returns the `extras.viewCones` that describes it."""
    (out_dir / URI).write_bytes(encode_view_cones(grid))
    return view_cones_extras(grid)


def build_view_cones(source: Path, out_dir: Path, tileset: Path | None = None) -> dict[str, object]:
    """`viewcones.bin` for a scan already packed: from its PLY, or from its tiles' leaves.

    With `tileset`, that `tileset.json` gains `extras.viewCones` too -- how a scan published
    before the grid gets one. `source` may be the PLY the tiles were packed from or the
    `tileset.json` itself.
    """
    if source.suffix == ".json":
        grid = cone_grid_from_tileset(source)
    else:
        from splat_tiles import _filter_rows, ply_layout

        layout = ply_layout(source)
        keep, _ = _filter_rows(layout, 0.02)
        grid = cone_grid(layout, keep)
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = write_view_cones(grid, out_dir)
    if tileset is not None:
        document = json.loads(tileset.read_text(encoding="utf-8"))
        # Where `splat_tiles.convert` puts it -- before the collision grid -- so a backfilled
        # tileset is the bytes a fresh pack writes.
        old = document["root"].get("extras", {})
        ordered: dict[str, object] = {}
        for key, value in old.items():
            if key == "collision":
                ordered["viewCones"] = extras
            if key != "viewCones":
                ordered[key] = value
        ordered.setdefault("viewCones", extras)
        document["root"]["extras"] = ordered
        tileset.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return extras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", type=Path, help="the scan's PLY, or its tileset.json")
    parser.add_argument("out_dir", type=Path, help="where viewcones.bin goes")
    parser.add_argument("--tileset", type=Path, help="a tileset.json to declare it on")
    args = parser.parse_args(argv)
    extras = build_view_cones(args.source, args.out_dir, args.tileset)
    json.dump(extras, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
