"""Ground pass: the terrain under a splat scan, every splat's height above it, and which
splats are the ground (docs/SCENE_OBJECTS.md §3a). Geometry only, CPU, NumPy and SciPy.

Shared by the segmentation bake-off's candidates (A ground first, B concept first, C feature
fields). **The API below is stable**: a candidate imports only these names, and a change to
them is a change to all three.

    import ground_pass as gp

    g = gp.ground_pass(positions, opacities=None, scales=None, params=gp.GroundParams())
    g.hag       # (n,) float32: height above the terrain, metres (z up, the tileset's ENU)
    g.label     # (n,) uint8: gp.ABOVE 0, gp.GROUND 1, gp.UNKNOWN 2, gp.BELOW 3
    g.ground    # (n,) bool: label == GROUND
    g.terrain   # gp.Terrain, the DTM raster: .at(xy) heights, .seen (bool raster), .cell
    g.layer_m   # GROUND is -below_m <= hag <= layer_m where the terrain was seen
    g.below_m
    g.stats     # what it found, JSON-ready (cell, slope, shares, timings)
    g.save(path) / gp.load(path)       # .npz: hag, label, the terrain, the thresholds
    gp.split_cells(cell, label)        # voxel cells cut at the ground surface

**Labels.** ``GROUND``: in the ground layer where the terrain was seen near the splat.
``ABOVE``: above the layer (things, plants, people). ``UNKNOWN``: in the layer, but where no
ground was seen within ``support_m`` -- under a canopy that hid the ground, under an
object's footprint: the height there is interpolated, so whether the splat is ground is not
known, and it is said so. ``BELOW``: under the terrain by more than ``below_m`` -- the
floaters a splat capture leaves beneath its surfaces.

**Method: a progressive morphological filter (SMRF, Pingel et al. 2013) on a robust lowest
surface.** Chosen over the three alternatives, on these splats:

* The **slope filter** already in the repo (`scene_plants.ground_model`, Vosselman 2000) is
  an erosion by a cone from each cell's *lowest* splat. Splat scans keep floaters beneath
  their surfaces, and one floater a metre down dents the cone-eroded surface for metres
  around; and the cone rises under an object (`slope` x its half width), so the terrain
  bulges up under a pumpkin by a quarter of its width and takes its lower part as ground.
* **CSF** (cloth simulation, Zhang et al. 2016) needs PDAL or a compiled binding, iterates a
  cloth over the whole grid, and its rigidness and resolution trade the same things SMRF's
  slope and window do; nothing here needs its robustness on steep relief.
* **SMRF**: openings with growing windows remove anything narrower than the window and
  steeper than the slope; what is left is interpolated *harmonically* under the objects, so
  the terrain under a pumpkin or a tent is flat, not raised. NumPy and SciPy only, linear in
  the raster at any window (scipy's separable min/max filters).

Floaters are dealt with before any of it: transparent splats (`opacity_min`) do not shape
the terrain; a cell's surface is the `low_quantile` of its lowest layer's heights, not
its lowest; and cells whose surface still sits under the low end of their neighbourhood (its 20th
percentile over 7 x 7 cells, past a threshold from the scan's own spread) are dropped.

Every threshold that is a length is either derived from the scan (its point spacing, the
spread of what rests on the ground) or bounded by a parameter in `GroundParams`. What
geometry cannot know: a thin board lying on the ground (the spool's bottom flange, a deck)
is ground to any height filter. A candidate that knows better from its masks claims such
splats back (segment_ground_first's refine pass).

Usage (inspection; the candidates call `ground_pass` in-process):
    python ground_pass.py TILESET.json|SPLAT.ply [--out ground.npz] [--png ground.png]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

__all__ = [
    "ABOVE",
    "BELOW",
    "GROUND",
    "LABELS",
    "UNKNOWN",
    "GroundParams",
    "GroundResult",
    "Terrain",
    "ground_pass",
    "load",
    "split_cells",
]

ABOVE, GROUND, UNKNOWN, BELOW = 0, 1, 2, 3
LABELS = {ABOVE: "above", GROUND: "ground", UNKNOWN: "unknown", BELOW: "below"}
FORMAT = "hexapod.ground"
VERSION = 1

#: Robust spread: the MAD's factor to a normal sigma, and how many sigmas a threshold sits out.
MAD_SIGMA = 1.4826
SIGMAS = 3.0
#: Pits: how many sigmas of the closing's lift a cell's surface may sit below it, and at
#: least this many floors (a grassy or hay-strewn surface is rough at the floor's scale).
PIT_SIGMAS = 5.0
PIT_FLOORS = 4.0
#: ... below this quantile of the surface over this many cells a side.
PIT_QUANTILE = 0.2
PIT_WINDOW = 7
#: A cell's surface is a low quantile of its lowest layer: the splats this many floors above
#: its lowest.
LOW_BAND_FLOORS = 2.0
#: Splats sampled for the point spacing.
SPACING_SAMPLE = 200_000
#: The raster's cell is at least the scan's robust extent over this.
MAX_CELLS_SIDE = 1500
#: The noise floor is at least this many times the ground's roughness: the robust spread of
#: its cells' surface about their 3 x 3 median.
ROUGHNESS_FLOORS = 3.0
#: The harmonic fill solves directly up to this many raster cells, else coarse to fine.
DIRECT_CELLS = 40_000
FINE_SWEEPS = 60


@dataclass(frozen=True)
class GroundParams:
    """What `ground_pass` may be told. Lengths in metres; None: derived from the scan."""

    #: The terrain raster's cell; None: where a cell holds `points_per_cell` splats on
    #: average over the area the scan covers, within [`min_cell_m`, `max_cell_m`].
    cell_m: float | None = None
    points_per_cell: float = 8.0
    min_cell_m: float = 0.02
    max_cell_m: float = 0.25
    #: Splats less opaque than this do not shape the terrain (they are still labelled).
    opacity_min: float = 0.1
    #: Splats whose largest axis is over this many cells do not shape it either (blobs).
    max_scale_cells: float = 4.0
    #: A cell's surface: this quantile of the heights of its splats.
    low_quantile: float = 0.1
    #: SMRF's slope (rise over run): steeper than this, over a window's width, is an object.
    slope: float = 0.35
    #: SMRF's largest window (radius); None: a quarter of the scan's shorter side, in
    #: [`min_window_m`, `max_window_m`].
    window_m: float | None = None
    min_window_m: float = 1.0
    max_window_m: float = 12.0
    #: The terrain is seen at a splat when a ground cell is this near; None: 3 cells.
    support_m: float | None = None
    #: The ground layer's thickness; None: from the scan (what rests on the terrain's seen
    #: cells: median + 3 sigma of their heights), within [floor, `max_layer_m`].
    layer_m: float | None = None
    max_layer_m: float = 0.3


@dataclass
class Terrain:
    """The DTM: heights on a raster whose cell ``(i, j)`` is centred at
    ``origin + (i + 0.5, j + 0.5) * cell`` (x east, y north), and where ground was seen."""

    origin: np.ndarray  # (2,)
    cell: float
    heights: np.ndarray  # (nx, ny) float32, metres
    #: (nx, ny) bool: ground observed in this cell (a splat rests on the surface here).
    seen: np.ndarray
    #: (nx, ny) float32: metres from the cell to the nearest seen cell.
    seen_distance: np.ndarray

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The raster cell holding each plan point (clamped to the raster)."""
        ij = np.floor((np.asarray(xy, np.float64) - self.origin) / self.cell).astype(np.int64)
        nx, ny = self.heights.shape
        return np.clip(ij[:, 0], 0, nx - 1), np.clip(ij[:, 1], 0, ny - 1)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear terrain height at plan points ``xy`` (n, 2), clamped to the raster."""
        xy = np.asarray(xy, np.float64)
        u = (xy[:, 0] - self.origin[0]) / self.cell - 0.5
        v = (xy[:, 1] - self.origin[1]) / self.cell - 0.5
        nx, ny = self.heights.shape
        u = np.clip(u, 0.0, nx - 1.0)
        v = np.clip(v, 0.0, ny - 1.0)
        i0 = np.minimum(np.floor(u).astype(np.int64), max(nx - 2, 0))
        j0 = np.minimum(np.floor(v).astype(np.int64), max(ny - 2, 0))
        i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
        fu, fv = u - i0, v - j0
        h = self.heights.astype(np.float64)
        return (
            h[i0, j0] * (1 - fu) * (1 - fv)
            + h[i1, j0] * fu * (1 - fv)
            + h[i0, j1] * (1 - fu) * fv
            + h[i1, j1] * fu * fv
        )

    def seen_at(self, xy: np.ndarray, within_m: float = 0.0) -> np.ndarray:
        """Whether ground was seen within ``within_m`` of each plan point's cell."""
        i, j = self.index(xy)
        return self.seen_distance[i, j] <= within_m + 1e-9

    def to_json(self) -> dict:
        return {
            "originM": [round(float(self.origin[0]), 5), round(float(self.origin[1]), 5)],
            "cellM": round(float(self.cell), 5),
            "shape": [int(self.heights.shape[0]), int(self.heights.shape[1])],
            "seenShare": round(float(self.seen.mean()), 4),
        }


@dataclass
class GroundResult:
    hag: np.ndarray  # (n,) float32
    label: np.ndarray  # (n,) uint8
    terrain: Terrain
    layer_m: float
    below_m: float
    params: GroundParams
    stats: dict = field(default_factory=dict)

    @property
    def ground(self) -> np.ndarray:
        return self.label == GROUND

    def save(self, path: Path) -> None:
        """An .npz: hag, label, the terrain raster and the thresholds (`load` reads it)."""
        np.savez_compressed(
            path,
            hag=self.hag.astype(np.float32),
            label=self.label.astype(np.uint8),
            heights=self.terrain.heights.astype(np.float32),
            seen=self.terrain.seen,
            seen_distance=self.terrain.seen_distance.astype(np.float32),
            origin=self.terrain.origin.astype(np.float64),
            cell=np.float64(self.terrain.cell),
            layer_m=np.float64(self.layer_m),
            below_m=np.float64(self.below_m),
            meta=np.frombuffer(
                json.dumps(
                    {"format": FORMAT, "version": VERSION, "params": asdict(self.params),
                     "stats": self.stats}
                ).encode("utf-8"),
                np.uint8,
            ),
        )  # fmt: skip


def load(path: Path) -> GroundResult:
    """A `GroundResult` written by `GroundResult.save`."""
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        terrain = Terrain(
            z["origin"], float(z["cell"]), z["heights"], z["seen"], z["seen_distance"]
        )
        return GroundResult(
            z["hag"], z["label"], terrain, float(z["layer_m"]), float(z["below_m"]),
            GroundParams(**meta["params"]), meta["stats"],
        )  # fmt: skip


# ----------------------------------------------------------------------------- helpers


def robust_spread(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-sigma of ``values`` after iterative 3-sigma clipping."""
    kept = np.asarray(values, np.float64)
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return 0.0, 0.0
    median = float(np.median(kept))
    sigma = MAD_SIGMA * float(np.median(np.abs(kept - median)))
    for _ in range(10):
        inside = kept[np.abs(kept - median) <= SIGMAS * max(sigma, 1e-12)]
        if inside.size in (0, kept.size):
            break
        kept = inside
        median = float(np.median(kept))
        sigma = MAD_SIGMA * float(np.median(np.abs(kept - median)))
    return median, sigma


def point_spacing(xyz: np.ndarray, seed: int = 0) -> float:
    """Median nearest-neighbour distance among a seeded sample of the splats, scaled to the
    whole scan's density (a sample of m of n splats spaces them sqrt(n / m) times wider on a
    surface)."""
    n = len(xyz)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    m = min(n, SPACING_SAMPLE)
    sample = xyz if m == n else xyz[np.sort(rng.choice(n, m, replace=False))]
    distance, _ = cKDTree(sample).query(sample, k=2)
    return float(np.median(distance[:, 1])) * math.sqrt(m / n)


def _cell_size(xy: np.ndarray, params: GroundParams) -> float:
    """`GroundParams.cell_m`, else the plan cell holding `points_per_cell` splats on average
    over the cells the scan occupies (two passes, so a ragged footprint is not diluted)."""
    if params.cell_m is not None:
        return float(params.cell_m)
    count = max(len(xy), 1)
    low = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - low, 1e-3)
    cell = math.sqrt(params.points_per_cell * float(span[0] * span[1]) / count)
    for _ in range(2):
        keys = np.floor((xy - low) / cell).astype(np.int64)
        occupied = np.unique(keys[:, 0] * (int(span[1] / cell) + 2) + keys[:, 1]).size
        cell = math.sqrt(params.points_per_cell * occupied * cell * cell / count)
    # A scan many hundred metres wide needs no centimetre raster: at least 1 / `MAX_CELLS_SIDE`
    # of its robust extent.
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    least = max(params.min_cell_m, float(np.max(hi - lo)) / MAX_CELLS_SIDE)
    return float(min(max(cell, least), params.max_cell_m))


def _fill_nearest(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Every cell its nearest known cell's value."""
    if known.all() or not known.any():
        return np.where(known, values, 0.0)
    _, (ii, jj) = ndimage.distance_transform_edt(~known, return_indices=True)
    return values[ii, jj]


def _laplace_direct(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The harmonic interpolation of ``values`` at ``known`` cells: every unknown cell the
    mean of its 4-neighbours (Neumann at the raster's edge), solved exactly."""
    h, w = values.shape
    unknown = np.flatnonzero(~known.reshape(-1))
    out = values.astype(np.float64).reshape(-1).copy()
    if unknown.size == 0:
        return out.reshape(h, w)
    position = np.full(h * w, -1, np.int64)
    position[unknown] = np.arange(unknown.size)
    rows, cols, data = [], [], []
    rhs = np.zeros(unknown.size)
    ui, uj = np.divmod(unknown, w)
    degree = np.zeros(unknown.size)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ui + di, uj + dj
        inside = (ni >= 0) & (ni < h) & (nj >= 0) & (nj < w)
        degree += inside
        flat = np.where(inside, ni * w + nj, 0)
        neighbour = position[flat]
        free = inside & (neighbour >= 0)
        rows.append(np.flatnonzero(free))
        cols.append(neighbour[free])
        data.append(-np.ones(int(free.sum())))
        fixed = inside & (neighbour < 0)
        rhs[fixed] += out[flat[fixed]]
    rows.append(np.arange(unknown.size))
    cols.append(np.arange(unknown.size))
    data.append(degree)
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(unknown.size, unknown.size),
    ).tocsr()
    out[unknown] = spsolve(matrix, rhs)
    return out.reshape(h, w)


def harmonic_fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """``values`` kept at ``known`` cells, the rest interpolated harmonically (a membrane
    stretched over the known heights): exactly up to `DIRECT_CELLS` cells, else coarse to
    fine -- the half-resolution solution upsampled, then `FINE_SWEEPS` Jacobi sweeps."""
    values = np.asarray(values, np.float64)
    known = np.asarray(known, bool)
    if not known.any():
        return np.zeros_like(values)
    if known.all():
        return values.copy()
    h, w = values.shape
    if h * w <= DIRECT_CELLS:
        return _laplace_direct(values, known)
    ph, pw = h + (h % 2), w + (w % 2)
    v = np.zeros((ph, pw))
    k = np.zeros((ph, pw))
    v[:h, :w] = np.where(known, values, 0.0)
    k[:h, :w] = known
    blocks_v = v.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    blocks_k = k.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    coarse_known = blocks_k > 0
    coarse = np.divide(blocks_v, blocks_k, out=np.zeros_like(blocks_v), where=coarse_known)
    coarse = harmonic_fill(coarse, coarse_known)
    x = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:h, :w]
    x = np.where(known, values, x)
    free = ~known
    for _ in range(FINE_SWEEPS):
        p = np.pad(x, 1, mode="edge")
        mean = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        x[free] = mean[free]
    return x


def _windows(cell: float, window_m: float) -> list[int]:
    """SMRF's window radii in cells: 1, 2, 3, then growing by half, up to `window_m`."""
    top = max(1, round(window_m / cell))
    radii: list[int] = []
    r = 1.0
    while round(r) <= top:
        if not radii or round(r) != radii[-1]:
            radii.append(round(r))
        r = r + 1 if r < 3 else r * 1.5
    if radii[-1] != top:
        radii.append(top)
    return radii


def _smrf(
    filled: np.ndarray,
    observed: np.ndarray,
    radii: list[int],
    slope: float,
    cell: float,
    floor: float,
) -> np.ndarray:
    """SMRF's object cells: progressive openings of the surface, each window's radius in
    `radii`; a cell that rises above the next opening by more than `slope` x the radius
    (plus the noise `floor`) is an object."""
    objects = np.zeros(filled.shape, bool)
    last = filled
    for r in radii:
        opened = ndimage.grey_opening(last, size=(2 * r + 1, 2 * r + 1))
        objects |= (last - opened > slope * r * cell + floor) & observed
        last = opened
    return objects


def _terrain_heights(filled: np.ndarray, ground_cells: np.ndarray) -> np.ndarray:
    """The ground cells' surface smoothed among ground cells (3 x 3), and a membrane over
    every other cell (`harmonic_fill`)."""
    known = ground_cells.astype(np.float64)
    numerator = ndimage.uniform_filter(np.where(ground_cells, filled, 0.0), 3, mode="nearest")
    denominator = ndimage.uniform_filter(known, 3, mode="nearest")
    smooth = np.divide(numerator, denominator, out=filled.copy(), where=denominator > 0)
    return harmonic_fill(np.where(ground_cells, smooth, 0.0), ground_cells)


# --------------------------------------------------------------------------- the pass


def ground_pass(
    positions: np.ndarray,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    params: GroundParams | None = None,
) -> GroundResult:
    """The terrain under ``positions`` (n, 3; z up), every splat's height above it and its
    label (module docstring). ``opacities`` (n,) in 0..1 and ``scales`` (n, 3) metres, when
    given, keep transparent splats and blobs from shaping the terrain. Deterministic."""
    params = params or GroundParams()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    pos = np.asarray(positions)
    n = len(pos)
    if n == 0:
        raise ValueError("ground_pass needs at least one splat")
    xy = np.asarray(pos[:, :2], np.float64)
    z = np.asarray(pos[:, 2], np.float64)
    spacing = point_spacing(pos)
    cell = _cell_size(xy, params)
    floor = max(2.0 * spacing, 0.005)

    # Which splats shape the terrain.
    shaping = np.ones(n, bool)
    if opacities is not None:
        shaping &= np.asarray(opacities) >= params.opacity_min
    if scales is not None:
        shaping &= np.asarray(scales).max(axis=1) <= params.max_scale_cells * cell
    if shaping.sum() < max(16, n // 100):
        shaping[:] = True

    origin = xy.min(axis=0) - cell
    span = xy.max(axis=0) - origin
    shape = (math.ceil(span[0] / cell) + 2, math.ceil(span[1] / cell) + 2)
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    flat = ij[:, 0] * shape[1] + ij[:, 1]

    # A robust lowest surface: per cell the `low_quantile` of its shaping splats' heights
    # among those within `LOW_BAND_FLOORS` floors of its lowest -- of the lowest layer, not
    # of the whole column (a cell under a crown holds one ground splat and thirty leaves).
    rows = np.flatnonzero(shaping)
    order = rows[np.lexsort((z[rows], flat[rows]))]
    cells_sorted = flat[order]
    starts = np.flatnonzero(np.r_[True, cells_sorted[1:] != cells_sorted[:-1]])
    counts = np.diff(np.r_[starts, cells_sorted.size])
    lowest = np.repeat(z[order[starts]], counts)
    group = np.repeat(np.arange(starts.size), counts)
    low = np.bincount(group, z[order] <= lowest + LOW_BAND_FLOORS * floor, starts.size)
    pick = starts + np.floor(params.low_quantile * (low.astype(np.int64) - 1)).astype(np.int64)
    del lowest, group
    surface = np.full(shape[0] * shape[1], np.nan)
    surface[cells_sorted[starts]] = z[order[pick]]
    surface = surface.reshape(shape)
    observed = np.isfinite(surface)
    timings["surfaceS"] = time.perf_counter() - started

    filled = _fill_nearest(np.nan_to_num(surface), observed)
    lo2, hi2 = np.percentile(xy, 2, axis=0), np.percentile(xy, 98, axis=0)
    window = params.window_m
    if window is None:
        window = min(max(0.25 * float(np.min(hi2 - lo2)), params.min_window_m),
                     params.max_window_m)  # fmt: skip
    radii = _windows(cell, window)

    # Pits: a cell whose surface sits under the low end of its neighbourhood -- its
    # `PIT_QUANTILE` over `PIT_WINDOW` cells a side -- past the scan's own spread is a
    # floater. The low end, not the median: ground seen through gaps in a crown sits below
    # most of its neighbours too, and is the ground; a floater is below nearly all of them.
    pits = 0
    for _ in range(2):
        reference = ndimage.percentile_filter(
            filled, 100 * PIT_QUANTILE, size=PIT_WINDOW, mode="nearest"
        )
        sink = reference - filled
        median, sigma = robust_spread(sink[observed])
        pit = observed & (sink > max(median + PIT_SIGMAS * sigma, PIT_FLOORS * floor))
        if not pit.any():
            break
        pits += int(pit.sum())
        observed &= ~pit
        filled = _fill_nearest(np.nan_to_num(surface), observed)
    timings["pitsS"] = time.perf_counter() - started

    # SMRF: progressive openings; what rises above an opening by more than the slope over
    # the window's radius (plus the noise floor) is an object. The floor is first the
    # splats' spacing, then also the roughness of the ground that finds: a forest floor or
    # a hay bed is rough at a few centimetres however densely it was captured.
    objects = _smrf(filled, observed, radii, params.slope, cell, floor)
    ground_cells = observed & ~objects
    if ground_cells.sum() >= 16:
        deviation = (filled - ndimage.median_filter(filled, size=3, mode="nearest"))[ground_cells]
        _, roughness = robust_spread(deviation)
        if ROUGHNESS_FLOORS * roughness > floor:
            floor = ROUGHNESS_FLOORS * roughness
            objects = _smrf(filled, observed, radii, params.slope, cell, floor)
    ground_cells = observed & ~objects
    if ground_cells.sum() < max(4, 0.01 * observed.sum()):
        ground_cells = observed.copy()  # nothing but objects: take the surface as it is
    timings["smrfS"] = time.perf_counter() - started

    heights = _terrain_heights(filled, ground_cells)
    seen_distance = (
        ndimage.distance_transform_edt(~ground_cells) * cell
        if not ground_cells.all()
        else np.zeros(shape)
    )
    terrain = Terrain(
        origin, cell, heights.astype(np.float32), ground_cells, seen_distance.astype(np.float32)
    )
    timings["terrainS"] = time.perf_counter() - started

    hag = np.empty(n, np.float32)
    step = 1 << 22
    for a in range(0, n, step):
        b = min(a + step, n)
        hag[a:b] = z[a:b] - terrain.at(xy[a:b])
    support = params.support_m if params.support_m is not None else 3.0 * cell
    supported = seen_distance.reshape(-1)[flat] <= support + 1e-9
    resting = shaping & ground_cells.reshape(-1)[flat] & (hag < params.max_layer_m)
    median, sigma = robust_spread(hag[resting])
    if params.layer_m is not None:
        layer = float(params.layer_m)
    else:
        layer = min(max(median + SIGMAS * sigma, floor), params.max_layer_m)
    below = max(-(median - SIGMAS * sigma), floor)

    label = np.full(n, GROUND, np.uint8)
    label[~supported] = UNKNOWN
    label[hag > layer] = ABOVE
    label[hag < -below] = BELOW
    timings["labelsS"] = time.perf_counter() - started
    counts_by = np.bincount(label, minlength=4)
    stats = {
        "splats": n,
        "spacingM": round(spacing, 5),
        "cellM": round(cell, 5),
        "floorM": round(floor, 5),
        "slope": params.slope,
        "windowM": round(float(window), 4),
        "windowsCells": radii,
        "layerM": round(layer, 4),
        "belowM": round(below, 4),
        "supportM": round(float(support), 4),
        "pitCells": pits,
        "observedCells": int(observed.sum()),
        "objectCells": int(objects.sum()),
        "groundCells": int(ground_cells.sum()),
        "terrain": terrain.to_json(),
        "labels": {LABELS[k]: int(counts_by[k]) for k in LABELS},
        "shares": {LABELS[k]: round(float(counts_by[k]) / n, 4) for k in LABELS},
        "timingsS": {k: round(v, 2) for k, v in timings.items()},
    }
    return GroundResult(hag, label, terrain, layer, below, params, stats)


def split_cells(cell: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, int]:
    """Voxel cells (per splat, 0..k-1) cut at the ground surface: a cell holding both ground
    (GROUND, BELOW) and other splats becomes two, so no cell carries both a flange and the
    soil under it. Returns the new cell per splat (0..m-1, the old order kept: a cell's
    ground part first) and m."""
    cell = np.asarray(cell, np.int64)
    low = np.isin(label, (GROUND, BELOW)).astype(np.int64)
    key = cell * 2 + (1 - low)
    unique, inverse = np.unique(key, return_inverse=True)
    return inverse.astype(np.int32).reshape(-1), int(unique.size)


# ------------------------------------------------------------------------------ the CLI


def _top_view(positions: np.ndarray, colours: np.ndarray, px: float, out: Path) -> None:
    """A plan view, the highest splat per pixel coloured, as a PNG."""
    from PIL import Image

    xy = positions[:, :2]
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / px).astype(np.int64)
    w, h = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    order = np.argsort(positions[:, 2], kind="stable")
    image = np.zeros((h, w, 3), np.uint8)
    image[h - 1 - ij[order, 1], ij[order, 0]] = colours[order]
    Image.fromarray(image).save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="a tileset.json (its leaves) or a 3DGS PLY")
    parser.add_argument("--out", type=Path, default=None, help="write the result (.npz)")
    parser.add_argument("--png", type=Path, default=None, help="a plan view by label")
    parser.add_argument("--slope", type=float, default=GroundParams.slope)
    args = parser.parse_args()
    from splat_render import load_ply, load_tileset

    splats = (
        load_tileset(args.source) if args.source.name.endswith(".json") else load_ply(args.source)
    )
    result = ground_pass(
        splats.positions, splats.opacities, splats.scales, GroundParams(slope=args.slope)
    )
    print(json.dumps(result.stats, indent=1))
    if args.out:
        result.save(args.out)
    if args.png:
        palette = np.array([[200, 70, 60], [90, 170, 70], [230, 200, 60], [70, 90, 220]], np.uint8)
        px = max(result.terrain.cell / 2, 0.01)
        _top_view(splats.positions, palette[result.label], px, args.png)


if __name__ == "__main__":
    main()
