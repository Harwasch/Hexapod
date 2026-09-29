"""Every plant in a splat capture, found from geometry and colour, rigged and ready to move.

``real_tree.py`` isolates *one* tree: the ground below the lowest camera, a trunk ring, a crown
cylinder. A capture of a park is many plants at once -- low shrubs, leafy trees, bare trunks --
beside buildings, paths and lawn that must stay exactly still. This module is the scene step
for that case (the geometry-only "Path B" of the scene-understanding plan): it takes a metric,
east-north-up splat (``canonical.ply``, as ``splat-ingest`` leaves it) and returns

* a **ground model** -- a raster of terrain heights -- and every splat's height above it;
* a **class** per splat: tree, shrub, snag, grass/low, ground, other-static;
* **plant instances**: id, stem point, height, crown polygon and radius, class, and which rig
  the evidence supports;

and then (``build``) a rig per plant, all of them in **one forest rig** for the tileset, a
motion sidecar covering every plant, and a **per-splat plant binding** keyed by every tile's
checksum, so the viewer can prove per tile which gaussians belong to which plant and which to
nothing (``plants.json``; apps/web ``splatTiles.ts`` reads it).

Nothing is tuned to a capture. Every threshold is derived from the capture itself or cited:

=========================  ===================================================================
ground cell                the plan spacing at which a cell holds ``GROUND_CELL_POINTS`` (8)
                           splats on average: splat_ground's own ``min_points``
ground surface             the highest surface below every cell's lowest splat whose slope
                           never exceeds S (Vosselman 2000, slope-based filtering); S the
                           capture's own cell-to-cell slope, median + 3 sigma (robust)
ground layer               median + 3 sigma (robust) of heights above that surface where it
                           rests on the capture, floored at the point spacing
greenness                  ExG = 2g - r - b on chromatic coordinates (Woebbecke et al. 1995),
                           split by Otsu (1979) on the capture's own histogram; VARI
                           (Gitelson et al. 2002) reported beside it
object seeds, classes      FAO FRA 2020: shrubs 0.5-5 m, trees >= 5 m
connectivity               ``LINK_FACTOR`` x the cloud's median neighbour spacing, as
                           skeleton.py links a band (density-relative, so it carries across
                           captures of any density)
tree tops                  local maxima in a window of crown width CW(H) = 2.51503 +
                           0.00901 H^2 (Popescu & Wynne 2004, mixed stands), trees only
leafy                      a majority of the instance's splats are green
snag                       tree height, not leafy, and plan spread under CW(H)/2: a standing
                           trunk that has lost at least three quarters of the crown width a
                           live tree of its height carries
rig                        skeleton.py's banded skeleton where the instance has as many
                           splats per metre of height as the thinnest cloud it is tested on
                           (the synthetic tree at a quarter density); a crown rig otherwise
=========================  ===================================================================

Motion per class (``motion_params.py`` for every rule it already has): a leafy tree is the
existing limb model on its rig; a shrub is the same laws on a small crown rig -- its short
height puts its sway frequency high (the pendulum law, extrapolated below the 4.7 m of its
data) and its swing small (the same bend angle on a short lever), and most of its motion is
leaf flutter; a snag is its trunk's mode alone, its bend scaled by its frontal area against a
leafy crown's, and no flutter; everything else is bound to a static anchor node that never
moves. ADR 0008's addendum on multi-plant scenes has the sources and what is estimated.

Usage (on a capture ``splat-ingest`` produced; the tiles are re-packed from the PLY with the
``package`` stage's parameters when ``--tiles`` is not given, byte-identical to its own)::

    uv run python scene_plants.py canonical.ply --tiles splat/ [--out splat/]
        [--lat 46.13 --lon -123.88 --height 0] [--truth labels.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay, cKDTree

import motion_params
import rig_tiles
import skeleton
import splat_tiles
from synthetic_tree import checksum_positions

#: The per-splat classes, in the order the class raster and the scores index them.
CLASSES = ("tree", "shrub", "snag", "grass/low", "ground", "other-static")
TREE, SHRUB, SNAG, LOW, GROUND, STATIC = range(len(CLASSES))
#: Classes whose instances are rigged and move.
PLANT_CLASSES = (TREE, SHRUB, SNAG)

#: FAO Global Forest Resources Assessment 2020, Terms and Definitions: a shrub is a woody
#: perennial plant "generally more than 0.5 m and less than 5 m in height at maturity", a tree
#: one "able to reach a minimum height of 5 m in situ".
FAO_SHRUB_MIN_M = 0.5
FAO_TREE_MIN_M = 5.0

#: Splats a ground cell holds on average: the ``min_points`` the pipeline's ``splat_ground``
#: stage asks of a cell before it samples it (tools/pipeline/recipes/splat-ingest.yaml defaults,
#: stages.py ``splat_ground``). Dimensionless.
GROUND_CELL_POINTS = 8
#: Robust clipping width, in standard deviations: Gaussian tails beyond 3 sigma are 0.27 %.
SIGMA_CLIP = 3.0
#: MAD to standard deviation for a normal distribution.
MAD_SIGMA = 1.4826

#: skeleton.py's own link radius, as a multiple of the median neighbour spacing, and its
#: smallest cluster: the same density-relative connectivity, one scale up.
LINK_FACTOR = 4.5
MIN_CLUSTER_POINTS = 8
#: Neighbours examined per splat when linking components (within the link radius).
LINK_NEIGHBOURS = 12
#: Splats a neighbour query handles at once, so memory stays flat in the capture's size.
QUERY_CHUNK = 1 << 20
#: Splats of an unrooted piece sampled when measuring its gap to the others (evenly).
GAP_SAMPLE = 1 << 16

#: Popescu & Wynne 2004, "Seeing the trees in the forest", PE&RS 70(5):589-604: crown width
#: from height for mixed stands, metres. Used as the tree-top search window and as the crown a
#: live tree of a given height carries. Fitted on trees, so applied only at tree heights.
POPESCU_WYNNE_MIXED = (2.51503, 0.0, 0.00901)

#: A majority vote: an instance is leafy when more than half its splats are green.
LEAFY_FRACTION = 0.5
#: A snag has lost at least three quarters of a live crown's width: its spread under CW(H)/2.
SNAG_SPREAD_OF_CROWN = 0.5

#: The evidence bar for the banded skeleton (skeleton.py), in splats per metre of height: the
#: thinnest cloud its recovery is tested on is the synthetic tree at a quarter of its density
#: (tests/test_skeleton.py ``test_recovery_survives_a_quarter_density_cloud``: 12,000 / 4
#: splats over the fixture's 6.4597 m, data/tiles/synthetic-tree/source/motion.json).
SKELETON_FIXTURE_SPLATS = 12000
SKELETON_FIXTURE_THINNING = 4
SKELETON_FIXTURE_HEIGHT_M = 6.4597
SKELETON_MIN_SPLATS_PER_M = (
    SKELETON_FIXTURE_SPLATS / SKELETON_FIXTURE_THINNING / SKELETON_FIXTURE_HEIGHT_M
)
#: Crown-rig limbs: four sectors about the stem. Few-bone rigs are the only ones whose
#: frequencies are identifiable from video (Chen & Lou, "Wind on Trees", 2026: 22 bones yes,
#: 224 no); SpeedTree moves one or two branch levels, Pivot Painter 2 at most four.
CROWN_SECTORS = 4

#: Node budget the skeleton gets per tree: skeleton.py's own default.
SKELETON_MAX_NODES = 200

#: The id of the static anchor node every non-plant splat is bound to.
STATIC_NODE_ID = "static"

PLANTS_FORMAT = "hexapod.plants"
PLANTS_VERSION = 1

#: The ``package`` stage's parameters (tools/pipeline/recipes/splat-ingest.yaml), so tiles
#: re-packed here are byte-identical to the ones the pipeline published.
PACKAGE_TILE_GAUSSIANS = splat_tiles.TILE_GAUSSIANS
PACKAGE_OPACITY_MIN = 0.02


# ---------------------------------------------------------------------------------- helpers


def popescu_wynne_crown_m(height_m: float | np.ndarray) -> float | np.ndarray:
    """Crown width, metres, for a tree ``height_m`` tall (mixed stands)."""
    a, b, c = POPESCU_WYNNE_MIXED
    return a + b * height_m + c * np.square(height_m)


def robust_sigma(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-sigma of ``values`` after iterative 3-sigma clipping."""
    kept = np.asarray(values, dtype=np.float64)
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return 0.0, 0.0
    median = float(np.median(kept))
    sigma = MAD_SIGMA * float(np.median(np.abs(kept - median)))
    for _ in range(10):
        inside = kept[np.abs(kept - median) <= SIGMA_CLIP * max(sigma, 1e-12)]
        if inside.size == 0 or inside.size == kept.size:
            break
        kept = inside
        median = float(np.median(kept))
        sigma = MAD_SIGMA * float(np.median(np.abs(kept - median)))
    return median, sigma


def otsu_threshold(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Otsu's (1979) threshold on ``values``, and its effectiveness sigma_B^2 / sigma_T^2."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    low, high = float(finite.min()), float(finite.max())
    if not high > low:
        return low, 0.0
    counts, edges = np.histogram(finite, bins=bins, range=(low, high))
    centres = (edges[:-1] + edges[1:]) / 2
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total * omega - mu) ** 2 / (omega * (1 - omega))
    between[~np.isfinite(between)] = 0.0
    k = int(np.argmax(between))
    variance = float(np.sum(p * (centres - total) ** 2))
    return float(edges[k + 1]), float(between[k] / variance) if variance > 0 else 0.0


def chromatic_indices(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ExG on chromatic coordinates (Woebbecke 1995) and VARI (Gitelson 2002), per splat."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    total = np.maximum(rgb.sum(axis=1), 1e-9)
    r, g, b = (rgb[:, k] / total for k in range(3))
    exg = 2 * g - r - b
    with np.errstate(divide="ignore", invalid="ignore"):
        vari = (rgb[:, 1] - rgb[:, 0]) / (rgb[:, 1] + rgb[:, 0] - rgb[:, 2])
    vari[~np.isfinite(vari)] = 0.0
    return exg, vari


def snap(xyz: np.ndarray) -> np.ndarray:
    """The SPZ grid, as the tiles store positions (skeleton.py ``_snap``)."""
    return skeleton._snap(xyz)


# ------------------------------------------------------------------------------ the ground


@dataclass
class GroundModel:
    """Terrain heights on a regular grid: cell ``(i, j)`` is centred at ``origin + (i+.5, j+.5)*cell``."""

    origin: np.ndarray
    cell: float
    heights: np.ndarray  # (nx, ny) metres, float64
    slope_bound: float
    layer_m: float
    contact: np.ndarray  # (nx, ny) bool: the surface rests on the capture here

    def at(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear terrain height at plan points ``xy``, clamped to the grid."""
        u = (xy[:, 0] - self.origin[0]) / self.cell - 0.5
        v = (xy[:, 1] - self.origin[1]) / self.cell - 0.5
        nx, ny = self.heights.shape
        u = np.clip(u, 0.0, nx - 1.0)
        v = np.clip(v, 0.0, ny - 1.0)
        i0 = np.minimum(np.floor(u).astype(np.int64), max(nx - 2, 0))
        j0 = np.minimum(np.floor(v).astype(np.int64), max(ny - 2, 0))
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        fu = u - i0
        fv = v - j0
        h = self.heights
        return (
            h[i0, j0] * (1 - fu) * (1 - fv)
            + h[i1, j0] * fu * (1 - fv)
            + h[i0, j1] * (1 - fu) * fv
            + h[i1, j1] * fu * fv
        )

    def to_json(self) -> dict:
        return {
            "originM": [float(self.origin[0]), float(self.origin[1])],
            "cellM": round(float(self.cell), 5),
            "shape": [int(self.heights.shape[0]), int(self.heights.shape[1])],
            "slopeBound": round(float(self.slope_bound), 5),
            "layerM": round(float(self.layer_m), 5),
            "contactShare": round(float(self.contact.mean()), 4),
        }


def ground_cell(xy: np.ndarray) -> float:
    """The plan cell at which a cell holds ``GROUND_CELL_POINTS`` splats on average.

    Two passes: the bounding box's area gives a first cell, and the area those cells actually
    occupy gives the second, so an L-shaped capture is not diluted by the corner it lacks.
    """
    count = max(int(xy.shape[0]), 1)
    low = xy.min(axis=0)
    span = np.maximum(xy.max(axis=0) - low, 1e-3)
    cell = math.sqrt(GROUND_CELL_POINTS * float(span[0] * span[1]) / count)
    keys = np.floor((xy - low) / cell).astype(np.int64)
    occupied = np.unique(keys[:, 0] * (int(span[1] / cell) + 2) + keys[:, 1]).size
    return math.sqrt(GROUND_CELL_POINTS * occupied * cell * cell / count)


def _lowest_per_cell(xyz: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]):
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, shape[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, shape[1] - 1)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    lowest = np.full(shape[0] * shape[1], np.inf)
    np.minimum.at(lowest, flat, xyz[:, 2])
    return lowest.reshape(shape), flat


def _neighbour_slopes(grid: np.ndarray, cell: float) -> np.ndarray:
    """|dz| / distance between every pair of adjacent cells that both hold splats."""
    out = []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        a = grid[max(di, 0) :, max(dj, 0) : grid.shape[1] + min(dj, 0)]
        b = grid[: grid.shape[0] - di, max(-dj, 0) : grid.shape[1] - max(dj, 0)]
        both = np.isfinite(a) & np.isfinite(b)
        out.append(np.abs(a[both] - b[both]) / (cell * math.hypot(di, dj)))
    return np.concatenate(out) if out else np.zeros(0)


def slope_envelope(lowest: np.ndarray, cell: float, slope: float) -> np.ndarray:
    """The highest surface below ``lowest`` whose slope never exceeds ``slope``.

    ``G(x) = min_y (lowest(y) + slope * |x - y|)``: Vosselman's (2000) slope-based filter,
    where a point is ground when no other point within distance d lies more than
    ``slope * d`` below it. Computed by relaxation over the eight neighbours (a chamfer
    distance, within 8 % of the Euclidean) until nothing changes; empty cells take the cone
    of their nearest neighbours.
    """
    g = lowest.copy()
    steps = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                steps.append((di, dj, slope * cell * math.hypot(di, dj)))
    padded = np.full((g.shape[0] + 2, g.shape[1] + 2), np.inf)
    for _ in range(g.shape[0] + g.shape[1] + 2):
        padded[1:-1, 1:-1] = g
        best = g.copy()
        for di, dj, rise in steps:
            shifted = padded[1 + di : 1 + di + g.shape[0], 1 + dj : 1 + dj + g.shape[1]]
            np.minimum(best, shifted + rise, out=best)
        if np.array_equal(best, g):
            break
        g = best
    return g


def ground_model(xyz: np.ndarray, spacing: float) -> GroundModel:
    """The terrain under a capture, from its lowest splats and its own slope statistics."""
    cell = ground_cell(xyz[:, :2])
    origin = xyz[:, :2].min(axis=0) - cell
    span = xyz[:, :2].max(axis=0) - origin
    shape = (math.ceil(span[0] / cell) + 2, math.ceil(span[1] / cell) + 2)
    lowest, flat = _lowest_per_cell(xyz, origin, cell, shape)
    median, sigma = robust_sigma(_neighbour_slopes(lowest, cell))
    slope = median + SIGMA_CLIP * sigma
    heights = slope_envelope(lowest, cell, slope)
    contact = np.isfinite(lowest) & (lowest <= heights + 1e-9)
    model = GroundModel(origin, cell, heights, slope, 0.0, contact)
    # The ground layer: how far splats that rest on the surface scatter above it.
    h = xyz[:, 2] - model.at(xyz[:, :2])
    resting = contact.reshape(-1)[flat] & (h < FAO_SHRUB_MIN_M)
    median_h, sigma_h = robust_sigma(h[resting])
    model.layer_m = max(median_h + SIGMA_CLIP * sigma_h, spacing)
    return model


# -------------------------------------------------------------------------------- instances


@dataclass
class Instance:
    """One plant (or a rejected object): its members are indices into the analysed arrays."""

    members: np.ndarray
    klass: int = STATIC
    height_m: float = 0.0
    stem: np.ndarray = field(default_factory=lambda: np.zeros(3))
    green_fraction: float = 0.0
    spread_m: float = 0.0
    crown_radius_m: float = 0.0
    crown_polygon: list[list[float]] = field(default_factory=list)
    splats_per_m: float = 0.0
    rig: str = "none"
    id: str = ""


def local_spacing(points: np.ndarray, tree: cKDTree | None = None) -> np.ndarray:
    """Each splat's own point spacing: the median nearest-neighbour distance of its neighbours.

    skeleton.py links a band at ``LINK_FACTOR`` times *one* spacing, which is right for one
    tree. A scene is captured at many densities -- near the camera and far from it, a dense
    tree and a sparse one -- and a single spacing welds the dense parts or shatters the
    sparse ones, so here each splat carries its own.
    """
    count = int(points.shape[0])
    if count < 2:
        return np.zeros(count)
    tree = tree or cKDTree(points)
    k = min(LINK_NEIGHBOURS + 1, count)
    nearest = np.empty(count)
    for start in range(0, count, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, count)
        nearest[start:stop] = tree.query(points[start:stop], k=2)[0][:, 1]
    out = np.empty(count)
    for start in range(0, count, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, count)
        _, index = tree.query(points[start:stop], k=k)
        out[start:stop] = np.median(nearest[index], axis=1)
    return out


def _components(points: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Connected components of ``points``: a pair links within ``LINK_FACTOR`` times the larger
    of their own spacings (``spacing``, per point), among each point's ``LINK_NEIGHBOURS``
    nearest. One label a point."""
    count = int(points.shape[0])
    if count <= 1:
        return np.zeros(count, dtype=np.int64)
    k = min(LINK_NEIGHBOURS + 1, count)
    tree = cKDTree(points)
    edges_from: list[np.ndarray] = []
    edges_to: list[np.ndarray] = []
    # In windows, the edges as 32-bit indices: a few million splats stay under a gigabyte.
    for start in range(0, count, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, count)
        distance, index = tree.query(points[start:stop], k=k)
        rows = np.repeat(np.arange(start, stop, dtype=np.int32), k)
        cols = index.reshape(-1).astype(np.int32)
        reach = LINK_FACTOR * np.maximum(spacing[rows], spacing[cols])
        ok = distance.reshape(-1) <= reach
        edges_from.append(rows[ok])
        edges_to.append(cols[ok])
    rows_all = np.concatenate(edges_from)
    graph = coo_matrix(
        (np.ones(rows_all.size, dtype=np.int8), (rows_all, np.concatenate(edges_to))),
        shape=(count, count),
    )
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def _tree_tops(chm: np.ndarray, cell: float) -> np.ndarray:
    """Cells that are the highest within half a crown width CW(H) of themselves, H >= 5 m."""
    filled = np.where(np.isfinite(chm), chm, -np.inf)
    tall = filled >= FAO_TREE_MIN_M
    if not tall.any():
        return np.zeros(chm.shape, dtype=bool)
    crown = np.where(tall, popescu_wynne_crown_m(np.where(tall, filled, 0.0)), 0.0)
    radius_cells = np.ceil(crown / 2.0 / cell).astype(np.int64)
    tops = np.zeros(chm.shape, dtype=bool)
    for radius in np.unique(radius_cells[tall]):
        r = int(radius)
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        highest = ndimage.maximum_filter(filled, footprint=disk, mode="constant", cval=-np.inf)
        tops |= tall & (radius_cells == radius) & (filled >= highest)
    # A plateau of equal maxima is one top: keep one cell per connected patch of them.
    labels, count = ndimage.label(tops, structure=np.ones((3, 3)))
    if count <= 1:
        return tops
    single = np.zeros_like(tops)
    for value, patch in enumerate(ndimage.find_objects(labels), start=1):
        where = np.argwhere(labels[patch] == value)[0]
        single[patch][tuple(where)] = True
    return single


def _merge_unrooted(basins: np.ndarray, rooted: np.ndarray) -> np.ndarray:
    """Relabels basins so every unrooted one joins the rooted neighbour it borders most.

    A plant stands on the ground. A basin of the canopy with no splat near the ground is a
    lobe of a crown whose stem is elsewhere -- a wide crown has more than one local maximum
    in a window fitted to average crowns -- so it joins the rooted basin it shares the longest
    boundary with, through other unrooted basins if it must. A component with no rooted
    basin at all (a canopy seen only from above) keeps the tops as they are.
    """
    count = int(rooted.size)
    if count <= 1 or not rooted.any() or rooted.all():
        return np.arange(count)
    shared: dict[tuple[int, int], int] = {}
    for a, b in (
        (basins[1:, :], basins[:-1, :]),
        (basins[:, 1:], basins[:, :-1]),
    ):
        differ = (a != b) & (a >= 0) & (b >= 0)
        pairs = np.stack([a[differ], b[differ]], axis=1)
        pairs.sort(axis=1)
        keys, counts = np.unique(pairs, axis=0, return_counts=True)
        for (u, v), n in zip(keys.tolist(), counts.tolist(), strict=True):
            shared[(u, v)] = shared.get((u, v), 0) + n
    target = np.arange(count)
    settled = rooted.copy()
    changed = True
    while changed:
        changed = False
        for basin in range(count):
            if settled[basin]:
                continue
            best, most = -1, 0
            for (u, v), n in shared.items():
                other = v if u == basin else u if v == basin else -1
                if other >= 0 and settled[other] and n > most:
                    best, most = other, n
            if best >= 0:
                target[basin] = target[best]
                settled[basin] = True
                changed = True
    return target


def _split_component(xy: np.ndarray, h: np.ndarray, cell: float) -> np.ndarray:
    """A component's splats split among its tree tops by watershed on its canopy height.

    A component with fewer than two tops is one instance (a shrub, a lone tree, a building);
    unrooted basins rejoin their neighbours (``_merge_unrooted``). Labels are 0-based.
    """
    low = xy.min(axis=0)
    ij = np.floor((xy - low) / cell).astype(np.int64)
    shape = (int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1)
    chm = np.full(shape, -np.inf)
    np.maximum.at(chm, (ij[:, 0], ij[:, 1]), h)
    chm_finite = np.where(np.isfinite(chm), chm, np.nan)
    tops = _tree_tops(chm_finite, cell)
    count = int(tops.sum())
    if count < 2:
        return np.zeros(xy.shape[0], dtype=np.int64)
    markers = np.zeros(shape, dtype=np.int32)
    for k, (i, j) in enumerate(np.argwhere(tops)):
        markers[i, j] = k + 1
    top = float(np.nanmax(chm_finite))
    depth = np.where(
        np.isfinite(chm), (top - np.where(np.isfinite(chm), chm, 0.0)) / max(top, 1e-6), 1.0
    )
    surface = np.clip(depth * 65534, 0, 65535).astype(np.uint16)
    basins = ndimage.watershed_ift(surface, markers, structure=np.ones((3, 3), dtype=int))
    basins = basins.astype(np.int64) - 1
    basins[~np.isfinite(chm)] = -1
    label = np.maximum(basins[ij[:, 0], ij[:, 1]], 0)
    rooted = np.zeros(count, dtype=bool)
    rooted[np.unique(label[h < FAO_SHRUB_MIN_M])] = True
    return _merge_unrooted(basins, rooted)[label]


def _inside_hull(hull_xy: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Which plan points lie inside the convex hull of ``hull_xy``."""
    if hull_xy.shape[0] < 3:
        return np.zeros(xy.shape[0], dtype=bool)
    try:
        return Delaunay(hull_xy).find_simplex(xy) >= 0
    except Exception:  # noqa: BLE001 -- a degenerate (collinear) hull holds nothing
        return np.zeros(xy.shape[0], dtype=bool)


def _hull_polygon(xy: np.ndarray) -> list[list[float]]:
    if xy.shape[0] < 3:
        return [[round(float(x), 3), round(float(y), 3)] for x, y in xy]
    try:
        hull = ConvexHull(xy)
    except Exception:  # noqa: BLE001
        return []
    return [[round(float(xy[v, 0]), 3), round(float(xy[v, 1]), 3)] for v in hull.vertices]


def _stem_xy(xyz: np.ndarray, height: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Where an object meets the ground, in plan: the median of its lowest tenth."""
    h = height[members]
    lowest = members[h <= np.percentile(h, 10)]
    return np.median(xyz[lowest, :2], axis=0)


def _assemble(instances: list[Instance], xyz: np.ndarray, height: np.ndarray) -> list[Instance]:
    """Joins every object that does not reach the ground to the one that holds it up.

    A plant stands on the ground. A capture often does not show that: a trunk under a dense
    crown is sparse, and a crown's lobe can be separated from the rest by a gap wider than the
    link. So an object with no splat within a shrub's height of the ground (*unrooted*) joins

    1. a rooted object standing under it -- whose stem is inside its plan hull and whose own
       spread is narrower than its own (a trunk under a crown, never a crown over a wall
       wider than itself); or
    2. failing that, the object nearest to it (among each of its splats' nearest neighbours),
       when the gap is smaller than the piece itself: a lobe hangs from the crown it is closer
       to than its own size.

    Anything left unrooted stays as it is: a canopy seen only from above has no stems to find.
    """
    if len(instances) < 2:
        return instances
    groups = [inst.members for inst in instances]
    count = len(groups)
    rooted = np.asarray([bool((height[m] < FAO_SHRUB_MIN_M).any()) for m in groups])
    if rooted.all():
        return instances
    stems = np.asarray([_stem_xy(xyz, height, m) for m in groups])
    spreads = np.asarray(
        [
            2.0 * float(np.median(np.linalg.norm(xyz[m, :2] - stems[i], axis=1)))
            for i, m in enumerate(groups)
        ]
    )
    owner = np.full(xyz.shape[0], -1, dtype=np.int64)
    for i, m in enumerate(groups):
        owner[m] = i
    target = np.arange(count)

    def find(i: int) -> int:
        while target[i] != i:
            target[i] = target[target[i]]
            i = int(target[i])
        return i

    for u in sorted(np.flatnonzero(~rooted).tolist(), key=lambda i: -groups[i].size):
        xy = xyz[groups[u], :2]
        centre = xy.mean(axis=0)
        own = 2.0 * float(np.median(np.linalg.norm(xy - centre, axis=1)))
        candidates = np.flatnonzero(rooted & (spreads < own))
        if candidates.size == 0:
            continue
        under = candidates[_inside_hull(xy, stems[candidates])]
        if under.size:
            best = int(under[np.argmin(np.linalg.norm(stems[under] - centre, axis=1))])
            target[u] = find(best)
    everything = np.flatnonzero(owner >= 0)
    tree = cKDTree(xyz[everything])
    for u in sorted(np.flatnonzero(~rooted).tolist(), key=lambda i: groups[i].size):
        if find(u) != u:
            continue
        points = xyz[groups[u]]
        size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        # Evenly sampled when large: a sampled gap is never smaller than the true one, so a
        # piece joins less readily, never more.
        sample = points[:: max(1, points.shape[0] // GAP_SAMPLE)]
        distance, index = tree.query(sample, k=min(2 * LINK_NEIGHBOURS, everything.size))
        roots = np.asarray([find(i) for i in range(count)], dtype=np.int64)
        others = roots[owner[everything[index]]]
        foreign = (others != u) & np.isfinite(distance)
        if not foreign.any():
            continue
        flat = int(np.argmin(np.where(foreign, distance, np.inf)))
        gap = float(distance.reshape(-1)[flat])
        if gap < size:
            target[u] = int(others.reshape(-1)[flat])
    merged: dict[int, list[np.ndarray]] = {}
    for i in range(count):
        merged.setdefault(find(i), []).append(groups[i])
    return [Instance(members=np.sort(np.concatenate(parts))) for _, parts in sorted(merged.items())]


@dataclass
class SceneAnalysis:
    """What ``analyse`` found. Per-splat arrays are over the analysed (kept) splats."""

    ground: GroundModel
    spacing: float
    link: float
    height: np.ndarray  # above ground, metres
    green: np.ndarray  # bool
    exg_threshold: float
    exg_effectiveness: float
    vari_green_share: float
    klass: np.ndarray  # uint8, index into CLASSES
    plant: np.ndarray  # int32, index into plants, -1 for none
    plants: list[Instance]
    rejected: int
    timings: dict[str, float]

    def thresholds(self) -> dict:
        """Every number the classification used, with where it came from."""
        return {
            "spacingM": {
                "value": round(self.spacing, 5),
                "rule": "median nearest-neighbour distance of the analysed splats",
                "source": "skeleton.py neighbour_spacing: every radius a multiple of it",
            },
            "groundCellM": {
                "value": round(self.ground.cell, 5),
                "rule": f"plan cell holding {GROUND_CELL_POINTS} splats on average",
                "source": "splat_ground's min_points (tools/pipeline stages.py)",
            },
            "groundSlopeBound": {
                "value": round(self.ground.slope_bound, 5),
                "rule": "median + 3 sigma (MAD) of adjacent lowest-cell slopes",
                "source": "Vosselman 2000, Slope based filtering of laser altimetry data, IAPRS 33(B3)",
            },
            "groundLayerM": {
                "value": round(self.ground.layer_m, 5),
                "rule": "median + 3 sigma (MAD) of heights above the surface where it rests",
                "source": "robust 3-sigma clipping, floored at the spacing",
            },
            "exgThreshold": {
                "value": round(self.exg_threshold, 5),
                "effectiveness": round(self.exg_effectiveness, 4),
                "rule": "Otsu on ExG = 2g - r - b (chromatic) of this capture",
                "source": "Woebbecke et al. 1995 (ExG); Otsu 1979 (threshold)",
            },
            "variGreenShareOfExgGreen": round(self.vari_green_share, 4),
            "linkM": {
                "value": round(self.link, 5),
                "rule": f"{LINK_FACTOR} x the local spacing (median nearest-neighbour distance of "
                f"each splat's {LINK_NEIGHBOURS} neighbours), shown here at the median spacing",
                "source": "skeleton.py link_factor",
            },
            "shrubMinM": {"value": FAO_SHRUB_MIN_M, "source": "FAO FRA 2020 Terms and Definitions"},
            "treeMinM": {"value": FAO_TREE_MIN_M, "source": "FAO FRA 2020 Terms and Definitions"},
            "crownWidth": {
                "rule": "CW = 2.51503 + 0.00901 H^2 (m), window CW/2",
                "source": "Popescu & Wynne 2004, PE&RS 70(5):589-604 (mixed stands)",
            },
            "leafyFraction": {"value": LEAFY_FRACTION, "rule": "majority of splats green"},
            "snagSpread": {
                "value": SNAG_SPREAD_OF_CROWN,
                "rule": "not leafy, tree height, spread < 0.5 CW(H)",
            },
            "skeletonMinSplatsPerM": {
                "value": round(SKELETON_MIN_SPLATS_PER_M, 2),
                "rule": "12,000 / 4 splats over 6.4597 m: the thinnest cloud skeleton.py is tested on",
                "source": "tests/test_skeleton.py test_recovery_survives_a_quarter_density_cloud",
            },
        }


def analyse(
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> SceneAnalysis:
    """Ground, classes and plant instances of a metric, z-up splat. ``xyz`` float64 metres."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    count = int(xyz.shape[0])
    spacing = skeleton.neighbour_spacing(xyz)
    link = LINK_FACTOR * spacing
    ground = ground_model(xyz, spacing)
    height = xyz[:, 2] - ground.at(xyz[:, :2])
    timings["groundS"] = time.perf_counter() - started

    exg, vari = chromatic_indices(rgb)
    threshold, effectiveness = otsu_threshold(exg)
    green = exg > threshold
    vari_share = float(np.mean(vari[green] > 0)) if green.any() else 0.0

    klass = np.full(count, STATIC, dtype=np.uint8)
    plant = np.full(count, -1, dtype=np.int32)
    on_ground = height <= ground.layer_m
    klass[on_ground & green] = LOW
    klass[on_ground & ~green] = GROUND

    # Objects: everything clear of the ground layer, linked by the density where each splat
    # stands; an object is a component that reaches a shrub's height with enough splats to be
    # a cluster. The rest of what stands above the ground layer is low: grass if green.
    spacing_at = local_spacing(xyz)
    above = np.flatnonzero(height > ground.layer_m)
    component = np.full(count, -1, dtype=np.int64)
    component[above] = _components(xyz[above], spacing_at[above])
    timings["componentsS"] = time.perf_counter() - started
    tall = np.zeros(count, dtype=bool)
    tall[above] = height[above] >= FAO_SHRUB_MIN_M
    seeds_per = np.bincount(component[above], weights=tall[above].astype(np.float64))
    real = seeds_per >= MIN_CLUSTER_POINTS
    low_left = above[~real[component[above]]]
    klass[low_left[green[low_left]]] = LOW
    component[low_left] = -1
    instances: list[Instance] = []
    objects = np.flatnonzero(component >= 0)
    order = objects[np.argsort(component[objects], kind="stable")]
    labels = component[order]
    bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    for a, b in itertools.pairwise(bounds):
        members = order[a:b]
        split = _split_component(xyz[members, :2], height[members], ground.cell)
        for value in np.unique(split):
            chosen = members[split == value]
            if chosen.size >= MIN_CLUSTER_POINTS:
                instances.append(Instance(members=chosen))
    instances = _assemble(instances, xyz, height)
    timings["instancesS"] = time.perf_counter() - started

    plants: list[Instance] = []
    rejected = 0
    for instance in instances:
        members = instance.members
        h = height[members]
        # Its top: connected to the rest, so a splat up there is the plant's, not a floater's.
        top = float(h.max())
        instance.height_m = top
        instance.green_fraction = float(green[members].mean())
        # The stem: where the instance meets the ground -- its lowest tenth, in plan.
        lowest = members[h <= np.percentile(h, 10)]
        stem_xy = np.median(xyz[lowest, :2], axis=0)
        stem_z = float(ground.at(stem_xy[None, :])[0])
        instance.stem = np.asarray([stem_xy[0], stem_xy[1], stem_z])
        radial = np.linalg.norm(xyz[members, :2] - stem_xy, axis=1)
        leafy = instance.green_fraction > LEAFY_FRACTION
        if leafy:
            instance.klass = (
                TREE if top >= FAO_TREE_MIN_M else SHRUB if top >= FAO_SHRUB_MIN_M else LOW
            )
        else:
            # A bare trunk's spread about its own axis is its diameter.
            instance.spread_m = 2.0 * float(np.median(radial))
            narrow = instance.spread_m < SNAG_SPREAD_OF_CROWN * float(popescu_wynne_crown_m(top))
            instance.klass = SNAG if top >= FAO_TREE_MIN_M and narrow else STATIC
        if instance.klass == LOW:
            klass[members[green[members]]] = LOW
            rejected += 1
            continue
        if instance.klass == STATIC:
            rejected += 1
            continue
        if instance.klass in (TREE, SHRUB):
            # A plant owns what stands under its foliage: the crown's plan hull, from its
            # green splats. Anything of the object outside it -- a wall the crown touches --
            # stays still.
            leaves = members[green[members]]
            hull_xy = xyz[leaves, :2]
            inside = _inside_hull(hull_xy, xyz[members, :2]) | green[members]
            outside = members[~inside]
            instance.members = members[inside]
            instance.crown_polygon = _hull_polygon(hull_xy)
            instance.crown_radius_m = float(np.percentile(radial[inside], 95))
            instance.spread_m = 2.0 * float(np.median(radial[inside]))
            klass[outside] = STATIC
        else:
            instance.crown_polygon = _hull_polygon(xyz[members, :2])
            instance.crown_radius_m = float(np.percentile(radial, 95))
        instance.splats_per_m = instance.members.size / max(top, 1e-6)
        if instance.klass == SNAG:
            instance.rig = "trunk"
        elif instance.klass == TREE and instance.splats_per_m >= SKELETON_MIN_SPLATS_PER_M:
            instance.rig = "skeleton"
        else:
            instance.rig = "crown"
        plants.append(instance)
    # Deterministic order and ids: by class, then west to east, south to north.
    plants.sort(key=lambda p: (p.klass, round(float(p.stem[0]), 3), round(float(p.stem[1]), 3)))
    counters: dict[int, int] = {}
    for index, instance in enumerate(plants):
        n = counters.get(instance.klass, 0)
        counters[instance.klass] = n + 1
        instance.id = f"{CLASSES[instance.klass]}-{n}"
        klass[instance.members] = instance.klass
        plant[instance.members] = index
    timings["classesS"] = time.perf_counter() - started
    return SceneAnalysis(
        ground=ground,
        spacing=spacing,
        link=link,
        height=height,
        green=green,
        exg_threshold=threshold,
        exg_effectiveness=effectiveness,
        vari_green_share=vari_share,
        klass=klass,
        plant=plant,
        plants=plants,
        rejected=rejected,
        timings=timings,
    )


# -------------------------------------------------------------------------------------- rigs


def _node(position, parent: int, band: str) -> dict:
    return {
        "position": np.asarray(position, dtype=np.float64),
        "parent": parent,
        "band": band,
        # Not read by the motion model (motion_params.py reads no radius).
        "radius": 0.0,
    }


def _dedupe_chain(points: list[np.ndarray], minimum: float) -> list[np.ndarray]:
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) >= minimum:
            out.append(p)
    return out


def crown_rig_nodes(
    xyz: np.ndarray, green: np.ndarray, stem: np.ndarray, height_m: float, spacing: float
) -> list[dict]:
    """A few-bone rig for a plant too sparse (or too small) for the banded skeleton.

    A stem from the ground to the crown's top through its base and its centroid, and up to
    ``CROWN_SECTORS`` limbs -- one per quarter of the crown about the stem, each two joints,
    the inner and outer halves of that quarter's foliage -- hung from the stem joint nearest
    their height. The trunk is the whole-plant mode; each limb is a mode of its own where
    eq. 17 puts it inside the plant's band (``motion_params.limb_modes``).
    """
    h = xyz[:, 2] - stem[2]
    leaves = xyz[green] if green.any() else xyz
    leaf_h = leaves[:, 2] - stem[2]
    crown_base = float(np.percentile(leaf_h, 5)) if leaves.shape[0] else 0.0
    top_share = xyz[h >= np.percentile(h, 95)]
    top = np.asarray([*np.median(top_share[:, :2], axis=0), stem[2] + height_m])
    centroid = leaves.mean(axis=0)

    def on_axis(z: float) -> np.ndarray:
        f = (z - stem[2]) / max(top[2] - stem[2], 1e-6)
        return stem + (top - stem) * min(max(f, 0.0), 1.0)

    axis = _dedupe_chain(
        [stem, on_axis(stem[2] + crown_base), on_axis(centroid[2]), top], max(spacing, 1e-3)
    )
    nodes = [_node(axis[0], -1, "trunk")]
    for p in axis[1:]:
        nodes.append(_node(p, len(nodes) - 1, "trunk"))
    axis_z = np.asarray([n["position"][2] for n in nodes])
    # Sectors about the stem, starting at the crown's own principal direction so the split
    # does not depend on which way north is.
    offset = leaves[:, :2] - np.asarray([on_axis(z)[:2] for z in leaves[:, 2]])
    if offset.shape[0] >= 2:
        cov = np.cov(offset.T)
        _, vectors = np.linalg.eigh(cov)
        principal = math.atan2(vectors[1, -1], vectors[0, -1])
    else:
        principal = 0.0
    azimuth = np.mod(np.arctan2(offset[:, 1], offset[:, 0]) - principal, 2 * math.pi)
    radial = np.linalg.norm(offset, axis=1)
    sector = np.minimum(
        (azimuth / (2 * math.pi) * CROWN_SECTORS).astype(np.int64), CROWN_SECTORS - 1
    )
    for s in range(CROWN_SECTORS):
        mine = sector == s
        if int(mine.sum()) < 2 * MIN_CLUSTER_POINTS:
            continue
        r = radial[mine]
        split = np.median(r)
        inner = leaves[mine][r <= split].mean(axis=0)
        outer = leaves[mine][r > split]
        if outer.shape[0] == 0:
            continue
        outer_point = outer.mean(axis=0)
        below = np.flatnonzero(axis_z <= inner[2])
        attach = int(below[-1]) if below.size else 0
        nodes.append(_node(inner, attach, "branch"))
        nodes.append(_node(outer_point, len(nodes) - 1, "leaf"))
    return nodes


def trunk_rig_nodes(xyz: np.ndarray, stem: np.ndarray, height_m: float) -> list[dict]:
    """A bare trunk's axis: the ground, then the median of each quarter of its height."""
    h = xyz[:, 2] - stem[2]
    nodes = [_node(stem, -1, "trunk")]
    for k in range(1, 5):
        z0, z1 = (k - 0.5) * height_m / 4, min(k + 0.5, 4.0) * height_m / 4
        slab = xyz[(h >= z0) & (h <= z1)]
        xy = np.median(slab[:, :2], axis=0) if slab.shape[0] else stem[:2]
        nodes.append(_node([xy[0], xy[1], stem[2] + k * height_m / 4], len(nodes) - 1, "trunk"))
    return nodes


def rig_from_nodes(nodes: list[dict], checksum: str, note: str) -> dict:
    """skeleton.build_rig, whatever made the nodes."""
    return skeleton.build_rig(nodes, checksum, note)


def snag_bend_scale(diameter_m: float, height_m: float) -> float:
    """A snag's bend against a leafy tree's of the same height: its frontal area's share.

    Deflection is linear in drag (Jackson et al. 2021), drag linear in frontal area (the drag
    equation, equal drag coefficients assumed). A bare trunk presents ``D * H``; a live tree
    of the same height adds its crown, ``CW(H)`` wide (Popescu & Wynne 2004) over the upper
    half of its height (estimate): ``D / (D + CW / 2)``.
    """
    crown = float(popescu_wynne_crown_m(height_m))
    return diameter_m / (diameter_m + crown / 2.0) if diameter_m > 0 else 0.0


SCENE_PROVENANCE = {
    "sceneClasses": {
        "rule": "tree >= 5 m, shrub 0.5-5 m, leafy = majority green (Otsu on ExG), snag = tree height, not leafy, spread < CW(H)/2; everything else static",
        "source": "FAO FRA 2020 Terms and Definitions; Woebbecke et al. 1995; Otsu 1979; Popescu & Wynne 2004",
        "status": "cited thresholds, majority and snag rules chosen",
    },
    "shrubMotion": {
        "rule": "the limb model on a crown rig; f0 = 2.4/sqrt(H) extrapolated below 4.7 m; the same bend angles, so a short plant swings little; flutter at the tips",
        "source": "Jackson et al. 2021 (pendulum law, trees 4.7-55.7 m); de Langre 2019 (plant frequencies of order 1-10 Hz from wheat to trees)",
        "status": "estimate",
    },
    "snagMotion": {
        "rule": "the trunk's mode alone, its bend scaled by D / (D + CW(H)/2), no leaf flutter",
        "source": "deflection linear in drag (Jackson et al. 2021); drag linear in frontal area; crown width Popescu & Wynne 2004; the crown over the upper half of the height and equal drag coefficients are estimates",
        "status": "estimate",
    },
    "crownRig": {
        "rule": "stem through crown base and centroid to top, and one two-joint limb per quarter of the crown",
        "source": "few-bone rigs are the identifiable ones (Chen & Lou 2026); SpeedTree 1-2 branch levels",
        "status": "design choice",
    },
    "staticAnchor": {
        "rule": "every splat that is not a plant is bound to one node that never moves",
        "source": "canonical-never-written (docs/LIVING_SURVEY.md)",
        "status": "design choice",
    },
}


def leaf_size(
    xyz: np.ndarray, log_scales: np.ndarray, green: np.ndarray, plant: np.ndarray
) -> float:
    """Median foliage splat diameter over every leafy plant, floored at a leaf's 5 cm."""
    foliage = green & (plant >= 0)
    if not foliage.any():
        return motion_params.DEFAULT_LEAF_SIZE_M
    diameters = 2.0 * np.exp(log_scales[foliage].max(axis=1))
    estimate = motion_params.round_to(float(np.median(diameters)), 4)
    return max(estimate, motion_params.MIN_LEAF_SIZE_M)


@dataclass
class PlantRig:
    instance: Instance
    rig: dict
    sidecar: dict


def plant_rigs(
    analysis: SceneAnalysis, xyz: np.ndarray, log_scales: np.ndarray, *, seed: int = 1
) -> tuple[list[PlantRig], float]:
    """A rig and a sidecar per plant, each in the scene's own frame."""
    size = leaf_size(xyz, log_scales, analysis.green, analysis.plant)
    out: list[PlantRig] = []
    for instance in analysis.plants:
        members = instance.members
        pts = snap(xyz[members]).astype(np.float64)
        note = f"{instance.id} ({CLASSES[instance.klass]}, {instance.rig} rig)"
        if instance.rig == "skeleton":
            nodes = skeleton.extract_skeleton(pts, max_nodes=SKELETON_MAX_NODES)
            rig = skeleton.build_rig(nodes, "fnv1a32:0:00000000", note)
        elif instance.rig == "crown":
            nodes = crown_rig_nodes(
                pts, analysis.green[members], instance.stem, instance.height_m, analysis.spacing
            )
            rig = rig_from_nodes(nodes, "fnv1a32:0:00000000", note)
        else:
            nodes = trunk_rig_nodes(pts, instance.stem, instance.height_m)
            rig = rig_from_nodes(nodes, "fnv1a32:0:00000000", note)
        sidecar = motion_params.derive_sidecar(
            rig, tree_height_m=instance.height_m, leaf_size_m=size, seed=seed
        )
        if instance.klass == SNAG:
            scale = snag_bend_scale(instance.spread_m, instance.height_m)
            columns = sidecar["nodes"]
            columns["gainRad"] = [motion_params.round_to(g * scale, 7) for g in columns["gainRad"]]
            columns["flutterM"] = [0 for _ in columns["flutterM"]]
        out.append(PlantRig(instance, rig, sidecar))
    return out, size


def forest(
    plants: list[PlantRig], leaf_size_m: float, *, seed: int = 1, source_note: str = ""
) -> tuple[dict, dict]:
    """Every plant's rig in one rig with one root per plant and a static anchor at node 0."""
    nodes: list[dict] = [
        {
            "id": STATIC_NODE_ID,
            "parent": -1,
            "position": [0.0, 0.0, 0.0],
            "radius": 0.0,
            "stiffness": 1.0,
            "band": "trunk",
        }
    ]
    columns: dict[str, list] = {
        key: [value]
        for key, value in (
            ("branch", 0),
            ("mode", 0),
            ("share", 0),
            ("frequencyHz", 1.0),
            ("damping", motion_params.TREE_DAMPING_SUMMER),
            ("gainRad", 0),
            ("flutterM", 0),
        )
    }
    rig_plants: list[dict] = []
    sidecar_plants: list[dict] = []
    reference = max((p.instance.height_m for p in plants), default=1.0)
    for p in plants:
        start = len(nodes)
        for node in p.rig["nodes"]:
            nodes.append(
                {
                    **node,
                    "id": f"{p.instance.id}/{node['id']}",
                    "parent": -1 if node["parent"] < 0 else node["parent"] + start,
                }
            )
        for key, column in columns.items():
            values = p.sidecar["nodes"][key]
            if key == "branch":
                values = [v + start for v in values]
            column.extend(values)
        end = len(nodes)
        wind = motion_params.default_sidecar_wind(p.instance.height_m)
        rig_plants.append(
            {
                "id": p.instance.id,
                "class": CLASSES[p.instance.klass],
                "nodeStart": start,
                "nodeEnd": end,
            }
        )
        sidecar_plants.append(
            {
                "id": p.instance.id,
                "class": CLASSES[p.instance.klass],
                "rig": p.instance.rig,
                "nodeStart": start,
                "nodeEnd": end,
                "heightM": motion_params.round_to(p.instance.height_m, 4),
                "turbulence": wind["turbulence"],
                "lengthScaleM": wind["lengthScaleM"],
            }
        )
    if len(nodes) > 65535:
        raise SystemExit(f"{len(nodes)} rig nodes: more than a Uint16 can index")
    rig = {
        "units": "meters",
        "canonicalChecksum": "fnv1a32:0:00000000",
        "sourceNote": source_note
        or f"scene step: {len(plants)} plants, {len(nodes)} nodes (tools/captures/scene_plants.py)",
        "nodes": nodes,
        "plants": rig_plants,
    }
    sidecar = {
        "format": motion_params.MOTION_SIDECAR_FORMAT,
        "version": motion_params.MOTION_SIDECAR_VERSION,
        "motionEvidence": "allometric",
        "rigChecksum": rig["canonicalChecksum"],
        "nodeCount": len(nodes),
        "seed": seed,
        "treeHeightM": motion_params.round_to(reference, 4),
        "leafSizeM": leaf_size_m,
        "referenceSpeedMps": motion_params.REFERENCE_SPEED_MPS,
        "wind": motion_params.default_sidecar_wind(reference),
        "seasons": {
            "winter": {
                "dampingScale": motion_params.round_to(
                    motion_params.TREE_DAMPING_WINTER / motion_params.TREE_DAMPING_SUMMER, 4
                ),
                "branchFrequencyScale": motion_params.LEAFLESS_FREQUENCY_SCALE,
                "flutterScale": 0,
            }
        },
        "nodes": columns,
        "plants": sidecar_plants,
        "provenance": {**motion_params.ALLOMETRIC_PROVENANCE, **SCENE_PROVENANCE},
        "generator": "tools/captures/scene_plants.py",
    }
    return rig, sidecar


def forest_rig_issues(rig: dict) -> list[str]:
    """``validateRig``'s checks for a forest rig: one root per plant, and a static anchor."""
    issues: list[str] = []
    nodes = rig["nodes"]
    plants = rig.get("plants", [])
    owner = [-1] * len(nodes)
    for k, plant in enumerate(plants):
        start, end = plant["nodeStart"], plant["nodeEnd"]
        if not 0 < start < end <= len(nodes):
            issues.append(f"plant {k}: node range [{start}, {end}) is not inside the rig")
            continue
        for i in range(start, end):
            if owner[i] >= 0:
                issues.append(f"node {i} belongs to two plants")
            owner[i] = k
        if nodes[start]["parent"] != -1:
            issues.append(f"plant {k}: its first node must be its root")
    for i, node in enumerate(nodes):
        parent = node["parent"]
        if owner[i] < 0:
            if parent != -1:
                issues.append(f"node {i}: a node outside every plant must be an anchor")
            continue
        start = plants[owner[i]]["nodeStart"]
        if i != start and not start <= parent < i:
            issues.append(f"node {i}: parent {parent} is not an earlier node of its plant")
    single = {**rig, "nodes": [{**n, "parent": -1 if i == 0 else 0} for i, n in enumerate(nodes)]}
    issues += [i for i in skeleton.rig_issues(single) if "parent" not in i]
    return issues


# ---------------------------------------------------------------------------- tile binding


def _rle(values: np.ndarray) -> list[int]:
    """Run-length pairs ``[value, count, value, count, ...]``."""
    if values.size == 0:
        return []
    change = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    lengths = np.diff(np.r_[change, values.size])
    out = np.empty(change.size * 2, dtype=np.int64)
    out[0::2] = values[change]
    out[1::2] = lengths
    return [int(v) for v in out]


def tile_labels(
    ply: Path, labels_by_row: np.ndarray, opacity_min: float, tile_gaussians: int | None
) -> dict[str, np.ndarray]:
    """Per tile uri, the label of every gaussian in the tile's own order.

    Replays the packer's own plan (``splat_tiles.prepare``: the same filter, Morton order and
    tree) rather than matching positions: a leaf holds its originals in PLY order, so its
    labels are the rows'; a merged parent's gaussian is one occupied cell of its level, and it
    takes a plant's label only if **every** original merged into it is that plant's -- a cell
    that mixes a plant with anything else stays still.
    """
    out: dict[str, np.ndarray] = {}
    with (
        tempfile.TemporaryDirectory(prefix=".scene_plants-") as work,
        splat_tiles.prepare(ply, opacity_min, tile_gaussians, Path(work)) as tree,
    ):
        rows = np.empty(tree.count, dtype=np.int64)
        step = 1 << 20
        for start in range(0, tree.count, step):
            rows[start : start + step] = tree.store.read(start, min(start + step, tree.count))[
                "row"
            ]
        sorted_labels = labels_by_row[rows]
        for tile in tree.root.walk():
            if not tile.children:
                index = np.concatenate([np.arange(a, b) for a, b in tile.ranges])
                order = np.argsort(rows[index], kind="stable")
                out[tile.uri] = sorted_labels[index[order]].astype(np.int64)
                continue
            (start, stop) = tile.ranges[0]
            shift = np.uint64(3 * (splat_tiles.GRID_BITS - tile.level))
            keys = (tree.codes[start:stop] >> shift).astype(np.int64)
            values = sorted_labels[start:stop]
            first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
            low = np.minimum.reduceat(values, first)
            high = np.maximum.reduceat(values, first)
            out[tile.uri] = np.where(low == high, low, 0).astype(np.int64)
    return out


def plant_binding(
    tiles_dir: Path,
    ply: Path,
    labels_by_row: np.ndarray,
    *,
    opacity_min: float,
    tile_gaussians: int | None,
) -> tuple[dict, dict[str, str]]:
    """``plants.json``: per tile checksum, the plant (1-based; 0 = static) of each gaussian.

    Every tile's checksum is taken from its own GLB, as ``rig_tiles`` stamps the rig, and each
    tile's label count is checked against its gaussian count: tiles packed from another PLY,
    or with other parameters, are refused rather than bound wrongly.
    """
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    uris = rig_tiles.tile_uris(tileset)
    labels = tile_labels(ply, labels_by_row, opacity_min, tile_gaussians)
    if sorted(labels) != sorted(uris):
        raise SystemExit(
            "the tileset's tiles are not the ones this PLY packs to with these parameters "
            f"({len(uris)} tiles in tileset.json, {len(labels)} planned)"
        )
    tiles: dict[str, list[int]] = {}
    checksums: dict[str, str] = {}
    for uri in uris:
        positions = rig_tiles.tile_positions(tiles_dir / uri)
        if positions.shape[0] != labels[uri].size:
            raise SystemExit(
                f"{uri} holds {positions.shape[0]} gaussians, the plan {labels[uri].size}: "
                "the tiles were not packed from this PLY with these parameters"
            )
        checksum = checksum_positions(positions)
        checksums[uri] = checksum
        tiles[checksum] = _rle(labels[uri])
    document = {
        "format": PLANTS_FORMAT,
        "version": PLANTS_VERSION,
        "encoding": "rle",
        "note": "per tile checksum, run-length pairs [plant, count, ...] in the tile's own "
        "gaussian order; plant 0 is static, k is rig.plants[k - 1]",
        "tiles": dict(sorted(tiles.items())),
    }
    return document, checksums


# ------------------------------------------------------------------------------- scoring


def score(
    analysis: SceneAnalysis,
    truth_class: np.ndarray,
    truth_instance: np.ndarray,
    truth_instances: list[dict],
) -> dict:
    """Per-class IoU over splats, instance counts and height errors against a known scene."""
    iou = {}
    for k, name in enumerate(CLASSES):
        predicted = analysis.klass == k
        actual = truth_class == k
        union = int((predicted | actual).sum())
        iou[name] = round(int((predicted & actual).sum()) / union, 4) if union else None
    matched = []
    for plant in analysis.plants:
        votes = truth_instance[plant.members]
        votes = votes[votes >= 0]
        if votes.size == 0:
            matched.append({"id": plant.id, "truth": None})
            continue
        best = int(np.bincount(votes).argmax())
        truth = truth_instances[best]
        matched.append(
            {
                "id": plant.id,
                "truth": truth["id"],
                "classMatches": CLASSES[plant.klass] == truth["class"],
                "heightErrorM": round(plant.height_m - float(truth["heightM"]), 3),
                "stemErrorM": round(
                    float(np.linalg.norm(plant.stem[:2] - np.asarray(truth["stem"][:2]))), 3
                ),
                "rig": plant.rig,
            }
        )
    counts = {
        name: {
            "found": sum(1 for p in analysis.plants if p.klass == k),
            "truth": sum(1 for t in truth_instances if t["class"] == name),
        }
        for k, name in enumerate(CLASSES)
        if k in PLANT_CLASSES
    }
    errors = [abs(m["heightErrorM"]) for m in matched if m.get("truth")]
    return {
        "classIoU": iou,
        "meanPlantIoU": round(
            float(np.mean([iou[CLASSES[k]] for k in PLANT_CLASSES if iou[CLASSES[k]] is not None])),
            4,
        ),
        "instances": counts,
        "matched": matched,
        "heightErrorMaxM": round(max(errors), 3) if errors else None,
        "heightErrorMeanM": round(float(np.mean(errors)), 3) if errors else None,
    }


# ------------------------------------------------------------------------------ the build


#: The PLY columns the scene step reads: position, colour, size. Nothing of the higher-order
#: spherical harmonics, which are most of a trained PLY's bytes.
CAPTURE_COLUMNS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2")


def read_capture(ply: Path, opacity_min: float) -> dict[str, np.ndarray]:
    """The columns the step reads, every row, in windows; and ``keep``: the rows the packer
    keeps (``splat_tiles._filter_rows``: finite, opaque enough, not a far floater)."""
    layout = splat_tiles.ply_layout(ply)
    keep, _ = splat_tiles._filter_rows(layout, opacity_min)
    data = {name: np.empty(layout.count, dtype=np.float32) for name in CAPTURE_COLUMNS}
    for start, chunk in splat_tiles.iter_ply_rows(layout, CAPTURE_COLUMNS):
        for name in CAPTURE_COLUMNS:
            data[name][start : start + chunk[name].size] = chunk[name]
    data["keep"] = keep
    return data


def build(
    ply: Path,
    out_dir: Path,
    *,
    tiles_dir: Path | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    height: float = 0.0,
    opacity_min: float = PACKAGE_OPACITY_MIN,
    tile_gaussians: int | None = PACKAGE_TILE_GAUSSIANS,
    seed: int = 1,
    truth: dict | None = None,
) -> dict:
    """Analyse, rig, bind and write: rig.json, motion.json, plants.json, scene.json beside the tiles."""
    started = time.perf_counter()
    if tiles_dir is None:
        tiles_dir = out_dir
        splat_tiles.convert(
            ply, tiles_dir, lat, lon, height, opacity_min=opacity_min, tile_gaussians=tile_gaussians
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_capture(ply, opacity_min)
    keep = data["keep"]
    rows = np.flatnonzero(keep)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)[keep].astype(np.float64)
    rgb = np.clip(
        0.5 + splat_tiles.SH_C0 * np.stack([data[f"f_dc_{k}"] for k in range(3)], axis=1)[keep],
        0.0,
        1.0,
    )
    log_scales = np.stack([data[f"scale_{k}"] for k in range(3)], axis=1)[keep].astype(np.float64)
    analysis = analyse(xyz, rgb)
    rigs, size = plant_rigs(analysis, xyz, log_scales, seed=seed)
    rig, sidecar = forest(
        rigs,
        size,
        seed=seed,
        source_note=(
            f"scene step on {ply.name}: {len(rigs)} plants "
            f"({sum(1 for p in rigs if p.instance.klass == TREE)} trees, "
            f"{sum(1 for p in rigs if p.instance.klass == SHRUB)} shrubs, "
            f"{sum(1 for p in rigs if p.instance.klass == SNAG)} snags) over "
            f"{xyz.shape[0]:,} splats (tools/captures/scene_plants.py)"
        ),
    )
    labels_by_row = np.zeros(int(keep.size), dtype=np.int64)
    labels_by_row[rows] = analysis.plant.astype(np.int64) + 1
    binding, checksums = plant_binding(
        tiles_dir, ply, labels_by_row, opacity_min=opacity_min, tile_gaussians=tile_gaussians
    )
    rig = rig_tiles.stamp(rig, tiles_dir)
    rig["canonicalChecksum"] = checksums["splat.glb"]
    rig["motion"] = "motion.json"
    rig["binding"] = "plants.json"
    sidecar["rigChecksum"] = rig["canonicalChecksum"]
    issues = forest_rig_issues(rig)
    if issues:
        raise SystemExit("forest rig is invalid:\n  " + "\n  ".join(issues))
    ordered_rig = {
        key: rig[key]
        for key in (
            "units",
            "canonicalChecksum",
            "sourceNote",
            "tileChecksums",
            "nodes",
            "motion",
            "plants",
            "binding",
        )
    }

    def dump(name: str, document: dict) -> int:
        text = json.dumps(document, separators=(",", ":")) + "\n"
        (out_dir / name).write_text(text, encoding="utf-8")
        return len(text)

    sizes = {
        "rig.json": dump("rig.json", ordered_rig),
        "motion.json": dump("motion.json", sidecar),
        "plants.json": dump("plants.json", binding),
    }
    (out_dir / "ground.f32").write_bytes(analysis.ground.heights.astype("<f4").tobytes())
    classes_by_row = np.full(int(keep.size), STATIC, dtype=np.uint8)
    classes_by_row[rows] = analysis.klass
    (out_dir / "classes.u8").write_bytes(classes_by_row.tobytes())
    class_counts = {name: int((analysis.klass == k).sum()) for k, name in enumerate(CLASSES)}
    report: dict = {
        "format": "hexapod.scene",
        "version": 1,
        "generator": "tools/captures/scene_plants.py",
        "splats": int(keep.size),
        "analysedSplats": int(rows.size),
        "classes": list(CLASSES),
        "classCounts": class_counts,
        "ground": {
            **analysis.ground.to_json(),
            "file": "ground.f32",
            "dtype": "<f4",
            "order": "C, x then y",
        },
        "classFile": {"file": "classes.u8", "note": "class index per PLY row; dropped rows static"},
        "thresholds": analysis.thresholds(),
        "leafSizeM": size,
        "plants": [
            {
                "id": p.instance.id,
                "class": CLASSES[p.instance.klass],
                "rig": p.instance.rig,
                "stem": [round(float(v), 3) for v in p.instance.stem],
                "heightM": round(p.instance.height_m, 3),
                "crownRadiusM": round(p.instance.crown_radius_m, 3),
                "crownPolygon": p.instance.crown_polygon,
                "greenFraction": round(p.instance.green_fraction, 4),
                "spreadM": round(p.instance.spread_m, 3),
                "splats": int(p.instance.members.size),
                "splatsPerM": round(p.instance.splats_per_m, 1),
                "nodes": len(p.rig["nodes"]),
                "f0Hz": p.sidecar["nodes"]["frequencyHz"][0],
            }
            for p in rigs
        ],
        "rejectedObjects": analysis.rejected,
        "rig": {
            "nodes": len(rig["nodes"]),
            "plants": len(rig["plants"]),
            "tiles": len(rig["tileChecksums"]),
            "bytes": sizes,
        },
    }
    if truth is not None:
        truth_class = np.asarray(truth["class"], dtype=np.int64)[rows]
        truth_instance = np.asarray(truth["instance"], dtype=np.int64)[rows]
        report["score"] = score(analysis, truth_class, truth_instance, truth["instances"])
    (out_dir / "scene.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    # Wall-clock is reported, never written: two runs over the same bytes write the same files.
    report["timingsS"] = {
        **{k: round(v, 3) for k, v in analysis.timings.items()},
        "totalS": round(time.perf_counter() - started, 3),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ply", type=Path, help="canonical.ply: metric, east/north/up")
    parser.add_argument(
        "--tiles",
        type=Path,
        default=None,
        help="the tileset packed from it (splat/); re-packed when absent",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: the tiles directory")
    parser.add_argument("--lat", type=float, default=0.0, help="only when re-packing")
    parser.add_argument("--lon", type=float, default=0.0, help="only when re-packing")
    parser.add_argument("--height", type=float, default=0.0, help="only when re-packing")
    parser.add_argument("--opacity-min", type=float, default=PACKAGE_OPACITY_MIN)
    parser.add_argument("--tile-gaussians", type=int, default=PACKAGE_TILE_GAUSSIANS)
    parser.add_argument("--truth", type=Path, default=None, help="labels.json of a synthetic scene")
    args = parser.parse_args()
    out = args.out or args.tiles
    if out is None:
        parser.error("--out is required when --tiles is not given")
    truth = json.loads(args.truth.read_text(encoding="utf-8")) if args.truth else None
    report = build(
        args.ply,
        out,
        tiles_dir=args.tiles,
        lat=args.lat,
        lon=args.lon,
        height=args.height,
        opacity_min=args.opacity_min,
        tile_gaussians=args.tile_gaussians,
        truth=truth,
    )
    summary = {k: report[k] for k in ("classCounts", "rig", "timingsS")}
    summary["plants"] = [(p["id"], p["rig"], p["heightM"], p["nodes"]) for p in report["plants"]]
    if "score" in report:
        summary["score"] = {
            k: report["score"][k] for k in ("classIoU", "instances", "heightErrorMaxM")
        }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
