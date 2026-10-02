"""Scene objects from 2D masks: lift class-free masks to 3D splat instances with a hierarchy.

docs/SCENE_OBJECTS.md §3-4. A scan is rendered from many viewpoints; a mask model cuts each
view into class-free masks at a few scales; every mask votes for the splats that own its
pixels; splats that keep landing in the same mask, across the views where both were seen,
become one instance. Coarse scales give parents, fine scales children. An image-text model
then embeds each instance's best views, and tags, attribute scores and a behaviour follow
from that embedding. Nothing here knows a class: the models come in through two Protocols
(`MaskSource`, `Embedder`), implemented for real by `segment_models.py` (SAM 2, SigLIP) and
here by test doubles (`OracleMasks`, `FakeEmbedder`).

**Cells.** Splats are first bucketed into small voxel cells (`supervoxels`), and everything
after works on cells, so the graph is bounded by `MAX_CELLS` whatever the scan's size (22.6M
gaussians on 15 GB) and a splat's instance is its cell's. Each view is rendered with the
cell id as the label (`splat_render.render(labels=...)`): per pixel the dominant cell and its
purity, which weights its votes.

**Votes.** In one view, a cell is *in* a mask when at least `MASK_SHARE` of its visible
(purity-weighted) pixels are; overlapping masks of one level go to the one holding the larger
share. A cell is *visible* in a view with at least `MIN_VISIBLE_PX` of weight.

**Instances, per mask level.** Two cells belong together when, over the views where both are
visible and at least one is in a mask of that level, they are in the *same* mask in at least
`MERGE_RATIO` of them (co-occurrence over co-visibility), and in at least `MIN_COVISIBLE`
views. Stage 1 asks it of spatial neighbours only (each cell's `NEIGHBOURS` nearest within
`NEIGHBOUR_CELLS` cell edges), so the graph is sparse; its components are fragments. Stage 2
asks it again of fragments, now of any two that ever shared a mask, wherever they are -- which
rejoins an object that occlusion or a gap split, without a quadratic pass over cells.
Fragments below `MIN_INSTANCE_SPLATS` are dropped. Cells no view ever saw (inside a crown,
under a roof) take their nearest seen cell's labels within `FILL_CELLS` edges; a seen cell in
no mask keeps 0.

**Hierarchy.** Levels are the mask model's scale hints (0 = coarsest). The instances of
level L are refined by those of L+1: a child is the cells of a parent that share one level
L+1 instance, so every child's splats are inside its parent by construction. A child that
would be its whole parent is not made, nor one below `MIN_INSTANCE_SPLATS`; a splat whose
cell has no finer instance keeps its parent's id. `level` in the output is depth in this
tree. Splat ids are the deepest instance (leaf-level), as the contract says.

**Meaning.** Per instance, crops of the `CROP_VIEWS` views where its splats cover most pixels
(bounding box padded `CROP_PAD`) are embedded and averaged (`embedding`, L2-normalised).
`tags`: softmax over the whole vocabulary of `LOGIT_SCALE` x cosine, top `TAGS_TOP_K`.
`properties`: per attribute, a two-way softmax of `LOGIT_SCALE` x cosine between a positive
and a contrast prompt (`PROPERTY_PROMPTS`), so each is a probability on its own and the
attributes do not compete. `behaviour`: `BEHAVIOUR_RULE`.

**Scale** (2026-10-01, 4 CPUs, other jobs running): the 22.6M-gaussian camp with 8 views
and random two-level masks -- 998k cells at 0.24 m, 5.7M stage-1 edges; cells 36 s, plan
19 s, render 52 s per view, votes 4 s, lift 46 s, describe 9 s; peak RSS 7.5 GB (the scan
held as float64 `Splats` is most of it). Rendering dominates: 24 views is ~21 min.

Usage:
    python segment_scene.py SPLAT.ply TILES_DIR --masks segment_models:Sam2Masks \\
        --embedder segment_models:SiglipEmbedder --vocabulary words.txt \\
        [--views 24] [--save-dir views/] [--render-instances instances.png]
    python segment_scene.py yard/source/splat.ply yard/splat --truth yard/source/labels.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import scene_plants
import splat_tiles
from splat_render import Camera, Splats, _from_columns, render

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "Instance",
    "Lifted",
    "Mask",
    "MaskSource",
    "OracleMasks",
    "View",
    "describe",
    "footprint_anchors",
    "instances_document",
    "lift",
    "link_instances",
    "load_embedder",
    "load_masks",
    "load_source",
    "plan_views",
    "render_instances",
    "render_views",
    "segment",
    "supervoxels",
    "tile_binding",
    "view_footprint",
    "write_instances",
]

FORMAT = "hexapod.instances"
VERSION = 1
FRAME = "tileset local ENU (the root transform's frame), metres"
TILES_ENCODING = "rle; a merged parent splat takes an id only if all its children share it"

#: Views rendered by default: half on a ring around the scan, half from where it was seen.
VIEW_COUNT = 24
#: View size and field of view: large enough for a mask model's smallest useful masks.
VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV_DEG = 512, 384, 60.0
#: Ring views alternate between these elevations (degrees above the horizon).
RING_ELEVATIONS_DEG = (30.0, 55.0)
#: Observer views look this far below the horizon, at golden-angle yaws.
OBSERVER_PITCH_DEG = 15.0
#: Observer views try this many yaws and take the one facing the most of the scan.
YAW_CANDIDATES = 16
#: An observer eye needs this much free space (no splat centre nearer), metres.
OBSERVER_CLEARANCE_M = 0.3
#: Local views (scans wider than one view): the frame's width at its target holds
#: `VIEW_WIDTH / CELL_PX` cells, so a cell spans about `CELL_PX` pixels.
CELL_PX = 4.0
#: Local view targets: a grid over the scan's footprint, this many per footprint width.
ANCHORS_PER_FOOTPRINT = 3
#: A grid bin holds the scan with at least this share of the median occupied bin's sample.
FOOTPRINT_DENSITY = 0.1
#: Oblique views per target, alternating between these elevations (degrees).
OBLIQUE_VIEWS = 4
OBLIQUE_ELEVATIONS_DEG = (35.0, 55.0, 20.0, 75.0)
#: ... at these factors of the distance where the frame spans one footprint.
OBLIQUE_DISTANCES = (1.0, 0.7, 0.5, 1.4)
#: Line of sight: `OBLIQUE_PROBES` points from the eye to `OBLIQUE_PROBE_FROM` of the way
#: back towards it from the target; a probe is clear with no sampled splat within
#: `OBLIQUE_CLEARANCE` of a footprint. The first candidate with `OBLIQUE_CLEAR` of its
#: probes clear is taken, else the clearest if at least `OBLIQUE_MIN_CLEAR`, else none.
OBLIQUE_PROBES = 12
OBLIQUE_PROBE_FROM = 0.25
OBLIQUE_CLEARANCE = 0.03
OBLIQUE_CLEAR = 0.9
OBLIQUE_MIN_CLEAR = 0.6
#: Extra eye-height observer views per target.
EYE_VIEWS = 2
#: At most this many views in all (rings, observers, local).
MAX_VIEWS = 480
#: Splats sampled for the scan's extent and the observers' clearance test (seeded).
PLAN_SAMPLE = 400_000

#: The smallest voxel cell edge tried, metres; it grows by `CELL_GROWTH` until the scan has
#: at most `MAX_CELLS` cells and at least `SPLATS_PER_CELL` splats per cell on average.
CELL_M = 0.05
CELL_GROWTH = 1.25
MAX_CELLS = 1_000_000
SPLATS_PER_CELL = 2.0

#: A cell is visible in a view with at least this much purity-weighted pixel area.
MIN_VISIBLE_PX = 1.0
#: Pixels a cell owns by less than this share of their coverage do not vote: at an
#: object's silhouette the dominant cell is often one of a crowd that belongs elsewhere.
MIN_PURITY = 0.5
#: A cell is in a mask when at least this share of its visible weight is.
MASK_SHARE = 0.5
#: Stage 1: neighbouring cells join outright when in the same mask in this share of views.
STRICT_RATIO = 0.9
#: Stage 2's far pairs: the largest regions of each mask, at most this many.
SHARED_REGIONS = 48
#: Stage 2 stops after this many rounds of mutual-best joins.
MAX_ROUNDS = 200
#: Two regions join when in the same mask in at least this share of the views
#: where both were seen and either was in a mask of the level ...
MERGE_RATIO = 0.6
#: ... and in at least this many such views.
MIN_COVISIBLE = 2
#: Stage 1's graph: each cell's nearest cells, within this many cell edges.
NEIGHBOURS = 10
NEIGHBOUR_CELLS = 3.0
#: An instance needs at least this many splats.
MIN_INSTANCE_SPLATS = 30
#: Cells no view saw take the nearest seen cell's labels within this many cell edges.
FILL_CELLS = 3.0

#: Views an instance is counted as seen in need this many of its pixels.
MIN_VIEW_PX = 16
#: Crops embedded per instance: its best views by pixel area.
CROP_VIEWS = 3
#: Crop padding, a share of the box's size each side, and the smallest crop side, pixels.
CROP_PAD = 0.15
MIN_CROP_PX = 24
#: CLIP's logit scale: softmax(LOGIT_SCALE * cosine) for tags and properties.
LOGIT_SCALE = 100.0
TAGS_TOP_K = 5

#: The attribute prompts of SCENE_OBJECTS.md §3 step 5: per property, a positive prompt and
#: a contrast prompt. Attributes, not classes.
PROPERTY_PROMPTS: dict[str, tuple[str, str]] = {
    "movable": ("a photo of an object that can be moved", "a photo of something fixed in place"),
    "rigid": ("a photo of a rigid, hard object", "a photo of something soft or flexible"),
    "elastic": (
        "a photo of something flexible that bends or sways",
        "a photo of something stiff and rigid",
    ),
    "static": (
        "a photo of a static structure, such as a building, a wall or the ground",
        "a photo of a loose object",
    ),
    "vegetation": ("a photo of vegetation, a plant or a tree", "a photo of something not a plant"),
    "water": ("a photo of water", "a photo of dry land or a dry object"),
    "vehicle": ("a photo of a vehicle", "a photo of something that is not a vehicle"),
    "creature": ("a photo of a person or an animal", "a photo of an inanimate thing"),
}
#: Behaviour from the property scores, first rule that holds.
BEHAVIOUR_RULE = (
    "movable if vehicle >= 0.5 or creature >= 0.5 or (movable >= 0.5 and movable >= static); "
    "else in-place if vegetation >= 0.5 or water >= 0.5 or elastic >= 0.5; else static"
)
#: The PLY columns a scan is read with.
SOURCE_COLUMNS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip


# --------------------------------------------------------------------------- the models


@dataclass
class Mask:
    """One class-free mask of a view: which pixels, at which scale, how sure."""

    #: (h, w) bool.
    mask: np.ndarray
    #: Scale hint, 0 = coarsest (SAM's whole / part / subpart, or area bins).
    level: int
    score: float = 1.0


@runtime_checkable
class MaskSource(Protocol):
    """Class-free automatic masks of an image at several scales (SAM 2 family)."""

    #: Model id, for reports.
    name: str

    def masks(self, rgb: np.ndarray) -> list[Mask]:
        """`rgb` (h, w, 3) uint8 -> masks of every scale the model gives."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """An image-text model with one embedding space (SigLIP / CLIP family).

    Both methods return (n, dim) float32 arrays, each row L2-normalised; `name` is the model
    id written to `instances.json`."""

    name: str
    dim: int

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """uint8 (h, w, 3) crops of any size -> (n, dim)."""
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Prompts -> (n, dim)."""
        ...


def _normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.float64)
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.where(norm > 0, norm, 1.0)


class FakeEmbedder:
    """A deterministic stand-in: an image is its coarse colour histogram, a text a hash.

    Both are projected by fixed seeded matrices into `dim` dimensions. It means nothing; it
    lets the pipeline and the files be tested without a model."""

    name = "fake-colour-histogram"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._project = np.random.default_rng(20261001).standard_normal((64, dim))

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images), self.dim))
        for k, image in enumerate(images):
            q = (np.asarray(image, np.uint8).reshape(-1, 3) // 64).astype(np.int64)
            hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64)
            out[k] = hist / max(int(hist.sum()), 1) @ self._project
        return _normalise(out).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = [
            np.random.default_rng(
                int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            ).standard_normal(self.dim)
            for t in texts
        ]
        return _normalise(np.asarray(rows).reshape(len(texts), self.dim)).astype(np.float32)


def _load(spec: str) -> object:
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"{spec!r}: expected 'module:Class'")
    return getattr(importlib.import_module(module), name)()


def load_masks(spec: str) -> MaskSource:
    """`module:Class`, constructed with no arguments (as `teacher_fill.make_filler`)."""
    source = _load(spec)
    if not isinstance(source, MaskSource):
        raise TypeError(f"{spec} is not a MaskSource (name, masks(rgb))")
    return source


def load_embedder(spec: str) -> Embedder:
    """`module:Class`, constructed with no arguments."""
    embedder = _load(spec)
    if not isinstance(embedder, Embedder):
        raise TypeError(f"{spec} is not an Embedder (name, dim, embed_images, embed_texts)")
    return embedder


# ------------------------------------------------------------------------------ the scan


def load_source(ply: Path, opacity_min: float) -> tuple[Splats, np.ndarray, int]:
    """The rows the packer keeps (`splat_tiles._filter_rows`), as splats; their PLY rows; and
    the PLY's row count. Only the degree-0 columns are read, in windows."""
    layout = splat_tiles.ply_layout(ply)
    keep, _ = splat_tiles._filter_rows(layout, opacity_min)
    rows = np.flatnonzero(keep)
    columns = {name: np.empty(rows.size, np.float32) for name in SOURCE_COLUMNS}
    held = 0
    for start, chunk in splat_tiles.iter_ply_rows(layout, SOURCE_COLUMNS):
        kept = keep[start : start + chunk["x"].size]
        n = int(kept.sum())
        for name in SOURCE_COLUMNS:
            columns[name][held : held + n] = chunk[name][kept]
        held += n
    return _from_columns(columns), rows, int(layout.count)


def supervoxels(
    positions: np.ndarray,
    *,
    edge_m: float = CELL_M,
    max_cells: int = MAX_CELLS,
    splats_per_cell: float = SPLATS_PER_CELL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per splat its cell (int32, 0..k-1), each cell's centroid (k, 3) and splat count, and
    the cell edge used: the first of `edge_m` x `CELL_GROWTH`^i with at most
    `min(max_cells, n / splats_per_cell)` occupied cells."""
    pos = np.asarray(positions)
    lo = pos.min(axis=0).astype(np.float64)
    span = pos.max(axis=0).astype(np.float64) - lo
    target = max(1, min(max_cells, int(len(pos) / splats_per_cell)))
    edge = edge_m
    while True:
        dims = np.floor(span / edge).astype(np.int64) + 1
        key = np.zeros(len(pos), np.int64)
        for axis in (2, 1, 0):
            q = np.floor((pos[:, axis] - lo[axis]) / edge).astype(np.int64)
            key = key * dims[axis] + np.clip(q, 0, dims[axis] - 1)
        unique, inverse = np.unique(key, return_inverse=True)
        del key
        if unique.size <= target:
            break
        edge *= CELL_GROWTH
    cell = inverse.astype(np.int32).reshape(-1)
    counts = np.bincount(cell, minlength=unique.size)
    centroids = (
        np.stack(
            [np.bincount(cell, pos[:, k].astype(np.float64), unique.size) for k in range(3)], axis=1
        )
        / counts[:, None]
    )
    return cell, centroids, counts, float(edge)


# ---------------------------------------------------------------------------- the views


def _extent(positions: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A seeded sample of the scan, and its robust (1st-99th percentile) box."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(positions)
    sample = pos if len(pos) <= PLAN_SAMPLE else pos[rng.choice(len(pos), PLAN_SAMPLE, False)]
    sample = np.asarray(sample, np.float64)
    return sample, np.percentile(sample, 1, axis=0), np.percentile(sample, 99, axis=0)


def observer_points(splats: Splats) -> np.ndarray:
    """Where the scan was seen from: `view_cones`' observers (the finest-detail region, the
    capture's own cameras standing in), for a scan held in memory."""
    import view_cones as vc

    step = 1 << 20

    def chunks():
        for start in range(0, len(splats), step):
            stop = min(start + step, len(splats))
            yield (
                np.asarray(splats.positions[start:stop], np.float64),
                vc._middle_axis(np.log(np.maximum(splats.scales[start:stop], 1e-12))),
            )

    return vc.cone_grid_from_chunks(chunks(), chunks(), len(splats)).observers


def _farthest(points: np.ndarray, count: int, start: int) -> list[int]:
    """`count` indices of `points` spread by farthest-point sampling from `start`."""
    chosen = [start]
    gaps = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(count, len(points)):
        chosen.append(int(gaps.argmax()))
        gaps = np.minimum(gaps, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return chosen


def view_footprint(edge: float, width: int = VIEW_WIDTH) -> float:
    """A local view's footprint (the frame's width at its target), metres: `CELL_PX`
    pixels per cell, so the cells it votes for are resolved."""
    return width * edge / CELL_PX


def footprint_anchors(
    sample: np.ndarray, lo: np.ndarray, hi: np.ndarray, step: float
) -> np.ndarray:
    """Targets for local views: the centres of a `step` grid over the robust box where the
    scan is (bins with at least `FOOTPRINT_DENSITY` of the median occupied bin's sample),
    each at its bin's ground (5th percentile of height)."""
    inside = np.all((sample >= lo) & (sample <= hi), axis=1)
    pts = sample[inside]
    if len(pts) == 0:
        return np.zeros((0, 3))
    dims = np.maximum(np.ceil((hi[:2] - lo[:2]) / step).astype(np.int64), 1)
    ij = np.minimum(np.floor((pts[:, :2] - lo[:2]) / step).astype(np.int64), dims - 1)
    key = ij[:, 0] * dims[1] + ij[:, 1]
    keys, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    dense = counts >= FOOTPRINT_DENSITY * np.median(counts)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(pts[order, 2], np.cumsum(counts)[:-1])
    out = []
    for k in np.flatnonzero(dense):
        i, j = divmod(int(keys[k]), int(dims[1]))
        ground = float(np.percentile(groups[k], 5))
        out.append([lo[0] + (i + 0.5) * step, lo[1] + (j + 0.5) * step, ground])
    return np.asarray(out, np.float64).reshape(-1, 3)


def plan_views(
    positions: np.ndarray,
    count: int = VIEW_COUNT,
    *,
    observers: np.ndarray | None = None,
    edge: float | None = None,
    solid: np.ndarray | None = None,
    max_views: int = MAX_VIEWS,
    width: int = VIEW_WIDTH,
    height: int = VIEW_HEIGHT,
    fov_deg: float = VIEW_FOV_DEG,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    eye_height_m: float = 1.6,
) -> list[Camera]:
    """Cameras: `count` views of the whole scan (rings and observer-near views), plus, for a
    scan wider than one view's footprint, local views that scale with its area.

    The rings (`count - count // 2` views, or all of them with no observers) see every side
    at alternating `RING_ELEVATIONS_DEG`: half from far enough out that the scan's robust
    box fills the frame, half from half as far at the half of the scan facing them -- two
    scales, so small things are not specks. Observer views stand `eye_height_m` above
    observer points with `OBSERVER_CLEARANCE_M` of free space, spread by farthest-point
    sampling, `OBSERVER_PITCH_DEG` down, each facing the most of the scan within reach
    (`YAW_CANDIDATES`) -- the close views of what a walk-in capture saw best.

    **Local views** (given the cell `edge`): a view's footprint is `view_footprint(edge)`
    (a cell spans `CELL_PX` pixels). When the robust box is wider than that, the footprint is
    gridded `ANCHORS_PER_FOOTPRINT` times per footprint width (`footprint_anchors`), and
    each anchor gets `OBLIQUE_VIEWS` obliques at the distance where the frame spans one
    footprint (azimuths turned by the golden angle per anchor; the elevation and distance
    are the first of `OBLIQUE_ELEVATIONS_DEG` x `OBLIQUE_DISTANCES` with a clear line of
    sight to the target past `solid` (points that block it: the occupied cells' centres;
    default the sample) -- under a canopy the view goes low, in the open high) and
    `EYE_VIEWS` more observer views; every local and observer view then has a far plane a
    footprint beyond its target. At most `max_views` in all (anchors spread by farthest
    points). Nothing in it knows the scene; small scans get the plain `count` views."""
    sample, lo, hi = _extent(positions)
    centre = (lo + hi) / 2
    up_a = np.asarray(up, np.float64)
    radius = max(0.5 * float(np.linalg.norm((hi - lo)[:2])), 1e-3)
    half_fov = math.radians(fov_deg) / 2
    anchors = np.zeros((0, 3))
    footprint = math.inf
    if edge is not None:
        footprint = view_footprint(edge, width)
        if float(np.max(hi[:2] - lo[:2])) > footprint:
            anchors = footprint_anchors(sample, lo, hi, footprint / ANCHORS_PER_FOOTPRINT)
            per_anchor = OBLIQUE_VIEWS + EYE_VIEWS
            room = max(0, (max_views - count) // per_anchor)
            if len(anchors) > room:
                start = int(np.argmin(np.linalg.norm(anchors[:, :2] - centre[:2], axis=1)))
                anchors = anchors[np.sort(_farthest(anchors, room, start))]
        if not len(anchors):
            footprint = math.inf
    local = len(anchors) > 0
    eyes: list[tuple[np.ndarray, np.ndarray, float]] = []
    near: list[np.ndarray] = []
    tree = cKDTree(sample)
    if observers is not None and len(observers) and count > 1:
        pool = np.asarray(observers, np.float64).reshape(-1, 3) + up_a * eye_height_m
        clearance = tree.query(pool, k=1)[0]
        pool = pool[clearance >= OBSERVER_CLEARANCE_M]
        if len(pool):
            start = int(np.argmin(np.linalg.norm(pool - centre, axis=1)))
            wanted = count // 2 + EYE_VIEWS * len(anchors)
            near = [pool[k] for k in _farthest(pool, wanted, start)]
    ring = count - min(len(near), count // 2)
    outer = (ring + 1) // 2
    distance = radius / math.tan(half_fov)
    for k in range(ring):
        # Two scales: the whole scan from the outer ring, a half of it from half as far.
        inner = k >= outer
        n_ring = ring - outer if inner else outer
        j = k - outer if inner else k
        azimuth = 2 * math.pi * (j + (0.5 if inner else 0.0)) / max(n_ring, 1)
        elevation = math.radians(RING_ELEVATIONS_DEG[j % len(RING_ELEVATIONS_DEG)])
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        target = centre.copy()
        if inner:
            target[:2] += 0.5 * radius * direction[:2] / max(math.cos(elevation), 1e-9)
        eyes.append((target + distance * (0.5 if inner else 1.0) * direction, target, math.inf))
    if local:
        reach = footprint / 2 / math.tan(half_fov)
        golden = math.pi * (3 - math.sqrt(5))
        sight = tree if solid is None else cKDTree(np.asarray(solid, np.float64))
        probes = np.linspace(OBLIQUE_PROBE_FROM, 1.0, OBLIQUE_PROBES)
        n_elev = len(OBLIQUE_ELEVATIONS_DEG)
        for i, target in enumerate(anchors):
            for j in range(OBLIQUE_VIEWS):
                azimuth = 2 * math.pi * j / OBLIQUE_VIEWS + i * golden
                # Candidates in order of preference: elevations from this view's turn in
                # the cycle, each at the distances in `OBLIQUE_DISTANCES`.
                candidates = []
                for e in range(n_elev):
                    elevation = math.radians(OBLIQUE_ELEVATIONS_DEG[(i + j + e) % n_elev])
                    direction = np.array(
                        [
                            math.cos(elevation) * math.cos(azimuth),
                            math.cos(elevation) * math.sin(azimuth),
                            math.sin(elevation),
                        ]
                    )
                    candidates += [reach * f * direction for f in OBLIQUE_DISTANCES]
                offsets = np.asarray(candidates)
                # Line of sight: probes from the eye towards the target, each clear when no
                # sampled splat is within `OBLIQUE_CLEARANCE` of a footprint.
                points = target + offsets[:, None, :] * probes[None, ::-1, None]
                gap = sight.query(points.reshape(-1, 3), k=1)[0].reshape(len(offsets), -1)
                clear = (gap >= OBLIQUE_CLEARANCE * footprint).mean(axis=1)
                good = np.flatnonzero(clear >= OBLIQUE_CLEAR)
                pick = int(good[0]) if good.size else int(np.argmax(clear))
                if clear[pick] >= OBLIQUE_MIN_CLEAR:
                    eye = target + offsets[pick]
                    eyes.append((eye, target, float(np.linalg.norm(offsets[pick])) + footprint))
    pitch = math.radians(OBSERVER_PITCH_DEG)
    yaws = 2 * math.pi * np.arange(YAW_CANDIDATES) / YAW_CANDIDATES
    within = min(radius, footprint)
    for eye in near:
        # Look where the scan is: the yaw whose wedge holds the most of it within reach.
        offset = sample[:, :2] - eye[:2]
        reach_m = np.linalg.norm(offset, axis=1)
        bearing = np.arctan2(offset[:, 1], offset[:, 0])
        close = reach_m <= within
        held = [
            int(np.sum(close & (np.abs(np.angle(np.exp(1j * (bearing - y)))) <= math.pi / 6)))
            for y in yaws
        ]
        yaw = float(yaws[int(np.argmax(held))])
        look = np.array(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
        )
        eyes.append((eye, eye + look, footprint))
    return [
        Camera.look_at(eye, target, fov_deg=fov_deg, width=width, height=height, up=up, far=far)
        for eye, target, far in eyes
    ]


@dataclass
class View:
    """One rendered view: the image a mask model sees, and per pixel which cell owns it."""

    camera: Camera
    #: (h, w, 3) uint8.
    rgb: np.ndarray
    #: (h, w) int32: the dominant cell (-1: nothing).
    cell: np.ndarray
    #: (h, w) float32: that cell's share of the pixel's coverage.
    purity: np.ndarray


def render_view(splats: Splats, camera: Camera, cells: np.ndarray) -> View:
    """The view the pipeline and `OracleMasks` both see (same renderer, same seed)."""
    frame = render(splats, camera, labels=cells)
    return View(
        camera,
        np.round(frame.rgb * 255).astype(np.uint8),
        frame.label.astype(np.int32),
        frame.purity.astype(np.float32),
    )


def render_views(splats: Splats, cameras: Sequence[Camera], cells: np.ndarray) -> list[View]:
    return [render_view(splats, camera, cells) for camera in cameras]


# ------------------------------------------------------------------------- the oracle


def _image_key(rgb: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(rgb, np.uint8).tobytes()).hexdigest()


class OracleMasks:
    """Masks from ground truth, for tests: per level, each connected region of one true label.

    The Protocol hands a mask model only pixels, as a real one gets, so the oracle renders
    the same cameras itself with the true labels and recognises a view by its image. Labels
    below 0 are background. Regions under `min_area` pixels are left out, as a mask model
    leaves out specks."""

    name = "oracle"

    def __init__(
        self,
        splats: Splats,
        levels: Sequence[np.ndarray],
        cameras: Sequence[Camera],
        *,
        min_area: int = 20,
        min_purity: float = 0.5,
        connected: bool = True,
    ) -> None:
        import cv2

        self._masks: dict[str, list[Mask]] = {}
        for camera in cameras:
            found: list[Mask] = []
            rgb = None
            for level, labels in enumerate(levels):
                frame = render(splats, camera, labels=np.asarray(labels, np.int64))
                rgb = np.round(frame.rgb * 255).astype(np.uint8)
                image = np.where(frame.purity >= min_purity, frame.label, -1).reshape(-1)
                h, w = frame.label.shape
                pixels = np.flatnonzero(image >= 0)
                if pixels.size == 0:
                    continue
                # Group pixels by label once; each label is then cut within its own box.
                order = pixels[np.argsort(image[pixels], kind="stable")]
                values = image[order]
                starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
                for group in np.split(order, starts[1:]):
                    if group.size < min_area:
                        continue
                    ys, xs = np.divmod(group, w)
                    y0, x0 = int(ys.min()), int(xs.min())
                    box = np.zeros((int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1), np.uint8)
                    box[ys - y0, xs - x0] = 1
                    if connected:
                        n, parts, stats, _ = cv2.connectedComponentsWithStats(box, connectivity=8)
                        areas = stats[:, cv2.CC_STAT_AREA]
                    else:
                        n, parts, areas = 2, box.astype(np.int32), np.array([0, group.size])
                    for k in np.flatnonzero(areas[1:n] >= min_area) + 1:
                        region = np.zeros((h, w), bool)
                        region[y0 : y0 + box.shape[0], x0 : x0 + box.shape[1]] = parts == k
                        found.append(Mask(region, level, 1.0))
            if rgb is not None:
                self._masks[_image_key(rgb)] = found

    def masks(self, rgb: np.ndarray) -> list[Mask]:
        return list(self._masks.get(_image_key(rgb), []))


# ---------------------------------------------------------------------------- lifting


@dataclass
class _Votes:
    """One view's votes: the cells visible in it, their weight and their mask per level."""

    cells: np.ndarray  # (m,) int32
    weight: np.ndarray  # (m,) float32
    masks: np.ndarray  # (levels, m) int32, -1: in no mask of that level


def vote(view: View, masks: Sequence[Mask], n_cells: int, levels: int) -> _Votes:
    """Which mask of each level every visible cell of `view` is in (`MASK_SHARE`)."""
    owner = view.cell.reshape(-1)
    weight = view.purity.reshape(-1).astype(np.float64)
    good = (owner >= 0) & (weight >= MIN_PURITY)
    seen = np.bincount(owner[good], weight[good], n_cells)
    visible = np.flatnonzero(seen >= MIN_VISIBLE_PX)
    best = np.full((levels, n_cells), -1, np.int32)
    best_share = np.zeros((levels, n_cells))
    best_score = np.zeros((levels, n_cells))
    for k, mask in enumerate(masks):
        inside = good & np.asarray(mask.mask, bool).reshape(-1)
        share = np.bincount(owner[inside], weight[inside], n_cells)
        share = np.divide(share, seen, out=np.zeros(n_cells), where=seen > 0)
        level = mask.level
        better = (share >= MASK_SHARE) & (
            (share > best_share[level])
            | ((share == best_share[level]) & (mask.score > best_score[level]))
        )
        best[level, better] = k
        best_share[level, better] = share[better]
        best_score[level, better] = mask.score
    return _Votes(visible.astype(np.int32), seen[visible].astype(np.float32), best[:, visible])


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1].astype(np.int64)


def _cell_graph(centroids: np.ndarray, edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1's edges: each cell's `NEIGHBOURS` nearest within `NEIGHBOUR_CELLS` edges."""
    k = min(NEIGHBOURS + 1, len(centroids))
    if k < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    distance, index = cKDTree(centroids).query(
        centroids, k=k, distance_upper_bound=NEIGHBOUR_CELLS * edge, workers=-1
    )
    a = np.repeat(np.arange(len(centroids)), k - 1)
    b = index[:, 1:].reshape(-1)
    ok = np.isfinite(distance[:, 1:].reshape(-1))
    a, b = np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])
    pairs = np.unique(a * len(centroids) + b)
    return pairs // len(centroids), pairs % len(centroids)


def _region_masks(
    votes: Sequence[_Votes], level: int, region: np.ndarray, n: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per view: which regions are visible, and the mask of this level holding `MASK_SHARE`
    of each one's visible weight (-1: none). A region is judged as a whole, so a few cells
    that bleed into a neighbour's mask do not carry it with them."""
    out = []
    for v in votes:
        reg = region[v.cells]
        ok = reg >= 0
        reg, weight, mask = reg[ok], v.weight[ok].astype(np.float64), v.masks[level][ok]
        visible = np.bincount(reg, weight, n)
        best = np.full(n, -1, np.int32)
        assigned = mask >= 0
        if assigned.any():
            n_masks = int(mask.max()) + 1
            keys, inverse = np.unique(
                reg[assigned].astype(np.int64) * n_masks + mask[assigned], return_inverse=True
            )
            sums = np.bincount(inverse, weight[assigned])
            kr, km = keys // n_masks, keys % n_masks
            order = np.lexsort((km, -sums, kr))
            first = order[np.r_[True, kr[order][1:] != kr[order][:-1]]]
            strong = sums[first] >= MASK_SHARE * visible[kr[first]]
            best[kr[first][strong]] = km[first][strong]
        out.append((visible >= MIN_VISIBLE_PX, best))
    return out


def _agreement(
    per_view: Sequence[tuple[np.ndarray, np.ndarray]], a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: views in the same mask, and views where both were seen and either masked."""
    together = np.zeros(a.size, np.int32)
    both = np.zeros(a.size, np.int32)
    for visible, best in per_view:
        informative = visible[a] & visible[b] & ((best[a] >= 0) | (best[b] >= 0))
        both += informative
        together += informative & (best[a] == best[b])
    return together, both


def _pairs(region: np.ndarray, a: np.ndarray, b: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    ra, rb = region[a], region[b]
    ok = (ra >= 0) & (rb >= 0) & (ra != rb)
    key = np.unique(np.minimum(ra[ok], rb[ok]) * n + np.maximum(ra[ok], rb[ok]))
    return key // n, key % n


def _relabel(region: np.ndarray) -> np.ndarray:
    out = np.full(region.shape, -1, np.int64)
    ok = region >= 0
    out[ok] = np.unique(region[ok], return_inverse=True)[1]
    return out


def _shared(
    per_view: Sequence[tuple[np.ndarray, np.ndarray]],
    region: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of regions that were in one mask in some view, wherever they are: the
    `SHARED_REGIONS` largest of each mask (so a mask over thousands of specks stays cheap)."""
    size = np.bincount(region[region >= 0], weights[region >= 0], n)
    keys: list[np.ndarray] = []
    for visible, best in per_view:
        members = np.flatnonzero(visible & (best >= 0))
        if members.size < 2:
            continue
        order = members[np.lexsort((members, -size[members], best[members]))]
        for group in np.split(order, np.flatnonzero(np.diff(best[order])) + 1):
            group = np.sort(group[:SHARED_REGIONS])
            if group.size > 1:
                i, j = np.triu_indices(group.size, 1)
                keys.append(group[i] * n + group[j])
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = np.unique(np.concatenate(keys))
    return key // n, key % n


def _joins(
    per_view: Sequence[tuple[np.ndarray, np.ndarray]],
    pa: np.ndarray,
    pb: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One round: the regions that join, and every region's best partner (-1: none).

    A pair qualifies with `MERGE_RATIO` agreement over at least `MIN_COVISIBLE` views. Each
    region's best partner has the most agreement (with a pseudo-view against small samples),
    then the most evidence, then the lower id. Sinks: of each mutual pair, the larger. A
    region joins its best partner only if that partner is a sink, which itself stays put:
    every join is one step, so many small regions can join a large one in a round and none
    chains through another."""
    n = size.size
    best = np.full(n, -1, np.int64)
    if pa.size == 0:
        return np.zeros(0, np.int64), best
    together, both = _agreement(per_view, pa, pb)
    ok = (both >= MIN_COVISIBLE) & (together >= MERGE_RATIO * both)
    if not ok.any():
        return np.zeros(0, np.int64), best
    score = together[ok] / (both[ok] + 1.0)
    src = np.concatenate([pa[ok], pb[ok]])
    dst = np.concatenate([pb[ok], pa[ok]])
    order = np.lexsort((dst, -np.tile(both[ok], 2), -np.tile(score, 2), src))
    first = order[np.r_[True, src[order][1:] != src[order][:-1]]]
    best[src[first]] = dst[first]
    mutual = np.flatnonzero((best >= 0) & (best[np.maximum(best, 0)] == np.arange(n)))
    partner = best[mutual]
    larger = (size[mutual] > size[partner]) | ((size[mutual] == size[partner]) & (mutual < partner))
    sink = np.zeros(n, bool)
    sink[mutual[larger]] = True
    movers = np.flatnonzero(~sink & (best >= 0))
    return movers[sink[best[movers]]], best


def _grow(
    votes: Sequence[_Votes],
    level: int,
    in_mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Per cell its region at this level (-1: in no mask), and the rounds it took.

    Stage 1: neighbour cells that share a mask in at least `STRICT_RATIO` of their
    informative views join (single linkage, so strict). Stage 2, in rounds: regions are
    re-judged as wholes (`_region_masks`), every neighbouring pair is scored, and each region
    joins its best partner when that partner is a sink (`_joins`) -- average evidence, never
    a chain through one ambiguous cell. When neighbours are done, pairs that shared a mask
    anywhere are scored too (`_shared`): an object's parts that a gap or another part
    separates in space. Up to `MAX_ROUNDS`. `weights`: splats per cell."""
    n_cells = in_mask.size
    region = np.where(in_mask, np.arange(n_cells), -1)
    ca, cb = _pairs(region, a, b, n_cells)
    together, both = _agreement(_region_masks(votes, level, region, n_cells), ca, cb)
    keep = (both >= MIN_COVISIBLE) & (together >= STRICT_RATIO * both)
    region = np.where(in_mask, _components(n_cells, ca[keep], cb[keep]), -1)
    region = _relabel(region)
    rounds = 0
    far = False
    while rounds < MAX_ROUNDS:
        n = int(region.max()) + 1
        if n < 2:
            break
        per_view = _region_masks(votes, level, region, n)
        pa, pb = _pairs(region, a, b, n)
        if far:
            fa, fb = _shared(per_view, region, weights, n)
            key = np.unique(np.concatenate([pa * n + pb, fa * n + fb]))
            pa, pb = key // n, key % n
        movers, best = _joins(
            per_view, pa, pb, np.bincount(region[region >= 0], weights[region >= 0], n)
        )
        rounds += 1
        if movers.size == 0:
            if far:
                break
            far = True  # neighbours are done; now regions that share masks anywhere
            continue
        joined = _components(n, movers, best[movers])
        region = _relabel(np.where(region >= 0, joined[np.maximum(region, 0)], -1))
    return region, rounds


@dataclass
class Lifted:
    """Instances over cells: the hierarchy, each cell's deepest instance, and its views."""

    #: Per cell its leaf instance id (1-based; 0: none).
    cell_id: np.ndarray
    #: Per instance (index id - 1): parent id or 0, and depth (0 = coarsest).
    parent: np.ndarray
    level: np.ndarray
    #: How many cells each mask level gave before the hierarchy, for reports.
    stats: dict[str, object] = field(default_factory=dict)


def lift(
    votes: Sequence[_Votes],
    centroids: np.ndarray,
    cell_counts: np.ndarray,
    edge: float,
    levels: int,
) -> Lifted:
    """Instances from the views' votes (module docstring: stages 1-2, fill, hierarchy)."""
    n_cells = len(centroids)
    a, b = _cell_graph(centroids, edge)
    seen = np.zeros(n_cells, bool)
    for v in votes:
        seen[v.cells] = True
    labels = np.zeros((levels, n_cells), np.int64)
    stats: dict[str, object] = {
        "cells": n_cells,
        "edges": int(a.size),
        "seenCells": int(seen.sum()),
    }
    in_mask = np.zeros((levels, n_cells), bool)
    for v in votes:
        for level in range(levels):
            in_mask[level, v.cells[v.masks[level] >= 0]] = True
    for level in range(levels):
        joined, rounds = _grow(votes, level, in_mask[level], a, b, cell_counts)
        stats[f"level{level}Rounds"] = rounds
        valid = joined >= 0
        size = np.bincount(joined[valid], cell_counts[valid])
        joined[valid & (size[np.maximum(joined, 0)] < MIN_INSTANCE_SPLATS)] = -1
        labels[level] = joined + 1
        stats[f"level{level}Instances"] = int(np.unique(joined[joined >= 0]).size)
    unseen = np.flatnonzero(~seen)
    if unseen.size and seen.any():
        seen_index = np.flatnonzero(seen)
        distance, nearest = cKDTree(centroids[seen_index]).query(
            centroids[unseen], k=1, distance_upper_bound=FILL_CELLS * edge, workers=-1
        )
        ok = np.isfinite(distance)
        labels[:, unseen[ok]] = labels[:, seen_index[nearest[ok]]]
        stats["filledCells"] = int(ok.sum())
    return _hierarchy(labels, cell_counts, stats)


def _hierarchy(labels: np.ndarray, cell_counts: np.ndarray, stats: dict[str, object]) -> Lifted:
    """Refine coarse instances by finer ones; number them breadth first, largest first."""
    n_cells = labels.shape[1]
    current = np.zeros(n_cells, np.int64)  # node id (1-based), 0: none
    parents: list[int] = []
    depths: list[int] = []
    totals: list[int] = []
    for level in range(labels.shape[0]):
        has = labels[level] > 0
        if not has.any():
            continue
        key = current[has] * (int(labels[level].max()) + 1) + labels[level][has]
        keys, inverse = np.unique(key, return_inverse=True)
        sizes = np.bincount(inverse, cell_counts[has])
        node_of_key = np.zeros(keys.size, np.int64)
        for k in range(keys.size):
            parent = int(keys[k] // (int(labels[level].max()) + 1))
            if sizes[k] < MIN_INSTANCE_SPLATS:
                node_of_key[k] = parent
            elif parent and sizes[k] >= totals[parent - 1]:
                node_of_key[k] = parent  # the whole parent: not a new instance
            else:
                parents.append(parent)
                depths.append(depths[parent - 1] + 1 if parent else 0)
                totals.append(int(sizes[k]))
                node_of_key[k] = len(parents)
        current[has] = node_of_key[inverse]
    # Renumber: by depth, then parent's new id, then size descending, then first cell.
    count = len(parents)
    first_cell = np.full(count + 1, n_cells, np.int64)
    np.minimum.at(first_cell, current, np.arange(n_cells))
    new_id = np.zeros(count + 1, np.int64)
    parent_arr = np.asarray(parents, np.int64)
    depth_arr = np.asarray(depths, np.int64)
    total_arr = np.asarray(totals, np.int64)
    next_id = 1
    for depth in range(int(depth_arr.max()) + 1 if count else 0):
        nodes = np.flatnonzero(depth_arr == depth) + 1
        order = np.lexsort(
            (first_cell[nodes], -total_arr[nodes - 1], new_id[parent_arr[nodes - 1]])
        )
        for node in nodes[order]:
            new_id[node] = next_id
            next_id += 1
    final_parent = np.zeros(count, np.int64)
    final_level = np.zeros(count, np.int64)
    for node in range(1, count + 1):
        final_parent[new_id[node] - 1] = new_id[parent_arr[node - 1]]
        final_level[new_id[node] - 1] = depth_arr[node - 1]
    stats["instances"] = count
    return Lifted(new_id[current], final_parent, final_level, stats)


# ----------------------------------------------------------------------------- meaning


@dataclass
class Instance:
    id: int
    parent: int | None
    level: int
    splats: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    centroid: np.ndarray
    views: int
    embedding: np.ndarray
    tags: list[dict[str, object]]
    properties: dict[str, float]
    behaviour: str


def _ancestors(parent: np.ndarray) -> list[list[int]]:
    """Per instance (index), itself and every ancestor, as 0-based indices."""
    chains: list[list[int]] = []
    for k in range(parent.size):
        chain = [k]
        while parent[chain[-1]]:
            chain.append(int(parent[chain[-1]]) - 1)
        chains.append(chain)
    return chains


def _up(values: np.ndarray, parent: np.ndarray, level: np.ndarray, how: str) -> np.ndarray:
    """Leaf values (instances along axis 0) folded into every ancestor, deepest first."""
    out = values.copy()
    for k in np.argsort(-level, kind="stable"):
        p = int(parent[k])
        if p:
            if how == "sum":
                out[p - 1] += out[k]
            elif how == "min":
                out[p - 1] = np.minimum(out[p - 1], out[k])
            else:
                out[p - 1] = np.maximum(out[p - 1], out[k])
    return out


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def behaviour(properties: dict[str, float]) -> str:
    """`BEHAVIOUR_RULE`."""
    p = properties
    if (
        p["vehicle"] >= 0.5
        or p["creature"] >= 0.5
        or (p["movable"] >= 0.5 and p["movable"] >= p["static"])
    ):
        return "movable"
    if p["vegetation"] >= 0.5 or p["water"] >= 0.5 or p["elastic"] >= 0.5:
        return "in-place"
    return "static"


def describe(
    lifted: Lifted,
    splats: Splats,
    cell: np.ndarray,
    views: Sequence[View],
    embedder: Embedder,
    vocabulary: Sequence[str],
) -> list[Instance]:
    """Per instance: bounds, centroid and splat counts, views, crops embedded, tags,
    properties and behaviour. Bounds and centroid cover the instance with its children;
    `splats` counts only the splats that carry its id (the contract's leaf level)."""
    n = lifted.parent.size
    if n == 0:
        return []
    splat_id = lifted.cell_id[cell]
    own = np.bincount(splat_id, minlength=n + 1)[1:]
    labelled = splat_id > 0
    index = splat_id[labelled] - 1
    pos = splats.positions[labelled]
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    total = np.zeros((n, 3))
    for k in range(3):
        np.minimum.at(lo[:, k], index, pos[:, k])
        np.maximum.at(hi[:, k], index, pos[:, k])
        total[:, k] = np.bincount(index, pos[:, k], n)
    del pos, index
    lo = _up(lo, lifted.parent, lifted.level, "min")
    hi = _up(hi, lifted.parent, lifted.level, "max")
    total = _up(total, lifted.parent, lifted.level, "sum")
    count = _up(own.astype(np.float64), lifted.parent, lifted.level, "sum")
    centroid = total / np.maximum(count, 1)[:, None]

    # Pixel area and box of every instance (with its children) in every view.
    area = np.zeros((len(views), n))
    boxes = np.zeros((len(views), n, 4))  # x0, y0, x1, y1
    for v, view in enumerate(views):
        h, w = view.cell.shape
        owner = view.cell.reshape(-1)
        pid = np.where(owner >= 0, lifted.cell_id[np.maximum(owner, 0)], 0)
        has = pid > 0
        idx = pid[has] - 1
        ys, xs = np.divmod(np.flatnonzero(has), w)
        a = np.bincount(idx, minlength=n).astype(np.float64)
        box = np.stack(
            [np.full(n, np.inf), np.full(n, np.inf), np.full(n, -1.0), np.full(n, -1.0)], 1
        )
        np.minimum.at(box[:, 0], idx, xs)
        np.minimum.at(box[:, 1], idx, ys)
        np.maximum.at(box[:, 2], idx, xs)
        np.maximum.at(box[:, 3], idx, ys)
        area[v] = _up(a, lifted.parent, lifted.level, "sum")
        box[:, :2] = _up(box[:, :2], lifted.parent, lifted.level, "min")
        box[:, 2:] = _up(box[:, 2:], lifted.parent, lifted.level, "max")
        boxes[v] = box
    seen_in = (area >= MIN_VIEW_PX).sum(axis=0)

    crops: list[np.ndarray] = []
    owner_of_crop: list[int] = []
    for k in range(n):
        best = [v for v in np.argsort(-area[:, k], kind="stable")[:CROP_VIEWS] if area[v, k] > 0]
        for v in best:
            image = views[v].rgb
            h, w = image.shape[:2]
            x0, y0, x1, y1 = boxes[v, k]
            pad_x = max(CROP_PAD * (x1 - x0 + 1), (MIN_CROP_PX - (x1 - x0 + 1)) / 2, 0)
            pad_y = max(CROP_PAD * (y1 - y0 + 1), (MIN_CROP_PX - (y1 - y0 + 1)) / 2, 0)
            xa, xb = int(max(0, x0 - pad_x)), int(min(w, x1 + 1 + pad_x))
            ya, yb = int(max(0, y0 - pad_y)), int(min(h, y1 + 1 + pad_y))
            crops.append(np.ascontiguousarray(image[ya:yb, xa:xb]))
            owner_of_crop.append(k)
    dim = int(embedder.dim)
    embedding = np.zeros((n, dim))
    if crops:
        rows = np.asarray(embedder.embed_images(crops), np.float64).reshape(len(crops), dim)
        np.add.at(embedding, np.asarray(owner_of_crop), rows)
    embedding = _normalise(embedding)

    words = list(vocabulary)
    names = list(PROPERTY_PROMPTS)
    tag_scores = np.zeros((n, 0))
    # An embedder that scores for itself (segment_models.ZeroShotScoring) uses its own prompts.
    own_tags = getattr(embedder, "score_tags", None)
    own_properties = getattr(embedder, "score_properties", None)
    if words:
        if own_tags is not None:
            tag_scores = np.asarray(own_tags(embedding, words), np.float64)
        else:
            text = np.asarray(embedder.embed_texts(words), np.float64).reshape(len(words), dim)
            tag_scores = _softmax(LOGIT_SCALE * embedding @ text.T)
    if own_properties is not None:
        scored = own_properties(embedding)
        property_scores = np.stack([np.asarray(scored[name], np.float64) for name in names], 1)
    else:
        prompts = [p for name in names for p in PROPERTY_PROMPTS[name]]
        prompt_rows = np.asarray(embedder.embed_texts(prompts), np.float64).reshape(
            len(prompts), dim
        )
        logits = LOGIT_SCALE * (embedding @ prompt_rows.T).reshape(n, len(names), 2)
        property_scores = _softmax(logits)[:, :, 0]

    instances: list[Instance] = []
    for k in range(n):
        has_embedding = bool(np.any(embedding[k]))
        tags: list[dict[str, object]] = []
        if words and has_embedding:
            top = np.argsort(-tag_scores[k], kind="stable")[:TAGS_TOP_K]
            tags = [{"label": words[t], "score": round(float(tag_scores[k, t]), 4)} for t in top]
        properties = {
            name: round(float(property_scores[k, j]) if has_embedding else 0.0, 4)
            for j, name in enumerate(names)
        }
        instances.append(
            Instance(
                id=k + 1,
                parent=int(lifted.parent[k]) or None,
                level=int(lifted.level[k]),
                splats=int(own[k]),
                bounds_min=lo[k],
                bounds_max=hi[k],
                centroid=centroid[k],
                views=int(seen_in[k]),
                embedding=embedding[k],
                tags=tags,
                properties=properties,
                behaviour=behaviour(properties),
            )
        )
    return instances


# ------------------------------------------------------------------------------ writers


def tile_binding(
    tiles_dir: Path,
    ply: Path,
    rows: np.ndarray,
    splat_id: np.ndarray,
    row_count: int,
    *,
    opacity_min: float,
    tile_gaussians: int | None,
) -> dict[str, list[int]]:
    """Per tile checksum, RLE pairs [id, count, ...] in the tile's own gaussian order -- the
    same keys and encoding as `plants.json` (`scene_plants.plant_binding`, which refuses
    tiles not packed from this PLY with these parameters)."""
    labels_by_row = np.zeros(row_count, np.int64)
    labels_by_row[rows] = splat_id
    document, _ = scene_plants.plant_binding(
        tiles_dir, ply, labels_by_row, opacity_min=opacity_min, tile_gaussians=tile_gaussians
    )
    return document["tiles"]


#: Neighbours a tile's gaussian consults when bound by position: its id only if all agree.
BIND_NEIGHBOURS = 8


def tile_binding_by_position(
    tiles_dir: Path, positions: np.ndarray, splat_id: np.ndarray
) -> dict[str, list[int]]:
    """`tile_binding` for a scan whose source PLY is not at hand: the segmented splats are
    the tileset's own leaves (`splat_render.load_tileset`), in the tiles' frame, so every
    tile's gaussian takes the id of the segmented splat at its position. A leaf gaussian is
    one of them exactly; a merged parent's gaussian takes an id only when all of its
    `BIND_NEIGHBOURS` nearest share it -- the rule `plants.json` applies to merged cells."""
    from rig_tiles import tile_positions, tile_uris
    from synthetic_tree import checksum_positions

    tree = cKDTree(np.asarray(positions, np.float64))
    ids = np.asarray(splat_id, np.int64)
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    tiles: dict[str, list[int]] = {}
    for uri in tile_uris(tileset):
        at = tile_positions(tiles_dir / uri)
        distance, index = tree.query(at.astype(np.float64), k=BIND_NEIGHBOURS)
        near = ids[index]
        agreed = np.where((near == near[:, :1]).all(axis=1), near[:, 0], 0)
        labels = np.where(distance[:, 0] == 0.0, near[:, 0], agreed)
        tiles[checksum_positions(at)] = scene_plants._rle(labels)
    return dict(sorted(tiles.items()))


def _round(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def instances_document(
    instances: Sequence[Instance],
    tiles: dict[str, list[int]],
    *,
    embedding_model: str,
    dim: int,
    vocabulary_model: str,
    vocabulary_size: int,
) -> dict:
    """`instances.json` v1 (docs/SCENE_OBJECTS.md §4), keys in the contract's order."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "frame": FRAME,
        "embedding": {
            "file": "instances.emb",
            "model": embedding_model,
            "dim": dim,
            "dtype": "float16",
        },
        "vocabulary": {"model": vocabulary_model, "size": vocabulary_size},
        "instances": [
            {
                "id": i.id,
                "parent": i.parent,
                "level": i.level,
                "splats": i.splats,
                "bounds": {"min": _round(i.bounds_min), "max": _round(i.bounds_max)},
                "centroid": _round(i.centroid),
                "tags": i.tags,
                "properties": i.properties,
                "behaviour": i.behaviour,
                "views": i.views,
            }
            for i in instances
        ],
        "tiles": dict(sorted(tiles.items())),
        "tilesEncoding": TILES_ENCODING,
    }


def write_instances(out_dir: Path, document: dict, instances: Sequence[Instance]) -> None:
    """`instances.json` and `instances.emb` (float16, count x dim, row k is id k + 1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = int(document["embedding"]["dim"])
    rows = np.zeros((len(instances), dim), "<f2")
    for k, instance in enumerate(instances):
        rows[k] = instance.embedding
    (out_dir / "instances.emb").write_bytes(rows.tobytes())
    (out_dir / "instances.json").write_text(
        json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def link_instances(tileset: Path, count: int, uri: str = "instances.json") -> None:
    """`root.extras.instances = {uri, count}` on a tileset.json, in place if already there
    (the other extras keep their order), else last."""
    document = json.loads(tileset.read_text(encoding="utf-8"))
    extras = document["root"].setdefault("extras", {})
    extras["instances"] = {"uri": uri, "count": count}
    tileset.write_text(json.dumps(document, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- the run


def _colours(ids: np.ndarray) -> np.ndarray:
    """A stable colour per id (golden-ratio hues), black for 0."""
    hue = (ids * 0.6180339887) % 1.0
    k = (np.array([5.0, 3.0, 1.0]) + hue[..., None] * 6) % 6
    rgb = 1 - 0.75 * np.clip(np.minimum(k, 4 - k), 0, 1)
    return np.where((ids > 0)[..., None], rgb, 0.0)


def render_instances(splats: Splats, splat_id: np.ndarray, camera: Camera, out: Path) -> np.ndarray:
    """The scan beside itself coloured by instance id, as a PNG: what a person checks."""
    from PIL import Image

    frame = render(splats, camera, labels=splat_id.astype(np.int64))
    colour = _colours(np.maximum(frame.label, 0)) * frame.alpha[..., None]
    image = np.concatenate([frame.rgb, colour], axis=1)
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(out)
    return pixels


def collect_votes(
    views: Sequence[View], source: MaskSource, n_cells: int
) -> tuple[list[_Votes], int, list[list[Mask]]]:
    """Masks of every view, and their votes. Levels are counted from the masks."""
    all_masks = [source.masks(view.rgb) for view in views]
    levels = max((m.level for masks in all_masks for m in masks), default=0) + 1
    return [vote(v, m, n_cells, levels) for v, m in zip(views, all_masks)], levels, all_masks


def _cache_key(camera: Camera, n_cells: int) -> str:
    text = json.dumps([camera.to_json(), n_cells], sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def cached_view(
    cache: Path | None, splats: Splats, camera: Camera, cells: np.ndarray, n_cells: int
) -> View:
    """`render_view`, kept in `cache` (keyed by the camera and the cell count) so a run that
    is stopped picks up where it was rather than rendering again."""
    if cache is None:
        return render_view(splats, camera, cells)
    path = cache / f"view-{_cache_key(camera, n_cells)}.npz"
    if path.exists():
        with np.load(path) as z:
            return View(camera, z["rgb"], z["cell"], z["purity"])
    view = render_view(splats, camera, cells)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rgb=view.rgb, cell=view.cell, purity=view.purity)
    tmp.replace(path)
    return view


def cached_masks(cache: Path | None, view: View, source: MaskSource, n_cells: int) -> list[Mask]:
    """`source.masks(view.rgb)`, kept in `cache` beside the view (per mask source)."""
    if cache is None:
        return source.masks(view.rgb)
    name = hashlib.sha1(str(getattr(source, "name", "")).encode()).hexdigest()[:8]
    path = cache / f"masks-{_cache_key(view.camera, n_cells)}-{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return [
                Mask(m.astype(bool), int(level), float(score))
                for m, level, score in zip(z["masks"], z["levels"], z["scores"], strict=True)
            ]
    masks = source.masks(view.rgb)
    h, w = view.rgb.shape[:2]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        masks=np.array([m.mask for m in masks], bool).reshape(len(masks), h, w),
        levels=np.array([m.level for m in masks], np.int64),
        scores=np.array([m.score for m in masks], np.float64),
    )
    tmp.replace(path)
    return masks


@dataclass
class Segmentation:
    splat_id: np.ndarray
    instances: list[Instance]
    lifted: Lifted
    views: list[View]
    cell: np.ndarray
    timings: dict[str, float]
    #: What `lift` was given, so it can be re-run (`relift`).
    votes: list[_Votes] = field(default_factory=list)
    cells: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    def relift(self) -> Lifted:
        """`lift` again from the same votes (it is deterministic)."""
        assert self.cells is not None
        _, centroids, counts, edge = self.cells
        levels = int(self.lifted.stats["levels"])
        return lift(self.votes, centroids, counts, edge, levels)


def segment(
    splats: Splats,
    source: MaskSource | None,
    embedder: Embedder,
    vocabulary: Sequence[str],
    *,
    cameras: Sequence[Camera] | None = None,
    view_count: int = VIEW_COUNT,
    cells: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
    source_factory=None,
    cache: Path | None = None,
    progress=None,
) -> Segmentation:
    """Cells, views, masks, votes, lifting and meaning, for a scan held in memory.

    `source_factory(cameras)` builds a mask source that needs the cameras (`OracleMasks`).
    `cache`: a directory where each view and its masks are kept as they are made, so a
    stopped run resumes (views and masks are most of the time on a large scan).
    `progress(message)` is told as each view is done."""
    timings: dict[str, float] = {}
    mark = time.perf_counter()
    cell, centroids, counts, edge = cells or supervoxels(splats.positions)
    timings["cellsS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    if cameras is None:
        cameras = plan_views(splats.positions, view_count, observers=observer_points(splats))
    timings["planS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    n_cells = len(centroids)
    views = []
    for k, camera in enumerate(cameras):
        views.append(cached_view(cache, splats, camera, cell, n_cells))
        if progress:
            progress(f"view {k + 1}/{len(cameras)} rendered")
    timings["renderS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    if source is None:
        source = source_factory(cameras)
    all_masks = []
    for k, view in enumerate(views):
        all_masks.append(cached_masks(cache, view, source, n_cells))
        if progress:
            progress(f"view {k + 1}/{len(views)} masked ({len(all_masks[-1])} masks)")
    levels = max((m.level for masks in all_masks for m in masks), default=0) + 1
    votes = [vote(v, m, n_cells, levels) for v, m in zip(views, all_masks, strict=True)]
    timings["masksS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    lifted = lift(votes, centroids, counts, edge, levels)
    lifted.stats["cellEdgeM"] = round(edge, 4)
    lifted.stats["levels"] = levels
    timings["liftS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    instances = describe(lifted, splats, cell, views, embedder, vocabulary)
    timings["describeS"] = time.perf_counter() - mark
    return Segmentation(
        lifted.cell_id[cell],
        instances,
        lifted,
        views,
        cell,
        timings,
        votes,
        (cell, centroids, counts, edge),
    )


def colour_family(colours: np.ndarray) -> np.ndarray:
    """0 green-dominant, 1 red-dominant, 2 anything else: a part split with no class in it
    (leaves from bark, a roof from its walls)."""
    r, g, b = np.asarray(colours, np.float64).T
    return np.where((g > r) & (g > b), 0, np.where(r > 1.6 * g, 1, 2)).astype(np.int64)


def truth_levels(labels: dict, rows: np.ndarray, colours: np.ndarray) -> list[np.ndarray]:
    """A synthetic scene's `labels.json` as oracle levels for the kept rows: level 0 the
    objects (each instance; splats of no instance by class), level 1 their parts (each
    object split by `colour_family`)."""
    klass = np.asarray(labels["class"], np.int64)[rows]
    instance = np.asarray(labels["instance"], np.int64)[rows]
    objects = np.where(instance >= 0, instance, len(labels["instances"]) + klass)
    return [objects, objects * 3 + colour_family(colours)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        type=Path,
        help="the scan's PLY (the one the tiles were packed from), or its tileset.json "
        "(the leaves are the splats, bound to every tile by position)",
    )
    parser.add_argument("tiles", type=Path, help="its tileset directory (tileset.json)")
    parser.add_argument("--masks", default=None, help="module:Class, a MaskSource")
    parser.add_argument("--truth", type=Path, default=None, help="labels.json: oracle masks")
    parser.add_argument(
        "--embedder", default="segment_scene:FakeEmbedder", help="module:Class, an Embedder"
    )
    parser.add_argument("--vocabulary", type=Path, default=None, help="one tag per line")
    parser.add_argument("--views", type=int, default=VIEW_COUNT)
    parser.add_argument("--save-dir", type=Path, default=None, help="write the views here")
    parser.add_argument(
        "--cache", type=Path, default=None, help="keep views and masks here; a rerun resumes"
    )
    parser.add_argument("--render-instances", type=Path, default=None, help="a PNG to check")
    parser.add_argument("--out", type=Path, default=None, help="default: the tiles directory")
    parser.add_argument("--opacity-min", type=float, default=scene_plants.PACKAGE_OPACITY_MIN)
    parser.add_argument("--tile-gaussians", type=int, default=scene_plants.PACKAGE_TILE_GAUSSIANS)
    args = parser.parse_args()
    if (args.masks is None) == (args.truth is None):
        parser.error("give exactly one of --masks and --truth")
    from_tiles = args.source.name.endswith(".json")
    if from_tiles:
        if args.truth:
            parser.error("--truth needs the source PLY (labels are per PLY row)")
        from splat_render import load_tileset

        splats = load_tileset(args.source)
        rows, row_count = np.arange(len(splats)), len(splats)
    else:
        splats, rows, row_count = load_source(args.source, args.opacity_min)
    embedder = load_embedder(args.embedder)
    vocabulary: list[str] = []
    if args.vocabulary:
        vocabulary = [
            line.strip()
            for line in args.vocabulary.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    source = load_masks(args.masks) if args.masks else None
    factory = None
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        levels = truth_levels(truth, rows, splats.colours)
        factory = lambda cameras: OracleMasks(splats, levels, cameras)
    result = segment(
        splats,
        source,
        embedder,
        vocabulary,
        view_count=args.views,
        source_factory=factory,
        cache=args.cache,
        progress=lambda message: print(message, flush=True),
    )
    if from_tiles:
        tiles = tile_binding_by_position(args.tiles, splats.positions, result.splat_id)
    else:
        tiles = tile_binding(
            args.tiles,
            args.source,
            rows,
            result.splat_id,
            row_count,
            opacity_min=args.opacity_min,
            tile_gaussians=args.tile_gaussians,
        )
    document = instances_document(
        result.instances,
        tiles,
        embedding_model=embedder.name,
        dim=int(embedder.dim),
        vocabulary_model=embedder.name,
        vocabulary_size=len(vocabulary),
    )
    out = args.out or args.tiles
    write_instances(out, document, result.instances)
    if out.resolve() == args.tiles.resolve():
        link_instances(args.tiles / "tileset.json", len(result.instances))
    if args.save_dir:
        from PIL import Image

        args.save_dir.mkdir(parents=True, exist_ok=True)
        for k, view in enumerate(result.views):
            Image.fromarray(view.rgb).save(args.save_dir / f"view_{k:03d}.png")
        (args.save_dir / "cameras.json").write_text(
            json.dumps([v.camera.to_json() for v in result.views], indent=1), encoding="utf-8"
        )
    if args.render_instances:
        render_instances(splats, result.splat_id, result.views[0].camera, args.render_instances)
    print(
        json.dumps(
            {
                "instances": len(result.instances),
                "assignedShare": round(float((result.splat_id > 0).mean()), 4),
                **result.lifted.stats,
                "timingsS": {k: round(v, 2) for k, v in result.timings.items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
