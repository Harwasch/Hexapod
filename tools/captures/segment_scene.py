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

**Views** (`plan_views`). `--views` views of the whole scan (rings, and eye-height views
where it was seen from) and, for a scan wider than one view's footprint, local views that
scale with its area: a footprint is `view_footprint` (a cell spans `CELL_PX` pixels), the
scan's footprint is gridded `ANCHORS_PER_FOOTPRINT` times per footprint width, and each
anchor gets `OBLIQUE_VIEWS` obliques placed by line of sight (under a canopy they go low, in
the open high) and `EYE_VIEWS` more eye-height views; every local view has a far plane a
footprint past its target, so it shows what it resolves and not the horizon as specks. At
most `MAX_VIEWS`. Views render in forked workers (`render_views`: the scan shared
copy-on-write, `splat_render.SplatIndex` culling what a view cannot reach, the same frames
as one process) while this process masks and votes each view as it arrives.

**Images** (`--renderer gsplat`, `make_renderer`). What the mask and image models see is
rasterized by gsplat on a GPU, as a viewer draws the scan -- not the CPU renderer's point
samples, speckle on black, far from the photos the models know; the per-pixel cells still
come from the CPU's samples of the same camera. `--max-scale-m` leaves the floaters out of
the views (gaussians larger than that, which a rasterizer draws as blobs over a view); an
unseen cell takes its nearest seen cell's instance within its own reach (`lift`).

**Coverage** (`coverage_views`, `--coverage-rounds`). After the lift, views are aimed at
what is still without an instance -- the coarse rim of a capture, a forest's crowns, a
corner the plan missed: unassigned splats binned in 3D, the heaviest bins targets, each seen
by two obliques and once from eye height (looking up into a canopy), from the azimuth with
the clearest line of sight; then everything is lifted again from all the votes.

**Votes.** In one view, a cell is *in* a mask when at least `MASK_SHARE` of its visible
(purity-weighted) pixels are; overlapping masks of one level go to the one holding the larger
share. A cell is *visible* in a view with at least `MIN_VISIBLE_PX` of weight. A mask over
more than `MAX_MASK_SHARE` of the view says only that the view is one thing, and does not
vote.

**Instances, per mask level.** Two cells belong together when, over the views where both are
visible and at least one is in a mask of that level, they are in the *same* mask in at least
`MERGE_RATIO` of them (co-occurrence over co-visibility), and in at least `MIN_COVISIBLE`
views. Stage 1 asks it of spatial neighbours only (each cell's `NEIGHBOURS` nearest within
`NEIGHBOUR_CELLS` cell edges), so the graph is sparse; its components are fragments. It is
single linkage, so it asks for `STRICT_RATIO` against `STRICT_PRIOR` pseudo-views as well:
over a million cells, pairs that agree in two views by chance chain across a whole scan (on
the camp, into one region of 63% of its splats). Stage 2 asks it again of regions as wholes,
in rounds of mutual-best joins, first of neighbours, then of any two that ever shared a mask,
wherever they are -- which rejoins an object that occlusion or a gap split, without a
quadratic pass over cells. It is incremental (`_grow`): a round re-judges only the regions it
merged and re-scores only their pairs, so its cost follows what changed, not the views.
Then specks (regions under `MIN_REGION_CELLS` cells) and seen cells in no mask take their
neighbours' region (`_absorb`), and regions below `MIN_INSTANCE_SPLATS` are dropped. Cells
no view ever saw (inside a crown, under a roof) take their nearest seen cell's labels within
`FILL_CELLS` edges.

**Hierarchy.** Levels are the mask model's scale hints (0 = coarsest). The instances of
level L are refined by those of L+1: a child is the cells of a parent that share one level
L+1 instance, so every child's splats are inside its parent by construction. A child that
would be its whole parent is not made, nor one below `MIN_INSTANCE_SPLATS`; a splat whose
cell has no finer instance keeps its parent's id. `level` in the output is depth in this
tree. Splat ids are the deepest instance (leaf-level), as the contract says.

**Meaning.** Instances whose best view gives them `DESCRIBE_MIN_PX` pixels are described;
smaller ones keep no embedding and no tags (they stay in the hierarchy, with their nearest
described ancestor's properties). Per instance, from the `CROP_VIEWS` views where it covers
most pixels (counted at half where it runs off the frame), square crops are embedded and
averaged (`embedding`, L2-normalised): in context (box padded `CROP_PAD`, the rest dimmed),
alone (its own pixels on grey), and with a renderer a portrait of its own splats from that
view's side, framed to its box at `PORTRAIT_PX` (no occluder, its own resolution).
`category`: a zero-shot head over the categories' prompts (`category_scores`) mixed with
its best labels' categories (`_categories`).
`tags`: softmax over the whole vocabulary of `LOGIT_SCALE` x cosine, top `TAGS_TOP_K`.
`properties`: per attribute, a two-way softmax of `LOGIT_SCALE` x cosine between a positive
and a contrast prompt (`PROPERTY_PROMPTS`), so each is a probability on its own and the
attributes do not compete. `behaviour`: `BEHAVIOUR_RULE`.

**Scale** (2026-10-02, Modal L4 + 32 CPUs, `infra/modal/segment.py`): the 22.6M-gaussian
camp (~110 m across) gets 252 views (24 whole-scan + 228 local); 998k cells at 0.24 m, 5.7M
stage-1 edges. Cells 29 s, plan 65 s (observers, index, line of sight), render 45 s of
waiting (24 workers, ~8 s a view each, overlapped with masking), SAM 2.1 masks 695 s (2.8 s
a view at 32 points a side: now the bulk), votes 11 s, lift 103 s, describe 60 s (1.9k of
7.3k instances); ~20 min in all. Before (24 views, one render process, the per-view lift):
render 406 s, lift 81 s, describe 604 s, and one instance held 26% of the scan. A camp view
renders in 1.6-1.9 GB at most (`RENDER_WORKER_BYTES`).

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
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import rebind_instances
import scene_categories
import scene_plants
import splat_tiles
from splat_render import Camera, SplatIndex, Splats, _from_columns, render

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
#: What one render worker may hold at its peak (a view of the 22.6M-gaussian camp: < 2 GB).
RENDER_WORKER_BYTES = 2.5e9
#: At most this many views in all (rings, observers, local).
MAX_VIEWS = 480
#: Splats sampled for the scan's extent and the observers' clearance test (seeded).
PLAN_SAMPLE = 400_000

#: Coverage rounds (`coverage_views`): after a lift, views aimed at what is still without an
#: instance, at most this many a round ...
COVERAGE_VIEWS = 96
#: ... `len(COVERAGE_SHOTS)` per target: (elevation in degrees, distance as a factor of the
#: distance where the frame spans one footprint). The last is from eye height (`EYE_M`
#: above the target's local ground), looking at the target -- up into a canopy.
COVERAGE_SHOTS = ((40.0, 1.0), (25.0, 0.6), (None, 0.8))
EYE_M = 1.6
#: Azimuths tried per shot (the clearest line of sight wins).
COVERAGE_AZIMUTHS = 8
#: Targets: unassigned splats binned in 3D at half a footprint; a bin needs this many, and
#: targets are at least half a footprint apart.
COVERAGE_MIN_SPLATS = 40
#: A round is not run once less than this share of the splats is without an instance.
COVERAGE_MIN_SHARE = 0.002
#: Rounds by default (the CLI's `--coverage-rounds`).
COVERAGE_ROUNDS = 2

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
#: ... counting this many pseudo-views against them, so two views are not enough: single
#: linkage over a million cells percolates through any pair that agrees by chance.
STRICT_PRIOR = 3.0
#: A mask over more than this share of a view's drawn pixels does not vote: it says the
#: whole view is one thing, which is no evidence about what in it belongs together.
MAX_MASK_SHARE = 0.8
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
#: A region of fewer cells is a speck: its cells, and seen cells in no mask, take their
#: neighbours' region, up to this many steps out (`_absorb`).
MIN_REGION_CELLS = 8
ABSORB_ROUNDS = 3
#: An instance needs at least this many splats, or `MIN_INSTANCE_CELLS` cells: where a scan
#: is sparse (the coarse rim of a capture: a gaussian a cell) a tree is a few dozen splats.
MIN_INSTANCE_SPLATS = 30
MIN_INSTANCE_CELLS = 16
#: Cells no view saw take the nearest seen cell's labels within this many cell edges, or
#: within twice their largest gaussian's scale when that is farther (a gaussian left out of
#: the views, `max_scale_m`, reaches its neighbours).
FILL_CELLS = 3.0

#: An instance is described (crops embedded, tags) when its best view gives it at least
#: this many pixels: SigLIP sees 16-pixel patches of a 224 crop, and a smaller crop is noise.
DESCRIBE_MIN_PX = 1024
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
                clear = _clear_share(sight, target, offsets, probes, footprint)
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


def _clear_share(
    sight: cKDTree, target: np.ndarray, offsets: np.ndarray, probes: np.ndarray, footprint: float
) -> np.ndarray:
    """Line of sight per candidate eye (`target + offset`): the share of probes from the eye
    towards the target that are clear, i.e. have no point of `sight` within
    `OBLIQUE_CLEARANCE` of a footprint."""
    points = target + offsets[:, None, :] * probes[None, ::-1, None]
    gap = sight.query(points.reshape(-1, 3), k=1)[0].reshape(len(offsets), -1)
    return (gap >= OBLIQUE_CLEARANCE * footprint).mean(axis=1)


def coverage_views(
    centroids: np.ndarray,
    missing: np.ndarray,
    edge: float,
    *,
    budget: int = COVERAGE_VIEWS,
    width: int = VIEW_WIDTH,
    height: int = VIEW_HEIGHT,
    fov_deg: float = VIEW_FOV_DEG,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[Camera]:
    """Views aimed at what a lift left without an instance (a coverage round).

    `missing`: per cell, its splats that carry no instance (0 for the rest). They are binned
    in 3D at half a footprint (`view_footprint(edge)`), so a canopy and the ground under it
    are separate targets; the heaviest bins with at least `COVERAGE_MIN_SPLATS` become
    targets (the weighted mean of their cells), greedily, at least half a footprint apart,
    up to `budget // len(COVERAGE_SHOTS)`. Each target gets `COVERAGE_SHOTS`: obliques at
    two scales and one from eye height above its local ground (the 5th percentile height of
    the cells within half a footprint), each from the azimuth (of `COVERAGE_AZIMUTHS`, turned
    by the golden angle per target) with the clearest line of sight past the occupied cells
    (`_clear_share`), with a far plane a footprint past the target. Nothing here knows the
    scene: the rim of a capture, a forest's crowns and a corner the plan missed are all just
    places with unassigned splats."""
    centroids = np.asarray(centroids, np.float64)
    missing = np.asarray(missing, np.float64)
    per_target = len(COVERAGE_SHOTS)
    if budget < per_target or not (missing > 0).any():
        return []
    footprint = view_footprint(edge, width)
    step = footprint / 2
    where = np.flatnonzero(missing > 0)
    pts, w = centroids[where], missing[where]
    key3 = np.floor((pts - pts.min(axis=0)) / step).astype(np.int64)
    dims = key3.max(axis=0) + 1
    key = (key3[:, 0] * dims[1] + key3[:, 1]) * dims[2] + key3[:, 2]
    keys, inverse = np.unique(key, return_inverse=True)
    weight = np.bincount(inverse, w, keys.size)
    centre = np.stack([np.bincount(inverse, w * pts[:, k], keys.size) for k in range(3)], 1)
    centre /= weight[:, None]
    order = np.argsort(-weight, kind="stable")
    order = order[weight[order] >= COVERAGE_MIN_SPLATS]
    targets: list[np.ndarray] = []
    for b in order:
        if len(targets) >= budget // per_target:
            break
        if targets and np.min(np.linalg.norm(np.asarray(targets) - centre[b], axis=1)) < step:
            continue
        targets.append(centre[b])
    if not targets:
        return []
    sight = cKDTree(centroids)
    flat = cKDTree(centroids[:, :2])
    half_fov = math.radians(fov_deg) / 2
    reach = footprint / 2 / math.tan(half_fov)
    probes = np.linspace(OBLIQUE_PROBE_FROM, 1.0, OBLIQUE_PROBES)
    golden = math.pi * (3 - math.sqrt(5))
    up_a = np.asarray(up, np.float64)
    cameras: list[Camera] = []
    for i, target in enumerate(targets):
        near = flat.query_ball_point(target[:2], step)
        ground = float(np.percentile(centroids[near, 2], 5)) if near else float(target[2])
        azimuths = i * golden + 2 * math.pi * np.arange(COVERAGE_AZIMUTHS) / COVERAGE_AZIMUTHS
        for j, (elevation_deg, factor) in enumerate(COVERAGE_SHOTS):
            if elevation_deg is None:
                # From eye height, `factor` of the reach away horizontally.
                offsets = np.stack(
                    [
                        reach * factor * np.cos(azimuths),
                        reach * factor * np.sin(azimuths),
                        np.full(azimuths.size, ground + EYE_M - target[2]),
                    ],
                    1,
                )
            else:
                e = math.radians(elevation_deg)
                offsets = (
                    reach
                    * factor
                    * np.stack(
                        [
                            math.cos(e) * np.cos(azimuths),
                            math.cos(e) * np.sin(azimuths),
                            np.full(azimuths.size, math.sin(e)),
                        ],
                        1,
                    )
                )
            clear = _clear_share(sight, target, offsets, probes, footprint)
            # Ties (all clear, as at the open rim) turn with the shot, so shots differ.
            pick = int(np.argmax(clear + 1e-6 * np.roll(np.arange(azimuths.size) == 0, j * 3)))
            eye = target + offsets[pick]
            if np.linalg.norm(eye - target) < 1e-6:
                continue
            far = float(np.linalg.norm(offsets[pick])) + footprint
            cameras.append(
                Camera.look_at(
                    eye, target, fov_deg=fov_deg, width=width, height=height,
                    up=tuple(up_a), far=far,
                )
            )  # fmt: skip
    return cameras


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
    #: (h, w, 3) uint8: the CPU renderer's point samples, when `rgb` was drawn by another.
    samples: np.ndarray | None = None


def render_view(
    splats: Splats, camera: Camera, cells: np.ndarray, index: SplatIndex | None = None
) -> View:
    """The view the pipeline and `OracleMasks` both see (same renderer, same seed; `index`
    only makes it faster)."""
    frame = render(splats, camera, labels=cells, index=index)
    return View(
        camera,
        np.round(frame.rgb * 255).astype(np.uint8),
        frame.label.astype(np.int32),
        frame.purity.astype(np.float32),
    )


def render_views(
    splats: Splats,
    cameras: Sequence[Camera],
    cells: np.ndarray,
    *,
    index: SplatIndex | None = None,
    workers: int = 1,
    cache: Path | None = None,
    tag: str = "",
) -> Iterator[View]:
    """The views, in camera order, rendered by `workers` processes. The processes are forked
    (the scan is shared copy-on-write, not copied), each renders whole views, and the views
    come back in order as they are done -- the same views as one process makes, so the run
    is deterministic. The caller can use each view (mask it on a GPU) while the rest render.
    Fork before anything starts a GPU context; with `workers` 1 or no fork, one process."""
    import multiprocessing as mp

    n_cells = int(cells.max()) + 1 if cells.size else 0
    _POOL_STATE.update(
        splats=splats, cameras=list(cameras), cells=cells, index=index, cache=cache,
        n_cells=n_cells, tag=tag,
    )  # fmt: skip
    try:
        if workers <= 1 or len(cameras) <= 1 or "fork" not in mp.get_all_start_methods():
            for k in range(len(cameras)):
                yield _render_job(k)
            return
        # An executor, not a Pool: a worker that dies (out of memory) fails the run
        # (BrokenProcessPool) rather than leaving it waiting for a view forever.
        context = mp.get_context("fork")
        with ProcessPoolExecutor(min(workers, len(cameras)), mp_context=context) as pool:
            yield from pool.map(_render_job, range(len(cameras)))
    finally:
        _POOL_STATE.clear()


#: What forked render workers read (set by `render_views` before the fork).
_POOL_STATE: dict[str, object] = {}


def _render_job(k: int) -> View:
    state = _POOL_STATE
    cameras = state["cameras"]
    assert isinstance(cameras, list)
    return cached_view(
        state["cache"],  # type: ignore[arg-type]
        state["splats"],  # type: ignore[arg-type]
        cameras[k],
        state["cells"],  # type: ignore[arg-type]
        int(state["n_cells"]),  # type: ignore[arg-type]
        state["index"],  # type: ignore[arg-type]
        str(state.get("tag", "")),
    )


def _memory_room() -> float | None:
    """Bytes this process's cgroup can still take (cgroup v2), or None if unknown."""
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError):
        return None
    if limit == "max":
        return None
    return float(int(limit) - used)


def default_workers() -> int:
    """Processes to render with: the CPUs this process may use, no more than the memory
    left holds at `RENDER_WORKER_BYTES` each."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    room = _memory_room()
    if room is not None:
        cpus = min(cpus, int(room // RENDER_WORKER_BYTES))
    return max(1, cpus)


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
    """Which mask of each level every visible cell of `view` is in (`MASK_SHARE`). Works on
    the view's own cells, so its cost does not grow with the scan."""
    owner = view.cell.reshape(-1)
    weight = view.purity.reshape(-1).astype(np.float64)
    good = (owner >= 0) & (weight >= MIN_PURITY)
    pixels = np.flatnonzero(good)
    cells, local = np.unique(owner[pixels], return_inverse=True)
    weight = weight[pixels]
    m = cells.size
    seen = np.bincount(local, weight, m)
    best = np.full((levels, m), -1, np.int32)
    best_share = np.zeros((levels, m))
    best_score = np.zeros((levels, m))
    for k, mask in enumerate(masks):
        inside = np.asarray(mask.mask, bool).reshape(-1)[pixels]
        if inside.sum() > MAX_MASK_SHARE * pixels.size:
            continue  # (nearly) the whole view: says nothing about what is one thing
        share = np.bincount(local[inside], weight[inside], m)
        share = np.divide(share, seen, out=np.zeros(m), where=seen > 0)
        level = mask.level
        better = (share >= MASK_SHARE) & (
            (share > best_share[level])
            | ((share == best_share[level]) & (mask.score > best_score[level]))
        )
        best[level, better] = k
        best_share[level, better] = share[better]
        best_score[level, better] = mask.score
    visible = seen >= MIN_VISIBLE_PX
    return _Votes(
        cells[visible].astype(np.int32), seen[visible].astype(np.float32), best[:, visible]
    )


def _pad_levels(votes: _Votes, levels: int) -> _Votes:
    """`votes` with -1 rows for the mask levels its view had none of."""
    have = votes.masks.shape[0]
    if have >= levels:
        return votes
    pad = np.full((levels - have, votes.cells.size), -1, np.int32)
    return _Votes(votes.cells, votes.weight, np.concatenate([votes.masks, pad]))


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


@dataclass
class _Observed:
    """Every view's votes of one mask level, as flat rows in view order (cells ascending
    within a view): the visible weight of each in-mask cell, and its mask (-1: none)."""

    cell: np.ndarray  # (e,) int64
    view: np.ndarray  # (e,) int64
    weight: np.ndarray  # (e,) float64
    mask: np.ndarray  # (e,) int64
    views: int
    masks: int  # 1 + the largest mask index


def _observed(votes: Sequence[_Votes], level: int, in_mask: np.ndarray) -> _Observed:
    cells, views, weights, masks = [], [], [], []
    for k, v in enumerate(votes):
        ok = in_mask[v.cells]
        cells.append(v.cells[ok].astype(np.int64))
        views.append(np.full(int(ok.sum()), k, np.int64))
        weights.append(v.weight[ok].astype(np.float64))
        masks.append(v.masks[level][ok].astype(np.int64))
    if not cells:
        z = np.zeros(0, np.int64)
        return _Observed(z, z, np.zeros(0), z, 0, 1)
    mask = np.concatenate(masks)
    return _Observed(
        np.concatenate(cells),
        np.concatenate(views),
        np.concatenate(weights),
        mask,
        len(votes),
        int(mask.max()) + 1 if mask.size else 1,
    )


@dataclass
class _Seen:
    """Per (region, view) where the region is visible (`MIN_VISIBLE_PX`): the mask holding
    `MASK_SHARE` of its visible weight there (-1: none). Rows sorted by region, then view."""

    region: np.ndarray  # (r,) int64
    view: np.ndarray  # (r,) int64
    best: np.ndarray  # (r,) int64
    #: Views in all (keys are region x views_bound + view).
    views_bound: int

    def rows(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where each id's rows start, and how many."""
        start = np.searchsorted(self.region, ids, "left")
        return start, np.searchsorted(self.region, ids, "right") - start


def _seen(obs: _Observed, region: np.ndarray, rows: np.ndarray | None = None) -> _Seen:
    """`_Seen` of the regions `region` (per cell, -1: none) gives the observed rows `rows`
    (default: all). A region is judged as a whole, so a few cells that bleed into a
    neighbour's mask do not carry it with them. Sums run in the rows' order."""
    cell = obs.cell if rows is None else obs.cell[rows]
    reg = region[cell]
    ok = reg >= 0
    view = (obs.view if rows is None else obs.view[rows])[ok]
    weight = (obs.weight if rows is None else obs.weight[rows])[ok]
    mask = (obs.mask if rows is None else obs.mask[rows])[ok]
    reg = reg[ok]
    key = reg * obs.views + view
    keys, inverse = np.unique(key, return_inverse=True)
    visible = np.bincount(inverse, weight, keys.size)
    best = np.full(keys.size, -1, np.int64)
    assigned = mask >= 0
    if assigned.any():
        mkeys, minv = np.unique(key[assigned] * obs.masks + mask[assigned], return_inverse=True)
        sums = np.bincount(minv, weight[assigned], mkeys.size)
        kr, km = mkeys // obs.masks, mkeys % obs.masks
        order = np.lexsort((km, -sums, kr))
        first = order[np.r_[True, kr[order][1:] != kr[order][:-1]]]
        at = np.searchsorted(keys, kr[first])
        strong = sums[first] >= MASK_SHARE * visible[at]
        best[at[strong]] = km[first][strong]
    keep = visible >= MIN_VISIBLE_PX
    keys = keys[keep]
    return _Seen(keys // obs.views, keys % obs.views, best[keep], obs.views)


#: Pair rows joined at a time in `_agreement` (bounds its memory).
JOIN_ROWS = 1 << 24


def _agreement(seen: _Seen, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: views in the same mask, and views where both were seen and either masked.
    A sorted join of the two regions' rows on the view."""
    together = np.zeros(a.size, np.int64)
    both = np.zeros(a.size, np.int64)
    if a.size == 0 or seen.region.size == 0:
        return together, both
    sa, la = seen.rows(a)
    sb, lb = seen.rows(b)
    cost = np.cumsum(la + lb)
    lo = 0
    while lo < a.size:
        hi = int(np.searchsorted(cost, (cost[lo - 1] if lo else 0) + JOIN_ROWS, "right"))
        hi = max(hi, lo + 1)
        part = slice(lo, hi)
        n_part = hi - lo
        pair_a = np.repeat(np.arange(n_part), la[part])
        rows_a = np.repeat(sa[part] - np.cumsum(la[part]) + la[part], la[part]) + np.arange(
            pair_a.size
        )
        pair_b = np.repeat(np.arange(n_part), lb[part])
        rows_b = np.repeat(sb[part] - np.cumsum(lb[part]) + lb[part], lb[part]) + np.arange(
            pair_b.size
        )
        # Both sides are sorted by (pair, view), so their keys are ascending.
        key_a = pair_a * seen.views_bound + seen.view[rows_a]
        key_b = pair_b * seen.views_bound + seen.view[rows_b]
        at = np.minimum(np.searchsorted(key_b, key_a), max(key_b.size - 1, 0))
        match = (key_b[at] == key_a) if key_b.size else np.zeros(key_a.size, bool)
        best_a = seen.best[rows_a[match]]
        best_b = seen.best[rows_b[at[match]]]
        informative = (best_a >= 0) | (best_b >= 0)
        same = informative & (best_a == best_b)
        owner = pair_a[match]
        both[part] = np.bincount(owner, informative, n_part)
        together[part] = np.bincount(owner, same, n_part)
        lo = hi
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


def _shared(seen: _Seen, size: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of regions that were in one mask in some view, wherever they are: the
    `SHARED_REGIONS` largest of each mask (so a mask over thousands of specks stays cheap)."""
    masked = seen.best >= 0
    reg, view, best = seen.region[masked], seen.view[masked], seen.best[masked]
    order = np.lexsort((reg, -size[reg], best, view))
    reg, view, best = reg[order], view[order], best[order]
    if reg.size < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, (view[1:] != view[:-1]) | (best[1:] != best[:-1])])
    lengths = np.diff(np.r_[starts, reg.size])
    rank = np.arange(reg.size) - np.repeat(starts, lengths)
    held = np.minimum(lengths, SHARED_REGIONS)
    keys: list[np.ndarray] = []
    for g in np.unique(held[held > 1]):
        # Every mask holding g regions (after the cut): a (k, g) table, ids sorted per row.
        first = starts[held == g]
        table = np.sort(reg[first[:, None] + np.arange(g)[None, :]], axis=1)
        i, j = np.triu_indices(int(g), 1)
        keys.append((table[:, i] * n + table[:, j]).reshape(-1))
    del rank
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = np.unique(np.concatenate(keys))
    return key // n, key % n


def _joins(
    together: np.ndarray,
    both: np.ndarray,
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
    re-judged as wholes (`_seen`), every neighbouring pair is scored, and each region
    joins its best partner when that partner is a sink (`_joins`) -- average evidence, never
    a chain through one ambiguous cell. When neighbours are done, pairs that shared a mask
    anywhere are scored too (`_shared`): an object's parts that a gap or another part
    separates in space. Up to `MAX_ROUNDS`. `weights`: splats per cell.

    Incremental: a region keeps the smallest id of what joined it (so ids keep their order
    and ties break as with compact ids), only the regions a round changed are re-judged,
    and only pairs touching them are scored again."""
    n_cells = in_mask.size
    obs = _observed(votes, level, in_mask)
    region = np.where(in_mask, np.arange(n_cells), -1)
    ca, cb = _pairs(region, a, b, n_cells)
    together, both = _agreement(_seen(obs, region), ca, cb)
    keep = (both >= MIN_COVISIBLE) & (together >= STRICT_RATIO * (both + STRICT_PRIOR))
    region = np.where(in_mask, _components(n_cells, ca[keep], cb[keep]), -1)
    region = _relabel(region)
    n = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
    if n < 2:
        return region, 0
    valid = region >= 0
    size = np.bincount(region[valid], weights[valid], n).astype(np.float64)
    seen = _seen(obs, region)
    pa, pb = _pairs(region, a, b, n)
    neighbours = pa * n + pb
    cache_keys = np.zeros(0, np.int64)
    cache_together = cache_both = np.zeros(0, np.int64)
    alive = n
    rounds = 0
    far = False
    while rounds < MAX_ROUNDS and alive >= 2:
        keys = neighbours
        if far:
            fa, fb = _shared(seen, size, n)
            keys = np.union1d(neighbours, fa * n + fb)
        at = np.minimum(np.searchsorted(cache_keys, keys), max(cache_keys.size - 1, 0))
        hit = (cache_keys[at] == keys) if cache_keys.size else np.zeros(keys.size, bool)
        together = np.zeros(keys.size, np.int64)
        both = np.zeros(keys.size, np.int64)
        together[hit], both[hit] = cache_together[at[hit]], cache_both[at[hit]]
        miss = np.flatnonzero(~hit)
        together[miss], both[miss] = _agreement(seen, keys[miss] // n, keys[miss] % n)
        cache_keys, cache_together, cache_both = keys, together, both
        movers, best = _joins(together, both, keys // n, keys % n, size)
        rounds += 1
        if movers.size == 0:
            if far:
                break
            far = True  # neighbours are done; now regions that share masks anywhere
            continue
        joined = _components(n, movers, best[movers])
        low = np.full(int(joined.max()) + 1, n, np.int64)
        np.minimum.at(low, joined, np.arange(n))
        new = low[joined]
        count = np.bincount(joined)
        member = count[joined] > 1
        alive -= int(member.sum()) - int((count > 1).sum())
        size = np.bincount(new, size, n)
        region = np.where(region >= 0, new[np.maximum(region, 0)], -1)
        # Re-judge the merged regions only; the others' rows stand.
        changed = np.zeros(n, bool)
        changed[new[member]] = True
        of_row = region[obs.cell]
        fresh = _seen(obs, region, np.flatnonzero((of_row >= 0) & changed[np.maximum(of_row, 0)]))
        kept = ~member[seen.region]
        old_keys = seen.region[kept] * obs.views + seen.view[kept]
        place = np.searchsorted(old_keys, fresh.region * obs.views + fresh.view)
        seen = _Seen(
            np.insert(seen.region[kept], place, fresh.region),
            np.insert(seen.view[kept], place, fresh.view),
            np.insert(seen.best[kept], place, fresh.best),
            obs.views,
        )
        na, nb = new[neighbours // n], new[neighbours % n]
        ok = na != nb
        neighbours = np.unique(np.minimum(na[ok], nb[ok]) * n + np.maximum(na[ok], nb[ok]))
        stale = member[cache_keys // n] | member[cache_keys % n]
        cache_keys = cache_keys[~stale]
        cache_together, cache_both = cache_together[~stale], cache_both[~stale]
    return _relabel(region), rounds


def _absorb(region: np.ndarray, a: np.ndarray, b: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Specks and gaps: a seen cell in no region, or in one of fewer than `MIN_REGION_CELLS`
    cells, takes the region most of its stage-1 neighbours are in (ties: the lower id), for
    up to `ABSORB_ROUNDS` steps outwards. A mask model leaves slivers between its masks and
    a few cells that no view resolved; they belong to what surrounds them."""
    region = region.copy()
    n = int(region.max()) + 1 if region.size else 0
    if n == 0:
        return region
    cells = np.bincount(region[region >= 0], minlength=n)
    open_ = seen & ((region < 0) | (cells[np.maximum(region, 0)] < MIN_REGION_CELLS))
    region[open_] = -1
    for _ in range(ABSORB_ROUNDS):
        ra, rb = region[a], region[b]
        to_a = (ra < 0) & (rb >= 0) & open_[a]
        to_b = (rb < 0) & (ra >= 0) & open_[b]
        cell = np.concatenate([a[to_a], b[to_b]])
        if cell.size == 0:
            break
        label = np.concatenate([rb[to_a], ra[to_b]])
        keys, counts = np.unique(cell * n + label, return_counts=True)
        kc, kl = keys // n, keys % n
        order = np.lexsort((kl, -counts, kc))
        first = order[np.r_[True, kc[order][1:] != kc[order][:-1]]]
        region[kc[first]] = kl[first]
    return region


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
    cell_reach: np.ndarray | None = None,
) -> Lifted:
    """Instances from the views' votes (module docstring: stages 1-2, fill, hierarchy).
    `cell_reach`: per cell, how far (metres) its unseen splats reach (twice their largest
    scale); an unseen cell takes the nearest seen cell's labels within `FILL_CELLS` edges or
    that reach, whichever is farther."""
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
        joined = _absorb(joined, a, b, seen)
        stats[f"level{level}Rounds"] = rounds
        valid = joined >= 0
        top = int(joined.max()) + 1 if joined.size else 0
        size = np.bincount(joined[valid], cell_counts[valid], minlength=max(top, 1))
        cells = np.bincount(joined[valid], minlength=max(top, 1))
        small = (size < MIN_INSTANCE_SPLATS) & (cells < MIN_INSTANCE_CELLS)
        joined[valid & small[np.maximum(joined, 0)]] = -1
        labels[level] = joined + 1
        stats[f"level{level}Instances"] = int(np.unique(joined[joined >= 0]).size)
    unseen = np.flatnonzero(~seen)
    if unseen.size and seen.any():
        seen_index = np.flatnonzero(seen)
        bound = np.full(unseen.size, FILL_CELLS * edge)
        if cell_reach is not None:
            bound = np.maximum(bound, np.asarray(cell_reach, np.float64)[unseen])
        distance, nearest = cKDTree(centroids[seen_index]).query(
            centroids[unseen], k=1, distance_upper_bound=float(bound.max()), workers=-1
        )
        ok = np.isfinite(distance) & (distance <= bound)
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
        n_cells_of = np.bincount(inverse)
        node_of_key = np.zeros(keys.size, np.int64)
        for k in range(keys.size):
            parent = int(keys[k] // (int(labels[level].max()) + 1))
            if sizes[k] < MIN_INSTANCE_SPLATS and n_cells_of[k] < MIN_INSTANCE_CELLS:
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
    #: Its broad scene category when described (`describe`); else the document's rule.
    category: str | None = None


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
    """Leaf values (instances along axis 0) folded into every ancestor, deepest first (one
    vectorised step per depth)."""
    out = values.copy()
    fold = {"sum": np.add, "min": np.minimum}.get(how, np.maximum)
    for depth in range(int(level.max()) if level.size else 0, 0, -1):
        nodes = np.flatnonzero((level == depth) & (parent > 0))
        fold.at(out, parent[nodes] - 1, out[nodes])
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


#: The crops an instance can be described by: from the views' images (`CROP_SPECS`), the
#: same from the CPU's point samples (`-samples`, when the views were drawn by another
#: renderer), and portraits of its own splats.
CROP_KINDS = (
    "context", "plain", "alone", "black", "wide", "wide-black",
    "context-samples", "plain-samples", "alone-samples", "black-samples", "wide-samples",
    "wide-black-samples", "portrait",
)  # fmt: skip
DESCRIBE_KINDS = ("context", "black")
#: The wide crop: this many times the instance's box, at least `WIDE_MIN_PX` a side.
WIDE_FACTOR = 3.0
WIDE_MIN_PX = 224
#: Crops embedded at a time (bounds what `describe` holds).
EMBED_CHUNK = 512
#: A view is used for an instance's crops when it shows at least this share of the pixels
#: of the instance's best view (and `MIN_VIEW_PX`); a view where the instance runs off the
#: frame counts its pixels at `TRUNCATED_WEIGHT`.
CROP_MIN_SHARE = 0.25
TRUNCATED_WEIGHT = 0.5
#: The grey of portraits' backgrounds (and of an empty crop).
BACKGROUND_GREY = 0.5
#: Crops are scaled down to at most this side (the image model sees 224).
CROP_MAX_SIDE = 256
#: Portraits: an instance's own splats alone on grey, square, framed to its box.
PORTRAIT_PX = 224
PORTRAIT_FOV_DEG = 40.0
#: The category: `CATEGORY_HEAD_WEIGHT` of the zero-shot head over the categories' prompts,
#: the rest from the `CATEGORY_TAGS` best labels' probabilities summed per category.
CATEGORY_HEAD_WEIGHT = 0.5
CATEGORY_TAGS = 10


@runtime_checkable
class SplatRenderer(Protocol):
    """Draws splats for a camera: `splat_render.GsplatRenderer` on a GPU, `CpuRenderer`."""

    name: str

    def __call__(
        self,
        splats: Splats,
        camera: Camera,
        *,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rows: np.ndarray | None = None,
    ) -> object:
        """A `splat_render.Frame` of `splats` (only `rows` of them, when given)."""
        ...


class CpuRenderer:
    """`splat_render.render` as a `SplatRenderer` (tests; the views' own renderer)."""

    name = "cpu"

    def __call__(self, splats, camera, *, background=(0.0, 0.0, 0.0), rows=None):
        return render(splats if rows is None else splats.take(rows), camera, background=background)


def make_renderer(spec: str) -> SplatRenderer | None:
    """`cpu` (None: the views keep the point-sampled image their labels come with) or
    `gsplat` (`splat_render.GsplatRenderer`, CUDA: the views are rasterized, as a viewer
    draws them, for the mask and image models; labels still come from the CPU's samples)."""
    if spec == "cpu":
        return None
    if spec == "gsplat":
        from splat_render import GsplatRenderer

        return GsplatRenderer()
    raise ValueError(f"renderer {spec!r}: 'cpu' or 'gsplat'")


def _crop_views(area: np.ndarray, boxes: np.ndarray, views: Sequence[View]) -> list[int]:
    """An instance's views for crops: up to `CROP_VIEWS`, by pixels (counted at
    `TRUNCATED_WEIGHT` where its box touches the frame's edge), each with at least
    `CROP_MIN_SHARE` of the best one's pixels and `MIN_VIEW_PX`."""
    if area.size == 0:
        return []
    weight = area.astype(np.float64).copy()
    for v in np.flatnonzero(area > 0):
        h, w = views[v].cell.shape
        x0, y0, x1, y1 = boxes[v]
        if x0 <= 0 or y0 <= 0 or x1 >= w - 1 or y1 >= h - 1:
            weight[v] *= TRUNCATED_WEIGHT
    order = np.argsort(-weight, kind="stable")[:CROP_VIEWS]
    floor = max(MIN_VIEW_PX, CROP_MIN_SHARE * float(weight[order[0]]))
    return [int(v) for v in order if weight[v] >= floor and area[v] > 0]


def _square(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, ...]:
    """The box grown to a square about its centre, within the frame where it can be."""
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xa, ya = round(cx - side / 2), round(cy - side / 2)
    xa = min(max(xa, 0), max(w - int(side), 0))
    ya = min(max(ya, 0), max(h - int(side), 0))
    return xa, ya, min(w, xa + math.ceil(side)), min(h, ya + math.ceil(side))


def _shrink(image: np.ndarray) -> np.ndarray:
    """At most `CROP_MAX_SIDE` a side (area-averaged), uint8."""
    from PIL import Image

    h, w = image.shape[:2]
    if max(h, w) <= CROP_MAX_SIDE:
        return np.ascontiguousarray(image, np.uint8)
    scale = CROP_MAX_SIDE / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(np.ascontiguousarray(image, np.uint8)).resize(size, 4))


def _save_gallery(path: Path, rows: list[list[np.ndarray]], side: int = 128) -> None:
    """Rows of crops, each scaled to `side` square, as one JPEG (a check of what was seen)."""
    from PIL import Image

    width = max((len(r) for r in rows), default=1)
    sheet = Image.new("RGB", (width * side, len(rows) * side))
    for i, row in enumerate(rows):
        for j, crop in enumerate(row):
            tile = Image.fromarray(np.ascontiguousarray(crop, np.uint8)).resize((side, side))
            sheet.paste(tile, (j * side, i * side))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


#: The crops of an instance in a view, by kind: the frame (`pad`: its box padded `CROP_PAD`;
#: `tight`: its box; `wide`: `WIDE_FACTOR` times its box, at least `WIDE_MIN_PX`) and what
#: becomes of the pixels that are not the instance (scaled by a factor, or filled with a grey
#: level). All square.
CROP_SPECS: dict[str, tuple[str, str, float]] = {
    "context": ("pad", "scale", 0.4),
    "plain": ("pad", "scale", 1.0),
    "alone": ("tight", "fill", 0.5),
    "black": ("tight", "fill", 0.0),
    "wide": ("wide", "scale", 0.7),
    "wide-black": ("wide", "fill", 0.0),
}


def _crops(
    view: View,
    image: np.ndarray,
    box: np.ndarray,
    cell_id: np.ndarray,
    ids: np.ndarray,
    kinds: Sequence[str] = tuple(CROP_SPECS),
) -> dict[str, np.ndarray]:
    """Square crops of an instance in a view, one per kind of `CROP_SPECS`. The instance's
    pixels are those whose cell carries one of `ids` (it and the instances below it), closed
    over the renderer's speckle (a 3x3 closing)."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = (float(b) for b in box)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    pad_x = max(CROP_PAD * bw, (MIN_CROP_PX - bw) / 2, 0)
    pad_y = max(CROP_PAD * bh, (MIN_CROP_PX - bh) / 2, 0)
    half = max(WIDE_FACTOR * max(bw, bh), WIDE_MIN_PX) / 2
    cx, cy = (x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2
    frames = {
        "pad": _square(x0 - pad_x, y0 - pad_y, x1 + 1 + pad_x, y1 + 1 + pad_y, w, h),
        "tight": _square(x0 - 1, y0 - 1, x1 + 2, y1 + 2, w, h),
        "wide": _square(cx - half, cy - half, cx + half, cy + half, w, h),
    }
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        frame, how, value = CROP_SPECS[kind]
        xa, ya, xb, yb = frames[frame]
        owner = view.cell[ya:yb, xa:xb]
        pid = np.where(owner >= 0, cell_id[np.maximum(owner, 0)], 0)
        inside = np.isin(pid, ids).astype(np.uint8)
        inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        crop = image[ya:yb, xa:xb].astype(np.float64)
        rest = value * crop if how == "scale" else np.full_like(crop, 255.0 * value)
        crop = np.where(inside[..., None], crop, rest)
        if not crop.size:
            crop = np.full((1, 1, 3), 255.0 * BACKGROUND_GREY)
        out[kind] = _shrink(np.round(crop).astype(np.uint8))
    return out


def _portraits(
    lifted: Lifted,
    renderer: SplatRenderer | None,
    splats: Splats | None,
    cell: np.ndarray | None,
):
    """`portrait(ids, camera, lo, hi, centroid)`: the splats of the instances `ids` alone,
    on grey, `PORTRAIT_PX` square, from the side `camera` saw them, framed to their box
    (`lo`, `hi`) -- no occluder, no background, at a resolution of its own whatever its size
    in the view. None without a renderer."""
    if renderer is None or splats is None or cell is None:
        return None
    leaf = lifted.cell_id[np.asarray(cell)]
    order = np.argsort(leaf, kind="stable")
    bounds = np.searchsorted(leaf[order], np.arange(lifted.parent.size + 2))
    grey = (BACKGROUND_GREY,) * 3
    half = math.radians(PORTRAIT_FOV_DEG) / 2

    def portrait(ids, camera, lo, hi, centroid) -> np.ndarray:
        rows = np.concatenate([order[bounds[i] : bounds[i + 1]] for i in ids])
        radius = max(0.5 * float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))), 1e-3)
        towards = np.asarray(centroid, np.float64) - np.asarray(camera.centre, np.float64)
        towards /= max(float(np.linalg.norm(towards)), 1e-9)
        eye = np.asarray(centroid, np.float64) - towards * (1.05 * radius / math.sin(half))
        shot = Camera.look_at(
            eye, centroid, fov_deg=PORTRAIT_FOV_DEG, width=PORTRAIT_PX, height=PORTRAIT_PX
        )
        frame = renderer(splats, shot, background=grey, rows=np.sort(rows))
        return np.round(np.clip(frame.rgb, 0, 1) * 255).astype(np.uint8)

    return portrait


def category_scores(embedder: Embedder, embedding: np.ndarray) -> tuple[list[str], np.ndarray]:
    """The zero-shot category head: per embedding row, a softmax over the categories that
    have prompts (`scene_categories.CATEGORIES`, each its phrasings' text embeddings
    averaged), at `LOGIT_SCALE`. An embedder that scores for itself (`score_categories`)
    uses its own banks."""
    prompted = [c for c in scene_categories.CATEGORIES if c.prompts]
    ids = [c.id for c in prompted]
    own = getattr(embedder, "score_categories", None)
    if own is not None:
        return ids, np.asarray(own(embedding, [c.prompts for c in prompted]), np.float64)
    flat = [p for c in prompted for p in c.prompts]
    text = np.asarray(embedder.embed_texts(flat), np.float64).reshape(len(flat), -1)
    bank, k = [], 0
    for c in prompted:
        bank.append(text[k : k + len(c.prompts)].mean(axis=0))
        k += len(c.prompts)
    return ids, _softmax(LOGIT_SCALE * np.atleast_2d(embedding) @ _normalise(np.stack(bank)).T)


def describe(
    lifted: Lifted,
    splats: Splats,
    cell: np.ndarray,
    views: Sequence[View],
    embedder: Embedder,
    vocabulary: Sequence[str],
    *,
    renderer: SplatRenderer | None = None,
    render_splats: Splats | None = None,
    render_cell: np.ndarray | None = None,
    categories: dict[str, str] | None = None,
    kinds: Sequence[str] | None = None,
    by_kind: dict[str, np.ndarray] | None = None,
    debug_dir: Path | None = None,
) -> list[Instance]:
    """Per instance: bounds, centroid and splat counts, views, crops embedded, tags,
    properties, category and behaviour. Bounds and centroid cover the instance with its
    children; `splats` counts only the splats that carry its id (the contract's leaf level).

    Crops (`_crops`) from its best views (`_crop_views`), and with a `renderer`, a portrait
    of its own splats from each of those views' sides (`_portraits`: `render_splats`, the
    splats the views were drawn from, and their cells `render_cell`). The category is voted
    by its tags' labels (`categories`, label to category; default the committed file) and
    a zero-shot head over the categories' own prompts (`category_scores`).

    `kinds`: which crops make the embedding (`CROP_KINDS`; default `DESCRIBE_KINDS`).
    `by_kind`: filled with every kind's own embedding (each crop kind is then made), to
    compare them. `debug_dir`: a sheet of some instances' crops is written there."""
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
        w = view.cell.shape[1]
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

    # Only what a crop can show is described: an instance whose best view gives it fewer
    # than `DESCRIBE_MIN_PX` pixels (with its children) keeps no embedding and no tags, and
    # takes its properties from its nearest described ancestor.
    described = area.max(axis=0) >= DESCRIBE_MIN_PX if len(views) else np.zeros(n, bool)
    # Per instance, the leaf ids below it (itself included): what its pixels and splats are.
    subtree: list[list[int]] = [[] for _ in range(n)]
    for k, chain in enumerate(_ancestors(lifted.parent)):
        for a in chain:
            subtree[a].append(k + 1)
    subtree_ids = [np.asarray(ids, np.int64) for ids in subtree]
    portrait = _portraits(lifted, renderer, render_splats, render_cell)
    dim = int(embedder.dim)
    wanted = set(DESCRIBE_KINDS if kinds is None else kinds)
    collect = wanted | (set(CROP_KINDS) if by_kind is not None else set())
    sums: dict[str, np.ndarray] = {}
    crops: list[np.ndarray] = []
    owner_of_crop: list[tuple[str, int]] = []
    shown = set(np.flatnonzero(described)[:: max(1, int(described.sum()) // 48)].tolist())
    gallery: list[list[np.ndarray]] = []

    def add(kind: str, k: int, crop: np.ndarray) -> None:
        if kind in collect:
            crops.append(crop)
            owner_of_crop.append((kind, k))
            if debug_dir is not None and k in shown:
                gallery[-1].append(crop)

    def flush() -> None:
        if crops:
            rows = np.asarray(embedder.embed_images(crops), np.float64).reshape(len(crops), dim)
            for (kind, k), row in zip(owner_of_crop, rows, strict=True):
                sums.setdefault(kind, np.zeros((n, dim)))[k] += row
            crops.clear()
            owner_of_crop.clear()

    for k in np.flatnonzero(described):
        if debug_dir is not None and k in shown:
            gallery.append([])
        for v in _crop_views(area[:, k], boxes[:, k], views):
            view = views[v]
            drawn = [c for c in CROP_SPECS if c in collect]
            for kind, crop in _crops(
                view, view.rgb, boxes[v, k], lifted.cell_id, subtree_ids[k], drawn
            ).items():
                add(kind, int(k), crop)
            sampled = [c for c in CROP_SPECS if f"{c}-samples" in collect]
            if view.samples is not None and sampled:
                for kind, crop in _crops(
                    view, view.samples, boxes[v, k], lifted.cell_id, subtree_ids[k], sampled
                ).items():
                    add(f"{kind}-samples", int(k), crop)
            if portrait is not None and "portrait" in collect:
                shot = portrait(subtree_ids[k], view.camera, lo[k], hi[k], centroid[k])
                add("portrait", int(k), shot)
        if len(crops) >= EMBED_CHUNK:
            flush()
    flush()
    # Each kind weighs the same, however many crops of it there were.
    parts = [_normalise(sums[c]) for c in sorted(wanted) if c in sums]
    embedding = _normalise(sum(parts, np.zeros((n, dim))))
    if by_kind is not None:
        by_kind.update({kind: _normalise(rows) for kind, rows in sums.items()})
    if debug_dir is not None and gallery:
        _save_gallery(debug_dir / "crops.jpg", gallery)

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

    has_embedding = np.any(embedding != 0, axis=1)
    category = _categories(embedder, embedding, has_embedding, tag_scores, words, categories)
    # Properties of what was not described: its nearest described ancestor's (parents come
    # first in id order), else none.
    source = np.where(has_embedding, np.arange(n), -1)
    for k in np.argsort(lifted.level, kind="stable"):
        if source[k] < 0 and lifted.parent[k]:
            source[k] = source[lifted.parent[k] - 1]
    instances: list[Instance] = []
    for k in range(n):
        tags: list[dict[str, object]] = []
        if words and has_embedding[k]:
            top = np.argsort(-tag_scores[k], kind="stable")[:TAGS_TOP_K]
            tags = [{"label": words[t], "score": round(float(tag_scores[k, t]), 4)} for t in top]
        properties = {
            name: round(float(property_scores[source[k], j]) if source[k] >= 0 else 0.0, 4)
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
                category=category[k],
            )
        )
    return instances


def _categories(
    embedder: Embedder,
    embedding: np.ndarray,
    described: np.ndarray,
    tag_scores: np.ndarray,
    words: Sequence[str],
    labels: dict[str, str] | None,
    head_weight: float = CATEGORY_HEAD_WEIGHT,
) -> list[str | None]:
    """Per instance, its category when described: `CATEGORY_HEAD_WEIGHT` of the zero-shot
    head (`category_scores`) and the rest from its `CATEGORY_TAGS` best labels, each
    label's probability added to its category and the sum normalised (the head alone when
    none of them has a category). None for what is not described."""
    out: list[str | None] = [None] * len(embedding)
    ids, rows, head, tags, total = _category_parts(
        embedder, embedding, described, tag_scores, words, labels
    )
    if rows.size == 0:
        return out
    mixed = np.where(total > 0, head_weight * head + (1 - head_weight) * tags, head)
    best = np.argmax(mixed, axis=1)
    for r, k in enumerate(rows):
        out[k] = ids[int(best[r])]
    return out


def _category_parts(
    embedder: Embedder,
    embedding: np.ndarray,
    described: np.ndarray,
    tag_scores: np.ndarray,
    words: Sequence[str],
    labels: dict[str, str] | None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`_categories`' inputs: the category ids, the described rows, the head's and the
    tags' distributions over the categories (rows), and the tags' total per row."""
    rows = np.flatnonzero(described)
    if rows.size == 0:
        return [], rows, np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 1))
    labels = scene_categories.load() if labels is None else labels
    ids, head = category_scores(embedder, embedding[rows])
    column = {c: j for j, c in enumerate(ids)}
    votes = np.zeros_like(head)
    if words and tag_scores.shape[1]:
        top = np.argsort(-tag_scores[rows], axis=1, kind="stable")[:, :CATEGORY_TAGS]
        for r in range(rows.size):
            for t in top[r]:
                c = labels.get(words[t]) or labels.get(words[t].strip().lower())
                if c in column:
                    votes[r, column[c]] += tag_scores[rows[r], t]
    total = votes.sum(axis=1, keepdims=True)
    tags = np.divide(votes, total, out=np.zeros_like(votes), where=total > 0)
    return ids, rows, head, tags, total


def _tag_scores(embedder: Embedder, embedding: np.ndarray, words: Sequence[str]) -> np.ndarray:
    """(n, len(words)) tag probabilities, as `describe` scores them."""
    own = getattr(embedder, "score_tags", None)
    if own is not None:
        return np.asarray(own(embedding, list(words)), np.float64)
    text = np.asarray(embedder.embed_texts(list(words)), np.float64)
    return _softmax(LOGIT_SCALE * embedding @ text.T)


#: `describe_variants`: crop kinds compared, and the category head's weights.
VARIANT_KINDS = (
    ("context",), ("plain",), ("alone",), ("black",), ("wide",), ("wide-black",),
    ("context-samples",), ("plain-samples",), ("black-samples",), ("wide-samples",),
    ("context", "alone"), ("context", "black"), ("plain", "black"), ("wide", "black"),
    ("wide", "wide-black"), ("context", "wide", "black"), ("context", "wide", "alone"),
    ("plain", "wide", "black"), ("context", "black", "wide-black"),
    ("context", "context-samples", "black", "black-samples"),
    ("plain-samples", "black-samples"), ("plain", "plain-samples", "black"),
    ("context", "plain-samples", "black"), ("wide", "plain-samples", "black"),
)  # fmt: skip
VARIANT_HEAD_WEIGHTS = (0.0, 0.5)


def describe_variants(
    embedder: Embedder,
    vocabulary: Sequence[str],
    by_kind: dict[str, np.ndarray],
    distributions: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, list]]:
    """Per variant (`VARIANT_KINDS` x `VARIANT_HEAD_WEIGHTS`, named `kinds@weight`), each
    instance's category and best label as `describe` would give them from those crops: to
    measure which crops describe a scan best, on the same instances. `distributions`, when
    given, gets per single crop kind its described rows' head and tag distributions over
    the categories (`<kind>/head`, `<kind>/tags`, `<kind>/rows`) and `categories`."""
    words = list(vocabulary)
    labels = scene_categories.load()
    out: dict[str, dict[str, list]] = {}
    if distributions is not None:
        for kind, embedding in by_kind.items():
            has = np.any(embedding != 0, axis=1)
            scores = _tag_scores(embedder, embedding, words)
            ids, rows, head, tags, _ = _category_parts(
                embedder, embedding, has, scores, words, labels
            )
            distributions["categories"] = np.asarray(ids)
            distributions[f"{kind}/rows"] = rows.astype(np.int32)
            distributions[f"{kind}/head"] = head.astype(np.float16)
            distributions[f"{kind}/tags"] = tags.astype(np.float16)
    for kinds in VARIANT_KINDS:
        if not all(k in by_kind for k in kinds):
            continue
        embedding = _normalise(sum(by_kind[k] for k in kinds))
        has = np.any(embedding != 0, axis=1)
        scores = _tag_scores(embedder, embedding, words)
        top = [words[int(t)] if h else None for t, h in zip(scores.argmax(axis=1), has)]
        for weight in VARIANT_HEAD_WEIGHTS:
            category = _categories(embedder, embedding, has, scores, words, labels, weight)
            out[f"{'+'.join(kinds)}@{weight:g}"] = {"category": category, "top": top}
    return out


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
    categories: dict[str, str] | None = None,
) -> dict:
    """`instances.json` v1 (docs/SCENE_OBJECTS.md §4), keys in the contract's order. Each
    instance also carries its broad scene `category` (`scene_categories.instance_categories`
    over `categories`, label to category id; default the committed `data/categories.json`),
    which the viewer otherwise works out from the tags itself. A described instance's own
    `category` (`describe`: its tags and the category head) is kept; the rest follow from
    it by the same rule."""
    records = [
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
            "category": scene_categories.OTHER,
        }
        for i in instances
    ]
    labels = scene_categories.load() if categories is None else categories
    given = {i.id: i.category for i in instances if i.category is not None}
    assigned = scene_categories.instance_categories(records, labels, given=given)
    for record in records:
        record["category"] = assigned.get(int(record["id"]), scene_categories.OTHER)
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
        "instances": records,
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


def render_instances(
    splats: Splats,
    splat_id: np.ndarray,
    camera: Camera | Sequence[Camera],
    out: Path,
    index: SplatIndex | None = None,
) -> np.ndarray:
    """The scan beside itself coloured by instance id, one row per camera, as a PNG: what a
    person checks."""
    from PIL import Image

    rows = []
    for cam in [camera] if isinstance(camera, Camera) else camera:
        frame = render(splats, cam, labels=splat_id.astype(np.int64), index=index)
        colour = _colours(np.maximum(frame.label, 0)) * frame.alpha[..., None]
        rows.append(np.concatenate([frame.rgb, colour], axis=1))
    image = np.concatenate(rows, axis=0)
    pixels = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(out)
    return pixels


def top_level(parent: np.ndarray) -> np.ndarray:
    """Per id (index 0: none), its level-0 ancestor (parents have lower ids)."""
    top = np.arange(parent.size + 1)
    for k in range(1, parent.size + 1):
        if parent[k - 1]:
            top[k] = top[parent[k - 1]]
    return top


def check_cameras(views: Sequence[View], count: int = 4) -> list[Camera]:
    """Cameras for `render_instances`: the first view (the whole scan) and `count - 1` more
    spread through the plan -- its local views (a far plane) when it has them."""
    if not views:
        return []
    pool = [v.camera for v in views if math.isfinite(v.camera.far)] or [v.camera for v in views[1:]]
    n = min(count - 1, len(pool))
    return [views[0].camera] + [pool[(2 * k + 1) * len(pool) // (2 * n)] for k in range(n)]


def collect_votes(
    views: Sequence[View], source: MaskSource, n_cells: int
) -> tuple[list[_Votes], int, list[list[Mask]]]:
    """Masks of every view, and their votes. Levels are counted from the masks."""
    all_masks = [source.masks(view.rgb) for view in views]
    levels = max((m.level for masks in all_masks for m in masks), default=0) + 1
    return [vote(v, m, n_cells, levels) for v, m in zip(views, all_masks)], levels, all_masks


def _cache_key(camera: Camera, n_cells: int, tag: str = "") -> str:
    text = json.dumps([camera.to_json(), n_cells] + ([tag] if tag else []), sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def cached_view(
    cache: Path | None,
    splats: Splats,
    camera: Camera,
    cells: np.ndarray,
    n_cells: int,
    index: SplatIndex | None = None,
    tag: str = "",
) -> View:
    """`render_view`, kept in `cache` (keyed by the camera, the cell count and `tag`, what
    else chose the splats) so a run that is stopped picks up where it was."""
    if cache is None:
        return render_view(splats, camera, cells, index)
    path = cache / f"view-{_cache_key(camera, n_cells, tag)}.npz"
    if path.exists():
        with np.load(path) as z:
            return View(camera, z["rgb"], z["cell"], z["purity"])
    view = render_view(splats, camera, cells, index)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, rgb=view.rgb, cell=view.cell, purity=view.purity)
    tmp.replace(path)
    return view


def cached_raster(
    cache: Path | None,
    renderer: SplatRenderer,
    splats: Splats,
    camera: Camera,
    n_cells: int,
    tag: str = "",
    index: SplatIndex | None = None,
) -> np.ndarray:
    """The view's image drawn by `renderer` (uint8), kept in `cache` like the view;
    `index` hands it only the gaussians that can reach the frame."""
    path = None
    if cache is not None:
        path = cache / f"raster-{_cache_key(camera, n_cells, tag)}-{renderer.name}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["rgb"]
    rows = None if index is None else index.visible(camera)
    frame = renderer(splats, camera, rows=rows)
    rgb = np.round(np.clip(frame.rgb, 0, 1) * 255).astype(np.uint8)
    if path is not None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, rgb=rgb)
        tmp.replace(path)
    return rgb


def cached_masks(
    cache: Path | None, view: View, source: MaskSource, n_cells: int, tag: str = ""
) -> list[Mask]:
    """`source.masks(view.rgb)`, kept in `cache` beside the view (per mask source, and per
    `tag`: what drew the image)."""
    if cache is None:
        return source.masks(view.rgb)
    name = hashlib.sha1(str(getattr(source, "name", "")).encode()).hexdigest()[:8]
    path = cache / f"masks-{_cache_key(view.camera, n_cells, tag)}-{name}.npz"
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
    #: Per cell, how far its unseen splats reach (`lift`'s `cell_reach`).
    reach: np.ndarray | None = None

    def relift(self) -> Lifted:
        """`lift` again from the same votes (it is deterministic)."""
        assert self.cells is not None
        _, centroids, counts, edge = self.cells
        levels = int(self.lifted.stats["levels"])
        return lift(self.votes, centroids, counts, edge, levels, cell_reach=self.reach)


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
    workers: int | None = None,
    max_views: int = MAX_VIEWS,
    renderer: SplatRenderer | None = None,
    max_scale_m: float | None = None,
    coverage_rounds: int = 0,
    coverage_budget: int = COVERAGE_VIEWS,
    describe_kinds: Sequence[str] | None = None,
    by_kind: dict[str, np.ndarray] | None = None,
    debug_dir: Path | None = None,
) -> Segmentation:
    """Cells, views, masks, votes, lifting and meaning, for a scan held in memory.

    `source_factory(cameras)` builds a mask source that needs the cameras (`OracleMasks`;
    called again for each coverage round's cameras).
    `cache`: a directory where each view and its masks are kept as they are made, so a
    stopped run resumes (views and masks are most of the time on a large scan).
    `progress(message)` is told as each view is done. `workers`: render processes
    (default: the CPUs this process may use).

    `renderer` (`make_renderer("gsplat")`): the image of every view the mask and image
    models see is drawn by it (labels still come from the CPU renderer's samples, the same
    camera), and the instances' portraits too (`describe`). `max_scale_m`: the views leave
    out gaussians larger than this (largest axis) -- the floaters at a capture's edge, which
    a rasterizer draws as blobs over a view; they take their cell's instance, or the nearest
    seen cell's within their own reach (`lift`'s `cell_reach`). `coverage_rounds`: after
    the lift, up to this many rounds of `coverage_views` (at most `coverage_budget` views
    each) aimed at what is still without an instance, each followed by a lift of all the
    votes; a round is not run once less than `COVERAGE_MIN_SHARE` of the splats is left.
    `describe_kinds`, `by_kind`, `debug_dir`: `describe`'s `kinds`, `by_kind` and
    `debug_dir` (which also gets a sheet of some views, as drawn and as sampled)."""
    timings: dict[str, float] = {}
    mark = time.perf_counter()
    cell, centroids, counts, edge = cells or supervoxels(splats.positions)
    timings["cellsS"] = time.perf_counter() - mark
    mark = time.perf_counter()
    largest = np.asarray(splats.scales).max(axis=1)
    if max_scale_m is None:
        view_splats, view_cell = splats, cell
    else:
        keep = np.flatnonzero(largest <= max_scale_m)
        view_splats, view_cell = splats.take(keep), cell[keep]
    index = SplatIndex.build(view_splats)
    if cameras is None:
        cameras = plan_views(
            splats.positions,
            view_count,
            observers=observer_points(splats),
            edge=edge,
            solid=centroids,
            max_views=max_views,
        )
    timings["planS"] = time.perf_counter() - mark
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    n_cells = len(centroids)
    tag = "" if max_scale_m is None else f"max{max_scale_m:g}"
    from_factory = source is None
    views: list[View] = []
    view_votes: list[_Votes] = []
    clock = {"renderS": 0.0, "rasterS": 0.0, "masksS": 0.0, "votesS": 0.0}

    def run_views(batch: Sequence[Camera]) -> None:
        # Views render in forked workers while this process masks the ones already done
        # (the mask model's GPU context starts after the first fork). Each view's masks
        # become its votes and are dropped; the views are kept for `describe`.
        nonlocal source
        if from_factory:
            source = source_factory(list(batch))
        first = len(views)
        mark = time.perf_counter()
        rendered = render_views(
            view_splats, batch, view_cell, index=index,
            workers=workers or default_workers(), cache=cache, tag=tag,
        )  # fmt: skip
        for k, view in enumerate(rendered):
            now = time.perf_counter()
            clock["renderS"] += now - mark
            if renderer is not None:
                view = View(
                    view.camera,
                    cached_raster(cache, renderer, view_splats, view.camera, n_cells, tag, index),
                    view.cell,
                    view.purity,
                    samples=view.rgb,
                )
            rastered = time.perf_counter()
            clock["rasterS"] += rastered - now
            views.append(view)
            if progress:
                progress(f"view {first + k + 1}/{first + len(batch)} rendered")
            masks = cached_masks(
                cache, view, source, n_cells, tag + ("" if renderer is None else renderer.name)
            )
            voted = time.perf_counter()
            view_levels = max((m.level for m in masks), default=0) + 1
            view_votes.append(vote(view, masks, n_cells, view_levels))
            if progress:
                progress(f"view {first + k + 1}/{first + len(batch)} masked ({len(masks)} masks)")
            mark = time.perf_counter()
            clock["masksS"] += voted - rastered
            clock["votesS"] += mark - voted

    # Unseen cells reach as far as twice their largest gaussian (a floater left out of the
    # views carries the instance of what it hangs over).
    reach = np.zeros(n_cells)
    np.maximum.at(reach, cell, 2.0 * largest)

    def lift_all() -> tuple[Lifted, int]:
        levels = max((v.masks.shape[0] for v in view_votes), default=1)
        votes = [_pad_levels(v, levels) for v in view_votes]
        lifted = lift(votes, centroids, counts, edge, levels, cell_reach=reach)
        lifted.stats["cellEdgeM"] = round(edge, 4)
        lifted.stats["levels"] = levels
        return lifted, levels

    run_views(cameras)
    mark = time.perf_counter()
    lifted, levels = lift_all()
    timings["liftS"] = time.perf_counter() - mark
    coverage: list[dict[str, float]] = []
    for _ in range(coverage_rounds):
        missing = np.where(lifted.cell_id == 0, counts, 0)
        share = float(missing.sum()) / max(float(counts.sum()), 1.0)
        if share < COVERAGE_MIN_SHARE:
            break
        mark = time.perf_counter()
        extra = coverage_views(centroids, missing, edge, budget=coverage_budget)
        timings["planS"] += time.perf_counter() - mark
        if not extra:
            break
        coverage.append({"unassignedShare": round(share, 4), "views": len(extra)})
        run_views(extra)
        mark = time.perf_counter()
        lifted, levels = lift_all()
        timings["liftS"] += time.perf_counter() - mark
    final = np.where(lifted.cell_id == 0, counts, 0).sum() / max(float(counts.sum()), 1.0)
    lifted.stats["coverageRounds"] = coverage
    lifted.stats["unassignedShare"] = round(float(final), 4)
    lifted.stats["views"] = len(views)
    # Rendering overlaps masking: renderS is the time spent waiting for views.
    timings.update(clock)
    mark = time.perf_counter()
    votes = [_pad_levels(v, levels) for v in view_votes]
    if debug_dir is not None and views:
        picked = views[:: max(1, len(views) // 24)]
        _save_gallery(
            debug_dir / "views.jpg",
            [[v.rgb] + ([v.samples] if v.samples is not None else []) for v in picked],
            side=256,
        )
    instances = describe(
        lifted, splats, cell, views, embedder, vocabulary,
        renderer=renderer, render_splats=view_splats, render_cell=view_cell,
        kinds=describe_kinds, by_kind=by_kind, debug_dir=debug_dir,
    )  # fmt: skip
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
        reach,
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
    parser.add_argument(
        "--views", type=int, default=VIEW_COUNT, help="views of the whole scan (rings, observers)"
    )
    parser.add_argument(
        "--max-views", type=int, default=MAX_VIEWS, help="at most this many with local views"
    )
    parser.add_argument(
        "--renderer",
        choices=("cpu", "gsplat"),
        default="cpu",
        help="what draws the views' images and the portraits (gsplat: a CUDA GPU)",
    )
    parser.add_argument(
        "--max-scale-m",
        type=float,
        default=None,
        help="leave gaussians larger than this (largest axis, metres) out of the views",
    )
    parser.add_argument(
        "--coverage-rounds",
        type=int,
        default=COVERAGE_ROUNDS,
        help="rounds of views aimed at what is still without an instance",
    )
    parser.add_argument(
        "--coverage-views", type=int, default=COVERAGE_VIEWS, help="at most this many a round"
    )
    parser.add_argument(
        "--describe-kinds",
        default=",".join(DESCRIBE_KINDS),
        help=f"crops an instance is described by, of {', '.join(CROP_KINDS)}",
    )
    parser.add_argument(
        "--variants", type=Path, default=None, help="write describe_variants here (JSON)"
    )
    parser.add_argument(
        "--debug-dir", type=Path, default=None, help="sheets of some views and crops"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="render processes (default: usable CPUs)"
    )
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
    by_kind: dict[str, np.ndarray] | None = {} if args.variants else None
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
        workers=args.workers,
        max_views=args.max_views,
        source_factory=factory,
        cache=args.cache,
        progress=lambda message: print(message, flush=True),
        renderer=make_renderer(args.renderer),
        max_scale_m=args.max_scale_m,
        coverage_rounds=args.coverage_rounds,
        coverage_budget=args.coverage_views,
        describe_kinds=[k for k in args.describe_kinds.split(",") if k],
        by_kind=by_kind,
        debug_dir=args.debug_dir,
    )
    if args.variants:
        distributions: dict[str, np.ndarray] = {}
        variants = describe_variants(embedder, vocabulary, by_kind, distributions)
        args.variants.write_text(json.dumps(variants, separators=(",", ":")), encoding="utf-8")
        np.savez_compressed(args.variants.with_suffix(".npz"), **distributions)
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
    # Merged splats take what most of the leaf splats near them are, not only what all of
    # them share (which left the coarse levels of detail almost without ids).
    tiles = rebind_instances.rebind(args.tiles, tiles)
    document = instances_document(
        result.instances,
        tiles,
        embedding_model=embedder.name,
        dim=int(embedder.dim),
        vocabulary_model=embedder.name,
        vocabulary_size=len(vocabulary),
    )
    document["tilesEncoding"] = rebind_instances.TILES_ENCODING
    out = args.out or args.tiles
    write_instances(out, document, result.instances)
    if out.resolve() == args.tiles.resolve():
        link_instances(args.tiles / "tileset.json", len(result.instances))
    if args.cache:
        (args.cache / "cameras.json").write_text(
            json.dumps([v.camera.to_json() for v in result.views]), encoding="utf-8"
        )
    if args.save_dir:
        from PIL import Image

        args.save_dir.mkdir(parents=True, exist_ok=True)
        for k, view in enumerate(result.views):
            Image.fromarray(view.rgb).save(args.save_dir / f"view_{k:03d}.png")
        (args.save_dir / "cameras.json").write_text(
            json.dumps([v.camera.to_json() for v in result.views], indent=1), encoding="utf-8"
        )
    if args.render_instances:
        # Coloured by object (each splat's top-level instance), from a few viewpoints.
        objects = top_level(result.lifted.parent)[result.splat_id]
        cameras = check_cameras(result.views)
        render_instances(
            splats, objects, cameras, args.render_instances, index=SplatIndex.build(splats)
        )
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
