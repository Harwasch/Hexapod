"""Bake-off candidate A, "ground first": the ground found from geometry before any mask is
lifted, things lifted above it, each thing refined from close views of its base, the ground
classified into cover classes, and things named by a vision-language model
(docs/SCENE_OBJECTS.md §3b; the research notes' §1.3).

1. **Ground pass** (`ground_pass`, CPU): the terrain, every splat's height above it and
   its label. The voxel cells are cut at the ground surface (`ground_pass.split_cells`),
   so no cell holds a spool's flange and the soil beside it.
2. **Things pass** (`segment_scene.segment`, unchanged but for two hooks): views, SAM 2.1
   *large* masks, votes, the lift -- with the ground's cells kept out of the object graph
   and out of `_absorb` (`exclude`): an object can never absorb ground.
3. **Refine pass** (`refine`, the hook): each top-level object, largest first (up to
   `REFINE_OBJECTS`), gets `REFINE_VIEWS` views around its base, two of them low; SAM is
   prompted with the object's box (extended down to the terrain) in each, which gives the
   whole object in one mask. Over those views: a ground cell inside the object's box that
   is in its mask in most views it is seen in is **claimed** by the object (a flange, a
   pumpkin's bottom in the hay); a cell of the object almost never in its masks is
   **released**; a smaller top-level object that is mostly inside its masks becomes **a part
   of it** (the spool's planks and flanges under one spool).
4. **Stuff pass**: low objects that describe as ground cover (`STUFF_CATEGORIES`, at most
   `STUFF_LAYERS` ground layers high: a tuft of grass, a clump of hay) are demoted to the
   ground. Every ground splat is classified into the cover classes of
   `data/ground_cover.json` -- grass, gravel, dirt, trail, hay, forest floor, ... -- by
   the image-text model on tiles of the views where its cell is seen (`cover_votes`),
   smoothed among neighbouring ground cells, and split into connected regions.
5. **Ground in `instances.json`**: one top-level instance per cover class present (category
   `ground`, `name` "Grass", "Hay", ...), its regions as its children. The objects panel lists
   them under "Ground & soil" like any object; they hide, highlight and select like any
   object. Things stay as they were lifted, never under the ground.
6. **Naming** (`Namer`, Qwen3-VL, Apache-2.0): each top-level thing (and the parts of the
   largest) shown in context and alone; its answer's `name` is written as the instance's
   `name` (`nameSource: "vlm"`), with `wholeOrPart`, `material` and `movable`.

Records also carry `kind` (`thing` | `ground`) and `scaleM` (half the bounds' diagonal).
The file is still `hexapod.instances` v1: readers that know none of the new fields read it
as before.

Usage (as `segment_scene.py`; the result is a variant, so `--out` is required and the
tileset is never linked):
    python segment_ground_first.py TILESET.json TILES_DIR --out OUT \\
        --masks segment_models:Sam2LargeMasks --boxes segment_models:Sam2BoxMasks \\
        --embedder segment_models:SiglipEmbedder --namer segment_models:QwenNamer \\
        --vocabulary data/open_vocabulary.txt [--renderer gsplat] [--cache DIR]
    python segment_ground_first.py yard/source/splat.ply yard/splat --out OUT \\
        --truth yard/source/labels.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import ground_pass as gp
import segment_scene as ss
from splat_render import Camera, Splats, render

__all__ = [
    "BoxMaskSource",
    "CoverClass",
    "GroundFirst",
    "Namer",
    "OracleBoxMasks",
    "cover_classes",
    "refine_cameras",
    "run",
]

VARIANT = "ground-first"
VARIANT_LABEL = "A · Ground first"
VARIANT_ABOUT = (
    "The ground is found from the scan's shape first and split into cover classes (grass, "
    "hay, gravel, ...); objects are cut out above it with SAM 2.1 large, checked from low "
    "views around each object's base, and named by a vision-language model."
)
COVER_FILE = Path(__file__).resolve().parent / "data" / "ground_cover.json"
GROUND_CATEGORY = "ground"

#: Refine pass: at most this many top-level objects, of at least this many splats ...
REFINE_OBJECTS = 32
REFINE_MIN_SPLATS = 150
#: ... each seen from `REFINE_AZIMUTHS` sides at each of `REFINE_ELEVATIONS_DEG` (the low
#: one shows where it meets the ground), the box spanning `REFINE_FILL` of the frame.
REFINE_AZIMUTHS = 5
REFINE_ELEVATIONS_DEG = (10.0, 40.0)
REFINE_FILL = 0.75
REFINE_FOV_DEG = 50.0
#: The object's box, padded by this share of its size each side (and two cell edges).
REFINE_PAD = 0.12
#: Of its refine views, this many per object are kept for `describe` and naming.
REFINE_KEEP = 4
#: A ground (or unassigned) cell inside the box is claimed when it is in the object's mask
#: in at least `CLAIM_SHARE` of the refine views it is seen in, and seen in `CLAIM_VIEWS`.
CLAIM_SHARE = 0.6
CLAIM_VIEWS = 3
#: An object's cell is released when in at most `RELEASE_SHARE` of `RELEASE_VIEWS`+ views.
RELEASE_SHARE = 0.1
RELEASE_VIEWS = 4
#: A smaller top-level object becomes a part when `MERGE_SHARE` of its cell-views (at least
#: `MERGE_VIEWS` x its cells... at least `MERGE_VIEWS`) are in the larger one's masks, and
#: `MERGE_SEEN` of its cells were seen at all.
MERGE_SHARE = 0.7
MERGE_VIEWS = 6
MERGE_SEEN = 0.3

#: Stuff pass: a top-level object whose splats' 90th-percentile height above ground is at
#: most this many ground layers (and `STUFF_MIN_M`) and whose category is one of these
#: is ground cover, not a thing; so is an undescribed speck that low.
STUFF_LAYERS = 2.0
STUFF_MIN_M = 0.12
STUFF_CATEGORIES = frozenset({"grass", "ground", "paths", "snow", "water"})
#: Cover votes (`cover_votes`): ground pixels in no mask are voted by square tiles of this
#: many pixels when at least `COVER_MIN_SHARE` of a tile is ground.
COVER_TILE = 64
COVER_MIN_SHARE = 0.4
#: ... their votes weighed this much against a mask's (a tile mixes what it straddles).
COVER_TILE_WEIGHT = 0.25
COVER_CHUNK = 256
#: A mask's ground pixels are pooled when there are at least this many, in at most this
#: many windows.
COVER_MIN_PIXELS = 48
COVER_MAX_WINDOWS = 6
#: Mask region keys: level x this + the mask's index in its view.
MASK_KEY = 1 << 20
#: Smoothing: each ground cell takes the mean of its `COVER_NEIGHBOURS` nearest ground
#: cells' class probabilities, `COVER_ROUNDS` times.
COVER_NEIGHBOURS = 16
COVER_ROUNDS = 2
#: A region (connected cells of one class) under this share of the ground's splats, or
#: `COVER_MIN_REGION` splats, takes the class most of its neighbours have.
COVER_MIN_REGION = 400
COVER_MIN_REGION_SHARE = 0.01
#: A class needs this share of the ground's splats to be an instance (else its regions
#: take their neighbours' class).
COVER_MIN_CLASS_SHARE = 0.01
#: Naming: at most this many instances, the top-level things first (largest first), then
#: the first-level parts of the `NAME_PART_OBJECTS` largest.
NAME_MAX = 60
NAME_PART_OBJECTS = 6
NAME_PARTS_EACH = 8
#: ... asked one at a time, for at most this long (seconds) in all.
NAME_BUDGET_S = 420.0


# ------------------------------------------------------------------------------ models


@runtime_checkable
class BoxMaskSource(Protocol):
    """A mask model prompted with boxes (SAM 2 family, `segment_models.Sam2BoxMasks`)."""

    name: str

    def box_masks(self, rgb: np.ndarray, boxes: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """`boxes` (k, 4) pixels `x0, y0, x1, y1` -> per box `(mask (h, w) bool, score)`."""
        ...


@runtime_checkable
class Namer(Protocol):
    """A vision-language model naming objects from crops (`segment_models.QwenNamer`)."""

    name: str

    def name_objects(self, crops: list[list[np.ndarray]]) -> list[dict | None]:
        """Per object, `{"name", "whole", "part_of", "material", "movable"}` or None."""
        ...


class OracleBoxMasks:
    """For tests: per box, the truth object owning most of the box's middle (its middle
    half), as a mask over the whole view -- what SAM gives a box around one object. Built
    from the cameras (as `segment_scene.OracleMasks`), recognising a view by its image."""

    name = "oracle-boxes"

    def __init__(
        self, splats: Splats, labels: np.ndarray, cameras: Sequence[Camera], min_purity=0.5
    ) -> None:
        self._labels: dict[str, np.ndarray] = {}
        for camera in cameras:
            frame = render(splats, camera, labels=np.asarray(labels, np.int64))
            rgb = np.round(frame.rgb * 255).astype(np.uint8)
            self._labels[ss._image_key(rgb)] = np.where(frame.purity >= min_purity, frame.label, -1)

    def box_masks(self, rgb: np.ndarray, boxes: np.ndarray) -> list[tuple[np.ndarray, float]]:
        image = self._labels.get(ss._image_key(rgb))
        out = []
        for x0, y0, x1, y1 in np.asarray(boxes, np.float64).reshape(-1, 4):
            if image is None:
                out.append((np.zeros(rgb.shape[:2], bool), 0.0))
                continue
            w, h = x1 - x0, y1 - y0
            mid = image[
                int(y0 + h / 4) : math.ceil(y1 - h / 4) + 1,
                int(x0 + w / 4) : math.ceil(x1 - w / 4) + 1,
            ]
            found = mid[mid >= 0]
            if found.size == 0:
                out.append((np.zeros(rgb.shape[:2], bool), 0.0))
                continue
            label = np.bincount(found).argmax()
            out.append((image == label, 1.0))
        return out


# ------------------------------------------------------------------------- cover classes


@dataclass(frozen=True)
class CoverClass:
    id: str
    name: str
    prompts: tuple[str, ...]
    vegetation: bool = False


def cover_classes(path: Path = COVER_FILE) -> tuple[list[CoverClass], list[str]]:
    """The ground-cover classes and the contrast prompts (`data/ground_cover.json`)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = [
        CoverClass(c["id"], c["name"], tuple(c["prompts"]), bool(c.get("vegetation", False)))
        for c in data["classes"]
    ]
    return classes, list(data.get("contrast", []))


def _text_bank(embedder: ss.Embedder, groups: Sequence[Sequence[str]]) -> np.ndarray:
    flat = [p for g in groups for p in g]
    rows = np.asarray(embedder.embed_texts(flat), np.float64).reshape(len(flat), -1)
    out, k = [], 0
    for g in groups:
        out.append(rows[k : k + len(g)].mean(axis=0))
        k += len(g)
    return ss._normalise(np.stack(out))


# --------------------------------------------------------------------------- refine pass


def refine_cameras(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    azimuths: int = REFINE_AZIMUTHS,
    elevations: Sequence[float] = REFINE_ELEVATIONS_DEG,
    turn: float = 0.0,
    width: int = ss.VIEW_WIDTH,
    height: int = ss.VIEW_HEIGHT,
    fov_deg: float = REFINE_FOV_DEG,
) -> list[Camera]:
    """Views of the box `lo`..`hi` (its floor the terrain) from `azimuths` sides at each
    elevation, aimed a third of the way up it, from where its bounding sphere spans
    `REFINE_FILL` of the frame's height; the far plane just past the box."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-3)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + (hi[2] - lo[2]) / 3])
    vertical = math.radians(fov_deg) * height / width
    distance = radius / math.tan(REFINE_FILL * vertical / 2)
    cameras = []
    for e, elevation in enumerate(elevations):
        a = math.radians(elevation)
        for k in range(azimuths):
            azimuth = turn + 2 * math.pi * (k + 0.5 * e) / azimuths
            direction = np.array(
                [math.cos(a) * math.cos(azimuth), math.cos(a) * math.sin(azimuth), math.sin(a)]
            )
            eye = target + distance * direction
            eye[2] = max(eye[2], lo[2] + 0.05 * (hi[2] - lo[2]) + 0.02)
            cameras.append(
                Camera.look_at(
                    eye, target, fov_deg=fov_deg, width=width, height=height,
                    far=distance + 2 * radius,
                )
            )  # fmt: skip
    return cameras


def project_box(camera: Camera, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    """The 2D box (x0, y0, x1, y1) of a 3D box's corners in a view, clipped to the frame;
    None when a corner is behind the camera or the box is a few pixels."""
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    uv, depth = camera.project(corners)
    if np.any(depth <= 0.05):
        return None
    x0, y0 = np.clip(uv.min(axis=0), 0, [camera.width - 1, camera.height - 1])
    x1, y1 = np.clip(uv.max(axis=0), 0, [camera.width - 1, camera.height - 1])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.array([x0, y0, x1, y1], np.float64)


def cached_box_mask(
    cache: Path | None, source: BoxMaskSource, view: ss.View, box: np.ndarray
) -> tuple[np.ndarray, float]:
    """`source.box_masks(view.rgb, [box])[0]`, kept in `cache` (by camera, box and model)."""
    if cache is None:
        return source.box_masks(view.rgb, box[None])[0]
    key = json.dumps([view.camera.to_json(), [round(float(b), 2) for b in box], source.name])
    path = cache / f"boxmask-{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["mask"].astype(bool), float(z["score"])
    mask, score = source.box_masks(view.rgb, box[None])[0]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, mask=np.asarray(mask, bool), score=np.float64(score))
    tmp.replace(path)
    return np.asarray(mask, bool), float(score)


@dataclass
class _Plan:
    top: int
    lo: np.ndarray
    hi: np.ndarray
    cameras: list[Camera]


def _depths(parent: np.ndarray) -> np.ndarray:
    """Per instance (index id - 1), its depth below the top (parents have lower ids)."""
    level = np.zeros(parent.size, np.int64)
    for k in range(parent.size):
        if parent[k]:
            level[k] = level[parent[k] - 1] + 1
    return level


def make_refine(
    splats: Splats,
    cell: np.ndarray,
    centroids: np.ndarray,
    exclude: np.ndarray,
    ground: gp.GroundResult,
    edge: float,
    *,
    box_source: BoxMaskSource | None = None,
    box_factory: Callable[[list[Camera]], BoxMaskSource] | None = None,
    cache: Path | None = None,
    objects: int = REFINE_OBJECTS,
    progress=None,
    report: dict | None = None,
):
    """The refine pass as `segment_scene.segment`'s `refine` hook (module docstring, step
    3). `exclude` (per cell) is updated in place: a claimed ground cell is no longer
    ground. `report` gets what it did."""
    n_cells = len(centroids)
    positions = splats.positions

    def refine(lifted: ss.Lifted, render_views) -> tuple[ss.Lifted, list[ss.View]]:
        n = lifted.parent.size
        leaf = lifted.cell_id.copy()
        top_of = ss.top_level(lifted.parent)
        splat_top = top_of[leaf][cell]
        size = np.bincount(splat_top, minlength=n + 1)
        tops = [t for t in np.argsort(-size[1:], kind="stable") + 1 if lifted.parent[t - 1] == 0]
        tops = [int(t) for t in tops if size[t] >= REFINE_MIN_SPLATS][:objects]
        order = np.argsort(splat_top, kind="stable")
        bounds = np.searchsorted(splat_top[order], np.arange(n + 2))
        plans: list[_Plan] = []
        golden = math.pi * (3 - math.sqrt(5))
        for i, t in enumerate(tops):
            pts = positions[order[bounds[t] : bounds[t + 1]]]
            lo = np.percentile(pts, 1, axis=0)
            hi = np.percentile(pts, 99, axis=0)
            pad = REFINE_PAD * (hi - lo) + 2 * edge
            lo, hi = lo - pad, hi + pad
            corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
            lo[2] = min(lo[2], float(ground.terrain.at(corners).min()) - edge)
            plans.append(_Plan(t, lo, hi, refine_cameras(lo, hi, turn=i * golden)))
        cameras = [c for p in plans for c in p.cameras]
        if report is not None:
            report.update({"objects": len(plans), "views": len(cameras)})
        if not cameras:
            return lifted, []
        source = box_factory(cameras) if box_factory is not None else box_source
        if source is None:
            return lifted, []
        owner = np.repeat(np.arange(len(plans)), [len(p.cameras) for p in plans])
        rows: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in plans]
        kept: list[ss.View] = []
        for k, view in enumerate(render_views(cameras)):
            plan = plans[owner[k]]
            box = project_box(view.camera, plan.lo, plan.hi)
            if box is None:
                continue
            mask, score = cached_box_mask(cache, source, view, box)
            votes = ss.vote(view, [ss.Mask(mask, 0, score)], n_cells, 1)
            rows[owner[k]].append((votes.cells.astype(np.int64), votes.masks[0] >= 0))
            if (k - int(np.searchsorted(owner, owner[k]))) % max(
                1, len(plan.cameras) // REFINE_KEEP
            ) == 0:
                kept.append(view)
            if progress is not None and (k + 1) % 20 == 0:
                progress(f"refine view {k + 1}/{len(cameras)}")
        parent = lifted.parent.copy()
        cell_top = top_of[leaf]
        claimed = released = merged = 0
        absorbed = set()
        for p, plan in enumerate(plans):
            t = plan.top
            if t in absorbed or not rows[p]:
                continue
            cells = np.concatenate([c for c, _ in rows[p]])
            inside = np.concatenate([m for _, m in rows[p]])
            seen = np.bincount(cells, minlength=n_cells)
            held = np.bincount(cells, inside, minlength=n_cells)
            share = np.divide(held, seen, out=np.zeros(n_cells), where=seen > 0)
            # Parts: smaller top-level objects mostly inside this one's masks.
            others = np.unique(cell_top[(seen > 0) & (cell_top > t)])
            for u in others:
                if u in absorbed or parent[u - 1] != 0:
                    continue
                mine = cell_top == u
                views_u = seen[mine].sum()
                if (
                    views_u >= MERGE_VIEWS
                    and held[mine].sum() >= MERGE_SHARE * views_u
                    and (seen[mine] > 0).mean() >= MERGE_SEEN
                ):
                    parent[u - 1] = t
                    cell_top[mine] = t
                    absorbed.add(int(u))
                    merged += 1
            # Claims: ground or unassigned cells inside the box, in its masks.
            c = centroids
            in_box = np.all((c >= plan.lo) & (c <= plan.hi), axis=1)
            free = exclude | (leaf == 0)
            claim = free & in_box & (seen >= CLAIM_VIEWS) & (share >= CLAIM_SHARE)
            leaf[claim] = t
            cell_top[claim] = t
            exclude[claim] = False
            claimed += int(claim.sum())
            # Releases: its own cells almost never in its masks.
            release = (cell_top == t) & (seen >= RELEASE_VIEWS) & (share <= RELEASE_SHARE)
            leaf[release] = 0
            cell_top[release] = 0
            released += int(release.sum())
        if report is not None:
            report.update(
                {"claimedCells": claimed, "releasedCells": released, "mergedObjects": merged}
            )
        stats = dict(lifted.stats)
        stats["refine"] = dict(report or {})
        return ss.Lifted(leaf, parent, _depths(parent), stats), kept

    return refine


# ----------------------------------------------------------------------------- stuff pass


def _mask_regions(
    view: ss.View, votes: ss._Votes | None, ground_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel of a view, its ground mask region (`level * MASK_KEY + mask`, the finest
    level whose mask holds its cell; -1: none) and whether it is a ground pixel. The view's
    masks are not kept after voting, but its votes say which mask each cell was in."""
    owner = view.cell
    good = (owner >= 0) & (view.purity >= ss.MIN_PURITY)
    good &= ground_cell[np.maximum(owner, 0)]
    region = np.full(owner.shape, -1, np.int64)
    if votes is None or votes.cells.size == 0 or not good.any():
        return region, good
    cells = owner[good]
    pos = np.minimum(np.searchsorted(votes.cells, cells), votes.cells.size - 1)
    hit = votes.cells[pos] == cells
    assigned = np.full(cells.size, -1, np.int64)
    for level in reversed(range(votes.masks.shape[0])):
        mask = np.where(hit, votes.masks[level][pos], -1)
        take = (assigned < 0) & (mask >= 0)
        assigned[take] = level * MASK_KEY + mask[take]
    region[good] = assigned
    return region, good


def _windows_of(ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, ...]]:
    """Square windows over a region's pixels: one around it when it is small, else a grid
    of `COVER_TILE` x 1.5 windows over its box, each holding `COVER_MIN_SHARE` of region
    pixels (at most `COVER_MAX_WINDOWS`, spread)."""
    h, w = shape
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    if side <= 2 * COVER_TILE:
        return [ss._square(x0, y0, x1, y1, w, h)]
    step = int(1.5 * COVER_TILE)
    inside = np.zeros((y1 - y0, x1 - x0), bool)
    inside[ys - y0, xs - x0] = True
    out = []
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            window = inside[y - y0 : y - y0 + step, x - x0 : x - x0 + step]
            if window.size and window.mean() >= COVER_MIN_SHARE and window.sum() >= 64:
                out.append((x, y, min(x + step, w), min(y + step, h)))
    if len(out) > COVER_MAX_WINDOWS:
        out = [out[i] for i in np.linspace(0, len(out) - 1, COVER_MAX_WINDOWS).astype(int)]
    return out


def cover_votes(
    views: Sequence[ss.View],
    ground_cell: np.ndarray,
    embedder: ss.Embedder,
    classes: Sequence[CoverClass],
    contrast: Sequence[str],
    *,
    votes: Sequence[ss._Votes] = (),
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell, the summed probability of each cover class (n_cells, k); per cell the
    weight voted; and per class the sum of its crops' embeddings (k, dim), weighted by its
    probability.

    Pooled in masks: in a view whose mask votes are given (`votes[v]`, the lift's), the
    ground pixels of each mask (`_mask_regions`, its finest level) are cropped -- one crop,
    or windows over a large mask (`_windows_of`) -- with what is not the mask black; the
    mask's probabilities are its crops' mean, voted to every ground cell in it by its
    pixels. Ground pixels in no mask (and views without votes) are voted by `COVER_TILE`
    tiles, weighed `COVER_TILE_WEIGHT`; only the masked views vote when there are any. A
    crop's probabilities: a softmax over the classes and the contrast prompts
    (`segment_scene.LOGIT_SCALE` x cosine), the classes renormalised. The image is the
    CPU's point samples when the view has them (what `describe` reads best)."""
    n_cells = ground_cell.size
    k = len(classes)
    bank = _text_bank(embedder, [c.prompts for c in classes] + [[p] for p in contrast])
    sums = np.zeros((n_cells, k), np.float64)
    weight = np.zeros(n_cells, np.float64)
    class_embedding = np.zeros((k, int(embedder.dim)))
    crops: list[np.ndarray] = []
    group_of: list[int] = []
    groups: list[tuple[np.ndarray, np.ndarray]] = []

    def flush() -> None:
        if not crops:
            return
        emb = np.asarray(embedder.embed_images(crops), np.float64).reshape(len(crops), -1)
        p = ss._softmax(ss.LOGIT_SCALE * emb @ bank.T)[:, :k]
        p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        class_embedding[:] += p.T @ emb
        owner = np.asarray(group_of)
        mean = np.zeros((len(groups), k))
        np.add.at(mean, owner, p)
        mean /= np.maximum(np.bincount(owner, minlength=len(groups)), 1)[:, None]
        for (cells, w), row in zip(groups, mean, strict=True):
            sums[cells] += w[:, None] * row[None, :]
            weight[cells] += w
        crops.clear()
        group_of.clear()
        groups.clear()

    def add(image: np.ndarray, keep: np.ndarray, owner: np.ndarray, windows, weight=1.0) -> None:
        cells, counts = np.unique(owner[keep], return_counts=True)
        if not windows or cells.size == 0:
            return
        groups.append((cells, weight * counts.astype(np.float64)))
        for xa, ya, xb, yb in windows:
            crop = np.where(keep[ya:yb, xa:xb, None], image[ya:yb, xa:xb], 0)
            crops.append(np.ascontiguousarray(crop, np.uint8))
            group_of.append(len(groups) - 1)

    # Only the views that were masked (the lift's): the refine pass's close views of objects
    # have no masks, and tiles alone blur a path into the lawn beside it.
    for v, view in enumerate(views[: len(votes)] if votes else views):
        image = view.samples if view.samples is not None else view.rgb
        owner = view.cell
        h, w = owner.shape
        region, good = _mask_regions(view, votes[v] if v < len(votes) else None, ground_cell)
        flat = region[good]
        for r in np.unique(flat[flat >= 0]):
            keep = region == r
            ys, xs = np.nonzero(keep)
            if ys.size >= COVER_MIN_PIXELS:
                add(image, keep, owner, _windows_of(ys, xs, (h, w)))
        rest = good & (region < 0)
        for y in range(0, h - COVER_TILE + 1, COVER_TILE):
            for x in range(0, w - COVER_TILE + 1, COVER_TILE):
                tile = rest[y : y + COVER_TILE, x : x + COVER_TILE]
                if tile.mean() >= COVER_MIN_SHARE:
                    keep = np.zeros((h, w), bool)
                    keep[y : y + COVER_TILE, x : x + COVER_TILE] = tile
                    windows = [(x, y, x + COVER_TILE, y + COVER_TILE)]
                    add(image, keep, owner, windows, COVER_TILE_WEIGHT if votes else 1.0)
        if len(crops) >= COVER_CHUNK:
            flush()
        if progress is not None and (v + 1) % 50 == 0:
            progress(f"cover view {v + 1}/{len(views)}")
    flush()
    return sums, weight, class_embedding


def _smooth(
    probability: np.ndarray, points: np.ndarray, neighbours: int, rounds: int
) -> np.ndarray:
    if len(points) < 2:
        return probability
    k = min(neighbours, len(points))
    _, index = cKDTree(points).query(points, k=k, workers=-1)
    out = probability
    for _ in range(rounds):
        out = out[index].mean(axis=1)
    return out


def _regions(
    klass: np.ndarray, splats: np.ndarray, a: np.ndarray, b: np.ndarray, min_splats: float
) -> tuple[np.ndarray, np.ndarray]:
    """Connected same-class regions over edges (a, b); a region under `min_splats` takes the
    class most of its edges to other classes lead to, until none is left (or nothing
    changes). Returns the class and region per node."""
    klass = klass.copy()
    n = klass.size
    region = np.zeros(n, np.int64)
    if n == 0:
        return klass, region
    for _ in range(8):
        same = klass[a] == klass[b]
        graph = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n))
        region = connected_components(graph, directed=False)[1]
        count = int(region.max()) + 1
        size = np.bincount(region, splats, count)
        small = size < min_splats
        cross = ~same & (small[region[a]] | small[region[b]])
        if not cross.any():
            break
        # Each small region votes for the classes across its edges.
        src = np.concatenate([region[a[cross]], region[b[cross]]])
        dst = np.concatenate([klass[b[cross]], klass[a[cross]]])
        mine = small[src]
        src, dst = src[mine], dst[mine]
        width = int(klass.max()) + 1
        keys, votes = np.unique(src * width + dst, return_counts=True)
        r, c = keys // width, keys % width
        order = np.lexsort((c, -votes, r))
        first = order[np.r_[True, r[order][1:] != r[order][:-1]]]
        region_class = np.zeros(count, np.int64)
        region_class[region] = klass
        before = region_class.copy()
        region_class[r[first]] = c[first]
        if np.array_equal(before, region_class):
            break
        klass = region_class[region]
    return klass, region


def _fold_small(
    region: np.ndarray, klass: np.ndarray, splats: np.ndarray, points: np.ndarray, least: float
) -> np.ndarray:
    """Regions still under `least` splats (cut off from every other ground cell, so
    `_regions` could not join them) become part of the nearest region of their class that
    is not, or of their class's largest."""
    region = region.copy()
    size = np.bincount(region, splats)
    for c in np.unique(klass):
        mine = klass == c
        ids = np.unique(region[mine])
        small = ids[size[ids] < least]
        if small.size == 0 or small.size == ids.size and ids.size == 1:
            continue
        large = ids[size[ids] >= least]
        cells = np.flatnonzero(mine & np.isin(region, small))
        if large.size == 0:
            region[cells] = ids[np.argmax(size[ids])]
            continue
        anchors = np.flatnonzero(mine & np.isin(region, large))
        _, near = cKDTree(points[anchors]).query(points[cells], k=1, workers=-1)
        region[cells] = region[anchors[near]]
    return region


# ------------------------------------------------------------------------------ the run


@dataclass
class GroundFirst:
    instances: list[ss.Instance]
    #: Per instance id, the record's extra fields (kind, name, ...).
    extra: dict[int, dict[str, Any]]
    splat_id: np.ndarray
    ground: gp.GroundResult
    segmentation: ss.Segmentation
    stats: dict[str, Any] = field(default_factory=dict)


def _subtree_ids(parent: np.ndarray) -> list[np.ndarray]:
    subtree: list[list[int]] = [[] for _ in range(parent.size)]
    for k, chain in enumerate(ss._ancestors(parent)):
        for a in chain:
            subtree[a].append(k + 1)
    return [np.asarray(ids, np.int64) for ids in subtree]


def _instance(
    k: int,
    parent: int | None,
    level: int,
    own: int,
    points: np.ndarray,
    dim: int,
    tags: list[dict[str, object]],
    vegetation: bool,
    embedding: np.ndarray | None = None,
) -> ss.Instance:
    properties = {name: 0.0 for name in ss.PROPERTY_PROMPTS}
    properties["static"] = 1.0
    properties["vegetation"] = 1.0 if vegetation else 0.0
    return ss.Instance(
        id=k,
        parent=parent,
        level=level,
        splats=own,
        bounds_min=points.min(axis=0),
        bounds_max=points.max(axis=0),
        centroid=points.mean(axis=0),
        views=0,
        embedding=np.zeros(dim) if embedding is None else embedding,
        tags=tags,
        properties=properties,
        behaviour="static",
        category=GROUND_CATEGORY,
    )


def run(
    splats: Splats,
    masks: ss.MaskSource | None,
    embedder: ss.Embedder,
    vocabulary: Sequence[str],
    *,
    box_source: BoxMaskSource | None = None,
    box_factory: Callable[[list[Camera]], BoxMaskSource] | None = None,
    namer: Namer | None = None,
    source_factory=None,
    params: gp.GroundParams | None = None,
    refine_objects: int = REFINE_OBJECTS,
    progress=None,
    **segment_options: Any,
) -> GroundFirst:
    """Candidate A on a scan held in memory (module docstring). `segment_options` go to
    `segment_scene.segment` (views, renderer, cache, coverage, workers, ...)."""
    say = progress or (lambda message: None)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    ground = gp.ground_pass(splats.positions, splats.opacities, splats.scales, params)
    timings["groundS"] = time.perf_counter() - started
    say(f"ground: {json.dumps(ground.stats['shares'])}, layer {ground.layer_m:.3f} m")
    cell0, _, _, edge = ss.supervoxels(splats.positions)
    cell, n_cells = gp.split_cells(cell0, ground.label)
    del cell0
    counts = np.bincount(cell, minlength=n_cells)
    centroids = (
        np.stack([np.bincount(cell, splats.positions[:, k], n_cells) for k in range(3)], axis=1)
        / np.maximum(counts, 1)[:, None]
    )
    low = np.isin(ground.label, (gp.GROUND, gp.BELOW))
    exclude = np.zeros(n_cells, bool)
    exclude[cell[low]] = True
    refine_report: dict[str, Any] = {}
    refine = make_refine(
        splats, cell, centroids, exclude, ground, edge,
        box_source=box_source, box_factory=box_factory,
        cache=segment_options.get("cache"), objects=refine_objects, progress=progress,
        report=refine_report,
    )  # fmt: skip
    if box_source is None and box_factory is None:
        refine = None
    seg = ss.segment(
        splats, masks, embedder, vocabulary,
        cells=(cell, centroids, counts, edge), exclude=exclude.copy() if refine is None else exclude,
        refine=refine, source_factory=source_factory, progress=progress,
        **segment_options,
    )  # fmt: skip
    timings.update({f"segment.{k}": v for k, v in seg.timings.items()})
    lifted = seg.lifted
    n = lifted.parent.size
    leaf = lifted.cell_id.copy()
    hag = ground.hag
    layer = ground.layer_m

    # Stuff: low top-level objects that describe as ground cover go to the ground.
    mark = time.perf_counter()
    top_of = ss.top_level(lifted.parent)
    splat_top = top_of[leaf][cell]
    stuff_max = max(STUFF_LAYERS * layer, STUFF_MIN_M)
    demoted: list[int] = []
    order = np.argsort(splat_top, kind="stable")
    bounds = np.searchsorted(splat_top[order], np.arange(n + 2))
    for t in range(1, n + 1):
        if lifted.parent[t - 1] or bounds[t + 1] == bounds[t]:
            continue
        rows = order[bounds[t] : bounds[t + 1]]
        if float(np.percentile(hag[rows], 90)) > stuff_max:
            continue
        category = seg.instances[t - 1].category
        described = bool(seg.instances[t - 1].tags)
        if (category in STUFF_CATEGORIES) or not described:
            demoted.append(t)
    demoted_set = np.zeros(n + 1, bool)
    demoted_set[demoted] = True
    gone = demoted_set[top_of]  # per id: under a demoted top
    gone[0] = False
    cell_gone = gone[leaf]
    ground_cell = (exclude & (leaf == 0)) | cell_gone
    # Unassigned cells within the ground layer (unknown splats under an object, released
    # cells) are ground too, so nothing in the layer is left without an id.
    cell_low = np.bincount(cell, hag <= layer, n_cells) > 0
    ground_cell |= (leaf == 0) & cell_low
    leaf[ground_cell] = 0
    timings["stuffS"] = time.perf_counter() - mark

    # Cover classes of the ground cells.
    mark = time.perf_counter()
    classes, contrast = cover_classes()
    sums, weight, class_embedding = cover_votes(
        seg.views,
        ground_cell,
        embedder,
        classes,
        contrast,
        votes=seg.votes,
        progress=progress,
    )
    gcells = np.flatnonzero(ground_cell)
    k = len(classes)
    probability = np.full((n_cells, k), 1.0 / k)
    voted = weight > 0
    probability[voted] = sums[voted] / weight[voted, None]
    if gcells.size and voted[gcells].any():
        known = gcells[voted[gcells]]
        missing = gcells[~voted[gcells]]
        if missing.size:
            _, near = cKDTree(centroids[known]).query(centroids[missing], k=1, workers=-1)
            probability[missing] = probability[known[near]]
    smooth = _smooth(probability[gcells], centroids[gcells], COVER_NEIGHBOURS, COVER_ROUNDS)
    klass = np.argmax(smooth, axis=1)
    confidence = smooth[np.arange(gcells.size), klass]
    ground_splats = counts[gcells].astype(np.float64)
    total = float(ground_splats.sum())
    # Classes too small to list take their neighbours' class, then regions are made.
    local = np.full(n_cells, -1, np.int64)
    local[gcells] = np.arange(gcells.size)
    a, b = ss._cell_graph(centroids, edge)
    keep = (local[a] >= 0) & (local[b] >= 0)
    ga, gb = local[a[keep]], local[b[keep]]
    share = np.bincount(klass, ground_splats, k) / max(total, 1.0)
    rare = share < COVER_MIN_CLASS_SHARE
    if rare.any() and (~rare).any():
        second = np.where(rare[None, :], -1.0, smooth).argmax(axis=1)
        klass = np.where(rare[klass], second, klass)
    min_region = max(COVER_MIN_REGION, COVER_MIN_REGION_SHARE * total)
    klass, region = _regions(klass, ground_splats, ga, gb, min_region)
    region = _fold_small(region, klass, ground_splats, centroids[gcells], min_region)
    timings["coverS"] = time.perf_counter() - mark

    # The instance table: things kept (renumbered), then the ground's classes and regions.
    keep_ids = [i for i in range(1, n + 1) if not gone[i]]
    new_id = np.zeros(n + 1, np.int64)
    new_id[keep_ids] = np.arange(1, len(keep_ids) + 1)
    instances: list[ss.Instance] = []
    extra: dict[int, dict[str, Any]] = {}
    dim = int(embedder.dim)
    for old in keep_ids:
        i = seg.instances[old - 1]
        parent = new_id[i.parent] if i.parent else 0
        instances.append(
            ss.Instance(
                int(new_id[old]), int(parent) or None, i.level, i.splats, i.bounds_min,
                i.bounds_max, i.centroid, i.views, i.embedding, i.tags, i.properties,
                i.behaviour, i.category,
            )
        )  # fmt: skip
        extent = float(np.linalg.norm(np.asarray(i.bounds_max) - np.asarray(i.bounds_min)))
        extra[int(new_id[old])] = {"kind": "thing", "scaleM": round(extent / 2, 3)}
    cell_id = new_id[leaf]
    splat_cell_ground = ground_cell[cell]
    cover_report = []
    next_id = len(instances) + 1
    for c in np.unique(klass):
        members = np.flatnonzero(klass == c)
        cover = classes[int(c)]
        regions = np.unique(region[members])
        class_id = next_id
        next_id += 1
        cells_of_class = gcells[members]
        mask = np.zeros(n_cells, bool)
        mask[cells_of_class] = True
        points = splats.positions[mask[cell]]
        score = round(float(confidence[members].mean()), 4)
        tags = [{"label": cover.name.lower(), "score": score}]
        emb = ss._normalise(class_embedding[int(c)][None])[0]
        children = regions if regions.size > 1 else np.zeros(0, np.int64)
        own = 0 if children.size else int(points.shape[0])
        instances.append(
            _instance(class_id, None, 0, own, points, dim, tags, cover.vegetation, emb)
        )
        extra[class_id] = {
            "kind": "ground", "name": cover.name, "nameSource": "ground-cover",
            "cover": cover.id, "scaleM": round(float(np.linalg.norm(np.ptp(points, axis=0))) / 2, 3),
        }  # fmt: skip
        if not children.size:
            cell_id[cells_of_class] = class_id
        for r in children:
            in_region = members[region[members] == r]
            region_cells = gcells[in_region]
            rmask = np.zeros(n_cells, bool)
            rmask[region_cells] = True
            rpoints = splats.positions[rmask[cell]]
            rtags = [
                {
                    "label": cover.name.lower(),
                    "score": round(float(confidence[in_region].mean()), 4),
                }
            ]
            instances.append(
                _instance(next_id, class_id, 1, int(rpoints.shape[0]), rpoints, dim, rtags,
                          cover.vegetation, emb)
            )  # fmt: skip
            extra[next_id] = {
                "kind": "ground", "name": cover.name, "nameSource": "ground-cover",
                "cover": cover.id,
                "scaleM": round(float(np.linalg.norm(np.ptp(rpoints, axis=0))) / 2, 3),
            }  # fmt: skip
            cell_id[region_cells] = next_id
            next_id += 1
        cover_report.append(
            {"class": cover.id, "splats": int(points.shape[0]), "regions": int(max(children.size, 1)),
             "meanConfidence": score}
        )  # fmt: skip
    splat_id = cell_id[cell]
    del splat_cell_ground

    # Names.
    mark = time.perf_counter()
    named = 0
    if namer is not None:
        named = _name(instances, extra, splat_id, cell, cell_id, seg.views, namer, say,
                      segment_options.get("cache"))  # fmt: skip
    timings["nameS"] = time.perf_counter() - mark
    timings["totalS"] = time.perf_counter() - started
    things = [i for i in instances if extra[i.id]["kind"] == "thing"]
    stats = {
        "variant": VARIANT,
        "ground": ground.stats,
        "refine": refine_report,
        "demotedObjects": len(demoted),
        "cover": sorted(cover_report, key=lambda r: -r["splats"]),
        "things": len(things),
        "topLevelThings": sum(1 for i in things if i.parent is None),
        "named": named,
        "groundShare": round(float(np.isin(splat_id, [i.id for i in instances if extra[i.id]["kind"] == "ground"]).mean()), 4),
        "unassignedShare": round(float((splat_id == 0).mean()), 4),
        "lift": {k: v for k, v in lifted.stats.items() if k != "refine"},
        "timingsS": {k: round(v, 2) for k, v in timings.items()},
    }  # fmt: skip
    return GroundFirst(instances, extra, splat_id, ground, seg, stats)


def _name(
    instances: list[ss.Instance],
    extra: dict[int, dict[str, Any]],
    splat_id: np.ndarray,
    cell: np.ndarray,
    cell_id: np.ndarray,
    views: Sequence[ss.View],
    namer: Namer,
    say,
    cache: Path | None,
) -> int:
    """Asks `namer` about the top-level things and the parts of the largest; writes `name`,
    `nameSource`, `wholeOrPart`, `material`, `movable` into `extra`. Returns how many were
    named. Each object is shown in its view where it has the most pixels: in context (the
    rest dimmed) and alone on black, from the view's own image."""
    parent = np.array([i.parent or 0 for i in instances], np.int64)
    subtree = _subtree_ids(parent)
    splats_of = np.bincount(splat_id, minlength=len(instances) + 1)
    total = np.array([splats_of[ids].sum() for ids in subtree])
    tops = [
        i.id for i in instances
        if extra[i.id]["kind"] == "thing" and i.parent is None and i.tags
    ]  # fmt: skip
    tops.sort(key=lambda t: -total[t - 1])
    chosen = list(tops)
    for t in tops[:NAME_PART_OBJECTS]:
        parts = [i.id for i in instances if i.parent == t and i.tags]
        parts.sort(key=lambda p: -total[p - 1])
        chosen += parts[:NAME_PARTS_EACH]
    chosen = chosen[:NAME_MAX]
    if not chosen:
        return 0
    # Per chosen instance, its best view: the most pixels, half for one the frame cuts.
    lookup = np.zeros((len(chosen), len(instances) + 1), bool)
    for j, k in enumerate(chosen):
        lookup[j, subtree[k - 1]] = True
    best = np.full(len(chosen), -1)
    best_area = np.zeros(len(chosen))
    best_box = np.zeros((len(chosen), 4))
    for v, view in enumerate(views):
        owner = view.cell
        h, w = owner.shape
        pid = np.where(owner >= 0, cell_id[np.maximum(owner, 0)], 0)
        for j in range(len(chosen)):
            inside = lookup[j][pid]
            area = float(inside.sum())
            if area <= max(best_area[j], ss.MIN_VIEW_PX - 1):
                continue
            ys, xs = np.nonzero(inside)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
            if box[0] <= 0 or box[1] <= 0 or box[2] >= w - 1 or box[3] >= h - 1:
                area *= ss.TRUNCATED_WEIGHT
            if area > best_area[j]:
                best[j], best_area[j], best_box[j] = v, area, box
    crops: list[list[np.ndarray]] = []
    for j, k in enumerate(chosen):
        if best[j] < 0:
            crops.append([])
            continue
        view = views[int(best[j])]
        made = ss._crops(view, view.rgb, best_box[j], cell_id, subtree[k - 1], ("context", "black"))
        crops.append([made["context"], made["black"]])
    answers = _cached_names(cache, namer, crops)
    named = 0
    for k, answer in zip(chosen, answers, strict=True):
        if not answer:
            continue
        record = extra[k]
        record["name"] = str(answer["name"])[:1].upper() + str(answer["name"])[1:]
        record["nameSource"] = "vlm"
        whole = answer.get("whole")
        if isinstance(whole, bool):
            record["wholeOrPart"] = "whole" if whole else "part"
        if isinstance(answer.get("part_of"), str) and answer["part_of"]:
            record["partOf"] = answer["part_of"][:60]
        if isinstance(answer.get("material"), str) and answer["material"]:
            record["material"] = answer["material"][:40]
        if isinstance(answer.get("movable"), bool):
            record["movable"] = answer["movable"]
        named += 1
    say(f"named {named} of {len(chosen)}: " + ", ".join(
        extra[k].get("name", "?") for k in chosen[:12]))  # fmt: skip
    return named


def _cached_names(
    cache: Path | None, namer: Namer, crops: list[list[np.ndarray]]
) -> list[dict | None]:
    """`namer.name_objects(crops)`, each answer kept in `cache/names.json` by its crops'
    bytes, so a run that is resumed (or re-assembled elsewhere) asks again only what is new."""
    path = None if cache is None else cache / "names.json"
    known: dict[str, Any] = {}
    if path is not None and path.exists():
        known = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        hashlib.sha1(b"".join(np.ascontiguousarray(c).tobytes() for c in group)).hexdigest()
        for group in crops
    ]
    ask = [i for i, key in enumerate(keys) if key not in known and crops[i]]
    started = time.perf_counter()
    for i in ask:
        if time.perf_counter() - started > NAME_BUDGET_S:
            break  # the rest keep their tags' names
        (known[keys[i]],) = namer.name_objects([crops[i]])
    if ask and path is not None:
        path.write_text(json.dumps(known), encoding="utf-8")
    return [known.get(key) for key in keys]


def document(result: GroundFirst, tiles: dict[str, list[int]], embedder: ss.Embedder,
             vocabulary_size: int, namer: Namer | None) -> dict:  # fmt: skip
    """`instances.json` (v1 and its extra fields: module docstring)."""
    doc = ss.instances_document(
        result.instances, tiles,
        embedding_model=embedder.name, dim=int(embedder.dim),
        vocabulary_model=embedder.name, vocabulary_size=vocabulary_size,
    )  # fmt: skip
    for record in doc["instances"]:
        record.update(result.extra.get(int(record["id"]), {}))
    ground = result.ground
    doc["ground"] = {
        "method": "SMRF on a robust lowest surface (tools/captures/ground_pass.py)",
        "layerM": round(ground.layer_m, 4),
        "cellM": round(ground.terrain.cell, 4),
        "seenShare": ground.stats["terrain"]["seenShare"],
        "cover": result.stats["cover"],
    }
    doc["variant"] = {
        "name": VARIANT,
        "label": VARIANT_LABEL,
        "about": VARIANT_ABOUT,
        "namer": None if namer is None else namer.name,
    }
    return doc


# ------------------------------------------------------------------------------ the CLI


def _palette(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rng.uniform(0.25, 1.0, (n + 1, 3))
    out[0] = 0.0
    return out


def render_check(
    splats: Splats,
    result: GroundFirst,
    cameras: Sequence[Camera],
    out: Path,
) -> None:
    """Rows per camera: the scan, its top-level things coloured by object (ground grey),
    and its ground coloured by cover class (things dark)."""
    from PIL import Image

    from splat_render import SplatIndex

    parent = np.array([i.parent or 0 for i in result.instances], np.int64)
    top = ss.top_level(parent)[result.splat_id]
    kinds = np.array(["none"] + [result.extra[i.id]["kind"] for i in result.instances])
    covers = sorted({e.get("cover") for e in result.extra.values() if e.get("cover")})
    cover_of = np.zeros(len(result.instances) + 1, np.int64)
    for i in result.instances:
        c = result.extra[i.id].get("cover")
        if c:
            cover_of[i.id] = covers.index(c) + 1
    palette = _palette(len(result.instances), 7)
    cover_palette = np.array(
        [[0, 0, 0], [0.55, 0.8, 0.3], [0.85, 0.75, 0.45], [0.6, 0.45, 0.3], [0.65, 0.65, 0.7],
         [0.3, 0.55, 0.25], [0.95, 0.6, 0.2], [0.4, 0.6, 0.9], [0.8, 0.4, 0.6], [0.5, 0.35, 0.2]]
    )  # fmt: skip
    index = SplatIndex.build(splats)
    rows = []
    for camera in cameras:
        frame = render(splats, camera, labels=top.astype(np.int64), index=index)
        label = np.maximum(frame.label, 0)
        is_ground = kinds[label] == "ground"
        things = np.where(is_ground[..., None], 0.35, palette[label]) * frame.alpha[..., None]
        cover_frame = render(splats, camera, labels=cover_of[result.splat_id], index=index)
        c = cover_palette[np.maximum(cover_frame.label, 0) % len(cover_palette)]
        c = np.where((cover_frame.label > 0)[..., None], c, 0.08) * cover_frame.alpha[..., None]
        rows.append(np.concatenate([frame.rgb, things, c], axis=1))
    image = np.round(np.clip(np.concatenate(rows, axis=0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(out)
    legend = out.with_suffix(".legend.json")
    legend.write_text(
        json.dumps({c: [round(float(x), 2) for x in cover_palette[(k + 1) % len(cover_palette)]]
                    for k, c in enumerate(covers)}), encoding="utf-8")  # fmt: skip


def overview_cameras(positions: np.ndarray, count: int = 4) -> list[Camera]:
    """A view from above and `count - 1` obliques around the scan's robust box."""
    lo, hi = np.percentile(positions, 2, axis=0), np.percentile(positions, 98, axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi[:2] - lo[:2])) / 2
    cams = [Camera.look_at(centre + [0, 0.01, 2.2 * radius], centre, fov_deg=50, width=512,
                           height=384)]  # fmt: skip
    for k in range(count - 1):
        a = 2 * math.pi * k / max(count - 1, 1) + 0.4
        eye = centre + np.array([math.cos(a) * 1.5 * radius, math.sin(a) * 1.5 * radius,
                                 0.55 * radius])  # fmt: skip
        cams.append(Camera.look_at(eye, centre, fov_deg=50, width=512, height=384))
    return cams


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="the scan's tileset.json, or its PLY")
    parser.add_argument("tiles", type=Path, help="its tileset directory")
    parser.add_argument("--out", type=Path, required=True, help="where instances.json goes")
    parser.add_argument("--masks", default=None, help="module:Class, a MaskSource")
    parser.add_argument("--truth", type=Path, default=None, help="labels.json: oracle masks")
    parser.add_argument("--boxes", default=None, help="module:Class, a BoxMaskSource")
    parser.add_argument("--namer", default=None, help="module:Class, a Namer")
    parser.add_argument("--embedder", default="segment_scene:FakeEmbedder")
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--views", type=int, default=ss.VIEW_COUNT)
    parser.add_argument("--max-views", type=int, default=ss.MAX_VIEWS)
    parser.add_argument("--renderer", choices=("cpu", "gsplat"), default="cpu")
    parser.add_argument("--max-scale-m", type=float, default=None)
    parser.add_argument("--coverage-rounds", type=int, default=ss.COVERAGE_ROUNDS)
    parser.add_argument("--coverage-views", type=int, default=ss.COVERAGE_VIEWS)
    parser.add_argument("--refine-objects", type=int, default=REFINE_OBJECTS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--check", type=Path, default=None, help="a PNG: by object, by cover")
    parser.add_argument("--opacity-min", type=float, default=ss.scene_plants.PACKAGE_OPACITY_MIN)
    parser.add_argument(
        "--tile-gaussians", type=int, default=ss.scene_plants.PACKAGE_TILE_GAUSSIANS
    )
    args = parser.parse_args()
    if (args.masks is None) == (args.truth is None):
        parser.error("give exactly one of --masks and --truth")
    from_tiles = args.source.name.endswith(".json")
    if from_tiles:
        from splat_render import load_tileset

        splats = load_tileset(args.source)
        rows, row_count = np.arange(len(splats)), len(splats)
    else:
        splats, rows, row_count = ss.load_source(args.source, args.opacity_min)
    embedder = ss.load_embedder(args.embedder)
    vocabulary: list[str] = []
    if args.vocabulary:
        vocabulary = [
            line.strip()
            for line in args.vocabulary.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    masks = ss.load_masks(args.masks) if args.masks else None
    boxes = ss._load(args.boxes) if args.boxes else None
    namer = ss._load(args.namer) if args.namer else None
    factory = box_factory = None
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        levels = ss.truth_levels(truth, rows, splats.colours)
        factory = lambda cameras: ss.OracleMasks(splats, levels, cameras)
        box_factory = lambda cameras: OracleBoxMasks(splats, levels[0], cameras)
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    result = run(
        splats, masks, embedder, vocabulary,
        box_source=boxes, box_factory=box_factory, namer=namer, source_factory=factory,
        refine_objects=args.refine_objects, progress=lambda m: print(m, flush=True),
        view_count=args.views, max_views=args.max_views, workers=args.workers,
        cpus=args.cpus,
        memory_bytes=None if args.memory_gb is None else args.memory_gb * float(1 << 30),
        cache=args.cache, renderer=ss.make_renderer(args.renderer),
        max_scale_m=args.max_scale_m, coverage_rounds=args.coverage_rounds,
        coverage_budget=args.coverage_views,
    )  # fmt: skip
    if from_tiles:
        tiles = ss.tile_binding_by_position(args.tiles, splats.positions, result.splat_id)
    else:
        tiles = ss.tile_binding(
            args.tiles, args.source, rows, result.splat_id, row_count,
            opacity_min=args.opacity_min, tile_gaussians=args.tile_gaussians,
        )  # fmt: skip
    tiles = ss.rebind_instances.rebind(args.tiles, tiles)
    doc = document(result, tiles, embedder, len(vocabulary), namer)
    doc["tilesEncoding"] = ss.rebind_instances.TILES_ENCODING
    ss.write_instances(args.out, doc, result.instances)
    if args.cache:
        (args.cache / "cameras.json").write_text(
            json.dumps([v.camera.to_json() for v in result.segmentation.views]), encoding="utf-8"
        )
    if args.check:
        render_check(splats, result, overview_cameras(splats.positions), args.check)
    usage = {"gaussians": len(splats), **ss.peak_usage(started)}
    print(ss.usage_line(usage), flush=True)
    summary = {
        "instances": len(result.instances),
        **result.stats,
        "views": len(result.segmentation.views),
        "usage": usage,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
