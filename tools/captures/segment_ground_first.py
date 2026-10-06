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
   the image-text model inside the views' masks where its cell is seen (`cover_votes`),
   smoothed among neighbouring ground cells, and split into connected regions. When there
   is a namer that can (`CoverNamer`), it checks the word of each of the largest classes
   from two crops of it, choosing among the same classes; its choice is the class's word
   (two classes it gives one word become one): the image-text model groups, the
   vision-language model names.
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
    "CoverNamer",
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
#: one shows where it meets the ground), its box filling `REFINE_FILL` of the frame (the
#: box's own extent, not its bounding sphere's: run 37383770486 framed by the sphere and
#: saw the spool at a third of the frame, too small for SAM to tell it from the ground).
REFINE_AZIMUTHS = 5
REFINE_ELEVATIONS_DEG = (10.0, 40.0)
REFINE_FILL = 0.8
REFINE_FOV_DEG = 50.0
#: The box SAM is prompted with: the object's 2nd-98th percentiles, down to the terrain,
#: padded by `PROMPT_PAD` of its size (and a cell edge) -- tight, or SAM answers with the
#: ground the box holds. Cells are claimed within the box padded by `REFINE_PAD`.
PROMPT_PAD = 0.04
REFINE_PAD = 0.12
#: Of SAM's three answers to a box, one is used when it holds `MASK_RECALL` of the object's
#: own pixels in the view (else the view does not vote: SAM answered about something
#: else), at most `MASK_FOREIGN` of its pixels belong to other things beyond its footprint
#: (a thing inside it, such as a flange's rim, may be a part of it), and at most
#: `MASK_SPILL` lie outside the box, and at most `MASK_LEAK` are ground beyond the object's
#: own footprint (its box in plan: a flange is inside it, the lawn around it is not); the
#: largest of those (the whole object, base included).
MASK_RECALL = 0.6
MASK_FOREIGN = 0.15
MASK_SPILL = 0.1
MASK_LEAK = 0.12
#: Of its refine views, this many per object are kept for `describe` and naming.
REFINE_KEEP = 4
#: A ground (or unassigned) cell inside the box is claimed when it is in the object's mask
#: in at least `CLAIM_SHARE` of the refine views it is seen in, and seen in `CLAIM_VIEWS`.
CLAIM_SHARE = 0.6
CLAIM_VIEWS = 3
#: An object's cell is released when in at most `RELEASE_SHARE` of `RELEASE_VIEWS`+ views.
RELEASE_SHARE = 0.1
RELEASE_VIEWS = 4
#: A smaller top-level object, mostly inside the box (`MERGE_INSIDE`), becomes a part when
#: `MERGE_SHARE` of its cell-views (at least `MERGE_VIEWS`) are in the larger one's masks,
#: and `MERGE_SEEN` of its cells were seen at all.
MERGE_SHARE = 0.7
MERGE_VIEWS = 6
MERGE_SEEN = 0.3
#: ... and when this share of its splats lie inside the (padded) box: a scatter of specks
#: across the scan is not a part.
MERGE_INSIDE = 0.8

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
#: A region's pixels are closed over holes this wide before the rest of a crop is blacked.
COVER_CLOSE_PX = 9
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
#: Cover classes the namer checks (the largest; the rest keep the image-text model's word).
COVER_ASK_MAX = 8
#: Its close look at a class: a square this share of the view's shorter side, where the
#: class is densest.
COVER_LOOK_SHARE = 0.35


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


@runtime_checkable
class CoverNamer(Protocol):
    """A namer that can also say which ground cover a kind of ground is."""

    name: str

    def choose_cover(
        self, crops: list[list[np.ndarray]], choices: Sequence[str]
    ) -> list[str | None]:
        """Per kind of ground (its crops), one of `choices`, or None."""
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
    elevation, aimed at 40% of its height, from where the box fills `REFINE_FILL` of the
    frame: its half plan diagonal against the horizontal field of view, its half height
    (seen at the elevation) against the vertical; the far plane just past the box."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    half_plan = max(0.5 * float(np.linalg.norm((hi - lo)[:2])), 1e-3)
    half_height = max(0.5 * float(hi[2] - lo[2]), 1e-3)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.4 * (hi[2] - lo[2])])
    across = math.tan(math.radians(fov_deg) / 2)
    up = across * height / width
    cameras = []
    for e, elevation in enumerate(elevations):
        a = math.radians(elevation)
        # Seen from above at `a`, the box's plan depth adds to its height on screen.
        tall = half_height * math.cos(a) + half_plan * math.sin(a)
        distance = max(half_plan / (REFINE_FILL * across), tall / (REFINE_FILL * up))
        distance += half_plan  # from the box's near face, not its centre
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
                    far=distance + 2 * half_plan,
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


def cached_box_candidates(
    cache: Path | None, source: BoxMaskSource, view: ss.View, box: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """SAM's answers to `box` in `view` -- `source.box_candidates` (its three) when it has
    them, else `box_masks`' one -- kept in `cache` (by camera, box and model)."""

    def ask() -> list[tuple[np.ndarray, float]]:
        many = getattr(source, "box_candidates", None)
        if many is not None:
            return [(np.asarray(m, bool), float(sc)) for m, sc in many(view.rgb, box)]
        mask, score = source.box_masks(view.rgb, box[None])[0]
        return [(np.asarray(mask, bool), float(score))]

    if cache is None:
        return ask()
    key = json.dumps([view.camera.to_json(), [round(float(b), 2) for b in box], source.name])
    path = cache / f"boxmask-{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
    if path.exists():
        with np.load(path) as z:
            if "masks" in z:
                return [(m.astype(bool), float(sc)) for m, sc in zip(z["masks"], z["scores"])]
            return [(z["mask"].astype(bool), float(z["score"]))]
    found = ask()
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        masks=np.array([m for m, _ in found], bool).reshape(len(found), *view.cell.shape),
        scores=np.array([sc for _, sc in found], np.float64),
    )
    tmp.replace(path)
    return found


def choose_candidate(
    candidates: Sequence[tuple[np.ndarray, float]],
    own: np.ndarray,
    other: np.ndarray,
    box: np.ndarray,
    beyond: np.ndarray | None = None,
) -> np.ndarray | None:
    """The answer to use (module docstring, refine): of those holding `MASK_RECALL` of the
    object's own pixels (`own`), with at most `MASK_FOREIGN` other things' pixels (`other`),
    `MASK_SPILL` outside the box and `MASK_LEAK` ground beyond the object's footprint
    (`beyond`), the largest; None when none qualifies."""
    if not own.any():
        return None
    h, w = own.shape
    x0, y0, x1, y1 = box
    in_box = np.zeros((h, w), bool)
    in_box[int(y0) : math.ceil(y1) + 1, int(x0) : math.ceil(x1) + 1] = True
    best, best_area = None, -1
    for mask, _ in candidates:
        area = int(mask.sum())
        if area == 0:
            continue
        recall = (mask & own).sum() / own.sum()
        foreign = (mask & other).sum() / area
        spill = (mask & ~in_box).sum() / area
        leak = 0.0 if beyond is None else (mask & beyond).sum() / area
        valid = (
            recall >= MASK_RECALL
            and foreign <= MASK_FOREIGN
            and spill <= MASK_SPILL
            and leak <= MASK_LEAK
        )
        if valid and area > best_area:
            best, best_area = mask, area
    return best


@dataclass
class _Plan:
    top: int
    #: The prompt box (tight) and the box cells are claimed in (padded).
    lo: np.ndarray
    hi: np.ndarray
    claim_lo: np.ndarray
    claim_hi: np.ndarray
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
    counts = np.bincount(cell, minlength=n_cells).astype(np.float64)

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
            lo = np.percentile(pts, 2, axis=0)
            hi = np.percentile(pts, 98, axis=0)
            corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
            lo[2] = min(lo[2], float(ground.terrain.at(corners).min()) - edge)
            tight = PROMPT_PAD * (hi - lo) + edge
            loose = REFINE_PAD * (hi - lo) + 2 * edge
            plans.append(
                _Plan(
                    t, lo - tight, hi + tight, lo - loose, hi + loose,
                    refine_cameras(lo - tight, hi + tight, turn=i * golden),
                )
            )  # fmt: skip
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
        cell_top0 = top_of[leaf]
        thing = cell_top0 > 0
        footprints: dict[int, np.ndarray] = {}
        skipped = 0
        for k, view in enumerate(render_views(cameras)):
            plan = plans[owner[k]]
            box = project_box(view.camera, plan.lo, plan.hi)
            if box is None:
                continue
            pixel_cell = np.maximum(view.cell, 0)
            good = (view.cell >= 0) & (view.purity >= ss.MIN_PURITY)
            own = good & (cell_top0[pixel_cell] == plan.top)
            if owner[k] not in footprints:
                footprints[owner[k]] = np.all(
                    (centroids[:, :2] >= plan.lo[:2]) & (centroids[:, :2] <= plan.hi[:2]), axis=1
                )
            outside = good & ~footprints[owner[k]][pixel_cell]
            other = outside & thing[pixel_cell] & ~own
            beyond = outside & ~thing[pixel_cell]
            candidates = cached_box_candidates(cache, source, view, box)
            mask = choose_candidate(candidates, own, other, box, beyond)
            if mask is None:
                skipped += 1
                continue
            votes = ss.vote(view, [ss.Mask(mask, 0, 1.0)], n_cells, 1)
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
            # Parts: smaller top-level objects, centred in its box, mostly inside its masks.
            others = np.unique(cell_top[(seen > 0) & (cell_top > t)])
            for u in others:
                if u in absorbed or parent[u - 1] != 0:
                    continue
                mine = cell_top == u
                within = np.all(
                    (centroids[mine] >= plan.claim_lo) & (centroids[mine] <= plan.claim_hi), axis=1
                )
                if np.average(within, weights=counts[mine]) < MERGE_INSIDE:
                    continue
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
            in_box = np.all((c >= plan.claim_lo) & (c <= plan.claim_hi), axis=1)
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
                {
                    "claimedCells": claimed,
                    "releasedCells": released,
                    "mergedObjects": merged,
                    "viewsSkipped": skipped,
                }
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
    image: str = "rgb",
    background: int | None = 0,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell, the summed probability of each cover class (n_cells, k); per cell the
    weight voted; and per class the sum of its crops' embeddings (k, dim), weighted by its
    probability.

    Pooled in masks: in a view whose mask votes are given (`votes[v]`, the lift's), the
    ground pixels of each mask (`_mask_regions`, its finest level, closed over the holes the
    sparse pixel ownership leaves: `COVER_CLOSE_PX`) are cropped -- one crop, or windows over
    a large mask (`_windows_of`) -- with what is not the mask black (`background`; None
    keeps it); the mask's probabilities are its crops' mean, voted to every ground cell in
    it by its pixels. Ground pixels in no mask (and views without votes) are voted by
    `COVER_TILE` tiles, weighed `COVER_TILE_WEIGHT`; only the masked views vote when there
    are any. A crop's probabilities: a softmax over the classes and the contrast prompts
    (`segment_scene.LOGIT_SCALE` x cosine); the classes' share of it (how much the crop
    reads as ground cover at all) weighs its vote, and the classes are renormalised. The
    image is the view's own (`image="rgb"`: gsplat's, as a viewer draws it -- the CPU's
    point samples, masked, are speckle; measured on the pumpkin, run 37383770486: they read
    the hay as dirt and leaf litter) or the samples (`"samples"`)."""
    import cv2

    n_cells = ground_cell.size
    k = len(classes)
    closing = np.ones((COVER_CLOSE_PX, COVER_CLOSE_PX), np.uint8)
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
        full = ss._softmax(ss.LOGIT_SCALE * emb @ bank.T)
        # How much a crop reads as ground cover at all (the classes against the contrast
        # prompts): a pumpkin's bottom in the ground layer barely votes.
        groundness = full[:, :k].sum(axis=1)
        p = full[:, :k] / np.maximum(groundness[:, None], 1e-12)
        class_embedding[:] += (p * groundness[:, None]).T @ emb
        owner = np.asarray(group_of)
        mean = np.zeros((len(groups), k))
        np.add.at(mean, owner, p * groundness[:, None])
        held = np.bincount(owner, groundness, len(groups))
        mean /= np.maximum(held, 1e-12)[:, None]
        trust = held / np.maximum(np.bincount(owner, minlength=len(groups)), 1)
        for (cells, w), row, t in zip(groups, mean, trust, strict=True):
            sums[cells] += (t * w)[:, None] * row[None, :]
            weight[cells] += t * w
        crops.clear()
        group_of.clear()
        groups.clear()

    def add(image: np.ndarray, keep: np.ndarray, owner: np.ndarray, windows, weight=1.0) -> None:
        cells, counts = np.unique(owner[keep], return_counts=True)
        if not windows or cells.size == 0:
            return
        groups.append((cells, weight * counts.astype(np.float64)))
        for xa, ya, xb, yb in windows:
            crop = image[ya:yb, xa:xb]
            if background is not None:
                # The region's pixels are a sparse sampling of its cells: close its holes
                # first, or the crop is speckle on black.
                inside = keep[ya:yb, xa:xb].astype(np.uint8)
                inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, closing) > 0
                crop = np.where(inside[..., None], crop, background)
            crops.append(np.ascontiguousarray(crop, np.uint8))
            group_of.append(len(groups) - 1)

    # Only the views that were masked (the lift's): the refine pass's close views of objects
    # have no masks, and tiles alone blur a path into the lawn beside it.
    for v, view in enumerate(views[: len(votes)] if votes else views):
        picture = view.samples if image == "samples" and view.samples is not None else view.rgb
        owner = view.cell
        h, w = owner.shape
        region, good = _mask_regions(view, votes[v] if v < len(votes) else None, ground_cell)
        flat = region[good]
        for r in np.unique(flat[flat >= 0]):
            keep = region == r
            ys, xs = np.nonzero(keep)
            if ys.size >= COVER_MIN_PIXELS:
                add(picture, keep, owner, _windows_of(ys, xs, (h, w)))
        rest = good & (region < 0)
        for y in range(0, h - COVER_TILE + 1, COVER_TILE):
            for x in range(0, w - COVER_TILE + 1, COVER_TILE):
                tile = rest[y : y + COVER_TILE, x : x + COVER_TILE]
                if tile.mean() >= COVER_MIN_SHARE:
                    keep = np.zeros((h, w), bool)
                    keep[y : y + COVER_TILE, x : x + COVER_TILE] = tile
                    windows = [(x, y, x + COVER_TILE, y + COVER_TILE)]
                    add(picture, keep, owner, windows, COVER_TILE_WEIGHT if votes else 1.0)
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
    asked: list[dict[str, Any]] = []
    if isinstance(namer, CoverNamer):
        klass, to, asked = _ask_cover(
            klass, gcells, ground_splats, n_cells, seg.views, classes, namer,
            segment_options.get("cache"), say,
        )  # fmt: skip
        merged = np.zeros_like(class_embedding)
        np.add.at(merged, to, class_embedding)
        class_embedding = merged
    by_vlm = {r["vlm"] for r in asked if r["vlm"] is not None}
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
        source = "vlm" if cover.id in by_vlm else "ground-cover"
        extra[class_id] = {
            "kind": "ground", "name": cover.name, "nameSource": source,
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
                "kind": "ground", "name": cover.name, "nameSource": source,
                "cover": cover.id,
                "scaleM": round(float(np.linalg.norm(np.ptp(rpoints, axis=0))) / 2, 3),
            }  # fmt: skip
            cell_id[region_cells] = next_id
            next_id += 1
        cover_report.append(
            {"class": cover.id, "splats": int(points.shape[0]), "regions": int(max(children.size, 1)),
             "meanConfidence": score, "nameSource": source,
             "fromClasses": sorted(r["siglip"] for r in asked if r["vlm"] == cover.id)}
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
        "coverAsked": asked,
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
            # A robust box: a few stray splats of it across the view do not widen the crop.
            x0, x1 = np.percentile(xs, [1, 99])
            y0, y1 = np.percentile(ys, [1, 99])
            box = np.array([x0, y0, x1, y1], np.float64)
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
    return _cached_answers(cache, crops, lambda group: namer.name_objects([group])[0])


def _cached_answers(
    cache: Path | None,
    crops: list[list[np.ndarray]],
    ask: Callable[[list[np.ndarray]], Any],
    salt: str = "",
) -> list[Any]:
    """`ask(group)` per group of crops (at most `NAME_BUDGET_S` in all), each answer kept in
    `cache/names.json` by the crops' bytes (and `salt`, the question when it is not the
    names')."""
    path = None if cache is None else cache / "names.json"
    known: dict[str, Any] = {}
    if path is not None and path.exists():
        known = json.loads(path.read_text(encoding="utf-8"))
    prefix = f"{hashlib.sha1(salt.encode()).hexdigest()[:12]}:" if salt else ""
    keys = [
        prefix
        + hashlib.sha1(b"".join(np.ascontiguousarray(c).tobytes() for c in group)).hexdigest()
        for group in crops
    ]
    todo = [i for i, key in enumerate(keys) if key not in known and crops[i]]
    started = time.perf_counter()
    for i in todo:
        if time.perf_counter() - started > NAME_BUDGET_S:
            break  # the rest keep the words they had
        known[keys[i]] = ask(crops[i])
    if todo and path is not None:
        path.write_text(json.dumps(known), encoding="utf-8")
    return [known.get(key) for key in keys]


def _cover_crops(
    views: Sequence[ss.View], class_of_cell: np.ndarray, wanted: Sequence[int]
) -> list[list[np.ndarray]]:
    """Per class of `wanted`, two crops from the view where it has the most pixels: its
    pixels in their surroundings (the rest dimmed, `segment_scene._crops`' "context") and a
    close look (an undimmed square, `COVER_LOOK_SHARE` of the view, where it is densest)."""
    ids_of_cell = np.where(class_of_cell >= 0, class_of_cell + 1, 0)
    best = {c: (0, -1) for c in wanted}
    for v, view in enumerate(views):
        owner = view.cell
        good = (owner >= 0) & (view.purity >= ss.MIN_PURITY)
        cls = np.where(good, class_of_cell[np.maximum(owner, 0)], -1)
        counts = np.bincount(cls[cls >= 0], minlength=max(len(class_of_cell), 1))
        for c in wanted:
            if c < counts.size and counts[c] > max(best[c][0], ss.MIN_VIEW_PX - 1):
                best[c] = (int(counts[c]), v)
    out: list[list[np.ndarray]] = []
    for c in wanted:
        if best[c][1] < 0:
            out.append([])
            continue
        view = views[best[c][1]]
        owner = view.cell
        good = (owner >= 0) & (view.purity >= ss.MIN_PURITY)
        inside = good & (class_of_cell[np.maximum(owner, 0)] == c)
        ys, xs = np.nonzero(inside)
        x0, x1 = np.percentile(xs, [1, 99])
        y0, y1 = np.percentile(ys, [1, 99])
        box = np.array([x0, y0, x1, y1], np.float64)
        context = ss._crops(view, view.rgb, box, ids_of_cell, np.array([c + 1]), ("context",))
        h, w = inside.shape
        side = max(int(COVER_LOOK_SHARE * min(h, w)), 8)
        step = max(side // 4, 1)
        total = np.pad(inside.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        ya = np.arange(0, max(h - side, 0) + 1, step)
        xa = np.arange(0, max(w - side, 0) + 1, step)
        yb, xb = np.minimum(ya + side, h), np.minimum(xa + side, w)
        held = (
            total[yb][:, xb] - total[ya][:, xb] - total[yb][:, xa] + total[ya][:, xa]
        )  # per window (rows: ya, columns: xa)
        r, q = np.unravel_index(int(np.argmax(held)), held.shape)
        look = view.rgb[ya[r] : yb[r], xa[q] : xb[q]]
        out.append([context["context"], ss._shrink(look)])
    return out


def _ask_cover(
    klass: np.ndarray,
    gcells: np.ndarray,
    weight: np.ndarray,
    n_cells: int,
    views: Sequence[ss.View],
    classes: Sequence[CoverClass],
    namer: CoverNamer,
    cache: Path | None,
    say,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """The namer's word for each of the `COVER_ASK_MAX` largest classes (`klass`: per
    ground cell of `gcells`, weighed by `weight`), chosen among `classes`' names. Returns
    the class per ground cell after (a class it named otherwise is that class now), the
    mapping (old class -> class) and per class asked `{siglip, vlm}` (`vlm`: None when it
    answered none of them)."""
    k = len(classes)
    share = np.bincount(klass, weight, k)
    wanted = [int(c) for c in np.argsort(-share, kind="stable") if share[c] > 0][:COVER_ASK_MAX]
    class_of_cell = np.full(n_cells, -1, np.int64)
    class_of_cell[gcells] = klass
    crops = _cover_crops(views, class_of_cell, wanted)
    choices = [c.name.lower() for c in classes]
    answers = _cached_answers(
        cache, crops, lambda group: namer.choose_cover([group], choices)[0],
        salt="cover:" + ",".join(choices),
    )  # fmt: skip
    to = np.arange(k)
    asked = []
    for c, answer in zip(wanted, answers, strict=True):
        if answer in choices:
            to[c] = choices.index(answer)
        asked.append(
            {"siglip": classes[c].id, "vlm": classes[to[c]].id if answer in choices else None}
        )
    say("cover checked: " + ", ".join(f"{r['siglip']} -> {r['vlm']}" for r in asked))
    return to[klass], to, asked


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


#: A cropped run binds a merged (parent) gaussian only within this of a segmented splat.
CROP_REACH_M = 1.0


def crop_binding(
    tiles_dir: Path, positions: np.ndarray, splat_id: np.ndarray, box: Sequence[float]
) -> dict[str, list[int]]:
    """Every tile's binding for a run on a crop (`box` = x0, y0, x1, y1 of the tileset's
    frame): a leaf gaussian is the segmented splat at its position (0 outside the crop); a
    merged gaussian takes what most of the 8 segmented splats nearest to it carry, when
    they are within `CROP_REACH_M` (`rebind_instances`' rule, on the crop alone); every
    tile wholly outside is one run of 0. The scan stays whole: the rest of it simply has no
    objects in this variant."""
    from rebind_instances import encode_runs, plurality, tile_tree
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    tree = cKDTree(np.asarray(positions, np.float64))
    ids = np.asarray(splat_id, np.int64)
    x0, y0, x1, y1 = (float(v) for v in box)
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for uri, leaf in tile_tree(tileset):
        at = tile_positions(tiles_dir / uri)
        labels = np.zeros(len(at), np.int64)
        near = (
            (at[:, 0] >= x0 - CROP_REACH_M) & (at[:, 0] <= x1 + CROP_REACH_M)
            & (at[:, 1] >= y0 - CROP_REACH_M) & (at[:, 1] <= y1 + CROP_REACH_M)
        )  # fmt: skip
        if near.any():
            rows = np.flatnonzero(near)
            k = 1 if leaf else 8
            distance, index = tree.query(at[rows].astype(np.float64), k=k, workers=-1)
            distance, index = np.atleast_2d(distance.T).T, np.atleast_2d(index.T).T
            close = distance[:, 0] <= (0.0 if leaf else CROP_REACH_M)
            if leaf:
                labels[rows[close]] = ids[index[close, 0]]
            elif close.any():
                labels[rows[close]] = plurality(ids[index[close]])
        out[checksum_positions(at)] = encode_runs(labels)
    return dict(sorted(out.items()))


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
    parser.add_argument(
        "--crop",
        default=None,
        help="x0,y0,x1,y1 (metres, the tileset's frame): segment only this part of the scan; "
        "every tile is still bound, the rest with no objects",
    )
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
    box = None
    if args.crop:
        if not from_tiles:
            parser.error("--crop needs the tileset (its leaves are the splats)")
        box = [float(v) for v in args.crop.split(",")]
        if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
            parser.error("--crop is x0,y0,x1,y1 with x0 < x1 and y0 < y1")
        p = splats.positions
        inside = (
            (p[:, 0] >= box[0]) & (p[:, 0] <= box[2]) & (p[:, 1] >= box[1]) & (p[:, 1] <= box[3])
        )
        splats = splats.take(np.flatnonzero(inside))
        print(f"crop {box}: {len(splats):,} of {len(p):,} splats", flush=True)
        del p, inside
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
    if box is not None:
        tiles = crop_binding(args.tiles, splats.positions, result.splat_id, box)
    elif from_tiles:
        tiles = ss.tile_binding_by_position(args.tiles, splats.positions, result.splat_id)
    else:
        tiles = ss.tile_binding(
            args.tiles, args.source, rows, result.splat_id, row_count,
            opacity_min=args.opacity_min, tile_gaussians=args.tile_gaussians,
        )  # fmt: skip
    if box is None:
        tiles = ss.rebind_instances.rebind(args.tiles, tiles)
    doc = document(result, tiles, embedder, len(vocabulary), namer)
    if box is not None:
        doc["variant"]["crop"] = box
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
