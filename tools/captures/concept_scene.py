"""Concept-first scene objects (bake-off candidate C; research notes §1.5, "Alternative B").

`segment_scene.py` cuts every view into class-free masks and names what it lifted
afterwards. This module turns that round: it first asks what is in the scene, then looks for
each named thing in every view, so objects are born with their names.

1. **Vocabulary** (`Vocabulary`). A vision-language model (`concept_models.QwenVocabulary`,
   Qwen3-VL, Apache-2.0) looks at `OVERVIEW_VIEWS` overview renders and lists the scene's
   *things* (countable objects: "cable spool", "pumpkin") and *stuff* (ground cover:
   "grass", "gravel", "dirt"), each with one of the viewer's broad categories
   (`data/categories.json`). At most `MAX_THINGS` and `MAX_STUFF`: the list is short.
2. **Concepts in every view** (`ConceptSource`). Things: every instance of each thing in
   every view, one mask each, along *camera paths* (`camera_paths`: the plan's views in an
   order a video tracker can follow), so a tracker that keeps ids along a path (SAM 3's)
   tells this module which masks are one object (`ConceptMask.track`). Stuff: a per-pixel
   class over the ground (`StuffMap`).
3. **Ground** (`ground_layer`). Which splats lie on the ground, from geometry. Today
   `scene_plants`' slope filter behind a small adapter (`GROUND_PASS`); the shared ground
   pass (`ground_pass.py`, bake-off candidate A's branch) replaces it in one line. Cells
   never straddle the ground (`ground_cells`), and a ground cell that the views keep
   seeing inside a thing's mask (`PROMOTE_SHARE`) is the thing's, not the ground's: the
   spool's lower flange.
4. **Lift** (`lift_concepts`), by `segment_scene`'s own voting: thing masks vote for the
   cells they cover and co-occurrence over co-visibility joins cells into objects
   (`segment_scene._grow`); each object is named by the concept most of its votes carry,
   and objects that one track spans are one. Ground cells take the stuff class most views
   gave them (smoothed over the cell graph); each class is one top-level instance in the
   category Ground ("Grass", "Gravel"), so the objects panel lists Ground with its cover
   classes, each hideable, with no new UI.
5. **Leftovers.** Cells that are neither ground nor in a named thing are lifted from a
   class-free pass (SAM 2 automatic masks, `segment_models.Sam2Masks`), as `segment_scene`
   does: what nobody named still becomes an object (named by its tags, as today). The same
   class-free masks give every object its parts (levels below the concept level).
6. **Meaning.** `segment_scene.describe` embeds and tags every instance as today (search
   stays SigLIP's); a named object's name, category and first tag come from its concept.

`instances.json` is the contract's v1 (docs/SCENE_OBJECTS.md §4), with optional fields per
instance: `name` (the concept, or none), `nameSource` (`concept` | `tags`), `kind`
(`thing` | `ground`), `concept`; and at the root `concepts` (the vocabulary, the models)
and `ground` (the ground pass's summary). Viewers that know nothing of them still work.

Usage (`infra/modal/segment_concepts.py` runs it on a GPU):
    python concept_scene.py TILESET.json TILES_DIR \\
        --vocabulary concept_models:QwenVocabulary \\
        --concepts concept_models:GroundedSam2Concepts \\
        --masks segment_models:Sam2Masks --embedder segment_models:SiglipEmbedder \\
        --renderer gsplat --out OUT --overviews OUT/overviews
    python concept_scene.py yard/source/splat.ply yard/splat --truth yard/source/labels.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial import cKDTree

import rebind_instances
import scene_categories
import scene_plants
import segment_scene as ss
from splat_render import Camera, SplatIndex, Splats, render

# ----------------------------------------------------------------------------- constants

#: Overview renders the vocabulary is read from (rings at two scales and elevations).
OVERVIEW_VIEWS = 12
#: The vocabulary is kept short: at most this many things and this much stuff.
MAX_THINGS = 12
MAX_STUFF = 6
#: Camera paths (`camera_paths`): consecutive views whose eyes are at most this share of
#: the scan's radius apart, and whose axes turn at most `PATH_TURN_DEG`, are one path.
PATH_STEP = 0.75
PATH_TURN_DEG = 50.0
#: A ground cell is the thing's when it was inside a thing's mask in at least this share of
#: its visible weight, in at least `PROMOTE_VIEWS` views (`lift_concepts`).
PROMOTE_SHARE = 0.6
PROMOTE_VIEWS = 2
#: A ground cell takes a stuff class with at least this much class weight (pixels x score).
STUFF_MIN_WEIGHT = 1.0
#: Ground cells without a class take the nearest classified ground cell's within this many
#: cell edges, or `GROUND_FILL_M`, whichever is farther.
GROUND_FILL_CELLS = 6.0
GROUND_FILL_M = 1.0
#: Rounds of majority smoothing of the cover classes over the cell graph.
GROUND_SMOOTH_ROUNDS = 2
#: Two objects of one concept are one when a track holds at least this share of each.
TRACK_SHARE = 0.5
#: Which ground pass `ground_layer` runs: "slope" (`scene_plants`' Vosselman filter, here
#: now) or "shared" (`ground_pass.py`, from bake-off candidate A's branch once merged).
GROUND_PASS = "slope"
#: Splats the slope filter is fitted on at most (seeded); every splat is then measured.
GROUND_SAMPLE = 1_000_000
#: Gaussians larger than this, or fainter than `GROUND_OPACITY`, are left out of the fit:
#: floaters under the ground would pull the terrain down.
GROUND_MAX_SCALE_M = 0.5
GROUND_OPACITY = 0.1
#: The category every ground cover instance is in, and the name of ground no class took.
GROUND_CATEGORY = "ground"
GROUND_NAME = "Ground"
#: Cover classes' colours in the overview renders (Tableau 10), ground with no class grey.
CLASS_COLOURS = (
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7",
    "#9c755f", "#bab0ac",
)  # fmt: skip
UNCLASSIFIED_COLOUR = "#808080"


# --------------------------------------------------------------------------- the concepts


@dataclass(frozen=True)
class Concept:
    """One named concept of a scene: a thing (countable) or stuff (ground cover)."""

    name: str
    kind: str  # "thing" | "stuff"
    #: A `data/categories.json` id.
    category: str = scene_categories.OTHER
    #: What the segmenter is asked for (default: the name).
    prompt: str | None = None

    @property
    def query(self) -> str:
        return self.prompt or self.name

    def to_json(self) -> dict[str, str]:
        out = {"name": self.name, "kind": self.kind, "category": self.category}
        if self.prompt and self.prompt != self.name:
            out["prompt"] = self.prompt
        return out


def clean_concepts(
    concepts: Sequence[Concept],
    *,
    max_things: int = MAX_THINGS,
    max_stuff: int = MAX_STUFF,
    categories: Sequence[str] | None = None,
) -> list[Concept]:
    """Things then stuff, each name once (lower case, trimmed), an unknown category made
    `other` (stuff: `ground`), at most `max_things` and `max_stuff`, in the order given."""
    known = set(categories or scene_categories.category_ids())
    seen: set[str] = set()
    things: list[Concept] = []
    stuff: list[Concept] = []
    for c in concepts:
        name = " ".join(str(c.name).strip().lower().split())
        kind = "stuff" if c.kind == "stuff" else "thing"
        if not name or name in seen:
            continue
        fallback = GROUND_CATEGORY if kind == "stuff" else scene_categories.OTHER
        category = c.category if c.category in known else fallback
        out = things if kind == "thing" else stuff
        if len(out) >= (max_things if kind == "thing" else max_stuff):
            continue
        seen.add(name)
        out.append(Concept(name, kind, category, c.prompt))
    return things + stuff


def split_concepts(concepts: Sequence[Concept]) -> tuple[list[Concept], list[Concept]]:
    return [c for c in concepts if c.kind == "thing"], [c for c in concepts if c.kind == "stuff"]


def load_concepts(path: Path) -> list[Concept]:
    """A vocabulary file: `{"things": [{"name", "category"?}, ...], "stuff": [...]}` (what
    `vocabulary.json` holds), or a list of `{"name", "kind", "category"?}`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    if isinstance(data, dict):
        for kind in ("things", "stuff"):
            for row in data.get(kind, []):
                rows.append({**row, "kind": "thing" if kind == "things" else "stuff"})
    else:
        rows = list(data)
    return clean_concepts(
        [
            Concept(
                str(r["name"]),
                str(r.get("kind", "thing")),
                str(r.get("category", "")),
                r.get("prompt"),
            )
            for r in rows
        ]
    )


@runtime_checkable
class Vocabulary(Protocol):
    """Lists a scene's things and stuff from overview images (a VLM)."""

    name: str

    def concepts(self, images: list[np.ndarray]) -> list[Concept]:
        """uint8 (h, w, 3) overview renders -> things and stuff (`clean_concepts`)."""
        ...


@dataclass
class FixedVocabulary:
    """A vocabulary given in advance (tests; a re-run with the same words)."""

    fixed: list[Concept]
    name: str = "fixed"

    def concepts(self, images: list[np.ndarray]) -> list[Concept]:
        return clean_concepts(self.fixed)


@dataclass
class ConceptMask:
    """One instance of a thing in one frame."""

    #: (h, w) bool.
    mask: np.ndarray
    #: Index into the things.
    concept: int
    score: float = 1.0
    #: The tracker's id for it along its path (-1: none; ids are per path).
    track: int = -1


@dataclass
class StuffMap:
    """A frame's ground cover: per pixel a stuff index (-1: none) and how sure."""

    label: np.ndarray  # (h, w) int16
    score: np.ndarray  # (h, w) float32


@runtime_checkable
class ConceptSource(Protocol):
    """Finds named concepts in frames: SAM 3 (`concept_models.Sam3Concepts`), or a stand-in
    (`concept_models.GroundedSam2Concepts`)."""

    name: str

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        """One camera path's frames (in order, a video) -> per frame its thing masks."""
        ...

    def stuff(
        self,
        rgb: np.ndarray,
        stuff: Sequence[Concept],
        masks: Sequence[ss.Mask],
        region: np.ndarray,
    ) -> StuffMap:
        """One frame's cover classes. `masks`: its class-free masks (a source that pools
        over them uses them); `region`: (h, w) bool, the pixels that show ground."""
        ...


# ----------------------------------------------------------------------------- the ground


@dataclass
class Ground:
    """The ground layer: per splat whether it is on it, and its height above the terrain."""

    flag: np.ndarray  # (n,) bool
    hag: np.ndarray  # (n,) float32, metres
    source: str
    info: dict = field(default_factory=dict)


def slope_ground(
    splats: Splats,
    *,
    sample: int = GROUND_SAMPLE,
    max_scale_m: float = GROUND_MAX_SCALE_M,
    opacity: float = GROUND_OPACITY,
    seed: int = 0,
) -> Ground:
    """`scene_plants.ground_model` (Vosselman's slope filter on the lowest splat per plan
    cell, its ground layer from how far the resting splats scatter): a splat is on the
    ground at most the layer above the terrain (or anywhere under it)."""
    import skeleton

    pos = np.asarray(splats.positions, np.float64)
    fit = np.flatnonzero(
        (np.asarray(splats.scales).max(axis=1) <= max_scale_m)
        & (np.asarray(splats.opacities) >= opacity)
    )
    if fit.size < 16:
        fit = np.arange(len(pos))
    if fit.size > sample:
        fit = np.sort(np.random.default_rng(seed).choice(fit, sample, replace=False))
    xyz = pos[fit]
    model = scene_plants.ground_model(xyz, skeleton.neighbour_spacing(xyz))
    hag = pos[:, 2] - model.at(pos[:, :2])
    flag = hag <= model.layer_m
    return Ground(
        flag,
        hag.astype(np.float32),
        "scene_plants.ground_model (slope filter)",
        {**model.to_json(), "share": round(float(flag.mean()), 4) if flag.size else 0.0},
    )


def ground_layer(splats: Splats) -> Ground:
    """The ground pass `GROUND_PASS` names. The one line to change when the shared pass
    lands is the constant."""
    if GROUND_PASS == "shared":
        import ground_pass  # bake-off candidate A's module (bakeoff-seg-ground-first)

        return ground_pass.concept_ground(splats)  # type: ignore[attr-defined]
    return slope_ground(splats)


def ground_cells(
    positions: np.ndarray, flag: np.ndarray, **kwargs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """`segment_scene.supervoxels`, with no cell holding both ground and other splats: each
    voxel is split by the flag. Returns the cells, centroids, counts, edge, and per cell
    whether it is ground."""
    cell, _, _, edge = ss.supervoxels(positions, **kwargs)  # type: ignore[arg-type]
    key = cell.astype(np.int64) * 2 + np.asarray(flag, np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    cell = inverse.astype(np.int32).reshape(-1)
    counts = np.bincount(cell, minlength=unique.size)
    pos = np.asarray(positions, np.float64)
    centroids = (
        np.stack([np.bincount(cell, pos[:, k], unique.size) for k in range(3)], axis=1)
        / counts[:, None]
    )
    return cell, centroids, counts, float(edge), (unique % 2).astype(bool)


# ---------------------------------------------------------------------------- the views


def camera_paths(cameras: Sequence[Camera], radius: float) -> list[list[int]]:
    """The cameras' indices cut into paths a tracker can follow: in the plan's order, a new
    path wherever the eye jumps more than `PATH_STEP` x `radius` or the view axis turns more
    than `PATH_TURN_DEG`. Every camera is in exactly one path, paths in order."""
    paths: list[list[int]] = []
    cos_turn = math.cos(math.radians(PATH_TURN_DEG))
    for k, camera in enumerate(cameras):
        if paths:
            last = cameras[paths[-1][-1]]
            step = float(np.linalg.norm(camera.centre - last.centre))
            turn = float(np.dot(camera.rotation[2], last.rotation[2]))
            if step <= PATH_STEP * radius and turn >= cos_turn:
                paths[-1].append(k)
                continue
        paths.append([k])
    return paths


def overview_cameras(positions: np.ndarray, count: int = OVERVIEW_VIEWS) -> list[Camera]:
    """`count` views of the whole scan: `segment_scene.plan_views`' rings, no local views."""
    return ss.plan_views(positions, count, observers=None, edge=None)


# ---------------------------------------------------------------------------- the votes


@dataclass
class ConceptVotes:
    """Every view's evidence, as `lift_concepts` reads it."""

    n_cells: int
    n_things: int
    n_stuff: int
    #: Per view: which thing mask each visible cell is in (one level), and per mask its
    #: concept and its global track (-1: none).
    things: list[ss._Votes] = field(default_factory=list)
    thing_concept: list[np.ndarray] = field(default_factory=list)
    thing_track: list[np.ndarray] = field(default_factory=list)
    #: Per view: the class-free masks' votes (`segment_scene.vote`, its levels).
    free: list[ss._Votes] = field(default_factory=list)
    #: Per cell: visible weight over all views, weight inside a thing mask, views inside one.
    seen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thing_in: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thing_views: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    #: Per cell and thing: weight inside that thing's masks.
    concept_weight: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: Per cell and stuff class: pixels x score.
    stuff_weight: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def __post_init__(self) -> None:
        if self.seen.size != self.n_cells:
            self.seen = np.zeros(self.n_cells)
            self.thing_in = np.zeros(self.n_cells)
            self.thing_views = np.zeros(self.n_cells, np.int32)
            self.concept_weight = np.zeros((self.n_cells, max(self.n_things, 1)), np.float32)
            self.stuff_weight = np.zeros((self.n_cells, max(self.n_stuff, 1)), np.float32)

    def add_free(self, view: ss.View, masks: Sequence[ss.Mask]) -> None:
        levels = max((m.level for m in masks), default=0) + 1
        self.free.append(ss.vote(view, masks, self.n_cells, levels))

    def add_things(self, view: ss.View, masks: Sequence[ConceptMask], track_base: int) -> None:
        votes = ss.vote(view, [ss.Mask(m.mask, 0, m.score) for m in masks], self.n_cells, 1)
        concept = np.array([m.concept for m in masks], np.int64)
        track = np.array([m.track + track_base if m.track >= 0 else -1 for m in masks], np.int64)
        self.things.append(votes)
        self.thing_concept.append(concept)
        self.thing_track.append(track)
        np.add.at(self.seen, votes.cells, votes.weight)
        inside = votes.masks[0] >= 0
        cells = votes.cells[inside]
        weight = votes.weight[inside].astype(np.float64)
        np.add.at(self.thing_in, cells, weight)
        np.add.at(self.thing_views, cells, 1)
        if cells.size and self.n_things:
            np.add.at(self.concept_weight, (cells, concept[votes.masks[0][inside]]), weight)

    def add_stuff(self, view: ss.View, stuff: StuffMap) -> None:
        if not self.n_stuff:
            return
        owner = view.cell.reshape(-1)
        purity = view.purity.reshape(-1)
        label = np.asarray(stuff.label).reshape(-1)
        ok = (owner >= 0) & (purity >= ss.MIN_PURITY) & (label >= 0)
        if not ok.any():
            return
        weight = purity[ok] * np.asarray(stuff.score, np.float32).reshape(-1)[ok]
        np.add.at(self.stuff_weight, (owner[ok], label[ok].astype(np.int64)), weight)


def ground_pixels(view: ss.View, ground_cell: np.ndarray) -> np.ndarray:
    """(h, w) bool: the pixels a ground cell owns."""
    owner = view.cell
    return (owner >= 0) & ground_cell[np.maximum(owner, 0)]


# ----------------------------------------------------------------------------- the lift


@dataclass
class ConceptLift:
    """`lift_concepts`' result: the hierarchy over cells, and what each object is."""

    lifted: ss.Lifted
    #: Per instance (index id - 1): "thing" | "ground".
    kind: list[str]
    #: Per instance: its concept's thing index (itself or the object it is part of), -1 none.
    thing: np.ndarray
    #: Per instance: the stuff index of a ground instance, -1 for none (or unclassified).
    stuff: np.ndarray
    #: Per instance: True for the coarsest instance of an object (level 0).
    top: np.ndarray
    #: Per cell: on the ground after promotion; and the promoted cells.
    ground: np.ndarray
    promoted: np.ndarray
    #: Per cell: its cover class (-1: unclassified; only ground cells have one).
    cover: np.ndarray
    stats: dict = field(default_factory=dict)


def _drop_small(region: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Regions below `MIN_INSTANCE_SPLATS` splats and `MIN_INSTANCE_CELLS` cells go to -1."""
    region = region.copy()
    valid = region >= 0
    if not valid.any():
        return region
    top = int(region.max()) + 1
    size = np.bincount(region[valid], counts[valid], top)
    cells = np.bincount(region[valid], minlength=top)
    small = (size < ss.MIN_INSTANCE_SPLATS) & (cells < ss.MIN_INSTANCE_CELLS)
    region[valid & small[np.maximum(region, 0)]] = -1
    return ss._relabel(region)


def _union(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(k) for k in range(n)], np.int64)


def track_joins(
    region: np.ndarray, region_concept: np.ndarray, votes: ConceptVotes
) -> tuple[np.ndarray, int]:
    """Objects one track spans are one object: for each track, the regions of its concept
    holding at least `TRACK_SHARE` of their own tracked weight in it, and at least a quarter
    of the track's, are joined. Returns the regions renumbered, and how many joins."""
    n = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
    rows_r, rows_t, rows_w = [], [], []
    for v, concept_track in zip(votes.things, votes.thing_track, strict=True):
        inside = v.masks[0] >= 0
        if not inside.any():
            continue
        cells = v.cells[inside]
        track = concept_track[v.masks[0][inside]]
        reg = region[cells]
        ok = (track >= 0) & (reg >= 0)
        rows_r.append(reg[ok])
        rows_t.append(track[ok])
        rows_w.append(v.weight[inside][ok].astype(np.float64))
    if n < 2 or not rows_r or not sum(r.size for r in rows_r):
        return region, 0
    r = np.concatenate(rows_r)
    t = np.concatenate(rows_t)
    w = np.concatenate(rows_w)
    keys, inverse = np.unique(r * (int(t.max()) + 1) + t, return_inverse=True)
    weight = np.bincount(inverse, w, keys.size)
    kr, kt = keys // (int(t.max()) + 1), keys % (int(t.max()) + 1)
    per_region = np.bincount(kr, weight, n)
    per_track = np.bincount(kt, weight, int(t.max()) + 1)
    strong = (weight >= TRACK_SHARE * per_region[kr]) & (weight >= 0.25 * per_track[kt])
    pairs: list[tuple[int, int]] = []
    for track in np.unique(kt[strong]):
        members = kr[strong & (kt == track)]
        for other in members[1:]:
            if region_concept[other] == region_concept[members[0]]:
                pairs.append((int(members[0]), int(other)))
    if not pairs:
        return region, 0
    root = _union(n, pairs)
    out = np.where(region >= 0, root[np.maximum(region, 0)], -1)
    return ss._relabel(out), len(pairs)


def _majority(
    labels: np.ndarray, a: np.ndarray, b: np.ndarray, who: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """One round of majority over the cell graph among `who` cells (labels >= 0 vote;
    a cell's own label counts with its weight, each neighbour's with its)."""
    n_labels = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    if n_labels == 0:
        return labels
    keep = who[a] & who[b] & (labels[a] >= 0) & (labels[b] >= 0)
    src = np.concatenate([a[keep], b[keep], np.flatnonzero(who & (labels >= 0))])
    lab = np.concatenate([labels[b[keep]], labels[a[keep]], labels[who & (labels >= 0)]])
    w = np.concatenate([weights[b[keep]], weights[a[keep]], weights[who & (labels >= 0)]])
    keys, inverse = np.unique(src * n_labels + lab, return_inverse=True)
    sums = np.bincount(inverse, w, keys.size)
    kc, kl = keys // n_labels, keys % n_labels
    order = np.lexsort((kl, -sums, kc))
    first = order[np.r_[True, kc[order][1:] != kc[order][:-1]]]
    out = labels.copy()
    out[kc[first]] = kl[first]
    return out


def cover_classes(
    votes: ConceptVotes,
    centroids: np.ndarray,
    counts: np.ndarray,
    edge: float,
    ground: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Per cell its cover class (-1: none): ground cells with `STUFF_MIN_WEIGHT` of class
    weight take their heaviest class; the rest of the ground the nearest classified ground
    cell's within `GROUND_FILL_CELLS` edges or `GROUND_FILL_M`; then `GROUND_SMOOTH_ROUNDS`
    of majority among ground neighbours. Cells off the ground have none."""
    cover = np.full(votes.n_cells, -1, np.int64)
    if not votes.n_stuff or not ground.any():
        return cover
    total = votes.stuff_weight.sum(axis=1)
    has = ground & (total >= STUFF_MIN_WEIGHT)
    cover[has] = np.argmax(votes.stuff_weight[has], axis=1)
    rest = np.flatnonzero(ground & ~has)
    if rest.size and has.any():
        source = np.flatnonzero(has)
        reach = max(GROUND_FILL_CELLS * edge, GROUND_FILL_M)
        distance, nearest = cKDTree(centroids[source]).query(
            centroids[rest], k=1, distance_upper_bound=reach, workers=-1
        )
        ok = np.isfinite(distance)
        cover[rest[ok]] = cover[source[nearest[ok]]]
    for _ in range(GROUND_SMOOTH_ROUNDS):
        cover = _majority(cover, a, b, ground, counts.astype(np.float64))
    return cover


def _fill_unseen(
    labels: np.ndarray,
    seen: np.ndarray,
    centroids: np.ndarray,
    edge: float,
    reach: np.ndarray | None,
) -> int:
    """`segment_scene.lift`'s fill, in place: unseen cells take the nearest seen cell's
    labels within `FILL_CELLS` edges or their own reach."""
    unseen = np.flatnonzero(~seen)
    if not unseen.size or not seen.any():
        return 0
    seen_index = np.flatnonzero(seen)
    bound = np.full(unseen.size, ss.FILL_CELLS * edge)
    if reach is not None:
        bound = np.maximum(bound, np.asarray(reach, np.float64)[unseen])
    distance, nearest = cKDTree(centroids[seen_index]).query(
        centroids[unseen], k=1, distance_upper_bound=float(bound.max()), workers=-1
    )
    ok = np.isfinite(distance) & (distance <= bound)
    labels[:, unseen[ok]] = labels[:, seen_index[nearest[ok]]]
    return int(ok.sum())


def lift_concepts(
    votes: ConceptVotes,
    centroids: np.ndarray,
    counts: np.ndarray,
    edge: float,
    ground_cell: np.ndarray,
    *,
    cell_reach: np.ndarray | None = None,
) -> ConceptLift:
    """Objects from the concept votes (module docstring, steps 3-5).

    Level 0 holds the named objects (thing masks, `segment_scene._grow`; joined along
    tracks), then the leftovers (class-free level-0 masks over what is neither ground nor
    named), then one instance per cover class over the ground. The class-free levels refine
    every object off the ground into parts below it (`segment_scene._hierarchy`)."""
    n_cells = len(centroids)
    a, b = ss._cell_graph(centroids, edge)
    stats: dict[str, object] = {"cells": n_cells, "edges": int(a.size)}
    seen_w = votes.seen.copy()
    for v in votes.free:
        np.maximum.at(seen_w, v.cells, v.weight)
    seen = seen_w >= ss.MIN_VISIBLE_PX
    promoted = (
        ground_cell
        & (votes.thing_in >= PROMOTE_SHARE * np.maximum(votes.seen, 1e-9))
        & (votes.thing_views >= PROMOTE_VIEWS)
    )
    ground = ground_cell & ~promoted
    stats.update(
        groundCells=int(ground_cell.sum()),
        promotedCells=int(promoted.sum()),
        seenCells=int(seen.sum()),
    )

    # Named things.
    in_thing = np.zeros(n_cells, bool)
    for v in votes.things:
        in_thing[v.cells[v.masks[0] >= 0]] = True
    in_thing &= ~ground
    region = np.full(n_cells, -1, np.int64)
    if votes.things and in_thing.any():
        region, rounds = ss._grow(votes.things, 0, in_thing, a, b, counts)
        stats["thingRounds"] = rounds
        region = _drop_small(ss._absorb(region, a, b, in_thing), counts)
    n_regions = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
    region_concept = np.full(n_regions, -1, np.int64)
    if n_regions and votes.n_things:
        valid = region >= 0
        weight = np.zeros((n_regions, votes.n_things))
        np.add.at(weight, region[valid], votes.concept_weight[valid, : votes.n_things])
        region_concept = np.argmax(weight, axis=1)
        region, joins = track_joins(region, region_concept, votes)
        stats["trackJoins"] = joins
        if joins:
            n_regions = int(region.max()) + 1
            valid = region >= 0
            weight = np.zeros((n_regions, votes.n_things))
            np.add.at(weight, region[valid], votes.concept_weight[valid, : votes.n_things])
            region_concept = np.argmax(weight, axis=1)
    stats["things"] = n_regions

    # Leftovers: the class-free level 0 over what is neither ground nor named.
    levels = max((v.masks.shape[0] for v in votes.free), default=0)
    free = [ss._pad_levels(v, levels) for v in votes.free]
    left = ~ground & (region < 0)
    leftover = np.full(n_cells, -1, np.int64)
    if levels:
        in_free = np.zeros((levels, n_cells), bool)
        for v in free:
            for level in range(levels):
                in_free[level, v.cells[v.masks[level] >= 0]] = True
        if (in_free[0] & left).any():
            leftover, _ = ss._grow(free, 0, in_free[0] & left, a, b, counts)
            leftover = _drop_small(ss._absorb(leftover, a, b, seen & left), counts)
    n_left = int(leftover.max()) + 1 if leftover.size and leftover.max() >= 0 else 0
    stats["leftovers"] = n_left

    # Level 0: things, leftovers, then slivers off the ground to their neighbours.
    objects = np.where(region >= 0, region, np.where(leftover >= 0, n_regions + leftover, -1))
    objects = ss._absorb(objects, a, b, seen & ~ground)
    objects[ground] = -1

    # The ground's cover classes, one instance per class (and one for ground no class took).
    cover = cover_classes(votes, centroids, counts, edge, ground, a, b)
    n_stuff = votes.n_stuff
    base = n_regions + n_left
    level0 = np.where(objects >= 0, objects + 1, 0)
    level0[ground] = base + 1 + np.where(cover[ground] >= 0, cover[ground], n_stuff)
    labels = [level0]
    # Parts: the class-free levels over everything off the ground.
    for level in range(levels):
        if not (in_free[level] & ~ground).any():
            continue
        joined, _ = ss._grow(free, level, in_free[level] & ~ground, a, b, counts)
        joined = _drop_small(ss._absorb(joined, a, b, seen & ~ground), counts)
        joined[ground] = -1
        labels.append(joined + 1)
    stacked = np.stack(labels).astype(np.int64)
    # Cells off the ground that no view saw take the nearest seen cell's labels (the ground
    # has its class wherever it is, seen or not).
    off = ~ground
    sub = stacked[:, off]
    reach = None if cell_reach is None else np.asarray(cell_reach)[off]
    stats["filledCells"] = _fill_unseen(sub, seen[off], centroids[off], edge, reach)
    stacked[:, off] = sub
    lifted = ss._hierarchy(stacked, counts, stats)
    lifted.stats["levels"] = int(stacked.shape[0])

    # What each instance is: its top-level ancestor's level-0 label.
    n = lifted.parent.size
    top_of = ss.top_level(lifted.parent)  # per id (0: none) its level-0 ancestor
    label_of_top = np.zeros(n + 1, np.int64)
    has = lifted.cell_id > 0
    label_of_top[top_of[lifted.cell_id[has]]] = stacked[0][has]
    kind: list[str] = []
    thing = np.full(n, -1, np.int64)
    stuff = np.full(n, -1, np.int64)
    for k in range(n):
        lab = int(label_of_top[top_of[k + 1]]) - 1  # level-0 label, 0-based
        if lab >= base:
            kind.append("ground")
            s = lab - base
            stuff[k] = s if s < n_stuff else -1
        else:
            kind.append("thing")
            if 0 <= lab < n_regions:
                thing[k] = int(region_concept[lab])
    top = lifted.parent == 0
    lifted.stats["unassignedShare"] = round(
        float(counts[lifted.cell_id == 0].sum()) / max(float(counts.sum()), 1.0), 4
    )
    return ConceptLift(lifted, kind, thing, stuff, top, ground, promoted, cover, lifted.stats)


# ------------------------------------------------------------------------------ the run


@dataclass
class ConceptSegmentation:
    splat_id: np.ndarray
    instances: list[ss.Instance]
    lift: ConceptLift
    concepts: list[Concept]
    views: list[ss.View]
    cell: np.ndarray
    ground: Ground
    overviews: list[np.ndarray]
    timings: dict[str, float]
    #: Per instance (index id - 1): the extra fields `concept_document` writes.
    extra: list[dict[str, object]] = field(default_factory=list)


def segment_concepts(
    splats: Splats,
    vocabulary: Vocabulary,
    source: ConceptSource,
    free: ss.MaskSource | None,
    embedder: ss.Embedder,
    words: Sequence[str],
    *,
    cameras: Sequence[Camera] | None = None,
    view_count: int = ss.VIEW_COUNT,
    max_views: int = ss.MAX_VIEWS,
    overview_count: int = OVERVIEW_VIEWS,
    renderer: ss.SplatRenderer | None = None,
    max_scale_m: float | None = None,
    workers: int | None = None,
    cpus: int | None = None,
    memory_bytes: float | None = None,
    ground: Ground | None = None,
    progress: Callable[[str], None] | None = None,
    free_factory: Callable[[list[Camera]], ss.MaskSource] | None = None,
) -> ConceptSegmentation:
    """Ground, cells, views, vocabulary, concept and class-free masks, the lift and
    meaning, for a scan held in memory. `free_factory(cameras)` builds the class-free mask
    source that needs the cameras (`segment_scene.OracleMasks` in tests)."""
    say = progress or (lambda message: None)
    timings: dict[str, float] = {}
    mark = time.perf_counter()
    ground = ground or ground_layer(splats)
    timings["groundS"] = time.perf_counter() - mark
    say(f"ground: {ground.source}, {ground.info.get('share')} of the splats")
    mark = time.perf_counter()
    cell, centroids, counts, edge, ground_cell = ground_cells(splats.positions, ground.flag)
    n_cells = len(centroids)
    largest = np.asarray(splats.scales).max(axis=1)
    if max_scale_m is None:
        view_splats, view_cell = splats, cell
    else:
        keep = np.flatnonzero(largest <= max_scale_m)
        view_splats, view_cell = splats.take(keep), cell[keep]
    index = SplatIndex.build(view_splats)
    if cameras is None:
        cameras = ss.plan_views(
            splats.positions,
            view_count,
            observers=ss.observer_points(splats),
            edge=edge,
            solid=centroids,
            max_views=max_views,
        )
    cameras = list(cameras)
    _, lo, hi = ss._extent(splats.positions)
    radius = max(0.5 * float(np.linalg.norm((hi - lo)[:2])), 1e-3)
    paths = camera_paths(cameras, radius)
    timings["planS"] = time.perf_counter() - mark
    say(f"{n_cells} cells at {edge:.3f} m, {len(cameras)} views in {len(paths)} paths")
    reach = np.zeros(n_cells)
    np.maximum.at(reach, cell, 2.0 * largest)
    tag = "" if max_scale_m is None else f"max{max_scale_m:g}"
    count = workers or ss.default_workers(
        cpus, memory_bytes, ss.render_worker_bytes(len(view_splats))
    )
    views: list[ss.View] = []
    overviews: list[np.ndarray] = []
    # Forked before any model starts threads of its own (`segment_scene.RenderPool`).
    with ss.RenderPool(view_splats, view_cell, index=index, workers=count, tag=tag) as pool:
        # The vocabulary, from overview renders.
        mark = time.perf_counter()
        for camera in overview_cameras(splats.positions, overview_count):
            if renderer is not None:
                overviews.append(
                    ss.cached_raster(None, renderer, view_splats, camera, n_cells, tag, index)
                )
            else:
                frame = render(view_splats, camera, index=index)
                overviews.append(np.round(np.clip(frame.rgb, 0, 1) * 255).astype(np.uint8))
        concepts = clean_concepts(vocabulary.concepts(overviews))
        close = getattr(vocabulary, "close", None)
        if close is not None:
            close()  # a VLM's GPU memory back before the segmenters load
        things, stuff = split_concepts(concepts)
        timings["vocabularyS"] = time.perf_counter() - mark
        say("vocabulary: " + ", ".join(f"{c.name} ({c.kind})" for c in concepts))
        votes = ConceptVotes(n_cells, len(things), len(stuff))
        if free is None and free_factory is not None:
            free = free_factory(cameras)
        clock = {"renderS": 0.0, "rasterS": 0.0, "freeS": 0.0, "thingsS": 0.0, "stuffS": 0.0}
        path_of = {k: p for p, path in enumerate(paths) for k in path}
        pending: list[ss.View] = []
        track_base = 0
        mark = time.perf_counter()
        for k, view in enumerate(pool.views(cameras)):
            now = time.perf_counter()
            clock["renderS"] += now - mark
            if renderer is not None:
                image = ss.cached_raster(
                    None, renderer, view_splats, view.camera, n_cells, tag, index
                )
                view = ss.View(view.camera, image, view.cell, view.purity, samples=view.rgb)
            rastered = time.perf_counter()
            clock["rasterS"] += rastered - now
            views.append(view)
            masks = free.masks(view.rgb) if free is not None else []
            votes.add_free(view, masks)
            freed = time.perf_counter()
            clock["freeS"] += freed - rastered
            if stuff:
                region = ground_pixels(view, ground_cell)
                votes.add_stuff(view, source.stuff(view.rgb, stuff, masks, region))
            clock["stuffS"] += time.perf_counter() - freed
            pending.append(view)
            if k + 1 == len(cameras) or path_of[k + 1] != path_of[k]:
                # A path is done: its frames go to the tracker together, as a video.
                started = time.perf_counter()
                found = source.things([v.rgb for v in pending], things) if things else []
                found = found or [[] for _ in pending]
                for v, masks_of in zip(pending, found, strict=True):
                    votes.add_things(v, masks_of, track_base)
                track_base += 1 + max((m.track for f in found for m in f), default=0)
                pending = []
                clock["thingsS"] += time.perf_counter() - started
            say(f"view {k + 1}/{len(cameras)} done")
            mark = time.perf_counter()
    timings.update(clock)
    mark = time.perf_counter()
    lift = lift_concepts(votes, centroids, counts, edge, ground_cell, cell_reach=reach)
    timings["liftS"] = time.perf_counter() - mark
    lift.stats["views"] = len(views)
    lift.stats["paths"] = len(paths)
    lift.stats["cellEdgeM"] = round(edge, 4)
    mark = time.perf_counter()
    words = list(dict.fromkeys([*words, *(c.name for c in concepts)]))
    instances = ss.describe(
        lift.lifted, splats, cell, views, embedder, words,
        renderer=renderer, render_splats=view_splats, render_cell=view_cell,
    )  # fmt: skip
    extra = name_instances(instances, lift, concepts)
    timings["describeS"] = time.perf_counter() - mark
    return ConceptSegmentation(
        lift.lifted.cell_id[cell],
        instances,
        lift,
        concepts,
        views,
        cell,
        ground,
        overviews,
        timings,
        extra,
    )


def name_instances(
    instances: list[ss.Instance], lift: ConceptLift, concepts: Sequence[Concept]
) -> list[dict[str, object]]:
    """Names, categories and first tags from the concepts, in place; and per instance the
    fields `concept_document` adds. A named object and its parts take the concept's category
    (so the panel lists the object whole); the object's first tag is the concept, so search
    and the selection card find it by name. A ground instance is in `GROUND_CATEGORY`, named
    by its cover class, static. Everything else keeps what `describe` gave it."""
    things, stuff = split_concepts(concepts)
    extra: list[dict[str, object]] = []
    for k, instance in enumerate(instances):
        record: dict[str, object] = {"kind": lift.kind[k]}
        concept: Concept | None = None
        if lift.kind[k] == "ground":
            s = int(lift.stuff[k])
            concept = stuff[s] if s >= 0 else None
            label = concept.name if concept else GROUND_NAME.lower()
            instance.category = GROUND_CATEGORY
            instance.behaviour = "static"
            source = "concept" if concept else "ground"
        elif int(lift.thing[k]) >= 0:
            concept = things[int(lift.thing[k])]
            label = concept.name
            instance.category = concept.category
            source = "concept"
        else:
            if instance.tags:
                record["nameSource"] = "tags"
            extra.append(record)
            continue
        if concept is not None:
            record["concept"] = concept.name
        if lift.top[k]:
            record["name"] = label[:1].upper() + label[1:] if lift.kind[k] == "ground" else label
            record["nameSource"] = source
            rest = [t for t in instance.tags if t["label"] != label]
            instance.tags = [{"label": label, "score": 1.0}, *rest][: ss.TAGS_TOP_K]
        extra.append(record)
    return extra


def concept_document(
    result: ConceptSegmentation,
    tiles: dict[str, list[int]],
    *,
    embedder: ss.Embedder,
    vocabulary_size: int,
    vocabulary_model: str,
    segmenter: str,
    stand_in: bool,
) -> dict:
    """`instances.json` (`segment_scene.instances_document`) with the concept fields: per
    instance `name`, `nameSource`, `kind`, `concept`; at the root `concepts` and `ground`."""
    document = ss.instances_document(
        result.instances,
        tiles,
        embedding_model=embedder.name,
        dim=int(embedder.dim),
        vocabulary_model=embedder.name,
        vocabulary_size=vocabulary_size,
    )
    for record, extra in zip(document["instances"], result.extra, strict=True):
        record.update(extra)
    things, stuff = split_concepts(result.concepts)
    document["concepts"] = {
        "vocabularyModel": vocabulary_model,
        "segmenter": segmenter,
        "standIn": bool(stand_in),
        "things": [c.to_json() for c in things],
        "stuff": [c.to_json() for c in stuff],
    }
    info = result.ground.info
    document["ground"] = {
        "source": result.ground.source,
        **{k: info[k] for k in ("layerM", "cellM", "share") if k in info},
        "promotedCells": int(result.lift.promoted.sum()),
        "classes": [c.name for c in stuff],
    }
    return document


# --------------------------------------------------------------------------- the renders


def _hex(colour: str) -> np.ndarray:
    return np.array([int(colour[k : k + 2], 16) / 255 for k in (1, 3, 5)])


def class_palette(n_stuff: int) -> np.ndarray:
    """(n_stuff + 2, 3): index 0 nothing (black), 1 ground no class took, 2.. each class."""
    rows = [np.zeros(3), _hex(UNCLASSIFIED_COLOUR)]
    rows += [_hex(CLASS_COLOURS[k % len(CLASS_COLOURS)]) for k in range(n_stuff)]
    return np.stack(rows)


def splat_classes(result: ConceptSegmentation) -> np.ndarray:
    """Per splat: 0 off the ground, 1 ground no class took, 2 + its cover class."""
    lift = result.lift
    per_cell = np.where(lift.ground, np.where(lift.cover >= 0, lift.cover + 2, 1), 0)
    return per_cell[result.cell]


def render_labels(
    splats: Splats,
    labels: np.ndarray,
    colours: Callable[[np.ndarray], np.ndarray],
    cameras: Sequence[Camera],
    out: Path,
    *,
    legend: Sequence[tuple[str, np.ndarray]] = (),
    index: SplatIndex | None = None,
) -> None:
    """The scan beside itself coloured by `labels` (per splat), one row per camera, with a
    legend strip when given: what a person checks."""
    from PIL import Image, ImageDraw

    rows = []
    for camera in cameras:
        frame = render(splats, camera, labels=np.asarray(labels, np.int64), index=index)
        colour = colours(np.maximum(frame.label, 0)) * frame.alpha[..., None]
        rows.append(np.concatenate([frame.rgb, colour], axis=1))
    pixels = np.round(np.clip(np.concatenate(rows), 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    if legend:
        strip = Image.new("RGB", (image.width, 22 * len(legend) + 8), (24, 24, 24))
        draw = ImageDraw.Draw(strip)
        for k, (text, rgb) in enumerate(legend):
            fill = tuple(round(255 * float(c)) for c in rgb)
            draw.rectangle([8, 6 + 22 * k, 26, 22 + 22 * k], fill=fill)
            draw.text((34, 8 + 22 * k), text, fill=(235, 235, 235))
        both = Image.new("RGB", (image.width, image.height + strip.height))
        both.paste(image, (0, 0))
        both.paste(strip, (0, image.height))
        image = both
    image.save(out)


def render_overviews(result: ConceptSegmentation, splats: Splats, out_dir: Path) -> list[Path]:
    """`by-object.png` (each splat coloured by its object; the largest named ones in the
    legend) and `by-ground-class.png` (the ground by cover class, the rest black), from two
    overview cameras and two of the plan's views (`segment_scene.check_cameras`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = SplatIndex.build(splats)
    cameras = overview_cameras(splats.positions, 4)[:2] + ss.check_cameras(result.views, 3)[1:]
    parent = result.lift.lifted.parent
    objects = ss.top_level(parent)[result.splat_id]
    sizes = np.bincount(objects, minlength=parent.size + 1)
    named = sorted(
        (
            (int(sizes[k + 1]), k + 1, str(extra["name"]))
            for k, extra in enumerate(result.extra)
            if parent[k] == 0 and extra.get("name")
        ),
        reverse=True,
    )[:14]
    legend = [
        (f"{name} ({size:,} splats)", ss._colours(np.array([i]))[0]) for size, i, name in named
    ]
    paths = [out_dir / "by-object.png", out_dir / "by-ground-class.png"]
    render_labels(splats, objects, ss._colours, cameras, paths[0], legend=legend, index=index)
    _, stuff = split_concepts(result.concepts)
    palette = class_palette(len(stuff))
    classes = splat_classes(result)
    share = np.bincount(classes, minlength=len(palette)) / max(len(classes), 1)
    legend = [(f"{GROUND_NAME}, no class ({share[1]:.1%} of splats)", palette[1])]
    legend += [(f"{c.name} ({share[k + 2]:.1%})", palette[k + 2]) for k, c in enumerate(stuff)]
    render_labels(
        splats, classes, lambda ids: palette[np.minimum(ids, len(palette) - 1)], cameras,
        paths[1], legend=legend, index=index,
    )  # fmt: skip
    return paths


# ---------------------------------------------------------------------------- the oracle


class OracleConcepts:
    """Concepts from ground truth, for tests (as `segment_scene.OracleMasks`): it renders the
    cameras itself with the true labels and recognises a frame by its image.

    `thing` (per splat: its thing's index, -1 none) and `instance` (per splat: its object,
    -1 none) give one mask per object in view; with `tracks`, an object's track is its
    instance, as a video tracker that never loses it would say. `stuff` (per splat: its
    cover class, -1 none) gives the cover map."""

    name = "oracle"

    def __init__(
        self,
        splats: Splats,
        thing: np.ndarray,
        instance: np.ndarray,
        stuff: np.ndarray,
        cameras: Sequence[Camera],
        *,
        tracks: bool = True,
        min_area: int = 20,
        min_purity: float = 0.5,
    ) -> None:
        self._things: dict[str, list[ConceptMask]] = {}
        self._stuff: dict[str, StuffMap] = {}
        thing = np.asarray(thing, np.int64)
        instance = np.asarray(instance, np.int64)
        held = (thing >= 0) & (instance >= 0)
        concept_of = dict(zip(instance[held].tolist(), thing[held].tolist(), strict=True))
        for camera in cameras:
            frame = render(splats, camera, labels=np.where(thing >= 0, instance, -1))
            key = ss._image_key(np.round(frame.rgb * 255).astype(np.uint8))
            image = np.where(frame.purity >= min_purity, frame.label, -1)
            masks = []
            for obj in np.unique(image[image >= 0]):
                region = image == obj
                if region.sum() < min_area:
                    continue
                concept = concept_of[int(obj)]
                masks.append(ConceptMask(region, concept, 1.0, int(obj) if tracks else -1))
            self._things[key] = masks
            cover = render(splats, camera, labels=np.asarray(stuff, np.int64))
            label = np.where(cover.purity >= min_purity, cover.label, -1).astype(np.int16)
            self._stuff[key] = StuffMap(label, np.ones(label.shape, np.float32))

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        return [list(self._things.get(ss._image_key(rgb), [])) for rgb in frames]

    def stuff(
        self,
        rgb: np.ndarray,
        stuff: Sequence[Concept],
        masks: Sequence[ss.Mask],
        region: np.ndarray,
    ) -> StuffMap:
        found = self._stuff.get(ss._image_key(rgb))
        if found is None:
            h, w = rgb.shape[:2]
            return StuffMap(np.full((h, w), -1, np.int16), np.zeros((h, w), np.float32))
        return found


#: The synthetic yard's concepts (`synthetic_yard.CLASSES`): what a VLM would list there.
YARD_CONCEPTS = (
    Concept("tree", "thing", "trees"),
    Concept("shrub", "thing", "shrubs"),
    Concept("dead tree", "thing", "wood"),
    Concept("house", "thing", "buildings"),
    Concept("grass", "stuff", "grass"),
    Concept("path", "stuff", "paths"),
)


def yard_truth(labels: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The yard's `labels.json` as `OracleConcepts`' inputs for the kept rows: per splat its
    thing (`YARD_CONCEPTS` order), its object, and its cover class."""
    classes = list(labels["classes"])
    klass = np.asarray(labels["class"], np.int64)[rows]
    instance = np.asarray(labels["instance"], np.int64)[rows]
    of_class = {"tree": 0, "shrub": 1, "snag": 2, "other-static": 3}
    thing = np.full(klass.size, -1, np.int64)
    for name, index in of_class.items():
        thing[klass == classes.index(name)] = index
    # The building is no instance of the yard's: one object of its own.
    obj = np.where(instance >= 0, instance, -1)
    obj[klass == classes.index("other-static")] = len(labels["instances"])
    stuff = np.full(klass.size, -1, np.int64)
    stuff[klass == classes.index("grass/low")] = 0
    stuff[klass == classes.index("ground")] = 1
    return thing, obj, stuff


# ----------------------------------------------------------------------------------- CLI


def _load_vocabulary(spec: str) -> Vocabulary:
    """`module:Class`, or a vocabulary JSON file (`FixedVocabulary`)."""
    if spec.endswith(".json"):
        return FixedVocabulary(load_concepts(Path(spec)), name=f"fixed:{Path(spec).name}")
    vocabulary = ss._load(spec)
    if not isinstance(vocabulary, Vocabulary):
        raise TypeError(f"{spec} is not a Vocabulary (name, concepts(images))")
    return vocabulary


def _load_source(spec: str) -> ConceptSource:
    source = ss._load(spec)
    if not isinstance(source, ConceptSource):
        raise TypeError(f"{spec} is not a ConceptSource (name, things, stuff)")
    return source


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="the scan's PLY, or its tileset.json")
    parser.add_argument("tiles", type=Path, help="its tileset directory (tileset.json)")
    parser.add_argument("--vocabulary", default=None, help="module:Class, or a vocabulary .json")
    parser.add_argument("--concepts", default=None, help="module:Class, a ConceptSource")
    parser.add_argument("--masks", default=None, help="module:Class, the class-free MaskSource")
    parser.add_argument("--truth", type=Path, default=None, help="labels.json: oracle concepts")
    parser.add_argument("--embedder", default="segment_scene:FakeEmbedder")
    parser.add_argument("--words", type=Path, default=None, help="tag vocabulary, one a line")
    parser.add_argument("--views", type=int, default=ss.VIEW_COUNT)
    parser.add_argument("--max-views", type=int, default=ss.MAX_VIEWS)
    parser.add_argument("--overview-views", type=int, default=OVERVIEW_VIEWS)
    parser.add_argument("--renderer", choices=("cpu", "gsplat"), default="cpu")
    parser.add_argument("--max-scale-m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--stand-in", action="store_true", help="mark the run a stand-in")
    parser.add_argument("--out", type=Path, required=True, help="instances.json etc. go here")
    parser.add_argument("--overviews", type=Path, default=None, help="PNGs to check, here")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--opacity-min", type=float, default=scene_plants.PACKAGE_OPACITY_MIN)
    parser.add_argument("--tile-gaussians", type=int, default=scene_plants.PACKAGE_TILE_GAUSSIANS)
    args = parser.parse_args()
    from_tiles = args.source.name.endswith(".json")
    if from_tiles:
        from splat_render import load_tileset

        splats = load_tileset(args.source)
        rows, row_count = np.arange(len(splats)), len(splats)
    else:
        splats, rows, row_count = ss.load_source(args.source, args.opacity_min)
    embedder = ss.load_embedder(args.embedder)
    words = []
    if args.words:
        text = args.words.read_text(encoding="utf-8").splitlines()
        words = [w.strip() for w in text if w.strip() and not w.lstrip().startswith("#")]
    say = lambda message: print(message, flush=True)
    cameras = None
    free_factory = None
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        ground = ground_layer(splats)
        _, centroids, _, edge, _ = ground_cells(splats.positions, ground.flag)
        cameras = ss.plan_views(
            splats.positions, args.views, observers=ss.observer_points(splats), edge=edge,
            solid=centroids, max_views=args.max_views,
        )  # fmt: skip
        thing, obj, stuff = yard_truth(truth, rows)
        source: ConceptSource = OracleConcepts(splats, thing, obj, stuff, cameras)
        vocabulary: Vocabulary = FixedVocabulary(list(YARD_CONCEPTS))
        levels = ss.truth_levels(truth, rows, splats.colours)

        def oracle_free(cams: list[Camera]) -> ss.MaskSource:
            return ss.OracleMasks(splats, levels, cams)

        free_factory = oracle_free
        free = None
    else:
        if not (args.vocabulary and args.concepts):
            parser.error("give --vocabulary and --concepts (or --truth)")
        vocabulary = _load_vocabulary(args.vocabulary)
        source = _load_source(args.concepts)
        free = ss.load_masks(args.masks) if args.masks else None
        ground = None
    result = segment_concepts(
        splats, vocabulary, source, free, embedder, words,
        cameras=cameras, view_count=args.views, max_views=args.max_views,
        overview_count=args.overview_views, renderer=ss.make_renderer(args.renderer),
        max_scale_m=args.max_scale_m, workers=args.workers, cpus=args.cpus,
        memory_bytes=None if args.memory_gb is None else args.memory_gb * float(1 << 30),
        ground=ground, progress=say, free_factory=free_factory,
    )  # fmt: skip
    if from_tiles:
        tiles = ss.tile_binding_by_position(args.tiles, splats.positions, result.splat_id)
    else:
        tiles = ss.tile_binding(
            args.tiles, args.source, rows, result.splat_id, row_count,
            opacity_min=args.opacity_min, tile_gaussians=args.tile_gaussians,
        )  # fmt: skip
    tiles = rebind_instances.rebind(args.tiles, tiles)
    document = concept_document(
        result,
        tiles,
        embedder=embedder,
        vocabulary_size=len(words) + len(result.concepts),
        vocabulary_model=vocabulary.name,
        segmenter=source.name,
        stand_in=args.stand_in,
    )
    document["tilesEncoding"] = rebind_instances.TILES_ENCODING
    ss.write_instances(args.out, document, result.instances)
    (args.out / "vocabulary.json").write_text(
        json.dumps(
            {
                "model": vocabulary.name,
                "things": [c.to_json() for c in result.concepts if c.kind == "thing"],
                "stuff": [c.to_json() for c in result.concepts if c.kind == "stuff"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    if args.overviews:
        from PIL import Image

        args.overviews.mkdir(parents=True, exist_ok=True)
        render_overviews(result, splats, args.overviews)
        sheet = np.concatenate(
            [np.concatenate(result.overviews[k : k + 4], axis=1) for k in range(0, 12, 4)
             if len(result.overviews[k : k + 4]) == 4], axis=0,
        ) if len(result.overviews) >= 4 else None  # fmt: skip
        if sheet is not None:
            Image.fromarray(sheet).save(args.overviews / "vocabulary-views.jpg", quality=85)
    usage = {"gaussians": len(splats), **ss.peak_usage(started)}
    say(ss.usage_line(usage))
    top = [r for r in document["instances"] if r["parent"] is None]
    summary = {
        "instances": len(result.instances),
        "objects": sum(1 for r in top if r.get("kind") == "thing"),
        "namedObjects": sum(
            1 for r in top if r.get("nameSource") == "concept" and r["kind"] == "thing"
        ),
        "groundInstances": sum(1 for r in top if r.get("kind") == "ground"),
        "assignedShare": round(float((result.splat_id > 0).mean()), 4),
        "concepts": [c.to_json() for c in result.concepts],
        **result.lift.stats,
        "ground": document["ground"],
        "timingsS": {k: round(v, 2) for k, v in result.timings.items()},
        "usage": usage,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
