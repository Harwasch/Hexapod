"""Concept-first scene objects: bake-off candidate C (docs/SCENE_OBJECTS.md §3b; research
notes §1.5, "Alternative B").

`segment_scene.py` cuts every view into class-free masks and names what it lifted
afterwards. This module turns that round: it first asks what is in the scene, then looks for
each named thing in every view, so objects are born with their names.

1. **Vocabulary** (`Vocabulary`). A vision-language model (`concept_models.QwenVocabulary`,
   Qwen3-VL, Apache-2.0) looks at `OVERVIEW_VIEWS` overview renders and lists the scene's
   *things* (countable objects: "cable spool", "pumpkin", with a few other words a detector
   may know them by), each with one of the viewer's broad categories
   (`data/categories.json`), and which of `data/ground_cover.json`'s classes the ground
   shows (*stuff*). At most `MAX_THINGS` and `MAX_STUFF`: the list is short.
2. **Concepts in every view** (`ConceptSource`). Things: every instance of each thing in
   every view, one mask each, along *camera paths* (`camera_paths`: the plan's views in an
   order a video tracker can follow), so a tracker that keeps ids along a path (SAM 3's)
   tells this module which masks are one object (`ConceptMask.track`). Stuff: a per-pixel
   cover class over the ground (`StuffMap`).
3. **Ground** (`ground_layer`). Which splats lie on the ground, from geometry: the
   bake-off's shared ground pass (`ground_pass.py`), behind a small adapter (`GROUND_PASS`;
   `scene_plants`' slope filter is the other choice). Its "unknown" splats (in the ground
   layer where no ground was seen: under an object's footprint, under a canopy) are ground
   unless the masks say otherwise. Cells never straddle the ground (`ground_cells`), and a
   ground cell that the views keep seeing inside a thing's mask (`PROMOTE_SHARE`, less for
   the unknown) is the thing's, not the ground's: the spool's lower flange.
4. **Lift** (`lift_concepts`), by `segment_scene`'s own voting: thing masks vote for the
   cells they cover and co-occurrence over co-visibility joins cells into objects
   (`segment_scene._grow`); each object is named by the concept most of its votes carry,
   and objects that one track spans are one. Ground cells take the cover class most views
   gave them, smoothed and cut into connected regions as candidate A cuts them
   (`segment_ground_first._regions`), in the shared ground schema (§3b): one top-level
   instance per class in the category Ground & soil, its regions as its children.
5. **Leftovers.** Cells that are neither ground nor in a named thing are lifted from a
   class-free pass (SAM 2 automatic masks), as `segment_scene` does: what nobody named
   still becomes an object (named by its tags, as today). The same class-free masks give
   every object its parts (levels below the concept level).
6. **Meaning.** `segment_scene.describe` embeds and tags every instance as today (search
   stays SigLIP's); a named object's name, category and first tag come from its concept.

`instances.json` is the contract's v1 with the bake-off's fields (§3b): per instance `kind`,
`name`, `nameSource` (`vlm` for a concept's name, `ground-cover`), `cover`, `scaleM`, and
`concept`; at the root `variant`, `ground`, and `concepts` (the vocabulary and the models).

Usage (`infra/modal/segment.py` runs it as the variant `concept-first-standin`: a `seg-*` push
whose head commit says `[segment|names=spool,pumpkin|variant=concept-first-standin|views=64]`):
    python concept_scene.py TILESET.json TILES_DIR --out OUT \\
        --vlm concept_models:QwenVocabulary --concepts concept_models:GroundedSam2Concepts \\
        --masks segment_models:Sam2LargeMasks --embedder segment_models:SiglipEmbedder \\
        --vocabulary data/open_vocabulary.txt --renderer gsplat --stand-in --check check.png
    python concept_scene.py yard/source/splat.ply yard/splat --out OUT \\
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
from scipy.spatial import cKDTree

import rebind_instances
import scene_categories
import scene_plants
import segment_ground_first as sgf
import segment_scene as ss
from splat_render import Camera, SplatIndex, Splats, render

# ----------------------------------------------------------------------------- constants

#: What a run is published as (`variant` in `instances.json`): SAM 3's run is candidate C;
#: the stand-in's has a name of its own, so it is never published as C.
VARIANTS = {
    "sam3": {
        "name": "concept-first",
        "label": "C · Concept first",
        "about": (
            "A vision-language model lists the scene's things and ground cover; SAM 3 finds "
            "each of them in every view, the masks are voted onto the splats, and what nobody "
            "named comes from a class-free pass."
        ),
    },
    "standin": {
        "name": "concept-first-standin",
        "label": "C (stand-in) · Concept first, Grounding DINO + SAM 2",
        "about": (
            "Concept first as C, with Grounding DINO and SAM 2 standing in for SAM 3 (whose "
            "weights are gated): a vision-language model lists the things and ground cover, "
            "each is found in every view and voted onto the splats."
        ),
    },
}
#: Overview renders the vocabulary is read from (rings at two scales and elevations).
OVERVIEW_VIEWS = 12
#: The vocabulary is kept short: at most this many things, cover classes, and other words
#: per thing.
MAX_THINGS = 12
MAX_STUFF = 6
MAX_SYNONYMS = 2
#: Camera paths (`camera_paths`): consecutive views whose eyes are at most this share of
#: the scan's radius apart, and whose axes turn at most `PATH_TURN_DEG`, are one path.
PATH_STEP = 0.75
PATH_TURN_DEG = 50.0
#: A ground cell is the thing's when it was inside a thing's mask in at least this share of
#: its visible weight, in at least `PROMOTE_VIEWS` views (`lift_concepts`); a cell the
#: ground pass was unsure of (`Ground.unsure`), in this share and one view.
PROMOTE_SHARE = 0.6
PROMOTE_VIEWS = 2
PROMOTE_SHARE_UNSURE = 0.35
#: A cell off the ground is a named thing's when the thing masks held at least this share
#: of its visible weight, in at least `THING_VIEWS` views: a pumpkin's mask that strays
#: over the hay in a view or two does not make the hay a pumpkin.
THING_SHARE = 0.5
THING_VIEWS = 2
#: Two objects of one concept are one when a track holds at least this share of each.
TRACK_SHARE = 0.5
#: Two touching objects of one concept are one when, over the views where both are in a
#: mask of it, they are in the same mask at least this share of the time (and in at least
#: `segment_scene.MIN_COVISIBLE` views): a spool's bottom flange, which the masks leave out
#: in some views, and its drum. Two pumpkins side by side are each in a mask of their own.
SAME_MASK_SHARE = 0.8
CONCEPT_JOIN_ROUNDS = 4
#: Which ground pass `ground_layer` runs: "shared" (`ground_pass.py`, the bake-off's one
#: ground pass) or "slope" (`scene_plants`' Vosselman filter).
GROUND_PASS = "shared"
#: Splats the slope filter is fitted on at most (seeded); every splat is then measured.
GROUND_SAMPLE = 1_000_000
#: Gaussians larger than this, or fainter than `GROUND_OPACITY`, are left out of the fit:
#: floaters under the ground would pull the terrain down.
GROUND_MAX_SCALE_M = 0.5
GROUND_OPACITY = 0.1
#: The category every ground instance is in (§3b).
GROUND_CATEGORY = sgf.GROUND_CATEGORY
#: Words a VLM may use for a cover class, beyond its id and name (`cover_of`).
COVER_WORDS = {
    "lawn": "grass", "short grass": "grass", "meadow": "tall-grass", "weeds": "tall-grass",
    "soil": "dirt", "earth": "dirt", "bare soil": "dirt", "bare ground": "dirt",
    "ground": "dirt", "pebbles": "gravel", "stones": "gravel", "crushed stone": "gravel",
    "stone": "rock", "bedrock": "rock", "pavement": "paving", "pavers": "paving",
    "brick": "paving", "road": "asphalt", "wood chips": "mulch", "bark": "mulch",
    "leaves": "leaf-litter", "fallen leaves": "leaf-litter", "pine needles": "forest-floor",
    "straw": "hay", "path": "trail", "footpath": "trail", "dirt path": "trail",
    "puddle": "water", "wooden deck": "deck",
}  # fmt: skip


# --------------------------------------------------------------------------- the concepts


@dataclass(frozen=True)
class Concept:
    """One named concept of a scene: a thing (countable) or stuff (a ground-cover class)."""

    name: str
    kind: str  # "thing" | "stuff"
    #: A `data/categories.json` id.
    category: str = scene_categories.OTHER
    #: What a segmenter is asked for, best first (default: the name). For stuff, the cover
    #: class's own prompts (`data/ground_cover.json`).
    prompts: tuple[str, ...] = ()
    #: For stuff, its `data/ground_cover.json` class id.
    cover: str | None = None

    @property
    def query(self) -> str:
        return self.prompts[0] if self.prompts else self.name

    @property
    def queries(self) -> tuple[str, ...]:
        return self.prompts or (self.name,)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "category": self.category}
        if self.kind == "stuff":
            out["cover"] = self.cover
        elif self.prompts and self.prompts != (self.name,):
            out["prompts"] = list(self.prompts)
        return out


def cover_of(word: str, classes: Sequence[sgf.CoverClass] | None = None) -> sgf.CoverClass | None:
    """The `data/ground_cover.json` class a word names: its id, its name, or `COVER_WORDS`."""
    classes = list(classes or sgf.cover_classes()[0])
    key = " ".join(str(word).strip().lower().replace("_", " ").split())
    for c in classes:
        if key in (c.id, c.id.replace("-", " "), c.name.lower()):
            return c
    found = COVER_WORDS.get(key)
    return next((c for c in classes if c.id == found), None)


def clean_concepts(
    concepts: Sequence[Concept],
    *,
    max_things: int = MAX_THINGS,
    max_stuff: int = MAX_STUFF,
    categories: Sequence[str] | None = None,
) -> list[Concept]:
    """Things then stuff, in the order given. A thing: its name once (lower case, trimmed),
    an unknown category made `other`, at most `MAX_SYNONYMS` other words, at most
    `max_things`. Stuff: the cover class it names (`cover_of`; a word that names none is
    dropped), each class once, with the class's name and prompts, at most `max_stuff`."""
    known = set(categories or scene_categories.category_ids())
    classes, _ = sgf.cover_classes()
    seen: set[str] = set()
    things: list[Concept] = []
    stuff: list[Concept] = []
    for c in concepts:
        name = " ".join(str(c.name).strip().lower().split())
        if not name:
            continue
        if c.kind == "stuff":
            cover = cover_of(c.cover or name, classes)
            if cover is None or f"cover:{cover.id}" in seen or len(stuff) >= max_stuff:
                continue
            seen.add(f"cover:{cover.id}")
            stuff.append(Concept(cover.name, "stuff", GROUND_CATEGORY, cover.prompts, cover.id))
            continue
        if name in seen or len(things) >= max_things:
            continue
        seen.add(name)
        others = [" ".join(str(p).strip().lower().split()) for p in c.prompts]
        others = [p for p in dict.fromkeys(others) if p and p != name][:MAX_SYNONYMS]
        category = c.category if c.category in known else scene_categories.OTHER
        things.append(Concept(name, "thing", category, (name, *others)))
    return things + stuff


def with_cover(concepts: Sequence[Concept]) -> list[Concept]:
    """`concepts`, and when they name no cover class, every class of
    `data/ground_cover.json` (as candidate A classifies the ground): the ground always has
    one."""
    if any(c.kind == "stuff" for c in concepts):
        return list(concepts)
    every = [Concept(c.id, "stuff") for c in sgf.cover_classes()[0]]
    return clean_concepts([*concepts, *every], max_stuff=len(every))


def split_concepts(concepts: Sequence[Concept]) -> tuple[list[Concept], list[Concept]]:
    return [c for c in concepts if c.kind == "thing"], [c for c in concepts if c.kind == "stuff"]


def load_concepts(path: Path) -> list[Concept]:
    """A vocabulary file: `{"things": [{"name", "category"?, "prompts"?}, ...], "cover":
    ["grass", ...]}` (a run's `concepts` block; entries of `stuff` are read as cover too)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = [
        Concept(str(r["name"]), "thing", str(r.get("category", "")), tuple(r.get("prompts", ())))
        for r in data.get("things", [])
    ]
    for entry in [*data.get("cover", []), *data.get("stuff", [])]:
        word = (entry.get("cover") or entry.get("name")) if isinstance(entry, dict) else entry
        out.append(Concept(str(word), "stuff"))
    return clean_concepts(out)


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
    #: (n,) bool, within `flag`: on the layer's height, but where no ground was seen.
    unsure: np.ndarray | None = None


def shared_ground(splats: Splats) -> Ground:
    """`ground_pass.ground_pass` (SMRF on a robust lowest surface; the bake-off's shared
    pass): its GROUND and BELOW (floaters under the terrain) are ground; its UNKNOWN (in the
    layer where no ground was seen) is ground the masks may claim back (`Ground.unsure`)."""
    import ground_pass as gp

    g = gp.ground_pass(splats.positions, splats.opacities, splats.scales)
    flag = g.label != gp.ABOVE
    return Ground(
        flag,
        g.hag,
        "ground_pass (SMRF)",
        {
            "layerM": round(float(g.layer_m), 4),
            "belowM": round(float(g.below_m), 4),
            "cellM": round(float(g.terrain.cell), 4),
            "share": round(float(flag.mean()), 4),
            "labels": g.stats.get("shares", {}),
            "seenShare": g.stats.get("terrain", {}).get("seenShare"),
        },
        g.label == gp.UNKNOWN,
    )


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
    """The ground pass `GROUND_PASS` names: swapping passes is that one constant."""
    return shared_ground(splats) if GROUND_PASS == "shared" else slope_ground(splats)


def ground_cells(
    positions: np.ndarray, ground: Ground, **kwargs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """`segment_scene.supervoxels`, with no cell holding both ground and other splats (as
    `ground_pass.split_cells` cuts them), nor the ground pass's sure and unsure ground: each
    voxel is split three ways. Returns the cells, centroids, counts, edge, and per cell
    whether it is ground, and whether that ground is unsure."""
    cell, _, _, edge = ss.supervoxels(positions, **kwargs)  # type: ignore[arg-type]
    state = np.asarray(ground.flag, np.int64)
    if ground.unsure is not None:
        state = np.where(np.asarray(ground.unsure, bool) & ground.flag, 2, state)
    key = cell.astype(np.int64) * 3 + state
    unique, inverse = np.unique(key, return_inverse=True)
    cell = inverse.astype(np.int32).reshape(-1)
    counts = np.bincount(cell, minlength=unique.size)
    pos = np.asarray(positions, np.float64)
    centroids = (
        np.stack([np.bincount(cell, pos[:, k], unique.size) for k in range(3)], axis=1)
        / counts[:, None]
    )
    kind = unique % 3
    return cell, centroids, counts, float(edge), kind > 0, kind == 2


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


def cached_things(
    cache: Path | None,
    views: Sequence[ss.View],
    source: ConceptSource,
    things: Sequence[Concept],
    n_cells: int,
    tag: str = "",
) -> list[list[ConceptMask]]:
    """`source.things` for one path's views, each frame's masks kept in `cache` as
    `boxmask-<camera key>-<things>.npz` (the name `infra/modal/segment.py` keeps in a run's
    `cache.tar`), so a run can be re-assembled on a CPU without the segmenter."""
    if not things:
        return [[] for _ in views]
    names = hashlib.sha1("|".join(c.query for c in things).encode()).hexdigest()[:8]
    paths = (
        [cache / f"boxmask-{ss._cache_key(v.camera, n_cells, tag)}-{names}.npz" for v in views]
        if cache is not None
        else []
    )
    if paths and all(p.exists() for p in paths):
        out = []
        for p in paths:
            with np.load(p) as z:
                out.append(
                    [
                        ConceptMask(m.astype(bool), int(c), float(s), int(t))
                        for m, c, s, t in zip(
                            z["masks"], z["concept"], z["score"], z["track"], strict=True
                        )
                    ]
                )
        return out
    found = source.things([v.rgb for v in views], things) or [[] for _ in views]
    for p, view, masks in zip(paths, views, found, strict=False):
        h, w = view.rgb.shape[:2]
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            masks=np.array([m.mask for m in masks], bool).reshape(len(masks), h, w),
            concept=np.array([m.concept for m in masks], np.int64),
            score=np.array([m.score for m in masks], np.float64),
            track=np.array([m.track for m in masks], np.int64),
        )
        tmp.replace(p)
    return found


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
    #: Per instance: the stuff index of a ground instance (a class or one of its regions).
    stuff: np.ndarray
    #: Per instance: True for the coarsest instance of an object (level 0).
    top: np.ndarray
    #: Per cell: on the ground after promotion; and the promoted cells.
    ground: np.ndarray
    promoted: np.ndarray
    #: Per cell: its cover class and region (-1 off the ground), and how sure the class is.
    cover: np.ndarray
    region: np.ndarray
    confidence: np.ndarray
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


def concept_joins(
    region: np.ndarray,
    region_concept: np.ndarray,
    votes: ConceptVotes,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Touching objects of one concept that the masks never tell apart are one object: for
    each pair of regions joined by an edge of the cell graph (`a`, `b`) and of one concept,
    over the views where each is in some thing mask (`segment_scene._seen`, the region judged
    as a whole), they join when they are in the same mask in at least `SAME_MASK_SHARE` of
    them and at least `segment_scene.MIN_COVISIBLE` of them. A view where one of them is in
    no mask says nothing (the mask left a flange out), unlike `segment_scene._grow`'s
    agreement. Returns the regions renumbered, and how many pairs joined."""
    n = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
    if n < 2 or not votes.things:
        return region, 0
    obs = ss._observed(votes.things, 0, region >= 0)
    seen = ss._seen(obs, region)
    pa, pb = ss._pairs(region, a, b, n)
    keep = region_concept[pa] == region_concept[pb]
    pa, pb = pa[keep], pb[keep]
    masked = seen.best >= 0
    best_of: dict[int, dict[int, int]] = {}
    for r, v, m in zip(seen.region[masked], seen.view[masked], seen.best[masked], strict=True):
        best_of.setdefault(int(r), {})[int(v)] = int(m)

    def agree(mx: dict[int, int], my: dict[int, int]) -> tuple[int, int]:
        both = mx.keys() & my.keys()
        return sum(1 for v in both if mx[v] == my[v]), len(both)

    scored = []
    for x, y in zip(pa.tolist(), pb.tolist(), strict=True):
        same, both = agree(best_of.get(x, {}), best_of.get(y, {}))
        if same >= ss.MIN_COVISIBLE and same >= SAME_MASK_SHARE * both:
            scored.append((-same, x, y))
    # Strongest first; a join is checked again between the groups as they have grown, so a
    # fragment touching two pumpkins does not chain them into one.
    root = np.arange(n)
    groups = {r: dict(best_of.get(r, {})) for r in range(n)}

    def find(r: int) -> int:
        while root[r] != r:
            root[r] = root[root[r]]
            r = int(root[r])
        return r

    joins = 0
    for _, x, y in sorted(scored):
        gx, gy = find(x), find(y)
        if gx == gy:
            continue
        same, both = agree(groups[gx], groups[gy])
        if same < SAME_MASK_SHARE * both:
            continue
        keep, gone = min(gx, gy), max(gx, gy)
        root[gone] = keep
        for view, mask in groups.pop(gone).items():
            groups[keep].setdefault(view, mask)
        joins += 1
    if not joins:
        return region, 0
    final = np.array([find(r) for r in range(n)], np.int64)
    out = np.where(region >= 0, final[np.maximum(region, 0)], -1)
    return ss._relabel(out), joins


def cover_regions(
    votes: ConceptVotes,
    centroids: np.ndarray,
    counts: np.ndarray,
    ground: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell its cover class (an index into the stuff; -1 off the ground), its region
    (-1 off the ground) and the class's smoothed probability there. As candidate A does it
    (`segment_ground_first.run`): each ground cell's class weights normalised (a cell no view
    classified takes the nearest classified ground cell's), smoothed over its
    `COVER_NEIGHBOURS` nearest ground cells, a class under `COVER_MIN_CLASS_SHARE` of the
    ground given to the cells' next best, then connected regions (`_regions`), those under
    `COVER_MIN_REGION` splats folded into their neighbours (`_fold_small`)."""
    n = votes.n_cells
    klass_of = np.full(n, -1, np.int64)
    region_of = np.full(n, -1, np.int64)
    confidence_of = np.zeros(n)
    gcells = np.flatnonzero(ground)
    k = votes.n_stuff
    if gcells.size == 0 or k == 0:
        return klass_of, region_of, confidence_of
    sums = votes.stuff_weight[:, :k].astype(np.float64)
    weight = sums.sum(axis=1)
    probability = np.full((n, k), 1.0 / k)
    voted = weight > 0
    probability[voted] = sums[voted] / weight[voted, None]
    if voted[gcells].any():
        known = gcells[voted[gcells]]
        missing = gcells[~voted[gcells]]
        if missing.size:
            _, near = cKDTree(centroids[known]).query(centroids[missing], k=1, workers=-1)
            probability[missing] = probability[known[near]]
    smooth = sgf._smooth(
        probability[gcells], centroids[gcells], sgf.COVER_NEIGHBOURS, sgf.COVER_ROUNDS
    )
    klass = np.argmax(smooth, axis=1)
    ground_splats = counts[gcells].astype(np.float64)
    total = float(ground_splats.sum())
    local = np.full(n, -1, np.int64)
    local[gcells] = np.arange(gcells.size)
    keep = (local[a] >= 0) & (local[b] >= 0)
    ga, gb = local[a[keep]], local[b[keep]]
    share = np.bincount(klass, ground_splats, k) / max(total, 1.0)
    rare = share < sgf.COVER_MIN_CLASS_SHARE
    if rare.any() and (~rare).any():
        second = np.where(rare[None, :], -1.0, smooth).argmax(axis=1)
        klass = np.where(rare[klass], second, klass)
    least = max(sgf.COVER_MIN_REGION, sgf.COVER_MIN_REGION_SHARE * total)
    klass, region = sgf._regions(klass, ground_splats, ga, gb, least)
    region = sgf._fold_small(region, klass, ground_splats, centroids[gcells], least)
    klass_of[gcells] = klass
    region_of[gcells] = region
    confidence_of[gcells] = smooth[np.arange(gcells.size), klass]
    return klass_of, region_of, confidence_of


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
    unsure_cell: np.ndarray | None = None,
    cell_reach: np.ndarray | None = None,
) -> ConceptLift:
    """Objects from the concept votes (module docstring, steps 3-5).

    Level 0 holds the named objects (thing masks, `segment_scene._grow`; joined along
    tracks), then the leftovers (class-free level-0 masks over what is neither ground nor
    named), then one instance per cover class over the ground. The class-free levels refine
    every object off the ground into parts below it (`segment_scene._hierarchy`).
    `unsure_cell`: ground cells the ground pass was unsure of, which the thing masks claim
    on less evidence (`PROMOTE_SHARE_UNSURE`)."""
    n_cells = len(centroids)
    a, b = ss._cell_graph(centroids, edge)
    stats: dict[str, object] = {"cells": n_cells, "edges": int(a.size)}
    seen_w = votes.seen.copy()
    for v in votes.free:
        np.maximum.at(seen_w, v.cells, v.weight)
    seen = seen_w >= ss.MIN_VISIBLE_PX
    unsure = np.zeros(n_cells, bool) if unsure_cell is None else unsure_cell & ground_cell
    share = votes.thing_in / np.maximum(votes.seen, 1e-9)
    promoted = ground_cell & (
        ((share >= PROMOTE_SHARE) & (votes.thing_views >= PROMOTE_VIEWS))
        | (unsure & (share >= PROMOTE_SHARE_UNSURE) & (votes.thing_views >= 1))
    )
    ground = ground_cell & ~promoted
    stats.update(
        groundCells=int(ground_cell.sum()),
        promotedCells=int(promoted.sum()),
        seenCells=int(seen.sum()),
    )

    # Named things: cells the thing masks hold most of the time (`THING_SHARE`), and the
    # ground they claimed.
    in_thing = ((share >= THING_SHARE) & (votes.thing_views >= THING_VIEWS) & ~ground) | promoted
    stats["thingCells"] = int(in_thing.sum())
    region = np.full(n_cells, -1, np.int64)
    if votes.things and in_thing.any():
        region, rounds = ss._grow(votes.things, 0, in_thing, a, b, counts)
        stats["thingRounds"] = rounds
        region = _drop_small(ss._absorb(region, a, b, in_thing), counts)

    def concepts_of(region: np.ndarray) -> np.ndarray:
        n = int(region.max()) + 1 if region.size and region.max() >= 0 else 0
        weight = np.zeros((n, max(votes.n_things, 1)))
        valid = region >= 0
        np.add.at(weight, region[valid], votes.concept_weight[valid, : weight.shape[1]])
        return np.argmax(weight, axis=1) if n else np.zeros(0, np.int64)

    region_concept = concepts_of(region)
    if region_concept.size and votes.n_things:
        region, joins = track_joins(region, region_concept, votes)
        stats["trackJoins"] = joins
        region_concept = concepts_of(region) if joins else region_concept
        stats["conceptJoins"] = 0
        for _ in range(CONCEPT_JOIN_ROUNDS):
            region, joins = concept_joins(region, region_concept, votes, a, b)
            if not joins:
                break
            stats["conceptJoins"] += joins
            region_concept = concepts_of(region)
    n_regions = int(region_concept.size)
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

    # The ground: one instance per cover class, its connected regions below it (§3b).
    cover, cover_region, confidence = cover_regions(votes, centroids, counts, ground, a, b)
    base = n_regions + n_left
    level0 = np.where(objects >= 0, objects + 1, 0)
    level0[ground] = base + 1 + np.maximum(cover[ground], 0)
    labels = [level0]
    # Parts: the class-free levels over everything off the ground.
    for level in range(levels):
        if not (in_free[level] & ~ground).any():
            continue
        joined, _ = ss._grow(free, level, in_free[level] & ~ground, a, b, counts)
        joined = _drop_small(ss._absorb(joined, a, b, seen & ~ground), counts)
        joined[ground] = -1
        labels.append(joined + 1)
    if len(labels) == 1:
        labels.append(np.zeros(n_cells, np.int64))
    # A class's regions are its children (a class of one region has none: `_hierarchy`
    # makes no child that is its whole parent).
    labels[1] = np.where(ground, cover_region + 1, labels[1])
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
            stuff[k] = lab - base
        else:
            kind.append("thing")
            if 0 <= lab < n_regions:
                thing[k] = int(region_concept[lab])
    top = lifted.parent == 0
    lifted.stats["unassignedShare"] = round(
        float(counts[lifted.cell_id == 0].sum()) / max(float(counts.sum()), 1.0), 4
    )
    lifted.stats["coverRegions"] = len(np.unique(cover_region[ground])) if ground.any() else 0
    return ConceptLift(
        lifted, kind, thing, stuff, top, ground, promoted, cover, cover_region, confidence,
        lifted.stats,
    )  # fmt: skip


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
    cache: Path | None = None,
) -> ConceptSegmentation:
    """Ground, cells, views, vocabulary, concept and class-free masks, the lift and
    meaning, for a scan held in memory. `free_factory(cameras)` builds the class-free mask
    source that needs the cameras (`segment_scene.OracleMasks` in tests). `cache`: every
    view's image and class-free masks are kept there (`segment_scene.cached_raster`,
    `cached_masks`), and the VLM's answer (`names.json`), so a run can be re-assembled."""
    say = progress or (lambda message: None)
    timings: dict[str, float] = {}
    mark = time.perf_counter()
    ground = ground or ground_layer(splats)
    timings["groundS"] = time.perf_counter() - mark
    say(f"ground: {ground.source}, {ground.info.get('share')} of the splats")
    mark = time.perf_counter()
    cell, centroids, counts, edge, ground_cell, unsure_cell = ground_cells(splats.positions, ground)
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
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
        for camera in overview_cameras(splats.positions, overview_count):
            if renderer is not None:
                overviews.append(
                    ss.cached_raster(cache, renderer, view_splats, camera, n_cells, tag, index)
                )
            else:
                frame = render(view_splats, camera, index=index)
                overviews.append(np.round(np.clip(frame.rgb, 0, 1) * 255).astype(np.uint8))
        concepts = with_cover(clean_concepts(vocabulary.concepts(overviews)))
        if cache is not None:
            (cache / "names.json").write_text(
                json.dumps(
                    {
                        "vocabulary": [c.to_json() | {"kind": c.kind} for c in concepts],
                        "answer": getattr(vocabulary, "answer", None),
                    }
                ),
                encoding="utf-8",
            )
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
                    cache, renderer, view_splats, view.camera, n_cells, tag, index
                )
                view = ss.View(view.camera, image, view.cell, view.purity, samples=view.rgb)
            rastered = time.perf_counter()
            clock["rasterS"] += rastered - now
            views.append(view)
            drawn = tag + ("" if renderer is None else renderer.name)
            masks = [] if free is None else ss.cached_masks(cache, view, free, n_cells, drawn)
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
                found = cached_things(cache, pending, source, things, n_cells, drawn)
                for v, masks_of in zip(pending, found, strict=True):
                    votes.add_things(v, masks_of, track_base)
                track_base += 1 + max((m.track for f in found for m in f), default=0)
                pending = []
                clock["thingsS"] += time.perf_counter() - started
            say(f"view {k + 1}/{len(cameras)} done")
            mark = time.perf_counter()
    timings.update(clock)
    mark = time.perf_counter()
    lift = lift_concepts(
        votes, centroids, counts, edge, ground_cell, unsure_cell=unsure_cell, cell_reach=reach
    )
    timings["liftS"] = time.perf_counter() - mark
    lift.stats["views"] = len(views)
    lift.stats["paths"] = len(paths)
    lift.stats["cellEdgeM"] = round(edge, 4)
    mark = time.perf_counter()
    words = list(dict.fromkeys([*words, *(c.name.lower() for c in concepts)]))
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


def _scale(instance: ss.Instance) -> float:
    """Half the bounds' diagonal, metres (§3b `scaleM`)."""
    extent = np.asarray(instance.bounds_max) - np.asarray(instance.bounds_min)
    return round(float(np.linalg.norm(extent)) / 2, 3)


def name_instances(
    instances: list[ss.Instance], lift: ConceptLift, concepts: Sequence[Concept]
) -> list[dict[str, object]]:
    """Names, categories and first tags from the concepts, in place; and per instance the
    fields `concept_document` adds (the §3b schema).

    A named object (`nameSource: "vlm"`: the vision-language model's word for it) and its
    parts take the concept's category, so the panel lists the object whole; the object's
    first tag is the concept, so search finds it by name. A ground instance -- a cover class,
    or a region of one -- is in `GROUND_CATEGORY`, named by its class
    (`nameSource: "ground-cover"`), with its class's `cover` id and one tag, the class at its
    mean confidence; static. Everything else keeps what `describe` gave it."""
    things, stuff = split_concepts(concepts)
    region_confidence: dict[int, float] = {}
    if lift.ground.any():
        leaf = lift.lifted.cell_id
        on = lift.ground & (leaf > 0)
        sums = np.bincount(leaf[on], lift.confidence[on], len(instances) + 1)
        cells = np.bincount(leaf[on], minlength=len(instances) + 1)
        for k in np.flatnonzero(cells):
            region_confidence[int(k)] = float(sums[k] / cells[k])
    extra: list[dict[str, object]] = []
    for k, instance in enumerate(instances):
        record: dict[str, object] = {"kind": lift.kind[k], "scaleM": _scale(instance)}
        if lift.kind[k] == "ground":
            cover = stuff[int(lift.stuff[k])]
            members = [k + 1] + [i.id for i in instances if i.parent == k + 1]
            known = [region_confidence[m] for m in members if m in region_confidence]
            score = round(float(np.mean(known)) if known else 0.0, 4)
            instance.category = GROUND_CATEGORY
            instance.behaviour = "static"
            instance.tags = [{"label": cover.name.lower(), "score": score}]
            record.update(name=cover.name, nameSource="ground-cover", cover=cover.cover)
        elif int(lift.thing[k]) >= 0:
            concept = things[int(lift.thing[k])]
            instance.category = concept.category
            record["concept"] = concept.name
            if lift.top[k]:
                record["name"] = concept.name[:1].upper() + concept.name[1:]
                record["nameSource"] = "vlm"
                rest = [t for t in instance.tags if t["label"] != concept.name]
                instance.tags = [{"label": concept.name, "score": 1.0}, *rest][: ss.TAGS_TOP_K]
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
    """`instances.json` (`segment_scene.instances_document`) with the bake-off's fields
    (§3b): per instance `kind`, `name`, `nameSource`, `cover`, `scaleM`, `concept`; at the
    root `variant` (`VARIANTS`: the stand-in's own name when `stand_in`), `ground` and
    `concepts` (the vocabulary and the models)."""
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
        "cover": [c.cover for c in stuff],
    }
    lift = result.lift
    splat_cover = lift.cover[result.cell]
    splat_region = lift.region[result.cell]
    report = []
    for s, c in enumerate(stuff):
        mine = splat_cover == s
        if not mine.any():
            continue
        on = lift.ground & (lift.cover == s)
        report.append(
            {
                "class": c.cover,
                "splats": int(mine.sum()),
                "regions": len(np.unique(splat_region[mine])),
                "meanConfidence": round(float(lift.confidence[on].mean()), 4),
            }
        )
    info = result.ground.info
    document["ground"] = {
        "method": result.ground.source,
        **{k: info[k] for k in ("layerM", "cellM", "seenShare") if k in info},
        "promotedCells": int(lift.promoted.sum()),
        "cover": sorted(report, key=lambda r: -r["splats"]),
    }
    document["variant"] = dict(VARIANTS["standin" if stand_in else "sam3"])
    return document


# --------------------------------------------------------------------------- the renders


def _palette(n: int, seed: int) -> np.ndarray:
    """`n + 1` colours, 0 black (as `segment_ground_first`'s check sheet)."""
    out = np.random.default_rng(seed).uniform(0.25, 1.0, (n + 1, 3))
    out[0] = 0.0
    return out


#: Cover classes' colours in the check sheet, in the vocabulary's order.
COVER_PALETTE = np.array(
    [[0.55, 0.8, 0.3], [0.85, 0.75, 0.45], [0.6, 0.45, 0.3], [0.65, 0.65, 0.7],
     [0.3, 0.55, 0.25], [0.95, 0.6, 0.2], [0.4, 0.6, 0.9], [0.8, 0.4, 0.6]]
)  # fmt: skip


def render_check(result: ConceptSegmentation, splats: Splats, out: Path) -> dict[str, Any]:
    """What a person checks (`--check`), as `segment_ground_first.render_check` lays it out:
    per camera (two overviews, two of the plan's views) the scan, its top-level objects
    coloured (ground grey), and its ground by cover class (things dark); under them a legend
    of the largest named objects and the classes, also written beside it as JSON
    (`<out>.legend.json`). Returns the legend."""
    from PIL import Image, ImageDraw

    index = SplatIndex.build(splats)
    cameras = overview_cameras(splats.positions, 4)[:2] + ss.check_cameras(result.views, 3)[1:]
    parent = result.lift.lifted.parent
    top = ss.top_level(parent)[result.splat_id]
    kinds = np.array(["none", *result.lift.kind])
    _, stuff = split_concepts(result.concepts)
    cover = np.where(result.lift.ground, result.lift.cover + 1, 0)[result.cell]
    palette = _palette(parent.size, 7)
    rows = []
    for camera in cameras:
        frame = render(splats, camera, labels=top.astype(np.int64), index=index)
        label = np.maximum(frame.label, 0)
        is_ground = kinds[label] == "ground"
        things = np.where(is_ground[..., None], 0.35, palette[label]) * frame.alpha[..., None]
        cover_frame = render(splats, camera, labels=cover.astype(np.int64), index=index)
        c = COVER_PALETTE[(np.maximum(cover_frame.label, 1) - 1) % len(COVER_PALETTE)]
        c = np.where((cover_frame.label > 0)[..., None], c, 0.08) * cover_frame.alpha[..., None]
        rows.append(np.concatenate([frame.rgb, things, c], axis=1))
    image = Image.fromarray(
        np.round(np.clip(np.concatenate(rows, axis=0), 0, 1) * 255).astype(np.uint8)
    )
    sizes = np.bincount(top, minlength=parent.size + 1)
    named = sorted(
        (
            (int(sizes[k + 1]), k + 1, str(e["name"]))
            for k, e in enumerate(result.extra)
            if parent[k] == 0 and e.get("name") and e["kind"] == "thing"
        ),
        reverse=True,
    )[:16]
    share = np.bincount(cover, minlength=len(stuff) + 1) / max(len(cover), 1)
    colour = [COVER_PALETTE[s % len(COVER_PALETTE)].round(2).tolist() for s in range(len(stuff))]
    legend: dict[str, Any] = {
        "objects": {
            f"{name} #{i} ({size:,} splats)": palette[i].round(2).tolist()
            for size, i, name in named
        },
        "cover": {
            f"{c.name} ({share[s + 1]:.1%} of the splats)": colour[s] for s, c in enumerate(stuff)
        },
    }
    entries = [*legend["objects"].items(), *legend["cover"].items()]
    strip = Image.new("RGB", (image.width, 20 * ((len(entries) + 2) // 3) + 10), (24, 24, 24))
    draw = ImageDraw.Draw(strip)
    for k, (text, rgb) in enumerate(entries):
        x, y = 8 + (k % 3) * (image.width // 3), 6 + 20 * (k // 3)
        draw.rectangle([x, y, x + 14, y + 14], fill=tuple(round(255 * v) for v in rgb))
        draw.text((x + 22, y + 2), text, fill=(235, 235, 235))
    sheet = Image.new("RGB", (image.width, image.height + strip.height))
    sheet.paste(image, (0, 0))
    sheet.paste(strip, (0, image.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    out.with_suffix(".legend.json").write_text(json.dumps(legend, indent=1), encoding="utf-8")
    return legend


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
    Concept("grass", "stuff"),
    Concept("path", "stuff"),
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
    """The command line. It takes `segment_scene.py`'s arguments (`infra/modal/segment.py`
    runs every bake-off variant with them): `--vocabulary` is the tag word list, as there;
    `--coverage-rounds` and `--coverage-views` are accepted and not used (this run's views
    are `--views`, with no coverage rounds)."""
    started = time.time()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="the scan's PLY, or its tileset.json")
    parser.add_argument("tiles", type=Path, help="its tileset directory (tileset.json)")
    parser.add_argument("--out", type=Path, required=True, help="instances.json goes here")
    parser.add_argument("--vlm", default=None, help="module:Class, or a vocabulary .json")
    parser.add_argument("--concepts", default=None, help="module:Class, a ConceptSource")
    parser.add_argument("--masks", default=None, help="module:Class, the class-free MaskSource")
    parser.add_argument("--truth", type=Path, default=None, help="labels.json: oracle concepts")
    parser.add_argument("--embedder", default="segment_scene:FakeEmbedder")
    parser.add_argument("--vocabulary", type=Path, default=None, help="tag words, one a line")
    parser.add_argument("--views", type=int, default=ss.VIEW_COUNT)
    parser.add_argument("--max-views", type=int, default=ss.MAX_VIEWS)
    parser.add_argument("--overview-views", type=int, default=OVERVIEW_VIEWS)
    parser.add_argument("--renderer", choices=("cpu", "gsplat"), default="cpu")
    parser.add_argument("--max-scale-m", type=float, default=None)
    parser.add_argument("--coverage-rounds", type=int, default=0, help="not used")
    parser.add_argument("--coverage-views", type=int, default=0, help="not used")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--cache", type=Path, default=None, help="views, masks, the answer")
    parser.add_argument("--stand-in", action="store_true", help="a run of the stand-in")
    parser.add_argument("--check", type=Path, default=None, help="a PNG: by object, by cover")
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
    if args.vocabulary:
        text = args.vocabulary.read_text(encoding="utf-8").splitlines()
        words = [w.strip() for w in text if w.strip() and not w.lstrip().startswith("#")]

    def say(message: str) -> None:
        print(message, flush=True)

    cameras = None
    free_factory = None
    ground = None
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        ground = ground_layer(splats)
        _, centroids, _, edge, _, _ = ground_cells(splats.positions, ground)
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
        if not (args.vlm and args.concepts):
            parser.error("give --vlm and --concepts (or --truth)")
        vocabulary = _load_vocabulary(args.vlm)
        source = _load_source(args.concepts)
        free = ss.load_masks(args.masks) if args.masks else None
    result = segment_concepts(
        splats, vocabulary, source, free, embedder, words,
        cameras=cameras, view_count=args.views, max_views=args.max_views,
        overview_count=args.overview_views, renderer=ss.make_renderer(args.renderer),
        max_scale_m=args.max_scale_m, workers=args.workers, cpus=args.cpus,
        memory_bytes=None if args.memory_gb is None else args.memory_gb * float(1 << 30),
        ground=ground, progress=say, free_factory=free_factory, cache=args.cache,
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
    if args.cache:
        (args.cache / "cameras.json").write_text(
            json.dumps([v.camera.to_json() for v in result.views]), encoding="utf-8"
        )
    if args.check:
        render_check(result, splats, args.check)
    usage = {"gaussians": len(splats), **ss.peak_usage(started)}
    say(ss.usage_line(usage))
    top = [r for r in document["instances"] if r["parent"] is None]
    summary = {
        "variant": document["variant"]["name"],
        "instances": len(result.instances),
        "topLevelThings": sum(1 for r in top if r["kind"] == "thing"),
        "named": sum(1 for r in top if r["kind"] == "thing" and r.get("nameSource") == "vlm"),
        "groundClasses": sum(1 for r in top if r["kind"] == "ground"),
        "assignedShare": round(float((result.splat_id > 0).mean()), 4),
        "concepts": document["concepts"],
        **result.lift.stats,
        "ground": document["ground"],
        "timingsS": {k: round(v, 2) for k, v in result.timings.items()},
        "usage": usage,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
