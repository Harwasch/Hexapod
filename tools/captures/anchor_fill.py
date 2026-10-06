"""Inferred fill, round 2: anchor, then propagate.

The owner's approved plan (the generative-fill plan, 2026-10-06): fill what a scan missed at
the poses that need it, with real photos as context, a few views first (anchors) and every
later view continuing them. Round 1 (`generative_fill`) walked 49-frame camera paths through
small video models and lifted the frames at unanchored monocular depth; it scored about
10 dB on the spool's held-out top. Here, with nothing picked by hand:

1. **Where** (`fill_quality`): each gaussian's best supervision quality over the real
   cameras (viewing angle x footprint x sharpness): known (frozen), weak (refined at partial
   strength) or unknown (generated); holes are connected groups of the latter two.
2. **Views** (`fill_views`): greedy weighted max coverage over rings of candidate poses:
   4-8 anchors, then 12-24 propagation views in a smooth order.
3. **Context** (`fill_views.retrieve_context`): the cameras that saw each hole's border
   vote; two of the top ones, a wide shot and a close-up rotate through the seeds.
4. **Anchors** (`Editor`, Qwen-Image-Edit-2511 with its Lightning LoRA): the masked render
   at the pose (+ two photos, or none in the no-photo arm), best of N seeds by the LPIPS on
   a ring of known pixels around the hole and agreement with the other anchors, registered
   to the render on the known pixels and composited, so known pixels never change.
5. **Lift** (`complete_depth`): depth from Prompt Depth Anything given the splat's own
   rendered depth as its prompt, fitted (robust scale and shift) to the rendered depth of
   the known pixels and its remaining residual spread over the hole harmonically, so the
   fill meets the scan at its edge; gaussians only on unknown pixels; carving.
6. **Propagate**: 6A (`propagate_sequential`) re-renders each propagation view with what is
   filled so far and fills only what is still unknown, lifting each before the next; 6B
   (`propagate_joint`) gives Wan2.1-VACE-14B every target view as a frame of one clip, the
   real photos and anchor fills unmasked (the ObjFiller-3D recipe).
7. **Fuse** (`fuse`): carve, one gaussian per voxel, a gsplat distil with the measured scan
   frozen (generated pixels weighted by seed agreement, real photos on what they saw), then
   dataset-update rounds: the fused layer rendered at every view, refined by the editor at
   decreasing strength, distilled again.
8. **Leave-out** (`score_held_out`): with the highest (spool) or lowest (pumpkin) cameras
   held out, their look withheld from the scan, the pipeline runs and the held-out photos
   score it inside the region only they saw well (masked PSNR, LPIPS, DreamSim).

Three arms share the setup and the views: `refs` (anchors with photos, then 6A), `norefs`
(the masked render alone, then 6A) and `vace` (the `refs` anchors, then 6B). Each is packaged
as an inferred layer (`anchor-refs`, `anchor-norefs`, `anchor-vace`).

Everything runs on the CPU with stand-ins (`StandInEditor`, `StandInSetFiller`, an oracle
depth, the CPU renderer) for tests; `infra/modal/fill.py` (`anchor:` and `leaveout:` jobs)
runs it with gsplat on an L4 and the models on H100s.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import fill_quality as fq
import fill_surface as fs
import fill_views as fv
import generative_fill as gf
import teacher_fill as tf
from splat_render import Camera, Splats

# --- constants --------------------------------------------------------------------------------------

#: The strength weak pixels are refined at (0 keeps them, 1 makes them anew).
WEAK_STRENGTH = 0.45
#: The strength unknown pixels are made at, from a smooth fill of their surroundings
#: (`prefill`): held to it only for the first steps, so the editor keeps its colours and
#: draws the rest. Run 37516117746 gave the editor those pixels in flat magenta (strength
#: 1) and it drew pink planks on the spool's top, with the photos more than without.
UNKNOWN_STRENGTH = 0.9
#: The ring of known pixels around a hole that seeds are scored on, px at a 1248-px width.
RING_PX = 24
#: A seed whose registered output keeps the known pixels below this (dB, blurred) drifted.
#: (Run 37508234704: the editor kept them at 16-29 dB, the outputs with photos at 16-18;
#: 18 called nearly every one of those drifted, and the choice among them was blind.)
DRIFT_DB = 15.0
#: Seed choice: each share of the pixels to make left in the key colour costs this much.
RESIDUE_WEIGHT = 2.0
#: Seed choice: LPIPS on the ring plus this times (1 - agreement with the other anchors).
AGREE_WEIGHT = 0.5
#: Agreement halves every this much colour difference (0..1, RMS over channels).
AGREE_COLOUR = 0.12
#: Depth completion: anchor pixels needed, the residual past which the fit is not trusted,
#: and the grid factor the residual is spread on.
DEPTH_MIN_PX = 200
DEPTH_MAX_RESIDUAL = 0.2
HARMONIC_FACTOR = 4
#: A propagation view is filled only when at least this share of its pixels needs it.
PROP_MIN_SHARE = 0.01
#: Lifted gaussians farther than this share of the focus radius from every gaussian to fill
#: are dropped (an unknown pixel of the background lifted into a floater).
LIFT_REACH = 0.08
#: Context photos are cropped about the hole they show, padded by this share, at least
#: this share of the photo across.
CROP_PAD = 0.25
CROP_MIN = 0.45
#: The distil: a generated view's weight against a real photo's 1, its weight outside its
#: mask, and real photos used.
GENERATED_WEIGHT = 0.5
GENERATED_OUTSIDE = 0.3
REAL_VIEWS = 12
#: The layer's budget (`generative_fill`): 15 % of the measured gaussians, at least 20,000.
BUDGET_SHARE = gf.BUDGET_SHARE
MIN_BUDGET = gf.MIN_BUDGET
#: A refined copy of a weak gaussian is kept only when its colour moved this much.
COPY_MIN_CHANGE = 0.04

#: Weak surfaces first (`fill_surface`): the reprojected surface's pixels are cleaned at
#: this strength (the photos' own pixels moved to a new angle: only seams and stretch to
#: fix); the hole clusters it is built for (largest first); its share of a layer's budget.
REPROJECT_STRENGTH = 0.3
SURFACE_CLUSTERS = 6
SURFACE_BUDGET_SHARE = 0.5
#: Hemisphere views (`hemisphere_targets`) closer than this to a chosen view are not added.
HEMISPHERE_NMS_DEG = 10.0
#: A camera does not carve through a pixel where the weak surfaces (alone) cover this much:
#: a ray through an under-constrained, semi-transparent surface is no proof of free space.
SOFT_ALPHA = 0.05

ARMS = ("refs", "norefs", "vace")
LAYERS = {"refs": "anchor-refs", "norefs": "anchor-norefs", "vace": "anchor-vace"}
ANCHOR_ARM = {"refs": "refs", "norefs": "norefs", "vace": "refs"}

#: The editor's prompt by placeholder (`Options.placeholder`: how the pixels to make are
#: shown to it).
EDIT_PROMPTS = {
    "smooth": (
        "Picture 1 is a view of {caption} rendered from a new camera position. Where the "
        "scene was never photographed, Picture 1 is only smeared in from around it: repaint "
        "those blurred parts as a sharp real photograph would show them, continuing the "
        "surfaces, materials, colours and lighting around them.{refs} Keep the camera, the "
        "framing and the sharp parts of Picture 1 exactly as they are."
    ),
    "flat": (
        "Picture 1 is a view of {caption} rendered from a new camera position. Where the "
        "scene was never photographed, Picture 1 is painted in flat patches of colour: paint "
        "those flat patches as a sharp real photograph would show the scene there, "
        "continuing the surfaces, materials, colours and lighting around them.{refs} Keep "
        "the camera, the framing and every other part of Picture 1 exactly as it is."
    ),
}
EDIT_PROMPT = EDIT_PROMPTS["smooth"]
PLACEHOLDERS = tuple(EDIT_PROMPTS)
REFS_SENTENCE = {
    1: (
        " Picture 2 is a close-up real photograph of the same scene taken from another "
        "position: match its materials, colours, wear and light, but do not copy its framing "
        "or paste anything from it into Picture 1."
    ),
    2: (
        " Pictures 2 and 3 are close-up real photographs of the same scene taken from other "
        "positions: match their materials, colours, wear and light, but do not copy their "
        "framing or paste anything from them into Picture 1."
    ),
}
UPDATE_PROMPT = (
    "Picture 1 is a view of {caption}. Make it a sharp, real photograph of the same scene, "
    "keeping the camera, the framing, every shape and the layout exactly as they are.{refs}"
)
SET_PROMPT = (
    "{caption}. The camera moves slowly around it; real footage in natural daylight, sharp "
    "detail; nothing in the scene moves."
)
RULE = (
    "weak or unknown gaussians (best quality over the real cameras: viewing angle x footprint "
    "x sharpness) filled at greedily chosen poses: anchors by an image editor given the masked "
    "render{refs}, then {propagate}; lifted at depth completed from the scan's own rendered "
    "depth, carved by real sight lines, distilled with the measured scan frozen"
)


# --- setup -----------------------------------------------------------------------------------------------


@dataclass
class Options:
    arms: tuple[str, ...] = ARMS
    fill_size: tuple[int, int] = (1024, 592)
    #: Each image the editor is given is encoded at about this many pixels.
    vae_area: int = 640 * 640
    set_size: tuple[int, int] = (832, 480)
    probe_size: tuple[int, int] = (156, 90)
    quality_width: int = 320
    elevations: tuple[float, ...] = fv.ELEVATIONS
    azimuths: int = fv.AZIMUTHS
    anchors: tuple[int, int] = fv.ANCHORS
    propagation: tuple[int, int] = fv.PROPAGATION
    seeds: int = 4
    prop_seeds: int = 2
    anchor_steps: int = 8
    prop_steps: int = 4
    update_steps: int = 4
    lightning: bool = True
    weak_strength: float = WEAK_STRENGTH
    #: How the editor is shown the pixels to make (`PLACEHOLDERS`): `smooth`, the smooth
    #: fill it is held to (`prefill`); `flat`, each hole one flat colour from around it
    #: (`flat_fill`), still held to the smooth fill.
    placeholder: str = "smooth"
    unknown_strength: float = UNKNOWN_STRENGTH
    #: Weak surfaces first, in the arms with photos (`build_surface`), and the strength the
    #: reprojected pixels are cleaned at.
    reproject: bool = True
    reproject_strength: float = REPROJECT_STRENGTH
    update_strengths: tuple[float, ...] = (0.4,)
    set_seeds: int = 2
    set_steps: int = 25
    set_distill: bool = False
    distill: int = 1500
    update_distill: int = 600
    distill_width: int = 640
    lift_stride: int = 4
    #: Refuse more editor calls than this in one run (the budget guard's own cap).
    max_edit_calls: int = 600
    max_set_calls: int = 4
    #: The exact-mask fallback (InstantX ControlNet inpainting) when every seed drifted.
    fallback: bool = False
    max_fallback_calls: int = 8
    #: The held-out photos are scored at this width.
    score_width: int = 480
    #: The real cameras carve at this width.
    carve_width: int = 320


@dataclass
class Setup:
    """What every arm shares: the scan, its cameras (kept and held out), the quality of each
    gaussian with the kept cameras, the leave-out's withheld and dropped gaussians, the
    focus and the hole clusters."""

    name: str
    caption: str
    measured: Splats
    views: list[gf.RealView]
    held_out: list[gf.RealView]
    quality: fq.Quality
    classes: np.ndarray
    withheld: np.ndarray
    dropped: np.ndarray
    focus: fq.Focus
    clusters: fq.Clusters
    leave_out: str = "none"
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def shown(self) -> np.ndarray:
        return np.flatnonzero(~self.dropped)

    def scene(self) -> Splats:
        """What the generator is shown: the measured scan without the dropped gaussians, the
        withheld ones mid grey (their shape stays, their look is the answer). Cached."""
        if "scene" not in self._cache:
            s = self.measured.take(self.shown)
            w = self.withheld[self.shown]
            if w.any():
                colours = s.colours.copy()
                colours[w] = 0.5
                s = Splats(s.positions, s.rotations, s.scales, colours, s.opacities)
            self._cache["scene"] = s
        return self._cache["scene"]

    def shown_classes(self) -> np.ndarray:
        return self.classes[self.shown]

    def shown_quality(self) -> fq.Quality:
        """The kept cameras' quality of the shown gaussians. Cached."""
        if "quality" not in self._cache:
            self._cache["quality"] = _quality_on(self.quality, self.shown)
        return self._cache["quality"]

    def frozen_index(self) -> np.ndarray:
        """The shown gaussians the distil keeps frozen beside the layer: all of them, but in
        a leave-out only the known and weak not withheld (what the scoring shows)."""
        if self.leave_out == "none":
            return np.arange(len(self.shown))
        c = self.shown_classes()
        return np.flatnonzero((c != fq.UNKNOWN) & ~self.withheld[self.shown])

    def photo(self, k: int, width: int = 1024) -> np.ndarray | None:
        key = f"photo{k}-{width}"
        if key not in self._cache:
            self._cache[key] = self.views[k].photo(width=min(width, self.views[k].camera.width))
        return self._cache[key]

    def context_photo(
        self, k: int, members: np.ndarray | None, tag: str = "", width: int = 2048
    ) -> np.ndarray | None:
        """Camera `k`'s photo cropped about the hole gaussians `members` (measured indices:
        what a view fills, `view_hole_members`) as it sees them: the middle 90 % of them in
        its frame, padded (`CROP_PAD`), at least `CROP_MIN` of the photo across, at the
        photo's aspect. A whole photo of the object beside the render made the editor paste
        it in, ghosted (run 37508234704); a close-up of the surface is a reference for its
        look. The whole photo when they are not in it. Cached by `tag`."""
        key = f"context{k}-{tag}-{width}"
        if key in self._cache:
            return self._cache[key]
        photo = self.photo(k, width)
        out = photo
        members = np.zeros(0, np.int64) if members is None else np.asarray(members, np.int64)
        if photo is not None and members.size >= 10:
            h, w = photo.shape[:2]
            cam = fv.scaled(self.views[k].camera, w, h)
            uv, z = cam.project(self.measured.positions[members])
            ok = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if ok.sum() >= 10:
                lo = np.percentile(uv[ok], 5, axis=0)
                hi = np.percentile(uv[ok], 95, axis=0)
                span = (hi - lo) * (1 + 2 * CROP_PAD)
                cw = min(float(w), max(span[0], span[1] * w / h, CROP_MIN * w))
                ch = min(float(h), cw * h / w)
                cw = ch * w / h
                mid = (lo + hi) / 2
                x0 = round(float(np.clip(mid[0] - cw / 2, 0, w - cw)))
                y0 = round(float(np.clip(mid[1] - ch / 2, 0, h - ch)))
                out = photo[y0 : y0 + max(1, round(ch)), x0 : x0 + max(1, round(cw))]
        self._cache[key] = out
        return out

    def region(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.focus.radius
        return self.focus.centre - r, self.focus.centre + r


def split_leave_out(
    views: Sequence[gf.RealView], focus: fq.Focus, which: str, share: float
) -> tuple[list[int], list[int]]:
    """(kept, held out) camera indices: the `share` of the cameras highest (`high`) or
    lowest (`low`) in elevation above the focus; none for `none`."""
    if which == "none" or share <= 0:
        return list(range(len(views))), []
    elev = np.array([focus.elevation(v.camera.centre) for v in views])
    order = np.argsort(-elev if which == "high" else elev, kind="stable")
    n = max(1, round(share * len(views)))
    held = set(order[:n].tolist())
    return [k for k in range(len(views)) if k not in held], sorted(held)


def make_setup(
    name: str,
    caption: str,
    measured: Splats,
    views: Sequence[gf.RealView],
    renderer: Any,
    *,
    leave_out: str = "none",
    share: float = 0.15,
    width: int = 320,
    log: Callable[[str], None] = print,
) -> tuple[Setup, dict[str, Any]]:
    """Quality with every camera, the leave-out split (and the withheld look), the classes
    with the kept cameras, and the hole clusters."""
    views = list(views)
    focus = fq.capture_focus([v.camera for v in views])
    photos = [v.photo(width=256) for v in views]
    sharp = fq.photo_sharpness(photos)
    t = time.time()
    everyone = fq.measure_quality(
        measured, [v.camera for v in views], renderer, focus, width=width, sharpness=sharp
    )
    log(
        f"{name}: quality of {len(measured)} gaussians over {len(views)} cameras in {time.time() - t:.0f}s"
    )
    kept_idx, held_idx = split_leave_out(views, focus, leave_out, share)
    kept = everyone.subset(kept_idx) if held_idx else everyone
    withheld = np.zeros(len(measured), bool)
    dropped = np.zeros(len(measured), bool)
    if held_idx:
        withheld, dropped = fq.withheld_by_holdout(kept, everyone, everyone.subset(held_idx))
    classes = kept.classes.copy()
    classes[withheld] = fq.UNKNOWN
    shown = ~dropped
    clusters_shown = fq.hole_clusters(
        measured.take(np.flatnonzero(shown)), classes[shown], focus, best=kept.best[shown]
    )
    labels = np.full(len(measured), -1, np.int64)
    labels[shown] = clusters_shown.labels
    clusters = fq.Clusters(labels, clusters_shown.info)
    setup = Setup(
        name,
        caption,
        measured,
        [views[k] for k in kept_idx],
        [views[k] for k in held_idx],
        kept,
        classes,
        withheld,
        dropped,
        focus,
        clusters,
        leave_out,
    )
    inside = focus.inside(measured.positions)
    info = {
        "focus": focus.to_json(),
        "cameras": len(views),
        "kept": len(kept_idx),
        "heldOut": [views[k].name for k in held_idx],
        "heldOutElevations": [round(focus.elevation(views[k].camera.centre), 1) for k in held_idx],
        "sharpness": {
            "min": round(float(sharp.min()), 3),
            "median": round(float(np.median(sharp)), 3),
        },
        "quality": {
            "all": everyone.summary(),
            "kept": kept.summary(),
            "keptInFocus": kept.summary(inside),
        },
        "withheld": int(withheld.sum()),
        "dropped": int(dropped.sum()),
        "clusters": clusters.info[:12],
    }
    return setup, info


# --- targets ---------------------------------------------------------------------------------------------


@dataclass
class Target:
    key: str
    role: str  # anchor | prop
    candidate: int
    elevation: float
    azimuth: float
    direction: np.ndarray
    camera: Camera
    cluster: int
    context: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "elevation": self.elevation,
            "azimuth": round(self.azimuth, 2),
            "cluster": self.cluster,
            "context": {k: v for k, v in self.context.items() if k != "votes"},
        }


def plan_targets(
    setup: Setup, renderer: Any, options: Options
) -> tuple[list[Target], dict[str, Any]]:
    """The anchors and propagation views (`fill_views.choose_views`), the propagation in
    order outwards from the anchors, and each view's context photos (its dominant hole's
    border votes)."""
    scene = setup.scene()
    shown = setup.shown
    q = setup.quality
    sub = setup.shown_quality()
    clusters = fq.Clusters(setup.clusters.labels[shown], setup.clusters.info)
    candidates, selection, cameras = fv.choose_views(
        scene,
        setup.shown_classes(),
        q.normals[shown],
        q.best[shown],
        clusters,
        setup.focus,
        renderer,
        fill_size=options.fill_size,
        probe_size=options.probe_size,
        elevations=options.elevations,
        azimuths=options.azimuths,
        anchors=options.anchors,
        propagation=options.propagation,
        quality=sub,
    )
    dirs = np.array([c.direction for c in candidates])
    order = fv.propagation_order(dirs, selection.anchors, selection.propagation)
    known_rows = np.flatnonzero(setup.shown_classes() == fq.KNOWN)
    spacing = fq.median_spacing(scene.positions) * fq.CLUSTER_SPACING * 2
    contexts: dict[int, dict[str, Any]] = {}
    centres = np.array([v.camera.centre for v in setup.views])
    for k, info in enumerate(clusters.info):
        members = clusters.members(k)
        border = fv.border_gaussians(scene.positions, members, known_rows, spacing)
        voters = border if border.size else members
        # Votes in the kept cameras' quality of the shown gaussians.
        ctx = fv.retrieve_context(sub.per_camera, voters, centres, np.asarray(info["centre"]))
        ctx["border"] = int(border.size)
        ctx["names"] = {r: setup.views[c].name for r, c in fv.context_roles(ctx).items()}
        contexts[k] = ctx
    targets = []
    for role, picks in (("anchor", selection.anchors), ("prop", order)):
        for j, c in enumerate(picks):
            cand = candidates[c]
            cluster = max(cand.clusters, key=cand.clusters.get) if cand.clusters else 0
            targets.append(
                Target(
                    f"{'a' if role == 'anchor' else 'p'}{j}",
                    role,
                    c,
                    cand.elevation,
                    cand.azimuth,
                    cand.direction,
                    cameras[c],
                    cluster,
                    contexts.get(cluster, {"top": [], "wide": None, "close": None}),
                )
            )
    hemisphere = hemisphere_targets(setup, renderer, options, targets, contexts) if targets else []
    targets += hemisphere
    report = {
        "selection": selection.report,
        "candidates": [c.to_json() for c in candidates],
        "targets": [t.to_json() for t in targets],
        "hemisphere": [t.key for t in hemisphere],
        "contexts": {str(k): {kk: vv for kk, vv in v.items()} for k, v in contexts.items()},
    }
    return targets, report


def hemisphere_targets(
    setup: Setup,
    renderer: Any,
    options: Options,
    targets: Sequence[Target],
    contexts: dict[int, dict[str, Any]],
) -> list[Target]:
    """Propagation views that make every direction of the largest hole's hemisphere seen
    (`HEADLINE_VIEWS`: straight above, eight azimuths at 15 and at 45 degrees), aimed at the
    hole: each one no chosen view is within `HEMISPHERE_NMS_DEG` of, that shows the hole,
    from an eye in the open (`fill_views.BACK_MAX`), with something to fill or clean. The
    pass/fail test is see-through from all of them, so each must be filled, not left to
    whichever views covered the hole once."""
    if not setup.clusters.info:
        return []
    scene = setup.scene()
    c = setup.shown_classes()
    known, weak = c == fq.KNOWN, c == fq.WEAK
    quality = setup.shown_quality()
    members = setup.measured.positions[setup.clusters.members(0)]
    centre = np.median(members, axis=0)
    chosen = [t.direction for t in targets]
    out: list[Target] = []
    for elev, az in HEADLINE_VIEWS:
        d = fv._direction(elev, az)
        if any(_angle_deg(d, o) < HEMISPHERE_NMS_DEG for o in chosen):
            continue
        cam = _headline_camera(
            centre, setup.focus.distance, setup.focus.hfov_deg, elev, az, options.fill_size
        )
        probe = fv.scaled(cam, options.probe_size[0], options.probe_size[1])
        face = fq.facing(quality, probe.centre, scene.positions)
        pc = fv.pixel_classes(scene, probe, renderer, known, weak, face)
        shares = pc.shares()
        if shares["shown"] < fv.SHOWN_MIN or shares["back"] > fv.BACK_MAX:
            continue
        if shares["deficit"] < PROP_MIN_SHARE:
            continue
        chosen.append(d)
        out.append(
            Target(
                f"h{len(out)}",
                "prop",
                -1,
                elev,
                az,
                d,
                cam,
                0,
                contexts.get(0, {"top": [], "wide": None, "close": None}),
            )
        )
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _quality_on(q: fq.Quality, rows: np.ndarray) -> fq.Quality:
    """`q` restricted to some gaussians (the shown ones)."""
    return fq.Quality(
        q.best[rows],
        q.good[rows],
        q.seen[rows],
        q.classes[rows],
        q.per_camera[:, rows],
        q.normals[rows],
        q.sharpness,
        q.centres,
        q.facing[rows],
        q.observed[rows],
        q.positions[rows],
        q.side_normals[rows],
    )


# --- one view's masks ----------------------------------------------------------------------------------


@dataclass
class ViewMasks:
    """A view's pixels with the scan (and any generated layer, as known): the classes, the
    scan's colour (`render`, uint8), what the editor is shown and held to (`condition`: the
    pixels to make smoothly filled from around them, `prefill`; the open background black),
    the strength map and the rendered depth of the known and weak pixels (NaN elsewhere:
    what depth completion is anchored to)."""

    camera: Camera
    known: np.ndarray
    weak: np.ndarray
    unknown: np.ndarray
    void: np.ndarray
    render: np.ndarray
    condition: np.ndarray
    strength: np.ndarray
    depth: np.ndarray
    #: What the editor's tokens are held to, when not `condition` (the smooth fill).
    hold: np.ndarray | None = None
    #: The pixels the reprojected surface draws (among `weak`: cleaned at low strength).
    surface: np.ndarray | None = None

    @property
    def edit(self) -> np.ndarray:
        return self.unknown | self.weak

    def share(self) -> float:
        return float(self.edit.sum() / max(int((~self.void).sum()), 1))


def prefill(image: np.ndarray, target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """`image` with its `target` pixels filled smoothly from its `source` pixels (Telea at a
    quarter of the size, so a large hole gets its surroundings' colours rather than streaks,
    then softened): what the editor is given where it has to make the scene."""
    import cv2

    if not target.any():
        return image.copy()
    h, w = target.shape
    sw, sh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    known_small = cv2.resize(source.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    holes = (known_small < 0.999).astype(np.uint8) * 255
    filled = cv2.inpaint(small, holes, 5, cv2.INPAINT_TELEA) if source.any() else small
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), max(1.0, w / 512))
    return np.where(target[..., None], big, image).astype(np.uint8)


def flat_fill(image: np.ndarray, target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """`image` with each connected piece of `target` one flat colour: the median of the
    `source` pixels in a ring around it (of all of them when none is near)."""
    import cv2

    out = image.copy()
    if not target.any():
        return out
    fallback = np.median(image[source], axis=0) if source.any() else np.full(3, 127.0)
    count, labels = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)
    for k in range(1, count):
        piece = labels == k
        ring = _dilate(piece, max(4, target.shape[1] // 80)) & source
        colour = np.median(image[ring], axis=0) if ring.sum() >= 5 else fallback
        out[piece] = np.round(colour).astype(np.uint8)
    return out


def view_masks(
    setup: Setup,
    camera: Camera,
    renderer: Any,
    layer: Splats | None = None,
    weak_strength: float = WEAK_STRENGTH,
    placeholder: str = "smooth",
    unknown_strength: float = UNKNOWN_STRENGTH,
    surface: Splats | None = None,
    surface_strength: float = REPROJECT_STRENGTH,
) -> ViewMasks:
    """The view's pixel classes over the scan, any generated `layer` (as known) and the
    reprojected weak `surface` (`fill_surface`): where that surface is the front, its
    pixels -- the photos' own, moved to this angle -- are weak, cleaned at
    `surface_strength`, and anchor the depth like known ones."""
    scene = setup.scene()
    c = setup.shown_classes()
    known = c == fq.KNOWN
    weak = c == fq.WEAK
    face = fq.facing(setup.shown_quality(), camera.centre, scene.positions)
    for extra in (layer, surface):
        if extra is not None and len(extra):
            scene = Splats.concat([scene, extra])
            ones = np.ones(len(extra), bool)
            known = np.concatenate([known, ones])
            weak = np.concatenate([weak, ~ones])
            face = np.concatenate([face, ones])
    pc = fv.pixel_classes(scene, camera, renderer, known, weak, face)
    front = np.zeros_like(pc.known)
    if surface is not None and len(surface):
        drawn = renderer(surface, camera)
        d = np.where(np.isfinite(drawn.depth), drawn.depth, np.inf)
        front = (
            pc.known & fv._covered(drawn.alpha) & (d <= pc.depth * (1 + fv.FRONT_TOLERANCE) + 1e-6)
        )
    known_px = pc.known & ~front
    weak_px = pc.weak | front
    render = tf.to_u8(pc.colour)
    hold = prefill(render, pc.unknown, known_px | weak_px)
    hold[pc.void] = 0
    if placeholder == "flat":
        condition = flat_fill(render, pc.unknown, known_px | weak_px)
        condition[pc.void] = 0
    elif placeholder == "smooth":
        condition = hold
    else:
        raise ValueError(f"placeholder {placeholder!r}: one of {PLACEHOLDERS}")
    strength = np.select(
        [known_px, front, pc.weak, pc.unknown],
        [0.0, surface_strength, weak_strength, unknown_strength],
        1.0,
    ).astype(np.float32)
    depth = np.where((known_px | weak_px) & np.isfinite(pc.depth), pc.depth, np.nan)
    return ViewMasks(
        camera,
        known_px,
        weak_px,
        pc.unknown,
        pc.void,
        render,
        condition,
        strength,
        depth,
        None if condition is hold else hold,
        front if front.any() else None,
    )


# --- the editor ---------------------------------------------------------------------------------------------


@dataclass
class EditRequest:
    key: str
    condition: np.ndarray
    render: np.ndarray
    strength: np.ndarray
    references: list[np.ndarray]
    prompt: str
    seed: int
    steps: int
    lightning: bool = True
    hold: bool = True
    vae_area: int = 640 * 640

    def wire(self) -> dict[str, Any]:
        import anchor_models as am

        h, w = self.condition.shape[:2]
        return {
            "image": am.encode_png(self.condition),
            "render": am.encode_png(self.render),
            "strength": am.encode_strength(self.strength),
            "references": [am.encode_png(_fit_long(r, 1024)) for r in self.references],
            "prompt": self.prompt,
            "seed": int(self.seed),
            "steps": int(self.steps),
            "lightning": bool(self.lightning),
            "hold": bool(self.hold),
            "size": [w, h],
            "vae_area": int(self.vae_area),
        }


@dataclass
class EditResult:
    key: str
    image: np.ndarray | None
    info: dict[str, Any] = field(default_factory=dict)


class Editor(Protocol):
    name: str

    def start(self, requests: Sequence[EditRequest]) -> Callable[[], list[EditResult]]: ...


def _fit_long(image: np.ndarray, long: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    if max(h, w) <= long:
        return image
    f = long / max(h, w)
    return cv2.resize(image, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


@dataclass
class StandInEditor:
    """The CPU stand-in: Telea inpainting of the pixels to make, the weak ones blended with
    it by their strength, a per-seed tint, and the photos' mean colour pulled in when given
    (so the arms differ in tests). Known pixels come back as rendered."""

    name: str = "standin-telea"
    calls: int = 0

    def start(self, requests: Sequence[EditRequest]) -> Callable[[], list[EditResult]]:
        def run() -> list[EditResult]:
            import cv2

            out = []
            for r in requests:
                self.calls += 1
                make = r.strength >= UNKNOWN_STRENGTH - 1e-3
                base = cv2.inpaint(
                    r.render, make.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA
                ).astype(np.float64)
                rng = np.random.default_rng(r.seed)
                base += rng.normal(0, 6.0, 3)
                if r.references:
                    mean = np.mean(
                        [ref.reshape(-1, 3).mean(axis=0) for ref in r.references], axis=0
                    )
                    base = 0.7 * base + 0.3 * mean
                s = np.clip(r.strength, 0, 1)[..., None]
                img = (1 - s) * r.render + s * base
                out.append(
                    EditResult(r.key, np.clip(img, 0, 255).astype(np.uint8), {"seconds": 0.0})
                )
            return out

        return run


Submit = Callable[[str, str, dict], Any]
Wait = Callable[[Any], dict]


@dataclass
class RemoteEditor:
    """Qwen-Image-Edit-2511 on a Modal H100 (`infra/modal/fill.py` `EditQwen.edit`)."""

    submit: Submit
    wait: Wait
    name: str = "qwen-image-edit-2511"
    cls: str = "EditQwen"
    calls: int = 0
    limit: int = 600

    def start(self, requests: Sequence[EditRequest]) -> Callable[[], list[EditResult]]:
        import anchor_models as am

        if self.calls + len(requests) > self.limit:
            raise RuntimeError(
                f"{self.calls + len(requests)} editor calls: past the cap {self.limit}"
            )
        self.calls += len(requests)
        handles = [(r.key, self.submit(self.cls, "edit", r.wire())) for r in requests]

        def collect() -> list[EditResult]:
            out = []
            for key, h in handles:
                try:
                    response = self.wait(h)
                except Exception as error:  # noqa: BLE001 - recorded; the others go on
                    out.append(EditResult(key, None, {"error": repr(error)[:1500]}))
                    continue
                if "error" in response:
                    out.append(EditResult(key, None, dict(response)))
                    continue
                info = {k: v for k, v in response.items() if k != "image"}
                out.append(EditResult(key, am.decode_png(response["image"]), info))
            return out

        return collect


@dataclass
class RemoteFallback:
    """The exact-mask fallback: Qwen-Image + InstantX ControlNet inpainting (`InpaintQwen`,
    `inpaint_models`), for a view whose every seed drifted."""

    submit: Submit
    wait: Wait
    name: str = "qwen-image+instantx-inpaint"
    calls: int = 0
    limit: int = 8

    def start(self, requests: Sequence[EditRequest]) -> Callable[[], list[EditResult]]:
        import anchor_models as am

        if self.calls + len(requests) > self.limit:
            return lambda: [EditResult(r.key, None, {"error": "fallback cap"}) for r in requests]
        self.calls += len(requests)
        handles = []
        for r in requests:
            body = {
                "image": am.encode_png(r.render),
                "mask": am.encode_png(
                    ((r.strength >= UNKNOWN_STRENGTH - 1e-3) * 255).astype(np.uint8)
                ),
                "prompt": r.prompt,
                "seed": int(r.seed),
            }
            handles.append((r.key, self.submit("InpaintQwen", "inpaint", body)))

        def collect() -> list[EditResult]:
            out = []
            for key, h in handles:
                try:
                    response = self.wait(h)
                    out.append(
                        EditResult(key, am.decode_png(response["image"]), {"fallback": True})
                    )
                except Exception as error:  # noqa: BLE001
                    out.append(EditResult(key, None, {"error": repr(error)[:1500]}))
            return out

        return collect


# --- registration and compositing ---------------------------------------------------------------------------


def _grey(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def register(
    out: np.ndarray, render: np.ndarray, known: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """`out` warped (affine, ECC on the known pixels) onto `render` when that is small and
    makes the known pixels agree better; colour matched (per channel, gain and offset) on
    the known pixels near the edit. Returns the aligned image and what was done."""
    import cv2

    h, w = render.shape[:2]
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    info: dict[str, Any] = {"warp": None}
    k = known.astype(np.uint8)
    aligned = out
    if known.sum() > 500:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            _, warp = cv2.findTransformECC(
                cv2.GaussianBlur(_grey(render), (0, 0), 1.5),
                cv2.GaussianBlur(_grey(out), (0, 0), 1.5),
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                k,
                5,
            )
            shift = float(np.abs(warp[:, 2]).max())
            scale = float(np.abs(np.linalg.svd(warp[:, :2], compute_uv=False) - 1).max())
            if shift <= 0.05 * w and scale <= 0.05:
                moved = cv2.warpAffine(
                    out,
                    warp,
                    (w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT,
                )
                if _mse(moved, render, known) < _mse(out, render, known):
                    aligned = moved
                    info["warp"] = np.round(warp, 5).tolist()
        except cv2.error as error:
            info["eccError"] = str(error).splitlines()[0][:200]
    # Colour: per channel gain and offset on the known pixels (near the edit when it has some).
    ring = known & ~_dilate(~known, max(4, round(RING_PX * w / 1248)) * 3)
    near = known & ~ring
    use = near if near.sum() > 200 else known
    gains = []
    result = aligned.astype(np.float64)
    if use.sum() > 50:
        for ch in range(3):
            x = aligned[..., ch][use].astype(np.float64)
            y = render[..., ch][use].astype(np.float64)
            a, b = np.polyfit(x, y, 1) if x.std() > 1e-3 else (1.0, float(y.mean() - x.mean()))
            a = float(np.clip(a, 0.7, 1.3))
            b = float(np.clip(b, -40, 40))
            result[..., ch] = a * result[..., ch] + b
            gains.append([round(a, 4), round(b, 2)])
    info["colour"] = gains
    return np.clip(result, 0, 255).astype(np.uint8), info


def _mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    d = a.astype(np.float64)[mask] - b.astype(np.float64)[mask]
    return float(np.mean(d**2))


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _close(mask: np.ndarray, px: int) -> np.ndarray:
    """Morphological closing: a region's pinholes and slivers between its specks filled."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def composite(aligned: np.ndarray, render: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The known pixels exactly as rendered, every other pixel as the editor drew it."""
    return np.where(known[..., None], render, aligned).astype(np.uint8)


def ring_mask(masks: ViewMasks) -> np.ndarray:
    """The known pixels within `RING_PX` (scaled to the width) of the edit."""
    px = max(3, round(RING_PX * masks.camera.width / 1248))
    return masks.known & _dilate(masks.edit, px)


def known_psnr(aligned: np.ndarray, masks: ViewMasks) -> float | None:
    """How well the editor kept the known pixels (blurred PSNR): its drift."""
    if masks.known.sum() < 50:
        return None
    return min(tf.psnr(tf._blur(aligned), tf._blur(masks.render), masks.known), tf.GATE_CAP_DB)


class Perceptual(Protocol):
    def lpips(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None: ...

    def dreamsim(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None: ...


@dataclass
class StandInPerceptual:
    """The CPU stand-in for LPIPS: the mean absolute difference of the blurred images (0..1);
    no DreamSim."""

    def lpips(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        d = np.abs(tf._blur(a).astype(np.float64) - tf._blur(b).astype(np.float64)).mean(axis=-1)
        return float(d[mask].mean() / 255.0)

    def dreamsim(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        return None


def key_residue(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The pixels of `region` the editor left in (or near) the key colour: magenta (hue
    within 20 degrees), saturated and bright."""
    import cv2

    hsv = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)  # 0..179; magenta is 150
    return region & (np.abs(hue - 150) <= 10) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 120)


def clean_residue(
    image: np.ndarray, weight: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Key-colour residue in `region` (and 2 px round it) painted over from its
    surroundings, its weight 0: (image, weight, residue mask). Run 37508234704 lifted the
    editor's leftover magenta specks into the layer."""
    import cv2

    residue = key_residue(image, region)
    if not residue.any():
        return image, weight, residue
    residue = _dilate(residue, 2) & region
    painted = cv2.inpaint(
        np.asarray(image, np.uint8), residue.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA
    )
    return painted, np.where(residue, 0.0, weight), residue


def seed_agreement(chosen: np.ndarray, others: Sequence[np.ndarray]) -> np.ndarray:
    """Per pixel, how far the chosen seed agrees with the other seeds (colour; the mean of
    `exp(-rms / AGREE_COLOUR * ln 2)` over them); 1 with no others."""
    if not others:
        return np.ones(chosen.shape[:2])
    a = tf._blur(chosen).astype(np.float64) / 255.0
    acc = np.zeros(chosen.shape[:2])
    for o in others:
        b = tf._blur(o).astype(np.float64) / 255.0
        rms = np.sqrt(np.mean((a - b) ** 2, axis=-1))
        acc += np.exp(-rms / AGREE_COLOUR * math.log(2))
    return acc / len(others)


# --- filled views ------------------------------------------------------------------------------------------------


@dataclass
class Candidate:
    """One seed's output for a view, registered: its scores."""

    seed: int
    aligned: np.ndarray
    ring: float
    drift: float | None
    context: list[str]
    info: dict[str, Any]
    agreement: float = 1.0
    #: The share of the pixels to make left in the key colour.
    residue: float = 0.0

    @property
    def drifted(self) -> bool:
        return self.drift is not None and self.drift < DRIFT_DB

    def score(self) -> float:
        """Lower is better: the ring's LPIPS, disagreement with the other anchors, key-colour
        residue, and a drifted seed last."""
        return (
            self.ring
            + AGREE_WEIGHT * (1.0 - self.agreement)
            + RESIDUE_WEIGHT * self.residue
            + (10.0 if self.drifted else 0.0)
        )


@dataclass
class FilledView:
    key: str
    role: str
    masks: ViewMasks
    image: np.ndarray
    weight: np.ndarray
    seeds: list[dict[str, Any]]
    chosen: int
    context: list[str]
    depth_info: dict[str, Any] = field(default_factory=dict)
    lifted: int = 0
    fallback: bool = False
    #: Pixels the editor left in the key colour (painted over, weight 0, never lifted).
    residue: np.ndarray | None = None

    @property
    def camera(self) -> Camera:
        return self.masks.camera

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "share": round(self.masks.share(), 4),
            "unknownPx": int(self.masks.unknown.sum()),
            "weakPx": int(self.masks.weak.sum()),
            "chosenSeed": self.chosen,
            "context": self.context,
            "seeds": self.seeds,
            "seedAgreement": round(float(self.weight[self.masks.edit].mean()), 4)
            if self.masks.edit.any()
            else None,
            "depth": self.depth_info,
            "lifted": self.lifted,
            "fallback": self.fallback,
            "residuePx": 0 if self.residue is None else int(self.residue.sum()),
        }


def prompt_for(
    caption: str, references: int, update: bool = False, placeholder: str = "smooth"
) -> str:
    refs = REFS_SENTENCE[min(references, 2)] if references else ""
    template = UPDATE_PROMPT if update else EDIT_PROMPTS[placeholder]
    return template.format(caption=caption, refs=refs)


def edit_requests(
    setup: Setup,
    key: str,
    masks: ViewMasks,
    context: dict[str, Any],
    *,
    refs: bool,
    seeds: Sequence[int],
    steps: int,
    lightning: bool,
    update: bool = False,
    vae_area: int = 640 * 640,
    members: np.ndarray | None = None,
    tag: str = "",
    placeholder: str = "smooth",
) -> list[tuple[EditRequest, list[str]]]:
    """One request per seed: the condition, the render and strength, and (with `refs`) the
    seed's pair of context photos, cropped about the hole gaussians the view fills
    (`members`, measured indices; the whole photos without; crops cached by `tag`)."""
    out = []
    for j, seed in enumerate(seeds):
        names: list[str] = []
        photos: list[np.ndarray] = []
        if refs:
            for role, c in fv.context_for_seed(context, j):
                photo = setup.context_photo(c, members, tag)
                if photo is not None:
                    photos.append(photo)
                    names.append(f"{role}:{setup.views[c].name}")
        # Held to the condition: known pixels as rendered, the ones to make as prefilled.
        request = EditRequest(
            f"{key}-s{seed}",
            masks.condition,
            masks.condition if masks.hold is None else masks.hold,
            masks.strength,
            photos,
            prompt_for(setup.caption, len(photos), update, placeholder),
            int(seed),
            steps,
            lightning,
            vae_area=vae_area,
        )
        out.append((request, names))
    return out


def score_candidates(
    masks: ViewMasks,
    results: Sequence[EditResult],
    names: Sequence[list[str]],
    perceptual: Perceptual,
) -> list[Candidate]:
    ring = ring_mask(masks)
    make = masks.unknown
    out = []
    for r, n in zip(results, names, strict=True):
        if r.image is None:
            continue
        aligned, info = register(r.image, masks.render, masks.known)
        lp = perceptual.lpips(aligned, masks.render, ring) if ring.any() else None
        seed = int(r.key.rsplit("-s", 1)[-1])
        residue = float(key_residue(aligned, make).sum() / max(int(make.sum()), 1))
        out.append(
            Candidate(
                seed,
                aligned,
                1.0 if lp is None else lp,
                known_psnr(aligned, masks),
                n,
                {**r.info, **info},
                residue=residue,
            )
        )
    return out


def choose(masks: ViewMasks, cands: Sequence[Candidate], key: str, role: str) -> FilledView | None:
    """The best candidate (`Candidate.score`), composited, key-colour residue painted over;
    its agreement with every other seed as per-pixel weight (a drifted seed counts too:
    with only drifted ones to compare, run 37508234704 weighted every pixel 1)."""
    if not cands:
        return None
    score = [c.score() for c in cands]
    best = int(np.argmin(score))
    chosen = cands[best]
    image = composite(chosen.aligned, masks.render, masks.known)
    others = [c.aligned for k, c in enumerate(cands) if k != best]
    weight = seed_agreement(chosen.aligned, others)
    image, weight, residue = clean_residue(image, weight, masks.edit)
    seeds = [
        {
            "seed": c.seed,
            "ringLpips": round(c.ring, 4),
            "keptDb": None if c.drift is None else round(c.drift, 2),
            "agreement": round(c.agreement, 4),
            "residue": round(c.residue, 4),
            "score": round(s, 4),
            "context": c.context,
            "seconds": c.info.get("seconds"),
            "warp": c.info.get("warp") is not None,
        }
        for c, s in zip(cands, score, strict=True)
    ]
    return FilledView(
        key, role, masks, image, weight, seeds, chosen.seed, chosen.context, residue=residue
    )


# --- agreement between anchors ----------------------------------------------------------------------------------


def anchor_points(
    setup: Setup, masks: ViewMasks, sample: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Of `sample` (shown gaussian indices of the holes), those this view sees in its edit
    region (depth tested), and their pixels."""
    scene = setup.scene()
    cam = masks.camera
    uv, z = cam.project(scene.positions[sample])
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    rows = np.flatnonzero(ok)
    depth = fq.nearest_surface(np.where(np.isfinite(masks.depth), masks.depth, np.inf))
    surf = depth[v[rows], u[rows]]
    # Holes have no anchored depth; their own render depth: test against the class depth
    # where there is one, else accept (the point is in an unknown pixel).
    vis = ~np.isfinite(surf) | (z[rows] <= surf * (1 + fq.DEPTH_TOLERANCE) + 0.02 * surf)
    rows = rows[vis]
    rows = rows[masks.edit[v[rows], u[rows]]]
    return sample[rows], np.column_stack([u[rows], v[rows]])


def agreement_scores(
    anchors: dict[str, tuple[list[Candidate], tuple[np.ndarray, np.ndarray]]],
    chosen: dict[str, int],
) -> None:
    """Each candidate's agreement with the other anchors' current choices: the colour at the
    hole gaussians both see (blurred), `exp(-rms / AGREE_COLOUR ln 2)` averaged."""
    colours: dict[str, dict[int, np.ndarray]] = {}
    for key, (cands, (ids, px)) in anchors.items():
        if not cands or not ids.size:
            continue
        img = tf._blur(cands[chosen[key]].aligned).astype(np.float64) / 255.0
        colours[key] = dict(zip(ids.tolist(), img[px[:, 1], px[:, 0]], strict=True))
    for key, (cands, (ids, px)) in anchors.items():
        for cand in cands:
            if not ids.size:
                cand.agreement = 1.0
                continue
            mine = tf._blur(cand.aligned).astype(np.float64)[px[:, 1], px[:, 0]] / 255.0
            scores = []
            for other, table in colours.items():
                if other == key:
                    continue
                shared = [j for j, g in enumerate(ids.tolist()) if g in table]
                if len(shared) < 10:
                    continue
                theirs = np.array([table[ids[j]] for j in shared])
                rms = np.sqrt(np.mean((mine[shared] - theirs) ** 2, axis=-1))
                scores.append(float(np.mean(np.exp(-rms / AGREE_COLOUR * math.log(2)))))
            cand.agreement = float(np.mean(scores)) if scores else 1.0


# --- depth completion and lift ---------------------------------------------------------------------------------


class DepthModel(Protocol):
    name: str

    def depth(self, image: np.ndarray, prompt: np.ndarray) -> np.ndarray: ...


@dataclass
class OracleDepth:
    """Tests: the true scene's depth at the view (the camera comes with the call)."""

    truth: Splats
    renderer: Any
    name: str = "oracle"
    camera: Camera | None = None

    def depth(self, image: np.ndarray, prompt: np.ndarray) -> np.ndarray:
        assert self.camera is not None
        d = self.renderer(self.truth, self.camera).depth
        return np.where(np.isfinite(d), d, np.nanmax(np.where(np.isfinite(d), d, np.nan)) * 1.5)


def harmonic(
    residual: np.ndarray, known: np.ndarray, target: np.ndarray, factor: int = HARMONIC_FACTOR
) -> np.ndarray:
    """`residual` (on `known` pixels) spread over the `target` pixels as a harmonic function
    (Laplace's equation, the known cells as its boundary; cells no boundary reaches stay 0),
    solved on a grid `factor` times coarser and brought back smoothly."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    h, w = residual.shape
    hs, ws = -(-h // factor), -(-w // factor)
    pad = ((0, hs * factor - h), (0, ws * factor - w))
    r = np.pad(np.where(known, residual, 0.0), pad)
    k = np.pad(known.astype(np.float64), pad)
    t = np.pad(target.astype(bool), pad)
    blocks = lambda a: a.reshape(hs, factor, ws, factor)
    count = blocks(k).sum(axis=(1, 3))
    value = np.where(count > 0, blocks(r).sum(axis=(1, 3)) / np.maximum(count, 1), 0.0)
    fixed = count > 0
    solve = blocks(t).any(axis=(1, 3)) & ~fixed
    field_ = np.where(fixed, value, 0.0)
    idx = -np.ones((hs, ws), np.int64)
    cells = np.flatnonzero(solve.ravel())
    idx.ravel()[cells] = np.arange(cells.size)
    if cells.size:
        rows, cols, vals = [], [], []
        rhs = np.zeros(cells.size)
        ys, xs = np.divmod(cells, ws)
        diag = np.full(cells.size, 1e-4)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inside = (ny >= 0) & (ny < hs) & (nx >= 0) & (nx < ws)
            ny_c, nx_c = np.clip(ny, 0, hs - 1), np.clip(nx, 0, ws - 1)
            nb_fixed = inside & fixed[ny_c, nx_c]
            nb_free = inside & solve[ny_c, nx_c]
            diag += nb_fixed | nb_free
            rhs += np.where(nb_fixed, value[ny_c, nx_c], 0.0)
            j = idx[ny_c, nx_c]
            sel = np.flatnonzero(nb_free)
            rows.append(sel)
            cols.append(j[sel])
            vals.append(-np.ones(sel.size))
        rows.append(np.arange(cells.size))
        cols.append(np.arange(cells.size))
        vals.append(diag)
        a = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(cells.size, cells.size),
        ).tocsr()
        field_.ravel()[cells] = spsolve(a, rhs)
    up = cv2.resize(
        field_.astype(np.float32), (ws * factor, hs * factor), interpolation=cv2.INTER_LINEAR
    )
    return np.where(target, up[:h, :w], 0.0)


def complete_depth(
    pred: np.ndarray,
    anchor_depth: np.ndarray,
    target: np.ndarray,
    *,
    disparity: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Depth for the `target` pixels: the model's `pred` (metric depth, or relative inverse
    depth with `disparity`) fitted to the anchored pixels (finite `anchor_depth`) by a
    robust affine map weighted towards the target, then corrected by the fit's residual
    spread harmonically over the target, so the result meets the scan at the edge."""
    import cv2

    anchors = np.isfinite(anchor_depth) & (anchor_depth > 0) & np.isfinite(pred)
    info: dict[str, Any] = {"anchorPx": int(anchors.sum())}
    if anchors.sum() < DEPTH_MIN_PX or not target.any():
        return None, info
    gap = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-gap / (0.1 * max(target.shape)))[anchors]
    x = pred[anchors].astype(np.float64)
    y = anchor_depth[anchors].astype(np.float64)
    if disparity:
        y = 1.0 / y
    use = np.ones(x.size, bool)
    a = b = 0.0
    for _ in range(4):
        sw = np.sqrt(weight[use])
        design = np.stack([x[use], np.ones(use.sum())], 1) * sw[:, None]
        (a, b), *_ = np.linalg.lstsq(design, y[use] * sw, rcond=None)
        err = np.abs(a * x + b - y) / np.maximum(np.abs(y), 1e-9)
        use = err <= max(np.quantile(err[use], 0.8), 1e-9)
    fit = a * pred + b
    fitted = 1.0 / np.maximum(fit, 1e-9) if disparity else fit
    rel = np.abs(fitted[anchors] - anchor_depth[anchors]) / np.maximum(anchor_depth[anchors], 1e-9)
    residual_med = float(np.median(rel))
    info.update({"a": float(a), "b": float(b), "residual": round(residual_med, 4)})
    if a <= 0 or residual_med > DEPTH_MAX_RESIDUAL:
        info["refused"] = "fit"
        return None, info
    resid = np.where(anchors, anchor_depth - fitted, 0.0)
    near = anchors & _dilate(target, 6)
    corr = harmonic(resid, near, target)
    out = np.where(target, fitted + corr, np.nan)
    limit = 4.0 * float(np.nanmax(anchor_depth[anchors]))
    out = np.where((out > 1e-3) & (out <= limit), out, np.nan)
    info["correctionMedian"] = (
        round(float(np.nanmedian(np.abs(corr[target]))), 5) if target.any() else 0.0
    )
    return out, info


def depth_prompt(masks: ViewMasks, layer_depth: np.ndarray | None = None) -> np.ndarray:
    """What depth completion is anchored to: the rendered depth of the known and weak
    pixels (and of the generated layer, as known)."""
    d = masks.depth.copy()
    if layer_depth is not None:
        d = np.where(np.isfinite(d), d, layer_depth)
    return d


def run_depth(
    model: DepthModel, image: np.ndarray, prompt: np.ndarray, camera: Camera
) -> np.ndarray:
    if isinstance(model, OracleDepth):
        model.camera = camera
    return model.depth(image, prompt)


def lift_view(
    fill: FilledView,
    depth_model: DepthModel,
    region: tuple[np.ndarray, np.ndarray],
    stride: int,
) -> tuple[Splats, np.ndarray]:
    """The view's unknown pixels as gaussians at completed depth (`complete_depth`), each
    with a confidence: distance from known pixels x depth fit x seed agreement."""
    import cv2

    masks = fill.masks
    target = masks.unknown if fill.residue is None else masks.unknown & ~fill.residue
    empty = Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
    if not target.any():
        return empty, np.zeros(0)
    prompt = depth_prompt(masks)
    try:
        pred = run_depth(depth_model, fill.image, prompt, masks.camera)
    except Exception as error:  # noqa: BLE001 - a view without depth is not lifted
        fill.depth_info = {"error": repr(error)[:300]}
        return empty, np.zeros(0)
    depth, info = complete_depth(
        pred, prompt, target, disparity=getattr(depth_model, "disparity_out", False)
    )
    fill.depth_info = info
    if depth is None:
        return empty, np.zeros(0)
    known = ~(masks.unknown | masks.void)
    distance = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 5)
    quality = math.exp(-info.get("residual", 0.0) / 0.1 * math.log(2))
    extra = fill.weight * quality
    splats, conf = gf.lift_frame(
        masks.camera,
        fill.image,
        depth,
        target,
        distance,
        region,
        stride=stride,
        extra_confidence=extra,
    )
    fill.lifted = len(splats)
    return splats, conf


class Carver:
    """Free-space carving against the real (kept) cameras, their nearest-surface depth
    rendered once: a point some camera saw through (in front of the measured surface around
    its pixel) is removed (`generative_fill.carve`'s test). Where that surface is seen at a
    grazing angle a pixel's depth is the mean over a long stretch of it, so the point must
    also be in front by the depth's spread there (`fill_quality.depth_spread`).

    `soft` (per measured gaussian): the weak and unknown ones. A camera does not carve where
    they alone cover its pixel (`SOFT_ALPHA`): a ray through an under-constrained,
    semi-transparent surface (the spool's top, from above) is no proof that the space
    behind it -- or the surface itself -- is empty."""

    def __init__(
        self,
        measured: Splats,
        cameras: Sequence[Camera],
        renderer: Any,
        region_size: float,
        soft: np.ndarray | None = None,
    ) -> None:
        from scipy.ndimage import maximum_filter, minimum_filter

        self.cameras = list(cameras)
        self.margin = gf.CARVE_REGION * region_size
        self.maps = []
        soft_splats = (
            measured.take(np.flatnonzero(soft)) if soft is not None and np.any(soft) else None
        )
        for cam in self.cameras:
            frame = renderer(measured, cam)
            depth = np.where(np.isfinite(frame.depth), frame.depth, np.inf)
            far = minimum_filter(depth, size=3) - fq.depth_spread(frame.depth, cap=0.25)
            covered = maximum_filter(frame.alpha, size=3) >= gf.COVERED
            solid = minimum_filter(covered.astype(np.uint8), size=3) > 0
            if soft_splats is not None:
                through_soft = maximum_filter(renderer(soft_splats, cam).alpha, size=3)
                solid &= through_soft < SOFT_ALPHA
            self.maps.append((far, solid))

    def keep(self, positions: np.ndarray) -> np.ndarray:
        keep = np.ones(len(positions), bool)
        if not len(positions):
            return keep
        for cam, (far, solid) in zip(self.cameras, self.maps, strict=True):
            uv, z = cam.project(positions)
            u = np.floor(uv[:, 0]).astype(np.int64)
            v = np.floor(uv[:, 1]).astype(np.int64)
            inside = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            rows = np.flatnonzero(inside & keep)
            if rows.size == 0:
                continue
            surf = far[v[rows], u[rows]]
            through = solid[v[rows], u[rows]] & (
                z[rows] < surf * (1 - gf.CARVE_SHARE) - self.margin
            )
            keep[rows[through]] = False
        return keep


@dataclass
class ArmState:
    name: str
    fills: list[FilledView] = field(default_factory=list)
    parts: list[Splats] = field(default_factory=list)
    confs: list[np.ndarray] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    carved: int = 0
    #: The reprojected weak surface (`build_surface`), in the arms with photos: drawn into
    #: every view as weak, kept whole (never carved, shape fixed) in the layer.
    surface: Splats | None = None

    def layer(self) -> Splats | None:
        parts = [p for p in self.parts if len(p)]
        return Splats.concat(parts) if parts else None

    def add(self, splats: Splats, conf: np.ndarray, carver: Carver | None) -> None:
        if not len(splats):
            return
        if carver is not None:
            keep = carver.keep(splats.positions)
            self.carved += int((~keep).sum())
            rows = np.flatnonzero(keep)
            splats, conf = splats.take(rows), conf[rows]
        self.parts.append(splats)
        self.confs.append(conf)


def soft_gaussians(setup: Setup) -> np.ndarray:
    """Per measured gaussian: shown and not known (weak, unknown, a leave-out's withheld)."""
    soft = np.zeros(len(setup.measured), bool)
    soft[setup.shown] = setup.shown_classes() != fq.KNOWN
    return soft


def build_surface(
    setup: Setup, renderer: Any, limit: int, log: Callable[[str], None] = print
) -> tuple[Splats | None, dict[str, Any]]:
    """The weak surfaces made solid from the photos that saw them (`fill_surface`): per hole
    cluster (the `SURFACE_CLUSTERS` largest), its gaussians some kept camera saw, their
    surface estimated from the best views' depth, coloured from the best photos, one opaque
    disc per coloured point; at most `limit` in all (shared by size). None when nothing
    could be coloured."""
    scene = setup.scene()
    shown = setup.shown
    labels = setup.clusters.labels[shown]
    seen = setup.quality.seen[shown] > 0
    centres = np.array([v.camera.centre for v in setup.views])
    found: list[tuple[int, fs.Surface]] = []
    report: dict[str, Any] = {"clusters": []}
    for k in range(min(SURFACE_CLUSTERS, len(setup.clusters.info))):
        region = seen & (labels == k)
        if region.sum() < fq.MIN_CLUSTER:
            continue
        cams = fs.best_cameras(
            setup.quality.per_camera,
            shown[np.flatnonzero(region)],
            centres,
            setup.focus.centre,
            max(fs.SURFACE_VIEWS, fs.COLOUR_VIEWS),
        )
        if not cams:
            continue
        positions, normals, spacing, info = fs.estimate_surface(
            scene,
            region,
            [setup.views[c].camera for c in cams[: fs.SURFACE_VIEWS]],
            renderer,
            setup.focus,
            facing=setup.shown_quality().facing,
        )
        if not len(positions):
            report["clusters"].append({"cluster": k, **info})
            continue
        views = []
        for c in cams[: fs.COLOUR_VIEWS]:
            photo = setup.photo(c, 1024)
            if photo is not None:
                views.append(
                    (fv.scaled(setup.views[c].camera, photo.shape[1], photo.shape[0]), photo)
                )
        colours, support = fs.colour_surface(
            positions, normals, views, scene, renderer, setup.focus.density
        )
        surface = fs.Surface(positions, normals, colours, support, spacing, info)
        info.update(
            {
                "cluster": k,
                "colourViews": [setup.views[c].name for c in cams[: fs.COLOUR_VIEWS]],
                "coloured": int(surface.coloured.sum()),
            }
        )
        report["clusters"].append(info)
        found.append((k, surface))
    total = sum(int(sf.coloured.sum()) for _, sf in found)
    parts = []
    for _, surface in found:
        share = int(limit * surface.coloured.sum() / max(total, 1))
        thinned = fs.thin_surface(surface, max(share, 1))
        parts.append(fs.surface_splats(thinned))
    parts = [p for p in parts if len(p)]
    report["gaussians"] = int(sum(len(p) for p in parts))
    log(f"{setup.name}: reprojected weak surface: {report['gaussians']} opaque gaussians")
    return (Splats.concat(parts) if parts else None), report


# --- the joint (set) filler -------------------------------------------------------------------------------------


class SetFiller(Protocol):
    name: str

    def start(
        self, clip: dict[str, Any], seeds: Sequence[int]
    ) -> Callable[[], list[EditResult]]: ...


@dataclass
class StandInSetFiller:
    """The CPU stand-in for VACE: every masked frame inpainted on its own (Telea)."""

    name: str = "standin-telea-frames"

    def start(self, clip: dict[str, Any], seeds: Sequence[int]) -> Callable[[], list[EditResult]]:
        def run() -> list[EditResult]:
            import cv2

            frames, masks = clip["frames"], clip["masks"]
            out = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                done = np.stack(
                    [
                        cv2.inpaint(f, m.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
                        for f, m in zip(frames, masks, strict=True)
                    ]
                ).astype(np.float64)
                done += rng.normal(0, 4.0, (len(frames), 1, 1, 3))
                done = np.where(masks[..., None], done, frames)
                out.append(
                    EditResult(
                        f"set-s{seed}", np.clip(done, 0, 255).astype(np.uint8), {"seconds": 0.0}
                    )
                )
            return out

        return run


@dataclass
class RemoteSetFiller:
    """Wan2.1-VACE-14B on a Modal H100 (`infra/modal/fill.py` `FillVace14.fill_set`)."""

    submit: Submit
    wait: Wait
    steps: int = 25
    distill: bool = False
    name: str = "wan2.1-vace-14b"
    calls: int = 0
    limit: int = 4

    def start(self, clip: dict[str, Any], seeds: Sequence[int]) -> Callable[[], list[EditResult]]:
        import video_fill_models as vfm

        if self.calls + len(seeds) > self.limit:
            raise RuntimeError(f"{self.calls + len(seeds)} set calls: past the cap {self.limit}")
        self.calls += len(seeds)
        packed = vfm.pack_clip(clip["frames"], clip["masks"], clip.get("void"))
        handles = []
        for seed in seeds:
            body = {
                "clip": packed,
                "prompt": clip["prompt"],
                "seed": int(seed),
                "steps": self.steps,
                "distill": self.distill,
            }
            handles.append((seed, self.submit("FillVace14", "fill_set", body)))

        def collect() -> list[EditResult]:
            out = []
            for seed, h in handles:
                try:
                    response = self.wait(h)
                except Exception as error:  # noqa: BLE001
                    out.append(EditResult(f"set-s{seed}", None, {"error": repr(error)[:1500]}))
                    continue
                if "error" in response:
                    out.append(EditResult(f"set-s{seed}", None, dict(response)))
                    continue
                frames, _ = vfm.unpack_clip(response["clip"])
                info = {k: v for k, v in response.items() if k != "clip"}
                out.append(EditResult(f"set-s{seed}", frames, info))
            return out

        return collect


def _cover_crop(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`image` scaled to cover `size` (w, h) and centre-cropped."""
    import cv2

    w, h = size
    ih, iw = image.shape[:2]
    f = max(w / iw, h / ih)
    resized = cv2.resize(
        image, (max(w, round(iw * f)), max(h, round(ih * f))), interpolation=cv2.INTER_AREA
    )
    y0 = (resized.shape[0] - h) // 2
    x0 = (resized.shape[1] - w) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def _contain(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`image` scaled to fit inside `size` (w, h), centred on black."""
    import cv2

    w, h = size
    ih, iw = image.shape[:2]
    f = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * f)), max(1, round(ih * f))
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def build_set(
    setup: Setup,
    anchors: Sequence[FilledView],
    props: Sequence[Target],
    layer: Splats | None,
    renderer: Any,
    options: Options,
    context: dict[str, Any],
    surface: Splats | None = None,
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    """The joint arm's clip: two real context photos, then every target view in a
    nearest-neighbour tour from the first anchor -- anchors as filled (nothing masked),
    propagation views rendered with the anchors' layer and masked where still to fill --
    padded at the front to 4k + 1 frames, at most 81. Returns the clip and per frame what
    it is (`("photo", name)`, `("anchor", key)`, `("prop", ViewMasks)`)."""
    import anchor_models as am

    size = options.set_size
    entries: list[tuple[str, Any]] = []
    frames, masks, voids = [], [], []
    for role, c in fv.context_for_seed(context, 0):
        photo = setup.photo(c)
        if photo is None:
            continue
        frames.append(_cover_crop(photo, size))
        masks.append(np.zeros(size[::-1], bool))
        voids.append(np.zeros(size[::-1], bool))
        entries.append(("photo", f"{role}:{setup.views[c].name}"))
    views: list[tuple[str, Any, np.ndarray]] = [
        ("anchor", a, a.masks.camera.rotation[2]) for a in anchors
    ]
    views += [("prop", t, t.camera.rotation[2]) for t in props]
    dirs = np.array([-v[2] for v in views])
    order = fv.tour(dirs, list(range(len(views))), start=0)
    room = am.VACE_MAX_FRAMES - len(frames)
    import cv2

    for k in order[:room]:
        kind, item, _ = views[k]
        if kind == "anchor":
            frames.append(cv2.resize(item.image, size, interpolation=cv2.INTER_AREA))
            masks.append(np.zeros(size[::-1], bool))
            voids.append(np.zeros(size[::-1], bool))
            entries.append(("anchor", item.key))
        else:
            cam = fv.scaled(item.camera, size[0], size[1])
            vm = view_masks(
                setup,
                cam,
                renderer,
                layer,
                options.weak_strength,
                options.placeholder,
                options.unknown_strength,
                surface,
                options.reproject_strength,
            )
            shown = vm.render.copy()
            shown[vm.unknown] = 127
            frames.append(shown)
            # The reprojected surface's pixels are the photos' own: shown, not masked.
            redo = vm.weak if vm.surface is None else vm.weak & ~vm.surface
            masks.append(vm.unknown | redo)
            voids.append(vm.unknown)
            entries.append(("prop", (item.key, vm)))
    n = len(frames)
    pad = (-(n - 1)) % 4 if n > 1 else 0
    if n == 0:
        raise ValueError("an empty set")
    for _ in range(pad):
        frames.insert(0, frames[0])
        masks.insert(0, masks[0])
        voids.insert(0, voids[0])
        entries.insert(0, ("pad", None))
    clip = {
        "frames": np.stack(frames).astype(np.uint8),
        "masks": np.stack(masks),
        "void": np.stack(voids),
        "prompt": SET_PROMPT.format(caption=setup.caption[0].upper() + setup.caption[1:]),
    }
    return clip, entries


# --- the arms ------------------------------------------------------------------------------------------------------


@dataclass
class Run:
    """One scan's run: every arm, the shared views, the models."""

    setup: Setup
    targets: list[Target]
    editor: Editor
    set_filler: SetFiller | None
    depth: DepthModel
    renderer: Any
    perceptual: Perceptual
    options: Options
    carver: Carver | None
    log: Callable[[str], None] = print
    fallback: Editor | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def seeds(self, n: int, base: int = 17) -> list[int]:
        return [base + 1000 * k for k in range(n)]

    def wait(self, pending: list[tuple[Any, Any]]) -> list[list[EditResult]]:
        out = []
        for _, collect in pending:
            results = collect()
            for r in results:
                self.calls.append(
                    {
                        "key": r.key,
                        **{
                            k: r.info.get(k)
                            for k in ("seconds", "loadSeconds", "error")
                            if k in r.info
                        },
                    }
                )
            out.append(results)
        return out


def view_hole_members(setup: Setup, masks: ViewMasks, target: Target | None) -> np.ndarray:
    """The hole gaussians (measured indices) a view shows in the pixels it fills, else its
    cluster's: what its context photos are cropped about. Cached by view (the first call's
    masks decide)."""
    key = f"members-{target.key if target is not None else id(masks)}"
    if key not in setup._cache:
        rows, _ = anchor_points(setup, masks, _hole_sample(setup))
        members = setup.shown[rows]
        if members.size < 10 and target is not None and target.cluster >= 0:
            members = np.flatnonzero(setup.clusters.labels == target.cluster)
        setup._cache[key] = members
    return setup._cache[key]


def fill_anchors(
    run: Run, arms: Sequence[str], surface: Splats | None = None
) -> dict[str, list[FilledView]]:
    """Every anchor under every anchor arm (`refs`, `norefs`): all seeds sent at once, then
    best of N with agreement between the anchors, composited. The `refs` arm's views carry
    the reprojected weak `surface` (its pixels cleaned at low strength)."""
    setup, opt = run.setup, run.options
    anchor_arms = sorted({ANCHOR_ARM[a] for a in arms})
    targets = [t for t in run.targets if t.role == "anchor"]
    per_arm_masks: dict[str, dict[str, ViewMasks]] = {}
    for arm in anchor_arms:
        per_arm_masks[arm] = {
            t.key: view_masks(
                setup,
                t.camera,
                run.renderer,
                None,
                opt.weak_strength,
                opt.placeholder,
                opt.unknown_strength,
                surface if arm == "refs" else None,
                opt.reproject_strength,
            )
            for t in targets
        }
    pending = []
    meta = []
    for arm in anchor_arms:
        masks = per_arm_masks[arm]
        for t in targets:
            if not masks[t.key].edit.any():
                continue
            reqs = edit_requests(
                setup,
                f"{arm}/{t.key}",
                masks[t.key],
                t.context,
                refs=arm == "refs",
                seeds=run.seeds(opt.seeds),
                steps=opt.anchor_steps,
                lightning=opt.lightning,
                vae_area=opt.vae_area,
                members=view_hole_members(setup, masks[t.key], t),
                tag=t.key,
                placeholder=opt.placeholder,
            )
            pending.append((t.key, run.editor.start([r for r, _ in reqs])))
            meta.append((arm, t, [n for _, n in reqs]))
    run.log(f"{setup.name}: {sum(len(m[2]) for m in meta)} anchor fills sent")
    results = run.wait(pending)
    scene_sample = _hole_sample(setup)
    out: dict[str, list[FilledView]] = {}
    for arm in anchor_arms:
        masks = per_arm_masks[arm]
        per_anchor: dict[str, tuple[list[Candidate], tuple[np.ndarray, np.ndarray]]] = {}
        for (a, t, names), res in zip(meta, results, strict=True):
            if a != arm:
                continue
            cands = score_candidates(masks[t.key], res, names, run.perceptual)
            per_anchor[t.key] = (cands, anchor_points(setup, masks[t.key], scene_sample))
        # First by each seed alone (agreement 1), then twice with the other anchors' choices.
        chosen = {
            k: int(np.argmin([c.score() for c in v[0]])) if v[0] else 0
            for k, v in per_anchor.items()
        }
        for _ in range(2):
            agreement_scores(per_anchor, chosen)
            chosen = {
                k: int(np.argmin([c.score() for c in v[0]])) if v[0] else 0
                for k, v in per_anchor.items()
            }
        fills = []
        for t in targets:
            if t.key not in per_anchor:
                continue
            cands = per_anchor[t.key][0]
            fill = choose(masks[t.key], cands, t.key, "anchor")
            if fill is not None and all(c.drifted for c in cands) and run.fallback is not None:
                fill = _fallback(run, masks[t.key], t, fill)
            if fill is not None:
                fills.append(fill)
        out[arm] = fills
    return out


def copy_fill(fill: FilledView) -> FilledView:
    """A filled view another arm may change (its image and weight copied)."""
    import dataclasses

    return dataclasses.replace(fill, image=fill.image.copy(), weight=fill.weight.copy())


def _fallback(run: Run, masks: ViewMasks, target: Target, fill: FilledView) -> FilledView:
    """Every seed drifted: the exact-mask inpainter fills it instead (one seed)."""
    assert run.fallback is not None
    reqs = edit_requests(
        run.setup,
        f"fallback/{target.key}",
        masks,
        target.context,
        refs=False,
        seeds=[17],
        steps=0,
        lightning=False,
    )
    (results,) = run.wait([(target.key, run.fallback.start([r for r, _ in reqs]))])
    if not results or results[0].image is None:
        return fill
    aligned, _ = register(results[0].image, masks.render, masks.known)
    fill.image = composite(aligned, masks.render, masks.known)
    fill.weight = np.full(fill.weight.shape, 0.5)
    fill.fallback = True
    return fill


def _hole_sample(setup: Setup, limit: int = 6000) -> np.ndarray:
    rows = np.flatnonzero(setup.clusters.labels[setup.shown] >= 0)
    if rows.size > limit:
        rows = np.sort(np.random.default_rng(1).choice(rows, limit, replace=False))
    return rows


def near_holes(setup: Setup, positions: np.ndarray) -> np.ndarray:
    """Which `positions` lie within `LIFT_REACH` of the focus radius of a gaussian to fill
    (a hole cluster's member). The tree is cached."""
    if not len(positions):
        return np.zeros(0, bool)
    tree = setup._cache.get("holeTree")
    if tree is None:
        from scipy.spatial import cKDTree

        members = np.flatnonzero(setup.clusters.labels >= 0)
        if members.size == 0:
            return np.ones(len(positions), bool)
        tree = cKDTree(setup.measured.positions[members])
        setup._cache["holeTree"] = tree
    d, _ = tree.query(positions, k=1)
    return d <= LIFT_REACH * setup.focus.radius


def lift_fills(run: Run, state: ArmState, fills: Sequence[FilledView]) -> None:
    """Each view lifted (`lift_view`), what lands far from every hole dropped (a floater
    from an unknown pixel of the background), carved, added to the arm."""
    region = run.setup.region()
    for fill in fills:
        splats, conf = lift_view(fill, run.depth, region, run.options.lift_stride)
        near = near_holes(run.setup, splats.positions)
        if len(splats) and not near.all():
            state.report["farFromHoles"] = state.report.get("farFromHoles", 0) + int((~near).sum())
            rows = np.flatnonzero(near)
            splats, conf = splats.take(rows), conf[rows]
            fill.lifted = len(splats)
        state.add(splats, conf, run.carver)
        state.fills.append(fill)


def propagate_sequential(run: Run, states: dict[str, ArmState]) -> None:
    """6A: each propagation view, in order, rendered with what each arm has filled so far;
    only what is still to fill is sent (both arms' requests together), the best seed kept
    and lifted before the next view."""
    setup, opt = run.setup, run.options
    props = [t for t in run.targets if t.role == "prop"]
    for t in props:
        pending, meta = [], []
        for arm, state in states.items():
            masks = view_masks(
                setup,
                t.camera,
                run.renderer,
                state.layer(),
                opt.weak_strength,
                opt.placeholder,
                opt.unknown_strength,
                state.surface,
                opt.reproject_strength,
            )
            if masks.share() < PROP_MIN_SHARE:
                state.report.setdefault("skipped", []).append(t.key)
                continue
            reqs = edit_requests(
                setup,
                f"{arm}/{t.key}",
                masks,
                t.context,
                refs=arm == "refs",
                seeds=run.seeds(opt.prop_seeds, base=29),
                steps=opt.prop_steps,
                lightning=opt.lightning,
                vae_area=opt.vae_area,
                members=view_hole_members(setup, masks, t),
                tag=t.key,
                placeholder=opt.placeholder,
            )
            pending.append((t.key, run.editor.start([r for r, _ in reqs])))
            meta.append((arm, masks, [n for _, n in reqs]))
        for (arm, masks, names), results in zip(meta, run.wait(pending), strict=True):
            cands = score_candidates(masks, results, names, run.perceptual)
            fill = choose(masks, cands, t.key, "prop")
            if fill is not None:
                lift_fills(run, states[arm], [fill])
        run.log(
            f"{setup.name}: 6A {t.key} done ({', '.join(f'{a}:{len(s.parts)}' for a, s in states.items())})"
        )


def start_joint(
    run: Run, anchors: Sequence[FilledView], layer: Splats | None, surface: Splats | None = None
) -> tuple[Callable[[], list[EditResult]], list[tuple[str, Any]], dict[str, Any]] | None:
    """6B, sent: the set of views (`build_set`) to the set filler, `set_seeds` seeds."""
    if run.set_filler is None:
        return None
    props = [t for t in run.targets if t.role == "prop"]
    context = run.targets[0].context if run.targets else {}
    clip, entries = build_set(
        run.setup, anchors, props, layer, run.renderer, run.options, context, surface
    )
    seeds = run.seeds(run.options.set_seeds, base=41)
    run.log(f"{run.setup.name}: 6B sent, {len(clip['frames'])} frames x {len(seeds)} seeds")
    return run.set_filler.start(clip, seeds), entries, clip


def finish_joint(
    run: Run,
    state: ArmState,
    sent: tuple[Callable[[], list[EditResult]], list[tuple[str, Any]], dict[str, Any]],
) -> None:
    """6B, back: each propagation frame composited (known pixels as rendered), the seeds'
    agreement as its weight, lifted."""
    collect, entries, clip = sent
    (back,) = run.wait([("set", collect)])
    results = [r for r in back if r.image is not None]
    state.report["setCalls"] = [r.info for r in back]
    if not results:
        state.report["setFailed"] = True
        return
    fills = []
    for k, (kind, item) in enumerate(entries):
        if kind != "prop":
            continue
        key, vm = item
        drawn = [r.image[k] for r in results]
        chosen = drawn[0]
        image = composite(chosen, vm.render, vm.known)
        weight = seed_agreement(chosen, drawn[1:])
        fills.append(
            FilledView(key, "frame", vm, image, weight, [{"seed": r.key} for r in results], 0, [])
        )
    lift_fills(run, state, fills)
    state.report["setFrames"] = len(clip["frames"])


# --- fuse ---------------------------------------------------------------------------------------------------------


def _resize_view(camera: Camera, image: np.ndarray, masks: list[np.ndarray], width: int):
    import cv2

    if camera.width <= width:
        return camera, image, masks
    small = fv.scaled(camera, width)
    image = cv2.resize(image, (small.width, small.height), interpolation=cv2.INTER_AREA)
    masks = [
        cv2.resize(m.astype(np.float32), (small.width, small.height), interpolation=cv2.INTER_AREA)
        for m in masks
    ]
    return small, image, masks


def distil_request(run: Run, state: ArmState, init: Splats, iterations: int) -> dict[str, Any]:
    """`distill_fill.run`'s request: the filled views (their edit pixels, weighted by seed
    agreement, `GENERATED_WEIGHT`), up to `REAL_VIEWS` real photos (weight 1, on what the
    frozen scan shows and the leave-out has not withheld), the frozen scan cropped near."""
    import distill_fill as df

    setup = run.setup
    scene = setup.scene()
    frozen = scene.take(setup.frozen_index())
    cams, images, masks, pweights, weights, outside = [], [], [], [], [], []
    for fill in state.fills:
        c, im, (m, w) = _resize_view(
            fill.camera,
            fill.image,
            [fill.masks.edit, fill.weight * fill.masks.edit],
            run.options.distill_width,
        )
        cams.append(c.to_json())
        images.append(im)
        masks.append(m > 0.5)
        pweights.append(np.clip(w, 0, 1))
        weights.append(GENERATED_WEIGHT)
        outside.append(GENERATED_OUTSIDE)
    withheld = setup.measured.take(np.flatnonzero(setup.withheld & ~setup.dropped))
    ranked = sorted(
        range(len(setup.views)),
        key=lambda k: float(np.linalg.norm(setup.views[k].camera.centre - setup.focus.centre)),
    )
    for k in ranked[:REAL_VIEWS]:
        photo = setup.photo(k, width=run.options.distill_width)
        if photo is None:
            continue
        cam = fv.scaled(setup.views[k].camera, photo.shape[1], photo.shape[0])
        mask = run.renderer(frozen, cam).alpha >= gf.COVERED
        if len(withheld):
            mask &= run.renderer(withheld, cam).alpha < 0.05
        cams.append(cam.to_json())
        images.append(photo)
        masks.append(mask)
        pweights.append(mask.astype(np.float32))
        weights.append(1.0)
        outside.append(0.0)
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    stack = np.zeros((len(images), h, w, 3), np.uint8)
    mstack = np.zeros((len(images), h, w), bool)
    pstack = np.zeros((len(images), h, w), np.float32)
    for j, (im, m, p) in enumerate(zip(images, masks, pweights, strict=True)):
        stack[j, : im.shape[0], : im.shape[1]] = im
        mstack[j, : m.shape[0], : m.shape[1]] = m
        pstack[j, : p.shape[0], : p.shape[1]] = p
    low, high = init.positions.min(axis=0), init.positions.max(axis=0)
    pad = 0.25 * float(np.max(high - low)) + 1e-6
    near = np.all((frozen.positions >= low - pad) & (frozen.positions <= high + pad), axis=1)
    return {
        "measured": df.pack_scan(tf._arrays(frozen.take(np.flatnonzero(near)))),
        "init": df.pack_scan(tf._arrays(init)),
        "views": df.pack_views(
            cams, stack, mstack, weights=weights, outside=outside, pixel_weights=pstack
        ),
        "iterations": iterations,
    }


def _weak_rows(setup: Setup) -> np.ndarray:
    c = setup.shown_classes()
    return np.flatnonzero(
        (c == fq.WEAK) & (setup.clusters.labels[setup.shown] >= 0) & ~setup.withheld[setup.shown]
    )


def _uncovered(setup: Setup, rows: np.ndarray, surface: Splats | None) -> np.ndarray:
    """Of the shown `rows`, those no reprojected surface point lies near (its disc size)."""
    if surface is None or not len(surface) or not rows.size:
        return rows
    from scipy.spatial import cKDTree

    reach = 2.0 * float(np.median(surface.scales[:, 0]))
    d, _ = cKDTree(surface.positions).query(setup.scene().positions[rows], k=1)
    return rows[d > reach]


def weak_count(run: Run, surface: Splats | None = None) -> int:
    """How many weak gaussians in the holes could be copied (none the surface covers)."""
    return int(_uncovered(run.setup, _weak_rows(run.setup), surface).size)


def weak_copies(run: Run, limit: int, surface: Splats | None = None) -> tuple[Splats, np.ndarray]:
    """Trainable copies of the weak gaussians in the holes (the distil refines their look;
    the originals stay frozen) that the reprojected `surface` does not cover: at most
    `limit`, the worst seen first. Returns them and their shown indices."""
    setup = run.setup
    rows = _uncovered(setup, _weak_rows(setup), surface)
    if rows.size > limit:
        best = setup.quality.best[setup.shown][rows]
        rows = np.sort(rows[np.argsort(best, kind="stable")[:limit]])
    return setup.scene().take(rows), rows


def _empty_splats() -> Splats:
    return Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))


def fuse(
    run: Run,
    states: dict[str, ArmState],
    distil: Callable[[dict], dict] | None,
) -> dict[str, tuple[Splats, np.ndarray, list[Camera]] | None]:
    """Each arm: its reprojected weak surface (if any) whole, its lifted gaussians thinned,
    copies of the weak gaussians the surface does not cover, distilled together (the
    surface's shape fixed and its opacity held above `fill_surface.OPACITY_FLOOR`), then the
    dataset-update rounds (every filled view rendered with the fused layer, refined by the
    editor at the round's strength, distilled again), a final carve -- of the lifted
    gaussians only: the surface is where the photos say a surface is. Returns per arm the
    layer, its confidence and the cameras that made it."""
    import distill_fill as df

    setup, opt = run.setup, run.options
    budget = max(MIN_BUDGET, int(BUDGET_SHARE * len(setup.measured)))
    out: dict[str, tuple[Splats, np.ndarray, list[Camera]] | None] = {}
    layers: dict[str, dict[str, Any]] = {}
    for arm, state in states.items():
        surface = state.surface if state.surface is not None else _empty_splats()
        weak = weak_count(run, surface)
        lifted = state.layer()
        if (lifted is None and not weak and not len(surface)) or not state.fills:
            state.report["skipped"] = "nothing lifted, nothing weak, no surface"
            out[arm] = None
            continue
        room = max(0, budget - len(surface))
        if lifted is None:
            # Only weak gaussians to refine (a scan seen everywhere, if badly somewhere):
            # the layer is the surface and their refined copies.
            gen, gen_conf = _empty_splats(), np.zeros(0)
        else:
            # Up to half the room is kept for the weak gaussians' copies: run 37508234704
            # thinned the lifted ones to the whole budget and refined no weak gaussian.
            reserve = min(weak, room // 2)
            thinned = gf.thin(
                gf.Lifted(lifted, np.concatenate(state.confs), [], []), max(room - reserve, 0)
            )
            gen, gen_conf = thinned.splats, thinned.confidence
        copies, copy_rows = weak_copies(run, max(0, room - len(gen)), surface)
        state.report.update(
            {
                "lifted": 0 if lifted is None else len(lifted),
                "carvedOnLift": state.carved,
                "thinned": len(gen),
                "weakCopies": len(copies),
                "surface": len(surface),
            }
        )
        layers[arm] = {
            "gen": gen,
            "conf": gen_conf,
            "copies": copies,
            "copyRows": copy_rows,
            "surface": surface,
        }
    rounds = [(opt.distill, None), *[(opt.update_distill, s) for s in opt.update_strengths]]
    for r, (iterations, strength) in enumerate(rounds):
        if strength is not None:
            _update_round(run, {a: states[a] for a in layers}, layers, strength)
        if distil is None or iterations <= 0:
            continue
        for arm, layer in layers.items():
            parts = [layer["gen"], layer["copies"], layer["surface"]]
            init = (
                Splats.concat([q for q in parts if len(q)]) if any(len(q) for q in parts) else None
            )
            if init is None:
                continue
            n_gen, n_copies, n_surface = (len(q) for q in parts)
            rigid = np.r_[np.zeros(n_gen + n_copies, bool), np.ones(n_surface, bool)]
            floor = np.where(rigid, fs.OPACITY_FLOOR, 0.0)
            t = time.time()
            request = distil_request(run, states[arm], init, iterations)
            if n_surface:
                request["constraints"] = df.pack_constraints(rigid, floor)
            response = distil(request)
            got = df.unpack_scan(response["inferred"])
            fused = Splats(*(got[k] for k in df.KEYS))
            layer["gen"] = fused.take(np.arange(n_gen))
            layer["copies"] = fused.take(np.arange(n_gen, n_gen + n_copies))
            layer["surface"] = fused.take(np.arange(n_gen + n_copies, len(fused)))
            states[arm].report.setdefault("distil", []).append(
                {
                    "round": r,
                    "seconds": round(time.time() - t, 1),
                    **{
                        k: response["report"].get(k)
                        for k in ("iterations", "meanDriftM", "gaussians")
                    },
                }
            )
    for arm, layer in layers.items():
        gen, conf = layer["gen"], layer["conf"]
        keep = run.carver.keep(gen.positions) if run.carver is not None else np.ones(len(gen), bool)
        region = setup.region()
        keep &= np.all((gen.positions >= region[0]) & (gen.positions <= region[1]), axis=1)
        states[arm].report["carvedFinal"] = int((~keep).sum())
        gen, conf = gen.take(np.flatnonzero(keep)), conf[keep]
        copies = layer["copies"]
        if len(copies):
            original = setup.scene().take(layer["copyRows"])
            moved = np.sqrt(np.mean((copies.colours - original.colours) ** 2, axis=1))
            keep_c = moved >= COPY_MIN_CHANGE
            states[arm].report["weakCopiesKept"] = int(keep_c.sum())
            copies = copies.take(np.flatnonzero(keep_c))
            conf = np.concatenate([conf, np.full(len(copies), 0.5)])
            gen = Splats.concat([gen, copies]) if len(gen) else copies
        surface = layer["surface"]
        if len(surface):
            if run.carver is not None:
                # Reported, not applied: the surface is never carved.
                states[arm].report["surfaceCarvable"] = int(
                    (~run.carver.keep(surface.positions)).sum()
                )
            conf = np.concatenate([conf, np.full(len(surface), 0.9)])
            gen = Splats.concat([gen, surface]) if len(gen) else surface
        cams = [f.camera for f in states[arm].fills]
        out[arm] = (gen, conf, cams) if len(gen) else None
    return out


def _update_round(
    run: Run, states: dict[str, ArmState], layers: dict[str, dict[str, Any]], strength: float
) -> None:
    """A dataset update: each arm's filled views rendered with its current layer over the
    frozen scan, refined at `strength` on their edit pixels, the views' images replaced
    there (known pixels never change)."""
    setup, opt = run.setup, run.options
    frozen = setup.scene().take(setup.frozen_index())
    pending, meta = [], []
    for arm, state in states.items():
        layer = layers[arm]
        scene = Splats.concat(
            [frozen, *(q for q in (layer["gen"], layer["copies"], layer["surface"]) if len(q))]
        )
        for fill in state.fills:
            frame = run.renderer(scene, fill.camera)
            rendered = tf.to_u8(fv.unpremultiply(frame))
            edit = fill.masks.edit
            strength_map = np.where(edit, strength, 0.0).astype(np.float32)
            if fill.masks.surface is not None:
                # The reprojected surface's pixels stay the photos': no stronger than before.
                strength_map[fill.masks.surface] = min(strength, opt.reproject_strength)
            masks = ViewMasks(
                fill.camera,
                ~edit & ~fill.masks.void,
                np.zeros_like(edit),
                edit,
                fill.masks.void,
                rendered,
                rendered,
                strength_map,
                fill.masks.depth,
            )
            target = next((t for t in run.targets if t.key == fill.key), None)
            reqs = edit_requests(
                setup,
                f"{arm}/{fill.key}/u{strength:g}",
                masks,
                target.context if target is not None else {},
                refs=ANCHOR_ARM[arm] == "refs",
                seeds=[fill.chosen],
                steps=opt.update_steps,
                lightning=opt.lightning,
                vae_area=opt.vae_area,
                update=True,
                members=view_hole_members(setup, fill.masks, target),
                tag=fill.key,
            )
            pending.append((fill.key, run.editor.start([r for r, _ in reqs])))
            meta.append((fill, masks))
    for (fill, masks), results in zip(meta, run.wait(pending), strict=True):
        if not results or results[0].image is None:
            continue
        aligned, _ = register(results[0].image, masks.render, masks.known)
        fill.image = np.where(fill.masks.edit[..., None], aligned, fill.image).astype(np.uint8)
    run.log(f"{setup.name}: update round at strength {strength:g}: {len(meta)} views")


# --- scores and pictures ------------------------------------------------------------------------------------------


def score_held_out(
    run: Run, layers: dict[str, Splats | None], out: Path, max_views: int = 6, width: int = 480
) -> dict[str, Any]:
    """The leave-out score: per held-out photo, inside the region whose look was withheld
    (what only the held-out cameras saw well), the photo against the scan as the kept
    cameras know it (before) and with each arm's layer: PSNR, LPIPS, DreamSim. Saves the
    side by side (photo | before | arms | the scan trained with them)."""
    from PIL import Image

    setup = run.setup
    scene = setup.scene()
    shown = setup.shown
    c = setup.shown_classes()
    visible = ((c != fq.UNKNOWN) & ~setup.withheld[shown]).astype(np.float64)
    withheld = setup.measured.take(np.flatnonzero(setup.withheld & ~setup.dropped))
    rows, numbers = [], []
    candidates = []
    for view in setup.held_out:
        photo = view.photo(width=width)
        if photo is None:
            continue
        cam = fv.scaled(view.camera, photo.shape[1], photo.shape[0])
        region = np.zeros(photo.shape[:2], bool)
        if len(withheld):
            # Where a withheld gaussian is the front surface (not behind what is shown).
            drawn = run.renderer(withheld, cam)
            front = run.renderer(scene, cam).depth
            ahead = np.isfinite(drawn.depth) & (
                drawn.depth <= np.where(np.isfinite(front), front, np.inf) * 1.03 + 1e-6
            )
            region = _close(tf._covered(drawn.alpha) & ahead, 3)
        candidates.append((int(region.sum()), view, photo, cam, region))
    candidates.sort(key=lambda x: -x[0])
    for area, view, photo, cam, region in candidates[:max_views]:
        if area < 50:
            continue
        before = tf.to_u8(run.renderer(scene, cam, opacity_scale=visible).rgb)
        row = [
            gf.label_image(photo, f"real, held out: {view.name}"),
            gf.label_image(before, "before"),
        ]
        score: dict[str, Any] = {
            "view": view.name,
            "regionPx": area,
            "before": _metrics(run, before, photo, region),
        }
        for arm, layer in layers.items():
            if layer is None or not len(layer):
                row.append(gf.label_image(np.zeros_like(photo), f"{LAYERS[arm]}: nothing"))
                continue
            weights = np.concatenate([visible, np.ones(len(layer))])
            after = tf.to_u8(
                run.renderer(Splats.concat([scene, layer]), cam, opacity_scale=weights).rgb
            )
            row.append(gf.label_image(after, LAYERS[arm]))
            score[arm] = _metrics(run, after, photo, region)
        full = tf.to_u8(run.renderer(setup.measured, cam).rgb)
        row.append(gf.label_image(full, "scan trained with them"))
        # The scored region outlined on the photo only (the others stay clean to compare).
        row[0][_dilate(region, 1) & ~region] = (255, 0, 255)
        rows.append(row)
        numbers.append(score)
    if rows:
        Image.fromarray(gf.grid(rows)).save(out / f"held-out-{setup.name}.png")
    means: dict[str, Any] = {}
    for key in ["before", *layers]:
        vals = [n[key] for n in numbers if key in n]
        if vals:
            means[key] = {m: _mean([v.get(m) for v in vals]) for m in ("psnr", "lpips", "dreamsim")}
    return {"views": numbers, "mean": means}


def _mean(values: Sequence[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return round(float(np.mean(v)), 4) if v else None


def _metrics(run: Run, image: np.ndarray, photo: np.ndarray, region: np.ndarray) -> dict[str, Any]:
    return {
        "psnr": round(tf.psnr(image, photo, region), 3),
        "lpips": _round(run.perceptual.lpips(image, photo, region)),
        "dreamsim": _round(run.perceptual.dreamsim(image, photo, region)),
    }


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(float(x), n)


def rerender_consistency(run: Run, state: ArmState, layer: Splats | None) -> dict[str, Any]:
    """How well the fused scene re-renders each filled view's generated pixels (NeRFiller's
    check): PSNR and LPIPS in the edit pixels, averaged."""
    if layer is None:
        return {}
    scene = Splats.concat([run.setup.scene().take(run.setup.frozen_index()), layer])
    psnrs, lps = [], []
    for fill in state.fills:
        m = fill.masks.edit
        if m.sum() < 50:
            continue
        cam, image, (mm,) = _resize_view(fill.camera, fill.image, [m], 480)
        mm = mm > 0.5
        frame = tf.to_u8(fv.unpremultiply(run.renderer(scene, cam)))
        psnrs.append(tf.psnr(frame, image, mm))
        lps.append(run.perceptual.lpips(frame, image, mm))
    return {"psnr": _mean(psnrs), "lpips": _mean(lps), "views": len(psnrs)}


def anchor_sheet(run: Run, fills: dict[str, list[FilledView]], out: Path) -> None:
    """Per anchor: its context photos (as the editor is given them, cropped about the
    hole), what the editor was given, and each arm's fill."""
    import cv2
    from PIL import Image

    setup = run.setup
    w, h = 384, round(384 * run.options.fill_size[1] / run.options.fill_size[0])
    small = lambda im: cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    by_key = {arm: {f.key: f for f in fs} for arm, fs in fills.items()}
    rows = []
    for t in run.targets:
        if t.role != "anchor":
            continue
        row = []
        for role, c in fv.context_roles(t.context).items():
            photo = setup.context_photo(c, setup._cache.get(f"members-{t.key}"), t.key)
            if photo is not None:
                row.append(
                    gf.label_image(_contain(photo, (w, h)), f"{role}: {setup.views[c].name}")
                )
        while len(row) < 4:
            row.append(np.zeros((h, w, 3), np.uint8))
        any_fill = next((by_key[a][t.key] for a in by_key if t.key in by_key[a]), None)
        if any_fill is None:
            continue
        given = any_fill.masks.condition.copy()
        unknown = any_fill.masks.unknown
        given[_dilate(unknown, 3) & ~unknown] = (255, 0, 255)  # what it had to make, outlined
        row.append(gf.label_image(small(given), f"{t.key} {t.elevation:g}/{t.azimuth:g} given"))
        for arm, table in by_key.items():
            f = table.get(t.key)
            row.append(
                gf.label_image(
                    small(f.image) if f else np.zeros((h, w, 3), np.uint8),
                    f"{arm} seed {f.chosen if f else '-'}",
                )
            )
        rows.append(row)
    if rows:
        Image.fromarray(gf.grid(rows)).save(out / f"anchors-{setup.name}.png")


def before_after(run: Run, layers: dict[str, Splats | None], out: Path) -> None:
    """The scan from a few angles, alone and with each layer."""
    from PIL import Image

    setup = run.setup
    scene = setup.scene()
    rows = []
    for elev, az in ((65.0, 30.0), (40.0, 150.0), (15.0, 270.0), (90.0, 0.0)):
        cam = fv.view_camera(setup.focus, fv._direction(elev, az), (480, 276))
        row = [gf.label_image(tf.to_u8(run.renderer(scene, cam).rgb), f"scan {elev:g}/{az:g}")]
        for arm, layer in layers.items():
            both = Splats.concat([scene, layer]) if layer is not None and len(layer) else scene
            row.append(gf.label_image(tf.to_u8(run.renderer(both, cam).rgb), f"+ {LAYERS[arm]}"))
        rows.append(row)
    Image.fromarray(gf.grid(rows)).save(out / f"before-after-{setup.name}.png")


#: The headline test (the owner's): the largest hole seen from straight above and from eight
#: azimuths at 15 and at 45 degrees; of its footprint (its gaussians' pixels), the share whose
#: opacity -- the gaussians in the hole's box, the measured ones a fill supersedes swapped
#: for the fill's -- is below `SEE_THROUGH_ALPHA` is see-through. Pass: at most
#: `SEE_THROUGH_PASS` from every direction, and a mean opacity of `MEAN_ALPHA_PASS`.
HEADLINE_VIEWS = (
    [(90.0, 0.0)]
    + [(15.0, float(a)) for a in range(0, 360, 45)]
    + [(45.0, float(a)) for a in range(0, 360, 45)]
)
SEE_THROUGH_ALPHA = 0.5
SEE_THROUGH_PASS = 0.05
MEAN_ALPHA_PASS = 0.95


def _known_rows(setup: Setup) -> np.ndarray:
    """'Before' as shown-gaussian indices: all; in a leave-out, only what the kept cameras
    know (the withheld and unknown gaussians left out, as `score_held_out` draws it)."""
    if setup.leave_out == "none":
        return np.arange(len(setup.shown))
    c = setup.shown_classes()
    return np.flatnonzero((c != fq.UNKNOWN) & ~setup.withheld[setup.shown])


def _known_scene(setup: Setup) -> Splats:
    return setup.scene().take(_known_rows(setup))


def _headline_camera(centre: np.ndarray, distance: float, fov: float, elev: float, az: float, size):
    d = fv._direction(elev, az)
    up = (0.0, 1.0, 0.0) if elev >= 89.0 else (0.0, 0.0, 1.0)
    return Camera.look_at(
        centre + distance * d, centre, fov_deg=fov, width=size[0], height=size[1], up=up
    )


def see_through(
    run: Run,
    scene: Splats,
    layer: Splats | None,
    members: np.ndarray,
    box: tuple[np.ndarray, np.ndarray],
    cam: Camera,
) -> tuple[float, float] | None:
    """(see-through share, mean opacity) over the hole's footprint (its gaussians' `members`
    positions projected, closed): of the gaussians in its `box` -- the scene's and the
    layer's -- the share of pixels less opaque than `SEE_THROUGH_ALPHA`, and the mean."""
    uv, z = cam.project(members)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    if ok.sum() < 10:
        return None
    foot = np.zeros((cam.height, cam.width), bool)
    foot[v[ok], u[ok]] = True
    foot = _close(_dilate(foot, 1), 2)
    parts = [scene]
    if layer is not None and len(layer):
        parts.append(layer)
    inside = [
        q.take(np.flatnonzero(np.all((q.positions >= box[0]) & (q.positions <= box[1]), axis=1)))
        for q in parts
    ]
    inside = [q for q in inside if len(q)]
    if not inside:
        return 1.0, 0.0
    alpha = run.renderer(Splats.concat(inside), cam).alpha[foot]
    return float((alpha < SEE_THROUGH_ALPHA).mean()), float(alpha.mean())


def headline(
    run: Run,
    layers: dict[str, Splats | None],
    out: Path,
    hidden: dict[str, np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """The owner's test on the largest hole (`HEADLINE_VIEWS`), before and with each arm's
    layer -- the measured gaussians the arm supersedes (`hidden`, per measured gaussian)
    swapped out, as the viewer draws the variant. Saves `headline-<scan>.png`: straight
    above and the two directions most see-through before."""
    from PIL import Image

    setup = run.setup
    if not setup.clusters.info:
        return {}
    hidden = hidden or {}
    members = setup.measured.positions[setup.clusters.members(0)]
    info = setup.clusters.info[0]
    low, high = np.asarray(info["low"], float), np.asarray(info["high"], float)
    pad = 0.05 * (high - low) + 1e-6
    box = (low - pad, high + pad)
    centre = np.median(members, axis=0)
    rows = _known_rows(setup)
    scene = setup.scene()
    before = scene.take(rows)
    after_base: dict[str, Splats] = {}
    for arm in layers:
        mask = hidden.get(arm)
        if mask is None:
            after_base[arm] = before
        else:
            keep = rows[~np.asarray(mask, bool)[setup.shown][rows]]
            after_base[arm] = scene.take(keep)
    width = run.options.score_width
    size = (width, max(1, round(width * 0.75)))
    distance = setup.focus.distance
    fov = 40.0
    table = []
    for elev, az in HEADLINE_VIEWS:
        cam = _headline_camera(centre, distance, fov, elev, az, size)
        row: dict[str, Any] = {"elevation": elev, "azimuth": az}
        got = see_through(run, before, None, members, box, cam)
        row["before"] = None if got is None else [round(got[0], 4), round(got[1], 4)]
        for arm, layer in layers.items():
            got = see_through(run, after_base[arm], layer, members, box, cam)
            row[arm] = None if got is None else [round(got[0], 4), round(got[1], 4)]
        table.append(row)
    scored = [r for r in table if r["before"] is not None]
    worst = sorted((r for r in scored if r["elevation"] < 89.0), key=lambda r: -r["before"][0])[:2]
    shown_views = [r for r in table if r["elevation"] >= 89.0] + worst
    sheet = []
    for r in shown_views:
        cam = _headline_camera(centre, distance, fov, r["elevation"], r["azimuth"], size)
        st = r["before"]
        line = [
            gf.label_image(
                tf.to_u8(run.renderer(before, cam).rgb),
                f"before {r['elevation']:g}/{r['azimuth']:g}: {_pct(st and st[0])} see-through",
            )
        ]
        for arm, layer in layers.items():
            base = after_base[arm]
            both = Splats.concat([base, layer]) if layer is not None and len(layer) else base
            st = r.get(arm)
            line.append(
                gf.label_image(
                    tf.to_u8(run.renderer(both, cam).rgb), f"{LAYERS[arm]}: {_pct(st and st[0])}"
                )
            )
        sheet.append(line)
    if sheet:
        Image.fromarray(gf.grid(sheet)).save(out / f"headline-{setup.name}.png")
    verdict = {}
    for key in ["before", *layers]:
        values = [r[key] for r in table if r.get(key) is not None]
        if values:
            worst_share = max(v[0] for v in values)
            mean_alpha = float(np.mean([v[1] for v in values]))
            verdict[key] = {
                "worstSeeThrough": round(worst_share, 4),
                "meanAlpha": round(mean_alpha, 4),
                "pass": worst_share <= SEE_THROUGH_PASS and mean_alpha >= MEAN_ALPHA_PASS,
            }
    return {
        "cluster": 0,
        "seeThroughAlpha": SEE_THROUGH_ALPHA,
        "views": table,
        "verdict": verdict,
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


# --- the run -----------------------------------------------------------------------------------------------------------


def run_arms(
    setup: Setup,
    editor: Editor,
    set_filler: SetFiller | None,
    depth: DepthModel,
    renderer: Any,
    perceptual: Perceptual,
    options: Options,
    measured_tileset: Path,
    out: Path,
    *,
    distil: Callable[[dict], dict] | None = None,
    fallback: Editor | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every arm on one setup: views, anchors, propagation, fusion, layers, scores and
    pictures under `out`; the report (also `out/report.json`)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(exist_ok=True)
    started = time.time()
    report: dict[str, Any] = {
        "scene": setup.name,
        "leaveOut": setup.leave_out,
        "arms": list(options.arms),
        "options": _options_json(options),
    }
    targets, plan = plan_targets(setup, renderer, options)
    report["plan"] = plan
    log(
        f"{setup.name}: {sum(t.role == 'anchor' for t in targets)} anchors, {sum(t.role == 'prop' for t in targets)} propagation views"
    )
    report["timings"] = {"planS": round(time.time() - started, 1)}
    if not targets:
        report["skipped"] = "no view to fill"
        gf.write_json(out / "report.json", report)
        return report
    # Carving is against everything measured (`generative_fill` did the same): the kept
    # cameras' sight lines end at every surface, a dropped one included; it only removes.
    carver = Carver(
        setup.measured,
        [fv.scaled(v.camera, options.carve_width) for v in setup.views],
        renderer,
        2 * setup.focus.radius,
        soft_gaussians(setup),
    )
    run = Run(
        setup,
        targets,
        editor,
        set_filler,
        depth,
        renderer,
        perceptual,
        options,
        carver,
        log,
        fallback,
    )
    # Weak surfaces first, in the arms with photos: the best photos reprojected onto the
    # surface they saw, as opaque gaussians every later view is drawn with.
    surface = None
    if options.reproject and any(ANCHOR_ARM[a] == "refs" for a in options.arms):
        t = time.time()
        budget = max(MIN_BUDGET, int(BUDGET_SHARE * len(setup.measured)))
        surface, report["surface"] = build_surface(
            setup, renderer, int(SURFACE_BUDGET_SHARE * budget), log
        )
        report["timings"]["surfaceS"] = round(time.time() - t, 1)
    t = time.time()
    anchors = fill_anchors(run, options.arms, surface)
    report["timings"]["anchorsS"] = round(time.time() - t, 1)
    states: dict[str, ArmState] = {}
    lifted: dict[str, ArmState] = {}
    for arm in sorted({ANCHOR_ARM[a] for a in options.arms}):
        lifted[arm] = ArmState(arm, surface=surface if arm == "refs" else None)
        lift_fills(run, lifted[arm], anchors.get(arm, []))
    for arm in options.arms:
        # Each arm its own copy of its anchors (the update rounds rewrite their images).
        base = lifted[ANCHOR_ARM[arm]]
        states[arm] = ArmState(
            arm,
            [copy_fill(f) for f in base.fills],
            list(base.parts),
            list(base.confs),
            {},
            base.carved,
            base.surface,
        )
    # 6B goes first (it needs only the anchors), so its GPU runs while 6A goes on.
    joint = None
    if "vace" in options.arms:
        try:
            joint = start_joint(
                run, anchors.get("refs", []), states["vace"].layer(), states["vace"].surface
            )
        except Exception as error:  # noqa: BLE001 - the arm fails, the others go on
            states["vace"].report["setFailed"] = repr(error)[:1500]
    t = time.time()
    sequential = {a: states[a] for a in options.arms if a != "vace"}
    if sequential:
        propagate_sequential(run, sequential)
    report["timings"]["sequentialS"] = round(time.time() - t, 1)
    if joint is not None:
        t = time.time()
        try:
            finish_joint(run, states["vace"], joint)
        except Exception as error:  # noqa: BLE001
            states["vace"].report["setFailed"] = repr(error)[:1500]
        report["timings"]["jointS"] = round(time.time() - t, 1)
    t = time.time()
    fused = fuse(run, states, distil)
    report["timings"]["fuseS"] = round(time.time() - t, 1)
    layers: dict[str, Splats | None] = {}
    hidden: dict[str, np.ndarray | None] = {}
    report["candidates"] = {}
    for arm, state in states.items():
        entry: dict[str, Any] = {**state.report, "fills": [f.to_json() for f in state.fills]}
        # Swap, don't stack: the measured gaussians the rebuilt surface replaces.
        hidden[arm] = None
        if state.surface is not None and len(state.surface) and arm in fused and fused[arm]:
            hidden[arm] = fs.superseded(setup.measured.positions, state.surface)
            entry["superseded"] = int(hidden[arm].sum())
        made = fused.get(arm)
        layers[arm] = None if made is None else made[0]
        if made is not None:
            splats, conf, cams = made
            entry["reRender"] = rerender_consistency(run, state, splats)
            agreement = [
                float(f.weight[f.masks.edit].mean()) for f in state.fills if f.masks.edit.any()
            ]
            entry["seedAgreement"] = _mean(agreement)
            layer_dir = out / LAYERS[arm] / "inferred"
            entry["evidence"] = tf.package_inferred(
                splats,
                conf,
                cams,
                measured_tileset,
                layer_dir,
                filler_name(arm, editor, set_filler),
                rule=RULE.format(
                    refs=(
                        " and two retrieved real photos, after the weak surfaces were made "
                        "solid from the best photos reprojected onto them"
                        if state.surface is not None and len(state.surface)
                        else " and two retrieved real photos"
                    )
                    if ANCHOR_ARM[arm] == "refs"
                    else " alone",
                    propagate="the rest sequentially, each view re-rendered with what is filled"
                    if arm != "vace"
                    else "the rest jointly by a video model over the set of views",
                ),
                extra={
                    "provenance": "inferred-generated",
                    "arm": arm,
                    "reprojected": 0 if state.surface is None else len(state.surface),
                    "anchors": sum(f.role == "anchor" for f in state.fills),
                    "propagated": sum(f.role != "anchor" for f in state.fills),
                    "seedAgreement": entry["seedAgreement"],
                    "depth": depth.name,
                },
            )
            entry["layer"] = str(layer_dir)
            if hidden[arm] is not None:
                entry["supersedes"] = write_supersedes(
                    layer_dir, measured_tileset.parent, hidden[arm], log
                )
        report["candidates"][arm] = entry
    report["calls"] = run.calls
    report["editorCalls"] = getattr(editor, "calls", None)
    t = time.time()
    if setup.held_out:
        report["heldOut"] = score_held_out(run, layers, out / "renders", width=options.score_width)
    anchor_sheet(run, anchors, out / "renders")
    before_after(run, layers, out / "renders")
    report["headline"] = headline(run, layers, out / "renders", hidden)
    report["timings"]["scoreS"] = round(time.time() - t, 1)
    report["timings"]["totalS"] = round(time.time() - started, 1)
    gf.write_json(out / "report.json", report)
    return report


def write_supersedes(
    layer_dir: Path, tiles_dir: Path, mask: np.ndarray, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """The measured gaussians a layer's rebuilt surface replaces, beside the layer
    (`supersedes.json`, `fill_surface.supersede_document`) and named in its root's extras, so
    the viewer hides them while the layer is shown. Nothing is written when the tiles are not
    this scan's (a test's stand-in tileset), or when they cannot be read: the layer, its scores
    and the headline still come back, and the report says why."""
    try:
        document = fs.supersede_document(tiles_dir, mask)
    except (OSError, ValueError, KeyError) as error:
        log(f"supersedes not written: {error}")
        return {"written": False, "superseded": int(mask.sum()), "error": str(error)}
    if document is None:
        return {"written": False, "superseded": int(mask.sum())}
    (layer_dir / "supersedes.json").write_text(json.dumps(document), encoding="utf-8")
    path = layer_dir / "tileset.json"
    tileset = json.loads(path.read_text(encoding="utf-8"))
    tileset["root"].setdefault("extras", {})["supersedes"] = {
        "uri": "supersedes.json",
        "superseded": document["superseded"],
    }
    path.write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return {"written": True, "superseded": document["superseded"]}


def filler_name(arm: str, editor: Editor, set_filler: SetFiller | None) -> str:
    base = getattr(editor, "name", "editor")
    if arm == "refs":
        return f"{base}+photos+6a"
    if arm == "norefs":
        return f"{base}+6a"
    return f"{base}+photos+{getattr(set_filler, 'name', 'set')}"


def _options_json(o: Options) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.__dict__.items()}


# --- the command line ----------------------------------------------------------------------------------------------

#: Set inside Modal (`infra/modal/fill.py`): spawn a GPU class's method and wait for it.
BACKEND: tuple[Submit, Wait] | None = None


def parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="every arm on one scan: layers, scores and renders")
    run.add_argument("tileset", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--scan", default="scan")
    run.add_argument("--caption", required=True, help="what the scan shows, for the prompts")
    run.add_argument("--poses", type=Path, required=True)
    run.add_argument("--placement", type=Path)
    run.add_argument("--frames", type=Path)
    run.add_argument("--leave-out", choices=("none", "high", "low"), default="none")
    run.add_argument("--leave-share", type=float, default=0.15)
    run.add_argument("--arms", default=",".join(ARMS))
    run.add_argument("--min-frame-psnr", type=float, default=13.0)
    run.add_argument("--editor", choices=("remote", "standin"), default="remote")
    run.add_argument("--set-filler", choices=("remote", "standin", "none"), default="remote")
    run.add_argument("--depth", choices=("prompt", "mono", "oracle"), default="prompt")
    run.add_argument("--renderer", choices=("cpu", "gsplat"), default="gsplat")
    run.add_argument("--perceptual", choices=("lpips", "standin"), default="lpips")
    run.add_argument("--distill-on", choices=("local", "none"), default="local")
    run.add_argument("--fallback", action="store_true")
    for name, default in Options().__dict__.items():
        if name in ("arms", "fallback"):
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            run.add_argument(
                flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=default
            )
        elif isinstance(default, tuple):
            run.add_argument(flag, default=",".join(str(x) for x in default))
        else:
            run.add_argument(flag, type=type(default), default=default)
    return p


def options_from(args: Any) -> Options:
    o = Options()
    for name, default in Options().__dict__.items():
        if name == "arms":
            o.arms = tuple(a for a in args.arms.split(",") if a)
            unknown = set(o.arms) - set(ARMS)
            if unknown:
                raise SystemExit(f"arms {sorted(unknown)}: one of {ARMS}")
            continue
        if name == "fallback":
            o.fallback = bool(args.fallback)
            continue
        value = getattr(args, name)
        if isinstance(default, tuple):
            kind = type(default[0]) if default else float
            value = tuple(kind(x) for x in str(value).split(",") if x != "")
        setattr(o, name, value)
    if o.placeholder not in PLACEHOLDERS:
        raise SystemExit(f"placeholder {o.placeholder!r}: one of {PLACEHOLDERS}")
    return o


def smoke_request() -> EditRequest:
    """A tiny request (a gradient with a box to make, one reference) that checks the editor
    runs before a job sends it hundreds; sent first, so its load overlaps the setup."""
    h, w = 144, 256
    yy, xx = np.mgrid[0:h, 0:w]
    render = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 128)], -1).astype(np.uint8)
    make = np.zeros((h, w), bool)
    make[48:96, 96:160] = True
    strength = np.where(make, UNKNOWN_STRENGTH, 0.0).astype(np.float32)
    condition = prefill(render, make, ~make)
    return EditRequest(
        "smoke-s1",
        condition,
        condition,
        strength,
        [render],
        prompt_for("a gradient", 1),
        1,
        4,
        True,
    )


def _log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    import distill_fill
    from splat_render import load_tileset

    args = parser().parse_args(argv)
    options = options_from(args)
    smoke = None
    if args.editor == "remote" and BACKEND is not None:
        # Sent now, read after the setup: the editor's container loads meanwhile.
        smoke = RemoteEditor(*BACKEND, limit=1).start([smoke_request()])
    renderer = tf.make_renderer(args.renderer)
    measured = load_tileset(args.tileset)
    placement = json.loads(args.placement.read_text()) if args.placement else None
    views = gf.real_views(args.poses, placement, args.frames)
    check = gf.frame_check(measured, views, renderer)
    _log(f"frame check: {json.dumps(check)}")
    if check["meanPsnr"] is None or check["meanPsnr"] < args.min_frame_psnr:
        raise SystemExit(f"the cameras do not match the tiles (frame check {check['meanPsnr']} dB)")
    setup, info = make_setup(
        args.scan,
        args.caption,
        measured,
        views,
        renderer,
        leave_out=args.leave_out,
        share=args.leave_share,
        width=options.quality_width,
        log=_log,
    )
    info["frameCheck"] = check
    if smoke is not None:
        (result,) = smoke()
        info["smoke"] = dict(result.info)
        _log(f"editor smoke test: {json.dumps(info['smoke'])[:2000]}")
        if result.image is None:
            raise SystemExit(f"the editor failed its smoke test: {result.info.get('error')}")
    _log(f"setup: {json.dumps({k: info[k] for k in ('focus', 'kept', 'withheld', 'dropped')})}")
    remote = args.editor == "remote" or args.set_filler == "remote" or args.fallback
    if remote and BACKEND is None:
        raise SystemExit("the remote models need a backend (run through infra/modal/fill.py)")
    editor: Editor = (
        RemoteEditor(*BACKEND, limit=options.max_edit_calls)  # type: ignore[misc]
        if args.editor == "remote"
        else StandInEditor()
    )
    set_filler: SetFiller | None = None
    if args.set_filler == "remote":
        set_filler = RemoteSetFiller(
            *BACKEND,
            steps=options.set_steps,
            distill=options.set_distill,
            limit=options.max_set_calls,
        )  # type: ignore[misc]
    elif args.set_filler == "standin":
        set_filler = StandInSetFiller()
    fallback = (
        RemoteFallback(*BACKEND, limit=options.max_fallback_calls)
        if args.fallback and BACKEND
        else None
    )  # type: ignore[misc]
    import anchor_models as am

    if args.depth == "prompt":
        depth: DepthModel = am.PromptDepth()
    elif args.depth == "mono":
        depth = am.MonoDepth()
    else:
        depth = OracleDepth(measured, renderer)
    perceptual: Perceptual = am.Perceptual() if args.perceptual == "lpips" else StandInPerceptual()
    report = run_arms(
        setup,
        editor,
        set_filler,
        depth,
        renderer,
        perceptual,
        options,
        args.tileset,
        args.out,
        distil=distill_fill.run if args.distill_on == "local" else None,
        fallback=fallback,
        log=_log,
    )
    report["setup"] = info
    gf.write_json(args.out / "report.json", report)
    brief = {
        arm: {
            "evidence": {k: e.get("evidence", {}).get(k) for k in ("gaussians", "meanConfidence")},
            "seedAgreement": e.get("seedAgreement"),
            "reRender": e.get("reRender"),
        }
        for arm, e in report.get("candidates", {}).items()
    }
    print(
        json.dumps(
            {
                "scene": setup.name,
                "candidates": brief,
                "heldOut": report.get("heldOut", {}).get("mean"),
            },
            default=gf._json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
