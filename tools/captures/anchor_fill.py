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
import fill_views as fv
import generative_fill as gf
import teacher_fill as tf
from splat_render import Camera, Splats

# --- constants --------------------------------------------------------------------------------------

#: The strength weak pixels are refined at (0 keeps them, 1 makes them anew).
WEAK_STRENGTH = 0.45
#: The ring of known pixels around a hole that seeds are scored on, px at a 1248-px width.
RING_PX = 24
#: A seed whose registered output keeps the known pixels below this (dB, blurred) drifted.
DRIFT_DB = 18.0
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

ARMS = ("refs", "norefs", "vace")
LAYERS = {"refs": "anchor-refs", "norefs": "anchor-norefs", "vace": "anchor-vace"}
ANCHOR_ARM = {"refs": "refs", "norefs": "norefs", "vace": "refs"}

EDIT_PROMPT = (
    "Picture 1 is a view of {caption} rendered from a new camera position; the parts of it "
    "that were never photographed are painted flat magenta. Replace every magenta pixel with "
    "what the real scene shows there, continuing the surfaces, materials, colours and "
    "lighting around it, as a sharp real photograph.{refs} Keep the camera, the framing and "
    "every other pixel of Picture 1 exactly as they are. No magenta may remain."
)
REFS_SENTENCE = {
    1: (
        " Picture 2 is a real photograph of the same scene taken from another position: use "
        "it for the true appearance of every surface."
    ),
    2: (
        " Pictures 2 and 3 are real photographs of the same scene taken from other positions: "
        "use them for the true appearance of every surface."
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
        withheld, dropped = fq.withheld_by_holdout(kept, everyone)
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
    report = {
        "selection": selection.report,
        "candidates": [c.to_json() for c in candidates],
        "targets": [t.to_json() for t in targets],
        "contexts": {str(k): {kk: vv for kk, vv in v.items()} for k, v in contexts.items()},
    }
    return targets, report


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
    scan's colour (`render`, uint8), what the editor is shown (`condition`: the pixels to
    make in the key colour), the strength map and the rendered depth of the known and weak
    pixels (NaN elsewhere: what depth completion is anchored to)."""

    camera: Camera
    known: np.ndarray
    weak: np.ndarray
    unknown: np.ndarray
    void: np.ndarray
    render: np.ndarray
    condition: np.ndarray
    strength: np.ndarray
    depth: np.ndarray

    @property
    def edit(self) -> np.ndarray:
        return self.unknown | self.weak

    def share(self) -> float:
        return float(self.edit.sum() / max(int((~self.void).sum()), 1))


def view_masks(
    setup: Setup,
    camera: Camera,
    renderer: Any,
    layer: Splats | None = None,
    weak_strength: float = WEAK_STRENGTH,
) -> ViewMasks:
    import anchor_models as am

    scene = setup.scene()
    c = setup.shown_classes()
    known = c == fq.KNOWN
    weak = c == fq.WEAK
    face = fq.facing(setup.shown_quality(), camera.centre, scene.positions)
    if layer is not None and len(layer):
        scene = Splats.concat([scene, layer])
        ones = np.ones(len(layer), bool)
        known = np.concatenate([known, ones])
        weak = np.concatenate([weak, ~ones])
        face = np.concatenate([face, ones])
    pc = fv.pixel_classes(scene, camera, renderer, known, weak, face)
    render = tf.to_u8(pc.colour)
    condition = render.copy()
    condition[pc.unknown] = am.KEY
    condition[pc.void] = 0
    strength = np.where(pc.known, 0.0, np.where(pc.weak, weak_strength, 1.0)).astype(np.float32)
    depth = np.where((pc.known | pc.weak) & np.isfinite(pc.depth), pc.depth, np.nan)
    return ViewMasks(
        camera, pc.known, pc.weak, pc.unknown, pc.void, render, condition, strength, depth
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
                make = r.strength >= 1.0
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
                "mask": am.encode_png(((r.strength >= 1.0) * 255).astype(np.uint8)),
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

    @property
    def drifted(self) -> bool:
        return self.drift is not None and self.drift < DRIFT_DB


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
        }


def prompt_for(caption: str, references: int, update: bool = False) -> str:
    refs = REFS_SENTENCE[min(references, 2)] if references else ""
    template = UPDATE_PROMPT if update else EDIT_PROMPT
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
) -> list[tuple[EditRequest, list[str]]]:
    """One request per seed: the condition, the render and strength, and (with `refs`) the
    seed's pair of context photos."""
    out = []
    for j, seed in enumerate(seeds):
        names: list[str] = []
        photos: list[np.ndarray] = []
        if refs:
            for role, c in fv.context_for_seed(context, j):
                photo = setup.photo(c)
                if photo is not None:
                    photos.append(photo)
                    names.append(f"{role}:{setup.views[c].name}")
        request = EditRequest(
            f"{key}-s{seed}",
            masks.condition,
            masks.render,
            masks.strength,
            photos,
            prompt_for(setup.caption, len(photos), update),
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
    out = []
    for r, n in zip(results, names, strict=True):
        if r.image is None:
            continue
        aligned, info = register(r.image, masks.render, masks.known)
        lp = perceptual.lpips(aligned, masks.render, ring) if ring.any() else None
        seed = int(r.key.rsplit("-s", 1)[-1])
        out.append(
            Candidate(
                seed,
                aligned,
                1.0 if lp is None else lp,
                known_psnr(aligned, masks),
                n,
                {**r.info, **info},
            )
        )
    return out


def choose(masks: ViewMasks, cands: Sequence[Candidate], key: str, role: str) -> FilledView | None:
    """The best candidate (ring LPIPS + agreement, drifted ones last), composited; its
    seed agreement with the rest as per-pixel weight."""
    if not cands:
        return None
    score = [
        c.ring + AGREE_WEIGHT * (1.0 - c.agreement) + (10.0 if c.drifted else 0.0) for c in cands
    ]
    best = int(np.argmin(score))
    chosen = cands[best]
    image = composite(chosen.aligned, masks.render, masks.known)
    others = [c.aligned for k, c in enumerate(cands) if k != best and not c.drifted]
    weight = seed_agreement(chosen.aligned, others)
    seeds = [
        {
            "seed": c.seed,
            "ringLpips": round(c.ring, 4),
            "keptDb": None if c.drift is None else round(c.drift, 2),
            "agreement": round(c.agreement, 4),
            "score": round(s, 4),
            "context": c.context,
            "seconds": c.info.get("seconds"),
            "warp": c.info.get("warp") is not None,
        }
        for c, s in zip(cands, score, strict=True)
    ]
    return FilledView(key, role, masks, image, weight, seeds, chosen.seed, chosen.context)


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
    target = masks.unknown
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
    also be in front by the depth's spread there (`fill_quality.depth_spread`)."""

    def __init__(
        self, measured: Splats, cameras: Sequence[Camera], renderer: Any, region_size: float
    ) -> None:
        from scipy.ndimage import maximum_filter, minimum_filter

        self.cameras = list(cameras)
        self.margin = gf.CARVE_REGION * region_size
        self.maps = []
        for cam in self.cameras:
            frame = renderer(measured, cam)
            depth = np.where(np.isfinite(frame.depth), frame.depth, np.inf)
            far = minimum_filter(depth, size=3) - fq.depth_spread(frame.depth, cap=0.25)
            covered = maximum_filter(frame.alpha, size=3) >= gf.COVERED
            solid = minimum_filter(covered.astype(np.uint8), size=3) > 0
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


def build_set(
    setup: Setup,
    anchors: Sequence[FilledView],
    props: Sequence[Target],
    layer: Splats | None,
    renderer: Any,
    options: Options,
    context: dict[str, Any],
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
            vm = view_masks(setup, cam, renderer, layer, options.weak_strength)
            shown = vm.render.copy()
            shown[vm.unknown] = 127
            frames.append(shown)
            masks.append(vm.unknown | vm.weak)
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


def fill_anchors(run: Run, arms: Sequence[str]) -> dict[str, list[FilledView]]:
    """Every anchor under every anchor arm (`refs`, `norefs`): all seeds sent at once, then
    best of N with agreement between the anchors, composited."""
    setup, opt = run.setup, run.options
    anchor_arms = sorted({ANCHOR_ARM[a] for a in arms})
    targets = [t for t in run.targets if t.role == "anchor"]
    masks = {
        t.key: view_masks(setup, t.camera, run.renderer, None, opt.weak_strength) for t in targets
    }
    pending = []
    meta = []
    for arm in anchor_arms:
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
            )
            pending.append((t.key, run.editor.start([r for r, _ in reqs])))
            meta.append((arm, t, [n for _, n in reqs]))
    run.log(f"{setup.name}: {sum(len(m[2]) for m in meta)} anchor fills sent")
    results = run.wait(pending)
    scene_sample = _hole_sample(setup)
    out: dict[str, list[FilledView]] = {}
    for arm in anchor_arms:
        per_anchor: dict[str, tuple[list[Candidate], tuple[np.ndarray, np.ndarray]]] = {}
        for (a, t, names), res in zip(meta, results, strict=True):
            if a != arm:
                continue
            cands = score_candidates(masks[t.key], res, names, run.perceptual)
            per_anchor[t.key] = (cands, anchor_points(setup, masks[t.key], scene_sample))
        chosen = {
            k: int(np.argmin([c.ring + (10.0 if c.drifted else 0.0) for c in v[0]])) if v[0] else 0
            for k, v in per_anchor.items()
        }
        for _ in range(2):
            agreement_scores(per_anchor, chosen)
            chosen = {
                k: int(
                    np.argmin(
                        [
                            c.ring + AGREE_WEIGHT * (1 - c.agreement) + (10.0 if c.drifted else 0.0)
                            for c in v[0]
                        ]
                    )
                )
                if v[0]
                else 0
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


def lift_fills(run: Run, state: ArmState, fills: Sequence[FilledView]) -> None:
    region = run.setup.region()
    for fill in fills:
        splats, conf = lift_view(fill, run.depth, region, run.options.lift_stride)
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
            masks = view_masks(setup, t.camera, run.renderer, state.layer(), opt.weak_strength)
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
    run: Run, anchors: Sequence[FilledView], layer: Splats | None
) -> tuple[Callable[[], list[EditResult]], list[tuple[str, Any]], dict[str, Any]] | None:
    """6B, sent: the set of views (`build_set`) to the set filler, `set_seeds` seeds."""
    if run.set_filler is None:
        return None
    props = [t for t in run.targets if t.role == "prop"]
    context = run.targets[0].context if run.targets else {}
    clip, entries = build_set(run.setup, anchors, props, layer, run.renderer, run.options, context)
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


def weak_copies(run: Run, limit: int) -> tuple[Splats, np.ndarray]:
    """Trainable copies of the weak gaussians in the holes (the distil refines their look;
    the originals stay frozen): at most `limit`, the worst seen first. Returns them and
    their shown indices."""
    setup = run.setup
    c = setup.shown_classes()
    rows = np.flatnonzero(
        (c == fq.WEAK) & (setup.clusters.labels[setup.shown] >= 0) & ~setup.withheld[setup.shown]
    )
    if rows.size > limit:
        best = setup.quality.best[setup.shown][rows]
        rows = np.sort(rows[np.argsort(best, kind="stable")[:limit]])
    return setup.scene().take(rows), rows


def fuse(
    run: Run,
    states: dict[str, ArmState],
    distil: Callable[[dict], dict] | None,
) -> dict[str, tuple[Splats, np.ndarray, list[Camera]] | None]:
    """Each arm: carve and thin its lifted gaussians, add copies of the weak ones, distil,
    then the dataset-update rounds (every filled view rendered with the fused layer,
    refined by the editor at the round's strength, distilled again), a final carve. Returns
    per arm the layer, its confidence and the cameras that made it."""
    import distill_fill as df

    setup, opt = run.setup, run.options
    budget = max(MIN_BUDGET, int(BUDGET_SHARE * len(setup.measured)))
    out: dict[str, tuple[Splats, np.ndarray, list[Camera]] | None] = {}
    layers: dict[str, dict[str, Any]] = {}
    for arm, state in states.items():
        lifted = state.layer()
        if lifted is None:
            state.report["skipped"] = "nothing lifted"
            out[arm] = None
            continue
        conf = np.concatenate(state.confs)
        thinned = gf.thin(gf.Lifted(lifted, conf, [], []), budget)
        copies, copy_rows = weak_copies(run, max(0, budget - len(thinned.splats)) // 2)
        state.report.update(
            {
                "lifted": len(lifted),
                "carvedOnLift": state.carved,
                "thinned": len(thinned.splats),
                "weakCopies": len(copies),
            }
        )
        layers[arm] = {
            "gen": thinned.splats,
            "conf": thinned.confidence,
            "copies": copies,
            "copyRows": copy_rows,
        }
    rounds = [(opt.distill, None), *[(opt.update_distill, s) for s in opt.update_strengths]]
    for r, (iterations, strength) in enumerate(rounds):
        if strength is not None:
            _update_round(run, {a: states[a] for a in layers}, layers, strength)
        if distil is None or iterations <= 0:
            continue
        for arm, layer in layers.items():
            init = (
                Splats.concat([layer["gen"], layer["copies"]])
                if len(layer["copies"])
                else layer["gen"]
            )
            t = time.time()
            response = distil(distil_request(run, states[arm], init, iterations))
            got = df.unpack_scan(response["inferred"])
            fused = Splats(*(got[k] for k in df.KEYS))
            n = len(layer["gen"])
            layer["gen"] = fused.take(np.arange(n))
            layer["copies"] = fused.take(np.arange(n, len(fused)))
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
            gen = Splats.concat([gen, copies])
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
            [frozen, layer["gen"], *([layer["copies"]] if len(layer["copies"]) else [])]
        )
        for fill in state.fills:
            frame = run.renderer(scene, fill.camera)
            rendered = tf.to_u8(fv.unpremultiply(frame))
            edit = fill.masks.edit
            strength_map = np.where(edit, strength, 0.0).astype(np.float32)
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
            ctx = next((t.context for t in run.targets if t.key == fill.key), {})
            reqs = edit_requests(
                setup,
                f"{arm}/{fill.key}/u{strength:g}",
                masks,
                ctx,
                refs=ANCHOR_ARM[arm] == "refs",
                seeds=[fill.chosen],
                steps=opt.update_steps,
                lightning=opt.lightning,
                vae_area=opt.vae_area,
                update=True,
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
        region = (
            tf._covered(run.renderer(withheld, cam).alpha)
            if len(withheld)
            else np.zeros(photo.shape[:2], bool)
        )
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
        edge = _dilate(region, 1) & ~region
        for im in row:
            im[edge] = (255, 0, 255)
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
    """Per anchor: its context photos, what the editor was given, and each arm's fill."""
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
            photo = setup.photo(c, 768)
            if photo is not None:
                row.append(
                    gf.label_image(_cover_crop(photo, (w, h)), f"{role}: {setup.views[c].name}")
                )
        while len(row) < 4:
            row.append(np.zeros((h, w, 3), np.uint8))
        any_fill = next((by_key[a][t.key] for a in by_key if t.key in by_key[a]), None)
        if any_fill is None:
            continue
        row.append(
            gf.label_image(
                small(any_fill.masks.condition), f"{t.key} {t.elevation:g}/{t.azimuth:g} given"
            )
        )
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
    t = time.time()
    anchors = fill_anchors(run, options.arms)
    report["timings"]["anchorsS"] = round(time.time() - t, 1)
    states: dict[str, ArmState] = {}
    lifted: dict[str, ArmState] = {}
    for arm in sorted({ANCHOR_ARM[a] for a in options.arms}):
        lifted[arm] = ArmState(arm)
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
        )
    # 6B goes first (it needs only the anchors), so its GPU runs while 6A goes on.
    joint = None
    if "vace" in options.arms:
        try:
            joint = start_joint(run, anchors.get("refs", []), states["vace"].layer())
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
    report["candidates"] = {}
    for arm, state in states.items():
        entry: dict[str, Any] = {**state.report, "fills": [f.to_json() for f in state.fills]}
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
                    refs=" and two retrieved real photos"
                    if ANCHOR_ARM[arm] == "refs"
                    else " alone",
                    propagate="the rest sequentially, each view re-rendered with what is filled"
                    if arm != "vace"
                    else "the rest jointly by a video model over the set of views",
                ),
                extra={
                    "provenance": "inferred-generated",
                    "arm": arm,
                    "anchors": sum(f.role == "anchor" for f in state.fills),
                    "propagated": sum(f.role != "anchor" for f in state.fills),
                    "seedAgreement": entry["seedAgreement"],
                    "depth": depth.name,
                },
            )
            entry["layer"] = str(layer_dir)
        report["candidates"][arm] = entry
    report["calls"] = run.calls
    report["editorCalls"] = getattr(editor, "calls", None)
    t = time.time()
    if setup.held_out:
        report["heldOut"] = score_held_out(run, layers, out / "renders", width=options.score_width)
    anchor_sheet(run, anchors, out / "renders")
    before_after(run, layers, out / "renders")
    report["timings"]["scoreS"] = round(time.time() - t, 1)
    report["timings"]["totalS"] = round(time.time() - started, 1)
    gf.write_json(out / "report.json", report)
    return report


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
    return o


def smoke_request() -> EditRequest:
    """A tiny request (a gradient with a magenta box, one reference) that checks the editor
    runs before a job sends it hundreds; sent first, so its load overlaps the setup."""
    import anchor_models as am

    h, w = 144, 256
    yy, xx = np.mgrid[0:h, 0:w]
    render = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 128)], -1).astype(np.uint8)
    strength = np.zeros((h, w), np.float32)
    strength[48:96, 96:160] = 1.0
    condition = render.copy()
    condition[strength >= 1.0] = am.KEY
    return EditRequest(
        "smoke-s1", condition, render, strength, [render], prompt_for("a gradient", 1), 1, 4, True
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
