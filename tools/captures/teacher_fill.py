"""Teacher B: what a scan never saw, filled in by an image model and lifted into a layer of
its own, labelled inferred.

LIVING_ENGINE.md section 3 and ADR 0009: a scan is right from where it was seen; from
anywhere else the globe now fades it (view cones), which leaves holes. This fills them --
in a **separate tileset**, never in the measured one, carrying its own view cones so it is
drawn only from the side it was made for, and its own `extras.evidence` so the viewer can
say so. Every stage runs on the CPU with the image model behind one interface (`Filler`):

1. **views** (`plan_views`) -- virtual cameras: at eye height above the observers (the
   view cones' finest-detail region, or the real cameras when known), or on a ring outside
   the scan, aimed at what is to be filled;
2. **conditioning** (`condition`) -- per view (rendered by `splat_render.render` on the CPU,
   or gsplat on a GPU: `make_renderer`), the scan rendered with each gaussian faded by
   its view cone (what was seen from here), the full render (what is there at all), and the
   **mask**: pixels the scan covers but only with what was never seen from this side (or,
   for the drop test, the pixels a dropped region covered);
3. **fill** -- `Filler.fill(rgb, mask)`: images with the masked pixels painted.
   `InpaintFiller` is the CPU stand-in (OpenCV Telea inpainting); NVIDIA Fixer and Cosmos are
   Fillers run on a GPU (infra/modal/world_models.py);
4. **gate** -- a fill that changed the unmasked pixels by more than `GATE_PSNR_DB` is
   refused: it was told what is there and repainted it (a filler that re-renders the whole
   frame is held to its layout: `GATE_FULL_RENDER_PSNR_DB` on blurred frames);
5. **lift** (`lift`) -- every `stride`-th masked pixel becomes a flat gaussian facing its
   camera, at the depth the scan has there (or the depth inpainted from around the hole;
   for a dropped region, always interpolated across the hole from around it),
   with the filled colour and an opacity scaled by **confidence**: how far the pixel is
   from anything measured in its view (`exp(-d / CONFIDENCE_PX)`);
6. **package** (`package_inferred`) -- the gaussians as their own tileset in the scan's frame,
   with view cones whose observers are the virtual cameras and `extras.evidence`.

`drop_and_fill` is the evaluation that needs no photographs: remove a region the capture
did see, fill it, and compare with what was there -- in the views used to fill it and, after
lifting, in a held-out view (S1/S6 in LIVING_ENGINE.md, without a GPU).
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

import view_cones as vc
from splat_render import Camera, Frame, Splats, render, save_ply

#: What every view is rendered with: `(splats, camera, *, opacity_scale=None) -> Frame`.
#: `splat_render.render` (the CPU's point samples) or `splat_render.GsplatRenderer` (gsplat
#: on a GPU, what a viewer draws and what Fixer is trained on); see `make_renderer`.
Renderer = Callable[..., Frame]


def make_renderer(spec: str = "cpu") -> Renderer:
    """`cpu` (`splat_render.render`) or `gsplat` (`splat_render.GsplatRenderer`, CUDA)."""
    if spec == "cpu":
        return render
    if spec == "gsplat":
        from splat_render import GsplatRenderer

        return GsplatRenderer()
    raise ValueError(f"renderer {spec!r}: 'cpu' or 'gsplat'")


__all__ = [
    "Conditioning",
    "Filler",
    "InpaintFiller",
    "clean_mask",
    "condition",
    "describe_surroundings",
    "drop_and_fill",
    "fill_hole",
    "fill_scan",
    "lift",
    "link_inferred",
    "make_filler",
    "make_renderer",
    "package_inferred",
    "plan_views",
    "read_instances",
    "refine",
]

#: A fill that changed the pixels it was not asked to fill by more than this (PSNR) is refused.
GATE_PSNR_DB = 25.0
#: A filler that re-renders the whole frame (`reads_full_render`, NVIDIA Fixer) changes every
#: pixel's texture -- the CPU renderer's point samples become a photograph's grain -- so it is
#: gated on the frame's layout instead: both blurred by `GATE_BLUR_PX`, and this PSNR. On
#: the first GPU runs (spool, pumpkin, yard from outside) Fixer kept the scene at 20-30 dB
#: blurred and 12-17 dB unblurred; an inverted frame scores under 10 either way.
GATE_FULL_RENDER_PSNR_DB = 20.0
GATE_BLUR_PX = 2.0
#: Untouched pixels score this rather than infinity, so reports stay JSON.
GATE_CAP_DB = 99.0
#: Confidence halves about every this many pixels from the nearest measured pixel.
CONFIDENCE_PX = 12.0
#: Below this coverage of what was seen from here, a pixel the scan covers is to be filled.
SEEN_ALPHA = 0.5
#: ...and only where the scan does cover it at least this much.
COVERED_ALPHA = 0.5
#: Mask speckle smaller than this (pixels) is closed over or opened away.
MASK_CLEAN_PX = 5
#: Eye height above the observer points, metres, for camera-free scans.
EYE_HEIGHT_M = 1.6


class Filler(Protocol):
    """Paints the masked pixels of a view. `rgb` (h, w, 3) uint8, `mask` (h, w) bool; returns
    one or more fills, each (h, w, 3) uint8.

    `rgb` is the scan faded by its view cones (masked pixels empty) unless the filler sets
    `reads_full_render`: then it is the whole scan as rendered from there, unseen side and
    all -- what an artifact-fixing model (NVIDIA Fixer) is trained to clean.

    Optional, read with `getattr`: `context` -- a filler that has it (a generative inpainter,
    `world_model_client.GenerativeFiller`) is told what surrounds what it fills
    (`describe_surroundings`: a prompt and what not to paint) before its views; and
    `chain_views` -- `fill_hole` then fills the views in turn, each shown what the earlier
    ones filled, lifted and re-rendered, with only the rest masked (`_chained_fill`)."""

    name: str

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]: ...


@dataclass
class InpaintFiller:
    """The CPU stand-in: OpenCV's Telea inpainting. Deterministic, local, and plausible only
    for small holes -- a floor for the image models to beat, not a fill to ship."""

    radius: int = 5
    name: str = "opencv-telea"

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        import cv2

        return [cv2.inpaint(rgb, mask.astype(np.uint8) * 255, self.radius, cv2.INPAINT_TELEA)]


def to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def seen_weights(splats: Splats, camera: Camera, grid: vc.ConeGrid | None) -> np.ndarray:
    """Each gaussian's view-cone weight seen from `camera` (1 everywhere without a grid)."""
    if grid is None:
        return np.ones(len(splats))
    texels = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, splats.positions)
    view = splats.positions - camera.centre
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    return vc.visibility(texels, view)


@dataclass
class Conditioning:
    camera: Camera
    seen: Frame
    full: Frame
    mask: np.ndarray
    #: Depth for every masked pixel: the scan's where it has any, else inpainted around it.
    depth: np.ndarray
    #: Distance in pixels from each masked pixel to the nearest unmasked, measured one.
    distance: np.ndarray


def condition(
    splats: Splats,
    camera: Camera,
    grid: vc.ConeGrid | None,
    mask: np.ndarray | None = None,
    seen_opacity: np.ndarray | None = None,
    *,
    hole_depth: str = "scan",
    surround: Splats | None = None,
    renderer: Renderer = render,
) -> Conditioning:
    """What `camera` is to be told and asked: the seen render, and the mask to fill.

    The depth a masked pixel is lifted at: `scan`, the scan's own there (the unseen side of
    what it covers), inpainted from around the hole where it has none; `surround`, always
    interpolated across the mask from the measured pixels around it (inverse depth, which is
    affine across a plane) -- for a region that was removed (the drop test), where what the
    scan shows through the hole is what was behind it. `surround` (gaussians) narrows "around
    it" to what they cover: the scan just outside the removed region, so a hole whose image
    neighbours are far background (a branch against the sky) still gets its own surface."""
    import cv2

    weights = seen_weights(splats, camera, grid) if seen_opacity is None else seen_opacity
    seen = renderer(splats, camera, opacity_scale=weights)
    full = renderer(splats, camera)
    if mask is None:
        mask = clean_mask((seen.alpha < SEEN_ALPHA) & (full.alpha >= COVERED_ALPHA))
    depth = np.where(np.isfinite(full.depth), full.depth, np.nan)
    known = np.isfinite(seen.depth) & (seen.alpha >= SEEN_ALPHA)
    if hole_depth == "surround":
        source, around = seen.depth, known & ~mask
        if surround is not None and len(surround):
            shell = renderer(surround, camera)
            near = np.isfinite(shell.depth) & (shell.alpha >= SURROUND_ALPHA) & ~mask
            if near.any():
                source, around = shell.depth, near
        if mask.any() and around.any():
            depth = np.where(mask, _interpolate_depth(source, around, mask), depth)
        else:
            depth = np.where(mask, np.nan, depth)
    elif hole_depth != "scan":
        raise ValueError(f"hole_depth {hole_depth!r}: 'scan' or 'surround'")
    missing = mask & ~np.isfinite(depth)
    if missing.any() and known.any():
        log_d = np.where(known, np.log(np.where(known, seen.depth, 1.0)), 0).astype(np.float32)
        filled = cv2.inpaint(log_d, (~known).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        depth = np.where(missing, np.exp(filled), depth)
    distance = cv2.distanceTransform((mask | ~known).astype(np.uint8), cv2.DIST_L2, 5)
    return Conditioning(camera, seen, full, mask, depth, distance)


#: `_interpolate_depth` reads the measured depth in a band this wide (pixels) around a hole
#: (wider when the band holds nothing).
HOLE_RING_PX = 6
#: Coverage at which a `surround` render's depth counts (it is a thin shell: sparse).
SURROUND_ALPHA = 0.2


def _interpolate_depth(depth: np.ndarray, known: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Depth everywhere, interpolated from the `known` pixels: inverse depth smoothed in from
    the ring of known pixels just outside the `hole` (normalised convolution at growing
    scales), so a hole the width of a region gets the surface around it, not what lies
    behind. Known pixels keep their depth."""
    import cv2

    gap = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
    ring = known & (gap <= HOLE_RING_PX)
    for wider in (4, 16):
        if ring.any():
            break
        ring = known & (gap <= wider * HOLE_RING_PX)
    if not ring.any():
        ring = known
    inv = np.where(ring, 1.0 / np.where(ring, depth, 1.0), 0.0).astype(np.float32)
    weight = ring.astype(np.float32)
    out = np.full(depth.shape, np.nan, np.float32)
    todo = ~known
    sigma = 2.0
    while todo.any() and sigma < 4 * max(depth.shape):
        w = cv2.GaussianBlur(weight, (0, 0), sigma)
        v = cv2.GaussianBlur(inv * weight, (0, 0), sigma)
        ok = todo & (w > 1e-3)
        out[ok] = v[ok] / w[ok]
        todo &= ~ok
        sigma *= 2.0
    with np.errstate(divide="ignore"):
        return np.where(known, depth, 1.0 / np.maximum(out, 1e-9)).astype(np.float64)


def clean_mask(mask: np.ndarray, px: int = MASK_CLEAN_PX) -> np.ndarray:
    """A mask without the renderer's sampling speckle: closed, then opened, by `px`."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel).astype(bool)


def plan_views(
    grid: vc.ConeGrid | None,
    targets: np.ndarray,
    *,
    cameras: Sequence[np.ndarray] | None = None,
    count: int = 4,
    mode: str = "near",
    ring_radius_m: float | None = None,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 60.0,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    distance_m: float | None = None,
) -> list[Camera]:
    """`count` cameras looking at the centroid of `targets`.

    `near`: from the direction of the observers nearest the targets (the real camera
    centres when given, else the view cones' observer points lifted `EYE_HEIGHT_M`), spread
    apart, at `distance_m` from the target when given (else where the observers are) -- the
    holes a capture left close to where it walked. `ring`: from a ring around the targets
    `ring_radius_m` out, a little above them -- the sides it never walked to."""
    target = np.asarray(targets, np.float64).reshape(-1, 3).mean(axis=0)
    eyes: list[np.ndarray] = []
    if mode == "ring":
        radius = ring_radius_m or 10.0
        for k in range(count):
            a = 2 * math.pi * k / count
            eyes.append(
                target + np.array([radius * math.cos(a), radius * math.sin(a), 0.3 * radius])
            )
    else:
        if cameras is not None and len(cameras):
            pool = np.asarray(cameras, np.float64).reshape(-1, 3)
        elif grid is not None and grid.observers.shape[0]:
            pool = grid.observers + np.asarray(up) * EYE_HEIGHT_M
        else:
            raise ValueError("no observers to stand at: give cameras or a view-cone grid")
        d = np.linalg.norm(pool - target, axis=1)
        near = pool[np.argsort(d)[: max(count * 8, count)]]
        # Farthest-point spread over the nearest few, so the views see the target from apart.
        chosen = [near[0]]
        while len(chosen) < min(count, near.shape[0]):
            gaps = np.min(np.linalg.norm(near[:, None] - np.array(chosen)[None], axis=2), axis=1)
            chosen.append(near[int(gaps.argmax())])
        eyes = list(chosen)
        if distance_m is not None:
            eyes = [
                target + distance_m * (e - target) / max(float(np.linalg.norm(e - target)), 1e-9)
                for e in eyes
            ]
    return [
        Camera.look_at(eye, target, fov_deg=fov_deg, width=width, height=height, up=up)
        for eye in eyes
        if np.linalg.norm(eye - target) > 1e-3
    ]


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR (dB) of two uint8 images over `mask` (everything when None)."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    if mask is not None:
        diff = diff[mask]
    mse = float(np.mean(diff**2)) if diff.size else 0.0
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM (grey, 7-px Gaussian windows) over `mask`."""
    import cv2

    x = a.astype(np.float64) @ [0.299, 0.587, 0.114]
    y = b.astype(np.float64) @ [0.299, 0.587, 0.114]
    blur = lambda v: cv2.GaussianBlur(v, (7, 7), 1.5)
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s[mask].mean()) if mask.any() else 1.0


@dataclass
class Filled:
    conditioning: Conditioning
    rgb: np.ndarray  # the fill kept (uint8)
    gate_psnr_db: float
    accepted: bool


def fill_views(conds: Sequence[Conditioning], filler: Filler) -> list[Filled]:
    """Each view filled and gated: the first fill whose unmasked pixels kept their render."""
    out = []
    full = bool(getattr(filler, "reads_full_render", False))
    threshold = GATE_FULL_RENDER_PSNR_DB if full else GATE_PSNR_DB
    for cond in conds:
        given = to_u8(cond.seen.rgb)
        shown = to_u8(cond.full.rgb) if full else given
        best: Filled | None = None
        for candidate in filler.fill(shown, cond.mask):
            if candidate.shape != given.shape:
                import cv2

                candidate = cv2.resize(candidate, (given.shape[1], given.shape[0]))
            kept = cond.seen.alpha >= SEEN_ALPHA
            # Held to what it was shown: the full render, when it reads that.
            a, b = (_blur(candidate), _blur(shown)) if full else (candidate, given)
            score = min(psnr(a, b, kept & ~cond.mask), GATE_CAP_DB)
            if best is None or score > best.gate_psnr_db:
                best = Filled(cond, candidate, score, score >= threshold)
        assert best is not None
        out.append(best)
    return out


#: A chained view reuses an earlier view's fill where that, re-rendered, covers this much.
CHAIN_ALPHA = 0.5


def _chained_fill(
    conds: Sequence[Conditioning], filler: Filler, renderer: Renderer, stride: int
) -> list[Filled]:
    """`fill_views` one view at a time, each shown the fill of the views before it: their
    accepted fills lifted (onto the same surface every view lifts to) and rendered from
    here, painted into the hole wherever they cover it, and only the rest of the hole left
    for the filler. What one view invented the next one sees rather than invents again.
    Each `Filled` keeps its view's whole mask, so lifting and distilling treat it as any."""
    out: list[Filled] = []
    for cond in conds:
        ask = cond
        if any(f.accepted for f in out):
            painted, _ = lift(out, stride=stride, hole_scaled=True)
            solid = Splats(
                painted.positions,
                painted.rotations,
                painted.scales,
                painted.colours,
                np.full(len(painted), 0.95),
            )
            frame = renderer(solid, cond.camera)
            reuse = cond.mask & (frame.alpha >= CHAIN_ALPHA)
            if reuse.any():
                colour = frame.rgb / np.maximum(frame.alpha, 1e-6)[..., None]
                rgb = np.where(reuse[..., None], np.clip(colour, 0, 1), cond.seen.rgb)
                seen = Frame(rgb, cond.seen.depth, cond.seen.alpha, cond.seen.label,
                             cond.seen.purity)  # fmt: skip
                ask = Conditioning(
                    cond.camera, seen, seen, cond.mask & ~reuse, cond.depth, cond.distance
                )
        (f,) = fill_views([ask], filler)
        out.append(Filled(cond, f.rgb, f.gate_psnr_db, f.accepted))
    return out


#: `describe_surroundings`: the labels named are those carrying at least this share of the
#: strongest one's weight, at most `SURROUNDINGS_LABELS` of them.
SURROUNDINGS_SHARE = 0.1
SURROUNDINGS_LABELS = 4
#: How a prompt is phrased for what a hole's views look down on, and for a scan from outside.
SURROUNDINGS_VIEWS = {
    "above": "{labels}: a top-down close-up photograph of the ground, a seamless natural "
    "texture, evenly lit, nothing lying on it, sharp detail",
    "outside": "{labels}, seen from outside, natural photograph, daylight, sharp detail",
}


def describe_surroundings(
    instances: Sequence[dict],
    low: np.ndarray,
    high: np.ndarray,
    *,
    object_ids: Sequence[int] = (),
    exclude_ids: Sequence[int] = (),
    view: str = "above",
    top_tags: int = 3,
) -> dict[str, object]:
    """What a generative filler is told about a region (`low`..`high`, the scan's local ENU):
    the tags of the instances (instances.json) that reach into its horizontal box and start
    no higher than its top, each tag weighted by its score, the instance's gaussians and the
    share of its own footprint inside the box (a scene-wide instance does not drown the
    patch at hand); the strongest few named in the prompt. `object_ids` (a removed object)
    and `exclude_ids` (its fragments) say nothing; the object's own top labels become the
    negative prompt and are never named: a hole is not painted with what was taken out of it."""
    silent = {int(k) for k in (*object_ids, *exclude_ids)}
    removed: list[str] = []
    for inst in instances:
        if int(inst["id"]) in {int(k) for k in object_ids}:
            removed += [str(t["label"]) for t in inst.get("tags", [])[:2]]
    removed = list(dict.fromkeys(removed))
    weights: dict[str, float] = {}
    for inst in instances:
        if int(inst["id"]) in silent:
            continue
        a = np.asarray(inst["bounds"]["min"], float)
        b = np.asarray(inst["bounds"]["max"], float)
        if a[2] > high[2]:
            continue
        overlap = np.clip(np.minimum(b[:2], high[:2]) - np.maximum(a[:2], low[:2]), 0, None)
        share = float(np.prod(overlap)) / float(np.prod(np.maximum(b[:2] - a[:2], 1e-3)))
        if share <= 0:
            continue
        for t in inst.get("tags", [])[:top_tags]:
            label = str(t["label"])
            gain = float(t["score"]) * int(inst.get("splats", 1)) * share
            weights[label] = weights.get(label, 0.0) + gain
    ranked = sorted(
        ((k, w) for k, w in weights.items() if k not in removed), key=lambda kv: (-kv[1], kv[0])
    )
    labels = [k for k, w in ranked[:SURROUNDINGS_LABELS] if w >= SURROUNDINGS_SHARE * ranked[0][1]]
    named = (
        " and ".join([", ".join(labels[:-1]), labels[-1]]) if len(labels) > 1 else "".join(labels)
    )
    prompt = SURROUNDINGS_VIEWS[view].format(labels=named or "the ground")
    return {
        "prompt": prompt[0].upper() + prompt[1:],
        "negative": ", ".join(removed),
        "labels": labels,
        "weights": {k: round(w, 2) for k, w in ranked[:8]},
    }


def read_instances(tileset: Path) -> list[dict]:
    """The instances a tileset declares (`root.extras.instances`), or none."""
    document = json.loads(tileset.read_text(encoding="utf-8"))
    uri = document["root"].get("extras", {}).get("instances", {}).get("uri")
    if not uri or not (tileset.parent / uri).exists():
        return []
    return json.loads((tileset.parent / uri).read_text(encoding="utf-8")).get("instances", [])


def _blur(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(rgb, (0, 0), GATE_BLUR_PX)


def _disc_rotations(normals: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z) turning local +z to each normal: discs facing the camera."""
    z = np.array([0.0, 0.0, 1.0])
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    s = np.linalg.norm(axis, axis=1)
    c = n @ z
    half = np.arctan2(s, c) / 2
    axis = np.where(s[:, None] > 1e-9, axis / np.maximum(s, 1e-12)[:, None], [1.0, 0.0, 0.0])
    return np.column_stack([np.cos(half), axis * np.sin(half)[:, None]])


def lift(
    filled: Sequence[Filled], stride: int = 2, *, hole_scaled: bool = False
) -> tuple[Splats, np.ndarray]:
    """The accepted fills' masked pixels as gaussians, and each one's confidence.

    `hole_scaled`: confidence halves every `CONFIDENCE_PX` or every half the view's deepest
    distance into its mask, whichever is longer -- a hole that must close (an object moved
    away) is not left transparent in its middle for being wide."""
    parts, confidences = [], []
    for f in filled:
        if not f.accepted:
            continue
        cond, camera = f.conditioning, f.conditioning.camera
        mask = cond.mask & np.isfinite(cond.depth)
        sel = np.zeros_like(mask)
        sel[::stride, ::stride] = mask[::stride, ::stride]
        v, u = np.nonzero(sel)
        if v.size == 0:
            continue
        z = cond.depth[v, u]
        local = np.column_stack(
            [
                (u + 0.5 - camera.width / 2) * z / camera.focal,
                (v + 0.5 - camera.height / 2) * z / camera.focal,
                z,
            ]
        )
        positions = camera.centre + local @ camera.rotation
        footprint = stride * z / camera.focal
        scales = np.column_stack([0.6 * footprint, 0.6 * footprint, 0.1 * footprint])
        halving = CONFIDENCE_PX
        if hole_scaled:
            halving = max(halving, 0.5 * float(cond.distance[mask].max()))
        confidence = np.exp(-cond.distance[v, u] / halving * math.log(2))
        parts.append(
            Splats(
                positions,
                _disc_rotations(camera.centre - positions),
                scales,
                f.rgb[v, u].astype(np.float64) / 255.0,
                0.95 * confidence,
            )
        )
        confidences.append(confidence)
    if not parts:
        empty = np.zeros((0, 3))
        return Splats(empty, np.zeros((0, 4)), empty, empty, np.zeros(0)), np.zeros(0)
    return Splats.concat(parts), np.concatenate(confidences)


@dataclass
class DropReport:
    region: dict[str, object]
    filler: str
    views: list[dict[str, float]] = field(default_factory=list)
    held_out: dict[str, float] = field(default_factory=dict)
    lifted: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "region": self.region,
            "filler": self.filler,
            "views": self.views,
            "heldOut": self.held_out,
            "lifted": self.lifted,
        }


def drop_and_fill(
    splats: Splats,
    centre: np.ndarray,
    half_size_m: float,
    filler: Filler,
    grid: vc.ConeGrid | None,
    *,
    views: int = 4,
    width: int = 480,
    height: int = 360,
    stride: int = 2,
    save_dir: Path | None = None,
    hole_depth: str = "surround",
    renderer: Renderer = render,
) -> DropReport:
    """Removes the gaussians in a cube the capture saw, fills it from `views` cameras near
    the observers, and scores the fill against what was there: per fill view (PSNR, SSIM on
    the dropped region's pixels, against leaving the hole), and -- after lifting -- from one
    more view the fill never used. The fill is lifted at the depth interpolated across the
    hole from the scan just outside the region (`hole_depth="surround"`, the gaussians within
    `SURROUND_SCALE` times its half size): where the dropped surface was, so the held-out
    view sees the fill in front of what was behind it."""
    centre = np.asarray(centre, np.float64)
    inside = np.all(np.abs(splats.positions - centre) <= half_size_m, axis=1)
    kept, dropped = splats.take(np.flatnonzero(~inside)), splats.take(np.flatnonzero(inside))
    # What the capture has just outside the region: the surface the hole is lifted onto.
    reach = np.all(np.abs(splats.positions - centre) <= SURROUND_SCALE * half_size_m, axis=1)
    shell = splats.take(np.flatnonzero(reach & ~inside))
    # Far enough that the region is about a third of the frame, from where it was seen.
    distance = 3.0 * half_size_m / math.tan(math.radians(60.0) / 2)
    cameras = plan_views(
        grid, centre[None], count=views + 1, width=width, height=height, distance_m=distance
    )
    held, used = cameras[-1], cameras[:-1]
    report = DropReport(
        {
            "centre": centre.round(3).tolist(),
            "halfSizeM": half_size_m,
            "dropped": int(inside.sum()),
        },
        filler.name,
    )
    conds = []
    for camera in used:
        truth = to_u8(renderer(splats, camera).rgb)
        hole = _hole(renderer(dropped, camera).alpha)
        cond = condition(
            kept,
            camera,
            None,
            mask=hole,
            seen_opacity=np.ones(len(kept)),
            hole_depth=hole_depth,
            surround=shell,
            renderer=renderer,
        )
        conds.append((cond, truth))
    filled = fill_views([c for c, _ in conds], filler)
    for k, (f, (cond, truth)) in enumerate(zip(filled, conds, strict=True)):
        if not cond.mask.any():
            continue
        hole_render = to_u8(cond.seen.rgb)
        if save_dir is not None:
            _save_strip(save_dir / f"view{k}.png", [truth, hole_render, f.rgb], cond.mask)
        report.views.append(
            {
                "maskPx": int(cond.mask.sum()),
                "psnrFill": round(psnr(f.rgb, truth, cond.mask), 2),
                "psnrHole": round(psnr(hole_render, truth, cond.mask), 2),
                "ssimFill": round(ssim(f.rgb, truth, cond.mask), 4),
                "ssimHole": round(ssim(hole_render, truth, cond.mask), 4),
                "gatePsnr": round(f.gate_psnr_db, 2),
            }
        )
    lifted, _ = lift(filled, stride=stride)
    report.lifted = len(lifted)
    truth = to_u8(renderer(splats, held).rgb)
    hole = _hole(renderer(dropped, held).alpha)
    if hole.any():
        without = to_u8(renderer(kept, held).rgb)
        with_fill = (
            to_u8(renderer(Splats.concat([kept, lifted]), held).rgb) if len(lifted) else without
        )
        if save_dir is not None:
            _save_strip(save_dir / "held-out.png", [truth, without, with_fill], hole)
        report.held_out = {
            "maskPx": int(hole.sum()),
            "psnrFill": round(psnr(with_fill, truth, hole), 2),
            "psnrHole": round(psnr(without, truth, hole), 2),
            "ssimFill": round(ssim(with_fill, truth, hole), 4),
            "ssimHole": round(ssim(without, truth, hole), 4),
        }
    return report


#: The drop test's lift surface: the scan within this many half sizes of the region's centre.
SURROUND_SCALE = 2.0


def _hole(alpha: np.ndarray) -> np.ndarray:
    """Where a dropped region covered the view: its coverage, closed over the gaps the
    renderer's samples leave (3 px)."""
    import cv2

    raw = (alpha >= 0.25).astype(np.uint8)
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)


#: A pixel of a removed object's silhouette needs filling when what the scan shows there now
#: lies more than this share of its depth behind the surface the object stood on (or nothing
#: covers it): the view sees through where the object stood.
SEE_THROUGH = 0.05
#: Hole views look down on where the object stood at least this steeply (degrees): its
#: footprint is what the capture could not see.
HOLE_ELEVATION_DEG = 35.0
#: ...from far enough that the object spans about this share of the frame's height.
HOLE_FRAME_SHARE = 0.45
#: The surface a removed object stood on: the scan around its footprint (`SURROUND_SCALE`
#: half sizes out) no higher than this share of its height above its base.
HOLE_BASE_SHARE = 0.25
#: Coverage is judged on the render's alpha dilated by this (px): the CPU renderer's samples
#: leave a measured surface speckled, and a speck is no hole.
HOLE_COVER_PX = 3
#: The footprint a hole is filled over: the object's ground plan padded by this share of its size.
HOLE_FOOTPRINT_PAD = 0.25
HOLE_RULE = (
    "the hole a split object left (split_objects.py): pixels of virtual views around where it "
    "stood that now see through the surface it stood on (a plane fitted to the scan around its "
    "footprint), filled by an image model, gated on the other pixels, lifted onto that plane as "
    "discs facing their camera, opacity by distance from measured pixels"
)


def hole_views(
    grid: vc.ConeGrid | None,
    removed: Splats,
    count: int,
    *,
    width: int = 480,
    height: int = 360,
    fov_deg: float = 60.0,
) -> list[Camera]:
    """`count` cameras on where `removed` stood: from the observers' side when the grid has
    observers (else a ring), at least `HOLE_ELEVATION_DEG` above it, looking at its lower
    part, at a distance where it fills `HOLE_FRAME_SHARE` of the frame."""
    low, high = removed.positions.min(axis=0), removed.positions.max(axis=0)
    size = float(np.max(high - low))
    target = np.array(
        [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] + 0.25 * (high - low)[2]]
    )
    distance = size / HOLE_FRAME_SHARE / (2 * math.tan(math.radians(fov_deg) / 2))
    distance = max(distance, 2.0 * size)
    if grid is not None and grid.observers.shape[0]:
        eyes = [c.centre for c in plan_views(grid, target[None], count=count, distance_m=distance)]
    else:
        eyes = []
    while len(eyes) < count:  # a ring makes up what the observers do not
        a = 2 * math.pi * len(eyes) / count + 0.3
        eyes.append(target + distance * np.array([math.cos(a), math.sin(a), 0.0]))
    cameras = []
    lowest = math.radians(HOLE_ELEVATION_DEG)
    for eye in eyes:
        d = np.asarray(eye, np.float64) - target
        flat = float(np.hypot(d[0], d[1]))
        azimuth = math.atan2(d[1], d[0]) if flat > 1e-9 else 0.0
        elevation = max(math.atan2(d[2], flat), lowest)
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        cameras.append(
            Camera.look_at(
                target + distance * direction, target, fov_deg=fov_deg, width=width, height=height
            )
        )
    return cameras


def support_plane(points: np.ndarray, rounds: int = 4) -> tuple[np.ndarray, float] | None:
    """The plane `n . x = c` (unit `n`, upward) through `points`, least squares with the
    farthest fifth dropped each round: the ground under a removed object, without the tufts
    and stems on it. None for fewer than 3 points."""
    pts = np.asarray(points, np.float64)
    if pts.shape[0] < 3:
        return None
    keep = np.ones(pts.shape[0], bool)
    normal, offset = np.array([0.0, 0.0, 1.0]), float(np.median(pts[:, 2]))
    for _ in range(rounds):
        centre = pts[keep].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[keep] - centre, full_matrices=False)
        normal = vt[-1] if vt[-1][2] >= 0 else -vt[-1]
        offset = float(normal @ centre)
        residual = np.abs(pts @ normal - offset)
        if keep.sum() < 15:
            break
        keep = residual <= np.quantile(residual[keep], 0.8)
    return normal, offset


def plane_depth(camera: Camera, normal: np.ndarray, offset: float) -> np.ndarray:
    """Per pixel, the camera depth (along its axis) at which its ray meets the plane; NaN
    where it does not (parallel, or behind the camera)."""
    rays = camera.rays()
    along = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - float(normal @ camera.centre)) / along
    z = t * (rays @ camera.rotation[2])
    return np.where(np.isfinite(z) & (t > 0), z, np.nan)


def _covered(alpha: np.ndarray) -> np.ndarray:
    import cv2

    kernel = np.ones((HOLE_COVER_PX, HOLE_COVER_PX), np.uint8)
    return cv2.dilate(alpha.astype(np.float32), kernel) >= COVERED_ALPHA


def see_through(full: Frame, silhouette: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Of a removed object's `silhouette`, the pixels that see through the `surface` it stood
    on (`plane_depth`): nothing covers them now, or what does lies `SEE_THROUGH` behind it."""
    on = np.isfinite(surface)
    limit = np.where(on, surface, np.inf) * (1 + SEE_THROUGH)
    behind = np.isfinite(full.depth) & (full.depth > limit)
    return silhouette & on & (~_covered(full.alpha) | behind)


def _hole_conditioning(
    kept: Splats,
    removed: Splats,
    camera: Camera,
    plane: tuple[np.ndarray, float],
    renderer: Renderer,
) -> tuple[Conditioning, np.ndarray]:
    """One hole view: the scan without the object rendered, the pixels that see through
    where it stood as the mask, the plane's depth there to lift at; and its silhouette."""
    import cv2

    silhouette = _hole(renderer(removed, camera).alpha)
    full = renderer(kept, camera)
    surface = plane_depth(camera, *plane)
    # Only where the plane is under the object (its footprint, padded): what lies beyond it
    # was not the object's to hide.
    low, high = removed.positions.min(axis=0), removed.positions.max(axis=0)
    pad = HOLE_FOOTPRINT_PAD * float(np.max(high - low))
    hit = (
        camera.centre + camera.rays() * (surface / (camera.rays() @ camera.rotation[2]))[..., None]
    )
    with np.errstate(invalid="ignore"):
        under = np.all((hit[..., :2] >= low[:2] - pad) & (hit[..., :2] <= high[:2] + pad), axis=-1)
    mask = clean_mask(see_through(full, silhouette, np.where(under, surface, np.nan)))
    depth = np.where(mask, surface, np.where(np.isfinite(full.depth), full.depth, np.nan))
    known = np.isfinite(full.depth) & (full.alpha >= SEEN_ALPHA)
    distance = cv2.distanceTransform((mask | ~known).astype(np.uint8), cv2.DIST_L2, 5)
    shown = _extend_background(full, mask)
    return Conditioning(camera, shown, shown, mask, depth, distance), silhouette


def _extend_background(frame: Frame, mask: np.ndarray) -> Frame:
    """`frame` with the hole and the empty pixels around it (nothing rendered there: the
    scan's edge) painted from the covered ones (Telea), so a filler is not shown the void as
    context -- and one that re-renders the whole frame (NVIDIA Fixer, an artifact cleaner,
    not an inpainter: on the pumpkin it kept an empty hole black) starts from a rough fill
    it can clean. Coverage and depth stay as rendered."""
    import cv2

    empty = ~_covered(frame.alpha) & ~mask
    todo = empty | mask
    if not todo.any() or todo.all():
        return frame
    painted = cv2.inpaint(to_u8(frame.rgb), todo.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    rgb = np.where(todo[..., None], painted.astype(np.float64) / 255.0, frame.rgb)
    return Frame(rgb, frame.depth, frame.alpha, frame.label, frame.purity)


def _coverage(
    renderer: Renderer, kept: Splats, lifted: Splats, camera: Camera, through: np.ndarray
) -> dict[str, float]:
    """How much of the see-through pixels the scan covers without and with the fill."""
    before = renderer(kept, camera).alpha
    after = renderer(Splats.concat([kept, lifted]), camera).alpha if len(lifted) else before
    n = int(through.sum())

    def share(alpha: np.ndarray) -> float:
        return round(float(_covered(alpha)[through].mean()), 4) if n else 1.0

    return {"throughPx": n, "coveredBefore": share(before), "coveredAfter": share(after)}


def fill_hole(
    kept: Splats,
    removed: Splats,
    filler: Filler,
    grid: vc.ConeGrid | None,
    *,
    views: int = 6,
    width: int = 480,
    height: int = 360,
    stride: int = 2,
    renderer: Renderer = render,
    save_dir: Path | None = None,
    distill_iterations: int = 0,
    distill_runner: Callable[[dict], dict] | None = None,
    context: dict[str, object] | None = None,
) -> tuple[Splats, np.ndarray, list[Camera], dict[str, object]]:
    """The hole `removed` (taken out of the scan, now `kept`) leaves, filled: `drop_and_fill`
    for a region that is gone for good. The surface it stood on is a plane fitted to the scan
    around its footprint (`support_plane`); from `views` cameras around where it stood
    (`hole_views`, and one more held out), the pixels of its silhouette that now see through
    that plane (`see_through`: the footprint nobody saw, the void behind) are filled, gated,
    lifted onto the plane and optionally distilled. `context` (`describe_surroundings`) is
    handed to a filler that reads one (a generative inpainter's prompt); a filler with
    `chain_views` fills the views in turn (`_chained_fill`).

    Returns the lifted gaussians, their confidence, the cameras that made them and a report:
    per view the silhouette and see-through pixels and the gate, and in the held-out view how
    much of what sees through the scan covers before and after the fill. `save_dir`: per view
    truth | hole | fill, and `held-out.png`: the scan as it was | without the object | with
    the fill | the object moved aside (`MOVE_SHARE` of its size), the fill showing."""
    low, high = removed.positions.min(axis=0), removed.positions.max(axis=0)
    centre, half = (low + high) / 2, float(np.max(high - low)) / 2
    # The surface it stood on: the scan around its footprint, no higher than its lower
    # quarter (the frame is east-north-up) -- not the neighbours' crowns beside it.
    p = kept.positions
    around = (
        np.all(np.abs(p[:, :2] - centre[:2]) <= SURROUND_SCALE * half, axis=1)
        & (p[:, 2] <= low[2] + HOLE_BASE_SHARE * (high[2] - low[2]))
        & (p[:, 2] >= low[2] - half)
    )
    plane = support_plane(p[around])
    report: dict[str, object] = {"filler": filler.name, "removed": len(removed)}
    if plane is None:
        empty = Splats(*(np.zeros((0, k)) for k in (3, 4, 3, 3)), np.zeros(0))
        return empty, np.zeros(0), [], {**report, "lifted": 0, "skipped": "no surface around it"}
    report["plane"] = {
        "normal": [round(float(v), 4) for v in plane[0]],
        "offset": round(plane[1], 4),
        "points": int(around.sum()),
    }
    cameras = hole_views(grid, removed, views + 1, width=width, height=height)
    held, used = cameras[-1], cameras[:-1]
    conds, silhouettes = [], []
    for camera in used:
        cond, silhouette = _hole_conditioning(kept, removed, camera, plane, renderer)
        conds.append(cond)
        silhouettes.append(int(silhouette.sum()))
    if context is not None and hasattr(filler, "context"):
        filler.context = context  # type: ignore[attr-defined]
        report["context"] = context
    calls_before = len(getattr(filler, "received", []))
    if getattr(filler, "chain_views", False):
        filled = _chained_fill(conds, filler, renderer, stride)
    else:
        filled = fill_views(conds, filler)
    if received := getattr(filler, "received", [])[calls_before:]:
        report["fillerCalls"] = received
    lifted, confidence = lift(filled, stride=stride, hole_scaled=True)
    distilled: dict[str, object] | None = None
    if distill_iterations > 0 and len(lifted):
        import distill_fill

        lifted, distilled = refine(
            lifted, kept, filled, distill_iterations, distill_runner or distill_fill.run
        )
        confidence = np.clip(lifted.opacities / 0.95, 0, 1)
    if save_dir is not None:
        for k, f in enumerate(filled):
            cond = f.conditioning
            truth = to_u8(renderer(Splats.concat([kept, removed]), cond.camera).rgb)
            _save_strip(save_dir / f"view{k}.png", [truth, to_u8(cond.full.rgb), f.rgb], cond.mask)
    report.update(
        {
            "lifted": len(lifted),
            "views": [
                {
                    "silhouettePx": s,
                    "throughPx": int(f.conditioning.mask.sum()),
                    "gatePsnr": round(f.gate_psnr_db, 2),
                    "accepted": f.accepted,
                }
                for s, f in zip(silhouettes, filled, strict=True)
            ],
        }
    )
    held_cond, _ = _hole_conditioning(kept, removed, held, plane, renderer)
    report["heldOut"] = _coverage(renderer, kept, lifted, held, held_cond.mask)
    if save_dir is not None:
        before = to_u8(renderer(Splats.concat([kept, removed]), held).rgb)
        without = to_u8(held_cond.full.rgb)
        with_fill = (
            to_u8(renderer(Splats.concat([kept, lifted]), held).rgb) if len(lifted) else without
        )
        moved = to_u8(
            renderer(Splats.concat([kept, lifted, _moved_aside(removed, held)]), held).rgb
        )
        _save_strip(save_dir / "held-out.png", [before, without, with_fill, moved], held_cond.mask)
    if distilled:
        report["distill"] = distilled
    used_cameras = [f.conditioning.camera for f in filled if f.accepted]
    return lifted, confidence, used_cameras, report


#: `fill_hole`'s last panel moves the object this share of its size, to the camera's right.
MOVE_SHARE = 0.9


def _moved_aside(removed: Splats, camera: Camera) -> Splats:
    """`removed` shifted level, to `camera`'s right, by `MOVE_SHARE` of its size."""
    right = np.array(camera.rotation[0], dtype=np.float64)
    right[2] = 0.0
    right /= max(float(np.linalg.norm(right)), 1e-9)
    size = float(np.max(removed.positions.max(axis=0) - removed.positions.min(axis=0)))
    shift = MOVE_SHARE * size * right
    return Splats(
        removed.positions + shift,
        removed.rotations,
        removed.scales,
        removed.colours,
        removed.opacities,
    )


def _save_strip(path: Path, images: Sequence[np.ndarray], mask: np.ndarray) -> None:
    """Truth, hole and fill side by side, the mask outlined in the hole."""
    import cv2
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    marked = images[1].copy()
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    marked[edge > 0] = (255, 0, 255)
    strip = np.concatenate([images[0], marked, *images[2:]], axis=1)
    Image.fromarray(strip).save(path)


def _save_before_after(
    save_dir: Path,
    splats: Splats,
    lifted: Splats,
    grid: vc.ConeGrid,
    filled: Sequence[Filled],
    renderer: Renderer = render,
) -> None:
    """Per accepted view, `after{k}.png`: the scan as the globe draws it from there (faded by
    its view cones) beside the same with the inferred layer added."""
    from PIL import Image

    both = Splats.concat([splats, lifted])
    for k, f in enumerate(filled):
        if not f.accepted:
            continue
        camera = f.conditioning.camera
        weights = np.concatenate([seen_weights(splats, camera, grid), np.ones(len(lifted))])
        after = to_u8(renderer(both, camera, opacity_scale=weights).rgb)
        strip = np.concatenate([to_u8(f.conditioning.seen.rgb), after], axis=1)
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strip).save(save_dir / f"after{k}.png")


def package_inferred(
    lifted: Splats,
    confidence: np.ndarray,
    cameras: Sequence[Camera],
    measured_tileset: Path,
    out_dir: Path,
    filler: str,
    *,
    rule: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """The lifted gaussians as their own tileset beside the measured one: the same frame
    (the measured root transform), view cones from the virtual cameras that made them, and
    `extras.evidence` saying what they are (`rule` replaces the fill-from-outside rule;
    `extra` adds keys, e.g. the instance whose hole it fills)."""
    import splat_tiles

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir.parent / f".{out_dir.name}-work"
    work.mkdir(parents=True, exist_ok=True)
    ply = work / "inferred.ply"
    save_ply(ply, lifted, comment=f"inferred by {filler}, tools/captures/teacher_fill.py")
    splat_tiles.convert(ply, out_dir, 0.0, 0.0, 0.0, opacity_min=0.0)
    observers = np.array([c.centre for c in cameras])
    layout = splat_tiles.ply_layout(ply)
    keep = np.ones(layout.count, bool)
    cones = vc.cone_grid(layout, keep, observers=observers)
    (out_dir / vc.URI).unlink(missing_ok=True)
    cone_extras = vc.write_view_cones(cones, out_dir)
    measured = json.loads(measured_tileset.read_text(encoding="utf-8"))
    tileset_path = out_dir / "tileset.json"
    document = json.loads(tileset_path.read_text(encoding="utf-8"))
    document["root"]["transform"] = measured["root"]["transform"]
    evidence = {
        "kind": "inferred",
        "filler": filler,
        "views": len(cameras),
        "gaussians": len(lifted),
        "meanConfidence": round(float(confidence.mean()), 4) if confidence.size else 0.0,
        "measured": measured_tileset.parent.name,
        "rule": rule
        or (
            "masked pixels of virtual views (what the scan covers but never saw from there), "
            "filled by an image model, gated on the unmasked pixels, lifted at the scan's "
            "depth as discs facing their camera, opacity by distance from measured pixels"
        ),
        **(extra or {}),
    }
    document["root"]["extras"]["viewCones"] = cone_extras
    document["root"]["extras"]["evidence"] = evidence
    tileset_path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return evidence


def _arrays(splats: Splats) -> dict[str, np.ndarray]:
    return {
        "positions": splats.positions,
        "rotations": splats.rotations,
        "scales": splats.scales,
        "colours": splats.colours,
        "opacities": splats.opacities,
    }


def refine(
    lifted: Splats,
    splats: Splats,
    filled: Sequence[Filled],
    iterations: int,
    runner: Callable[[dict], dict],
) -> tuple[Splats, dict[str, object]]:
    """The lifted gaussians refined against the accepted views (`distill_fill`), the scan
    cropped to their neighbourhood and frozen. `runner` takes `distill_fill.run`'s request:
    `distill_fill.run` itself on this machine, or the `Distill` function on a GPU."""
    import distill_fill as df

    kept = [f for f in filled if f.accepted]
    low, high = lifted.positions.min(axis=0), lifted.positions.max(axis=0)
    margin = 0.25 * float(np.max(high - low)) + 1e-6
    near = np.all((splats.positions >= low - margin) & (splats.positions <= high + margin), axis=1)
    request = {
        "measured": df.pack_scan(_arrays(splats.take(np.flatnonzero(near)))),
        "init": df.pack_scan(_arrays(lifted)),
        "views": df.pack_views(
            [f.conditioning.camera.to_json() for f in kept],
            np.stack([f.rgb for f in kept]),
            np.stack([f.conditioning.mask for f in kept]),
        ),
        "iterations": iterations,
    }
    response = runner(request)
    out = df.unpack_scan(response["inferred"])
    return Splats(*(out[k] for k in df.KEYS)), response["report"]


def fill_scan(
    splats: Splats,
    grid: vc.ConeGrid,
    filler: Filler,
    measured_tileset: Path,
    out_dir: Path,
    *,
    views: int = 8,
    mode: str = "ring",
    width: int = 640,
    height: int = 480,
    stride: int = 2,
    save_dir: Path | None = None,
    distill_iterations: int = 0,
    distill_runner: Callable[[dict], dict] | None = None,
    renderer: Renderer = render,
    max_scale_m: float | None = None,
) -> dict[str, object]:
    """The whole of Teacher B on one scan: views at what its view cones fade (`ring`: from
    outside, the sides never walked to; `near`: from the observers), conditioned, filled,
    gated, lifted and packaged beside `measured_tileset` in `out_dir`. Returns the
    evidence block, with a per-view report under `views`.

    `max_scale_m`: condition on the scan without its gaussians larger than this (largest
    axis) -- the floaters at a capture's edge, which a rasterizer draws as blobs over most
    of a view from outside (the camp: 0.4% of its gaussians at 0.5 m). The CPU renderer
    all but hides them (few samples, spread thin); gsplat does not."""
    if max_scale_m is not None:
        splats = splats.take(np.flatnonzero(splats.scales.max(axis=1) <= max_scale_m))
    texels = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, splats.positions)
    faded = texels[:, 2] != vc.OMNI
    if not faded.any():
        return {"kind": "inferred", "gaussians": 0, "skipped": "nothing is faded"}
    targets = splats.positions[faded]
    centre = targets.mean(axis=0)
    reach = float(np.percentile(np.linalg.norm(splats.positions - centre, axis=1), 95))
    cameras = plan_views(
        grid,
        targets,
        count=views,
        mode=mode,
        ring_radius_m=1.5 * reach,
        width=width,
        height=height,
    )
    conds = [condition(splats, camera, grid, renderer=renderer) for camera in cameras]
    context: dict[str, object] | None = None
    if hasattr(filler, "context") and (instances := read_instances(measured_tileset)):
        # A generative filler is told what the scan holds (what its views look at).
        low, high = np.percentile(targets, 2, axis=0), np.percentile(targets, 98, axis=0)
        context = describe_surroundings(instances, low, high, view="outside")
        filler.context = context  # type: ignore[attr-defined]
    filled = fill_views(conds, filler)
    if save_dir is not None:
        for k, f in enumerate(filled):
            given = to_u8(f.conditioning.full.rgb)
            seen = to_u8(f.conditioning.seen.rgb)
            _save_strip(save_dir / f"view{k}.png", [given, seen, f.rgb], f.conditioning.mask)
    lifted, confidence = lift(filled, stride=stride)
    per_view = [
        {
            "maskPx": int(f.conditioning.mask.sum()),
            "gatePsnr": round(f.gate_psnr_db, 2),
            "accepted": f.accepted,
        }
        for f in filled
    ]
    if len(lifted) == 0:
        return {
            "kind": "inferred",
            "gaussians": 0,
            "skipped": "no accepted fill",
            "views": per_view,
        }
    distilled: dict[str, object] | None = None
    if distill_iterations > 0:
        import distill_fill

        lifted, distilled = refine(
            lifted, splats, filled, distill_iterations, distill_runner or distill_fill.run
        )
    used = [f.conditioning.camera for f in filled if f.accepted]
    evidence = package_inferred(lifted, confidence, used, measured_tileset, out_dir, filler.name)
    if save_dir is not None:
        _save_before_after(save_dir, splats, lifted, grid, filled, renderer)
    extra: dict[str, object] = {"perView": per_view}
    if distilled:
        extra["distill"] = distilled
    if context is not None:
        extra["context"] = context
    if received := getattr(filler, "received", None):
        extra["fillerCalls"] = received
    return {**evidence, **extra}


def make_filler(spec: str) -> Filler:
    """`telea` (the CPU stand-in) or `module:Class` -- a GPU filler such as
    `world_model_client:FixerFiller` -- constructed with the keyword arguments after a `?`
    (`world_model_client:FixerFiller?timestep=50`; numbers are parsed as numbers)."""
    if spec == "telea":
        return InpaintFiller()
    import importlib

    path, _, query = spec.partition("?")
    module, _, name = path.partition(":")
    if not name:
        raise ValueError(f"filler {spec!r}: expected 'telea' or 'module:Class[?key=value&...]'")
    kwargs: dict[str, object] = {}
    for pair in filter(None, query.split("&")):
        key, _, value = pair.partition("=")
        try:
            kwargs[key] = int(value)
        except ValueError:
            try:
                kwargs[key] = float(value)
            except ValueError:
                kwargs[key] = value
    return getattr(importlib.import_module(module), name)(**kwargs)


def link_inferred(measured_tileset: Path, inferred_tileset: Path) -> list[dict[str, object]]:
    """Declares an inferred layer on the measured tileset's root (`extras.inferredLayers`,
    a path relative to it and the layer's evidence), which is how the viewer finds it
    (apps/web/src/lib/inferred.ts). Linking the same layer again replaces its entry."""
    import os

    measured = json.loads(measured_tileset.read_text(encoding="utf-8"))
    inferred = json.loads(inferred_tileset.read_text(encoding="utf-8"))
    evidence = inferred["root"].get("extras", {}).get("evidence")
    if not evidence or evidence.get("kind") != "inferred":
        raise ValueError(f"{inferred_tileset} carries no inferred evidence")
    uri = Path(os.path.relpath(inferred_tileset, measured_tileset.parent)).as_posix()
    extras = measured["root"].setdefault("extras", {})
    layers = [layer for layer in extras.get("inferredLayers", []) if layer.get("uri") != uri]
    layers.append({"uri": uri, "evidence": evidence})
    extras["inferredLayers"] = layers
    measured_tileset.write_text(json.dumps(measured, indent=1), encoding="utf-8")
    return layers


def _distill_runner(where: str) -> Callable[[dict], dict]:
    if where == "local":
        import distill_fill

        return distill_fill.run
    from world_model_client import modal_remote

    return lambda request: modal_remote("Distill", "run", request)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from splat_render import load_tileset

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    link = sub.add_parser("link", help="declare an inferred layer on its measured tileset")
    link.add_argument("tileset", type=Path, help="the measured tileset.json")
    link.add_argument("inferred", type=Path, help="the inferred layer's tileset.json")
    for name in ("drop", "fill"):
        p = sub.add_parser(name)
        p.add_argument("tileset", type=Path, help="the measured tileset.json (with viewcones.bin)")
        p.add_argument("--filler", default="telea", help="'telea' or module:Class")
        p.add_argument("--views", type=int, default=4 if name == "drop" else 8)
        p.add_argument("--width", type=int, default=480 if name == "drop" else 640)
        p.add_argument("--height", type=int, default=360 if name == "drop" else 480)
        p.add_argument("--stride", type=int, default=2)
        p.add_argument("--save", type=Path, help="where to write truth/hole/fill strips")
        p.add_argument(
            "--renderer",
            choices=("cpu", "gsplat"),
            default="cpu",
            help="render the views on the CPU or with gsplat on a CUDA GPU",
        )
    sub.choices["drop"].add_argument("--half-size-m", type=float, default=0.15)
    sub.choices["drop"].add_argument(
        "--centre", type=float, nargs=3, help="local ENU; default: beside the main observer"
    )
    sub.choices["fill"].add_argument("out", type=Path, help="the inferred tileset's directory")
    sub.choices["fill"].add_argument("--mode", choices=("ring", "near"), default="ring")
    sub.choices["fill"].add_argument(
        "--max-scale-m",
        type=float,
        help="condition on the scan without gaussians larger than this (floaters)",
    )
    sub.choices["fill"].add_argument(
        "--distill", type=int, default=0, help="refine the lifted fill this many steps"
    )
    sub.choices["fill"].add_argument(
        "--distill-on", choices=("local", "modal"), default="modal", help="where to refine"
    )
    args = parser.parse_args(argv)

    if args.command == "link":
        print(json.dumps(link_inferred(args.tileset, args.inferred), indent=1))
        return 0
    splats = load_tileset(args.tileset)
    grid = vc.cone_grid_from_tileset(args.tileset)
    filler = make_filler(args.filler)
    renderer = make_renderer(args.renderer)
    if args.command == "drop":
        if args.centre is not None:
            centre = np.array(args.centre)
        else:
            observer = grid.observers[int(np.argmax(grid.observer_weights))]
            near = np.argsort(np.linalg.norm(splats.positions - observer, axis=1))[:2000]
            centre = splats.positions[near].mean(axis=0)
        report = drop_and_fill(
            splats,
            centre,
            args.half_size_m,
            filler,
            grid,
            views=args.views,
            width=args.width,
            height=args.height,
            stride=args.stride,
            save_dir=args.save,
            renderer=renderer,
        )
        print(json.dumps(report.to_json(), indent=1))
    else:
        evidence = fill_scan(
            splats,
            grid,
            filler,
            args.tileset,
            args.out,
            views=args.views,
            mode=args.mode,
            width=args.width,
            height=args.height,
            stride=args.stride,
            save_dir=args.save,
            distill_iterations=args.distill,
            distill_runner=_distill_runner(args.distill_on) if args.distill else None,
            renderer=renderer,
            max_scale_m=args.max_scale_m,
        )
        print(json.dumps(evidence, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
