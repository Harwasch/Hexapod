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
2. **conditioning** (`condition`) -- per view, the scan rendered with each gaussian faded by
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
   camera, at the depth the scan has there (or the depth inpainted from around the hole),
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

__all__ = [
    "Conditioning",
    "Filler",
    "InpaintFiller",
    "clean_mask",
    "condition",
    "drop_and_fill",
    "fill_scan",
    "lift",
    "link_inferred",
    "make_filler",
    "package_inferred",
    "plan_views",
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
    all -- what an artifact-fixing model (NVIDIA Fixer) is trained to clean."""

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
) -> Conditioning:
    """What `camera` is to be told and asked: the seen render, and the mask to fill."""
    import cv2

    weights = seen_weights(splats, camera, grid) if seen_opacity is None else seen_opacity
    seen = render(splats, camera, opacity_scale=weights)
    full = render(splats, camera)
    if mask is None:
        mask = clean_mask((seen.alpha < SEEN_ALPHA) & (full.alpha >= COVERED_ALPHA))
    depth = np.where(np.isfinite(full.depth), full.depth, np.nan)
    known = np.isfinite(seen.depth) & (seen.alpha >= SEEN_ALPHA)
    missing = mask & ~np.isfinite(depth)
    if missing.any() and known.any():
        log_d = np.where(known, np.log(np.where(known, seen.depth, 1.0)), 0).astype(np.float32)
        filled = cv2.inpaint(log_d, (~known).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        depth = np.where(missing, np.exp(filled), depth)
    distance = cv2.distanceTransform((mask | ~known).astype(np.uint8), cv2.DIST_L2, 5)
    return Conditioning(camera, seen, full, mask, depth, distance)


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
            a, b = (_blur(candidate), _blur(given)) if full else (candidate, given)
            score = min(psnr(a, b, kept & ~cond.mask), GATE_CAP_DB)
            if best is None or score > best.gate_psnr_db:
                best = Filled(cond, candidate, score, score >= threshold)
        assert best is not None
        out.append(best)
    return out


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


def lift(filled: Sequence[Filled], stride: int = 2) -> tuple[Splats, np.ndarray]:
    """The accepted fills' masked pixels as gaussians, and each one's confidence."""
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
        confidence = np.exp(-cond.distance[v, u] / CONFIDENCE_PX * math.log(2))
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
) -> DropReport:
    """Removes the gaussians in a cube the capture saw, fills it from `views` cameras near
    the observers, and scores the fill against what was there: per fill view (PSNR, SSIM on
    the dropped region's pixels, against leaving the hole), and -- after lifting -- from one
    more view the fill never used."""
    centre = np.asarray(centre, np.float64)
    inside = np.all(np.abs(splats.positions - centre) <= half_size_m, axis=1)
    kept, dropped = splats.take(np.flatnonzero(~inside)), splats.take(np.flatnonzero(inside))
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
        truth = to_u8(render(splats, camera).rgb)
        hole = _hole(render(dropped, camera).alpha)
        cond = condition(kept, camera, None, mask=hole, seen_opacity=np.ones(len(kept)))
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
    truth = to_u8(render(splats, held).rgb)
    hole = _hole(render(dropped, held).alpha)
    if hole.any():
        without = to_u8(render(kept, held).rgb)
        with_fill = (
            to_u8(render(Splats.concat([kept, lifted]), held).rgb) if len(lifted) else without
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


def _hole(alpha: np.ndarray) -> np.ndarray:
    """Where a dropped region covered the view: its coverage, closed over the gaps the
    renderer's samples leave (3 px)."""
    import cv2

    raw = (alpha >= 0.25).astype(np.uint8)
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)


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
    save_dir: Path, splats: Splats, lifted: Splats, grid: vc.ConeGrid, filled: Sequence[Filled]
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
        after = to_u8(render(both, camera, opacity_scale=weights).rgb)
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
) -> dict[str, object]:
    """The lifted gaussians as their own tileset beside the measured one: the same frame
    (the measured root transform), view cones from the virtual cameras that made them, and
    `extras.evidence` saying what they are."""
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
        "rule": (
            "masked pixels of virtual views (what the scan covers but never saw from there), "
            "filled by an image model, gated on the unmasked pixels, lifted at the scan's "
            "depth as discs facing their camera, opacity by distance from measured pixels"
        ),
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
) -> dict[str, object]:
    """The whole of Teacher B on one scan: views at what its view cones fade (`ring`: from
    outside, the sides never walked to; `near`: from the observers), conditioned, filled,
    gated, lifted and packaged beside `measured_tileset` in `out_dir`. Returns the
    evidence block, with a per-view report under `views`."""
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
    conds = [condition(splats, camera, grid) for camera in cameras]
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
        _save_before_after(save_dir, splats, lifted, grid, filled)
    return {**evidence, "perView": per_view, **({"distill": distilled} if distilled else {})}


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
    sub.choices["drop"].add_argument("--half-size-m", type=float, default=0.15)
    sub.choices["drop"].add_argument(
        "--centre", type=float, nargs=3, help="local ENU; default: beside the main observer"
    )
    sub.choices["fill"].add_argument("out", type=Path, help="the inferred tileset's directory")
    sub.choices["fill"].add_argument("--mode", choices=("ring", "near"), default="ring")
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
        )
        print(json.dumps(evidence, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
