"""Living view: slight wind on the plants of a still render, from a video model, anchored.

The idea under test (the bake-off in `infra/modal/living_bakeoff.py`, run by
`.github/workflows/living-view.yml`): the viewer shows the measured splat; when the camera
stops, a video model adds slight wind to the plants, conditioned on the exact render of the
splat, in short segments that always start again from that render, so nothing can drift. This
module is the half that runs anywhere -- on the GPU box beside the renderer and on a laptop
over the clips that come back:

* **Plants** (`plant_instances`, `plant_flags`): which gaussians are plants. In a scene scan,
  an instance of `instances.json` is a plant when its category is one (`PLANT_CATEGORIES`) or
  its best tag names one (`PLANT_TAGS`: the categories are noisy -- most of the camp's bushes
  sit under `ground` or `household` with `bush` as their best tag). A scan that is one
  isolated plant (the Minnetonka tree) is a plant everywhere above its ground, except the
  base of its stem, which does not sway and anchors the camera fit below.
* **Masks** (`feather`): the plant label render thresholded, grown a little and blurred, so a
  warp fades out across the plant's outline instead of tearing at it.
* **Motion only** (`motion_only`): dense optical flow from every generated frame back to the
  clip's frame 0 with OpenCV's DIS (no learned weights -- RAFT's trace to non-commercial
  data); the camera's own creep taken out (`global_motion`: a homography fitted by RANSAC to
  the flow outside the plants, where the texture can carry it); the flow zeroed outside the
  feathered plant mask, scaled to the render's size, eased to nothing at the segment's ends
  (`loop_window`, a Tukey window) so the clip returns to the render and loops; and **our
  render** backward-warped by it. The appearance stays the measured scan's; only the motion
  is the model's.
* **Edge-aware** (`MODES`): the model's flow is coarser than our render (848 or 832 px wide
  against 1280) and DIS blurs it across edges anyway, so scaling it up bilinearly and fading
  it with a feathered mask drags a ring of sky with every twig. The `guided` mode scales it up
  with a guided filter (`GuidedFilter`, He et al. 2010) whose guide is our full-size render and
  its plant share -- a normalised convolution in which only plant pixels vote -- and keeps a
  pixel's motion only where it reads plant (`source_masked`). Both are made, side by side.
* **Numbers** (`motion_stats`, `psnr_outside`, `MotionReport`): how far pixels move inside and
  outside the plants before the camera is taken out (camera creep and things that should not
  move show up outside), how far the fitted camera crept, how much of what is not a plant the
  model redrew (PSNR against our render), and per mode the motion left on the ring of sky
  round the plants (`haloPx`).

    living_view.py process --render camp-1.png --mask camp-1-mask.png \\
        --share camp-1-share.png --clip clips/ltx/camp-1.mp4 --out site/clips --name camp-1-ltx
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MODES",
    "PLANT_CATEGORIES",
    "PLANT_TAGS",
    "GuidedFilter",
    "MotionReport",
    "dis_flow",
    "edge_guide",
    "edge_ring",
    "feather",
    "global_motion",
    "loop_window",
    "motion_only",
    "motion_stats",
    "plant_flags",
    "plant_instances",
    "psnr_outside",
    "resize_frames",
    "source_masked",
    "upsample_flow",
    "upsample_flow_guided",
    "warp",
]

#: instances.json categories that sway in wind.
PLANT_CATEGORIES = frozenset({"shrubs", "trees", "grass", "flowers"})
#: Best tags that name a plant whatever the category says (SigLIP vocabulary labels).
PLANT_TAGS = frozenset(
    {
        "bush",
        "shrub",
        "fern",
        "branch",
        "tree",
        "conifer",
        "pine tree",
        "leaf",
        "foliage",
        "plant",
        "hedge",
        "ivy",
        "vine",
        "grass",
        "sapling",
        "evergreen",
    }
)
#: The low, leafy kinds a living view is mostly about (weighted up when framing a view).
SHRUB_TAGS = frozenset({"bush", "shrub", "fern"})

#: Mask: a pixel is a plant when at least this share of its coverage is plant.
PLANT_SHARE = 0.5
#: ...and not a plant at all below this (the "outside" of the numbers and the camera fit).
NOT_PLANT_SHARE = 0.05
#: Feathering, as a fraction of the image's width: grown by this, then blurred by this.
FEATHER_GROW = 0.004
FEATHER_BLUR = 0.008
#: The camera fit: background pixels need this much image gradient (0-255 grey per pixel) to
#: carry a flow vector; fewer than `MIN_BACKGROUND` such pixels and no camera is fitted.
MIN_GRADIENT = 6.0
MIN_BACKGROUND = 200
RANSAC_PX = 1.0
#: Share of a segment eased in and out at each end.
LOOP_RAMP = 0.15


# --- plants --------------------------------------------------------------------------------


def _is_plant(instance: dict) -> bool | None:
    """True / False from the instance's own category and best tag; None when it has neither."""
    category = instance.get("category")
    tags = instance.get("tags") or []
    if category is None and not tags:
        return None
    if category in PLANT_CATEGORIES:
        return True
    return bool(tags) and tags[0].get("label") in PLANT_TAGS


def _is_shrub(instance: dict) -> bool:
    tags = instance.get("tags") or []
    return instance.get("category") == "shrubs" or (
        bool(tags) and tags[0].get("label") in SHRUB_TAGS
    )


def plant_instances(instances: Sequence[dict]) -> dict[int, float]:
    """Plant instances of an `instances.json` and their framing weight: 1 for shrubs, bushes
    and ferns, 0.5 for other plants. An instance with no category or tags of its own takes
    its parent's verdict."""
    by_id = {int(i["id"]): i for i in instances}
    out: dict[int, float] = {}
    for instance in instances:
        verdict, at = _is_plant(instance), instance
        while verdict is None and at.get("parent") is not None and int(at["parent"]) in by_id:
            at = by_id[int(at["parent"])]
            verdict = _is_plant(at)
        if verdict:
            out[int(instance["id"])] = 1.0 if _is_shrub(at) else 0.5
    return out


def plant_flags(ids: np.ndarray, plants: dict[int, float]) -> np.ndarray:
    """Per gaussian (its instance id in `ids`), its plant weight (0 for not a plant)."""
    ids = np.asarray(ids, np.int64)
    if ids.size == 0 or not plants:
        return np.zeros(ids.shape, np.float64)
    keys = np.fromiter(plants.keys(), np.int64)
    values = np.fromiter(plants.values(), np.float64)
    order = np.argsort(keys)
    keys, values = keys[order], values[order]
    at = np.clip(np.searchsorted(keys, ids), 0, keys.size - 1)
    return np.where(keys[at] == ids, values[at], 0.0)


def isolated_plant_flags(
    positions: np.ndarray,
    *,
    ground_m: float = 0.15,
    stem_radius_m: float = 0.6,
    stem_m: float = 1.8,
) -> np.ndarray:
    """A scan that is one plant on its own (origin at the stem's foot, z up): everything above
    `ground_m` is plant, except the stem within `stem_radius_m` of the axis below `stem_m`."""
    p = np.asarray(positions, np.float64)
    above = p[:, 2] > ground_m
    stem = (np.hypot(p[:, 0], p[:, 1]) < stem_radius_m) & (p[:, 2] < stem_m)
    return (above & ~stem).astype(np.float64)


# --- masks ---------------------------------------------------------------------------------


def feather(
    share: np.ndarray, *, grow: float = FEATHER_GROW, blur: float = FEATHER_BLUR
) -> np.ndarray:
    """A soft plant mask in [0, 1] from a per-pixel plant share: thresholded at
    `PLANT_SHARE`, dilated by `grow` and blurred by `blur` (fractions of the width)."""
    import cv2

    share = np.asarray(share, np.float32)
    w = share.shape[1]
    hard = (share >= PLANT_SHARE).astype(np.uint8)
    r = max(1, round(grow * w))
    hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    sigma = max(0.5, blur * w)
    return np.clip(cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma), 0.0, 1.0)


# --- flow ----------------------------------------------------------------------------------


def _grey(frame: np.ndarray) -> np.ndarray:
    import cv2

    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def dis_flow(frame: np.ndarray, reference: np.ndarray, preset: str = "medium") -> np.ndarray:
    """Dense flow (h, w, 2) from `frame` back to `reference`: `frame(x) ~ reference(x + f(x))`.
    OpenCV's DIS (Kroeger et al. 2016), no learned weights."""
    import cv2

    presets = {
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    }
    dis = cv2.DISOpticalFlow_create(presets[preset])
    return dis.calc(_grey(frame), _grey(reference), None).astype(np.float32)


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


def global_motion(
    flow: np.ndarray, background: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, int]:
    """The camera's own motion in `flow`: a homography `H` with `x + f(x) ~ H x`, fitted by
    RANSAC to (at most 4000 of) the `background` pixels (bool, same grid). Identity, and 0,
    when fewer than `MIN_BACKGROUND` pixels are there to carry it; else `H` and how many
    pixels were offered."""
    import cv2

    ys, xs = np.nonzero(background)
    if ys.size < MIN_BACKGROUND:
        return np.eye(3), 0
    rng = rng or np.random.default_rng(0)
    if ys.size > 4000:
        pick = rng.choice(ys.size, 4000, replace=False)
        ys, xs = ys[pick], xs[pick]
    src = np.stack([xs, ys], axis=1).astype(np.float32)
    dst = src + flow[ys, xs]
    H, _inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=2000)
    if H is None or not np.all(np.isfinite(H)):
        return np.eye(3), int(ys.size)
    return H, int(ys.size)


def homography_flow(H: np.ndarray, h: int, w: int) -> np.ndarray:
    """`H x - x` on an (h, w) pixel grid."""
    x, y = _grid(h, w)
    p = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.asarray(H, np.float64).T
    mapped = p[..., :2] / p[..., 2:3]
    return (mapped - np.stack([x, y], axis=-1)).astype(np.float32)


def upsample_flow(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    """A flow field resized to (height, width), its vectors scaled with it."""
    import cv2

    h, w = flow.shape[:2]
    if (w, h) == (width, height):
        return flow.astype(np.float32)
    big = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    return (big * np.array([width / w, height / h], np.float32)).astype(np.float32)


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward warp: `out(x) = image(x + flow(x))`, bilinear, edges reflected."""
    import cv2

    h, w = flow.shape[:2]
    x, y = _grid(h, w)
    return cv2.remap(
        np.asarray(image),
        x + flow[..., 0],
        y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def loop_window(n: int, ramp: float = LOOP_RAMP) -> np.ndarray:
    """A Tukey window over `n` frames: 0 at both ends, cosine ramps over `ramp` of the
    segment at each, 1 between. Multiplying a segment's flow by it starts and ends the
    segment on the render itself, so it loops and re-anchors without a jump."""
    if n <= 1:
        return np.zeros(max(n, 0))
    t = np.arange(n) / (n - 1)
    w = np.ones(n)
    if ramp > 0:
        rising = t < ramp
        falling = t > 1 - ramp
        w[rising] = 0.5 - 0.5 * np.cos(np.pi * t[rising] / ramp)
        w[falling] = 0.5 - 0.5 * np.cos(np.pi * (1 - t[falling]) / ramp)
    w[0] = w[-1] = 0.0
    return w


def resize_frames(frames: Iterable[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    """Frames resized to (width, height): Lanczos up, area down."""
    import cv2

    out = []
    for f in frames:
        h, w = f.shape[:2]
        if (w, h) == (width, height):
            out.append(np.asarray(f))
            continue
        method = cv2.INTER_LANCZOS4 if width * height > w * h else cv2.INTER_AREA
        out.append(cv2.resize(np.asarray(f), (width, height), interpolation=method))
    return out


# --- edge-aware upsampling ---------------------------------------------------------------------


class GuidedFilter:
    """He, Sun and Tang's guided filter (ECCV 2010, Algorithm 2) with a multi-channel guide.

    The output is locally a linear function of the guide, so it steps where the guide steps:
    a smooth low-resolution flow filtered with our full-resolution render (and its plant
    share) as the guide takes the render's edges. The guide's statistics -- its box means and
    the inverse of its regularised local covariance -- are computed once, since the render is
    the guide for every frame of a clip; each input then costs a handful of box filters."""

    def __init__(self, guide: np.ndarray, radius: int, eps: float) -> None:
        g = np.asarray(guide, np.float32)
        if g.ndim == 2:
            g = g[..., None]
        self.guide = g
        self.radius = int(radius)
        self.eps = float(eps)
        self.height, self.width, channels = g.shape
        size = (2 * self.radius + 1, 2 * self.radius + 1)
        self._size = size
        self.mean = np.stack([self._box(g[..., c]) for c in range(channels)], axis=-1)
        cov = np.empty((self.height, self.width, channels, channels), np.float32)
        for i in range(channels):
            for j in range(i, channels):
                v = self._box(g[..., i] * g[..., j]) - self.mean[..., i] * self.mean[..., j]
                cov[..., i, j] = v
                cov[..., j, i] = v
        cov += self.eps * np.eye(channels, dtype=np.float32)
        self.inverse = np.linalg.inv(cov).astype(np.float32)

    def _box(self, x: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.boxFilter(
            np.asarray(x, np.float32), -1, self._size, borderType=cv2.BORDER_REFLECT
        )

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float32)
        g, channels = self.guide, self.guide.shape[-1]
        mean_p = self._box(p)
        cov_gp = np.stack(
            [self._box(g[..., c] * p) - self.mean[..., c] * mean_p for c in range(channels)],
            axis=-1,
        )
        a = np.einsum("hwij,hwj->hwi", self.inverse, cov_gp)
        b = mean_p - np.einsum("hwi,hwi->hw", a, self.mean)
        mean_a = np.stack([self._box(a[..., c]) for c in range(channels)], axis=-1)
        return np.einsum("hwi,hwi->hw", mean_a, g) + self._box(b)


#: Edge-aware flow: the guided filter's radius (render pixels) and regularisation (the guide
#: in [0, 1]), and how much the plant share weighs in the guide beside the colour.
GUIDED_RADIUS = 6
GUIDED_EPS = 1e-3
GUIDE_SHARE_WEIGHT = 0.5


def edge_guide(render: np.ndarray, share: np.ndarray) -> GuidedFilter:
    """The guided filter for a start: our render's colour (0-1) and its plant share."""
    rgb = np.asarray(render, np.float32) / 255.0
    weight = GUIDE_SHARE_WEIGHT * np.asarray(share, np.float32)[..., None]
    return GuidedFilter(np.concatenate([rgb, weight], axis=2), GUIDED_RADIUS, GUIDED_EPS)


def upsample_flow_guided(flow: np.ndarray, guide: GuidedFilter, weight: np.ndarray) -> np.ndarray:
    """A low-resolution flow to the guide's size, edge-aware: bilinear first, then each
    component guided-filtered as a normalised convolution weighted by `weight` (the plant
    share at full size) -- only plant pixels vote for a plant's motion, so the sky between two
    branches does not average the branches' motion with its own zero, and the flow steps
    where the render steps rather than where the coarse flow happened to blur."""
    up = upsample_flow(flow, guide.width, guide.height)
    weight = np.asarray(weight, np.float32)
    den = guide(weight)
    out = np.zeros_like(up)
    valid = den > 0.05
    for c in range(2):
        num = guide(up[..., c] * weight)
        out[..., c] = np.where(valid, num / np.maximum(den, 0.05), 0.0)
    return out


def source_masked(flow: np.ndarray, share: np.ndarray) -> np.ndarray:
    """`flow` kept only where it comes from a plant: a backward flow at `x` reads the render at
    `x + f(x)`, so a pixel moves when what it shows is plant. A sky pixel next to a branch
    that the branch sways over reads the branch and moves; one that only caught the coarse
    flow's blur reads sky and stays -- no halo of dragged sky."""
    import cv2

    h, w = flow.shape[:2]
    x, y = _grid(h, w)
    source = cv2.remap(
        np.asarray(share, np.float32),
        x + flow[..., 0],
        y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return flow * np.clip(source, 0.0, 1.0)[..., None]


def edge_ring(share: np.ndarray, width_px: int = 6) -> np.ndarray:
    """Not-plant pixels within `width_px` of a plant: where a halo would show."""
    import cv2

    hard = (np.asarray(share) >= PLANT_SHARE).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width_px + 1, 2 * width_px + 1))
    return (cv2.dilate(hard, k) > 0) & (np.asarray(share) < NOT_PLANT_SHARE)


# --- motion only and the numbers -------------------------------------------------------------

#: The ways a clip's motion reaches our render (`motion_only`'s `modes`):
#: `bilinear` -- the flow zeroed outside the feathered plant mask at the model's size, then
#: scaled up bilinearly (the plain recipe); `guided` -- scaled up edge-aware
#: (`upsample_flow_guided`, our render and its plant share as the guide) and kept only where
#: it reads plant (`source_masked`).
MODES = ("bilinear", "guided")


@dataclass
class MotionReport:
    """What one clip did, at the render's scale (pixels)."""

    frames: int
    inside_mean_px: float
    inside_p95_px: float
    outside_mean_px: float
    outside_p95_px: float
    #: The fitted camera's mean displacement over the frame, worst frame (0: none fitted).
    camera_creep_px: float
    #: Pixels offered to the camera fit in the first frame that had any (0: no background).
    background_px: int
    #: PSNR of the not-plant pixels of the model's own frames against our render (dB).
    psnr_outside_db: float | None = None
    #: Per mode: the mean applied motion on the not-plant ring round the plants (`edge_ring`),
    #: render pixels, mean over frames -- what drags sky at branch edges.
    halo_px: dict[str, float] | None = None
    #: Per mode: the mean applied motion on the plants, render pixels, mean over frames.
    applied_px: dict[str, float] | None = None

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {
            "frames": self.frames,
            "insideMeanPx": round(self.inside_mean_px, 3),
            "insideP95Px": round(self.inside_p95_px, 3),
            "outsideMeanPx": round(self.outside_mean_px, 3),
            "outsideP95Px": round(self.outside_p95_px, 3),
            "cameraCreepPx": round(self.camera_creep_px, 3),
            "backgroundPx": self.background_px,
        }
        if self.psnr_outside_db is not None:
            out["psnrOutsideDb"] = round(self.psnr_outside_db, 2)
        if self.halo_px is not None:
            out["haloPx"] = {k: round(v, 3) for k, v in self.halo_px.items()}
        if self.applied_px is not None:
            out["appliedPx"] = {k: round(v, 3) for k, v in self.applied_px.items()}
        return out


def motion_stats(
    flows: Sequence[np.ndarray], mask: np.ndarray, scale: tuple[float, float]
) -> tuple[float, float, float, float]:
    """(inside mean, inside p95, outside mean, outside p95) of the flow magnitudes over every
    frame, in render pixels (`scale`: render size over flow size, x and y). Inside is mask >=
    0.5, outside mask < `NOT_PLANT_SHARE`; `mask` is on the flows' grid."""
    inside = mask >= 0.5
    outside = mask < NOT_PLANT_SHARE
    sx, sy = scale
    ins, outs = [], []
    for f in flows:
        mag = np.hypot(f[..., 0] * sx, f[..., 1] * sy)
        ins.append(mag[inside])
        outs.append(mag[outside])

    def summary(parts: list[np.ndarray]) -> tuple[float, float]:
        values = np.concatenate(parts) if parts else np.zeros(0)
        if values.size == 0:
            return 0.0, 0.0
        return float(values.mean()), float(np.percentile(values, 95))

    return (*summary(ins), *summary(outs))


def psnr_outside(frames: Sequence[np.ndarray], render: np.ndarray, mask: np.ndarray) -> float:
    """PSNR (dB) of the pixels outside the plants (mask < `NOT_PLANT_SHARE`) of `frames` (at
    the render's size) against `render`, the squared error pooled over every frame after
    frame 0."""
    outside = np.asarray(mask) < NOT_PLANT_SHARE
    if not outside.any() or len(frames) < 2:
        return float("inf")
    ref = np.asarray(render, np.float64)[outside]
    errors = [np.mean((np.asarray(f, np.float64)[outside] - ref) ** 2) for f in frames[1:]]
    mse = float(np.mean(errors))
    return float("inf") if mse <= 0 else 10.0 * math.log10(255.0**2 / mse)


def motion_only(
    render: np.ndarray,
    frames: Sequence[np.ndarray],
    soft_mask: np.ndarray,
    *,
    share: np.ndarray | None = None,
    modes: Sequence[str] = ("bilinear",),
    ramp: float = LOOP_RAMP,
    preset: str = "medium",
) -> tuple[dict[str, list[np.ndarray]], MotionReport]:
    """Our render moved by the plant motion of a generated clip (frames at the model's size,
    frame 0 being the model's copy of the render), once per mode in `modes` (`MODES`), and
    the clip's numbers. `soft_mask` is the feathered plant mask and `share` the plant share
    (unfeathered; `soft_mask >= 0.5` when not given), both at the render's size."""
    import cv2

    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
    H_r, W_r = render.shape[:2]
    h, w = frames[0].shape[:2]
    soft_mask = np.asarray(soft_mask, np.float32)
    if share is None:
        share = (soft_mask >= 0.5).astype(np.float32)
    share = np.asarray(share, np.float32)
    mask_small = cv2.resize(soft_mask, (w, h), interpolation=cv2.INTER_AREA)
    grey0 = _grey(frames[0])
    gradient = np.hypot(cv2.Sobel(grey0, cv2.CV_32F, 1, 0), cv2.Sobel(grey0, cv2.CV_32F, 0, 1))
    background = (mask_small < NOT_PLANT_SHARE) & (gradient / 4.0 >= MIN_GRADIENT)
    window = loop_window(len(frames), ramp)
    scale = (W_r / w, H_r / h)
    guide = edge_guide(render, share) if "guided" in modes else None
    ring = edge_ring(share)
    plant = share >= PLANT_SHARE
    out: dict[str, list[np.ndarray]] = {m: [] for m in modes}
    halo: dict[str, list[float]] = {m: [] for m in modes}
    applied: dict[str, list[float]] = {m: [] for m in modes}
    raw = []
    creep, offered = 0.0, 0
    for k, frame in enumerate(frames):
        if k == 0:
            flow = np.zeros((h, w, 2), np.float32)
        else:
            flow = dis_flow(frame, frames[0], preset)
        raw.append(flow)
        H, n = global_motion(flow, background)
        offered = offered or n
        camera = homography_flow(H, h, w)
        creep = max(
            creep, float(np.hypot(camera[..., 0] * scale[0], camera[..., 1] * scale[1]).mean())
        )
        local = (flow - camera) * float(window[k])
        for mode in modes:
            if mode == "bilinear":
                full = upsample_flow(local * mask_small[..., None], W_r, H_r)
            else:
                assert guide is not None
                full = source_masked(upsample_flow_guided(local, guide, share), share)
            out[mode].append(warp(render, full))
            if k:
                magnitude = np.hypot(full[..., 0], full[..., 1])
                halo[mode].append(float(magnitude[ring].mean()) if ring.any() else 0.0)
                applied[mode].append(float(magnitude[plant].mean()) if plant.any() else 0.0)
    inside_mean, inside_p95, outside_mean, outside_p95 = motion_stats(raw[1:], mask_small, scale)
    report = MotionReport(
        frames=len(frames),
        inside_mean_px=inside_mean,
        inside_p95_px=inside_p95,
        outside_mean_px=outside_mean,
        outside_p95_px=outside_p95,
        camera_creep_px=creep,
        background_px=offered,
        halo_px={m: float(np.mean(v)) if v else 0.0 for m, v in halo.items()},
        applied_px={m: float(np.mean(v)) if v else 0.0 for m, v in applied.items()},
    )
    return out, report


# --- files ---------------------------------------------------------------------------------


def read_png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_clip(path: Path) -> tuple[list[np.ndarray], float]:
    """An mp4's frames (RGB uint8) and its frame rate."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"{path}: no frames")
    return frames, float(fps)


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: float, crf: int = 24) -> None:
    """H.264 (yuv420p, +faststart) through the ffmpeg on PATH: what a browser plays inline."""
    h, w = frames[0].shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        command, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), check=True
    )


#: The clips `process` writes per mode, beside `<name>-a.mp4` (the model's pixels).
MODE_SUFFIX = {"bilinear": "b", "guided": "g"}


def process(
    render_path: Path,
    mask_path: Path,
    clip_path: Path,
    out: Path,
    name: str,
    *,
    share_path: Path | None = None,
    modes: Sequence[str] = MODES,
    crf: int = 24,
) -> dict:
    """One clip into what the page shows -- `<name>-a.mp4` (the model's pixels at the render's
    size) and per mode `<name>-b.mp4` / `<name>-g.mp4` (our render moved by the model's plant
    motion, bilinear / edge-aware) -- and its numbers."""
    render = read_png(render_path)
    soft = read_mask(mask_path)
    share = read_mask(share_path) if share_path is not None else None
    frames, fps = read_clip(clip_path)
    H, W = render.shape[:2]
    pixels = resize_frames(frames, W, H)
    moved, report = motion_only(render, frames, soft, share=share, modes=modes)
    report.psnr_outside_db = psnr_outside(pixels, render, soft)
    write_mp4(out / f"{name}-a.mp4", pixels, fps, crf)
    for mode, clip in moved.items():
        write_mp4(out / f"{name}-{MODE_SUFFIX[mode]}.mp4", clip, fps, crf)
    return {"name": name, "fps": fps, "modelSize": [frames[0].shape[1], frames[0].shape[0]]} | (
        report.to_json()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("process", help="a clip into its pixels (A) and motion-only (B) clips")
    run.add_argument("--render", type=Path, required=True)
    run.add_argument("--mask", type=Path, required=True, help="the feathered plant mask (PNG)")
    run.add_argument("--share", type=Path, help="the unfeathered plant share (PNG)")
    run.add_argument("--clip", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--modes", default=",".join(MODES), help=f"of {', '.join(MODES)}")
    run.add_argument("--crf", type=int, default=24)
    args = parser.parse_args(argv)
    result = process(
        args.render,
        args.mask,
        args.clip,
        args.out,
        args.name,
        share_path=args.share,
        modes=[m for m in args.modes.split(",") if m],
        crf=args.crf,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
