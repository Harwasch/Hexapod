"""How many gaussians a capture should train: its surface, measured in its own pixels.

**The problem this replaces.** `cap_max: 500000` was the same number for a teacup and a
building. On the spool-table capture (a ~2 m table, 173 frames) every one of nine runs
ended at exactly 500,000 (experiments/run_variants.py, 2026-09-27): the cap was binding
every time, so the scene was spread thinner than its data could support, while a small
object got more gaussians than it had detail for.

**The principle.** A splat needs about a fixed number of gaussians per *pixel of surface
at the finest resolution any frame saw it* -- its ground sampling distance, or footprint.
A 2 m table filmed from 1 m at 1,160 px focal has a footprint near 0.9 mm and about
2.5 million footprints of table top; the same table filmed from 4 m has a sixteenth of
that, and training cannot recover detail no frame recorded. So:

    budget = density x sum over the supported surface of (area / footprint^2)

`density` is the one constant, calibrated below. Everything else comes from the pose
stage's sparse model:

* **Footprint of a point** = depth / focal, for the observing camera that sees it
  largest (the smallest footprint in its track), in *training* pixels: the model's focal
  lengths are scaled by the training frames' size over the posed frames' size, exactly as
  gsplat's parser rescales its intrinsics (`training.build_dataset`). So a frame trained
  at 2,400 px is budgeted 2.25x one trained at 1,600, and the preview's 800 px a quarter.
* **Surface area**, from sparse points, is the area of the voxels they occupy: one face
  (edge squared) per occupied voxel, i.e. one surface per voxel. The voxel edge is set
  relative to the footprint of the points in it -- `VOXEL_FOOTPRINTS` of them -- by
  octave: a point whose footprint is between f and 2f (f the scene's median) goes into
  the grid whose edge is `VOXEL_FOOTPRINTS x f`, one twice as coarse holds the next
  octave, and so on. A fixed cell count across the scene would give a near object and
  the far background the same cell, which is the bug this module exists to avoid.
* Per occupied voxel the **median** finest footprint of its points is the voxel's
  footprint (robust to the odd badly triangulated point), and the voxel contributes
  `edge^2 / footprint^2` footprints -- a number that is unitless, so a model in metres,
  in millimetres or in COLMAP's arbitrary scale gets the same budget.

**Why 32 footprints to a voxel.** SfM points are sparse: 4,096 SIFT features on a 1600 px
frame are ~20 px apart, a third to a half of them triangulate, and the finest view of a
point is by construction closer than the average one. So the points on a surface are
roughly 30-60 finest-footprints apart. A voxel much smaller than that counts each point
as its own speck of surface (Truck at 16 footprints: 60,340 voxels from 136,029 points);
one much larger counts a corner clipped by a surface as a whole face. 32 is about one
point spacing. Measured on the real models below (2026-09-27), halving it roughly halves
every scene's estimate and doubling it roughly doubles it (Truck: 7.8M, 15.2M, 25.3M
footprints^2 at 16, 32, 64), so what matters for fairness between scenes is that it is
*fixed* -- the constant below is calibrated for 32 and would have to be recalibrated with
it.

**The constant, `DEFAULT_DENSITY` = 0.1 gaussians per footprint^2** -- one gaussian per
ten pixels of finest-seen surface. Calibrated on the one scene with a published answer,
Tanks and Temples Truck (experiments/benchmark.py: 3DGS's own `tandt_db` model, 251
frames trained at 979x546): this module measures 15.2 million footprints^2 there
(29,962 voxels), so 0.1 gives **1.52M**. The reasoning for landing there:

* 3DGS's Table 1 stores 411 MB for Tanks and Temples at 30k: at 248 bytes a gaussian that
  is ~1.7M averaged over Truck and Train, and its adaptive densification is the field's
  standing estimate of what a scene "wants". 1.5M is MCMC at about that count, where the
  MCMC paper reports it ahead of 3DGS.
* The same train stage with MCMC capped at 500k already matched 3DGS's published PSNR on
  Truck (25.12 against 25.19; the `recipe` config of the benchmark). So Truck is not
  starved at 500k; 1.5M is 3x that, headroom that gsplat's own MCMC table (1M, 2M, 3M on
  Mip-NeRF 360: 29.18, 29.53, 29.65 dB) says is worth having, not a count far past where
  the returns stop.
* The other real models on hand, measured the same day at the size each would train at:
  a 100-photo bulldozer set at 1600 px, 4.84M (-> 484k); three pose runs of one phone
  video at 900x1600 -- 51, 60 and 98 frames registered -- 4.22M, 7.57M and 8.08M (-> 422k,
  757k, 808k); an 87-frame 640x480 tree, 1.77M (-> the 200k floor). So a phone capture of
  a table-sized scene lands around the 500k every spool run hit, and one with more frames
  of a bigger scene above it -- the spool table (173 frames of a ~2 m table) was not on
  hand to measure; its `budget` in train_metrics.json is the check. Note the last three:
  a sparser model finds less surface, so the estimate is only as good as the pose run.

It is a recipe parameter (`gaussian_density`), and a phone's quality tier multiplies it
(`density_scale`): Quick 0.5, Best 2.

**The clamp.** At least `DEFAULT_FLOOR` (the preview's own cap: a full run never gets
fewer gaussians than the preview that forecast it), and at most the GPU's ceiling --
or `budget_max`, when a run sets one and it is lower -- recorded when it is the one
that applied:

* **GPU memory**, `memory_ceiling`: gsplat's own measurement of MCMC at 1M, 2M and 3M
  gaussians (docs/source/tests/eval.rst at v1.5.3, "Feature Ablation", A100, Mip-NeRF 360
  averaged over 7 scenes): peak `max_memory_allocated` 1.98, 3.43 and 4.99 GiB. A straight
  line through them is 1,616 bytes a gaussian plus 0.475 GiB. 944 of those bytes are the
  59 parameters of a degree-3 gaussian in float32 with their gradient and Adam's two
  moments (59 x 4 x 4), which do not depend on the image size; the other 672 are
  projection, SH evaluation and tile-intersection buffers, which grow with how many tiles
  a gaussian covers -- all of them are scaled with the training frame's pixel count here
  (the benchmark's frames average ~1.38 MP: three outdoor scenes at factor 4, four indoor
  at factor 2), which overstates them, the safe direction for a ceiling. The intercept is
  ~370 bytes a pixel (the image, the render, its alpha, their gradients, the fused SSIM's
  maps). The card's memory less 1.5 GiB (CUDA context and library workspaces, which
  `max_memory_allocated` does not see) is then shared out at two thirds, leaving a third
  for the caching allocator's fragmentation under MCMC's repeated `torch.cat` growth.
  On the L4 (24 GB): ~8.7M gaussians at 1600x900, ~5.4M at 2400x1350, ~12M at 979x546.
  **Packed rasterization** (`--packed`, on since `training.gsplat_argv` passes it) is not
  counted yet. gsplat's profile of the rasterizer alone (docs/source/tests/profile.rst:
  forward + backward, TITAN RTX) measured it 0.48 -> 0.35 GB on a small scene at batch 1
  (27% less) and 5.67 -> 3.08 GB at 49M gaussians (46% less), numerically unchanged;
  applied to `RASTER_BYTES` at the small-scene 27% -- the least it saved, and the case a
  phone orbit is, every frame seeing most of the scene -- that would be ~9.9M at 1600 px
  and ~6.6M at 2400. Those bytes were fitted to whole training runs, not to the
  rasterizer alone, so the saving stays headroom under the dense model until an L4 run
  with packed on measures `max_memory_allocated` against it (train_metrics.json's
  per-step memory; the plan's pass criterion is within 15% of the model).
* **`budget_max`**, an optional override: a run or recipe may cap the count below the
  GPU's ceiling. It used to be the binding one -- 2M, because `quality`, `place`,
  `thumbnail` and `package` each loaded the whole splat, measured at ~0.75 GB a million
  gaussians on the 2 GB Fly worker (apps/api/app/worker/README.md). They now read it a
  chunk at a time (`splat_io`, `splat_stream`, `outofcore`) in memory that does not grow
  with it -- quality, place, thumbnail and ground samples of an 8M-gaussian splat peaked
  at 193 MB (tests/test_bounded_memory.py) -- and `package` is made the same by the
  large-scene plan's out-of-core tiler (tools/captures/splat_tiles.py). So nothing after
  training bounds the count, and the recipe sets no `budget_max`; a deployment whose
  packager still loads the whole splat should set one for its worker.

An explicit integer `cap_max` is an override and is passed through untouched (the preview
preset's 200k is one); only `auto` computes. A model that cannot be read -- a hand-made
dataset, a test's placeholder bytes -- falls back to `FALLBACK_CAP`, the old fixed cap,
and says why rather than failing the run.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sfm

__all__ = [
    "AUTO",
    "DEFAULT_DENSITY",
    "DEFAULT_FLOOR",
    "DEFAULT_GPU_MEMORY_GB",
    "FALLBACK_CAP",
    "MAX_SCHEDULE_FACTOR",
    "REFERENCE_BUDGET",
    "VOXEL_FOOTPRINTS",
    "Budget",
    "Surface",
    "describe",
    "finest_footprints",
    "memory_ceiling",
    "parse_cap",
    "plan",
    "schedule_factor",
    "surface",
]

F64 = npt.NDArray[np.float64]

#: The `cap_max` value that asks for a budget computed from the capture.
AUTO = "auto"

#: Gaussians per footprint^2 of supported surface. See the module docstring.
DEFAULT_DENSITY = 0.1

#: The voxel edge, in the footprints of the points it holds. See the module docstring.
VOXEL_FOOTPRINTS = 32.0

#: The fewest gaussians an `auto` budget gives: the preview preset's own cap
#: (apps/web/src/upload/options.ts PREVIEW_TRAIN), so a full run is never thinner than
#: the preview that forecast it.
DEFAULT_FLOOR = 200_000

#: The cap when the model cannot be measured: the fixed number every run used before.
FALLBACK_CAP = 500_000

#: The L4's memory, the tier `photo-reconstruct` trains on. A10 is also 24 GB.
DEFAULT_GPU_MEMORY_GB = 24.0

#: The count gsplat's 30k-step schedule is the default for (`MCMCStrategy.cap_max`), and
#: the smallest count in its own MCMC table. Budgets above it may get a longer maximum
#: schedule (`schedule_factor`).
REFERENCE_BUDGET = 1_000_000
#: The longest maximum schedule, as a multiple of the recipe's (30k -> 60k).
MAX_SCHEDULE_FACTOR = 2.0

# --- the memory model (module docstring, "GPU memory") ---------------------------------

GIB = 1024**3
#: 59 float32 parameters x (value, gradient, Adam exp_avg, exp_avg_sq).
PARAMETER_BYTES = 59 * 4 * 4
#: gsplat's measured 1,616 bytes a gaussian, less the parameters: all taken to scale
#: with the training frame's pixel count.
RASTER_BYTES = 1616 - PARAMETER_BYTES
#: The mean pixel count of the frames that measurement trained on.
REFERENCE_PIXELS = 1_380_000
#: The measurement's intercept, 0.475 GiB, over those pixels.
PIXEL_BYTES = 370
#: CUDA context and cuBLAS/cuDNN workspaces, which `max_memory_allocated` does not count.
RESERVED_BYTES = int(1.5 * GIB)
#: The share of the rest the modelled peak may use; the remainder is for fragmentation.
USABLE_FRACTION = 2.0 / 3.0


def parse_cap(value: object) -> int | str | None:
    """`cap_max` as given: None (no cap), `"auto"`, or a positive integer override."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == AUTO:
        return AUTO
    if isinstance(value, bool):
        raise ValueError(f"cap_max must be a positive integer or {AUTO!r}, not {value!r}")
    try:
        number = int(str(value))
    except ValueError:
        raise ValueError(f"cap_max must be a positive integer or {AUTO!r}, not {value!r}") from None
    if number <= 0:
        raise ValueError(f"cap_max must be a positive integer or {AUTO!r}, not {value!r}")
    return number


def memory_ceiling(pixels: int, gpu_memory_gb: float = DEFAULT_GPU_MEMORY_GB) -> int:
    """The most gaussians gsplat can train in `gpu_memory_gb` (decimal GB, as a card is
    sold) on frames of `pixels` pixels, by the model in the module docstring."""
    usable = (gpu_memory_gb * 1e9 - RESERVED_BYTES) * USABLE_FRACTION - pixels * PIXEL_BYTES
    per_gaussian = PARAMETER_BYTES + RASTER_BYTES * pixels / REFERENCE_PIXELS
    return max(0, int(usable // per_gaussian))


def schedule_factor(cap: int) -> float:
    """How much longer than the recipe's schedule a run of `cap` gaussians may go.

    `sqrt(cap / REFERENCE_BUDGET)`, between 1 and `MAX_SCHEDULE_FACTOR`: 1M or fewer keep
    the 30k gsplat's MCMC was measured on, 2M may run 1.41x, 4M or more 2x. The square
    root, not the ratio, because the evidence for needing more steps is thin -- gsplat's
    own 1M/2M/3M runs all used 30k and still gained with count -- and the extension is
    a *maximum*: the convergence rule (`convergence.py`) ends the tail when held-out
    quality has stopped improving. Rounded to hundredths, as `schedule_scale` is.
    """
    if cap <= 0:
        return 1.0
    return round(min(MAX_SCHEDULE_FACTOR, max(1.0, math.sqrt(cap / REFERENCE_BUDGET))), 2)


def finest_footprints(model: Path, *, pixel_scale: float = 1.0) -> tuple[F64, F64]:
    """Every sparse point's centre, and its finest footprint in training pixels.

    Footprint = depth / focal for each observation in the point's track (depth along the
    observing camera's optical axis, COLMAP's `R x + t`), the smallest one kept: the
    camera that saw that bit of surface largest. `pixel_scale` is the training frames'
    size over the posed frames' (focal lengths scale with it). A point no camera sees in
    front of itself gets an infinite footprint, which `surface` leaves out.

    The focal length is the geometric mean of fx and fy where a model has both, so a
    footprint is the side of a square pixel of the same area.
    """
    colmap = sfm.read_model(model)
    cameras = {camera.id: camera for camera in colmap.cameras}
    images = colmap.images
    points = sfm.read_points3d(model / "points3D.bin")
    if not images or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    size = max(image.id for image in images) + 1
    axis = np.zeros((size, 3), dtype=np.float64)
    offset = np.zeros(size, dtype=np.float64)
    focal = np.full(size, np.nan, dtype=np.float64)
    for image in images:
        axis[image.id] = image.rotation[2]
        offset[image.id] = image.tvec[2]
        camera = cameras.get(image.camera_id)
        if camera is not None:
            focal[image.id] = _focal(camera) * pixel_scale
    lengths = np.diff(points.track_offsets)
    rows = np.repeat(np.arange(len(points)), lengths)
    ids = points.track[:, 0].astype(np.int64) if len(points.track) else np.zeros(0, np.int64)
    known = ids < size
    ids = np.where(known, ids, 0)
    depth = np.einsum("ij,ij->i", axis[ids], points.xyz[rows]) + offset[ids]
    with np.errstate(divide="ignore", invalid="ignore"):
        each = np.where(known & (depth > 0) & (focal[ids] > 0), depth / focal[ids], np.inf)
    finest = np.full(len(points), np.inf, dtype=np.float64)
    np.minimum.at(finest, rows, each)
    return points.xyz, finest


def _focal(camera: sfm.Camera) -> float:
    named = dict(zip(camera.param_names, camera.params, strict=False))
    if "fx" in named and "fy" in named:
        return math.sqrt(abs(named["fx"] * named["fy"]))
    return camera.focal_px


@dataclass(frozen=True)
class Surface:
    """The supported surface of a set of points, measured in their own footprints."""

    points: int
    voxels: int
    #: Sum of voxel faces, in the model's units squared (COLMAP's scale, often arbitrary).
    area: float
    #: Sum of face / footprint^2: the surface in finest-view pixels. Unitless.
    footprints: float
    #: The median finest footprint, in model units per training pixel.
    median_footprint: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "points": self.points,
            "voxels": self.voxels,
            "area": _round(self.area),
            "footprints": round(self.footprints),
            "medianFootprint": _round(self.median_footprint),
        }


EMPTY = Surface(points=0, voxels=0, area=0.0, footprints=0.0, median_footprint=None)


def surface(
    xyz: npt.ArrayLike, footprint: npt.ArrayLike, *, voxel_footprints: float = VOXEL_FOOTPRINTS
) -> Surface:
    """Occupied voxels, each `voxel_footprints` of its own points' footprints across.

    Octaves of footprint about the median `m`: a point with footprint in
    `[m 2^k, m 2^(k+1))` goes into grid `k`, whose edge is `voxel_footprints m 2^k`. Each
    occupied (grid, cell) is one face of surface, and contributes `edge^2 / f^2` with `f`
    the median footprint of its points. A surface whose points straddle two octaves is
    counted in both where they meet, which overstates it by at most a band one voxel wide.
    """
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    size = np.asarray(footprint, dtype=np.float64).reshape(-1)
    good = np.isfinite(points).all(axis=1) & np.isfinite(size) & (size > 0)
    points, size = points[good], size[good]
    if size.size == 0:
        return EMPTY
    median = float(np.median(size))
    octave = np.floor(np.log2(size / median)).astype(np.int64)
    edge = voxel_footprints * median * np.exp2(octave)
    cells = np.floor(points / edge[:, None]).astype(np.int64)
    keys = np.concatenate([octave[:, None], cells], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # Median footprint per voxel, vectorised: sort by (voxel, footprint), then the middle.
    order = np.lexsort((size, inverse))
    ordered = size[order]
    counts = np.bincount(inverse, minlength=unique.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    low = ordered[starts + (counts - 1) // 2]
    high = ordered[starts + counts // 2]
    voxel_footprint = 0.5 * (low + high)
    face = (voxel_footprints * median * np.exp2(unique[:, 0])) ** 2
    return Surface(
        points=int(size.size),
        voxels=int(unique.shape[0]),
        area=float(face.sum()),
        footprints=float((face / voxel_footprint**2).sum()),
        median_footprint=median,
    )


@dataclass(frozen=True)
class Budget:
    """The gaussian cap a run trains under, and everything it was computed from."""

    #: `auto` (computed here), `explicit` (an integer `cap_max`, passed through) or
    #: `fallback` (asked for `auto`, but the model could not be measured).
    mode: str
    cap: int
    #: density x density_scale x footprints, before the clamp. None unless `auto`.
    raw: int | None = None
    #: Which bound applied: `floor`, `gpu-memory` or `budget-max`; None when none did.
    clamp: str | None = None
    floor: int = DEFAULT_FLOOR
    memory_ceiling: int | None = None
    budget_max: int | None = None
    density: float = DEFAULT_DENSITY
    density_scale: float = 1.0
    voxel_footprints: float = VOXEL_FOOTPRINTS
    #: The region a Refine trains in, counted in full, and the rest of the scene,
    #: counted at `outside_weight` (a tenth: the share of its SfM points the crop keeps).
    #: Without a region everything is `inside`.
    inside: Surface = EMPTY
    outside: Surface | None = None
    outside_weight: float = 0.0
    train_size: tuple[int, int] | None = None
    pixel_scale: float | None = None
    gpu_memory_gb: float = DEFAULT_GPU_MEMORY_GB
    reason: str = ""

    @property
    def footprints(self) -> float:
        extra = 0.0 if self.outside is None else self.outside_weight * self.outside.footprints
        return self.inside.footprints + extra

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "capMax": self.cap,
            "raw": self.raw,
            "clamp": self.clamp,
            "floor": self.floor,
            "memoryCeiling": self.memory_ceiling,
            "budgetMax": self.budget_max,
            "density": self.density,
            "densityScale": self.density_scale,
            "voxelFootprints": self.voxel_footprints,
            "footprints": round(self.footprints),
            "surface": self.inside.to_dict(),
            "outside": None if self.outside is None else self.outside.to_dict(),
            "outsideWeight": self.outside_weight if self.outside is not None else None,
            "trainImageSize": None if self.train_size is None else list(self.train_size),
            "pixelScale": _round(self.pixel_scale),
            "gpuMemoryGb": self.gpu_memory_gb,
            "reason": self.reason or None,
        }


def plan(
    requested: int | str | None,
    *,
    model: Path,
    train_size: tuple[int, int] | None,
    pixel_scale: float | None,
    region_contains: Callable[[F64], npt.NDArray[np.bool_]] | None = None,
    outside_weight: float = 0.1,
    density: float = DEFAULT_DENSITY,
    density_scale: float = 1.0,
    floor: int = DEFAULT_FLOOR,
    budget_max: int | None = None,
    gpu_memory_gb: float = DEFAULT_GPU_MEMORY_GB,
    voxel_footprints: float = VOXEL_FOOTPRINTS,
) -> Budget | None:
    """The cap for `requested` (`parse_cap`'s result): None for no cap at all.

    `train_size` is the size of the frames the trainer reads and `pixel_scale` their size
    over the posed frames' (both from the built dataset: `training.build_dataset` may have
    shrunk them). `region_contains` is a Refine's region -- a support mask's or an ROI's
    membership test -- whose points count in full while the rest count at
    `outside_weight`: the trainer is given a tenth of the points outside to explain the
    background with (`training.ROI_OUTSIDE_EVERY`), and a tenth of that surface is what
    those gaussians are budgeted.
    """
    if requested is None:
        return None
    pixels = 0 if train_size is None else train_size[0] * train_size[1]
    ceiling = memory_ceiling(pixels, gpu_memory_gb) if pixels else None
    common: dict[str, Any] = {
        "floor": floor,
        "memory_ceiling": ceiling,
        "budget_max": budget_max,
        "density": density,
        "density_scale": density_scale,
        "voxel_footprints": voxel_footprints,
        "train_size": train_size,
        "pixel_scale": pixel_scale,
        "gpu_memory_gb": gpu_memory_gb,
    }
    if isinstance(requested, int):
        return Budget(mode="explicit", cap=requested, **common)
    if not (density > 0 and density_scale > 0):
        raise ValueError(
            f"gaussian_density and density_scale must be positive, not {density!r} and "
            f"{density_scale!r}"
        )
    if train_size is None or pixel_scale is None or not pixel_scale > 0:
        return Budget(
            mode="fallback",
            cap=FALLBACK_CAP,
            reason="the training frames' size is unknown, so no footprint can be measured",
            **common,
        )
    try:
        xyz, finest = finest_footprints(model, pixel_scale=pixel_scale)
    except (OSError, ValueError, KeyError, IndexError, struct.error) as error:
        return Budget(
            mode="fallback",
            cap=FALLBACK_CAP,
            reason=f"the sparse model could not be read: {error}",
            **common,
        )
    measured = np.isfinite(finest)
    if not measured.any():
        return Budget(
            mode="fallback",
            cap=FALLBACK_CAP,
            reason="no sparse point is in front of a camera that sees it",
            **common,
        )
    if region_contains is None:
        inside, outside, weight = surface(xyz, finest, voxel_footprints=voxel_footprints), None, 0.0
    else:
        mask = np.asarray(region_contains(xyz), dtype=bool)
        inside = surface(xyz[mask], finest[mask], voxel_footprints=voxel_footprints)
        outside = surface(xyz[~mask], finest[~mask], voxel_footprints=voxel_footprints)
        weight = outside_weight
    budget = Budget(
        mode="auto", cap=0, inside=inside, outside=outside, outside_weight=weight, **common
    )
    raw = round(density * density_scale * budget.footprints)
    cap, clamp = raw, None
    if cap < floor:
        cap, clamp = floor, "floor"
    # The ceilings after the floor, so they win over it: a floor the GPU cannot hold is a
    # run that runs out of memory, which is worse than a thin one.
    upper = [
        (bound, name)
        for bound, name in ((ceiling, "gpu-memory"), (budget_max, "budget-max"))
        if bound is not None
    ]
    if upper:
        bound, name = min(upper)
        if cap > bound:
            cap, clamp = max(1, bound), name
    return Budget(
        mode="auto",
        cap=cap,
        raw=raw,
        clamp=clamp,
        inside=inside,
        outside=outside,
        outside_weight=weight,
        **common,
    )


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def describe(budget: Budget) -> str:
    """One log line: what was measured and what it came to."""
    if budget.mode == "explicit":
        return f"cap_max {budget.cap} given; no budget computed"
    if budget.mode == "fallback":
        return f"cap_max auto -> {budget.cap} (fallback: {budget.reason})"
    surface_ = budget.inside
    parts = [
        f"cap_max auto -> {budget.cap}",
        f"{budget.footprints / 1e6:.2f}M footprints^2 of surface "
        f"({surface_.voxels} voxels of {budget.voxel_footprints:g} footprints, "
        f"median footprint {surface_.median_footprint:.4g} at "
        f"{budget.train_size[0]}x{budget.train_size[1]})"
        if budget.train_size is not None and surface_.median_footprint is not None
        else f"{budget.footprints / 1e6:.2f}M footprints^2 of surface",
        f"x {budget.density:g}"
        + ("" if budget.density_scale == 1.0 else f" x {budget.density_scale:g}")
        + f" = {budget.raw}",
    ]
    if budget.outside is not None:
        parts.append(
            f"region {budget.inside.footprints / 1e6:.2f}M + outside "
            f"{budget.outside.footprints / 1e6:.2f}M x {budget.outside_weight:g}"
        )
    if budget.clamp is not None:
        parts.append(f"clamped to the {budget.clamp}")
    return "; ".join(parts)
