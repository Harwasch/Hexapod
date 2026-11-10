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

    density target = density x sum over the supported surface of (area / footprint^2)

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

**The default, `budget_mode: auto`.** `train_admission.py` uses this density target
(with the floor and optional lower `budget_max`) to select the smallest allowed GPU,
then applies rule B: the recipe's base tier keeps the target; a promoted tier fills
its ceiling. Optional `budget_headroom` is off by default (None); when set it budgets
min(ceiling, headroom x target) on every tier. The beach's LPIPS improved through
20M (2.5x its 8.05M target) with flat PSNR at opacity_reg 0.002; there is no evidence
beyond that. Targets beyond the maximum split into blocks, each applying the same
rule to its own density target and allocated tier. `budget_mode: density` forces
headroom 1.0 and keeps the density-sized cap. `raw`
always preserves the measured density target in these two modes. The floor and
`budget_max` still bound the final cap. The survey streams all observations without
sampling: O(points + images + chunk) memory, not O(observations).

**The density clamp.** At least `DEFAULT_FLOOR` (the preview's own cap: a full run never gets
fewer gaussians than the preview that forecast it), and at most the GPU's ceiling --
or `budget_max`, when a run sets one and it is lower -- recorded when it is the one
that applied:

* **GPU memory**, `memory_ceiling`: steady allocated memory plus a per-view tile
  intersection burst and allocator fragmentation. See the constants and helpers below
  for the beach SH3 evidence and pinned gsplat buffer accounting. Admission retains
  the historical ceiling and one-third headroom, so this correction cannot increase
  an auto cap. Packed mode's savings are not an extra discount.
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
from dataclasses import dataclass, replace
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

#: Beach LPIPS improved through 20M / 8.05M density target; no evidence beyond 2.5x.
DEFAULT_HEADROOM: float | None = None

#: Maximum observations evaluated together, independent of total track count/length.
OBSERVATION_CHUNK = 65_536

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

#: Each GPU tier's memory, decimal GB as the cards are sold (`modal_adapter.GPU_NAMES`:
#: `a100` is the 80 GB part). The ceiling is modelled on the GPU the stage was *placed*
#: on, so a run sent to a bigger card is budgeted for it rather than for an L4.
GPU_MEMORY_GB: dict[str, float] = {
    "t4": 16.0,
    "l4": 24.0,
    "a10": 24.0,
    "l40s": 48.0,
    "a100": 80.0,
    "a100-40gb": 40.0,
    "h100": 80.0,
    "h200": 141.0,
    "b200": 180.0,
}


def gpu_memory_for(tier: str | None, given: object = None) -> float:
    """`gpu_memory_gb` as the run gives it, else the memory of `tier`, else the L4's."""
    if given is not None:
        return float(str(given))
    if tier is not None and tier in GPU_MEMORY_GB:
        return GPU_MEMORY_GB[tier]
    return DEFAULT_GPU_MEMORY_GB


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

#: Beach 4769ee65 SH3 allocated fit: 1440 B/N + 0.18 decimal GB at native pixels.
BURST_REFERENCE_PIXELS = 1080 * 1920
STEADY_RASTER_BYTES = 1440 - PARAMETER_BYTES
STEADY_PIXEL_BYTES = 0.18e9 / BURST_REFERENCE_PIXELS
#: 7ba40224 failed requesting 2.34 GiB at 10M/native. Conservatively treat that
#: as one int64 intersection key buffer. At gsplat 512d366b, Intersect.cpp's
#: intersect_tile allocates int64 keys + int32 IDs and empty_like sorted copies:
#: 2 * (8 + 4) = 24 B/intersection, three times that one 8 B key allocation.
#: IntersectTile.cu uses CUB DoubleBuffer; its cudaMallocAsync scratch is outside
#: torch's allocated peak. The context reserve/headroom cover unmeasured scratch.
#: Tile coverage (fixed 16px tiles) grows with projected area: N * frame pixels
#: for unchanged views/geometry. Visibility, radii and clipping vary by view, so
#: this is a conservative envelope, not a measured universal intersection rate.
BURST_REFERENCE_COUNT = 10_000_000
BURST_REFERENCE_BYTES = 3 * 2.34 * GIB
#: Reserved minus allocated was 2.46 GiB at OOM. Keep at least 2.5 GiB, or the
#: existing one-third headroom if larger; do not add both fragmentation reserves.
FRAGMENTATION_BYTES = 2.5 * GIB


def view_burst_bytes(count: int, pixels: int) -> float:
    """Full intersection sorting buffers, scaled from the failed native view."""
    return BURST_REFERENCE_BYTES * count / BURST_REFERENCE_COUNT * pixels / BURST_REFERENCE_PIXELS


def steady_memory_bytes(count: int, pixels: int) -> float:
    """Beach SH3 allocated fit; parameters stay fixed as resolution changes."""
    slope = PARAMETER_BYTES + STEADY_RASTER_BYTES * pixels / BURST_REFERENCE_PIXELS
    return count * slope + pixels * STEADY_PIXEL_BYTES


def peak_memory_bytes(count: int, pixels: int) -> float:
    """Physical peak envelope, including observed allocator fragmentation, in bytes.

    Excludes CUDA context/workspaces (RESERVED_BYTES); unlike admission this does
    not add the historical one-third policy margin. It deliberately overestimates
    the successful 7.25M run's observed <=12.7 GiB reserved peak.
    """
    return (
        steady_memory_bytes(count, pixels) + view_burst_bytes(count, pixels) + FRAGMENTATION_BYTES
    )


def memory_fits(count: int, pixels: int, gpu_memory_gb: float = DEFAULT_GPU_MEMORY_GB) -> bool:
    """Admission under the shared burst model and retained historical safety policy."""
    return count <= memory_ceiling(pixels, gpu_memory_gb)


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
    available = gpu_memory_gb * 1e9 - RESERVED_BYTES
    slope = PARAMETER_BYTES + STEADY_RASTER_BYTES * pixels / BURST_REFERENCE_PIXELS
    slope += view_burst_bytes(1, pixels)
    intercept = pixels * STEADY_PIXEL_BYTES
    # The larger of measured fragmentation and the historical one-third margin.
    usable = min(available - FRAGMENTATION_BYTES, available * USABLE_FRACTION)
    burst_cap = max(0, int((usable - intercept) // slope))
    # Never loosen the pre-burst auto model at any tier/resolution.
    legacy_slope = PARAMETER_BYTES + RASTER_BYTES * pixels / REFERENCE_PIXELS
    legacy_cap = max(0, int((available * USABLE_FRACTION - pixels * PIXEL_BYTES) // legacy_slope))
    return min(burst_cap, legacy_cap)


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

    Observations stream in chunks of at most `OBSERVATION_CHUNK`, including within
    a single long track. All observations are retained; results match the whole-array
    calculation exactly. Memory is O(points + images + chunk), not O(observations).

    The focal length is the geometric mean of fx and fy where a model has both, so a
    footprint is the side of a square pixel of the same area.
    """
    colmap = sfm.read_model(model)
    cameras = {camera.id: camera for camera in colmap.cameras}
    images = colmap.images
    if not images or colmap.points3d == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    # Compact lookup also bounds memory when COLMAP image IDs have large gaps.
    images = tuple(sorted(images, key=lambda image: image.id))
    image_ids = np.asarray([image.id for image in images], dtype=np.uint32)
    axis = np.asarray([image.rotation[2] for image in images])
    offset = np.asarray([image.tvec[2] for image in images])
    focal = np.asarray(
        [
            _focal(cameras[image.camera_id]) * pixel_scale if image.camera_id in cameras else np.nan
            for image in images
        ]
    )
    # Keep only the output (32 bytes/point) plus fixed observation buffers. A single
    # arbitrarily long track is split too; no sampling or change to the arithmetic.
    xyz = np.empty((colmap.points3d, 3), dtype=np.float64)
    finest = np.full(colmap.points3d, np.inf, dtype=np.float64)
    rows = np.empty(OBSERVATION_CHUNK, dtype=np.int64)
    ids = np.empty(OBSERVATION_CHUNK, dtype=np.uint32)

    def reduce_chunk(count: int) -> None:
        slots = np.searchsorted(image_ids, ids[:count])
        np.minimum(slots, len(image_ids) - 1, out=slots)
        known = image_ids[slots] == ids[:count]
        depth = np.einsum("ij,ij->i", axis[slots], xyz[rows[:count]]) + offset[slots]
        with np.errstate(divide="ignore", invalid="ignore"):
            each = np.where(known & (depth > 0) & (focal[slots] > 0), depth / focal[slots], np.inf)
        np.minimum.at(finest, rows[:count], each)

    filled = 0
    with (model / "points3D.bin").open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        if count != colmap.points3d:
            raise ValueError("sparse point count changed during density survey")
        for row in range(count):
            record = handle.read(51)  # id, xyz, rgb, error, track length
            xyz[row] = struct.unpack_from("<ddd", record, 8)
            remaining = struct.unpack_from("<Q", record, 43)[0]
            while remaining:
                take = min(remaining, OBSERVATION_CHUNK - filled)
                track = np.frombuffer(handle.read(8 * take), dtype="<u4").reshape(take, 2)
                ids[filled : filled + take] = track[:, 0]
                rows[filled : filled + take] = row
                remaining -= take
                filled += take
                if filled == OBSERVATION_CHUNK:
                    reduce_chunk(filled)
                    filled = 0
        if filled:
            reduce_chunk(filled)
    return xyz, finest


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
    budget_mode: str = "auto"
    budget_headroom: float | None = DEFAULT_HEADROOM
    gpu_tier: str | None = None
    recipe_gpu_tier: str | None = None
    max_gpu_tier: str | None = None
    tier_reason: str = ""

    @property
    def footprints(self) -> float:
        extra = 0.0 if self.outside is None else self.outside_weight * self.outside.footprints
        return self.inside.footprints + extra

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "budgetMode": self.budget_mode,
            "budgetHeadroom": 1.0 if self.budget_mode == "density" else self.budget_headroom,
            "gpuTier": self.gpu_tier,
            "recipeGpuTier": self.recipe_gpu_tier,
            "maxGpuTier": self.max_gpu_tier,
            "tierReason": self.tier_reason,
            "capMax": self.cap,
            "raw": self.raw,
            "densityTarget": self.raw if self.budget_mode != "gpu-max" else None,
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
    images_per_step: int = 1,
    budget_mode: str = "auto",
    budget_headroom: float | None = DEFAULT_HEADROOM,
) -> Budget | None:
    """The cap for `requested` (`parse_cap`'s result): None for no cap at all.

    `train_size` is the size of the frames the trainer reads and `pixel_scale` their size
    over the posed frames' (both from the built dataset: `training.build_dataset` may have
    shrunk them). `region_contains` is a Refine's region -- a support mask's or an ROI's
    membership test -- whose points count in full while the rest count at
    `outside_weight`: the trainer is given a tenth of the points outside to explain the
    background with (`training.ROI_OUTSIDE_EVERY`), and a tenth of that surface is what
    those gaussians are budgeted.

    `images_per_step` is the trainer's `batch_size`: a step rasterises that many frames at
    once, so the image and raster terms of the memory model are that many frames' worth
    (`train_size` stays one frame's size in the budget's record).
    """
    if requested is None:
        return None
    pixels = 0 if train_size is None else train_size[0] * train_size[1]
    ceiling = memory_ceiling(pixels * max(1, images_per_step), gpu_memory_gb) if pixels else None
    if budget_headroom is not None and (
        not math.isfinite(budget_headroom) or budget_headroom < 1.0
    ):
        raise ValueError("budget_headroom must be finite and at least 1.0")
    headroom = 1.0 if budget_mode == "density" else budget_headroom
    common: dict[str, Any] = {
        "budget_headroom": headroom,
        "budget_mode": budget_mode,
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
    if budget_mode not in ("auto", "density", "gpu-max"):
        raise ValueError(f"unknown budget_mode {budget_mode!r}")
    if budget_mode == "gpu-max" and ceiling is not None:
        target = min(ceiling, budget_max) if budget_max is not None else ceiling
        return Budget(mode="auto", cap=target, raw=target, **common)
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
    return apply_tier_policy(replace(budget, raw=raw))


def apply_tier_policy(budget: Budget, *, promoted: bool = False) -> Budget:
    """Apply rule B or explicit headroom within the tier and recipe limits."""
    if budget.mode != "auto":
        return budget
    raw = budget.raw or 0
    headroom = 1.0 if budget.budget_mode == "density" else budget.budget_headroom
    fill = budget.budget_mode == "auto" and headroom is None and promoted
    cap = (
        budget.memory_ceiling
        if fill and budget.memory_ceiling is not None
        else round(raw * (headroom if headroom is not None else 1.0))
    )
    clamp = None
    if budget.budget_mode == "gpu-max":
        cap = raw
    elif cap < budget.floor:
        cap, clamp = budget.floor, "floor"
    upper = [
        (bound, name)
        for bound, name in (
            (budget.memory_ceiling, "gpu-memory"),
            (budget.budget_max, "budget-max"),
        )
        if bound is not None
    ]
    if upper:
        bound, name = min(upper)
        if cap > bound:
            cap, clamp = max(1, bound), name
    return replace(budget, cap=cap, clamp=clamp)


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
    if budget.budget_mode == "gpu-max":
        return f"cap_max auto -> {budget.cap} (gpu-max; {budget.tier_reason or 'memory ceiling'})"
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
    if budget.budget_mode == "auto":
        parts.append(
            "rule B" if budget.budget_headroom is None else f"headroom x {budget.budget_headroom:g}"
        )
    if budget.tier_reason:
        parts.append(budget.tier_reason)
    if budget.clamp is not None:
        parts.append(f"clamped to the {budget.clamp}")
    return "; ".join(parts)
