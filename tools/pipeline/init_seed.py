"""A Refine that starts from the preview's splat instead of from COLMAP's sparse points.

gsplat v1.5.3 cannot resume -- its `--ckpt` evaluates a checkpoint and trains nothing --
so "continue the preview" is not on offer without patching the trainer. What *is* on
offer is its initialisation. `create_splats_with_optimizers` (read at the v1.5.3 tag)
turns `parser.points` and `parser.points_rgb` -- `points3D.bin`, all of it -- into the
starting gaussians:

* **means** are the points, **colours** their RGB (as the SH DC term, higher bands 0);
* **scales** are `log(init_scale * sqrt(mean d^2 to the 3 nearest neighbours))`, so a
  denser cloud starts with smaller gaussians -- the right thing for a denser cloud;
* **opacities** are `init_opa` (0.5 in the `mcmc` preset) and **rotations** random.

So writing the preview's gaussian centres and colours into the dataset's `points3D.bin`
hands the Refine a dense, surface-hugging cloud in place of a sparse one: the geometry and
colour a 3,000-step preview already found. Shapes, opacities and view-dependent colour are
re-learned, which is cheap next to finding the surfaces. It is sound under MCMC because
nothing there assumes a sparse start: `relocate` moves dead gaussians (opacity below
0.005) onto live ones and `add` grows the count 5% per refine up to `cap_max`, so a seed
below the cap is simply a head start. The seed is kept at `budget` points (half the cap
by default) so there is room left to grow into.

**The tracks.** The same file is the depth loss's supervision: v1.5.3's parser turns each
point's track into `point_indices[image]`, and `--depth_loss` projects those points into
the frame. A preview gaussian has no observations, and inventing some (every frame it
projects into) would feed the depth loss points hidden behind surfaces. So the seed points
are **appended with empty tracks** beside COLMAP's own points, which keep theirs: the
parser loads a zero-length track (pycolmap's `SceneManager._load_points3D_bin` reshapes an
empty `2*0`-int record to `(0, 2)`), init sees every point, and the depth loss sees exactly
the observations it saw before. Every frame keeps its SfM points either way.

**Where the preview's splat comes from.** A Refine re-runs `train` in the same workdir,
which clears `out/` before the new trainer writes a new `trained.ply`. `checkpoint/` is the
one directory the executor keeps across attempts and re-runs, `CloudRunner` carries it to
and from the GPU box, and the worker's tidy-up leaves it alone. So every `train` run leaves
a compact seed there (centres, colours, opacities of the visible gaussians -- 19 bytes
each, a few MB) when it finishes, and a later run with `init_from: preview` reads it before
it trains. The seed records a fingerprint of the poses it is in: a run whose poses were
recomputed (retry-from-pose, or a lost workdir) is in another frame, and its seed is
refused rather than applied to the wrong reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sfm
from support_mask import SupportMask

__all__ = [
    "DEFAULT_SCHEDULE_SCALE",
    "MIN_OPACITY",
    "SEED_DIR",
    "Seed",
    "SeedApplied",
    "apply",
    "budget_for",
    "load",
    "poses_fingerprint",
    "save",
    "schedule_for",
]

#: Under the stage's `checkpoint/`.
SEED_DIR = "init-seed"
_ARRAYS = "seed.npz"
_META = "seed.json"
_VERSION = 1

#: Gaussians fainter than this are not seeded: MCMC would relocate them first thing.
MIN_OPACITY = 0.1

#: The schedule a Refine runs when it starts from a seed: the full one. Measured on the
#: spool capture (179 frames, L4, 2026-09-28; all three Refines cropped to the same
#: preview support mask): from the preview on half the schedule 25.11 dB / LPIPS 0.147;
#: from COLMAP's points on the full one 25.79 / 0.133; from the preview on the full one
#: 25.85 / 0.125 -- the best of the three, for $0.48 with the preview against $0.75. Half
#: a schedule skipped too much of MCMC's refinement; the seed's head start is worth
#: keeping, not cashing in as fewer steps.
DEFAULT_SCHEDULE_SCALE = 1.0

#: Of the seed points outside a region, one in this many is kept -- the rule
#: `training.crop_initial_points` applies to COLMAP's points, for the same reason.
OUTSIDE_EVERY = 10

#: With no cap to go by, the most seed points used.
DEFAULT_BUDGET = 250_000

#: Where to seed: a `training.Roi` sphere, a `SupportMask`, or None for everywhere.
Region = Any


def poses_fingerprint(poses: Path) -> str:
    """What makes a seed usable: the exact cameras and extrinsics it was trained against."""
    digest = hashlib.sha256()
    for name in ("cameras.bin", "images.bin"):
        path = poses / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    return digest.hexdigest()


@dataclass(frozen=True)
class Seed:
    xyz: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.uint8]
    alpha: npt.NDArray[np.float32]
    meta: Mapping[str, Any]

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


def save(
    checkpoint_dir: Path,
    columns: Mapping[str, Any],
    poses: Path,
    *,
    settings: Mapping[str, Any],
    sh_c0: float,
    scratch: Path,
) -> int:
    """Write the seed of a finished run. Returns how many gaussians it holds.

    Written in `scratch` (the stage's `work/`) and renamed into place, so the remote's
    checkpoint syncer, which runs on an interval, never ships a half-written seed or a
    temporary file.
    """
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float32)
    with np.errstate(over="ignore"):
        alpha = (1.0 / (1.0 + np.exp(-np.asarray(columns["opacity"], dtype=np.float64)))).astype(
            np.float32
        )
    dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)
    rgb = np.clip(np.round((sh_c0 * dc + 0.5) * 255.0), 0, 255)
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1) & (alpha >= MIN_OPACITY)
    directory = checkpoint_dir / SEED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    arrays = directory / _ARRAYS
    scratch.mkdir(parents=True, exist_ok=True)
    partial = scratch / f"{_ARRAYS}.partial"
    with partial.open("wb") as handle:
        np.savez(
            handle,
            xyz=xyz[keep],
            rgb=rgb[keep].astype(np.uint8),
            alpha=alpha[keep],
        )
    os.replace(partial, arrays)
    meta = {
        "version": _VERSION,
        "posesFingerprint": poses_fingerprint(poses),
        "gaussians": int(keep.sum()),
        "gaussiansTrained": int(xyz.shape[0]),
        "minOpacity": MIN_OPACITY,
        "trainedWith": dict(settings),
    }
    partial_meta = scratch / f"{_META}.partial"
    partial_meta.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial_meta, directory / _META)
    return int(keep.sum())


def load(checkpoint_dir: Path, poses: Path) -> tuple[Seed | None, str]:
    """The seed a previous run of this stage left, or None and why not."""
    directory = checkpoint_dir / SEED_DIR
    arrays, meta_path = directory / _ARRAYS, directory / _META
    if not arrays.is_file() or not meta_path.is_file():
        return None, "no earlier run of this stage left a seed (a preview from before seeds)"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(arrays) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32).reshape(-1, 3)
            rgb = np.asarray(data["rgb"], dtype=np.uint8).reshape(-1, 3)
            alpha = np.asarray(data["alpha"], dtype=np.float32).reshape(-1)
    except (OSError, ValueError, KeyError) as error:
        return None, f"the seed could not be read ({error})"
    if not isinstance(meta, dict) or meta.get("version") != _VERSION:
        return None, "the seed is from another version of this stage"
    if meta.get("posesFingerprint") != poses_fingerprint(poses):
        return None, (
            "the seed was trained against other poses (they were recomputed since), so it "
            "is in another frame"
        )
    if not (xyz.shape[0] == rgb.shape[0] == alpha.shape[0]) or xyz.shape[0] == 0:
        return None, "the seed holds no usable gaussians"
    return Seed(xyz=xyz, rgb=rgb, alpha=alpha, meta=meta), ""


def budget_for(cap_max: int | None, requested: int | None) -> int:
    """How many seed points to use: as asked, else half the MCMC cap, else a default."""
    if requested is not None:
        return max(1, requested)
    if cap_max is not None:
        return max(1, cap_max // 2)
    return DEFAULT_BUDGET


@dataclass(frozen=True)
class SeedApplied:
    """What seeding the initial points did, for `train_metrics.json`."""

    seed_gaussians: int
    inside: int
    outside_sampled: int
    duplicates: int
    budget: int
    seeded: int
    sfm_points: int
    trained_with: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "from": "preview",
            "seedGaussians": self.seed_gaussians,
            "insideRegion": self.inside,
            "outsideSampled": self.outside_sampled,
            "duplicatesDropped": self.duplicates,
            "budget": self.budget,
            "seedPoints": self.seeded,
            "sfmPoints": self.sfm_points,
            "initialPoints": self.seeded + self.sfm_points,
            "previewSettings": dict(self.trained_with),
        }


def _inside(region: Region, xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    if region is None:
        return np.ones(xyz.shape[0], dtype=bool)
    if isinstance(region, SupportMask) or callable(getattr(region, "contains", None)):
        # A support mask, or a block's region (`blocks.BlockRegion`): each answers itself.
        inside_region: npt.NDArray[np.bool_] = np.asarray(region.contains(xyz), dtype=bool)
        return inside_region
    centre = np.asarray(region.center, dtype=np.float64)
    distance = np.linalg.norm(xyz.astype(np.float64) - centre, axis=1)
    inside: npt.NDArray[np.bool_] = np.isfinite(distance) & (distance <= region.radius)
    return inside


def apply(sparse: Path, seed: Seed, region: Region, *, budget: int) -> SeedApplied:
    """Append the seed's points to `sparse/points3D.bin`, with empty tracks.

    Chosen: inside `region` (all of the seed when there is none) plus one in
    `OUTSIDE_EVERY` of the rest; exact duplicates dropped, because the trainer's
    nearest-neighbour scale is `log` of a distance and three coincident points make it
    `-inf`; then a deterministic uniform sample down to `budget`.
    """
    path = sparse / "points3D.bin"
    points = sfm.read_points3d(path)
    inside = _inside(region, seed.xyz)
    outside = np.flatnonzero(~inside)
    chosen = inside.copy()
    chosen[outside[::OUTSIDE_EVERY]] = True
    index = np.flatnonzero(chosen)
    _, first = np.unique(seed.xyz[index], axis=0, return_index=True)
    unique = index[np.sort(first)]
    duplicates = int(index.shape[0] - unique.shape[0])
    if unique.shape[0] > budget:
        rng = np.random.default_rng(0)
        unique = np.sort(rng.choice(unique, size=budget, replace=False))
    count = int(unique.shape[0])
    first_id = int(points.ids.max()) + 1 if len(points) else 1
    added = sfm.Points3D(
        ids=np.arange(first_id, first_id + count, dtype=np.uint64),
        xyz=seed.xyz[unique].astype(np.float64),
        rgb=seed.rgb[unique],
        error=np.zeros(count, dtype=np.float64),
        track=np.zeros((0, 2), dtype=np.uint32),
        track_offsets=np.zeros(count + 1, dtype=np.int64),
    )
    sfm.write_points3d(path, _concat(points, added))
    trained_with = seed.meta.get("trainedWith")
    return SeedApplied(
        seed_gaussians=seed.count,
        inside=int(inside.sum()),
        outside_sampled=int(outside[::OUTSIDE_EVERY].shape[0]),
        duplicates=duplicates,
        budget=budget,
        seeded=count,
        sfm_points=len(points),
        trained_with=trained_with if isinstance(trained_with, dict) else {},
    )


def _concat(a: sfm.Points3D, b: sfm.Points3D) -> sfm.Points3D:
    return sfm.Points3D(
        ids=np.concatenate([a.ids, b.ids]),
        xyz=np.concatenate([a.xyz, b.xyz]),
        rgb=np.concatenate([a.rgb, b.rgb]),
        error=np.concatenate([a.error, b.error]),
        track=np.concatenate([a.track, b.track]).astype(np.uint32),
        track_offsets=np.concatenate([a.track_offsets, a.track_offsets[-1] + b.track_offsets[1:]]),
    )


def schedule_for(requested: float | None, default: float = DEFAULT_SCHEDULE_SCALE) -> float:
    """The schedule a seeded run gets: `init_schedule_scale` if given, else the default."""
    value = default if requested is None else float(requested)
    if not math.isfinite(value):
        raise ValueError(f"init_schedule_scale must be a number, not {requested!r}")
    return value
