"""Dispatching a 3DGS training run: the dataset it needs, the argv, the PLY it writes.

**No training run has been executed anywhere in this repository.** `gsplat` needs CUDA
and there is no GPU on any machine this was written on. What is built here is the
*dispatch*, and since 2026-09-23 it is checked against the trainer it dispatches rather
than against a memory of it: gsplat **v1.5.3**'s `examples/simple_trainer.py`, read at
that tag, and the argv below **parsed by that file's own `tyro` CLI** (tyro 1.0.16) in a
CPU venv holding the exact wheel the training image installs
(`gsplat-1.5.3+pt24cu124-cp310`), torch 2.4.1 and the example's pinned requirements.
Parsing is all a CPU can do: `Runner` allocates CUDA tensors in its constructor.

Reading the source corrected four things the first version of this file transcribed from
memory, and each one would have cost a GPU run to discover:

* **`--ckpt` is evaluation, not resume.** `main()` in v1.5.3: "if cfg.ckpt is not None:
  # run eval only". The old argv passed the last checkpoint on a second attempt, so a
  preempted run would have evaluated the stale checkpoint, written no PLY, and failed.
  There is no resume in `simple_trainer.py` at all (`init_step = 0`), so a second
  attempt now **restarts from zero** and says so; nothing is passed that pretends
  otherwise.
* **No PLY without `--save_ply`.** It defaults to False. Without it the stage would have
  trained for half an hour and then failed on "wrote no .ply".
* **`normalize_world_space` defaults to True**, which rotates, recentres and rescales
  the scene (`datasets/normalize.py`: `similarity_from_cameras` then
  `align_principal_axes`) and exports the PLY in *that* frame. Every downstream
  transform -- the EXIF similarity, the camera-up estimate -- is expressed in COLMAP's
  frame, so `--no-normalize-world-space` is passed and the PLY comes back in the frame
  the poses are in.
* **The file names**: `ply/point_cloud_<step>.ply` and `stats/val_step<step:04d>.json`
  (no rank suffix on the validation stats; `train_step*_rank<n>.json` carries only
  memory, time and count), where `<step>` is the zero-based index of the last step --
  `max_steps - 1`. And stdout's progress line is `Step:  <n> {..., 'num_GS': <n>}`.

`parse_metrics` is still written to survive being wrong: it reads whatever `stats/*.json`
it finds, falls back to the trainer's stdout, and records `null` for anything it cannot
find rather than inventing a number.

The dataset layout is COLMAP's, because that is what `gsplat`'s parser reads and it is
exactly what the `pose` stage already produced -- `images/` beside `sparse/0/`. Building
it in `work/` rather than reshaping the `poses` artifact keeps the artifact COLMAP's own
model, which is what `opensplat` and `nerfstudio` want too. Two things the dataset may
differ from the artifact in, both read out of v1.5.3's `examples/datasets/colmap.py`:

* **Smaller images, same poses** (`train_max_side`). The parser loads the first image,
  divides its size by the COLMAP camera's, and scales every intrinsic matrix and image
  size by that ratio before it builds the undistortion maps. So frames downscaled in
  place under `images/`, with `--data_factor 1`, train against correctly scaled
  intrinsics and unchanged extrinsics. `--data_factor N` is the other route and the
  worse one here: it wants an `images_N/` directory, is integers only, and for JPEGs
  re-derives PNGs from `images/` itself. Checked by loading a downscaled dataset with
  that parser on CPU: `K` came back scaled by the ratio of the sizes.
* **Fewer initial points** (`roi`). `points3D.bin` is the trainer's initialisation (the
  `sfm` init) and, with `--depth_loss`, its depth supervision: the parser turns each
  point's track into per-image `point_indices`, and the dataset projects those into the
  frame. The crop rewrites that one file and keeps every track that survives, so both
  uses see a consistent, smaller model.

**2DGS is not offered, and not because it was not looked at.** gsplat v1.5.3's
`examples/simple_trainer_2dgs.py` was read at the tag. It cannot produce what this
pipeline consumes: it has no `--save_ply` and no PLY writer at all (the splat leaves it
only as `ckpts/ckpt_<step>.pt`, a torch state dict); it has no MCMC strategy, so
no `cap_max` and no bound on the gaussian count, which is what bounds this stage's cost;
and its primitives are surfels -- flat oriented discs -- rasterised by
`rasterization_2dgs`, a ray-splat intersection that the 3DGS renderer the globe uses does
not perform. A converter from its checkpoint to a PLY is a page of torch code, but the
PLY it produced would be 2DGS surfels drawn as flattened 3D gaussians: an approximation
of what was trained, with no way to measure the difference on a machine without a GPU.
The trainer is also a separate CLI (`tyro.cli(Config)`, no `default`/`mcmc` subcommand)
that `gsplat_argv` would have to be forked for. `variant: 2dgs` is refused by name for
those reasons (`VARIANT_REFUSALS`), rather than dispatched to a trainer whose output
nothing downstream could honestly read.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from support_mask import SupportMask

__all__ = [
    "GSPLAT_VERSION",
    "ROI_KEEP_RADII",
    "SCHEDULE_SCALE_RANGE",
    "TEST_EVERY",
    "VARIANT_REFUSALS",
    "Roi",
    "RoiCrop",
    "TrainMetrics",
    "build_dataset",
    "check_schedule_scale",
    "converge_argv",
    "crop_initial_points",
    "crop_rows",
    "crop_splat",
    "gsplat_argv",
    "held_out_split",
    "in_region",
    "latest_ply",
    "parse_metrics",
    "step_of",
    "trainer_python",
    "trainer_script",
]

_STEP_RE = re.compile(r"(\d+)")
#: The stdout fallback when no stats JSON exists. v1.5.3 prints
#: `print("Step: ", step, stats)` at each save step, where `stats` is a dict holding
#: `num_GS` -- read from the source, not from a run.
_STDOUT_GS_RE = re.compile(r"Step:\s+(\d+)\s+\{[^}]*'num_GS':\s*(\d+)")
#: `(?<!CC_)` because with `--use_bilateral_grid` the same line goes on with
#: `CC_PSNR: ..., CC_SSIM: ..., CC_LPIPS: ...` (the colour-corrected numbers), and without
#: it the last match of each pattern would be the corrected one.
_STDOUT_PSNR_RE = re.compile(r"(?<!CC_)PSNR:\s*([0-9.]+)")
#: The rest of `eval()`'s line: `PSNR: 26.110, SSIM: 0.8410, LPIPS: 0.172 Time: ...`.
_STDOUT_SSIM_RE = re.compile(r"(?<!CC_)PSNR:[^\n]*?(?<!CC_)SSIM:\s*([0-9.]+)")
_STDOUT_LPIPS_RE = re.compile(r"(?<!CC_)PSNR:[^\n]*?(?<!CC_)LPIPS:\s*([0-9.]+)")

#: The gsplat release every flag and file name here was read from.
GSPLAT_VERSION = "1.5.3"

#: v1.5.3's `Config.test_every`, which this stage never overrides. The parser sorts the
#: registered images by name and holds out every one whose index is a multiple of it
#: (`indices % test_every == 0`) as the `val` split; the trainer never trains on those,
#: and `eval()` measures PSNR, SSIM and LPIPS on them. So every quality number in
#: `train_metrics.json` is a held-out number.
TEST_EVERY = 8

#: What a caller may ask `schedule_scale` to be. The floor is where a run stops being a
#: shorter schedule and starts being a few hundred steps of initialisation; 1.0 is the
#: whole schedule, and a scale above it would be a longer run than the recipe prices.
#: The one way past 1.0 is the stage's, not a caller's: with `converge` on, a big
#: gaussian budget may lengthen the *maximum* (`gaussian_budget.schedule_factor`, at most
#: 2x), which the convergence rule then ends early when it has gone flat.
SCHEDULE_SCALE_RANGE: tuple[float, float] = (0.05, 1.0)

#: How far from an ROI's centre a trained gaussian may be, in radii, before it is dropped.
#: More than one: a gaussian whose centre is just outside the sphere still paints its
#: edge, and the sphere is a person's rough circle round the subject, not a surveyed hull.
ROI_KEEP_RADII = 1.5

#: Of the initial SfM points outside an ROI, one in this many is kept. Not zero: with no
#: points at all behind the subject the optimiser has nothing to explain the background
#: pixels with except the subject's own gaussians, which it then stretches out into
#: floaters to do it.
ROI_OUTSIDE_EVERY = 10

#: The fewest initial points any registered frame keeps through an ROI crop. gsplat's
#: depth loss looks each training frame up in `point_indices` (a KeyError for a frame
#: with none) and averages over the points it projects (NaN over none), so a frame that
#: looked mostly away from the subject keeps some of what it saw.
ROI_MIN_POINTS_PER_IMAGE = 32

#: The `variant` values this stage refuses, and why. See the module docstring.
VARIANT_REFUSALS: dict[str, str] = {
    "2dgs": (
        "variant=2dgs is not offered: gsplat 1.5.3's simple_trainer_2dgs.py writes no PLY "
        "(only torch checkpoints), has no MCMC strategy and so no cap_max, and trains "
        "surfels that the globe's 3DGS renderer would draw as flattened gaussians rather "
        "than as what was trained. See training.py's module docstring"
    ),
}


class TrainerMissingError(RuntimeError):
    """No trainer script. Says where one comes from instead of a bare FileNotFoundError."""


def trainer_script(configured: object) -> Path:
    """The trainer to run: the `trainer` param, else `$GSPLAT_TRAINER`.

    Deliberately not discovered by importing `gsplat`: this process runs on a CPU box and
    importing a CUDA extension to find out where a file is would make the stage
    un-runnable exactly where it is dispatched from.
    """
    raw = str(configured) if configured else os.environ.get("GSPLAT_TRAINER", "")
    if not raw:
        raise TrainerMissingError(
            "no trainer: set the stage's `trainer` param or $GSPLAT_TRAINER to gsplat's "
            "examples/simple_trainer.py. The training box's image is where that lives; "
            "this stage does not vendor it."
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise TrainerMissingError(f"the trainer {path} does not exist on this machine")
    return path.resolve()


def trainer_python(configured: object) -> str:
    """The interpreter to run the trainer with: the `python` param, else `$GSPLAT_PYTHON`,
    else this one.

    Separate from the interpreter running the stage on purpose. gsplat publishes its
    prebuilt CUDA wheels for CPython 3.10 only (docs.gsplat.studio/whl, read 2026-09-23:
    every `gsplat-1.5.x+pt2xcu1xx` wheel is `cp310`), while this project needs 3.12. The
    training image therefore carries a 3.10 venv for the trainer beside the 3.12 one the
    pipeline runs in, and says where through `$GSPLAT_PYTHON`.
    """
    if configured:
        return str(configured)
    return os.environ.get("GSPLAT_PYTHON") or sys.executable


def build_dataset(frames: Path, poses: Path, root: Path, *, max_side: int | None = None) -> Path:
    """COLMAP's on-disk layout, assembled in `work/` from the two input artifacts.

    `images/` and `sparse/0/` are what every 3DGS trainer's COLMAP parser looks for.
    Files are copied rather than linked: the trainer may be in another container with
    this directory mounted, and a symlink out of it resolves to nothing there.

    `max_side` shrinks each frame whose long side is bigger, under its own name, and
    leaves `sparse/0/` exactly as the pose stage wrote it: gsplat v1.5.3's parser rescales
    the intrinsics to the size of the images it finds (module docstring). The pixels are
    written as they are stored, with no EXIF orientation applied, because COLMAP posed the
    stored pixels and the trainer reads them the same way.
    """
    images = root / "images"
    sparse = root / "sparse" / "0"
    for directory in (images, sparse):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for frame in sorted(p for p in frames.iterdir() if p.is_file()):
        if max_side is None or not _shrink(frame, images / frame.name, max_side):
            shutil.copyfile(frame, images / frame.name)
    for entry in sorted(p for p in poses.iterdir() if p.is_file()):
        shutil.copyfile(entry, sparse / entry.name)
    return root


def _shrink(source: Path, target: Path, max_side: int) -> bool:
    """Write `source` to `target` with its long side at most `max_side`; False if it
    already fits (nothing written), so an unchanged frame is copied, not re-encoded."""
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
        if max(width, height) <= max_side:
            return False
        scale = max_side / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        kind = "JPEG" if target.suffix.lower() in {".jpg", ".jpeg"} else None
        resized.save(target, format=kind, quality=95)
    return True


def held_out_split(registered: int, test_every: int = TEST_EVERY) -> tuple[int, int]:
    """(train, val) frame counts for `registered` posed frames, as v1.5.3's parser splits.

    Indices `0, test_every, 2 * test_every, ...` are `val`, so there are
    `ceil(registered / test_every)` of them -- one of 4, 11 of 87.
    """
    if registered <= 0:
        return 0, 0
    val = -(-registered // test_every)
    return registered - val, val


@dataclass(frozen=True)
class Roi:
    """A sphere round what the capture is of, in the COLMAP frame `trained.ply` is in."""

    center: tuple[float, float, float]
    radius: float

    @staticmethod
    def parse(value: object) -> Roi | None:
        """`{"center": [x, y, z], "radius": r}`, or None for no ROI. Anything else is
        refused by name: a malformed ROI silently ignored is a full-scene run that was
        paid for as a cropped one."""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f'roi must be {{"center": [x, y, z], "radius": r}}, not {value!r}')
        center = value.get("center")
        radius = _finite(value.get("radius"))
        listed: list[object] = []
        if isinstance(center, Sequence) and not isinstance(center, str):
            listed = list(center)
        values = [v for v in (_finite(item) for item in listed) if v is not None]
        if len(listed) != 3 or len(values) != 3:
            raise ValueError(f"roi.center must be three finite numbers, not {center!r}")
        if radius is None or radius <= 0.0:
            raise ValueError(f"roi.radius must be a positive number, not {value.get('radius')!r}")
        x, y, z = values
        return Roi(center=(x, y, z), radius=radius)

    def to_dict(self) -> dict[str, object]:
        return {"center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class RoiCrop:
    """What cropping the initial points to an ROI kept, for `train_metrics.json`."""

    points_in: int
    inside: int
    outside_sampled: int
    kept_for_coverage: int
    points_kept: int

    def to_dict(self) -> dict[str, int]:
        return {
            "initialPoints": self.points_in,
            "insideRoi": self.inside,
            "outsideSampled": self.outside_sampled,
            "keptForFrameCoverage": self.kept_for_coverage,
            "initialPointsKept": self.points_kept,
        }


def in_region(region: Roi | SupportMask, xyz: Any) -> Any:
    """Which of `xyz` are in `region`, as `crop_initial_points` counts inside: within one
    radius of an ROI's centre, or in a support mask's voxels. For the gaussian budget,
    which counts a Refine's region in full."""
    return _inside(region, xyz)


def _inside(region: Roi | SupportMask, xyz: Any, *, radii: float = 1.0) -> Any:
    """Which points are in `region`: within `radii` radii of an ROI sphere, or in a
    support mask's voxels (which already carry their margin). Anything else with a
    `contains(xyz)` -- a block's region (`blocks.BlockRegion`) -- answers for itself."""
    import numpy as np

    from support_mask import SupportMask

    if isinstance(region, SupportMask):
        return region.contains(xyz)
    contains = getattr(region, "contains", None)
    if callable(contains):
        return contains(xyz)
    distance = np.linalg.norm(np.asarray(xyz, dtype=np.float64) - np.asarray(region.center), axis=1)
    return np.isfinite(distance) & (distance <= radii * region.radius)


def crop_initial_points(
    sparse: Path,
    roi: Roi | SupportMask,
    *,
    outside_every: int = ROI_OUTSIDE_EVERY,
    min_per_image: int = ROI_MIN_POINTS_PER_IMAGE,
) -> RoiCrop:
    """Rewrite `sparse/points3D.bin` to the points inside `roi`, plus a sparse sample.

    Kept: every point within `radius` of the centre; one in `outside_every` of the rest,
    by file order (deterministic, and spread over the scene the way COLMAP numbered it);
    and, for any registered frame left with fewer than `min_per_image` observed points,
    enough of its own observations to reach that (see `ROI_MIN_POINTS_PER_IMAGE`).
    Tracks travel with their points, so gsplat's `point_indices` stay consistent.
    """
    import numpy as np

    import sfm

    path = sparse / "points3D.bin"
    points = sfm.read_points3d(path)
    count = len(points)
    inside = _inside(roi, points.xyz)
    outside = np.flatnonzero(~inside)
    sampled = np.zeros(count, dtype=bool)
    sampled[outside[:: max(1, outside_every)]] = True
    keep = inside | sampled

    lengths = np.diff(points.track_offsets)
    row_point = np.repeat(np.arange(count), lengths)
    row_image = points.track[:, 0] if len(points.track) else np.zeros(0, dtype=np.uint32)
    added = np.zeros(count, dtype=bool)
    for image_id in np.unique(row_image):
        seen = row_point[row_image == image_id]
        have = int(np.count_nonzero(keep[seen] | added[seen]))
        if have >= min_per_image:
            continue
        missing = seen[~(keep[seen] | added[seen])]
        added[missing[: min_per_image - have]] = True
    keep |= added
    sfm.write_points3d(path, points.subset(keep))
    return RoiCrop(
        points_in=count,
        inside=int(inside.sum()),
        outside_sampled=int(sampled.sum()),
        kept_for_coverage=int(added.sum()),
        points_kept=int(keep.sum()),
    )


def crop_splat(
    columns: Mapping[str, Any], roi: Roi | SupportMask, *, radii: float = ROI_KEEP_RADII
) -> tuple[dict[str, Any], int]:
    """The gaussians whose centre is in the region -- within `radii * roi.radius` of an
    ROI's centre, or in a support mask's voxels -- and how many that is. A gaussian with a
    non-finite centre is dropped too: it is nowhere, so it is not inside anything."""
    keep = crop_rows(columns, roi, radii=radii)
    return {name: values[keep] for name, values in columns.items()}, int(keep.sum())


def crop_rows(
    columns: Mapping[str, Any], roi: Roi | SupportMask, *, radii: float = ROI_KEEP_RADII
) -> Any:
    """Which rows `crop_splat` keeps, as a boolean mask -- for anything else held per
    gaussian in the same order (the held-out error arrays) to be cropped alike."""
    import numpy as np

    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    return _inside(roi, xyz, radii=radii)


def _finite(value: object) -> float | None:
    """`value` as a float if it is a finite real number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def gsplat_argv(
    python: str,
    trainer: Path,
    dataset: Path,
    result_dir: Path,
    *,
    strategy: str = "default",
    max_steps: int = 30_000,
    data_factor: int = 1,
    steps_scaler: float = 1.0,
    cap_max: int | None = None,
    antialiased: bool = False,
    opacity_reg: float | None = None,
    depth_loss: bool = False,
    pose_opt: bool = False,
    app_opt: bool = False,
    bilateral_grid: bool = False,
    live_steps: Sequence[int] = (),
    eval_steps: Sequence[int] = (),
    extra: Sequence[str] = (),
) -> list[str]:
    """The command line, built in one place so a test can read it without a GPU.

    Every flag was parsed by v1.5.3's own CLI; see the module docstring. Four of them are
    not optional:

    * `--disable_viewer`: the default opens a viewer server and, after training, sleeps
      for 1,000,000 seconds (`main()`), which is a stage billed until the tier's timeout;
    * `--no-normalize-world-space`: keep the PLY in COLMAP's frame, which every transform
      downstream is expressed in;
    * `--save_ply --ply_steps N`: without them there is no PLY at all;
    * `--save_steps N --eval_steps N`: one checkpoint and one evaluation, at the end,
      rather than the defaults' extra pair at 7,000 -- a checkpoint nothing can resume
      from is only disk, and the evaluation at N is what `train_metrics.json` reads.

    `--steps_scaler` is v1.5.3's own way to run a shorter schedule (`Config.adjust_steps`):
    it multiplies `max_steps`, the save/ply/eval steps, the SH-degree interval and the
    strategy's refine window together, and the position learning rate decays over the
    scaled `max_steps`. A short run is then a *complete* short run -- decayed and
    densified on schedule -- where passing a smaller `--max_steps` alone would stop a
    30k schedule a quarter of the way in, learning rate still high. So `max_steps` and
    the three step lists stay the unscaled number here and the trainer scales them.

    `--strategy.cap-max` (spelled as v1.5.3's `examples/benchmarks/mcmc.sh` spells it) is
    `MCMCStrategy`'s ceiling on the number of gaussians; the
    default strategy has no such field and refuses the flag, so it is only passed with
    `mcmc`.

    `live_steps` are extra `--ply_steps`, after the final one: v1.5.3 exports a PLY at
    every listed step (scaled with the rest), which `live.SplatWatch` turns into the
    intermediate splats the live viewer shows. Order does not matter to the trainer (it
    tests membership), and the final step stays first so it reads as the one that counts.

    `eval_steps` are extra `--eval_steps`, after the final one and unscaled like it: the
    held-out evaluations the convergence rule reads (`convergence.eval_steps`). Each one
    writes `stats/val_step<i>.json`; the furthest along is still the final one, so
    `parse_metrics` reads the same file it always did.

    `--disable_video` too: the trajectory render after evaluation is a video nobody reads.
    There is no `--ckpt`, and there must not be: in v1.5.3 it means "evaluate this and do
    not train".

    Three quality switches, each passed only when asked for, so the default argv is the
    one every earlier run used:

    * `--antialiased` rasterises in v1.5.3's `antialiased` mode (Mip-Splatting's 2D
      filter), which scales each gaussian's opacity by how much the screen-space blur
      grew it. The PLY stores the opacities that compensation was trained against, so a
      renderer that does not apply the same compensation draws small gaussians more
      opaque than they were trained to be -- which is why it is not a default here.
    * `--opacity_reg X` replaces the preset's opacity regularisation weight: the `mcmc`
      preset sets 0.01 (with `scale_reg` 0.01, the MCMC paper's values), `default` 0.
    * `--depth_loss` adds v1.5.3's sparse depth term: every training frame's own SfM
      points (`load_depths`, from `points3D.bin` tracks) are projected into it and the
      rendered expected depth is pulled toward theirs, in disparity, weighted
      `depth_lambda` 0.01. No extra dataset is needed; the model the pose stage wrote
      already carries the tracks.

    And three that phone captures are expected to need -- each off until
    `experiments/run_variants.py` has measured it on a real capture -- read from v1.5.3's
    `Config` and `Runner`:

    * `--pose_opt` learns a per-training-image pose correction (`CameraOptModule`: a
      translation and a 6D rotation delta, zero-initialised, lr 1e-5 decayed to 1% over
      the run, weight decay 1e-6). The splat stays in the frame the poses are in; only
      the training cameras move. Held-out frames are rendered from their COLMAP poses
      uncorrected, so `val` PSNR shows the benefit only through a better splat, never
      through a better-fitted val camera.
    * `--app_opt` learns a per-image appearance embedding (`AppearanceOptModule`, 16-dim,
      plus a 32-dim feature per gaussian) that absorbs exposure / white-balance drift.
      The PLY is *baked*: `sh0` is the colour under a zero embedding viewed head-on, and
      `shN` is empty, so the export is degree 0 -- which is all `canonical.ply` carries
      anyway (`gaussians.CANONICAL_PROPERTIES`). Held-out frames render with the zero
      embedding.
    * `--use_bilateral_grid` fits a 16x16x8 bilateral grid per training image (BilaRF's
      colour model, `examples/lib_bilagrid.py`: plain torch plus `tensorly`, both in the
      training image) to the render before the loss. Nothing of it reaches the PLY.
      `eval()` then reports the raw metrics *and* colour-corrected ones (`cc_psnr`, ...),
      which `parse_metrics` keeps as `colorCorrected`.
    """
    steps = str(max_steps)
    argv = [
        python,
        str(trainer),
        strategy,
        "--data_dir",
        str(dataset),
        "--data_factor",
        str(data_factor),
        "--result_dir",
        str(result_dir),
        "--max_steps",
        steps,
        "--disable_viewer",
        "--disable_video",
        "--no-normalize-world-space",
        "--save_ply",
        "--ply_steps",
        steps,
        *(str(step) for step in live_steps if 0 < step < max_steps),
        "--save_steps",
        steps,
        "--eval_steps",
        steps,
        *(str(step) for step in eval_steps if 0 < step < max_steps),
    ]
    if steps_scaler != 1.0:
        argv += ["--steps_scaler", f"{steps_scaler:g}"]
    if cap_max is not None:
        if strategy != "mcmc":
            raise ValueError(
                f"cap_max bounds MCMCStrategy's gaussian count; strategy {strategy!r} has "
                f"no cap and gsplat {GSPLAT_VERSION} would refuse the flag"
            )
        argv += ["--strategy.cap-max", str(cap_max)]
    if antialiased:
        argv.append("--antialiased")
    if opacity_reg is not None:
        argv += ["--opacity_reg", f"{opacity_reg:g}"]
    if depth_loss:
        argv.append("--depth_loss")
    if pose_opt:
        argv.append("--pose_opt")
    if app_opt:
        argv.append("--app_opt")
    if bilateral_grid:
        argv.append("--use_bilateral_grid")
    argv += list(extra)
    return argv


def converge_argv(
    argv: Sequence[str],
    *,
    script: Path,
    every: int,
    window: int,
    min_gain_db: float,
) -> list[str]:
    """`gsplat_argv`'s command, run through `converge_trainer.py` instead of directly.

    The same interpreter, the wrapper where the trainer was, the rule, `--`, and then the
    trainer's own arguments unchanged -- so everything `gsplat_argv` says about them still
    holds, and the wrapper hands them to the trainer as its `sys.argv`.
    """
    python, trainer, *rest = argv
    return [
        python,
        str(script),
        "--trainer",
        trainer,
        "--every",
        str(every),
        "--window",
        str(window),
        "--min-gain-db",
        f"{min_gain_db:g}",
        "--",
        *rest,
    ]


def schedule_scale(images: int, *, full_at: int, floor: float) -> float:
    """How much of the full schedule a capture of `images` frames gets.

    Linear in the frame count up to `full_at`, never below `floor`. Every iteration
    renders one frame, so a 4-frame capture sees each frame 7,500 times over a 30k
    schedule: past the point of learning anything new about the scene, and paid for by
    the GPU-second. Rounded to hundredths so the argv, and the log, stay readable.
    """
    if full_at <= 0:
        return 1.0
    return round(min(1.0, max(floor, images / full_at)), 2)


def check_schedule_scale(value: float) -> float:
    """A requested `schedule_scale`, or a refusal naming the range it must be in."""
    low, high = SCHEDULE_SCALE_RANGE
    if not low <= value <= high:
        raise ValueError(
            f"schedule_scale={value:g} is outside {low:g}-{high:g}: it is the fraction of "
            f"the full schedule to run, and below {low:g} a run is initialisation only"
        )
    return value


def scaled_steps(max_steps: int, steps_scaler: float) -> int:
    """The step count the trainer will actually run: `int(max_steps * factor)`, as
    v1.5.3's `adjust_steps` computes it."""
    return int(max_steps * steps_scaler)


def latest_ply(directory: Path) -> Path | None:
    """The furthest-along PLY under `directory`, by the step number in its name.

    By step rather than by mtime or by name: `point_cloud_999.ply` sorts after
    `point_cloud_29999.ply` as text.
    """
    if not directory.is_dir():
        return None
    found = sorted(directory.rglob("*.ply"), key=lambda p: (step_of(p), p.name))
    return found[-1] if found else None


def step_of(path: Path) -> int:
    """The step number a trainer put in a file name, or -1. Orders checkpoints and plys.

    The *largest* number in the stem, not the last one: gsplat's names carry a rank
    suffix (`ckpt_29000_rank0.pt`, `val_step29999_rank0.json`), and taking the last
    number would order every checkpoint by its rank and call step 500 the newest.
    """
    numbers = _STEP_RE.findall(path.stem)
    return max(int(number) for number in numbers) if numbers else -1


@dataclass(frozen=True)
class TrainMetrics:
    """`train_metrics.json`: what B3's three-way comparison reads.

    Every field is nullable, and that is the contract: a trainer whose output this
    version does not recognise produces a row of nulls next to the iteration count and
    the gaussian count read off the PLY, which says "it trained and nothing was measured"
    rather than putting a made-up PSNR into a comparison table.
    """

    trainer: str
    iterations: int | None = None
    requested_iterations: int | None = None
    gaussians: int | None = None
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    train_seconds: float | None = None
    eval_seconds_per_image: float | None = None
    peak_memory_gb: float | None = None
    resumed_from_step: int | None = None
    attempts: int = 1
    source: str = "none"
    #: The held-out split psnr/ssim/lpips were measured on: (train, val) frame counts.
    split: tuple[int, int] | None = None
    #: `eval()`'s colour-corrected numbers, written only with `--use_bilateral_grid`.
    cc_psnr: float | None = None
    cc_ssim: float | None = None
    cc_lpips: float | None = None

    def to_dict(self) -> dict[str, object]:
        train, val = self.split if self.split is not None else (None, None)
        return {
            "trainer": self.trainer,
            "iterations": self.iterations,
            "requestedIterations": self.requested_iterations,
            "gaussians": self.gaussians,
            # Held-out numbers, every one: see `heldOut`.
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "heldOut": {
                "split": "val",
                "testEvery": TEST_EVERY,
                "valFrames": val,
                "trainFrames": train,
                "note": (
                    f"psnr, ssim and lpips are gsplat's eval() on the val split: every "
                    f"{TEST_EVERY}th registered frame by name (index % {TEST_EVERY} == 0), "
                    f"which the trainer never trains on"
                ),
            },
            # Wall seconds from the first step to the last save, off the trainer's own
            # `train_step*` stats; not the stage's duration, which includes loading.
            "trainSeconds": self.train_seconds,
            "evalSecondsPerImage": self.eval_seconds_per_image,
            "peakMemoryGb": self.peak_memory_gb,
            "resumedFromStep": self.resumed_from_step,
            "attempts": self.attempts,
            # Which of the two readers produced the numbers above, so a reader of the
            # file can tell a parsed stats file from a scraped log line.
            "source": self.source,
            # With a bilateral grid, the val renders scored again after a per-image
            # affine colour fit to the ground truth (v1.5.3's `color_correct`): what the
            # splat scores once exposure is forgiven. Null without one.
            "colorCorrected": (
                None
                if self.cc_psnr is None and self.cc_ssim is None and self.cc_lpips is None
                else {"psnr": self.cc_psnr, "ssim": self.cc_ssim, "lpips": self.cc_lpips}
            ),
        }


def parse_metrics(
    result_dir: Path,
    log_text: str = "",
    *,
    trainer: str = "gsplat",
    requested_iterations: int | None = None,
    resumed_from_step: int | None = None,
    attempts: int = 1,
    registered: int | None = None,
) -> TrainMetrics:
    """Read back what the trainer measured: its stats files first, its stdout second.

    Two files, because v1.5.3 writes two and they mean different things by the same key:
    `val_step<i>.json` is `eval()`'s held-out psnr/ssim/lpips, where `ellipse_time` is the
    mean seconds to *render one val image*; `train_step<i>_rank0.json` is the save step's,
    where `ellipse_time` is seconds since training started, beside peak `mem` in GB.
    `registered` is how many posed frames the trainer split, for the held-out counts.
    """
    split = held_out_split(registered) if registered is not None else None
    val = _read_stats(result_dir, "val_step*.json")
    train = _read_stats(result_dir, "train_step*.json")
    if val is not None or train is not None:
        step = max(found[0] for found in (val, train) if found is not None)
        quality: Mapping[str, object] = val[1] if val is not None else {}
        timing: Mapping[str, object] = train[1] if train is not None else {}
        return TrainMetrics(
            trainer=trainer,
            # Zero-based in every file name v1.5.3 writes: `val_step29999` is 30,000 steps.
            iterations=step + 1,
            requested_iterations=requested_iterations,
            gaussians=_int(quality.get("num_GS", timing.get("num_GS"))),
            psnr=_float(quality.get("psnr")),
            ssim=_float(quality.get("ssim")),
            lpips=_float(quality.get("lpips")),
            train_seconds=_float(timing.get("ellipse_time")),
            eval_seconds_per_image=_float(quality.get("ellipse_time")),
            peak_memory_gb=_float(timing.get("mem")),
            resumed_from_step=resumed_from_step,
            attempts=attempts,
            source="stats",
            split=split,
            cc_psnr=_float(quality.get("cc_psnr")),
            cc_ssim=_float(quality.get("cc_ssim")),
            cc_lpips=_float(quality.get("cc_lpips")),
        )
    scraped = _scrape(log_text)
    return TrainMetrics(
        trainer=trainer,
        iterations=_completed(scraped.get("step")),
        requested_iterations=requested_iterations,
        gaussians=_int(scraped.get("gaussians")),
        psnr=scraped.get("psnr"),
        ssim=scraped.get("ssim"),
        lpips=scraped.get("lpips"),
        resumed_from_step=resumed_from_step,
        attempts=attempts,
        source="stdout" if scraped else "none",
        split=split,
    )


def _read_stats(result_dir: Path, pattern: str) -> tuple[int, Mapping[str, object]] | None:
    """The furthest-along `stats/<pattern>` file that parses as a JSON object."""
    stats = result_dir / "stats"
    if not stats.is_dir():
        return None
    candidates = sorted(stats.glob(pattern))
    best: tuple[int, Mapping[str, object]] | None = None
    for path in candidates:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(document, dict):
            continue
        step = step_of(path)
        if best is None or step > best[0]:
            best = (step, document)
    return best


def _scrape(text: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for match in _STDOUT_GS_RE.finditer(text):
        out["step"] = int(match.group(1))
        out["gaussians"] = int(match.group(2).replace(",", ""))
    for key, pattern in (
        ("psnr", _STDOUT_PSNR_RE),
        ("ssim", _STDOUT_SSIM_RE),
        ("lpips", _STDOUT_LPIPS_RE),
    ):
        found = pattern.findall(text)
        if found:
            out[key] = float(found[-1])
    return out


def _completed(step: object) -> int | None:
    """A zero-based step index as the number of steps completed."""
    index = _int(step)
    return None if index is None else index + 1


def _int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 6)
