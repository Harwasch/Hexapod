"""Block training: a scene whose gaussian budget is more than one GPU holds, trained as
blocks, one after another, each on the same single L4.

**Not executed on a GPU in this repository yet.** What is tested here is everything
around the trainer: the partition, the camera rule, the per-block datasets and budgets,
the frozen ring's selection, the blend, the merge and the resume -- driven end to end by
the stand-in trainers in `tests/`. The GPU pieces are `block_views.py` (renders the
prior to decide which cameras a block needs) and `block_trainer.py` (the frozen ring and
the convergence stop, as a wrapper round gsplat's own trainer, and the merged model's
evaluation).

**When.** Only when the density budget (`gaussian_budget.plan`, `auto`) is more than one
GPU trains: its raw count over `gpu_cap`, the smaller of the budget's two ceilings (the
L4's memory model and `budget_max`), rounded up. `blocks: <n>` forces n blocks, for
validating the machinery on a capture that fits one GPU (the spool at `blocks: 2`).

**The recipe, and where each part comes from** (`block_maths.py` has the rules' code):

1. *Prior*: a whole-scene coarse model -- CityGaussian §3.2 trains one on every image
   first and uses it to partition, to assign cameras and to initialise every block;
   H3DGS §6.1 trains one as the "scaffold". Ours is the Preview, which every capture
   already has: every `train` run leaves its splat's shapes in `checkpoint/block-prior/`
   (`save_prior`). A run with none trains a short coarse pass first (`_coarse_prior`):
   7,500 steps (a quarter of the schedule; CityGS V2's fast "-t" variant trains its coarse
   model for 7k where the full one trains 30k) at the Preview's 800 px, capped at 1M
   gaussians (5x the Preview's 200k, still a fraction of what the L4 holds at 800 px).
2. *Partition*: per-axis quantiles of the prior's gaussians in the region to train
   (CityGS V2 `build_partition_coordinates`), in the ground plane of the cameras' up.
3. *Cameras*: in the block, or `1 - SSIM > epsilon` when the block is removed from the
   prior's render (CityGS; Tab. 4: 25.77 dB with the test, 23.43 without). At least 50
   training frames a block (V2 asserts > 50), else merged into a neighbour.
4. *Training*: each block from the prior's gaussians in it (plus its margin), with a
   *frozen* ring of prior gaussians round it (H3DGS's scaffold) so its edges are occluded
   and lit as in the scene, without loading the whole scene as CityGS does; its own
   density budget and the single run's schedule and convergence stop.
5. *Merge*: each block keeps the gaussians whose final centre is in its core, their
   opacity faded across a thin band at the shared edges (H3DGS `getWeight`); the blocks
   are concatenated into one PLY by a chunked writer, so `quality`, `place` and `package`
   read one `trained.ply` as always.

**On Modal: one GPU per block, at once.** Measured on the spool forced to 2 blocks, one
after another on one L4: a 7,500-step coarse pass (no Preview existed), then block 1 in
2,530 s and block 2 in 2,207 s -- 2.3 h and $1.73, against ~1 h and $0.67 for the scene
whole, at the same quality (26.42 dB / LPIPS 0.1056 merged, 26.39 / 0.1088 whole). Modal
bills per GPU-second, so the same blocks on N GPUs at once cost the same seconds and end
when the longest does. `cloud.CloudRunner` fans the stage out (`contracts.FanOut`):

1. *head* -- the stage's usual call. The prior (or the coarse pass) and the camera test's
   renders happen here, once, and the plan goes into `checkpoint/blocks/plan.json`. With
   more than one block left to train and `block_parallel` above 1, it answers with one
   part per block instead of training them; otherwise (one left, or `block_parallel: 1`)
   it trains and merges here, as before.
2. *part* -- one block per call, `block_parallel` (4) at a time, each on its own
   checkpoint key with the prior and the plan. Its cropped splat and its record
   (`blocks/block_NNN/`, `blocks/block_NNN.json`) are what come home. A part that fails or
   is preempted runs again alone while the rest carry on.
3. *join* -- every block home: the merge, the evaluation and the held-out error, which
   write the stage's outputs.

A finished block is never trained again: the head of a later attempt lists only the blocks
with no record. Every call is priced in the attempt ledger, so the stage's cost is the sum.
Expected on the spool: the 2.3 h less the shorter block's 2,207 s, about 1.7 h, for the
same ~$1.7 plus two more container starts and dataset copies (a few cents); with a
Preview's prior there is no coarse pass either.

Without a fanning runner (`LocalRunner`, `block_parallel: 1`) one call loops the blocks,
bounded by the Modal function's 6 h timeout: before starting another block the
stage checks it has time for one more as long as the longest so far, and otherwise ends
the attempt (`BlocksYieldError`) after the checkpoint has synced; the worker's retry
resumes it.

**Each block's schedule follows its own data** (`block_schedule`, the param of the same
name): the single run's rule, `training.schedule_scale` -- linear in the frame count up to
`schedule_full_at`, never below `schedule_floor` -- applied to the frames the block trains
on, so a block that sees few cameras trains proportionally shorter; the gaussian factor
and the convergence stop then apply per block as they do to a whole run. A block that sees
every frame (the spool's: its 156 training cameras in both blocks) keeps the run's schedule
exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

import block_maths
import convergence
import gaussian_budget
import gaussians
import holdout
import init_seed
import live
import sfm
import training
from captures_bridge import read_ply
from contracts import FANOUT_PARAM, FanOut, FanOutPart, MetricValue, StageContext
from support_mask import SupportMask

__all__ = [
    "BLOCK_PARALLEL",
    "BLOCK_SCHEDULES",
    "BLOCK_TRAINER",
    "BLOCK_VIEWS",
    "PRIOR_DIR",
    "STATE_DIR",
    "BlockRegion",
    "BlocksYieldError",
    "Prior",
    "Settings",
    "block_count",
    "block_schedule",
    "fan_out",
    "finished",
    "gpu_cap",
    "load_prior",
    "merge_plys",
    "parse_blocks",
    "part_index",
    "pending",
    "prepare",
    "save_prior",
    "train",
    "train_part",
]

F32 = npt.NDArray[np.float32]
Region = training.Roi | SupportMask | None

#: Under the stage's `checkpoint/`: the prior (kept across runs, like `init-seed/`) and
#: one block run's resumable state (a fresh run clears it).
PRIOR_DIR = "block-prior"
STATE_DIR = "blocks"
_PRIOR_ARRAYS = "prior.npz"
_PRIOR_META = "prior.json"
_PRIOR_VERSION = 1
_PLAN = "plan.json"
_STATE = "state.json"
_STATE_VERSION = 1

#: The prior keeps gaussians at least this opaque: the quality stage's own `min_opacity`
#: (recipes/photo-reconstruct.yaml), below which nothing downstream draws them either.
PRIOR_MIN_OPACITY = 0.02
#: The largest splat left as a prior: 34 bytes a gaussian (float32 centres, float16
#: shapes and colour), so 68 MB, which the checkpoint carries between runs.
PRIOR_MAX_GAUSSIANS = 2_000_000

#: The coarse pass a run with no prior trains first; see the module docstring.
COARSE_SCHEDULE_SCALE = 0.25
COARSE_CAP = 1_000_000
COARSE_MAX_SIDE = 800

#: The camera test renders the prior at most this size (the Preview's own frame size):
#: CityGS renders the coarse model at its training resolution; a 1 - SSIM of a whole
#: block appearing or not is not a fine-detail measurement.
ASSIGN_MAX_SIDE = 800

#: Modal's function timeout is 6 h (`infra/modal/app.py`, TIMEOUT_S); a block is not
#: started unless the attempt has this long minus the longest block so far left, which
#: leaves an hour for the merge, the evaluation and the transfers.
ATTEMPT_BUDGET_S = 5 * 3600.0
#: How long an attempt that yields waits for the checkpoint syncer before it ends: more
#: than `CloudRunner`'s 60 s interval, so the last finished block is in object storage.
SYNC_WAIT_S = 75.0
#: Rows a checkpoint file of a finished block holds (`write_parts`): ~31 MB at SH degree 3.
PART_ROWS = 131_072
#: Params that pace an attempt without changing what a block trains -- including how many
#: blocks train at once and which call of a fan-out this is, so that neither can make a
#: finished block look like another plan's.
RUNTIME_PARAMS = frozenset(
    {"block_attempt_budget_s", "block_sync_wait_s", "live", "block_parallel", FANOUT_PARAM}
)

#: Blocks trained at once when the runner fans out (`block_parallel`). Four L4s: the spool's
#: two blocks and a large scene's first four at once, well under a Modal workspace's GPU
#: concurrency limit even with a couple of runs training together; a scene of more blocks
#: queues the rest, longest first. `cloud.CloudRunner(max_parallel=)` is the deployment's
#: own ceiling over this.
BLOCK_PARALLEL = 4

#: How a block's schedule follows its data (`block_schedule`):
#:   frames  the single run's rule (`training.schedule_scale`) on the block's own frames;
#:   share   the run's schedule times the block's share of the frames -- each frame visited
#:           as often as in the whole run, so the blocks add up to about one run's steps.
#:           Opt-in: nothing has measured a block trained that short;
#:   full    every block the run's whole schedule (before this rule existed).
BLOCK_SCHEDULES = ("frames", "share", "full")

#: The two scripts run with the trainer's interpreter, beside this file.
BLOCK_TRAINER = Path(__file__).resolve().parent / "block_trainer.py"
BLOCK_VIEWS = Path(__file__).resolve().parent / "block_views.py"


class BlocksYieldError(RuntimeError):
    """This attempt has no time left for another block; the next one resumes."""


# --- the prior -------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """A whole-scene splat's shapes: what blocks are partitioned, seeded and ringed by."""

    xyz: F32
    log_scales: F32
    quats: F32
    logit_opacity: F32
    f_dc: F32
    meta: Mapping[str, Any]

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    def as_seed(self) -> init_seed.Seed:
        """The prior as `init_seed.apply` takes a seed: centres, colours, opacities."""
        with np.errstate(over="ignore"):
            alpha = (1.0 / (1.0 + np.exp(-self.logit_opacity.astype(np.float64)))).astype(
                np.float32
            )
        rgb = np.clip(np.round((gaussians.SH_C0 * self.f_dc + 0.5) * 255.0), 0, 255)
        return init_seed.Seed(
            xyz=self.xyz, rgb=rgb.astype(np.uint8), alpha=alpha, meta={"trainedWith": self.meta}
        )


def save_prior(
    checkpoint_dir: Path,
    columns: Mapping[str, Any],
    poses: Path,
    *,
    scratch: Path,
    source: str,
    max_gaussians: int = PRIOR_MAX_GAUSSIANS,
) -> int | None:
    """Leave `columns` (a trained splat, COLMAP frame) as the prior for a later block run.

    None when there are more than `max_gaussians` of them (nothing is written, and an
    earlier prior stays). Written in `scratch` and renamed in, as `init_seed.save` is.
    """
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float32)
    logit = np.asarray(columns["opacity"], dtype=np.float32)
    with np.errstate(over="ignore"):
        alpha = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(logit) & (alpha >= PRIOR_MIN_OPACITY)
    count = int(keep.sum())
    if count > max_gaussians or count == 0:
        return None
    arrays = {
        "xyz": xyz[keep],
        "log_scales": np.stack([columns[f"scale_{i}"] for i in range(3)], 1)[keep],
        "quats": np.stack([columns[f"rot_{i}"] for i in range(4)], 1)[keep],
        "logit_opacity": logit[keep],
        "f_dc": np.stack([columns[f"f_dc_{i}"] for i in range(3)], 1)[keep],
    }
    stored: dict[str, Any] = {
        name: (values.astype(np.float32) if name == "xyz" else values.astype(np.float16))
        for name, values in arrays.items()
    }
    directory = checkpoint_dir / PRIOR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    partial = scratch / f"{_PRIOR_ARRAYS}.partial"
    with partial.open("wb") as handle:
        np.savez(handle, **stored)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    os.replace(partial, directory / _PRIOR_ARRAYS)
    meta = {
        "version": _PRIOR_VERSION,
        "posesFingerprint": init_seed.poses_fingerprint(poses),
        "gaussians": count,
        "source": source,
        "digest": digest,
    }
    partial_meta = scratch / f"{_PRIOR_META}.partial"
    partial_meta.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial_meta, directory / _PRIOR_META)
    return count


def load_prior(checkpoint_dir: Path, poses: Path) -> tuple[Prior | None, str]:
    """The prior an earlier run left, or None and why not."""
    directory = checkpoint_dir / PRIOR_DIR
    arrays, meta_path = directory / _PRIOR_ARRAYS, directory / _PRIOR_META
    if not arrays.is_file() or not meta_path.is_file():
        return None, "no earlier run of this stage left a prior"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(arrays) as data:
            loaded = {name: np.asarray(data[name], dtype=np.float32) for name in data.files}
    except (OSError, ValueError, KeyError) as error:
        return None, f"the prior could not be read ({error})"
    if not isinstance(meta, dict) or meta.get("version") != _PRIOR_VERSION:
        return None, "the prior is from another version of this stage"
    if meta.get("posesFingerprint") != init_seed.poses_fingerprint(poses):
        return None, "the prior was trained against other poses, so it is in another frame"
    try:
        prior = Prior(
            xyz=loaded["xyz"].reshape(-1, 3),
            log_scales=loaded["log_scales"].reshape(-1, 3),
            quats=loaded["quats"].reshape(-1, 4),
            logit_opacity=loaded["logit_opacity"].reshape(-1),
            f_dc=loaded["f_dc"].reshape(-1, 3),
            meta=meta,
        )
    except KeyError as error:
        return None, f"the prior is missing {error}"
    if prior.count == 0:
        return None, "the prior holds no gaussians"
    return prior, ""


def prior_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / PRIOR_DIR / _PRIOR_ARRAYS


# --- when, and how many ----------------------------------------------------------------


def parse_blocks(value: object) -> int | None:
    """`blocks`: `auto` (or unset) is None, a positive integer is that many."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"blocks must be 'auto' or a positive integer, not {value!r}")
    try:
        number = int(str(value))
    except ValueError:
        raise ValueError(f"blocks must be 'auto' or a positive integer, not {value!r}") from None
    if number < 1:
        raise ValueError(f"blocks must be 'auto' or a positive integer, not {value!r}")
    return number


def gpu_cap(budget: gaussian_budget.Budget | None) -> int | None:
    """The most gaussians one GPU run is given: the smaller of the budget's ceilings (the
    GPU's memory model at the frame size, and `budget_max`), read off the budget so a
    change to either is followed here."""
    if budget is None:
        return None
    bounds = [b for b in (budget.memory_ceiling, budget.budget_max) if b is not None and b > 0]
    return min(bounds) if bounds else None


def block_count(requested: int | None, budget: gaussian_budget.Budget | None) -> tuple[int, str]:
    """How many blocks, and why. `auto` is one unless an `auto` budget's raw count is more
    than one GPU's cap: then that ratio, rounded up."""
    if requested is not None:
        return requested, f"blocks: {requested} given"
    cap = gpu_cap(budget)
    if budget is None or budget.mode != "auto" or budget.raw is None or cap is None:
        return 1, "no auto budget to compare with a GPU"
    if budget.raw <= cap:
        return 1, f"the budget ({budget.raw}) fits one GPU ({cap})"
    count = math.ceil(budget.raw / cap)
    return count, f"the budget ({budget.raw}) is {budget.raw / cap:.2f}x one GPU's {cap}"


# --- a block's region ------------------------------------------------------------------


@dataclass(frozen=True)
class BlockRegion:
    """Where a block trains: its cells grown by the margin, within the run's own region.

    Duck-types as a region for `training.crop_initial_points`, `init_seed.apply` and the
    budget, all of which ask only `contains(xyz)`.
    """

    partition: block_maths.Partition
    block: int
    margin: float
    within: Region = None

    def contains(self, xyz: Any) -> npt.NDArray[np.bool_]:
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        inside = np.asarray(
            self.partition.in_expanded(self.partition.project(points), self.block, self.margin),
            dtype=bool,
        )
        if self.within is not None:
            inside &= np.asarray(training.in_region(self.within, points), dtype=bool)
        return inside


# --- a block's dataset -----------------------------------------------------------------


def _read_image_records(path: Path) -> list[tuple[int, str, bytes, bytes]]:
    """`images.bin` as (id, name, the 64-byte pose record, the points2D block)."""
    out: list[tuple[int, str, bytes, bytes]] = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            head = handle.read(64)
            (image_id,) = struct.unpack_from("<I", head, 0)
            name = bytearray()
            while (char := handle.read(1)) not in (b"\0", b""):
                name += char
            raw_count = handle.read(8)
            (points,) = struct.unpack("<Q", raw_count)
            out.append((image_id, name.decode("utf-8"), head, raw_count + handle.read(24 * points)))
    return out


def write_images_subset(source: Path, target: Path, rename: Mapping[str, str]) -> set[int]:
    """`images.bin` with only the images in `rename`, under their new names, in order of
    their new names. Returns the kept image ids."""
    records = [r for r in _read_image_records(source) if r[1] in rename]
    records.sort(key=lambda r: rename[r[1]])
    with target.open("wb") as handle:
        handle.write(struct.pack("<Q", len(records)))
        for _image_id, name, head, rest in records:
            handle.write(head + rename[name].encode("utf-8") + b"\0" + rest)
    return {r[0] for r in records}


def keep_tracks(points: sfm.Points3D, image_ids: set[int]) -> sfm.Points3D:
    """`points` with every track row of an image not in `image_ids` removed. gsplat's
    parser looks each row's image up by id, so a row of a removed image is a KeyError; a
    point left with no rows is legal (`init_seed` says why)."""
    if len(points.track) == 0:
        return points
    keep_rows = np.isin(points.track[:, 0], np.fromiter(image_ids, dtype=np.uint32))
    lengths = np.diff(points.track_offsets)
    owner = np.repeat(np.arange(len(points)), lengths)
    kept_lengths = np.bincount(owner[keep_rows], minlength=len(points))
    return sfm.Points3D(
        ids=points.ids,
        xyz=points.xyz,
        rgb=points.rgb,
        error=points.error,
        track=points.track[keep_rows],
        track_offsets=np.concatenate([[0], np.cumsum(kept_lengths)]).astype(np.int64),
    )


def build_block_dataset(
    dataset: Path, target: Path, order: Sequence[tuple[str, bool]]
) -> dict[str, str]:
    """A COLMAP dataset of `order`'s frames, renamed `<position>_<name>` so gsplat's
    parser's name sort puts them in that order (`block_maths.val_order`). Returns the
    renaming."""
    images, sparse = target / "images", target / "sparse" / "0"
    for directory in (images, sparse):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    rename = {name: f"{position:05d}_{name}" for position, (name, _val) in enumerate(order)}
    for old, new in rename.items():
        shutil.copyfile(dataset / "images" / old, images / new)
    source = dataset / "sparse" / "0"
    for entry in sorted(p for p in source.iterdir() if p.is_file()):
        if entry.name not in ("images.bin", "points3D.bin"):
            shutil.copyfile(entry, sparse / entry.name)
    kept = write_images_subset(source / "images.bin", sparse / "images.bin", rename)
    sfm.write_points3d(
        sparse / "points3D.bin", keep_tracks(sfm.read_points3d(source / "points3D.bin"), kept)
    )
    return rename


# --- PLY, a block at a time ------------------------------------------------------------

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}


@dataclass(frozen=True)
class PlyLayout:
    """A binary little-endian, vertex-only PLY's header: what gsplat and this file write."""

    count: int
    names: tuple[str, ...]
    dtype: np.dtype[Any]
    offset: int


def ply_layout(path: Path) -> PlyLayout:
    with path.open("rb") as handle:
        count, names, types, offset = 0, [], [], 0
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError(f"{path.name} is not a PLY")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path.name}: no end_header")
            words = line.decode("ascii", errors="replace").split()
            if words[:1] == ["format"] and words[1] != "binary_little_endian":
                raise ValueError(f"{path.name}: {words[1]} is not binary_little_endian")
            if words[:2] == ["element", "vertex"]:
                count = int(words[2])
            elif words[:1] == ["element"]:
                raise ValueError(f"{path.name}: element {words[1]} besides vertex")
            elif words[:1] == ["property"]:
                if words[1] == "list" or words[1] not in _PLY_TYPES:
                    raise ValueError(f"{path.name}: property {' '.join(words[1:])}")
                types.append(_PLY_TYPES[words[1]])
                names.append(words[2])
            elif words[:1] == ["end_header"]:
                offset = handle.tell()
                break
    return PlyLayout(count, tuple(names), np.dtype(list(zip(names, types, strict=True))), offset)


def write_columns(path: Path, columns: Mapping[str, Any], names: Sequence[str]) -> int:
    """A binary little-endian PLY of `names`, each as float32. The header is
    `gaussians.write_ply`'s, so the canonical fourteen come out byte-identical to it."""
    count = int(np.asarray(columns[names[0]]).shape[0])
    record = np.empty(count, dtype=np.dtype([(name, "<f4") for name in names]))
    for name in names:
        record[name] = columns[name]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    payload = ("\n".join(header) + "\n").encode("ascii") + record.tobytes()
    path.write_bytes(payload)
    return len(payload)


def merge_plys(
    parts: Sequence[Path], target: Path, names: Sequence[str] | None = None, *, chunk: int = 1 << 20
) -> tuple[int, int]:
    """Concatenate `parts` into one PLY of `names` (default: the first part's), reading
    and writing `chunk` rows at a time -- the merged splat is never in memory, only one
    chunk of one part. Returns (gaussians, bytes).

    A local, minimal chunked append: the Phase 2 splat I/O (chunked, memory-mapped, Morton
    ordered) should replace it when it lands.
    """
    layouts = [ply_layout(part) for part in parts]
    if not layouts:
        raise ValueError("nothing to merge")
    wanted = tuple(names) if names is not None else layouts[0].names
    for part, layout in zip(parts, layouts, strict=True):
        missing = [name for name in wanted if name not in layout.names]
        if missing:
            raise ValueError(f"{part.name} has no {', '.join(missing)}")
    total = sum(layout.count for layout in layouts)
    out_dtype = np.dtype([(name, "<f4") for name in wanted])
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {total}"]
    header += [f"property float {name}" for name in wanted]
    header.append("end_header")
    written = 0
    with target.open("wb") as handle:
        written += handle.write(("\n".join(header) + "\n").encode("ascii"))
        for part, layout in zip(parts, layouts, strict=True):
            if layout.count == 0:
                continue
            rows = np.memmap(
                part, dtype=layout.dtype, mode="r", offset=layout.offset, shape=(layout.count,)
            )
            for start in range(0, layout.count, chunk):
                piece = rows[start : start + chunk]
                record = np.empty(piece.shape[0], dtype=out_dtype)
                for name in wanted:
                    record[name] = piece[name]
                written += handle.write(record.tobytes())
            del rows
    return total, written


# --- the settings a block run shares with the single run -------------------------------


@dataclass(frozen=True)
class Settings:
    """Everything the stage worked out before it knew there would be blocks."""

    frames: Path
    poses: Path
    dataset: Path
    region: Region
    budget: gaussian_budget.Budget | None
    frame_size: tuple[int, int] | None
    pixel_scale: float | None
    strategy: str
    full_iterations: int
    #: The run's schedule, `init_from: preview` included, before any budget factor.
    steps_scaler: float
    data_factor: int
    trainer: Path
    python: str
    switches: Mapping[str, Any]
    extra: tuple[str, ...]
    converge: bool
    rule: convergence.Rule
    live: bool
    up: list[float] | None
    requested: int | None
    count: int
    reason: str
    init_max_points: int | None = None
    holdout_error: bool = False
    holdout_script: Path | None = None
    holdout_budget_s: float = holdout.DEFAULT_BUDGET_S
    params: Mapping[str, Any] = field(default_factory=dict)
    #: When this attempt's block work began (a coarse pass counts against its budget).
    started: float = field(default_factory=time.monotonic)
    #: What the run's own schedule was worked out from, for each block's (`block_schedule`):
    #: its frame count, the recipe's `schedule_full_at` and `schedule_floor`, and whether
    #: the run named its `schedule_scale` (which, as for a whole run, the frame rule leaves
    #: alone).
    images: int = 0
    schedule_full_at: int | None = None
    schedule_floor: float = 0.25
    schedule_requested: bool = False
    block_schedule: str = "frames"
    #: Images per training step (`training.gsplat_argv`'s `batch_size`).
    batch_size: int = 1
    #: Blocks at once when the runner fans out (`BLOCK_PARALLEL`); 1 trains them in turn.
    parallel: int = BLOCK_PARALLEL


@dataclass(frozen=True)
class Plan:
    """The partition and each block's cameras, as decided once per block run."""

    partition: block_maths.Partition
    names: tuple[str, ...]
    val: tuple[bool, ...]
    assigned: npt.NDArray[np.bool_]
    inside: npt.NDArray[np.bool_]
    method: str
    epsilon: float
    merges: tuple[tuple[int, int, int], ...]
    fingerprint: str
    prior_source: str
    prior_count: int
    gaussians: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "fingerprint": self.fingerprint,
            "partition": self.partition.to_dict(),
            "names": list(self.names),
            "val": list(self.val),
            "assigned": self.assigned.astype(int).tolist(),
            "inside": self.inside.astype(int).tolist(),
            "method": self.method,
            "epsilon": self.epsilon,
            "merges": [list(m) for m in self.merges],
            "priorSource": self.prior_source,
            "priorGaussians": self.prior_count,
            "gaussians": list(self.gaussians),
        }

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> Plan:
        return Plan(
            partition=block_maths.Partition.from_dict(document["partition"]),
            names=tuple(str(n) for n in document["names"]),
            val=tuple(bool(v) for v in document["val"]),
            assigned=np.asarray(document["assigned"], dtype=bool),
            inside=np.asarray(document["inside"], dtype=bool),
            method=str(document["method"]),
            epsilon=float(document["epsilon"]),
            merges=tuple((int(a), int(b), int(c)) for a, b, c in document["merges"]),
            fingerprint=str(document["fingerprint"]),
            prior_source=str(document["priorSource"]),
            prior_count=int(document["priorGaussians"]),
            gaussians=tuple(int(g) for g in document["gaussians"]),
        )


def _param_float(settings: Settings, name: str, default: float) -> float:
    value = settings.params.get(name)
    return default if value is None else float(value)


def _param_bool(settings: Settings, name: str, default: bool) -> bool:
    value = settings.params.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, not {value!r}")


def _fingerprint(settings: Settings, prior: Prior) -> str:
    """What makes a finished block reusable: these params, these poses, this prior. The
    params that only pace an attempt (`RUNTIME_PARAMS`) are left out, so an operator can
    change them between attempts without losing finished blocks."""
    digest = hashlib.sha256()
    params = {k: v for k, v in settings.params.items() if k not in RUNTIME_PARAMS}
    digest.update(json.dumps(params, sort_keys=True, default=str).encode())
    digest.update(init_seed.poses_fingerprint(settings.poses).encode())
    digest.update(str(prior.meta.get("digest")).encode())
    return digest.hexdigest()


def _state_dir(ctx: StageContext) -> Path:
    return ctx.checkpoint_dir / STATE_DIR


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _write_json_atomic(path: Path, document: Mapping[str, Any], scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    partial = scratch / f"{path.name}.partial"
    partial.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, path)


# --- deciding the blocks ---------------------------------------------------------------


def prepare(ctx: StageContext, settings: Settings, *, role: str | None = None) -> Plan | None:
    """The prior, the partition and each block's cameras -- or None when, after the
    minimum-cameras rule, one block is all there is (then the single run trains).

    `role` is the call's part in a fan-out (`contracts.fanout_role`). A `part` or a `join`
    never plans: it runs on the plan and prior the head call made in this same attempt,
    and refuses rather than re-plan (which would re-render every camera on each GPU and
    could disagree with the plan the other blocks trained on)."""
    work = ctx.work_dir / "blocks"
    work.mkdir(parents=True, exist_ok=True)
    state_dir = _state_dir(ctx)
    follower = role in ("part", "join")
    prior, why = load_prior(ctx.checkpoint_dir, settings.poses)
    if prior is None:
        if follower:
            raise ValueError(f"blocks: this {role} call was given no prior: {why}")
        ctx.log(f"blocks: {why}; training a coarse whole-scene pass first")
        prior = _coarse_prior(ctx, settings, work / "coarse")
    ctx.log(
        f"blocks: prior of {prior.count} gaussians ({prior.meta.get('source', 'unknown')}); "
        f"{settings.reason} -> {settings.count} blocks"
    )
    fingerprint = _fingerprint(settings, prior)
    # A fresh run (attempt 1, which a person's retry also resets to) starts over; a later
    # attempt resumes only what this same plan finished. Only the head decides that: the
    # parts and the join of attempt 1 carry on from what its head left.
    if ctx.attempt == 1 and not follower and state_dir.exists():
        shutil.rmtree(state_dir)
    existing = _read_json(state_dir / _PLAN)
    if existing is not None and existing.get("fingerprint") == fingerprint:
        plan = Plan.from_dict(existing)
        if not follower:
            ctx.log(f"blocks: attempt {ctx.attempt} resumes the plan an earlier attempt made")
        return plan if plan.partition.count > 1 else None
    if follower:
        raise ValueError(
            f"blocks: this {role} call's checkpoint holds no plan for these params, poses "
            f"and prior; only the head call plans"
        )
    if state_dir.exists():
        shutil.rmtree(state_dir)
    plan = _plan(ctx, settings, prior, fingerprint, work)
    _write_json_atomic(state_dir / _PLAN, plan.to_dict(), work)
    if plan.partition.count <= 1:
        ctx.log("blocks: one block is left after the minimum-cameras rule; training it whole")
        return None
    return plan


def _region_points(settings: Settings, prior: Prior) -> F32:
    xyz = prior.xyz
    if settings.region is None:
        return xyz
    inside = np.asarray(training.in_region(settings.region, xyz), dtype=bool)
    return xyz[inside] if inside.any() else xyz


def _plan(
    ctx: StageContext, settings: Settings, prior: Prior, fingerprint: str, work: Path
) -> Plan:
    model = sfm.read_model(settings.dataset / "sparse" / "0")
    up = settings.up or _mean_up(model)
    part = block_maths.partition(
        _region_points(settings, prior), up, settings.count, exact=settings.requested is not None
    )
    names = tuple(image.name for image in model.images)  # sorted, as gsplat's parser sorts
    val = tuple(index % training.TEST_EVERY == 0 for index in range(len(names)))
    centres = model.centres()
    inside = block_maths.cameras_inside(part, centres)
    epsilon = _param_float(settings, "block_epsilon", block_maths.DEFAULT_EPSILON)
    loss, method = _contribution(ctx, settings, part, names, inside, work)
    assigned = block_maths.assign(inside, loss, epsilon)
    counts = np.bincount(part.block_of(part.project(prior.xyz)), minlength=part.count)
    minimum = int(settings.params.get("block_min_images") or block_maths.MIN_IMAGES)
    train_mask = ~np.asarray(val, dtype=bool)
    before = [int((assigned[:, b] & train_mask).sum()) for b in range(part.count)]
    merged, matrix, merges = block_maths.merge_small(
        part, assigned, train_mask, [int(c) for c in counts], minimum
    )
    inside_merged = np.stack(
        [
            inside[:, [b for b, cells in enumerate(part.blocks) if set(cells) <= set(m)]].any(1)
            for m in merged.blocks
        ],
        axis=1,
    )
    ctx.log(
        f"blocks: {part.columns}x{part.rows} grid; training frames per block by {method} "
        f"(epsilon {epsilon:g}): {before}"
        + (
            ""
            if not merges
            else f"; {len(merges)} block(s) under {minimum} merged into a neighbour -> "
            f"{merged.count} blocks"
        )
    )
    merged_counts = np.bincount(merged.block_of(merged.project(prior.xyz)), minlength=merged.count)
    return Plan(
        partition=merged,
        names=names,
        val=val,
        assigned=matrix,
        inside=inside_merged,
        method=method,
        epsilon=epsilon,
        merges=tuple(merges),
        fingerprint=fingerprint,
        prior_source=str(prior.meta.get("source", "unknown")),
        prior_count=prior.count,
        gaussians=tuple(int(c) for c in merged_counts),
    )


def _mean_up(model: sfm.Model) -> list[float]:
    estimate = sfm.camera_up(model)
    return [0.0, -1.0, 0.0] if estimate is None else list(estimate.up)


def _contribution(
    ctx: StageContext,
    settings: Settings,
    part: block_maths.Partition,
    names: Sequence[str],
    inside: npt.NDArray[np.bool_],
    work: Path,
) -> tuple[npt.NDArray[np.float64] | None, str]:
    """(cameras, blocks) of `1 - SSIM` from `block_views.py` on the GPU; if that cannot
    run, VastGaussian's visibility test on the CPU, expressed as 0/1."""
    out = work / "views"
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part.to_dict()), encoding="utf-8")
    result = out / "views.json"
    result.unlink(missing_ok=True)
    script = settings.params.get("block_views_script")
    argv = [
        settings.python,
        str(script or BLOCK_VIEWS),
        "--data_dir",
        str(settings.dataset),
        "--prior",
        str(prior_path(ctx.checkpoint_dir)),
        "--partition",
        str(out / "partition.json"),
        "--out",
        str(result),
        "--trainer",
        str(settings.trainer),
        "--max-side",
        str(int(settings.params.get("block_assign_max_side") or ASSIGN_MAX_SIDE)),
    ]
    try:
        ctx.run(argv)
        document = json.loads(result.read_text(encoding="utf-8"))
        order = {str(name): row for row, name in enumerate(document["names"])}
        matrix = np.asarray(document["loss"], dtype=np.float64)
        loss = np.zeros((len(names), part.count), dtype=np.float64)
        for row, name in enumerate(names):
            if name in order:
                loss[row] = matrix[order[name]]
        return loss, "render"
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, IndexError) as error:
        ctx.log(
            f"WARNING: blocks: the prior could not be rendered for the camera test "
            f"({type(error).__name__}: {str(error)[:300]}); using VastGaussian's visibility "
            f"test on the CPU instead"
        )
    model = sfm.read_model(settings.dataset / "sparse" / "0")
    prior, _ = load_prior(ctx.checkpoint_dir, settings.poses)
    xyz = prior.xyz if prior is not None else np.zeros((0, 3), dtype=np.float32)
    visible = visibility(part, model, xyz, settings.dataset)
    return visible.astype(np.float64), "visibility"


#: VastGaussian §3.3: a cell is visible to a camera when its projected airspace-aware box
#: covers more than this share of the image.
VISIBILITY_SHARE = 0.25


def visibility(
    part: block_maths.Partition, model: sfm.Model, xyz: Any, dataset: Path
) -> npt.NDArray[np.bool_]:
    """VastGaussian's visibility-based camera selection, the fallback to the render test.

    Each block's box: its rectangle in the ground plane, from the lowest to the highest of
    the prior's gaussians in it ("airspace-aware", §3.3); projected into each camera, the
    share of the image its projection covers. VastGS takes the projected corners' convex
    hull; this takes their bounding box (clipped to the image), which is never smaller, so
    it errs toward more cameras. A box with a corner behind the camera covers the image.
    """
    cameras = {camera.id: camera for camera in model.cameras}
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv = part.project(points)
    up = np.cross(np.asarray(part.axis_u), np.asarray(part.axis_v))
    height = (points - np.asarray(part.origin)) @ up
    block = part.block_of(uv)
    out = np.zeros((len(model.images), part.count), dtype=bool)
    for b in range(part.count):
        members = height[block == b]
        low, high = (float(members.min()), float(members.max())) if members.size else (0.0, 0.0)
        u0, u1, v0, v1 = part.rect(b)
        corners = np.array(
            [
                np.asarray(part.origin)
                + u * np.asarray(part.axis_u)
                + v * np.asarray(part.axis_v)
                + h * up
                for u in (u0, u1)
                for v in (v0, v1)
                for h in (low, high)
            ]
        )
        for row, image in enumerate(model.images):
            camera = cameras.get(image.camera_id)
            if camera is None:
                continue
            fx, fy, cx, cy = _intrinsics(camera)
            local = (image.rotation @ corners.T).T + np.asarray(image.tvec)
            if (local[:, 2] <= 1e-9).any():
                out[row, b] = bool((local[:, 2] > 1e-9).any())
                continue
            px = fx * local[:, 0] / local[:, 2] + cx
            py = fy * local[:, 1] / local[:, 2] + cy
            x0, x1 = np.clip([px.min(), px.max()], 0, camera.width)
            y0, y1 = np.clip([py.min(), py.max()], 0, camera.height)
            share = (x1 - x0) * (y1 - y0) / max(1.0, float(camera.width * camera.height))
            out[row, b] = share > VISIBILITY_SHARE
    return out


def _intrinsics(camera: sfm.Camera) -> tuple[float, float, float, float]:
    named = dict(zip(camera.param_names, camera.params, strict=False))
    f = float(named.get("f", named.get("fx", camera.focal_px)))
    fy = float(named.get("fy", f))
    cx = float(named.get("cx", camera.width / 2.0))
    cy = float(named.get("cy", camera.height / 2.0))
    return f, fy, cx, cy


# --- the coarse pass -------------------------------------------------------------------


def _coarse_prior(ctx: StageContext, settings: Settings, work: Path) -> Prior:
    """A short whole-scene run, left as the prior (module docstring, 1). Resumable in the
    sense that matters: its result is the prior, which a retry finds and does not redo."""
    max_side = int(settings.params.get("coarse_max_side") or COARSE_MAX_SIDE)
    scale = _param_float(settings, "coarse_schedule_scale", COARSE_SCHEDULE_SCALE)
    cap = int(settings.params.get("coarse_cap") or COARSE_CAP)
    ceiling = gpu_cap(settings.budget)
    if ceiling is not None:
        cap = min(cap, ceiling)
    dataset = training.build_dataset(
        settings.frames, settings.poses, work / "dataset", max_side=max_side
    )
    if settings.region is not None:
        training.crop_initial_points(dataset / "sparse" / "0", settings.region)
    result = work / "gsplat"
    if result.exists():
        shutil.rmtree(result)
    result.mkdir(parents=True)
    scaler = training.batch_steps_scaler(training.check_schedule_scale(scale), settings.batch_size)
    argv = training.gsplat_argv(
        settings.python,
        settings.trainer,
        dataset,
        result,
        strategy="mcmc",
        max_steps=settings.full_iterations,
        steps_scaler=scaler,
        cap_max=cap,
        antialiased=bool(settings.switches.get("antialiased")),
        batch_size=settings.batch_size,
        extra=settings.extra,
    )
    ctx.log(
        f"blocks: coarse pass: {training.scaled_steps(settings.full_iterations, scaler)} steps "
        f"at {max_side} px, at most {cap} gaussians"
    )
    ctx.run(argv)
    ply = training.latest_ply(result)
    if ply is None:
        raise ValueError(f"the coarse pass wrote no .ply under {result}")
    columns = read_ply(ply)
    saved = save_prior(ctx.checkpoint_dir, columns, settings.poses, scratch=work, source="coarse")
    if saved is None:
        raise ValueError("the coarse pass left no usable gaussians for a prior")
    prior, why = load_prior(ctx.checkpoint_dir, settings.poses)
    if prior is None:
        raise ValueError(f"the coarse prior could not be read back: {why}")
    return prior


# --- training the blocks ---------------------------------------------------------------


@dataclass(frozen=True)
class BlockRun:
    """What `train` hands back to the stage: the documents and the flat metrics."""

    document: dict[str, Any]
    metrics: dict[str, MetricValue]
    gaussians: int
    written: int


def finished(ctx: StageContext, plan: Plan) -> dict[str, Any]:
    """The records of this plan's blocks that are trained and still in `checkpoint/`,
    by block index (as a string). A block's record sits beside its splat, one file each,
    so blocks trained on different machines at once never write the same file; a run
    checkpointed before that (one `state.json`) is still read."""
    state_dir = _state_dir(ctx)
    done: dict[str, Any] = {}
    legacy = _read_json(state_dir / _STATE)
    if legacy is not None and legacy.get("fingerprint") == plan.fingerprint:
        done.update({str(k): v for k, v in dict(legacy.get("blocks") or {}).items()})
    for index in range(plan.partition.count):
        record = _read_json(_record_path(state_dir, index))
        if record is not None and record.get("fingerprint") == plan.fingerprint:
            done[str(index)] = record
    return {k: v for k, v in done.items() if (state_dir / _block_dir(int(k))).is_dir()}


def pending(ctx: StageContext, plan: Plan) -> list[int]:
    """The blocks of `plan` still to train."""
    done = finished(ctx, plan)
    return [index for index in range(plan.partition.count) if str(index) not in done]


def part_id(index: int) -> str:
    return f"b{index}"


def part_index(part: str | None, plan: Plan) -> int:
    """The block a `part` call trains, from the id `fan_out` gave it."""
    try:
        index = int(str(part).removeprefix("b"))
    except ValueError:
        raise ValueError(f"blocks: {part!r} is not a block part") from None
    if not 0 <= index < plan.partition.count:
        raise ValueError(f"blocks: part {part!r} is outside the plan's {plan.partition.count}")
    return index


def fan_out(settings: Settings, plan: Plan, todo: Sequence[int]) -> FanOut:
    """One part per block still to train, the one expected to take longest first.

    Longest first because the stage ends when the last block does: with more blocks than
    `parallel`, starting the long ones early is what keeps the tail short (the classic
    longest-processing-time rule). Expected work is the block's steps -- its schedule
    (`block_schedule`) -- times its gaussians in the prior, the two things a step's cost
    and the number of steps grow with.
    """
    frames = plan.assigned.sum(axis=0)

    def work(index: int) -> float:
        scale, _ = block_schedule(settings, int(frames[index]), len(plan.names))
        return scale * max(1, plan.gaussians[index] if index < len(plan.gaussians) else 1)

    order = sorted(todo, key=lambda index: (-work(index), index))
    parts = tuple(
        FanOutPart(
            id=part_id(index),
            collect=(f"{STATE_DIR}/{_block_dir(index)}", f"{STATE_DIR}/{_block_dir(index)}.json"),
        )
        for index in order
    )
    return FanOut(
        parts=parts, parallel=settings.parallel, share=(PRIOR_DIR, f"{STATE_DIR}/{_PLAN}")
    )


def train_part(ctx: StageContext, settings: Settings, plan: Plan, index: int) -> dict[str, Any]:
    """One block, alone: what a `part` call of a fan-out does. Its splat and record land in
    `checkpoint/blocks/`, which is all that comes home; nothing is written to `out/`."""
    prior, why = load_prior(ctx.checkpoint_dir, settings.poses)
    if prior is None:
        raise ValueError(f"blocks: the prior this plan was made from is gone: {why}")
    uv_prior = plan.partition.project(prior.xyz)
    work = ctx.work_dir / "blocks"
    return _train_block(ctx, settings, plan, prior, uv_prior, index, work / f"b{index}")


def train(ctx: StageContext, settings: Settings, plan: Plan, trained_ply: Path) -> BlockRun:
    """Train every block not already finished, merge them into `trained_ply`, measure."""
    started = settings.started
    work = ctx.work_dir / "blocks"
    done = finished(ctx, plan)
    prior, why = load_prior(ctx.checkpoint_dir, settings.poses)
    if prior is None:
        raise ValueError(f"blocks: the prior this plan was made from is gone: {why}")
    uv_prior = plan.partition.project(prior.xyz)
    budget_s = _param_float(settings, "block_attempt_budget_s", ATTEMPT_BUDGET_S)
    sync_wait_s = _param_float(settings, "block_sync_wait_s", SYNC_WAIT_S)
    trained_here = 0
    for index in range(plan.partition.count):
        key = str(index)
        if key in done:
            ctx.log(
                f"blocks: block {index + 1} of {plan.partition.count} finished in an "
                f"earlier call; not trained again"
            )
            continue
        longest = max((float(r.get("seconds") or 0.0) for r in done.values()), default=0.0)
        elapsed = time.monotonic() - started
        if trained_here >= 1 and elapsed + longest > budget_s:
            ctx.log(
                f"blocks: {len(done)} of {plan.partition.count} blocks done; another could "
                f"take {longest:.0f} s and this attempt has used {elapsed:.0f} of its "
                f"{budget_s:.0f} s, so it ends here and the next attempt resumes"
            )
            time.sleep(max(0.0, sync_wait_s))
            raise BlocksYieldError(
                f"{len(done)} of {plan.partition.count} blocks trained; the rest resume on "
                f"the next attempt"
            )
        record = _train_block(ctx, settings, plan, prior, uv_prior, index, work / f"b{index}")
        done[key] = record
        trained_here += 1
    return _merge_and_measure(ctx, settings, plan, done, trained_ply, work, started)


def _block_dir(index: int) -> str:
    return f"block_{index:03d}"


def _record_path(state_dir: Path, index: int) -> Path:
    return state_dir / f"{_block_dir(index)}.json"


def block_schedule(settings: Settings, frames: int, of: int) -> tuple[float, str]:
    """A block's schedule scale, before its gaussian factor, and the rule that set it.

    `frames` are the posed frames the block is given (training and held out, as a whole
    run counts its dataset's), `of` the run's. Under `frames`, the default, it is the
    whole run's rule on the block: `training.schedule_scale(frames,
    full_at=schedule_full_at, floor=schedule_floor)`, carried as a ratio to the run's own
    (`settings.images`) so that a Refine's `init_schedule_scale` is shortened alike. A run
    that named its `schedule_scale` keeps it, as a whole run does; and a block given every
    frame keeps the run's schedule exactly -- the spool's two blocks, each with all 156
    training cameras, train what they did before. `share` scales by `frames / of` instead;
    `full` is the run's schedule for every block. Never below `schedule_floor` of the full
    schedule, unless the run itself is (a Preview's 0.1).

    With the recipe's `schedule_full_at: 60` and the 50-frame minimum a block must have
    (`block_maths.MIN_IMAGES`), `frames` shortens only a block of 50-59 frames, by at most
    a sixth: the rule was set for small captures, and a block of a big scene is not one.
    `share` is the rule that shrinks a big scene's blocks -- to the steps each frame gets
    in the whole run -- and is opt-in until a GPU run has measured its quality.
    """
    mode = settings.block_schedule
    run = settings.steps_scaler
    if mode == "full" or of <= 0 or frames >= of:
        return run, "the run's (the block sees every frame)" if frames >= of else "the run's"
    if mode == "frames":
        full_at = settings.schedule_full_at
        if settings.schedule_requested or not full_at:
            given = "schedule_scale given" if settings.schedule_requested else "no schedule_full_at"
            return run, f"the run's ({given})"
        floor = settings.schedule_floor
        mine = training.schedule_scale(frames, full_at=full_at, floor=floor)
        whole = training.schedule_scale(max(settings.images, of), full_at=full_at, floor=floor)
        ratio = mine / whole if whole > 0 else 1.0
        rule = f"{frames} frames against the full schedule at {full_at}"
    else:
        ratio = frames / of
        rule = f"{frames} of the run's {of} frames"
    scale = max(round(run * ratio, 4), min(run, settings.schedule_floor))
    return scale, rule


def write_parts(directory: Path, columns: Mapping[str, Any], names: Sequence[str]) -> int:
    """A block's splat as PLYs of at most `PART_ROWS` rows each (at least one, maybe
    empty), so no file in the checkpoint is bigger than a few tens of megabytes: the
    worker's transfer reads each file whole into its 2 GB (`apps/api/app/worker/cloud.py`)
    when it carries a resumed run's checkpoint back to the GPU. Returns the rows written."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    count = int(np.asarray(columns[names[0]]).shape[0])
    for part, start in enumerate(range(0, max(count, 1), PART_ROWS)):
        piece = {name: np.asarray(columns[name])[start : start + PART_ROWS] for name in names}
        write_columns(directory / f"part_{part:04d}.ply", piece, names)
    return count


def block_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob("part_*.ply"))


def _block_cap(
    settings: Settings, region: BlockRegion, share: float
) -> tuple[int | None, gaussian_budget.Budget | None]:
    """The block's gaussian budget: the capture's density over the block's own surface
    when the run's budget is `auto` (clamped by the same GPU ceilings), otherwise the run's
    cap shared by the prior's gaussians in the block."""
    budget = settings.budget
    if budget is None:
        return None, None
    if budget.mode != "auto":
        return max(1, round(budget.cap * share)), None
    block_budget = gaussian_budget.plan(
        gaussian_budget.AUTO,
        model=settings.poses,
        train_size=settings.frame_size,
        pixel_scale=settings.pixel_scale,
        region_contains=region.contains,
        outside_weight=1.0 / training.ROI_OUTSIDE_EVERY,
        density=budget.density,
        density_scale=budget.density_scale,
        floor=max(1, budget.floor // max(1, settings.count)),
        budget_max=budget.budget_max,
        gpu_memory_gb=budget.gpu_memory_gb,
        images_per_step=settings.batch_size,
    )
    return (None if block_budget is None else block_budget.cap), block_budget


def _train_block(
    ctx: StageContext,
    settings: Settings,
    plan: Plan,
    prior: Prior,
    uv_prior: Any,
    index: int,
    work: Path,
) -> dict[str, Any]:
    part = plan.partition
    total = part.count
    margin = _param_float(settings, "block_margin", block_maths.DEFAULT_MARGIN)
    falloff = _param_float(settings, "block_blend", block_maths.DEFAULT_FALLOFF)
    region = BlockRegion(part, index, margin, settings.region)
    names = [n for n, a in zip(plan.names, plan.assigned[:, index], strict=True) if a]
    val_flags = dict(zip(plan.names, plan.val, strict=True))
    train_names = [n for n in names if not val_flags[n]]
    val_names = [n for n in names if val_flags[n]]
    order, every = block_maths.val_order(train_names, val_names)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    dataset = work / "dataset"
    build_block_dataset(settings.dataset, dataset, order)
    sparse = dataset / "sparse" / "0"
    crop = training.crop_initial_points(sparse, cast("Any", region))
    trainable = np.asarray(region.contains(prior.xyz), dtype=bool)
    share = float(trainable.sum()) / max(1, prior.count)
    cap, block_budget = _block_cap(settings, region, share)
    applied = init_seed.apply(
        sparse,
        prior.as_seed(),
        region,
        budget=init_seed.budget_for(cap, settings.init_max_points),
    )
    # The frozen ring: H3DGS excludes its scaffold from densification; here it is not
    # optimised at all (block_trainer.py), and needs MCMC, whose densification reads no
    # per-gaussian screen-space gradients of the gaussians it is rendered with.
    ring_on = _param_bool(settings, "block_ring", True) and settings.strategy == "mcmc"
    ring_outer = _param_float(settings, "block_ring_outer", block_maths.DEFAULT_RING_OUTER)
    ring_mask = (
        np.asarray(
            block_maths.ring(part, uv_prior, index, margin=margin, outer=ring_outer), dtype=bool
        )
        if ring_on
        else np.zeros(prior.count, dtype=bool)
    )
    if settings.region is not None and ring_on:
        # Nothing outside the run's own region is scene worth occluding with.
        ring_mask &= np.asarray(training.in_region(settings.region, prior.xyz), dtype=bool)
    ring_path: Path | None = None
    if ring_mask.any():
        ring_path = work / "ring.npz"
        np.savez(
            ring_path,
            means=prior.xyz[ring_mask],
            log_scales=prior.log_scales[ring_mask],
            quats=prior.quats[ring_mask],
            logit_opacity=prior.logit_opacity[ring_mask],
            f_dc=prior.f_dc[ring_mask],
        )
    # The schedule: the single run's rule on this block's own frames (`block_schedule`),
    # lengthened for a bigger block budget as a whole run's would be, and divided by the
    # images a step sees (`batch_size`) so the images trained on stay the same.
    scaler, schedule_rule = block_schedule(settings, len(names), len(plan.names))
    data_scale = scaler
    factor = 1.0
    if settings.converge and settings.strategy == "mcmc" and cap is not None:
        factor = gaussian_budget.schedule_factor(cap)
        scaler = round(scaler * factor, 4)
    scaler = training.batch_steps_scaler(scaler, settings.batch_size)
    iterations = training.scaled_steps(settings.full_iterations, scaler)
    refine_stop = convergence.REFINE_STOP_ITER.get(settings.strategy)
    extra_evals = (
        convergence.eval_steps(settings.full_iterations, refine_stop, every=settings.rule.every)
        if settings.converge and refine_stop is not None
        else []
    )
    live_steps = live.train_ply_steps(settings.full_iterations) if settings.live else []
    result = work / "gsplat"
    result.mkdir(parents=True)
    argv = training.gsplat_argv(
        settings.python,
        settings.trainer,
        dataset,
        result,
        strategy=settings.strategy,
        max_steps=settings.full_iterations,
        data_factor=settings.data_factor,
        steps_scaler=scaler,
        cap_max=cap,
        antialiased=bool(settings.switches.get("antialiased")),
        opacity_reg=settings.switches.get("opacity_reg"),
        depth_loss=bool(settings.switches.get("depth_loss")),
        pose_opt=bool(settings.switches.get("pose_opt")),
        app_opt=bool(settings.switches.get("app_opt")),
        bilateral_grid=bool(settings.switches.get("bilateral_grid")),
        live_steps=live_steps,
        eval_steps=extra_evals,
        batch_size=settings.batch_size,
        # The block's own held-out frames (block_maths.val_order).
        extra=[*settings.extra, "--test_every", str(every)],
    )
    if extra_evals or ring_path is not None:
        argv = trainer_argv(
            argv,
            script=Path(settings.params.get("block_trainer_script") or BLOCK_TRAINER),
            rule=settings.rule if extra_evals else None,
            ring=ring_path,
        )
    ctx.log(
        f"blocks: block {index + 1} of {total}: {len(train_names)} training frames "
        f"({int(plan.inside[:, index].sum())} standing in it), {len(val_names)} held out "
        f"(test_every {every}); cap {cap}; {applied.seeded} prior gaussians seeded, "
        f"{int(ring_mask.sum())} frozen in the ring; schedule {data_scale:g} "
        f"({schedule_rule}) -> {iterations} steps"
        + ("" if settings.batch_size == 1 else f" of {settings.batch_size} images")
    )
    began = time.monotonic()
    with live.SplatWatch(
        result / "ply",
        ctx.checkpoint_dir / live.LIVE_DIR,
        ctx.log,
        indices=live.ply_indices(live_steps, scaler),
        total=iterations,
        key_prefix=f"{ctx.checkpoint_key}/{live.LIVE_DIR}",
        up=settings.up,
    ):
        ctx.run(argv)
    seconds = time.monotonic() - began
    ply = training.latest_ply(result)
    if ply is None:
        raise ValueError(f"block {index + 1}: the trainer wrote no .ply under {result}")
    columns = read_ply(ply)
    trained = int(np.asarray(columns["x"]).shape[0])
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    weight = np.asarray(block_maths.weights(part, part.project(xyz), index, falloff=falloff))
    keep = weight > 0.0
    keep &= np.isfinite(xyz).all(axis=1)
    if settings.region is not None:
        keep &= np.asarray(training.crop_rows(columns, settings.region), dtype=bool)
    kept = {name: np.asarray(values)[keep] for name, values in columns.items()}
    kept["opacity"] = block_maths.blended_opacity(kept["opacity"], weight[keep])
    partial = work / _block_dir(index)
    write_parts(partial, kept, list(columns))
    target = _state_dir(ctx) / _block_dir(index)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)
    metrics = training.parse_metrics(
        result, "", trainer=f"gsplat:{settings.trainer.name}", requested_iterations=iterations
    )
    report = convergence.read_report(result) if extra_evals else None
    record: dict[str, Any] = {
        "fingerprint": plan.fingerprint,
        "index": index,
        "cells": list(part.blocks[index]),
        "rect": list(part.rect(index)),
        "cameras": {
            "train": len(train_names),
            "val": len(val_names),
            "inside": int(plan.inside[:, index].sum()),
            "testEvery": every,
        },
        "capMax": cap,
        "budget": None if block_budget is None else block_budget.to_dict(),
        "scheduleScale": scaler,
        "scheduleData": data_scale,
        "scheduleRule": schedule_rule,
        "scheduleFactor": factor,
        "batchSize": settings.batch_size,
        "stepsMax": iterations,
        "stepsRun": metrics.iterations,
        "stoppedEarly": bool(report and report.get("stoppedEarly")),
        "seedPoints": applied.seeded,
        "initialPoints": crop.points_kept,
        "ring": int(ring_mask.sum()),
        "gaussiansTrained": trained,
        "gaussiansKept": int(keep.sum()),
        "gaussiansFaded": int((keep & (weight < 1.0)).sum()),
        "psnr": metrics.psnr,
        "ssim": metrics.ssim,
        "lpips": metrics.lpips,
        "peakMemoryGb": metrics.peak_memory_gb,
        "seconds": round(seconds, 1),
        "attempt": ctx.attempt,
    }
    # After the splat is in place: a block with a splat and no record is trained again,
    # never merged half-written.
    _write_json_atomic(_record_path(_state_dir(ctx), index), record, work)
    return record


def trainer_argv(
    argv: Sequence[str],
    *,
    script: Path,
    rule: convergence.Rule | None,
    ring: Path | None,
) -> list[str]:
    """`gsplat_argv`'s command run through `block_trainer.py`, as `training.converge_argv`
    runs it through `converge_trainer.py`: the wrapper's own options, `--`, then the
    trainer's argv unchanged."""
    python, trainer, *rest = argv
    own = [python, str(script), "--trainer", trainer]
    if rule is not None:
        own += [
            "--converge",
            "--every",
            str(rule.every),
            "--window",
            str(rule.window),
            "--min-gain-db",
            f"{rule.min_gain_db:g}",
        ]
    if ring is not None:
        own += ["--ring", str(ring)]
    return [*own, "--", *rest]


# --- the merge, and what it measured ---------------------------------------------------


def _merge_and_measure(
    ctx: StageContext,
    settings: Settings,
    plan: Plan,
    done: Mapping[str, Any],
    trained_ply: Path,
    work: Path,
    started: float,
) -> BlockRun:
    directories = [_state_dir(ctx) / _block_dir(i) for i in range(plan.partition.count)]
    parts = [part for directory in directories for part in block_parts(directory)]
    full = work / "merged_full.ply"
    count, _ = merge_plys(parts, full)
    _, written = merge_plys(parts, trained_ply, gaussians.CANONICAL_PROPERTIES)
    ctx.log(
        f"blocks: merged {plan.partition.count} blocks into {count} gaussians "
        f"({', '.join(str(done[str(i)]['gaussiansKept']) for i in range(plan.partition.count))})"
    )
    evaluation = _evaluate(ctx, settings, full, work / "eval")
    held_out = holdout.measure(
        ctx,
        enabled=settings.holdout_error,
        python=settings.python,
        trainer=settings.trainer,
        dataset=settings.dataset,
        ply=full,
        rows=count,
        keep=None,
        data_factor=settings.data_factor,
        test_every=training.TEST_EVERY,
        antialiased=bool(settings.switches.get("antialiased")),
        script=settings.holdout_script,
        budget_s=settings.holdout_budget_s,
    )
    # The splats are in the merged output now; the checkpoint keeps only the records, so
    # a later run does not carry gigabytes of finished blocks back and forth.
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
    records = [
        {k: v for k, v in dict(done[str(i)]).items() if k != "fingerprint"}
        for i in range(plan.partition.count)
    ]
    seconds = sum(float(r.get("seconds") or 0.0) for r in records)
    # A block's share of the training time: the GPU is priced per stage (the attempt
    # ledger, outside this container), so a block's cost is the stage's times this.
    for record in records:
        record["costShare"] = round(float(record.get("seconds") or 0.0) / max(seconds, 1e-9), 4)
    document: dict[str, Any] = {
        "count": plan.partition.count,
        "requested": settings.requested,
        "reason": settings.reason,
        "gpuCap": gpu_cap(settings.budget),
        "grid": [plan.partition.columns, plan.partition.rows],
        "partition": plan.partition.to_dict(),
        "prior": {"source": plan.prior_source, "gaussians": plan.prior_count},
        "cameraTest": {
            "method": plan.method,
            "epsilon": plan.epsilon,
            "minImages": int(settings.params.get("block_min_images") or block_maths.MIN_IMAGES),
            "merges": [{"into": a, "absorbed": b, "trainFrames": c} for a, b, c in plan.merges],
        },
        "margin": _param_float(settings, "block_margin", block_maths.DEFAULT_MARGIN),
        "ring": {
            "enabled": _param_bool(settings, "block_ring", True) and settings.strategy == "mcmc",
            "outer": _param_float(settings, "block_ring_outer", block_maths.DEFAULT_RING_OUTER),
        },
        "blend": _param_float(settings, "block_blend", block_maths.DEFAULT_FALLOFF),
        "exposure": "bilateral_grid" if settings.switches.get("bilateral_grid") else None,
        "blocks": records,
        "merged": {"gaussians": count, "trainedBytes": written},
        "evaluation": evaluation[1],
        "seconds": round(seconds, 1),
        "stageSeconds": round(time.monotonic() - started, 1),
    }
    metrics: dict[str, MetricValue] = {
        "blocks": plan.partition.count,
        "blockCameras": ",".join(str(r["cameras"]["train"]) for r in records),
        "blockGaussians": ",".join(str(r["gaussiansKept"]) for r in records),
        "blockCaps": ",".join(str(r["capMax"]) for r in records),
        "blockSteps": ",".join(str(r["stepsRun"]) for r in records),
        "blockSchedules": ",".join(str(r.get("scheduleScale")) for r in records),
        "blockSeconds": ",".join(str(r["seconds"]) for r in records),
        "blockEpsilon": plan.epsilon,
        "blockCameraTest": plan.method,
        "blockMerges": len(plan.merges),
        "blockPrior": plan.prior_source,
    }
    return BlockRun(
        document={
            "blocks": document,
            "evaluationMetrics": None if evaluation[0] is None else evaluation[0].to_dict(),
            "holdoutSummary": held_out,
        },
        metrics=metrics,
        gaussians=count,
        written=written,
    )


def _evaluate(
    ctx: StageContext, settings: Settings, merged: Path, work: Path
) -> tuple[training.TrainMetrics | None, dict[str, Any]]:
    """The merged splat on the run's held-out frames, by gsplat's own `eval()`: the numbers
    a single run reports, on the same frames. `block_trainer.py --ply-to-ckpt` turns the PLY
    into the trainer's checkpoint format, and the trainer's `--ckpt` (v1.5.3: "run eval
    only") scores it. Cannot fail the stage."""
    if not _param_bool(settings, "block_eval", True):
        return None, {"status": "off"}
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    checkpoint = work / "merged.pt"
    script = Path(settings.params.get("block_trainer_script") or BLOCK_TRAINER)
    try:
        ctx.run(
            [
                settings.python,
                str(script),
                "--ply-to-ckpt",
                str(merged),
                str(checkpoint),
                "--step",
                str(training.scaled_steps(settings.full_iterations, settings.steps_scaler) - 1),
            ]
        )
        result = work / "gsplat"
        result.mkdir()
        switches = dict(settings.switches)
        argv = training.gsplat_argv(
            settings.python,
            settings.trainer,
            settings.dataset,
            result,
            strategy=settings.strategy,
            max_steps=settings.full_iterations,
            data_factor=settings.data_factor,
            steps_scaler=settings.steps_scaler,
            antialiased=bool(switches.get("antialiased")),
            # The PLY is baked (degree-0 colour under a zero embedding), so it is scored
            # as the plain splat it is, never through an appearance module.
            bilateral_grid=bool(switches.get("bilateral_grid")),
            extra=[*settings.extra, "--ckpt", str(checkpoint)],
        )
        ctx.run(argv)
        metrics = training.parse_metrics(result, "", trainer=f"gsplat:{settings.trainer.name}")
        if metrics.psnr is None:
            raise ValueError("the evaluation wrote no held-out numbers")
        return metrics, {"status": "ok", "psnr": metrics.psnr, "lpips": metrics.lpips}
    except Exception as problem:  # anything: the merged splat is already written
        reason = f"{type(problem).__name__}: {problem}"[:600]
        ctx.log(f"WARNING: blocks: the merged splat was not evaluated ({reason})")
        return None, {"status": "failed", "reason": reason}
