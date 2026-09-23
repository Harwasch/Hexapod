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
model, which is what `opensplat` and `nerfstudio` want too.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GSPLAT_VERSION",
    "TrainMetrics",
    "build_dataset",
    "gsplat_argv",
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
_STDOUT_PSNR_RE = re.compile(r"PSNR:\s*([0-9.]+)")

#: The gsplat release every flag and file name here was read from.
GSPLAT_VERSION = "1.5.3"


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


def build_dataset(frames: Path, poses: Path, root: Path) -> Path:
    """COLMAP's on-disk layout, assembled in `work/` from the two input artifacts.

    `images/` and `sparse/0/` are what every 3DGS trainer's COLMAP parser looks for.
    Files are copied rather than linked: the trainer may be in another container with
    this directory mounted, and a symlink out of it resolves to nothing there.
    """
    images = root / "images"
    sparse = root / "sparse" / "0"
    for directory in (images, sparse):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for frame in sorted(p for p in frames.iterdir() if p.is_file()):
        shutil.copyfile(frame, images / frame.name)
    for entry in sorted(p for p in poses.iterdir() if p.is_file()):
        shutil.copyfile(entry, sparse / entry.name)
    return root


def gsplat_argv(
    python: str,
    trainer: Path,
    dataset: Path,
    result_dir: Path,
    *,
    strategy: str = "default",
    max_steps: int = 30_000,
    data_factor: int = 1,
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

    `--disable_video` too: the trajectory render after evaluation is a video nobody reads.
    There is no `--ckpt`, and there must not be: in v1.5.3 it means "evaluate this and do
    not train".
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
        "--save_steps",
        steps,
        "--eval_steps",
        steps,
    ]
    argv += list(extra)
    return argv


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
    resumed_from_step: int | None = None
    attempts: int = 1
    source: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "trainer": self.trainer,
            "iterations": self.iterations,
            "requestedIterations": self.requested_iterations,
            "gaussians": self.gaussians,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "trainSeconds": self.train_seconds,
            "resumedFromStep": self.resumed_from_step,
            "attempts": self.attempts,
            # Which of the two readers produced the numbers above, so a reader of the
            # file can tell a parsed stats file from a scraped log line.
            "source": self.source,
        }


def parse_metrics(
    result_dir: Path,
    log_text: str = "",
    *,
    trainer: str = "gsplat",
    requested_iterations: int | None = None,
    resumed_from_step: int | None = None,
    attempts: int = 1,
) -> TrainMetrics:
    """Read back what the trainer measured: its stats files first, its stdout second."""
    stats = _read_stats(result_dir)
    if stats is not None:
        step, document = stats
        return TrainMetrics(
            trainer=trainer,
            # Zero-based in every file name v1.5.3 writes: `val_step29999` is 30,000 steps.
            iterations=step + 1,
            requested_iterations=requested_iterations,
            gaussians=_int(document.get("num_GS")),
            psnr=_float(document.get("psnr")),
            ssim=_float(document.get("ssim")),
            lpips=_float(document.get("lpips")),
            train_seconds=_float(document.get("ellipse_time")),
            resumed_from_step=resumed_from_step,
            attempts=attempts,
            source="stats",
        )
    scraped = _scrape(log_text)
    return TrainMetrics(
        trainer=trainer,
        iterations=_completed(scraped.get("step")),
        requested_iterations=requested_iterations,
        gaussians=_int(scraped.get("gaussians")),
        psnr=scraped.get("psnr"),
        resumed_from_step=resumed_from_step,
        attempts=attempts,
        source="stdout" if scraped else "none",
    )


def _read_stats(result_dir: Path) -> tuple[int, Mapping[str, object]] | None:
    """The furthest-along `stats/val_step*.json`, or any `stats/*.json` if there is none."""
    stats = result_dir / "stats"
    if not stats.is_dir():
        return None
    candidates = sorted(stats.glob("val_step*.json")) or sorted(stats.glob("*.json"))
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
    psnrs = _STDOUT_PSNR_RE.findall(text)
    if psnrs:
        out["psnr"] = float(psnrs[-1])
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
