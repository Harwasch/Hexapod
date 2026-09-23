"""Dispatching a 3DGS training run: the dataset it needs, the argv, the resume, the PLY.

**No training run has been executed anywhere in this repository.** `gsplat` needs CUDA,
there is no GPU on the machine this was written on, and none of `modal.com`,
`api.modal.com`, `rest.runpod.io`, `api.runpod.io` or `console.vast.ai` is reachable
through this session's proxy. So what is built here is the *dispatch*: the directory
layout the trainer is handed, the command line it is given, where its checkpoints go so a
preempted attempt resumes, how its metrics are read back, and how its PLY becomes
`canonical.ply`. Every one of those is exercised by a test. What is **not** exercised is
`gsplat` itself, and the two facts below are transcribed from its `examples/` directory
rather than observed here:

* the trainer is `examples/simple_trainer.py`, whose first positional argument selects a
  strategy (`default` or `mcmc`) and which takes `--data_dir`, `--data_factor`,
  `--result_dir`, `--max_steps` and `--ckpt`;
* under `--result_dir` it writes `ckpts/ckpt_<step>_rank<n>.pt`, `stats/val_step<step>_
  rank<n>.json` (`psnr`, `ssim`, `lpips`, `num_GS`, `ellipse_time`) and
  `ply/point_cloud_<step>.ply`.

`parse_metrics` is written to survive both being wrong: it reads whatever `stats/*.json`
it finds, falls back to the trainer's stdout, and records `null` for anything it cannot
find rather than inventing a number. A `train_metrics.json` full of nulls is a stage that
ran and told you nothing; a plausible one that nobody produced is worse.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CHECKPOINT_GLOB",
    "TrainMetrics",
    "build_dataset",
    "gsplat_argv",
    "latest_checkpoint",
    "latest_ply",
    "parse_metrics",
    "step_of",
    "trainer_script",
]

#: What `gsplat`'s trainer names its checkpoints. The step number is what orders them.
CHECKPOINT_GLOB = "ckpt_*.pt"
_STEP_RE = re.compile(r"(\d+)")
#: `Step 6999: 123456 GSs` and friends -- the stdout fallback when no stats JSON exists.
_STDOUT_GS_RE = re.compile(r"[Ss]tep\s+(\d+).*?([\d,]+)\s*GSs")
_STDOUT_PSNR_RE = re.compile(r"[Pp]snr[:=]\s*([0-9.]+)")


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
    checkpoint: Path | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """The command line, built in one place so a test can read it without a GPU.

    `--disable_viewer` is not optional in this context: the trainer's default is to open
    a viewer server and block, which on a headless preemptible box is a stage that never
    finishes and is billed for the whole tier.
    """
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
        str(max_steps),
        "--disable_viewer",
    ]
    if checkpoint is not None:
        argv += ["--ckpt", str(checkpoint)]
    argv += list(extra)
    return argv


def latest_checkpoint(directory: Path) -> Path | None:
    """The furthest-along checkpoint under `directory`, by the step number in its name.

    By step rather than by mtime: a checkpoint synced back from object storage after a
    preemption has whatever mtime the copy gave it, and "newest file" would then be a
    property of the transfer rather than of the training run.
    """
    if not directory.is_dir():
        return None
    found = sorted(directory.rglob(CHECKPOINT_GLOB), key=lambda p: (step_of(p), p.name))
    return found[-1] if found else None


def latest_ply(directory: Path) -> Path | None:
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
            iterations=step,
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
        iterations=_int(scraped.get("step")),
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


def _int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 6)
