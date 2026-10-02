"""When to stop training: held-out quality has stopped improving, and not before it can.

Plain Python with no numpy, and nothing newer than 3.10, because two interpreters import
it: the pipeline's (3.12) to build the schedule and read the curve back, and the
trainer's (gsplat's CUDA wheels are CPython 3.10 only), where `converge_trainer.py` makes
the decision inside the training loop -- as `holdout_maths.py` is shared by `holdout.py`
and `holdout_error.py`.

**What the schedule allows.** gsplat v1.5.3's `simple_trainer.py`, read at the tag:

* `MCMCStrategy.step_post_backward` relocates dead gaussians and adds new ones (5% more a
  time, up to `cap_max`) every `refine_every` steps while `step < refine_stop_iter` --
  25,000 of the 30,000 (the default strategy's densification stops at 15,000) -- and all
  of these are scaled by `--steps_scaler` with `max_steps`. Stopping before then stops a
  run that is still moving gaussians around, whose held-out numbers say nothing about
  where it would settle: **no stop is ever considered before densification ends.**
* The means' learning rate decays exponentially to 1% over `max_steps`
  (`ExponentialLR(gamma=0.01 ** (1 / max_steps))`), and MCMC's position noise is scaled
  by it. So the tail after densification is where the splat settles, and the only part
  of a run that can end early: at most the last sixth of an MCMC schedule. That is the
  honest size of what this saves on a run of the recipe's length -- it is a bound on
  paying for a tail that has gone flat, and it is what makes a *longer* maximum schedule
  for a big budget (`gaussian_budget.schedule_factor`) affordable.

**The rule.** gsplat evaluates the `val` split (every 8th frame, never trained on) at
each `--eval_steps` entry. The stage asks for an evaluation at the end of densification,
every `every` steps after it, and a coarse curve before it (`eval_steps`); all unscaled,
as the trainer scales them. After each evaluation at or past densification's end, with a
full `window` of them behind it:

    gain = best PSNR in the last `window` steps - best PSNR before them
    stop if gain < min_gain_db

`min_gain_db` 0.05 dB over 2,000 steps: smaller than anything a second run could tell
apart -- the spool capture's baseline and its repeat differed by 0.22 dB, and this
stage's Truck result was 0.07 dB from the paper's (experiments/run_variants.py and
experiments/benchmark.py). The best, not the last, on either side so that one noisy
evaluation neither starts nor stops anything. PSNR rather than the training loss because
the loss is on frames the splat is fitted to; rather than LPIPS because PSNR is what the
benchmark and the phone's comparisons report.

Stopping is done by the trainer's own code, not by killing it: `converge_trainer.py`
adds the next step to the trainer's `ply_steps`, `save_steps` and `eval_steps`, so the
final PLY, checkpoint and statistics are written where and how v1.5.3 always writes them
(`training.latest_ply` and `parse_metrics` find them unchanged), and ends the loop after.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_CURVE_EVERY",
    "DEFAULT_EVERY",
    "DEFAULT_MIN_GAIN_DB",
    "DEFAULT_WINDOW",
    "REFINE_STOP_ITER",
    "REPORT",
    "TIMING",
    "Decision",
    "Rule",
    "decide",
    "eval_steps",
    "read_curve",
    "read_report",
    "read_timing",
]

#: gsplat v1.5.3's densification ends, unscaled: `MCMCStrategy.refine_stop_iter` and
#: `DefaultStrategy.refine_stop_iter`.
REFINE_STOP_ITER: dict[str, int] = {"mcmc": 25_000, "default": 15_000}

#: Steps between held-out evaluations after densification, unscaled. Each one renders
#: every val frame (22 of the spool capture's 173) and scores it: seconds on an L4.
DEFAULT_EVERY = 500
#: The span the gain is measured over, unscaled: four evaluations.
DEFAULT_WINDOW = 2_000
#: The gain, in dB of held-out PSNR over `DEFAULT_WINDOW`, below which a run has converged.
DEFAULT_MIN_GAIN_DB = 0.05
#: Evaluations before densification ends, unscaled, for the curve only.
DEFAULT_CURVE_EVERY = 5_000

#: What `converge_trainer.py` writes beside the trainer's `stats/`.
REPORT = "converge.json"
#: And where the trainer's own time went (`converge_trainer.Timing`), written by both
#: wrappers whether or not the convergence rule is on.
TIMING = "trainer_timing.json"


@dataclass(frozen=True)
class Rule:
    """The stopping rule's three numbers, unscaled."""

    every: int = DEFAULT_EVERY
    window: int = DEFAULT_WINDOW
    min_gain_db: float = DEFAULT_MIN_GAIN_DB

    def __post_init__(self) -> None:
        if self.every <= 0 or self.window < self.every:
            raise ValueError(
                f"converge_every must be positive and converge_window at least one "
                f"evaluation long, not {self.every} and {self.window}"
            )
        if not (math.isfinite(self.min_gain_db) and self.min_gain_db >= 0.0):
            raise ValueError(f"converge_min_gain_db must be >= 0, not {self.min_gain_db!r}")

    def to_dict(self) -> dict[str, object]:
        return {"every": self.every, "window": self.window, "minGainDb": self.min_gain_db}


@dataclass(frozen=True)
class Decision:
    stop: bool
    #: `densifying`, `warming up` (less than a window since densification ended),
    #: `improving` or `converged`.
    reason: str
    gain_db: float | None = None


def eval_steps(
    max_steps: int,
    refine_stop: int,
    *,
    every: int = DEFAULT_EVERY,
    curve_every: int = DEFAULT_CURVE_EVERY,
) -> list[int]:
    """The extra `--eval_steps` (unscaled, as the trainer scales them): a coarse curve up
    to densification's end, one at its end, then every `every` steps to `max_steps`
    (exclusive: the final evaluation is always requested separately).

    Empty when densification runs to the end of the schedule: there is then no tail to
    stop early, and no reason to pay for the evaluations.
    """
    if refine_stop >= max_steps or every <= 0:
        return []
    steps = set(range(curve_every, refine_stop, curve_every)) if curve_every > 0 else set()
    steps.update(range(refine_stop, max_steps, every))
    return sorted(step for step in steps if 0 < step < max_steps)


def decide(
    history: Sequence[tuple[int, float]],
    *,
    start: int,
    window: int,
    min_gain_db: float,
) -> Decision:
    """Whether to stop after the latest of `history`'s (zero-based step, PSNR) pairs.

    `start` is the scaled `refine_stop_iter`: an evaluation at step `s` counts once
    `s + 1 >= start` (gsplat evaluates list entry `i` at zero-based step `i - 1`, so the
    one the stage places at densification's end is exactly `start - 1`). `window` is
    scaled too. A non-finite PSNR counts as the worst possible, never as a plateau.
    """
    settled = [(step, _finite(psnr)) for step, psnr in history if step + 1 >= start]
    if not settled:
        return Decision(False, "densifying")
    latest = settled[-1][0]
    before = [psnr for step, psnr in settled if step <= latest - window]
    recent = [psnr for step, psnr in settled if step > latest - window]
    if not before or not recent:
        return Decision(False, "warming up")
    gain = max(recent) - max(before)
    if not math.isfinite(gain):
        return Decision(False, "improving", None)
    rounded = round(gain, 4)
    if gain < min_gain_db:
        return Decision(True, "converged", rounded)
    return Decision(False, "improving", rounded)


def _finite(value: float) -> float:
    return value if isinstance(value, int | float) and math.isfinite(value) else -math.inf


def read_curve(stats: Path) -> list[dict[str, object]]:
    """Every `val_step<i>.json` gsplat wrote, in step order: the held-out curve.

    `step` is the number of steps completed (the file's zero-based index plus one), as
    `train_metrics.json`'s `iterations` counts them.
    """
    points: list[dict[str, object]] = []
    if not stats.is_dir():
        return points
    for path in stats.glob("val_step*.json"):
        digits = "".join(ch for ch in path.stem[len("val_step") :] if ch.isdigit())
        if not digits:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        point: dict[str, object] = {"step": int(digits) + 1}
        for key in ("psnr", "ssim", "lpips", "num_GS"):
            value = document.get(key)
            # Finite only: an evaluation that skipped LPIPS (`converge_trainer.py`) has
            # none, and a NaN would make `train_metrics.json` invalid JSON.
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                point["gaussians" if key == "num_GS" else key] = (
                    int(value) if key == "num_GS" else round(float(value), 4)
                )
        points.append(point)
    return sorted(points, key=lambda point: int(str(point["step"])))


def read_report(result_dir: Path) -> dict[str, object] | None:
    """`converge.json`, if the hook wrote one that parses."""
    path = result_dir / REPORT
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def read_timing(result_dir: Path) -> dict[str, object] | None:
    """`trainer_timing.json`, if a wrapper wrote one that parses."""
    try:
        document = json.loads((result_dir / TIMING).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def history_of(points: Iterable[dict[str, object]]) -> list[tuple[int, float]]:
    """`read_curve`'s points as `decide`'s (zero-based step, PSNR) pairs."""
    out: list[tuple[int, float]] = []
    for point in points:
        psnr = point.get("psnr")
        if isinstance(psnr, int | float):
            out.append((int(str(point["step"])) - 1, float(psnr)))
    return out
