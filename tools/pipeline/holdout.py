"""Measured accuracy: the `holdout` artifact, how `train` makes it, how `quality` reads it.

Coverage says a gaussian *could* be right: enough frames saw it, from enough angles, close
enough. Held-out error says whether it *is*: gsplat never trains on every 8th registered
frame (`training.TEST_EVERY`), so rendering the splat from those frames and comparing it
with them measures the reconstruction against photographs it was not fitted to. Per
gaussian, `holdout_error.py` (on the GPU, with the trainer's interpreter) shares each
frame's per-pixel error out among the gaussians that painted the pixel, by blending
weight; `holdout_maths.py` has the arithmetic and says what attribution cannot do.

Three halves, one module, because they share the file layout and nothing else does:

* `measure` -- called by the `train` stage after `trained.ply` is written. Runs the
  script, keeps its arrays aligned with `trained.ply` through the stage's own ROI or
  support-mask crop, and writes `holdout/`. **It never fails the stage**: a script that
  errors, times out, or writes something malformed leaves `holdout/holdout.json` saying
  `failed` and why, a warning in the log, and the trained splat exactly as it was.
* `load` -- the `quality` stage's reader. Anything it cannot trust (missing, failed, the
  wrong length) comes back as None and a reason; quality then judges by coverage alone.
* `judge`, `tier_report`, `accuracy_tip` -- the rule: a gaussian that coverage puts in
  *keep* stays there only if its held-out error is under the limit wherever it has
  held-out evidence. One with no such evidence -- no held-out frame saw it -- is judged
  by coverage alone, and quality.json counts how many of each there are.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import holdout_maths
import splat_io
from artifacts import ArtifactDecl
from contracts import StageContext

__all__ = [
    "HOLDOUT",
    "SCRIPT",
    "Accuracy",
    "HeldOut",
    "HeldOutColumns",
    "TierStats",
    "accuracy_tip",
    "judge",
    "limit_of",
    "load",
    "measure",
    "open_columns",
    "report",
    "tier_report",
]

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
Bools = npt.NDArray[np.bool_]

HOLDOUT = ArtifactDecl(
    "holdout",
    kind="dir",
    content_type="inode/directory",
    summary="per-gaussian error on gsplat's held-out frames, in trained.ply's row order, "
    "and holdout.json (frames used, per-frame PSNR); status says when it was not measured",
    required_members=(holdout_maths.SUMMARY_FILE,),
    stub_members=(holdout_maths.SUMMARY_FILE,),
)

#: The GPU half, run with the trainer's interpreter. Beside this file in the image too
#: (`infra/modal/app.py` copies `tools/pipeline` whole).
SCRIPT = Path(__file__).resolve().parent / "holdout_error.py"

#: Seconds the script may take before its own watchdog stops it. A held-out pass is a
#: dozen renders and backward passes -- seconds on an L4 -- so this is a hang, not a slow
#: run; it bounds how long a stuck script can hold a finished training run hostage.
DEFAULT_BUDGET_S = 900.0

#: Held-out frames whose luminance differs from the render's by this much on average
#: (either way; 0..1 scale), or whose biases spread this much, read as an exposure change.
EXPOSURE_BIAS = 0.05
EXPOSURE_SPREAD = 0.03
#: A held-out frame this much less sharp than the render was motion-blurred; a render
#: this much less sharp than its frames (median) was trained on blurred frames.
BLURRED_FRAME = 0.6
BLURRED_SPLAT = 1.4
#: The share of coverage-keep gaussians near the subject that must fail accuracy before
#: the capture is told why. The same 15% the coverage tips use.
TIP_SHARE = 0.15


# ---------------------------------------------------------------------------------------
# train: measure
# ---------------------------------------------------------------------------------------


def measure(
    ctx: StageContext,
    *,
    enabled: bool,
    python: str,
    trainer: Path,
    dataset: Path,
    ply: Path,
    rows: int,
    keep: Bools | None,
    data_factor: int = 1,
    test_every: int = 8,
    antialiased: bool = False,
    script: Path | None = None,
    budget_s: float = DEFAULT_BUDGET_S,
    sh_degree: int = 0,
) -> dict[str, Any]:
    """Write `holdout/` for the `train` stage, and return its `holdout.json`.

    `ply` is the trainer's own export (`rows` gaussians); `keep` is the stage's crop of it
    into `trained.ply` (None for no crop). The arrays written are `keep`'s rows, in order.
    `sh_degree` is the degree `trained.ply` ships: the script renders the held-out frames
    at it (`--sh-degree`), so the error `quality` gates with is the error of what is
    published, whatever colour detail that is.
    """
    out = ctx.output(HOLDOUT.name)
    if not enabled:
        summary: dict[str, Any] = {"status": "off", "reason": "holdout_error is false"}
        holdout_maths.write_summary(out, summary)
        return summary
    scratch = ctx.work_dir / "holdout"
    if scratch.exists():
        for member in scratch.iterdir():
            member.unlink()
    argv = [
        python,
        str(script or SCRIPT),
        "--data_dir",
        str(dataset),
        "--ply",
        str(ply),
        "--out",
        str(scratch),
        "--trainer",
        str(trainer),
        "--data_factor",
        str(data_factor),
        "--test_every",
        str(test_every),
        "--budget-s",
        f"{budget_s:g}",
        "--sh-degree",
        str(sh_degree),
    ]
    if antialiased:
        argv.append("--antialiased")
    try:
        ctx.run(argv)
        summary = _read_summary(scratch)
        error, weight, views = _read_arrays(scratch, rows)
        if keep is not None:
            if keep.shape != (rows,):
                raise ValueError(f"the crop has {keep.shape[0]} rows for {rows} gaussians")
            error, weight = error[keep], weight[keep]
            views = None if views is None else views[keep]
        summary = {
            **summary,
            "status": "ok",
            "gaussiansExported": rows,
            "gaussians": int(error.shape[0]),
            "cropped": keep is not None,
        }
        holdout_maths.write(out, error, weight, views, summary)
        return summary
    # Anything at all: nothing in here may cost the stage the splat it already trained.
    except Exception as problem:
        reason = f"{type(problem).__name__}: {problem}"[:600]
        ctx.log(
            f"WARNING: held-out error was not measured ({reason}). The trained splat is "
            f"unaffected; quality will judge it by coverage alone"
        )
        for name in (holdout_maths.ERROR_FILE, holdout_maths.WEIGHT_FILE, holdout_maths.VIEWS_FILE):
            (out / name).unlink(missing_ok=True)
        summary = {"status": "failed", "reason": reason}
        holdout_maths.write_summary(out, summary)
        return summary


def _read_summary(directory: Path) -> dict[str, Any]:
    path = directory / holdout_maths.SUMMARY_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return document


def _read_arrays(directory: Path, rows: int) -> tuple[F32, F32, npt.NDArray[np.uint16] | None]:
    error = np.load(directory / holdout_maths.ERROR_FILE, allow_pickle=False)
    weight = np.load(directory / holdout_maths.WEIGHT_FILE, allow_pickle=False)
    views_path = directory / holdout_maths.VIEWS_FILE
    views = np.load(views_path, allow_pickle=False) if views_path.is_file() else None
    for name, array in (("error", error), ("weight", weight), ("views", views)):
        if array is not None and array.shape != (rows,):
            raise ValueError(
                f"held-out {name} has shape {array.shape}; the splat has {rows} gaussians"
            )
    return (
        error.astype(np.float32),
        weight.astype(np.float32),
        None if views is None else views.astype(np.uint16),
    )


# ---------------------------------------------------------------------------------------
# quality: load, judge, report
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOut:
    """`holdout/`, read and checked: per gaussian of `trained.ply`, and the summary."""

    error: F32
    weight: F32
    views: npt.NDArray[np.uint16] | None
    summary: Mapping[str, Any]


def load(directory: Path, count: int) -> tuple[HeldOut | None, dict[str, Any]]:
    """The artifact for a `count`-gaussian splat, or None and what was wrong with it."""
    try:
        summary = _read_summary(directory)
    except (OSError, ValueError) as problem:
        return None, {"status": "unreadable", "reason": f"{type(problem).__name__}: {problem}"}
    status = str(summary.get("status", "unknown"))
    if status != "ok":
        return None, {"status": status, "reason": summary.get("reason")}
    try:
        error, weight, views = _read_arrays(directory, count)
    except (OSError, ValueError) as problem:
        return None, {"status": "mismatch", "reason": f"{type(problem).__name__}: {problem}"}
    return HeldOut(error=error, weight=weight, views=views, summary=summary), {"status": "ok"}


@dataclass(frozen=True)
class HeldOutColumns:
    """`holdout/` checked as `load` checks it, but read by row range: the quality stage's
    reader, which never holds a per-gaussian array whole (10 bytes a gaussian here)."""

    error: splat_io.Column
    weight: splat_io.Column
    summary: Mapping[str, Any]

    def read(self, start: int, stop: int) -> tuple[F32, F32]:
        """Rows `[start, stop)` of the error and the weight, as `load` types them."""
        return (
            self.error.read(start, stop).astype(np.float32),
            self.weight.read(start, stop).astype(np.float32),
        )


def open_columns(directory: Path, count: int) -> tuple[HeldOutColumns | None, dict[str, Any]]:
    """`load`, without loading: the same statuses and reasons, and columns to read."""
    try:
        summary = _read_summary(directory)
    except (OSError, ValueError) as problem:
        return None, {"status": "unreadable", "reason": f"{type(problem).__name__}: {problem}"}
    status = str(summary.get("status", "unknown"))
    if status != "ok":
        return None, {"status": status, "reason": summary.get("reason")}
    try:
        names = (
            ("error", holdout_maths.ERROR_FILE),
            ("weight", holdout_maths.WEIGHT_FILE),
            ("views", holdout_maths.VIEWS_FILE),
        )
        columns: dict[str, splat_io.Column] = {}
        for name, filename in names:
            path = directory / filename
            if name == "views" and not path.is_file():
                continue
            shape, _, _ = splat_io.npy_header(path)
            if shape != (count,):
                raise ValueError(
                    f"held-out {name} has shape {shape}; the splat has {count} gaussians"
                )
            columns[name] = splat_io.npy_column(path)
    except (OSError, ValueError) as problem:
        return None, {"status": "mismatch", "reason": f"{type(problem).__name__}: {problem}"}
    return (
        HeldOutColumns(error=columns["error"], weight=columns["weight"], summary=summary),
        {"status": "ok"},
    )


@dataclass(frozen=True)
class Accuracy:
    """The verdict: tiers after accuracy, and who was measured, verified, or demoted."""

    tiers: U8
    #: Enough held-out evidence to be judged (`min_weight` pixels of it).
    measured: Bools
    #: In keep after the verdict *and* measured: keep that held-out frames confirmed.
    verified: Bools
    #: In keep by coverage, measured, and over the limit.
    demoted: Bools
    #: The scene's typical error the relative limit is a multiple of, and the limit.
    reference: float | None
    limit: float | None


def limit_of(
    reference: float | None, *, max_ratio: float | None, max_abs: float | None
) -> float | None:
    """The error a keep gaussian may have: the lower of `max_ratio` times the scene's
    `reference` and `max_abs`, whichever are set; None when neither applies."""
    limits: list[float] = []
    if max_ratio is not None and reference is not None:
        limits.append(max_ratio * reference)
    if max_abs is not None:
        limits.append(max_abs)
    return min(limits) if limits else None


def judge(
    tiers: U8,
    held: HeldOut,
    *,
    max_ratio: float | None,
    max_abs: float | None,
    min_weight: float,
    keep: int,
    demote_to: int,
) -> Accuracy:
    """Keep only what held-out frames confirm, where they saw it; the rest by coverage.

    The limit is the lower of `max_ratio` times the median error of the measured
    coverage-keep gaussians (the well-covered part of this scene: what "as good as this
    capture gets" means here) and `max_abs`. A demoted gaussian goes to `demote_to` (the
    context tier): its coverage stands, and `balanced` still shows it faded.
    """
    error = np.asarray(held.error, dtype=np.float32)
    measured = np.isfinite(error) & (np.asarray(held.weight) >= min_weight)
    in_keep = tiers == keep
    population = error[measured & in_keep]
    if population.size == 0:
        population = error[measured]
    reference = float(np.median(population)) if population.size else None
    limit = limit_of(reference, max_ratio=max_ratio, max_abs=max_abs)
    demoted = (
        in_keep & measured & (error > np.float32(limit))
        if limit is not None
        else np.zeros(tiers.shape, dtype=bool)
    )
    after = tiers.copy()
    after[demoted] = np.uint8(demote_to)
    return Accuracy(
        tiers=after,
        measured=measured,
        verified=(after == keep) & measured,
        demoted=demoted,
        reference=reference,
        limit=limit,
    )


@dataclass(frozen=True)
class TierStats:
    """One tier's held-out evidence: its size, how much of it was measured, the median
    error of the measured, and the error-times-weight and weight sums of the mean."""

    gaussians: int
    measured: int
    median_error: float | None
    weighted_error: float
    weight: float


def tier_report(
    accuracy: Accuracy,
    held: HeldOut,
    names: Mapping[int, str],
    *,
    max_ratio: float | None,
    max_abs: float | None,
    min_weight: float,
) -> dict[str, Any]:
    """`quality.json`'s `heldOut`: per tier, how much was measured and how wrong it was."""
    error = held.error
    stats: dict[str, TierStats] = {}
    for tier, name in names.items():
        members = accuracy.tiers == tier
        measured = members & accuracy.measured
        values = error[measured]
        weights = held.weight[measured].astype(np.float64)
        stats[name] = TierStats(
            gaussians=int(members.sum()),
            measured=int(measured.sum()),
            median_error=float(np.median(values)) if values.size else None,
            weighted_error=float((values * weights).sum()),
            weight=float(weights.sum()),
        )
    keep_tier = next(t for t, n in names.items() if n == "keep")
    return report(
        stats,
        summary=held.summary,
        reference=accuracy.reference,
        limit=accuracy.limit,
        verified=int(accuracy.verified.sum()),
        kept=int((accuracy.tiers == keep_tier).sum()),
        demoted=int(accuracy.demoted.sum()),
        max_ratio=max_ratio,
        max_abs=max_abs,
        min_weight=min_weight,
    )


def report(
    stats: Mapping[str, TierStats],
    *,
    summary: Mapping[str, Any],
    reference: float | None,
    limit: float | None,
    verified: int,
    kept: int,
    demoted: int,
    max_ratio: float | None,
    max_abs: float | None,
    min_weight: float,
) -> dict[str, Any]:
    """`tier_report`'s document from its statistics, however they were gathered -- the
    quality stage gathers them a chunk at a time."""
    tiers: dict[str, Any] = {
        name: {
            "gaussians": tier.gaussians,
            "measured": tier.measured,
            "medianError": _round(tier.median_error, 5),
            # Evidence-weighted: a gaussian that painted more held-out pixels counts more.
            "meanError": _round(tier.weighted_error / tier.weight if tier.weight > 0 else None, 5),
        }
        for name, tier in stats.items()
    }
    per_view = summary.get("perView")
    views: list[dict[str, Any]] = []
    if isinstance(per_view, list):
        for entry in per_view:
            if isinstance(entry, Mapping):
                views.append({key: entry.get(key) for key in ("name", "psnr", "bias", "sharpness")})
    criterion_parts: list[str] = []
    if max_ratio is not None:
        criterion_parts.append(f"{max_ratio:g}x the median of measured coverage-keep")
    if max_abs is not None:
        criterion_parts.append(f"{max_abs:g}")
    return {
        "status": "ok",
        "views": summary.get("views"),
        "meanPsnr": summary.get("meanPsnr"),
        "meanPsnrFullSh": summary.get("meanPsnrFullSh"),
        "shDegree": summary.get("shDegree"),
        "rendered": summary.get("rendered"),
        "seconds": summary.get("seconds"),
        "perView": views,
        "referenceError": _round(reference, 5),
        "limit": _round(limit, 5),
        "criterion": (
            "keep needs a mean held-out error (0.8 L1 + 0.2 (1 - SSIM), per pixel, shared "
            f"by blending weight) at most the lower of {' and '.join(criterion_parts)}, "
            f"wherever it has at least {min_weight:g} px of held-out evidence; keep with "
            "less is judged by coverage alone"
            if criterion_parts
            else "no limit set: measured and reported, not applied"
        ),
        "tiers": tiers,
        "keepVerified": verified,
        "keepGeometryOnly": kept - verified,
        "demotedFromKeep": demoted,
        # Of the gaussians kept, the share whose accuracy held-out frames confirmed.
        "keepVerifiedShare": _round(100.0 * verified / kept if kept else None, 1),
    }


def accuracy_tip(summary: Mapping[str, Any], demoted_share: float) -> dict[str, str] | None:
    """Advice when accuracy, not coverage, is what failed, from the held-out frames' own
    numbers: an exposure change shows as brightness bias, motion blur as sharpness."""
    if demoted_share < TIP_SHARE:
        return None
    views = [v for v in summary.get("perView") or [] if isinstance(v, Mapping)]
    biases = [float(v["bias"]) for v in views if _finite(v.get("bias"))]
    sharpness = [float(v["sharpness"]) for v in views if _finite(v.get("sharpness"))]
    percent = round(100 * demoted_share)
    lead = (
        f"About {percent}% of what was well covered did not match the frames held back to check it"
    )
    if biases and (
        max(abs(b) for b in biases) >= EXPOSURE_BIAS or float(np.std(biases)) >= EXPOSURE_SPREAD
    ):
        return {
            "id": "accuracy-exposure",
            "text": (
                f"{lead}: the light changed between frames. Lock exposure (press and hold "
                "on the subject) and keep to one kind of light -- not in and out of shade."
            ),
        }
    if sharpness and (
        min(sharpness) <= BLURRED_FRAME or float(np.median(sharpness)) >= BLURRED_SPLAT
    ):
        return {
            "id": "accuracy-blur",
            "text": (
                f"{lead}: frames were blurred. Move more slowly, or stop for each photo, "
                "and keep the phone steady."
            ),
        }
    return {
        "id": "accuracy",
        "text": (
            f"{lead}. Move more slowly, keep the light steady, and avoid things that move "
            "(people, leaves, screens) while you capture."
        ),
    }


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)
