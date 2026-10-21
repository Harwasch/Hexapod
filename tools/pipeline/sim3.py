"""A similarity between two sets of named points, by a seeded RANSAC or LMedS over 3-point
samples -- so a georeference, or an epoch's alignment, is reproducible.

`colmap model_aligner` (COLMAP 3.9.1) solves the same problem with a RANSAC whose last
digits move between runs over the same input, and has no seed to pin them. This module is
the numpy replacement: every random draw comes from `np.random.default_rng(seed)`, small
sets are enumerated rather than sampled, and ties are broken by sample order, so the same
inputs and seed give the same bytes on one machine (`test_sim3.py` holds that). Across
machines the last bits of a float may differ (BLAS, SVD); nothing byte-gates this output.

The algorithm, pinned (M3b design section 5):

1. The names common to both sides, sorted. Fewer than `MIN_POINTS` is refused.
2. Every 3-combination in index order when there are at most `iterations` of them;
   otherwise `iterations` draws of `rng.choice(n, 3, replace=False)`.
3. A sample whose three points are (nearly) collinear on either side is skipped:
   `|ab x ac| < 1e-6 * max(|ab|, |ac|)^2`. Otherwise Umeyama (1991) on the three points.
4. The score:
   * **ransac** (a `max_error_m` threshold given): the most residuals within it, then the
     smaller sum of squared inlier residuals, then the earlier sample;
   * **lmeds** (no threshold): the smaller median of the squared residuals, then the
     earlier sample. The threshold is then `LMEDS_CUTOFF` times the robust scale
     `1.4826 * (1 + 5 / (n - 3)) * sqrt(med r^2)` (Rousseeuw & Leroy 1987, eq. 5.3.6),
     floored at 1e-9 m. With exactly three points `n - 3` is zero; the correction is
     then taken at `n - 3 = 1`, its largest finite value, rather than dividing by zero.
5. Umeyama refitted on the inliers; residuals and the inlier set recomputed once with the
   same threshold, and refitted again if the set changed (never on fewer than three).
6. **ransac** with fewer than three final inliers is no fit (`NoConsensusError`): no
   similarity is fixed by fewer points than it has to agree on, and the sampled one would
   be a guess. LMedS sets its own threshold from the median and always keeps its half.

Pure numpy; nothing here imports pycolmap.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

import sfm

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_SEED",
    "LMEDS_CUTOFF",
    "MIN_POINTS",
    "NoConsensusError",
    "Sim3Fit",
    "fit_sim3",
    "residuals",
    "similarity_dict",
]

F64 = npt.NDArray[np.float64]

#: A similarity has seven degrees of freedom; three points are the fewest that fix one.
MIN_POINTS = 3
DEFAULT_ITERATIONS = 2000
DEFAULT_SEED = 0
#: Inliers are within this many robust standard deviations under LMedS (Rousseeuw & Leroy).
LMEDS_CUTOFF = 2.5
#: Rousseeuw & Leroy's consistency factor: 1 / Phi^-1(0.75).
_MAD_FACTOR = 1.4826
#: The relative area below which three points are a line, not a triangle.
_COLLINEAR = 1e-6
#: The robust scale is never taken as smaller than this (exact correspondences).
_SCALE_FLOOR_M = 1e-9


@dataclass(frozen=True)
class Sim3Fit:
    """A fitted similarity, `target ~= scale * R @ source + t`, and how well it fits."""

    similarity: sfm.Similarity
    #: Every name both sides had, sorted.
    names: tuple[str, ...]
    #: The names within `threshold_m` of the final fit, sorted.
    inliers: tuple[str, ...]
    #: Every common name's residual (metres in the target frame) under the final fit.
    residuals_m: dict[str, float]
    threshold_m: float
    method: Literal["ransac", "lmeds"]
    #: How many 3-point samples were scored (enumerated or drawn).
    iterations: int
    seed: int

    def stats(self) -> dict[str, float]:
        """RMS, median, 90th percentile and maximum of the inliers' residuals."""
        values = np.asarray([self.residuals_m[name] for name in self.inliers], dtype=np.float64)
        if values.size == 0:
            return {"rms": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "rms": float(np.sqrt(float(np.mean(values**2)))),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90.0)),
            "max": float(values.max()),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "iterations": self.iterations,
            "thresholdM": self.threshold_m,
            "correspondences": len(self.names),
            "inliers": len(self.inliers),
            "inlierNames": list(self.inliers),
            "similarity": similarity_dict(self.similarity),
            "residualM": {
                **self.stats(),
                "perName": {name: self.residuals_m[name] for name in self.names},
            },
        }


def similarity_dict(similarity: sfm.Similarity) -> dict[str, object]:
    """`{scale, rotation, translationM}`, the shape georef.json's `frame` carries."""
    return {
        "scale": float(similarity.scale),
        "rotation": [[float(v) for v in row] for row in similarity.rotation],
        "translationM": [float(v) for v in similarity.translation],
    }


class NoConsensusError(ValueError):
    """RANSAC found fewer than `MIN_POINTS` correspondences agreeing within its threshold."""


def residuals(similarity: sfm.Similarity, source: F64, target: F64) -> F64:
    """Per-row distance between `similarity(source)` and `target`."""
    moved = similarity.apply(source)
    return np.asarray(np.linalg.norm(moved - target, axis=1), dtype=np.float64)


def _degenerate(points: F64) -> bool:
    ab, ac = points[1] - points[0], points[2] - points[0]
    longest = max(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)))
    if longest == 0.0:
        return True
    return float(np.linalg.norm(np.cross(ab, ac))) < _COLLINEAR * longest**2


def _samples(n: int, iterations: int, seed: int) -> list[tuple[int, int, int]]:
    if math.comb(n, MIN_POINTS) <= iterations:
        return [(a, b, c) for a, b, c in itertools.combinations(range(n), MIN_POINTS)]
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int, int]] = []
    for _ in range(iterations):
        a, b, c = (int(v) for v in rng.choice(n, MIN_POINTS, replace=False))
        out.append((a, b, c))
    return out


def fit_sim3(
    source: Mapping[str, Sequence[float]],
    target: Mapping[str, Sequence[float]],
    *,
    max_error_m: float | None,
    seed: int = DEFAULT_SEED,
    iterations: int = DEFAULT_ITERATIONS,
) -> Sim3Fit:
    """The similarity taking `source` onto `target`, robust to outliers; see the module.

    `max_error_m` None is LMedS (no threshold needed); a number is RANSAC with that
    inlier threshold in target metres. Raises `ValueError` for fewer than three common
    names, a non-positive threshold, or when every sample is collinear, and its subclass
    `NoConsensusError` when RANSAC's best fit has fewer than three inliers.
    """
    if max_error_m is not None and not (math.isfinite(max_error_m) and max_error_m > 0):
        raise ValueError(f"max_error_m must be a positive number of metres, not {max_error_m!r}")
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, not {iterations}")
    names = tuple(sorted(set(source) & set(target)))
    if len(names) < MIN_POINTS:
        raise ValueError(
            f"a similarity needs at least {MIN_POINTS} common points; got {len(names)}"
        )
    src = np.asarray([[float(v) for v in source[name]] for name in names], dtype=np.float64)
    tgt = np.asarray([[float(v) for v in target[name]] for name in names], dtype=np.float64)
    if src.shape != (len(names), 3) or tgt.shape != (len(names), 3):
        raise ValueError("every point must have three coordinates")
    if not (np.isfinite(src).all() and np.isfinite(tgt).all()):
        raise ValueError("every coordinate must be a finite number")
    n = len(names)
    samples = _samples(n, iterations, seed)
    best: tuple[float, float] | None = None
    best_fit: sfm.Similarity | None = None
    for sample in samples:
        index = list(sample)
        if _degenerate(src[index]) or _degenerate(tgt[index]):
            continue
        candidate = sfm.umeyama(src[index], tgt[index])
        squared = residuals(candidate, src, tgt) ** 2
        if max_error_m is None:
            key = (float(np.median(squared)), 0.0)
        else:
            within = squared <= max_error_m**2
            key = (-float(within.sum()), float(squared[within].sum()))
        # Strictly better only: an equal score keeps the earlier sample.
        if best is None or key < best:
            best, best_fit = key, candidate
    if best is None or best_fit is None:
        raise ValueError(
            f"every 3-point sample of the {n} correspondences is collinear; a line does not "
            f"fix a rotation about itself"
        )
    if max_error_m is None:
        sigma = _MAD_FACTOR * (1.0 + 5.0 / max(n - MIN_POINTS, 1)) * math.sqrt(best[0])
        threshold = LMEDS_CUTOFF * max(sigma, _SCALE_FLOOR_M)
    else:
        threshold = float(max_error_m)
    keep = residuals(best_fit, src, tgt) <= threshold
    fit = best_fit
    if int(keep.sum()) >= MIN_POINTS and not _degenerate_set(src[keep]):
        fit = sfm.umeyama(src[keep], tgt[keep])
        again = residuals(fit, src, tgt) <= threshold
        if (
            not np.array_equal(again, keep)
            and int(again.sum()) >= MIN_POINTS
            and not _degenerate_set(src[again])
        ):
            keep = again
            fit = sfm.umeyama(src[keep], tgt[keep])
    final = residuals(fit, src, tgt)
    inliers = tuple(name for name, value in zip(names, final, strict=True) if value <= threshold)
    if max_error_m is not None and len(inliers) < MIN_POINTS:
        raise NoConsensusError(
            f"no consensus: {len(inliers)} of {n} points agree within {max_error_m:g} m, "
            f"and a similarity needs {MIN_POINTS}"
        )
    return Sim3Fit(
        similarity=fit,
        names=names,
        inliers=inliers,
        residuals_m={name: float(value) for name, value in zip(names, final, strict=True)},
        threshold_m=float(threshold),
        method="lmeds" if max_error_m is None else "ransac",
        iterations=len(samples),
        seed=int(seed),
    )


def _degenerate_set(points: F64) -> bool:
    """All of `points` on one line (or one point): the second singular value vanishes."""
    centred = points - points.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[0] == 0.0 or singular[1] < _COLLINEAR * singular[0])
