"""Parent gaussians for the level-of-detail octree, by moment matching.

A parent tile of the REPLACE octree (`splat_tiles.build_tiles`) holds one gaussian per
occupied grid cell at its level, standing in for every original gaussian (leaf) in the
cell. This module turns leaves into additive *moments*, merges them per cell, and turns
merged moments back into gaussians. Pure numpy; it does not import `splat_tiles` (the
dependency runs the other way). Written in October 2026 from published mathematics alone
(docs/DECISIONS/0012-licence-clean-room.md).

**Gaussians.** Centre mu, unit quaternion q = (w, x, y, z), per-axis standard deviations
sigma, opacity alpha = 1 / (1 + e^-l) from its logit l, DC colour c and optional higher SH
coefficients h. The covariance is Sigma = R(q) diag(sigma^2) R(q)^T, the standard 3DGS
parameterisation (Kerbl et al. 2023, arXiv 2308.04079, eq. 6); R's columns are the axes.

**Weight.** w = max(alpha * S(sigma_hat), 1e-30), sigma_hat the scales clipped to what SPZ
stores, S the surface area of the 1-sigma ellipsoid by Knud Thomsen's approximation,
S(a, b, c) = 4 pi [((ab)^p + (ac)^p + (bc)^p) / 3]^(1/p), p = 1.6075 (relative error at
most 1.061%, exact for spheres, homogeneous of degree 2). By Cauchy's surface-area formula
a convex body's orthogonal projection, averaged over all view directions, has area S / 4,
so alpha * S is proportional to the ink a gaussian leaves on screen averaged over
viewpoints -- what a parent must preserve. PlayCanvas splat-transform 3.8.0 (MIT) weighs
by the same opacity x Thomsen area. Clipping first means a parent merges exactly the sizes
its children are drawn at.

**Merge.** Per cell, with W = sum w_i: mu = sum w_i mu_i / W; Sigma = sum w_i (Sigma_i +
d_i d_i^T) / W with d_i = mu_i - mu, the law of total covariance, computed against the
cell's own mean in two passes (never E[xx^T] - mu mu^T, which cancels catastrophically
for a small cell far from the origin); colour and SH are the weighted means (SH is linear
in its coefficients, so the merged colour from any direction is the weighted mean of the
members' from that direction); the weight stays W, unnormalised, so cells merge again.
Merging fine cells and then coarser ones equals merging the leaves at the coarse level
directly: the pairwise combination identity for weighted means and covariances (Chan,
Golub and LeVeque, Am. Statistician 37(3), 1983).

**Back to a gaussian.** The eigen-decomposition of Sigma gives the axes (made a proper
rotation) and sigma_k = sqrt(max(lambda_k, e^(2 lo), 2^-20 lambda_max)): floored at the
smallest scale SPZ stores, and at 1/1024 of the longest axis (`MERGED_EIGEN_FLOOR`, which says
why). F = W / S(sigma) is the opacity that would deposit exactly the members' ink. If F
is at most `MERGED_OPACITY_MAX` the scales stay and alpha = F; otherwise every axis widens
by k = clip(sqrt(F / alpha_max), 1, `MERGED_WIDEN_MAX`), so S grows by k^2 and alpha =
F / k^2 conserves alpha * S = W until k reaches its cap (the widening idea Spark, MIT,
applies with `lodInflate`; splat-transform caps alpha at 1 without widening, which leaves
see-through parents). alpha is clipped to [`MERGED_OPACITY_MIN`, `MERGED_OPACITY_MAX`].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The largest merged opacity: 511/512, whose logit is finite and small (~6.24) and whose
#: SPZ alpha byte, round(255 alpha), is still 255.
MERGED_OPACITY_MAX = 511.0 / 512.0
#: The most a merged gaussian widens to keep its members' ink. Past twice its moment-matched
#: size a parent spreads beyond its cell's neighbourhood; F > 4 alpha_max means about four
#: or more stacked layers, whose extra mass is not recoverable as coverage.
MERGED_WIDEN_MAX = 2.0
#: A numeric guard only.
MERGED_OPACITY_MIN = 1e-6
#: The smallest variance a merged axis keeps, relative to its largest: 2^-20, an axis 1/1024 of
#: the longest. A cell whose members lie on a line or in a plane (rank-1 or rank-2 scatter)
#: has eigenvalues down at e^(2 lo) under ones up to SPZ's top, a 10^14 spread, and nothing
#: downstream can use it. A viewer rebuilds R diag(s^2) R^T in float32 (unit roundoff
#: u = 2^-24), where every entry carries a few u of the largest variance, so a smaller
#: eigenvalue is lost and can come back negative: a gaussian that is not one. Measured over
#: 200,000 random rotations of a needle and of a disc: at 2^-20 the rebuilt smallest
#: eigenvalue was within 11% and always positive; at 2^-24 0.1% came back negative; at 1e-10
#: half did. Nor would SPZ show the thinner axis: its rotation bytes are good to 1/255 per
#: component, a tilt of up to ~1/128 rad, which already blurs an axis thinner than 1/128 of the
#: longest; 1/1024 sits 8x below that. The float64 merge's own rounding (~n 2^-53 of the largest
#: variance for n members) is far below both.
MERGED_EIGEN_FLOOR = 2.0**-20
#: Thomsen's exponent for the ellipsoid's surface area.
THOMSEN_P = 1.6075
#: Weights are floored here so a cell of invisible members still divides.
WEIGHT_MIN = 1e-30

MOMENT_FIELDS = ("key", "weight", "mean", "cov", "colour", "sh")


@dataclass
class Moments:
    """Additive moments, one row per gaussian or merged cell: `key` (int64 cell id),
    `weight` (W), `mean` (n, 3), `cov` (n, 6: xx, xy, xz, yy, yz, zz), `colour` (n, 3 DC
    coefficients) and `sh` (n, 3K: higher bands coefficient-major, the colour channel
    innermost; (n, 0) without any). float64 throughout."""

    key: np.ndarray
    weight: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    colour: np.ndarray
    sh: np.ndarray

    def __len__(self) -> int:
        return int(self.key.shape[0])

    @staticmethod
    def concat(parts: list[Moments]) -> Moments:
        """The rows of `parts`, one after another, field by field."""
        if not parts:
            raise ValueError("Moments.concat needs at least one part")
        stacked = [
            np.concatenate([getattr(part, field) for part in parts]) for field in MOMENT_FIELDS
        ]
        return Moments(*stacked)

    def take(self, index: np.ndarray) -> Moments:
        """The rows `index` selects (integer positions or a boolean mask), field by field."""
        return Moments(*(getattr(self, field)[index] for field in MOMENT_FIELDS))


@dataclass
class MergedSplats:
    """Merged cells as gaussians, in `pack_spz`'s terms (float32)."""

    xyz: np.ndarray
    sh0: np.ndarray
    opacity_logit: np.ndarray
    log_scales: np.ndarray
    quat_xyzw: np.ndarray
    #: (n, K, 3) coefficient-major; None without higher bands.
    sh_rest: np.ndarray | None


def thomsen_area(scales: np.ndarray) -> np.ndarray:
    """Surface area of the ellipsoid with semi-axes `scales` (n, 3), Thomsen's formula."""
    s = np.asarray(scales, dtype=np.float64)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    p = THOMSEN_P
    mean = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
    out: np.ndarray = 4.0 * np.pi * mean ** (1.0 / p)
    return out


def rotation_matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotations of quaternions (w, x, y, z), normalised (norm floored at 1e-12);
    the columns are the gaussian's axes."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - w * z)
    r[:, 0, 2] = 2.0 * (x * z + w * y)
    r[:, 1, 0] = 2.0 * (x * y + w * z)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - w * x)
    r[:, 2, 0] = 2.0 * (x * z - w * y)
    r[:, 2, 1] = 2.0 * (y * z + w * x)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def quaternions(rotations: np.ndarray) -> np.ndarray:
    """(n, 4) unit quaternions (w, x, y, z), w >= 0, of proper rotations (n, 3, 3).

    Shepperd's method (J. Guidance and Control 1(3), 1978): of 4w^2 - 1 = trace and
    4x^2 - 1 = 2 R11 - trace (and likewise y, z), the largest is taken as the square root,
    and the other three components follow from the off-diagonal sums and differences, so no
    square root is ever taken of a small number."""
    r = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    trace = m00 + m11 + m22
    pick = np.argmax(np.stack([trace, m00, m11, m22], axis=1), axis=1)
    q = np.empty((r.shape[0], 4), dtype=np.float64)
    # Each case: s = 4 * (the chosen component), from 1 + the matching combination.
    for case in range(4):
        rows = pick == case
        if not rows.any():
            continue
        a = r[rows]
        if case == 0:
            s = 2.0 * np.sqrt(np.maximum(1.0 + trace[rows], 0.0))
            q[rows, 0] = 0.25 * s
            q[rows, 1] = (a[:, 2, 1] - a[:, 1, 2]) / s
            q[rows, 2] = (a[:, 0, 2] - a[:, 2, 0]) / s
            q[rows, 3] = (a[:, 1, 0] - a[:, 0, 1]) / s
        elif case == 1:
            s = 2.0 * np.sqrt(np.maximum(1.0 + a[:, 0, 0] - a[:, 1, 1] - a[:, 2, 2], 0.0))
            q[rows, 0] = (a[:, 2, 1] - a[:, 1, 2]) / s
            q[rows, 1] = 0.25 * s
            q[rows, 2] = (a[:, 0, 1] + a[:, 1, 0]) / s
            q[rows, 3] = (a[:, 0, 2] + a[:, 2, 0]) / s
        elif case == 2:
            s = 2.0 * np.sqrt(np.maximum(1.0 - a[:, 0, 0] + a[:, 1, 1] - a[:, 2, 2], 0.0))
            q[rows, 0] = (a[:, 0, 2] - a[:, 2, 0]) / s
            q[rows, 1] = (a[:, 0, 1] + a[:, 1, 0]) / s
            q[rows, 2] = 0.25 * s
            q[rows, 3] = (a[:, 1, 2] + a[:, 2, 1]) / s
        else:
            s = 2.0 * np.sqrt(np.maximum(1.0 - a[:, 0, 0] - a[:, 1, 1] + a[:, 2, 2], 0.0))
            q[rows, 0] = (a[:, 1, 0] - a[:, 0, 1]) / s
            q[rows, 1] = (a[:, 0, 2] + a[:, 2, 0]) / s
            q[rows, 2] = (a[:, 1, 2] + a[:, 2, 1]) / s
            q[rows, 3] = 0.25 * s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q


def leaf_weight(
    opacity_logit: np.ndarray, log_scales: np.ndarray, *, log_scale_range: tuple[float, float]
) -> np.ndarray:
    """w = max(alpha * S(exp(clip(log sigma, lo, hi))), 1e-30) per gaussian."""
    logit = np.asarray(opacity_logit, dtype=np.float64)
    with np.errstate(over="ignore"):
        alpha = 1.0 / (1.0 + np.exp(-logit))
    scales = np.exp(np.clip(np.asarray(log_scales, dtype=np.float64), *log_scale_range))
    out: np.ndarray = np.maximum(alpha * thomsen_area(scales), WEIGHT_MIN)
    return out


def leaf_moments(
    keys: np.ndarray,
    xyz: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    quat_wxyz: np.ndarray,
    f_dc: np.ndarray,
    sh_flat: np.ndarray,
    *,
    log_scale_range: tuple[float, float],
) -> Moments:
    """One row of moments per gaussian, in input order."""
    n = int(np.asarray(keys).shape[0])
    sigma = np.exp(
        np.clip(np.asarray(log_scales, dtype=np.float64).reshape(n, 3), *log_scale_range)
    )
    r = rotation_matrices(np.asarray(quat_wxyz, dtype=np.float64).reshape(n, 4))
    var = sigma * sigma
    cov = np.empty((n, 6), dtype=np.float64)
    # Element-wise sums over the three axes, no matrix product: the same bits on every BLAS.
    for column, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        cov[:, column] = (
            r[:, i, 0] * r[:, j, 0] * var[:, 0]
            + r[:, i, 1] * r[:, j, 1] * var[:, 1]
            + r[:, i, 2] * r[:, j, 2] * var[:, 2]
        )
    return Moments(
        key=np.asarray(keys).astype(np.int64),
        weight=leaf_weight(opacity_logit, log_scales, log_scale_range=log_scale_range),
        mean=np.asarray(xyz, dtype=np.float64).reshape(n, 3),
        cov=cov,
        colour=np.asarray(f_dc, dtype=np.float64).reshape(n, 3),
        sh=_rows(np.asarray(sh_flat, dtype=np.float64), n),
    )


def _rows(values: np.ndarray, n: int) -> np.ndarray:
    """`values` as n rows (an empty input keeps its column count)."""
    array = np.asarray(values)
    if n == 0:
        # reshape(0, -1) is ambiguous, so the width comes from the input when it has one
        return array.reshape(0, array.shape[-1] if array.ndim >= 2 else 0)
    return array.reshape(n, -1)


#: Index pairs (a, b) of the six stored covariance entries, in `Moments.cov` column order.
_COV_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def merge_cells(m: Moments) -> Moments:
    """One row per distinct key (keys ascending), each its members merged. `m.key` must be
    non-decreasing (equal keys contiguous)."""
    rows = len(m)
    if rows == 0:
        return m
    # A cell is a run of equal keys; `starts` holds each run's first row and `cell` maps
    # every row to its run, so all sums below are run-wise reductions (np.add.reduceat).
    new_run = np.ones(rows, dtype=bool)
    new_run[1:] = m.key[1:] != m.key[:-1]
    starts = np.flatnonzero(new_run)
    cell = np.cumsum(new_run) - 1
    total = np.add.reduceat(m.weight, starts)
    # Each member's share of its cell's weight: w_i / W, exactly 1.0 for a lone member.
    share = (m.weight / total[cell])[:, None]

    def weighted_mean(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values * share, starts, axis=0)

    centre = weighted_mean(m.mean)
    # Law of total covariance, second pass: deviations from the cell's own centre.
    dev = m.mean - centre[cell]
    scatter = np.stack([dev[:, a] * dev[:, b] for a, b in _COV_PAIRS], axis=1)
    cov = weighted_mean(m.cov + scatter)
    colour = weighted_mean(m.colour)
    if m.sh.shape[1]:
        sh = weighted_mean(m.sh)
    else:
        sh = np.zeros((len(starts), 0), dtype=np.float64)
    merged = Moments(m.key[starts], total, centre, cov, colour, sh)
    # A cell of one member is that member, bit for bit (adding a zero scatter would turn a
    # stored -0.0 into +0.0), so lone rows are copied rather than recomputed.
    lone = np.flatnonzero(np.diff(np.append(starts, rows)) == 1)
    if len(lone):
        source = starts[lone]
        for field in MOMENT_FIELDS:
            getattr(merged, field)[lone] = getattr(m, field)[source]
    return merged


def coarsen(m: Moments, levels: int) -> Moments:
    """The cells `levels` octree levels up (each level an octant, 3 bits of key), merged."""
    if levels <= 0:
        return m
    # Dropping 3 bits per level maps each cell to its ancestor and keeps the keys sorted.
    parent_keys = m.key >> np.int64(3 * levels)
    return merge_cells(Moments(parent_keys, m.weight, m.mean, m.cov, m.colour, m.sh))


def to_splats(m: Moments, *, log_scale_range: tuple[float, float]) -> MergedSplats:
    """Each row's gaussian: axes and scales from its covariance (floored at SPZ's smallest
    scale and at `MERGED_EIGEN_FLOOR` of the largest variance), opacity conserving its weight, widened up to `MERGED_WIDEN_MAX` past
    `MERGED_OPACITY_MAX`."""
    n = len(m)
    c = m.cov
    sigma_matrix = np.empty((n, 3, 3), dtype=np.float64)
    sigma_matrix[:, 0, 0], sigma_matrix[:, 1, 1], sigma_matrix[:, 2, 2] = c[:, 0], c[:, 3], c[:, 5]
    sigma_matrix[:, 0, 1] = sigma_matrix[:, 1, 0] = c[:, 1]
    sigma_matrix[:, 0, 2] = sigma_matrix[:, 2, 0] = c[:, 2]
    sigma_matrix[:, 1, 2] = sigma_matrix[:, 2, 1] = c[:, 4]
    if n:
        values, vectors = np.linalg.eigh(sigma_matrix)
    else:
        values, vectors = np.zeros((0, 3)), np.zeros((0, 3, 3))
    flip = np.linalg.det(vectors) < 0 if n else np.zeros(0, dtype=bool)
    vectors[flip, :, 0] *= -1.0
    lo = log_scale_range[0]
    # eigh's values ascend, so the last is the largest; it is positive (every leaf's
    # covariance is at least e^(2 lo) I), but a NaN-free floor needs it clipped at 0.
    floor = np.maximum(np.exp(2.0 * lo), MERGED_EIGEN_FLOOR * np.maximum(values[:, 2:], 0.0))
    sigma = np.sqrt(np.maximum(values, floor))
    fill = m.weight / thomsen_area(sigma)
    widen = np.clip(np.sqrt(fill / MERGED_OPACITY_MAX), 1.0, MERGED_WIDEN_MAX)
    alpha = np.clip(fill / (widen * widen), MERGED_OPACITY_MIN, MERGED_OPACITY_MAX)
    quat = quaternions(vectors)
    k = m.sh.shape[1] // 3
    return MergedSplats(
        xyz=m.mean.astype(np.float32),
        sh0=m.colour.astype(np.float32),
        opacity_logit=np.log(alpha / (1.0 - alpha)).astype(np.float32),
        log_scales=np.log(widen[:, None] * sigma).astype(np.float32),
        quat_xyzw=quat[:, [1, 2, 3, 0]].astype(np.float32),
        sh_rest=m.sh.reshape(n, k, 3).astype(np.float32) if k else None,
    )
