"""`splat_merge`: parents by moment matching, each property checked against a reference
written here from the maths (docs/DECISIONS/0012-licence-clean-room.md, Appendix)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import splat_merge as sm

RANGE = (-10.0, 255 / 16 - 10.0)  # SPZ's log-scale range, as splat_tiles passes it
P = 1.6075


def thomsen(scales: np.ndarray) -> np.ndarray:
    a, b, c = scales[:, 0], scales[:, 1], scales[:, 2]
    return 4 * np.pi * (((a * b) ** P + (a * c) ** P + (b * c) ** P) / 3) ** (1 / P)


def rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def sym(c: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = c
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def gaussians(n: int, seed: int, sh: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {
        "xyz": rng.normal(size=(n, 3)) * 3 + [1000.0, -2000.0, 50.0],
        "logit": rng.uniform(-3, 4, n),
        "log_scales": rng.normal(-3, 0.7, size=(n, 3)),
        "quat": q,
        "f_dc": rng.normal(0, 0.5, size=(n, 3)),
        "sh": rng.normal(0, 0.2, size=(n, 3 * sh)),
    }


def moments(g: dict[str, np.ndarray], keys: np.ndarray) -> sm.Moments:
    return sm.leaf_moments(
        keys,
        g["xyz"],
        g["logit"],
        g["log_scales"],
        g["quat"],
        g["f_dc"],
        g["sh"],
        log_scale_range=RANGE,
    )


def reconstructed(out: sm.MergedSplats, i: int) -> np.ndarray:
    x, y, z, w = out.quat_xyzw[i].astype(np.float64)
    r = rotation(np.array([w, x, y, z]))
    return r @ np.diag(np.exp(2 * out.log_scales[i].astype(np.float64))) @ r.T


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def test_a_single_gaussian_merges_to_itself() -> None:
    g = gaussians(1, 1)
    g["logit"][:] = 1.2  # alpha 0.77 < alpha_max
    m = sm.merge_cells(moments(g, np.zeros(1, np.int64)))
    out = sm.to_splats(m, log_scale_range=RANGE)
    assert np.array_equal(m.mean, g["xyz"])
    r = rotation(g["quat"][0])
    sigma = r @ np.diag(np.exp(2 * g["log_scales"][0])) @ r.T
    assert np.linalg.norm(reconstructed(out, 0) - sigma) <= 1e-5 * np.linalg.norm(sigma)
    assert abs(float(out.opacity_logit[0]) - 1.2) < 1e-5


def test_coincident_gaussians_widen_to_keep_their_mass() -> None:
    g = gaussians(1, 2)
    g = {k: np.repeat(v, 2, axis=0) for k, v in g.items()}
    g["logit"][:] = math.log(0.8 / 0.2)
    m = moments(g, np.zeros(2, np.int64))
    out = sm.to_splats(sm.merge_cells(m), log_scale_range=RANGE)
    alpha = float(sigmoid(out.opacity_logit[0]))
    assert alpha == pytest.approx(sm.MERGED_OPACITY_MAX, rel=1e-5)
    area = float(thomsen(np.exp(out.log_scales[:1].astype(np.float64)))[0])
    assert alpha * area == pytest.approx(2 * float(m.weight[0]), rel=1e-5)


def test_mass_beyond_the_widening_cap_is_not_smeared() -> None:
    g = gaussians(1, 3)
    g = {k: np.repeat(v, 8, axis=0) for k, v in g.items()}
    g["logit"][:] = math.log(0.9 / 0.1)
    out = sm.to_splats(sm.merge_cells(moments(g, np.zeros(8, np.int64))), log_scale_range=RANGE)
    sigma = np.sort(np.exp(g["log_scales"][0]))
    np.testing.assert_allclose(
        np.sort(np.exp(out.log_scales[0].astype(np.float64))), 2 * sigma, rtol=1e-6
    )
    assert float(sigmoid(out.opacity_logit[0])) == pytest.approx(sm.MERGED_OPACITY_MAX, rel=1e-6)


def test_the_moments_are_the_law_of_total_covariance() -> None:
    g = gaussians(50, 4, sh=9)
    m = sm.merge_cells(moments(g, np.zeros(50, np.int64)))
    w = sigmoid(g["logit"]) * thomsen(np.exp(np.clip(g["log_scales"], *RANGE)))
    total = w.sum()
    mean = sum(w[i] * g["xyz"][i] for i in range(50)) / total
    cov = np.zeros((3, 3))
    for i in range(50):
        r = rotation(g["quat"][i])
        d = g["xyz"][i] - mean
        cov += w[i] * (r @ np.diag(np.exp(2 * g["log_scales"][i])) @ r.T + np.outer(d, d))
    cov /= total
    assert m.weight[0] == pytest.approx(total, rel=1e-12)
    np.testing.assert_allclose(m.mean[0], mean, rtol=1e-12)
    np.testing.assert_allclose(sym(m.cov[0]), cov, rtol=1e-12, atol=1e-15)


def test_merging_merged_cells_is_merging_their_gaussians() -> None:
    n = 5000
    g = gaussians(n, 5, sh=3)
    rng = np.random.default_rng(6)
    fine = np.sort(rng.integers(0, 1 << 12, n)).astype(np.int64)
    leaves = moments(g, fine)
    two_step = sm.coarsen(sm.merge_cells(leaves), 2)
    direct = sm.merge_cells(moments(g, fine >> 6))
    assert np.array_equal(two_step.key, direct.key)
    np.testing.assert_allclose(two_step.weight, direct.weight, rtol=1e-12)
    for name in ("mean", "cov", "colour", "sh"):
        np.testing.assert_allclose(
            getattr(two_step, name), getattr(direct, name), rtol=1e-9, atol=1e-12
        )
    assert sm.coarsen(leaves, 0) is leaves


def test_the_weight_is_opacity_times_thomsen_area() -> None:
    # A sphere: exact.
    w = sm.leaf_weight(np.array([0.0]), np.log(np.full((1, 3), 0.3)), log_scale_range=RANGE)
    assert w[0] == pytest.approx(0.5 * 4 * np.pi * 0.09, rel=1e-12)
    # A 3:2:1 ellipsoid: within Thomsen's 1.061% of the area, here by quadrature.
    a, b, c = 3.0, 2.0, 1.0
    theta = (np.arange(2000) + 0.5) * np.pi / 2000
    phi = (np.arange(4000) + 0.5) * 2 * np.pi / 4000
    t, f = np.meshgrid(theta, phi, indexing="ij")
    st, ct, sf, cf = np.sin(t), np.cos(t), np.sin(f), np.cos(f)
    # |dr/dtheta x dr/dphi| of r = (a sin t cos f, b sin t sin f, c cos t)
    n1 = b * c * st**2 * cf
    n2 = a * c * st**2 * sf
    n3 = a * b * st * ct
    exact = float(np.sqrt(n1**2 + n2**2 + n3**2).sum() * (np.pi / 2000) * (2 * np.pi / 4000))
    got = sm.leaf_weight(np.array([50.0]), np.log([[a, b, c]]), log_scale_range=RANGE)[0]
    assert abs(got / exact - 1) < 0.011
    # Scales outside SPZ's range are clipped first.
    huge = sm.leaf_weight(np.array([0.0]), np.full((1, 3), 40.0), log_scale_range=RANGE)
    top = math.exp(RANGE[1])
    assert huge[0] == pytest.approx(0.5 * 4 * np.pi * top * top, rel=1e-12)


def test_quaternions_round_trip() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(2000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    # Near pi about each axis: w within 1e-6 of 0.
    for axis in range(3):
        near = np.zeros((3, 4))
        near[:, 0] = [1e-6, 3e-7, 0.0]
        near[:, axis + 1] = 1.0
        q = np.vstack([q, near / np.linalg.norm(near, axis=1, keepdims=True)])
    r = sm.rotation_matrices(q)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
    back = sm.quaternions(r)
    error = np.minimum(np.abs(back - q).max(axis=1), np.abs(back + q).max(axis=1))
    assert error.max() < 1e-12
    assert (back[:, 0] >= 0).all()


def test_the_axes_reproduce_the_covariance() -> None:
    rng = np.random.default_rng(8)
    n = 500
    a = rng.normal(size=(n, 3, 3)) * np.exp(rng.uniform(-4, 1, size=(n, 1, 1)))
    cov = a @ a.transpose(0, 2, 1) + 1e-6 * np.eye(3)
    six = np.stack(
        [cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1
    )
    m = sm.Moments(
        key=np.arange(n, dtype=np.int64),
        weight=np.full(n, 1e-9),  # faint: no widening
        mean=np.zeros((n, 3)),
        cov=six,
        colour=np.zeros((n, 3)),
        sh=np.zeros((n, 0)),
    )
    out = sm.to_splats(m, log_scale_range=RANGE)
    for i in range(n):
        assert (
            np.linalg.norm(reconstructed(out, i) - cov[i]) <= 1e-5 * np.linalg.norm(cov[i]) + 1e-12
        )


FLOOR = 2.0**-20  # the smallest variance an axis keeps, relative to the largest


def _floor_scale(m: sm.Moments) -> float:
    """The log-scale the floor gives a degenerate axis: max(lo, log of 2^-10 the longest)."""
    largest = float(np.linalg.eigvalsh(sym(m.cov[0]))[-1])
    return max(RANGE[0], 0.5 * math.log(FLOOR * largest))


def _float32_rebuild(out: sm.MergedSplats) -> np.ndarray:
    """The covariance as a viewer rebuilds it: in float32, element by element."""
    x, y, z, w = out.quat_xyzw[0].astype(np.float64)
    r = rotation(np.array([w, x, y, z])).astype(np.float32)
    var = np.exp(2 * out.log_scales[0].astype(np.float64)).astype(np.float32)
    rebuilt = np.zeros((3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            rebuilt[i, j] = sum(r[i, k] * r[j, k] * var[k] for k in range(3))
    return rebuilt.astype(np.float64)


def test_a_flat_cell_is_floored_relative_to_its_longest_axis() -> None:
    """Rank-2 scatter: coplanar centres of gaussians flat in that plane, 3 m across. The thin
    axis would be SPZ's smallest (e^lo, 45 um) under a 3 m one; it is kept at 1/1024 of the
    longest instead, and a float32 rebuild of the result is still a gaussian."""
    g = gaussians(40, 9)
    g["xyz"][:, 2] = 0.0
    g["log_scales"][:, 2] = -30.0
    g["quat"][:] = [1.0, 0.0, 0.0, 0.0]
    g["logit"][:] = -3.0  # faint, so nothing widens
    m = sm.merge_cells(moments(g, np.zeros(40, np.int64)))
    assert float(np.linalg.eigvalsh(sym(m.cov[0]))[0]) < 1e-8  # degenerate before the floor
    out = sm.to_splats(m, log_scale_range=RANGE)
    assert float(sigmoid(out.opacity_logit[0])) < sm.MERGED_OPACITY_MAX
    want = _floor_scale(m)
    assert want > RANGE[0] + 1  # the relative floor is the one that binds here
    assert float(out.log_scales[0].min()) == pytest.approx(want, abs=1e-5)
    # The thin axis is the plane's normal, z.
    x, y, z, w = out.quat_xyzw[0].astype(np.float64)
    axes = rotation(np.array([w, x, y, z]))
    assert abs(axes[2, int(np.argmin(out.log_scales[0]))]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.eigvalsh(_float32_rebuild(out))[0] > 0


def test_a_cell_on_a_line_is_floored_in_both_thin_axes() -> None:
    """Rank-1 scatter: needles strung along one direction over 40 m. Both thin axes take the
    floor, the long one keeps the line's direction, and the float32 rebuild stays positive."""
    rng = np.random.default_rng(13)
    n = 30
    direction = np.array([2.0, -1.0, 0.5]) / np.linalg.norm([2.0, -1.0, 0.5])
    along = rng.uniform(-20, 20, n)
    g = gaussians(n, 13)
    g["xyz"] = along[:, None] * direction + [300.0, -40.0, 12.0]
    # Each a needle along the line: x is its long axis, rotated onto `direction`.
    half = np.cross([1.0, 0.0, 0.0], direction)
    angle = math.acos(float(direction[0]))
    axis = half / np.linalg.norm(half)
    g["quat"][:] = [math.cos(angle / 2), *(math.sin(angle / 2) * axis)]
    g["log_scales"][:] = [-1.0, -30.0, -30.0]
    g["logit"][:] = -3.0
    m = sm.merge_cells(moments(g, np.zeros(n, np.int64)))
    values = np.linalg.eigvalsh(sym(m.cov[0]))
    assert values[1] / values[2] < 1e-10  # rank 1 before the floor
    out = sm.to_splats(m, log_scale_range=RANGE)
    scales = np.sort(out.log_scales[0].astype(np.float64))
    assert scales[:2] == pytest.approx([_floor_scale(m)] * 2, abs=1e-5)
    x, y, z, w = out.quat_xyzw[0].astype(np.float64)
    long_axis = rotation(np.array([w, x, y, z]))[:, int(np.argmax(out.log_scales[0]))]
    assert abs(float(long_axis @ direction)) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.eigvalsh(_float32_rebuild(out))[0] > 0


def test_a_small_flat_cell_is_floored_at_the_smallest_spz_scale() -> None:
    """A flat cell 1 cm across: 2^-20 of its largest variance is below SPZ's smallest scale,
    so the absolute floor, e^lo, is the one that binds."""
    g = gaussians(40, 9)
    g["xyz"] = (g["xyz"] - g["xyz"].mean(0)) * 1e-3
    g["xyz"][:, 2] = 0.0
    g["log_scales"][:] = [-6.0, -6.0, -30.0]
    g["quat"][:] = [1.0, 0.0, 0.0, 0.0]
    g["logit"][:] = -3.0
    m = sm.merge_cells(moments(g, np.zeros(40, np.int64)))
    assert 0.5 * math.log(FLOOR * float(np.linalg.eigvalsh(sym(m.cov[0]))[-1])) < RANGE[0]
    out = sm.to_splats(m, log_scale_range=RANGE)
    assert float(out.log_scales[0].min()) == pytest.approx(RANGE[0], abs=1e-6)


def test_the_floor_is_the_modules() -> None:
    assert sm.MERGED_EIGEN_FLOOR == FLOOR


def test_colour_and_sh_are_weighted_means() -> None:
    g = gaussians(30, 10, sh=8)
    m = sm.merge_cells(moments(g, np.zeros(30, np.int64)))
    w = sigmoid(g["logit"]) * thomsen(np.exp(np.clip(g["log_scales"], *RANGE)))
    np.testing.assert_allclose(m.colour[0], (w[:, None] * g["f_dc"]).sum(0) / w.sum(), rtol=1e-12)
    np.testing.assert_allclose(m.sh[0], (w[:, None] * g["sh"]).sum(0) / w.sum(), rtol=1e-11)
    out = sm.to_splats(m, log_scale_range=RANGE)
    assert out.sh_rest is not None and out.sh_rest.shape == (1, 8, 3)
    np.testing.assert_allclose(out.sh_rest[0].reshape(-1), m.sh[0], rtol=1e-6, atol=1e-7)
    assert (
        sm.to_splats(
            sm.merge_cells(moments(gaussians(3, 1), np.zeros(3, np.int64))), log_scale_range=RANGE
        ).sh_rest
        is None
    )


def test_opacity_stays_finite_and_in_range() -> None:
    rng = np.random.default_rng(12)
    n = 2000
    sigma = np.exp(rng.uniform(-6, 1, size=(n, 3)))
    fill = np.exp(rng.uniform(np.log(1e-9), np.log(1e3), n))
    m = sm.Moments(
        key=np.arange(n, dtype=np.int64),
        weight=fill * thomsen(sigma),
        mean=np.zeros((n, 3)),
        cov=np.stack(
            [
                sigma[:, 0] ** 2,
                0 * sigma[:, 0],
                0 * sigma[:, 0],
                sigma[:, 1] ** 2,
                0 * sigma[:, 0],
                sigma[:, 2] ** 2,
            ],
            1,
        ),
        colour=np.zeros((n, 3)),
        sh=np.zeros((n, 0)),
    )
    out = sm.to_splats(m, log_scale_range=RANGE)
    assert np.isfinite(out.opacity_logit).all() and np.isfinite(out.log_scales).all()
    alpha = sigmoid(out.opacity_logit)
    assert (alpha >= sm.MERGED_OPACITY_MIN * (1 - 1e-3)).all()
    assert (alpha <= sm.MERGED_OPACITY_MAX * (1 + 1e-6)).all()
    # Under the cap the ink is conserved exactly: alpha * S(sigma_out) = W.
    kept = (fill > sm.MERGED_OPACITY_MIN * 1.001) & (fill < sm.MERGED_OPACITY_MAX * 0.999)
    assert kept.sum() > 500
    np.testing.assert_allclose(alpha[kept], fill[kept], rtol=1e-4)


def test_an_empty_cell_list_merges_to_itself() -> None:
    m = moments(gaussians(0, 1), np.zeros(0, np.int64))
    assert len(sm.merge_cells(m)) == 0


def test_a_lone_member_cell_is_that_member_bit_for_bit() -> None:
    # Cells 3 and 9 have one member each, cell 5 has three; a -0.0 covariance entry must
    # survive (adding a zero scatter would make it +0.0).
    g = gaussians(5, 13, sh=2)
    m = moments(g, np.array([3, 5, 5, 5, 9], np.int64))
    m.cov[0, 1] = -0.0
    merged = sm.merge_cells(m)
    assert merged.key.tolist() == [3, 5, 9]
    for field in sm.MOMENT_FIELDS:
        for row, source in ((0, 0), (2, 4)):
            assert getattr(merged, field)[row].tobytes() == getattr(m, field)[source].tobytes()
    assert np.signbit(merged.cov[0, 1])


def test_concat_and_take_are_field_wise() -> None:
    a = moments(gaussians(3, 14, sh=1), np.arange(3, dtype=np.int64))
    b = moments(gaussians(2, 15, sh=1), np.arange(3, 5, dtype=np.int64))
    both = sm.Moments.concat([a, b])
    assert len(both) == 5
    picked = both.take(np.array([4, 0]))
    for field in sm.MOMENT_FIELDS:
        whole = np.concatenate([getattr(a, field), getattr(b, field)])
        assert np.array_equal(getattr(both, field), whole)
        assert np.array_equal(getattr(picked, field), whole[[4, 0]])
    assert len(both.take(both.key > 2)) == 2
