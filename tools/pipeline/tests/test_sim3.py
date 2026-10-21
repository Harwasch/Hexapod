"""`sim3.fit_sim3`: a seeded, reproducible similarity between named point sets.

Pure numpy. The noise levels and outlier offsets are chosen here, so every bound below is
derived from them (or is machine precision for exact data), not measured off one run.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import sfm
import sim3


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return sfm.quat_to_matrix([float(v) for v in q])


def _scene(n: int, seed: int = 1) -> tuple[dict[str, list[float]], sfm.Similarity]:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-10.0, 10.0, size=(n, 3))
    truth = sfm.Similarity(
        scale=float(rng.uniform(0.3, 3.0)),
        rotation=_rotation(rng),
        translation=rng.uniform(-50.0, 50.0, size=3),
    )
    return {f"p{i:03d}": points[i].tolist() for i in range(n)}, truth


def _apply(points: dict[str, list[float]], similarity: sfm.Similarity) -> dict[str, list[float]]:
    return {
        name: similarity.apply(np.asarray([value]))[0].tolist() for name, value in points.items()
    }


def test_exact_correspondences_are_recovered_to_machine_precision() -> None:
    source, truth = _scene(20)
    fit = sim3.fit_sim3(source, _apply(source, truth), max_error_m=None)

    assert np.abs(fit.similarity.rotation - truth.rotation).max() < 1e-9
    assert fit.similarity.scale == pytest.approx(truth.scale, rel=1e-12)
    assert np.abs(fit.similarity.translation - truth.translation).max() < 1e-8
    assert len(fit.inliers) == 20
    assert fit.method == "lmeds"


def test_a_fixed_threshold_rejects_planted_outliers() -> None:
    source, truth = _scene(30, seed=2)
    target = _apply(source, truth)
    rng = np.random.default_rng(3)
    names = sorted(source)
    outliers = set(names[::3][:9])  # 30%
    for name in names:
        # 1 cm per axis: a 3D residual within the 0.5 m threshold by 28 sigma.
        target[name] = (np.asarray(target[name]) + rng.normal(scale=0.01, size=3)).tolist()
        if name in outliers:
            target[name] = (np.asarray(target[name]) + np.array([10.0, 0.0, 0.0])).tolist()

    fit = sim3.fit_sim3(source, target, max_error_m=0.5)

    assert set(fit.inliers) == set(names) - outliers
    assert fit.method == "ransac" and fit.threshold_m == 0.5
    assert all(fit.residuals_m[name] > 5.0 for name in outliers)


def test_lmeds_needs_no_threshold_and_keeps_the_majority() -> None:
    source, truth = _scene(30, seed=4)
    target = _apply(source, truth)
    rng = np.random.default_rng(5)
    names = sorted(source)
    outliers = set(names[:12])  # 40%
    for name in names:
        noise = rng.normal(scale=0.005, size=3)
        shift = rng.uniform(5.0, 20.0, size=3) if name in outliers else np.zeros(3)
        target[name] = (np.asarray(target[name]) + noise + shift).tolist()

    fit = sim3.fit_sim3(source, target, max_error_m=None)

    assert set(fit.inliers) == set(names) - outliers
    # 2.5 robust sigmas of 5 mm noise per axis: centimetres, far below the 5 m shifts.
    assert fit.threshold_m < 0.1


def test_the_same_seed_gives_the_same_bytes() -> None:
    source, truth = _scene(80, seed=6)  # C(80, 3) > 2000: sampled, not enumerated
    target = _apply(source, truth)
    rng = np.random.default_rng(7)
    for name in sorted(source)[:20]:
        target[name] = (np.asarray(target[name]) + rng.uniform(1, 5, size=3)).tolist()

    first = sim3.fit_sim3(source, target, max_error_m=None, seed=11)
    second = sim3.fit_sim3(source, target, max_error_m=None, seed=11)

    assert first.iterations == sim3.DEFAULT_ITERATIONS
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_small_sets_are_enumerated() -> None:
    source, truth = _scene(6, seed=8)
    fit = sim3.fit_sim3(source, _apply(source, truth), max_error_m=None, seed=123)

    assert fit.iterations == math.comb(6, 3) == 20
    # Enumerated, so the seed cannot matter.
    other = sim3.fit_sim3(source, _apply(source, truth), max_error_m=None, seed=456)
    assert other.to_dict() | {"seed": 0} == fit.to_dict() | {"seed": 0}


def test_fewer_than_three_or_all_collinear_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 3 common points"):
        sim3.fit_sim3(
            {"a": [0, 0, 0], "b": [1, 0, 0]}, {"a": [0, 0, 0], "b": [1, 0, 0]}, max_error_m=None
        )
    line = {f"p{i}": [float(i), 2.0 * i, 0.0] for i in range(6)}
    with pytest.raises(ValueError, match="collinear"):
        sim3.fit_sim3(line, line, max_error_m=None)
    with pytest.raises(ValueError, match="positive"):
        sim3.fit_sim3(line, line, max_error_m=0.0)
    # Names that only one side has are not correspondences.
    with pytest.raises(ValueError, match="got 2"):
        sim3.fit_sim3(
            {"a": [0, 0, 0], "b": [1, 0, 0], "c": [0, 1, 0]},
            {"a": [0, 0, 0], "b": [1, 0, 0], "z": [0, 1, 0]},
            max_error_m=None,
        )


def test_it_agrees_with_umeyama_when_nothing_is_an_outlier() -> None:
    source, truth = _scene(25, seed=9)
    target = _apply(source, truth)
    rng = np.random.default_rng(10)
    for name in sorted(target):
        target[name] = (np.asarray(target[name]) + rng.normal(scale=0.02, size=3)).tolist()
    names = sorted(source)
    expected = sfm.umeyama(
        np.asarray([source[n] for n in names]), np.asarray([target[n] for n in names])
    )

    fit = sim3.fit_sim3(source, target, max_error_m=None)

    # Every point an inlier, so the final refit is Umeyama on all of them: equal up to
    # floating-point summation order.
    assert len(fit.inliers) == 25
    assert fit.similarity.scale == pytest.approx(expected.scale, rel=1e-12)
    assert np.abs(fit.similarity.rotation - expected.rotation).max() < 1e-12
    assert np.abs(fit.similarity.translation - expected.translation).max() < 1e-9
    stats = fit.stats()
    assert stats["median"] <= stats["p90"] <= stats["max"]
    assert stats["rms"] > 0.0


def test_three_points_fit_without_dividing_by_zero() -> None:
    source = {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0], "c": [0.0, 1.0, 0.0]}
    target = {k: [2 * v[0] + 1, 2 * v[1], 2 * v[2]] for k, v in source.items()}
    fit = sim3.fit_sim3(source, target, max_error_m=None)

    assert fit.similarity.scale == pytest.approx(2.0)
    assert len(fit.inliers) == 3 and math.isfinite(fit.threshold_m)


def test_ransac_with_fewer_than_three_inliers_is_no_fit() -> None:
    """R2 review (#51, medium): when no 3-point sample found three points within the
    threshold, the sampled fit was returned anyway, and `exif_gps` applied it as the
    capture's frame. No consensus is now refused, by name."""
    source = {"a": (0, 0, 0), "b": (1, 0, 0), "c": (0, 1, 0), "d": (1, 1, 0.5)}
    target = {"a": (0, 0, 0), "b": (10, 0, 0), "c": (0, 0.1, 0), "d": (3, -7, 5)}
    with pytest.raises(sim3.NoConsensusError, match=r"0 of 4 points agree within 0\.05 m"):
        sim3.fit_sim3(source, target, max_error_m=0.05)
    assert issubclass(sim3.NoConsensusError, ValueError)
    # LMedS sets its own threshold from the median, so it always has its half.
    assert len(sim3.fit_sim3(source, target, max_error_m=None).inliers) >= 2
