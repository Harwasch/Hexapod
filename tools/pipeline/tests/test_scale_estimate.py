"""`scale_estimate`: metres per model unit from how high a handheld phone is.

Synthetic, with known answers: ground and cameras are laid out in a levelled frame in
"model units", and the camera height in those units is chosen, so the right scale is
`1.5 / height` exactly. What is asserted is how close the estimate lands, that its
uncertainty is said rather than hidden, and -- as much as anything -- that it refuses
(None) when the capture cannot carry an estimate, because the caller then keeps the
scale unresolved, and that is an honest answer where a confident wrong one is not.
"""

from __future__ import annotations

import numpy as np
import pytest

import scale_estimate
from scale_estimate import estimate_metres_per_unit


def _orbit(count: int, *, radius: float, height: float, seed: int = 0) -> np.ndarray:
    """`count` cameras round a full circle at `height` above z = 0, a little jitter."""
    rng = np.random.default_rng(seed)
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), height + rng.normal(0, 0.02, count)],
        axis=1,
    )


def _ground(
    extent: float, count: int, *, slope: tuple[float, float] = (0.0, 0.0), seed: int = 1
) -> np.ndarray:
    """Points on the plane `z = sx * x + sy * y` over a square of side `2 * extent`."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-extent, extent, (2, count))
    z = slope[0] * x + slope[1] * y + rng.normal(0, 0.005, count)
    return np.stack([x, y, z], axis=1)


def _subject(count: int, *, top: float, seed: int = 2) -> np.ndarray:
    """A table-top subject at the middle: what an orbit mostly reconstructs."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.normal(0, 0.3, count), rng.normal(0, 0.3, count), rng.uniform(0.0, top, count)],
        axis=1,
    )


def test_cameras_one_point_eight_units_over_flat_ground_are_five_sixths_of_a_metre() -> None:
    """The spool's shape: an orbit 1.8 units up, round a subject, over a floor."""
    cameras = _orbit(120, radius=3.0, height=1.8)
    points = np.concatenate([_ground(6.0, 30_000), _subject(20_000, top=1.2)])

    found = estimate_metres_per_unit(cameras, points)

    assert found is not None
    assert found.scale == pytest.approx(1.5 / 1.8, rel=0.02)
    assert found.camera_height_units == pytest.approx(1.8, rel=0.02)
    assert found.n_cameras == 120 and found.cameras_offered == 120
    # Tight heights: the uncertainty is the handheld prior's ±20% and not much more.
    assert 20.0 <= found.uncertainty_pct < 21.0
    assert found.spread < 0.05


def test_the_estimate_follows_the_reconstructions_units_not_metres() -> None:
    """The whole point: the same capture at any normalisation gives the same metres.

    COLMAP's ten units across is arbitrary, so the model is scaled by 7 here and the
    estimate must come out 7 times smaller per unit -- the cell is sized from the capture,
    never in metres, so nothing in it can know which normalisation it was given.
    """
    cameras = _orbit(60, radius=3.0, height=1.8)
    points = np.concatenate([_ground(6.0, 20_000), _subject(10_000, top=1.2)])

    unit = estimate_metres_per_unit(cameras, points)
    scaled = estimate_metres_per_unit(cameras * 7.0, points * 7.0)

    assert unit is not None and scaled is not None
    assert scaled.scale == pytest.approx(unit.scale / 7.0, rel=1e-9)
    assert scaled.uncertainty_pct == pytest.approx(unit.uncertainty_pct, rel=1e-9)


def test_a_walk_up_a_slope_is_measured_against_the_ground_under_each_camera() -> None:
    """Per cell, not one plane: on a 15% slope the lowest ground is metres below most
    cameras, and only the ground under each one says how high it was held."""
    slope = (0.15, 0.05)
    x = np.linspace(-8.0, 8.0, 40)
    y = np.full(40, 1.0)
    cameras = np.stack([x, y, slope[0] * x + slope[1] * y + 1.8], axis=1)
    points = _ground(10.0, 60_000, slope=slope)

    found = estimate_metres_per_unit(cameras, points)

    # Measured 2.5% low: a cell's 5th percentile on a slope is its downhill side.
    assert found is not None
    assert found.scale == pytest.approx(1.5 / 1.8, rel=0.04)
    # What one global ground (its 5th percentile) would have said: far off.
    naive = float(np.median(cameras[:, 2]) - np.percentile(points[:, 2], 5.0))
    assert abs(1.5 / naive - 1.5 / 1.8) > 0.2 * (1.5 / 1.8)


def test_a_few_cameras_held_high_or_low_move_the_median_little_and_the_uncertainty_up() -> None:
    """A fifth of the frames from a crouch or held overhead: the median holds, and the
    spread they add is reported in the ±%."""
    cameras = _orbit(100, radius=3.0, height=1.8)
    rng = np.random.default_rng(5)
    odd = rng.choice(100, 20, replace=False)
    cameras[odd[:10], 2] = 0.6
    cameras[odd[10:], 2] = 2.6
    points = np.concatenate([_ground(6.0, 30_000), _subject(10_000, top=1.2)])
    steady = estimate_metres_per_unit(_orbit(100, radius=3.0, height=1.8), points)

    found = estimate_metres_per_unit(cameras, points)

    assert found is not None and steady is not None
    assert found.scale == pytest.approx(1.5 / 1.8, rel=0.05)
    assert found.uncertainty_pct > steady.uncertainty_pct


def test_too_few_cameras_is_no_estimate() -> None:
    points = _ground(6.0, 10_000)
    assert estimate_metres_per_unit(_orbit(7, radius=3.0, height=1.8), points) is None
    assert estimate_metres_per_unit(_orbit(8, radius=3.0, height=1.8), points) is not None


def test_cameras_with_no_ground_under_them_are_no_estimate() -> None:
    """An orbit whose reconstruction is all subject: there is nothing to be high above."""
    cameras = _orbit(60, radius=3.0, height=1.8)
    assert estimate_metres_per_unit(cameras, _subject(20_000, top=1.2)) is None
    # Ground only far beyond the ring is not ground under the cameras either.
    far = _ground(40.0, 40_000)
    far = far[np.hypot(far[:, 0], far[:, 1]) > 15.0]
    assert estimate_metres_per_unit(cameras, np.concatenate([far, _subject(5_000, top=1.2)])) is (
        None
    )


def test_ground_under_only_some_of_the_cameras_is_no_estimate() -> None:
    """Ground under a third of the orbit: those cameras are not evidence for the rest."""
    cameras = _orbit(60, radius=3.0, height=1.8)
    ground = _ground(6.0, 30_000)
    ground = ground[ground[:, 0] > 2.0]
    assert estimate_metres_per_unit(cameras, ground) is None


def test_heights_that_disagree_too_much_are_no_estimate() -> None:
    """Cameras anywhere from knee to well overhead: no one height describes them."""
    cameras = _orbit(60, radius=3.0, height=1.8)
    cameras[:, 2] = np.linspace(0.3, 4.0, 60)
    assert estimate_metres_per_unit(cameras, _ground(6.0, 30_000)) is None


def test_a_surface_above_the_cameras_is_not_their_ground() -> None:
    """Under a canopy, or a ceiling: the lowest surface the grid finds is over the cameras.
    The heights come out negative, and that is a refusal, not a tiny scale."""
    cameras = _orbit(60, radius=3.0, height=1.8)
    canopy = _ground(6.0, 30_000) + np.array([0.0, 0.0, 3.0])
    assert estimate_metres_per_unit(cameras, canopy) is None


def test_the_per_cell_ground_is_numpys_percentile_of_the_cell() -> None:
    """The vectorised percentile is `np.percentile`, cell by cell, not an approximation."""
    rng = np.random.default_rng(9)
    xyz = rng.uniform(0.0, 3.0, (5_000, 3))
    ground = scale_estimate._cell_ground(xyz, np.zeros(2), 0.5, 5.0, 8)

    assert len(ground) == 36
    ix = np.floor(xyz[:, 0] / 0.5).astype(int)
    iy = np.floor(xyz[:, 1] / 0.5).astype(int)
    for (cx, cy), value in ground.items():
        inside = (ix == cx) & (iy == cy)
        assert value == pytest.approx(float(np.percentile(xyz[inside, 2], 5.0)), abs=1e-12)


def test_the_evidence_says_what_was_assumed_and_how_well_it_is_known() -> None:
    cameras = _orbit(40, radius=3.0, height=1.8)
    found = estimate_metres_per_unit(cameras, _ground(6.0, 20_000))
    assert found is not None

    evidence = found.to_dict()

    assert evidence["method"] == "camera-height"
    assert evidence["priorM"] == 1.5
    assert evidence["priorRangeM"] == [1.2, 1.8]
    assert evidence["cameras"] == 40
    assert evidence["uncertaintyPct"] == round(found.uncertainty_pct, 1)
    assert "estimate, not a measurement" in str(evidence["note"])
