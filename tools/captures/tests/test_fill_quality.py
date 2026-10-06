"""`fill_quality`: quality-aware support, its classes, the leave-out's withheld look, and
hole clusters, on synthetic scenes (CPU renderer)."""

from __future__ import annotations

import math

import numpy as np
from fill_scenes import (
    ball_scene,
    pumpkin_cameras,
    ring_cameras,
    splats,
    spool_cameras,
    table_scene,
)

import fill_quality as fq
from splat_render import render


def _plane(n: int = 21, size: float = 1.0) -> np.ndarray:
    g = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def test_capture_focus_is_where_an_orbit_looks() -> None:
    cams = ring_cameras(8, 20.0, 3.0, target=(0.3, -0.2, 0.5))
    focus = fq.capture_focus(cams)
    assert np.allclose(focus.centre, [0.3, -0.2, 0.5], atol=1e-6)
    assert math.isclose(focus.distance, 3.0, rel_tol=1e-6)
    assert math.isclose(focus.hfov_deg, 60.0, rel_tol=1e-6)
    assert math.isclose(focus.elevation(cams[0].centre), 20.0, abs_tol=1e-6)


def test_normals_come_from_the_neighbours_when_the_splats_say_nothing() -> None:
    plane = splats(_plane(), [0.5, 0.5, 0.5])  # random orientations, isotropic scales
    normals = fq.surface_normals(plane)
    assert np.abs(normals[:, 2]).min() > 0.99


def test_shortest_axis_is_a_flat_splats_normal() -> None:
    s = splats(_plane(5), [0.5, 0.5, 0.5])
    s.scales[:] = [0.05, 0.05, 0.001]
    s.rotations[:] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]  # 90 deg about x
    axes = fq.shortest_axes(s)
    assert np.allclose(np.abs(axes), [0.0, 1.0, 0.0], atol=1e-9)


def _quality(scene, cams):
    focus = fq.capture_focus(cams)
    return fq.measure_quality(scene, cams, render, focus, width=96), focus


def test_a_plane_seen_only_at_grazing_angles_is_not_known() -> None:
    plane = splats(_plane(), [0.5, 0.5, 0.5])
    grazing = ring_cameras(8, 20.0, 2.0, target=(0, 0, 0))
    q, _ = _quality(plane, grazing)
    inner = np.linalg.norm(plane.positions[:, :2], axis=1) < 0.3
    assert (q.classes[inner] != fq.KNOWN).all()
    assert (q.best[inner] <= math.sin(math.radians(23.0)) + 0.01).all()
    assert (q.classes[inner] == fq.WEAK).mean() > 0.9
    # The same plane from well above is known.
    above = ring_cameras(8, 60.0, 2.0, target=(0, 0, 0))
    q2, _ = _quality(plane, above)
    assert (q2.classes[inner] == fq.KNOWN).all()


def test_the_spool_top_is_weak_and_its_side_known() -> None:
    scene, m = table_scene()
    cams = spool_cameras()
    q, _ = _quality(scene, cams)
    # Every camera sees the top at 23 degrees or less.
    top_centre = np.array([0.0, 0.0, 0.7])
    for c in cams:
        d = c.centre - top_centre
        assert math.degrees(math.asin(d[2] / np.linalg.norm(d))) <= 23.5
    inner_top = m["top"] & (np.linalg.norm(scene.positions[:, :2], axis=1) < 0.3)
    assert (q.classes[inner_top] != fq.KNOWN).mean() > 0.95
    upper_side = m["side"] & (scene.positions[:, 2] > 0.15)
    assert (q.classes[upper_side] == fq.KNOWN).mean() > 0.8
    summary = q.summary()
    assert summary["gaussians"] == len(scene)


def test_a_ball_seen_from_above_has_a_weak_lower_belt() -> None:
    scene, m = ball_scene()
    q, _ = _quality(scene, pumpkin_cameras())
    belt = m["lower"] & (scene.positions[:, 2] < 0.12) & (scene.positions[:, 2] > 0.05)
    assert (q.classes[belt] != fq.KNOWN).mean() > 0.9
    upper = m["ball"] & (scene.positions[:, 2] > 0.55)
    assert (q.classes[upper] == fq.KNOWN).mean() > 0.9


def test_a_blurred_photo_scores_lower_and_lowers_quality() -> None:
    import cv2

    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    s = fq.photo_sharpness([sharp, sharp, blurred, None])
    assert s[0] == s[1] == 1.0 and s[2] < 0.5 and s[3] == 1.0
    plane = splats(_plane(), [0.5, 0.5, 0.5])
    cams = ring_cameras(4, 60.0, 2.0, target=(0, 0, 0))
    focus = fq.capture_focus(cams)
    full = fq.measure_quality(plane, cams, render, focus, width=96)
    soft = fq.measure_quality(plane, cams, render, focus, width=96, sharpness=np.full(4, 0.3))
    assert soft.best.max() < full.best.max()


def test_far_cameras_count_less_than_near_ones() -> None:
    plane = splats(_plane(), [0.5, 0.5, 0.5])
    near = ring_cameras(6, 60.0, 2.0, target=(0, 0, 0))
    far = ring_cameras(6, 60.0, 8.0, target=(0, 0, 0))
    focus = fq.capture_focus(near)  # the capture's own resolution: the near cameras'
    q_far = fq.measure_quality(plane, far, render, focus, width=96)
    q_near = fq.measure_quality(plane, near, render, focus, width=96)
    inner = np.linalg.norm(plane.positions[:, :2], axis=1) < 0.3
    assert q_far.best[inner].mean() < 0.5 * q_near.best[inner].mean()


def test_a_subset_of_cameras_and_the_withheld_look() -> None:
    scene, m = table_scene()
    low = spool_cameras()
    high = ring_cameras(6, 65.0, 2.2)
    cams = low + high
    focus = fq.capture_focus(cams)
    everyone = fq.measure_quality(scene, cams, render, focus, width=96)
    kept = everyone.subset(range(len(low)))
    alone = fq.measure_quality(scene, low, render, focus, width=96)
    assert np.allclose(kept.best, alone.best) and (kept.classes == alone.classes).all()
    inner_top = m["top"] & (np.linalg.norm(scene.positions[:, :2], axis=1) < 0.3)
    assert (everyone.classes[inner_top] == fq.KNOWN).mean() > 0.9
    withheld, dropped = fq.withheld_by_holdout(kept, everyone)
    assert withheld[inner_top].mean() > 0.9
    assert not (withheld & dropped).any()
    assert not withheld[m["side"] & (scene.positions[:, 2] > 0.2)].any()


def test_hole_clusters_are_found_and_ranked_without_a_region() -> None:
    a = _plane(9, 0.4) + [1.0, 0.0, 0.0]
    b = _plane(13, 0.6) + [-1.0, 0.0, 0.0]
    c = _plane(5, 0.2) + [0.0, 1.5, 0.0]  # tiny: noise
    s = splats(np.concatenate([a, b, c, _plane(41, 3.0) + [0, 0, -0.5]]), [0.5, 0.5, 0.5])
    classes = np.full(len(s), fq.KNOWN, np.int8)
    classes[: len(a) + len(b) + len(c)] = fq.UNKNOWN
    focus = fq.Focus(np.zeros(3), 4.0, 60.0, 50.0)
    found = fq.hole_clusters(s, classes, focus, min_size=30)
    assert len(found.info) == 2
    assert found.info[0]["gaussians"] == len(b) and found.info[1]["gaussians"] == len(a)
    assert (found.labels[len(a) + len(b) :] == -1).all()
    assert found.info[0]["weight"] > found.info[1]["weight"]
