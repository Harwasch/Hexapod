"""Tiny, analytic fixtures: no teacher imports, model files, network or GPU."""

import numpy as np
import pytest

from light_bake import (
    fit_albedo,
    linear_to_srgb,
    pack_sidecar,
    srgb_to_linear,
    sun_path,
    trace_visibility,
)
from splat_render import Camera, Splats


def test_colour_round_trip_and_binary_layout():
    linear = np.array([[0.001, 0.18, 1.0]])
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(linear)), linear)
    blob = pack_sidecar(
        linear,
        np.array([[0.0, -1.0, 1.0]]),
        np.linspace(0, 1, 32)[None],
        np.array([0.5]),
        np.array([1.0]),
    )
    assert len(blob) == 48
    row = np.frombuffer(blob, np.uint8)
    assert row[3] == 255
    np.testing.assert_array_equal(row[4:8], [128, 1, 255, 128])
    assert row[8] == 0 and row[39] == 255
    assert blob[40:44] == b"LL01"


def test_day_path_enu_east_morning_west_evening():
    path = sun_path()
    assert path.shape == (32, 3)
    np.testing.assert_allclose(np.linalg.norm(path, axis=1), 1)
    assert path[0, 0] > 0 and path[-1, 0] < 0
    assert np.all(path[:, 2] > 0)
    np.testing.assert_allclose(path[:, 2], path[::-1, 2])


def test_visibility_blocker_direction_and_empty_sky():
    grid = np.zeros((12, 12, 12))
    grid[6, :, :] = 20
    points = np.array([[2.0, 5.0, 5.0]])
    directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    visibility = trace_visibility(grid, np.zeros(3), 1, points, directions)[0]
    assert visibility[0] == pytest.approx(np.exp(-20), abs=1e-12)
    np.testing.assert_array_equal(visibility[1:], [1.0, 1.0])


def test_fit_shared_albedo_rejects_hidden_and_keeps_unseen_flagged():
    camera = Camera.look_at([0, 0, 5], [0, 0, 0], width=16, height=16)
    splats = Splats(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]]),
        np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        np.tile([0.1, 0.1, 0.01], (2, 1)),
        np.full((2, 3), 0.2),
        np.ones(2),
    )
    albedo, normal, confidence = fit_albedo(
        splats,
        [camera, camera],
        [np.full((16, 16, 3), 0.4), np.full((16, 16, 3), 0.6)],
        [np.full((16, 16), 5.0)] * 2,
        [np.ones((16, 16))] * 2,
    )
    np.testing.assert_allclose(albedo[0], 0.5)
    np.testing.assert_allclose(albedo[1], srgb_to_linear(np.array([0.2] * 3)))
    np.testing.assert_array_equal(confidence, [1.0, 0.0])
    np.testing.assert_array_equal(normal, [[0.0, 0.0, 1.0]] * 2)
