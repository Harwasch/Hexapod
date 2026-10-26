"""Analytical fixtures only: private capture, model and network are never required."""

import numpy as np
import pytest

from light_photos import observations, placed_camera, robust_colour
from splat_render import Camera, Splats


def test_similarity_preserves_projection_and_scales_camera_depth():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([3.0, 4, -2])
    image = {"qvec": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], "centre": [1.0, 2, 3]}
    intrinsics = {
        "model": "SIMPLE_RADIAL",
        "width": 100,
        "height": 80,
        "params": {"f": 80.0, "cx": 50.0, "cy": 40.0, "k": 0.01},
    }
    camera = placed_camera(
        image, intrinsics, {"rotation": rotation, "scale": 2.0, "translation": translation}, 50
    )
    # Q rotates world +x to camera +y; camera-space offset is (.5, -.25, 2).
    point = np.array([[0.75, 1.5, 5.0]])
    projected, depth = camera.project(2 * point @ rotation.T + translation)
    np.testing.assert_allclose(projected, [[35.0, 15.0]])
    np.testing.assert_allclose(depth, [4.0])
    intrinsics["model"] = "OPENCV"
    with pytest.raises(ValueError, match="SIMPLE_RADIAL"):
        placed_camera(image, intrinsics, {}, 50)


def test_robust_fit_rejects_view_outlier_and_zero_weight_samples():
    samples = np.array(
        [
            [[0.2] * 3, [100.0] * 3],
            [[0.21] * 3, [-100.0] * 3],
            [[0.19] * 3, [3.0] * 3],
            [[0.9] * 3, [4.0] * 3],
        ],
        np.float32,
    )
    weights = np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0.5, 0]], np.float32)
    fitted, retained = robust_colour(samples, weights)
    np.testing.assert_allclose(fitted[0], 0.2, atol=1e-7)
    np.testing.assert_array_equal(fitted[1], 0)
    np.testing.assert_array_equal(retained, [3, 0])


def test_visibility_rejects_occluded_sample_before_robust_fit():
    camera = Camera.look_at([0, 0, 5], [0, 0, 0], width=16, height=16)
    splats = Splats(
        np.array([[0.0, 0, 0], [0, 0, -2.0]]),
        np.tile([1.0, 0, 0, 0], (2, 1)),
        np.tile([0.1, 0.1, 0.01], (2, 1)),
        np.full((2, 3), 0.2),
        np.ones(2),
    )
    ids, colours, weights, direction = observations(
        splats,
        camera,
        np.tile([0.0, 0, 1], (2, 1)),
        np.tile(np.eye(3), (2, 1, 1)),
        np.full((16, 16, 3), 0.4),
        np.full((16, 16), 5.0),
        np.ones((16, 16)),
    )
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_allclose(colours, 0.4)
    assert weights[0] > 0
    np.testing.assert_allclose(direction, [[0, 0, 1]])
