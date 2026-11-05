"""A common, explicit evaluation yardstick; no CUDA required."""

import numpy as np
import pytest


def test_affine_fit_matches_known_channel_gain_and_handles_flat_channels() -> None:
    from experiments.bench.rescore_maths import affine_correct

    target = np.linspace(0.1, 0.8, 300).reshape(100, 3)
    prediction = target * [0.8, 0.9, 1.1] + [0.03, 0.04, -0.02]
    corrected, gain, offset = affine_correct(prediction, target)
    np.testing.assert_allclose(corrected, target, atol=1e-12)
    np.testing.assert_allclose(gain, [0.8, 0.9, 1.1], atol=1e-12)
    np.testing.assert_allclose(offset, [0.03, 0.04, -0.02], atol=1e-12)
    assert np.isfinite(affine_correct(np.ones((10, 3)) * 0.2, np.ones((10, 3)) * 0.3)[0]).all()


def test_half_gt_is_area_average_with_scaled_camera() -> None:
    from experiments.bench.rescore_maths import half_camera, half_image

    image = np.arange(8 * 12 * 3).reshape(8, 12, 3)
    small = half_image(image)
    np.testing.assert_array_equal(small[1, 2], image[2:4, 4:6].mean((0, 1)))
    camera = {
        "width": 12,
        "height": 8,
        "K": [[12, 0, 6], [0, 12, 4], [0, 0, 1]],
        "viewmat": np.eye(4).tolist(),
    }
    half = half_camera(camera)
    assert (half["width"], half["height"]) == (6, 4)
    np.testing.assert_array_equal(half["K"], [[6, 0, 3], [0, 6, 2], [0, 0, 1]])
    assert half["viewmat"] == camera["viewmat"]
    with pytest.raises(ValueError):
        half_image(np.zeros((3, 4, 3)))


def test_se3_and_motion_use_camera_centre_and_scene_extent() -> None:
    from experiments.bench.rescore_maths import pose_motion, se3

    transform = se3([0, 0, 0, 0, 0.2, 0])
    np.testing.assert_array_equal(transform[:3, 3], [0, 0.2, 0])
    assert pose_motion(np.eye(4), transform, 10) == {"translationPct": 2.0, "rotationDeg": 0.0}
    rotated = se3([0, 0, np.pi / 2, 0, 0, 0])
    np.testing.assert_allclose(rotated[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    assert pose_motion(np.eye(4), rotated, 10)["rotationDeg"] == pytest.approx(90)


def test_shared_overheads_depend_only_on_frozen_support_and_exclude_outlier() -> None:
    from experiments.bench.rescore_maths import footprint, shared_cameras

    x, y = np.meshgrid(np.arange(11), np.arange(11))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    points = np.concatenate([points, [[1000, 1000, 1000]]])
    cameras = [[0, 0, 2], [5, 0, 2], [5, 5, 2]]
    result = shared_cameras(points, cameras, [0, 0, 1])
    assert result["extent"] < 20
    assert [c["downAngleDeg"] for c in result["cameras"]] == [90, 75, 60]
    for camera in result["cameras"]:
        view = np.asarray(camera["viewmat"])
        vertices = np.asarray(result["footprint"]) @ view[:3, :3].T + view[:3, 3]
        projected = vertices @ np.asarray(camera["K"]).T
        pixels = projected[:, :2] / projected[:, 2:]
        assert (pixels > 0).all() and (pixels < camera["width"]).all()
        mask = footprint(camera, result["footprint"])
        assert 0 < mask.mean() < 1
        forward = np.asarray(camera["viewmat"])[2, :3]
        assert np.rad2deg(np.arcsin(-forward[2])) == pytest.approx(camera["downAngleDeg"])
    assert result == shared_cameras(points, cameras, [0, 0, 1])


def test_far_field_never_affects_needles_or_weighted_denominator() -> None:
    from experiments.bench.rescore_maths import needle_scores

    axes = np.log([[0.1, 0.1, 0.1], [3, 0.1, 0.1], [1000, 0.01, 0.01]])
    columns = {
        "x": np.zeros(3),
        "opacity": np.array([0.0, np.log(3), 50.0]),
        "far_field": np.array([0, 0, 1]),
        **{f"scale_{i}": axes[:, i] for i in range(3)},
    }
    result = needle_scores(columns, 100)
    assert result["farFieldCount"] == 1 and result["foregroundCount"] == 2
    assert result["elongation"]["share"] == 0.5
    assert result["elongation"]["opacityWeightedShare"] == pytest.approx(0.75 / 1.25)
    assert result["oversized"]["2"]["count"] == 1
    assert result["oversized"]["5"]["count"] == 0
