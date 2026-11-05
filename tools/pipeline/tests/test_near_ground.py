import json
import struct
from pathlib import Path

import numpy as np
import pytest

import sfm
from experiments import near_ground
from experiments.near_ground import distribution, observations, pool_support, project


def test_projection_uses_distortion_and_world_to_camera() -> None:
    camera = sfm.Camera(1, "SIMPLE_RADIAL", 100, 90, (10, 50, 45, 0.1), ("f", "cx", "cy", "k"))
    image = sfm.Image(1, "a.jpg", (1, 0, 0, 0), (1, 0, 0), 1, 0, 0)
    xy, depth = project(np.array([[0, 1, 1], [0, 0, -1]]), image, camera)
    np.testing.assert_allclose(xy, [[62, 57], [39, 45]])
    np.testing.assert_array_equal(depth, [1, -1])


def test_support_counts_distinct_pool_images_not_heldout_tracks() -> None:
    points = sfm.Points3D(
        np.array([1, 2], dtype=np.uint64),
        np.zeros((2, 3)),
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros(2),
        np.array([[1, 0], [1, 1], [2, 0], [3, 0], [3, 1]], dtype=np.uint32),
        np.array([0, 4, 5], dtype=np.int64),
    )
    counts, sharp = pool_support(points, {1, 2}, {1: 10.0, 2: 30.0, 3: 1000.0})
    np.testing.assert_array_equal(counts, [2, 0])
    assert sharp[0] == 20 and np.isnan(sharp[1])
    assert distribution(sharp)["count"] == 1


def test_observation_reader_keeps_signed_unmatched_and_xy(tmp_path: Path) -> None:
    path = tmp_path / "images.bin"
    path.write_bytes(
        struct.pack("<QIdddddddI", 1, 42, 1, 0, 0, 0, 0, 0, 0, 1)
        + b"frame.jpg\0"
        + struct.pack("<Qddqddq", 2, 1.5, 2.5, 900, 3, 4, -1)
    )
    data = observations(path)[42]
    assert data["point"].tolist() == [900, -1]
    assert data["xy"][0].tolist() == [1.5, 2.5]
    path.write_bytes(path.read_bytes()[:72])
    with pytest.raises(ValueError, match="truncated"):
        observations(path)


def test_analysis_regions_use_pool_residuals_not_colmap_or_heldout_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera = sfm.Camera(1, "SIMPLE_PINHOLE", 90, 90, (30, 45, 45), ("f", "cx", "cy"))
    images = tuple(
        sfm.Image(i, name, (1, 0, 0, 0), (0, 0, 0), 1, 3, 3)
        for i, name in ((3, "train.jpg"), (7, "test.jpg"))
    )
    points = sfm.Points3D(
        np.array([11, 22, 33], dtype=np.uint64),
        np.array([[0, 2, 3], [0, 0, 3], [0, 0, -3]]),
        np.zeros((3, 3), dtype=np.uint8),
        np.full(3, 123.0),
        np.tile(np.array([[3, 0], [7, 0], [99, 0]], dtype=np.uint32), (3, 1)),
        np.array([0, 3, 6, 9], dtype=np.int64),
    )
    model = type("Model", (), {"images": images, "cameras": (camera,)})()
    monkeypatch.setattr(sfm, "read_model", lambda _: model)
    monkeypatch.setattr(sfm, "read_points3d", lambda _: points)
    monkeypatch.setattr(near_ground, "sharpness", lambda p: 10 if p.name == "train.jpg" else 100)
    data = {}
    for image, offset in zip(images, (1, 2), strict=True):
        xy, _ = project(points.xyz, image, camera)
        records = np.zeros(3, dtype=[("xy", "f8", (2,)), ("point", "i8")])
        records["xy"] = xy + np.array([offset, 0])
        records["point"] = points.ids
        data[image.id] = records
    monkeypatch.setattr(near_ground, "observations", lambda _: data)
    (tmp_path / "split.json").write_text(json.dumps({"pool": ["train.jpg"], "test": ["test.jpg"]}))
    result = near_ground.analyse(tmp_path)
    regions = result["perFrame"][0]["regions"]
    assert regions["bottom"]["projected"]["points"] == 1
    assert regions["upper"]["projected"]["points"] == 1  # Behind-camera point excluded.
    assert regions["bottom"]["trackedInTest"]["trainViews"]["median"] == 1
    assert regions["bottom"]["trackedInTest"]["supportSharpness"]["median"] == 10
    assert regions["bottom"]["trackedInTest"]["poolReprojectionPx"]["median"] == 1
    assert regions["bottom"]["testReprojectionPx"]["median"] == 2
