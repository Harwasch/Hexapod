"""CPU numerical/export tests. Synthetic arrays below test math, never model quality."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import (
    export_depth_mesh,
    export_gaussians,
    export_points,
    geometry_diagnostics,
    points_from_depth,
)
from sources import SourceError, prepare_sources


def test_depth_unprojection_honors_camera_intrinsics_pose_and_invalid_depth():
    depth = np.array([[[2, 0], [2, 2]]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, 3] = [10, 20, 30]
    intrinsic = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 1]]], dtype=np.float32)
    points = points_from_depth(depth, pose, intrinsic)
    np.testing.assert_allclose(points[0, 1, 1], [11, 21, 32])
    assert np.isnan(points[0, 0, 1]).all()


def test_extreme_aspect_images_have_bounded_preprocessing(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(2):
        Image.new("RGB", (1, 10000), "blue").save(inputs / f"{index}.png")
    frames, _ = prepare_sources(inputs, tmp_path / "frames", target_size=280)
    assert frames.shape == (2, 280, 280, 3)
    assert np.isfinite(frames).all()


def read_ply(path):
    payload = path.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    fields = []
    for line in header.decode().splitlines():
        if line.startswith("property"):
            _, kind, name = line.split()
            fields.append((name, "u1" if kind == "uchar" else "<f4"))
    return np.frombuffer(body, dtype=fields)


def test_gaussian_export_standard_activation_and_invalid_filter(tmp_path):
    path = tmp_path / "scene.ply"
    count = export_gaussians(
        path,
        [[1, 2, 3], [np.nan, 1, 2]],
        [[0.1, 0.2, 0.3]] * 2,
        [[2, 0, 0, 0]] * 2,
        [[0.4, 0.5, 0.6]] * 2,
        [0.8, 0.5],
    )
    data = read_ply(path)
    assert count == len(data) == 1
    assert data["x"][0] == 1
    assert np.exp(data["scale_0"][0]) == pytest.approx(0.1)
    assert 1 / (1 + np.exp(-data["opacity"][0])) == pytest.approx(0.8)
    assert data["rot_0"][0] == 1
    assert data["f_dc_1"][0] == pytest.approx(0.5)


def test_empty_prediction_does_not_export_dummy_geometry(tmp_path):
    with pytest.raises(ValueError, match="No finite"):
        export_points(tmp_path / "empty.ply", [[np.nan, 0, 0]], [[1, 0, 0]], [1])
    assert not (tmp_path / "empty.ply").exists()


def test_export_bounds_keep_real_samples(tmp_path):
    points = np.arange(300, dtype=np.float32).reshape(100, 3)
    count = export_points(
        tmp_path / "points.ply", points, np.ones_like(points), np.ones(100), limit=7
    )
    result = read_ply(tmp_path / "points.ply")
    assert count == 7
    assert set(result["x"]).issubset(set(points[:, 0]))


def plane():
    y, x = np.mgrid[:4, :4]
    points = np.stack((x, y, np.ones_like(x)), axis=-1).astype(float)
    return np.stack((points, points))


def test_depth_mesh_is_valid_glb_with_only_observed_geometry(tmp_path):
    points = plane()
    path = tmp_path / "mesh.glb"
    count = export_depth_mesh(
        path, points, np.ones_like(points), np.ones(points.shape[:-1]), stride=1
    )
    data = path.read_bytes()
    magic, version, length, json_size, json_kind = struct.unpack("<5I", data[:20])
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    assert json_kind == 0x4E4F534A
    document = json.loads(data[20 : 20 + json_size])
    assert document["accessors"][2]["count"] == count * 3
    assert count == 36
    assert document["accessors"][0]["min"] == [0, 0, 1]
    assert document["accessors"][0]["max"] == [3, 3, 1]


def test_self_consistency_is_measured_not_accuracy():
    points = plane()
    depth = np.ones((2, 4, 4, 1))
    poses = np.stack((np.eye(4), np.eye(4)))
    intrinsics = np.stack((np.eye(3), np.eye(3)))
    metrics = geometry_diagnostics(points, np.ones((2, 4, 4)), depth, poses, intrinsics)
    assert metrics["medianRelativeDepthResidual"] == 0
    assert metrics["adjacentProjectedOverlap"] == 1
    assert metrics["cameraSpread"] == 0
    assert metrics["scope"] == "prediction-self-consistency-not-ground-truth-accuracy"
    depth[1] *= 2
    metrics = geometry_diagnostics(points, np.ones((2, 4, 4)), depth, poses, intrinsics)
    assert metrics["medianRelativeDepthResidual"] == 0.5


def test_source_selection_spans_entire_sequence_and_flags_static_input(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(10):
        Image.new("RGB", (64, 48), "red").save(inputs / f"source-{index:03d}.png")
    frames, diagnostics = prepare_sources(
        inputs, tmp_path / "frames", max_frames=3, target_size=280
    )
    assert frames.shape == (3, 280, 280, 3)
    assert diagnostics["sourceFileCount"] == 10
    assert diagnostics["selectedFrameCount"] == 3
    assert diagnostics["warnings"]
    assert diagnostics["inputAssessment"] == "heuristics-not-reconstruction-accuracy"


def test_source_requires_multiple_views(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "one.png")
    with pytest.raises(SourceError, match="at least two"):
        prepare_sources(tmp_path, tmp_path / "frames")
