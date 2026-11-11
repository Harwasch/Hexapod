"""Streaming footprints retain every observation and match the former vector calculation."""

import struct
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from density_memory_probe import make
from test_gaussian_budget import model, patch

import gaussian_budget as gb
import sfm
from synthetic_scene import look_at, write_model


def whole_reference(directory: Path, pixel_scale: float) -> tuple[gb.F64, gb.F64]:
    """Pre-streaming implementation, kept only as a small-fixture numerical oracle."""
    colmap = sfm.read_model(directory)
    cameras = {camera.id: camera for camera in colmap.cameras}
    points = sfm.read_points3d(directory / "points3D.bin")
    size = max(image.id for image in colmap.images) + 1
    axis = np.zeros((size, 3), dtype=np.float64)
    offset = np.zeros(size, dtype=np.float64)
    focal = np.full(size, np.nan, dtype=np.float64)
    for image in colmap.images:
        axis[image.id] = image.rotation[2]
        offset[image.id] = image.tvec[2]
        if image.camera_id in cameras:
            focal[image.id] = gb._focal(cameras[image.camera_id]) * pixel_scale
    rows = np.repeat(np.arange(len(points)), np.diff(points.track_offsets))
    ids = points.track[:, 0].astype(np.int64)
    known = ids < size
    ids = np.where(known, ids, 0)
    depth = np.einsum("ij,ij->i", axis[ids], points.xyz[rows]) + offset[ids]
    with np.errstate(divide="ignore", invalid="ignore"):
        each = np.where(known & (depth > 0) & (focal[ids] > 0), depth / focal[ids], np.inf)
    finest = np.full(len(points), np.inf, dtype=np.float64)
    np.minimum.at(finest, rows, each)
    return points.xyz, finest


@pytest.mark.parametrize("chunk", [1, 7, 64, gb.OBSERVATION_CHUNK])
@pytest.mark.parametrize("pixel_scale", [0.5, 1.0, 1.7])
def test_chunks_match_whole_reference_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int, pixel_scale: float
) -> None:
    directory = model(tmp_path / "model", patch((0.0, 0.0, 0.0), 0.2), [2.0, 5.0, 7.0])
    points = sfm.read_points3d(directory / "points3D.bin")
    # Empty tracks, unknown images, views behind the camera, variable track lengths,
    # and a track longer than a chunk. Tilted views also exercise all three dot terms.
    points.xyz[1] = [0.2, 0.1, 20.0]
    points.track[3, 0] = 999
    points.track[4, 0] = 0
    points.track_offsets[1] = 0
    sfm.write_points3d(directory / "points3D.bin", points)
    centres = [np.array(value) for value in [(0.2, 0.7, 2.0), (0.3, -0.4, 5.0), (0.0, 0.0, 7.0)]]
    cameras = [
        (look_at(centre, np.zeros(3), up=np.array([0.0, 1.0, 0.0])), centre) for centre in centres
    ]
    write_model(directory, cameras, focal=1000.0)
    sfm.write_points3d(directory / "points3D.bin", points)
    expected_xyz, expected = whole_reference(directory, pixel_scale)
    monkeypatch.setattr(gb, "OBSERVATION_CHUNK", chunk)
    xyz, finest = gb.finest_footprints(directory, pixel_scale=pixel_scale)
    np.testing.assert_array_equal(xyz, expected_xyz)
    np.testing.assert_array_equal(finest, expected)
    assert gb.surface(xyz, finest) == gb.surface(expected_xyz, expected)


def test_observation_memory_is_bounded(tmp_path: Path) -> None:
    directory = model(
        tmp_path / "model",
        np.zeros((25_000, 3), dtype=np.float64),
        [float(i + 1) for i in range(30)],
    )
    tracemalloc.start()
    try:
        xyz, finest = gb.finest_footprints(directory)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(finest, np.full(25_000, 1.0 / 1000.0))
    assert xyz.shape == (25_000, 3)
    # 750k observations: the previous implementation measured 59.5 MiB here.
    # 16 MiB leaves room for the fixed 65,536-observation buffers and the output.
    assert peak < 16 * 1024**2, f"survey allocated {peak / 1024**2:.1f} MiB"


def test_image_metadata_streams_across_chunk_boundaries(tmp_path: Path) -> None:
    count = 65_539
    make(tmp_path, count, 1)
    path = tmp_path / "images.bin"
    with path.open("r+b") as handle:
        # Last observation has no 3D match, just beyond the first 65,536-row chunk.
        handle.seek(-8, 2)
        handle.write(struct.pack("<Q", 0xFFFFFFFFFFFFFFFF))
    colmap = sfm.read_model(tmp_path)
    assert colmap.images[0].points2d == count
    assert colmap.images[0].points3d == count - 1
    assert colmap.points3d == count
    assert colmap.mean_track_length == 1.0
