"""Shared committed replays: format, movie, analytic geometry and tracking gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from test_capture_bundle import pose_context

import arkit_ingest
import capture_bundle
from scripts.make_fixture_bundles import write_archive

FIXTURES = Path(__file__).resolve().parents[3] / "apps/ios/Fixtures/bundles"


@pytest.mark.parametrize("name,normal", [("yard-small", 22), ("yard-limited", 3)])
def test_shared_bundle(tmp_path: Path, name: str, normal: int) -> None:
    fixture = FIXTURES / name
    source = capture_bundle.read_bundle(fixture / "bundle")
    unpacked = capture_bundle.unpack(fixture / "capture.hexcap", tmp_path / "unpacked")
    assert source.manifest == unpacked.manifest
    assert source.frames == unpacked.frames
    assert source.locations == unpacked.locations
    assert source.location is not None
    assert source.location["lat"] == source.locations[0]["lat"]
    for path in source.directory.iterdir():
        assert path.read_bytes() == (unpacked.directory / path.name).read_bytes()
    # Archive headers are independent of filesystem ownership, modes and mtimes.
    rebuilt = tmp_path / "rebuilt.hexcap"
    write_archive(source.directory, rebuilt)
    assert rebuilt.read_bytes() == (fixture / "capture.hexcap").read_bytes()
    assert len(source.frames) == 24
    assert [f["trackingState"] for f in source.frames[:2]] == ["limited:initializing"] * 2
    assert sum(f["trackingState"] == "normal" for f in source.frames) == normal
    truth = json.loads((fixture / "ground_truth.json").read_text())
    assert truth["synthetic"] and truth["normalFrames"] == normal
    assert truth["counts"] == {"shed": 1, "tree": 1}
    objects = {obj["label"]: obj for obj in truth["objects"]}
    crown = objects["crown"]
    assert truth["treeHeight"] == crown["centre"][1] + crown["extents"][1] / 2
    for probe in truth["probes"]:
        frame = source.frames[probe["frame"]]
        point = np.array(probe["world"])
        # Independently verify the point lies on the labelled analytic surface.
        obj = objects[probe["label"]]
        offset = point - obj["centre"]
        half = np.array(obj["extents"]) / 2
        if probe["label"] == "ground":
            assert abs(point[1]) < 1e-10
            assert (np.abs(point[[0, 2]]) <= half[[0, 2]]).all()
        elif probe["label"] == "shed":
            assert (np.abs(offset) <= half + 1e-10).all()
            assert np.min(np.abs(np.abs(offset) - half)) < 1e-10
        elif probe["label"] == "trunk":
            assert abs(offset[1]) <= half[1] + 1e-10
            assert abs(np.linalg.norm(offset[[0, 2]]) - half[0]) < 1e-10
        else:
            assert abs(np.linalg.norm(offset) - half[0]) < 1e-10
        rotation, translation = capture_bundle.world_to_camera(
            capture_bundle.matrix(frame["transform"], 4)
        )
        camera = rotation @ point + translation
        k = capture_bundle.matrix(frame["intrinsics"], 3).copy()
        k[:2] *= truth["depthIntrinsicsScale"]
        pixel = (k @ camera)[:2] / camera[2]
        np.testing.assert_allclose(pixel, probe["pixel"], atol=1e-10, rtol=0)
        depth_ref = frame["depth"]
        depth = np.fromfile(source.directory / depth_ref["file"], dtype="<f2").reshape(36, 48)
        confidence = np.fromfile(source.directory / depth_ref["confidence"], dtype="u1").reshape(
            36, 48
        )
        u, v = probe["pixel"]
        # Exact correctly rounded binary16, rather than a fitted depth tolerance.
        assert depth[v, u] == np.float16(camera[2])
        assert confidence[v, u] == 2
        np.testing.assert_array_equal(confidence, np.where(np.isfinite(depth), 2, 0))
    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "info",
            "-threads",
            "1",
            "-i",
            str(source.directory / "rgb.mov"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    assert b"Video: hevc" in decoded.stderr
    assert len(decoded.stdout) == 24 * 192 * 144 * 3
    rgb = np.frombuffer(decoded.stdout, dtype="u1").reshape(24, 144, 192, 3)
    # Flat interiors remain recognisable after lossy HEVC and chroma subsampling.
    for probe in truth["probes"]:
        u, v = probe["pixel"]
        r, g, b = map(int, rgb[probe["frame"], v * 4, u * 4])
        if probe["label"] == "crown":
            assert g > r and g > b
        else:
            assert r > g and r > b


def test_limited_replay_fails_before_feature_extraction(tmp_path: Path) -> None:
    source = capture_bundle.read_bundle(FIXTURES / "yard-limited/bundle")
    ctx = pose_context(tmp_path, source.frames)
    with pytest.raises(
        ValueError, match="has 3 of 24 frames with normal ARKit tracking; at least 10"
    ):
        arkit_ingest.pose(ctx)
