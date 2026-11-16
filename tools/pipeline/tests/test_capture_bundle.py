from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import arkit_ingest
import capture_bundle as bundle
import sfm


def frame(index: int = 0) -> dict[str, Any]:
    return {
        "index": index,
        "timestamp": index / 30,
        "transform": np.eye(4).ravel().tolist(),
        "intrinsics": [500, 0, 160, 0, 520, 120, 0, 0, 1],
        "imageWidth": 320,
        "imageHeight": 240,
        "trackingState": "normal",
        "depth": {
            "file": f"depth_{index}.f16",
            "confidence": f"depth_{index}.u8",
            "width": 2,
            "height": 1,
        },
    }


def make_bundle(root: Path, frames: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    root.mkdir()
    frames = frames or [frame()]
    (root / "rgb.mov").write_bytes(b"fake video; parser does not decode")
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    for record in frames:
        (root / record["depth"]["file"]).write_bytes(np.array([1, 2], dtype="<f2").tobytes())
        (root / record["depth"]["confidence"]).write_bytes(bytes([1, 2]))
    manifest = {
        "version": 1,
        "format": "hex-capture",
        "device": "synthetic",
        "startedAt": "2026-01-01T00:00:00Z",
        "video": "rgb.mov",
        "frameLog": "frames.jsonl",
        "frameCount": len(frames),
        "depthFrameCount": len(frames),
        "smoothedDepthFrameCount": 0,
        "files": [{"path": p.name, "bytes": p.stat().st_size} for p in sorted(root.iterdir())],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def archive(root: Path, target: Path) -> None:
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(root.iterdir()):
            tar.add(path, arcname=path.name)


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, -0.7])
def test_pose_projection_inverse_and_quaternion(angle: float) -> None:
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    transform[:3, 3] = [2, -3, 4]
    r, t = bundle.world_to_camera(transform)
    np.testing.assert_allclose(bundle.camera_to_world(r, t), transform, atol=1e-12)
    np.testing.assert_allclose(sfm.quat_to_matrix(bundle.quaternion(r).tolist()), r, atol=1e-12)
    camera_point = np.array([0.4, 0.3, -2])
    world_point = transform[:3, :3] @ camera_point + transform[:3, 3]
    colmap_point = r @ world_point + t
    np.testing.assert_allclose(colmap_point, [0.4, -0.3, 2], atol=1e-12)
    ar_pixel = [
        500 * camera_point[0] / -camera_point[2] + 160,
        520 * -camera_point[1] / -camera_point[2] + 120,
    ]
    colmap_pixel = [
        500 * colmap_point[0] / colmap_point[2] + 160,
        520 * colmap_point[1] / colmap_point[2] + 120,
    ]
    np.testing.assert_allclose(ar_pixel, colmap_pixel, atol=1e-10)
    np.testing.assert_allclose(-r.T @ t, transform[:3, 3], atol=1e-12)


def test_axis_flip_and_reject_reflection() -> None:
    r, t = bundle.world_to_camera(np.eye(4))
    np.testing.assert_array_equal(r, np.diag([1, -1, -1]))
    np.testing.assert_array_equal(t, np.zeros(3))
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        bundle.world_to_camera(reflection)


def test_bundle_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = make_bundle(source, [frame(0), frame(1)])
    tar = tmp_path / "capture.hexcap"
    archive(source, tar)
    parsed = bundle.unpack(tar, tmp_path / "parsed")
    assert parsed.manifest == manifest
    assert parsed.frames == [frame(0), frame(1)]
    assert (parsed.directory / "depth_0.f16").read_bytes() == np.array(
        [1, 2], dtype="<f2"
    ).tobytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "count",
        "size",
        "path",
        "intrinsics",
        "index",
        "timestamp",
        "extra",
        "depth",
        "nan",
    ],
)
def test_malformed_bundle(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source"
    manifest = make_bundle(source)
    if mutation == "version":
        manifest["version"] = 2
    elif mutation == "count":
        manifest["frameCount"] = 2
    elif mutation == "size":
        manifest["files"][0]["bytes"] += 1
    elif mutation == "path":
        manifest["files"][0]["path"] = "../outside"
    elif mutation == "extra":
        (source / "undeclared").write_bytes(b"extra")
    else:
        record = frame()
        if mutation == "intrinsics":
            record["intrinsics"] = [0] * 9
        elif mutation == "index":
            record["index"] = 1
        elif mutation == "timestamp":
            record["timestamp"] = 1
        elif mutation == "nan":
            record["transform"][0] = float("nan")
        elif mutation == "depth":
            record["depth"]["width"] = 3
        (source / "frames.jsonl").write_text(json.dumps(record) + "\n")
        for entry in manifest["files"]:
            entry["bytes"] = (source / entry["path"]).stat().st_size
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.read_bundle(source)


@pytest.mark.parametrize(
    "kind", ["traversal", "absolute", "symlink", "hardlink", "duplicate", "directory"]
)
def test_archive_rejects_unsafe_members(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "bad.hexcap"
    with tarfile.open(path, "w") as tar:
        entry = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape"}.get(kind, "file"))
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../escape"
        elif kind == "hardlink":
            entry.type = tarfile.LNKTYPE
            entry.linkname = "../escape"
        elif kind == "directory":
            entry.type = tarfile.DIRTYPE
        tar.addfile(entry, io.BytesIO())
        if kind == "duplicate":
            tar.addfile(entry, io.BytesIO())
    with pytest.raises(ValueError):
        bundle.unpack(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_seed_uses_database_image_ids_and_per_frame_calibration(tmp_path: Path) -> None:
    db_path = tmp_path / "colmap.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        db.execute(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, "
            "width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)"
        )
        db.execute("INSERT INTO images VALUES (17, 'frame_00000000.png', 1)")
        db.execute("INSERT INTO images VALUES (3, 'frame_00000001.png', 1)")
    records = [frame(0), frame(1)]
    records[1]["intrinsics"][0] = 600
    records[1]["transform"][3] = 2
    model = tmp_path / "model"
    arkit_ingest.seed_model(db_path, records, model)
    lines = (model / "images.txt").read_text().splitlines()
    assert lines[0].startswith("17 ") and lines[2].startswith("3 ")
    assert lines[1] == lines[3] == ""
    assert lines[2].split()[5:8] == ["-2.0", "0.0", "0.0"]
    with sqlite3.connect(db_path) as db:
        params = db.execute("SELECT params FROM cameras WHERE camera_id=3").fetchone()[0]
    np.testing.assert_array_equal(np.frombuffer(params, dtype="<f8"), [600, 520, 160, 120])


def test_normalize_keeps_every_frame_at_native_size(tmp_path: Path) -> None:
    from test_normalize import make_clip

    from conftest import make_recipe
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    source = tmp_path / "source"
    make_bundle(source, [frame(i) for i in range(4)])
    (source / "rgb.mov").unlink()
    make_clip(source / "rgb.mov", frames=4, fps=30)
    # make_clip creates stills beside its output; only archive the declared payload.
    import shutil

    shutil.rmtree(source / "stills")
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        entry["bytes"] = (source / entry["path"]).stat().st_size
    (source / "manifest.json").write_text(json.dumps(manifest))
    work = Workdir.create(tmp_path / "run")
    upload = work.input_path("upload")
    upload.mkdir()
    archive(source, upload / "capture.hexcap")
    recipe = make_recipe(
        [{"id": "normalize", "impl": "ffmpeg_frames", "params": {"capture_bundle": True}}],
        inputs=["upload"],
    )
    execute(recipe, work, RunnerSet(cpu=LocalRunner()))
    images = list((work.stage_dir("normalize") / "out" / "frames").glob("*.png"))
    assert len(images) == 4


def test_placement_keeps_arkit_metres_and_gravity(tmp_path: Path) -> None:
    from typing import cast

    import stages
    from contracts import StageContext

    class Defaults:
        def param(self, name: str, default: object = None) -> object:
            return default

    (tmp_path / "poses.json").write_text(json.dumps({"source": "hex-capture-v1"}))
    placed = stages._frame_from_poses(cast(StageContext, Defaults()), tmp_path, estimate_scale=True)
    assert placed["scale"] == 1.0
    assert placed["scaleSource"] == "arkit"
    assert placed["scaleEstimate"] is None
    np.testing.assert_allclose(np.asarray(placed["rotation"]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
