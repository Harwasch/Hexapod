"""4K ARKit bundles and the optional exposure/format records (app 2026-11+)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image
from test_capture_bundle import archive, frame, location_fixes, make_bundle

import arkit_ingest
import arkit_refine
import capture_bundle as bundle
import sfm

W4K, H4K = 3840, 2160


def frame_4k(index: int) -> dict[str, Any]:
    record = frame(index)
    record.update(
        imageWidth=W4K,
        imageHeight=H4K,
        intrinsics=[2830.0, 0, 1920.0, 0, 2830.0, 1080.0, 0, 0, 1],
        exposureDurationS=1 / 250,
        exposureOffsetEV=-0.3,
        iso=320.0,
        whiteBalanceGains=[1.9, 1.0, 1.6],
    )
    return record


def settings(**overrides: Any) -> dict[str, Any]:
    return {
        "videoFormat": {"kind": "4k", "width": W4K, "height": H4K, "framesPerSecond": 30},
        "exposureControl": {
            "mode": "custom",
            "whiteBalance": "locked",
            "durationS": 1 / 250,
            "iso": 320.0,
        },
        **overrides,
    }


def with_settings(root: Path, extra: dict[str, Any]) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(extra)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_frame_suffix_keeps_1440p_lossless() -> None:
    assert bundle.frame_suffix(1920, 1440) == ".png"
    assert bundle.frame_suffix(1440, 1920) == ".png"
    assert bundle.frame_suffix(W4K, H4K) == ".jpg"
    assert bundle.frame_name(7, ".jpg") == "frame_00000007.jpg"


def test_exposure_and_format_records_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    records = [frame_4k(0), frame_4k(1)]
    make_bundle(root, records)
    with_settings(root, settings())
    capture = bundle.read_bundle(root)
    assert capture.frames[1]["iso"] == 320.0


def test_default_format_records_the_4k_probe_evidence(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    make_bundle(root, [frame(0)])
    evidence = {
        "reason": "depthAspect",
        "imageWidth": W4K,
        "imageHeight": H4K,
        "depthWidth": 256,
        "depthHeight": 192,
    }
    video_format = {"kind": "default", "width": 320, "height": 240, "framesPerSecond": 60}
    control = {"mode": "custom", "whiteBalance": "locked", "durationS": 1 / 120, "iso": 1600}
    with_settings(
        root,
        {
            "videoFormat": {**video_format, "fallback": evidence},
            "exposureControl": {**control, "shortfallEV": 0.0},
        },
    )
    bundle.read_bundle(root)


def test_bundles_without_new_fields_stay_valid(tmp_path: Path) -> None:
    make_bundle(tmp_path / "bundle", [frame(0), frame(1)])
    bundle.read_bundle(tmp_path / "bundle")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("exposureDurationS", 0, "exposureDurationS"),
        ("exposureDurationS", 2.0, "exposureDurationS"),
        ("exposureOffsetEV", float("nan"), "exposureOffsetEV"),
        ("iso", -1, "iso"),
        ("iso", True, "iso"),
        ("whiteBalanceGains", [1, 1], "whiteBalanceGains"),
        ("whiteBalanceGains", [1, 0, 1], "whiteBalanceGains"),
    ],
)
def test_invalid_exposure_fields_are_rejected(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    record = frame_4k(0)
    record[field] = value
    root = tmp_path / "bundle"
    make_bundle(root, [record])
    with pytest.raises(ValueError, match=message):
        bundle.read_bundle(root)


@pytest.mark.parametrize(
    "extra,message",
    [
        (
            {"videoFormat": {"kind": "8k", "width": W4K, "height": H4K, "framesPerSecond": 30}},
            "kind",
        ),
        (
            {"videoFormat": {"kind": "4k", "width": 1920, "height": 1440, "framesPerSecond": 30}},
            "differs",
        ),
        ({"videoFormat": {"kind": "4k", "width": W4K, "height": H4K}}, "framesPerSecond"),
        ({"exposureControl": {"mode": "manual", "whiteBalance": "locked"}}, "known mode"),
        (
            {"exposureControl": {"mode": "custom", "whiteBalance": "locked", "shortfallEV": -1}},
            "shortfallEV",
        ),
        (
            {
                "videoFormat": {
                    "kind": "4k",
                    "width": W4K,
                    "height": H4K,
                    "framesPerSecond": 30,
                    "fallback": {"reason": "tooDark", "imageWidth": W4K, "imageHeight": H4K},
                }
            },
            "fallback.reason",
        ),
        (
            {"exposureControl": {"mode": "custom", "whiteBalance": "locked", "durationS": 3}},
            "durationS",
        ),
    ],
)
def test_invalid_capture_settings_are_rejected(
    tmp_path: Path, extra: dict[str, Any], message: str
) -> None:
    root = tmp_path / "bundle"
    make_bundle(root, [frame_4k(0)])
    with_settings(root, extra)
    with pytest.raises(ValueError, match=message):
        bundle.read_bundle(root)


def encode_4k(path: Path, frames: int) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={W4K}x{H4K}:rate=30",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_normalize_writes_4k_bundle_frames_as_jpeg(tmp_path: Path) -> None:
    from conftest import make_recipe
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    records = [frame_4k(i) for i in range(3)]
    source = tmp_path / "source"
    make_bundle(source, records, locations=location_fixes())
    encode_4k(source / "rgb.mov", len(records))
    manifest = json.loads((source / "manifest.json").read_text())
    for entry in manifest["files"]:
        entry["bytes"] = (source / entry["path"]).stat().st_size
    manifest.update(settings())
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
    out = work.stage_dir("normalize") / "out"
    jpegs = sorted((out / "frames").glob("*.jpg"))
    assert [p.name for p in jpegs] == [bundle.frame_name(i, ".jpg") for i in range(3)]
    assert not list((out / "frames").glob("*.png"))
    with Image.open(jpegs[0]) as image:
        assert image.size == (W4K, H4K)
        assert image.format == "JPEG"
    meta = json.loads((out / "frames_meta.json").read_text())
    assert [f["name"] for f in meta["frames"]] == [p.name for p in jpegs]
    assert meta["arkit"][0]["exposureDurationS"] == 1 / 250


def test_seed_and_verify_follow_the_extracted_suffix(tmp_path: Path) -> None:
    db_path = tmp_path / "colmap.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        db.execute(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, "
            "width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)"
        )
        db.execute("INSERT INTO images VALUES (1, 'frame_00000000.jpg', 1)")
    records = [frame_4k(0)]
    arkit_ingest.seed_model(db_path, records, tmp_path / "model", ".jpg")
    assert (tmp_path / "model" / "images.txt").read_text().split()[9] == "frame_00000000.jpg"
    with pytest.raises(ValueError, match="do not match"):
        arkit_ingest.seed_model(db_path, records, tmp_path / "png", ".png")
    r, t = bundle.world_to_camera(bundle.matrix(records[0]["transform"], 4))
    image = sfm.Image(
        id=1,
        qvec=tuple(bundle.quaternion(r)),
        tvec=tuple(t),
        camera_id=1,
        name="frame_00000000.jpg",
        points2d=0,
        points3d=0,
    )
    model = sfm.Model(cameras=(), images=(image,), points3d=1, mean_track_length=0.0)
    arkit_refine.verify(model, records, None, suffix=".jpg")
    with pytest.raises(KeyError):
        arkit_refine.verify(model, records, None)


def test_4k_frames_fit_the_worker_disk_as_jpeg(tmp_path: Path) -> None:
    """A recorded-photo stand-in: JPEG keeps a 720-frame 4K scan far inside 20 GB."""
    rng = np.random.default_rng(0)
    # Smooth gradients plus moderate texture: a rough proxy for a natural frame.
    y, x = np.mgrid[0:H4K, 0:W4K]
    base = np.stack([x / W4K, y / H4K, (x + y) / (W4K + H4K)], axis=-1) * 200
    pixels = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    png, jpg = tmp_path / "f.png", tmp_path / "f.jpg"
    Image.fromarray(pixels).save(png)
    Image.fromarray(pixels).save(jpg, quality=95)
    assert jpg.stat().st_size * 720 < 6e9 < png.stat().st_size * 720
