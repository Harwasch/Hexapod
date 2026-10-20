"""`normalize` describes every kept frame from its original: `frames_meta.json`, the
platform, and the capture's UTC time in `source_meta.json`.

Drone and phone photos are synthetic (`drone_frames.py`); the iPhone-like clip is encoded
here with the wheel's ffmpeg and given Apple's QuickTime keys by a `-c copy` re-mux.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import pytest
from test_normalize import make_clip, run_normalize

from drone_frames import dji_m3e_photo, iphone_photo
from workdir import Workdir


def _out(workdir: Workdir, name: str) -> dict[str, Any]:
    document = json.loads((workdir.out_dir("normalize") / name).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def iphone_clip(path: Path, *, frames: int = 10, fps: int = 10) -> Path:
    """A clip with the QuickTime keys an iPhone writes, re-muxed without re-encoding."""
    raw = make_clip(path.parent / "raw" / "raw.mov", frames=frames, fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-i", str(raw),
         "-c", "copy", "-movflags", "use_metadata_tags",
         "-metadata", "com.apple.quicktime.location.ISO6709=+51.5074-000.1278+031.482/",
         "-metadata", "com.apple.quicktime.location.accuracy.horizontal=4.766775",
         "-metadata", "com.apple.quicktime.creationdate=2025-06-14T10:41:02+0100",
         "-metadata", "com.apple.quicktime.make=Apple",
         "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
         str(path)],
        capture_output=True, check=True,
    )  # fmt: skip
    return path


def test_drone_photos_are_described_from_their_originals_before_shrinking(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "upload"
    for index in range(3):
        dji_m3e_photo(upload / f"DJI_{index:04d}.JPG", index, size=(3000, 2250))

    workdir = run_normalize(tmp_path, {"select": "all", "keep": 3, "max_side": 1600}, upload)

    frames = _out(workdir, "frames_meta.json")
    assert frames["schema"] == "frames_meta/1"
    assert frames["format"] == "images"
    assert [r["name"] for r in frames["frames"]] == [f"frame_{i:04d}.jpg" for i in range(3)]
    for record in frames["frames"]:
        assert record["originalSize"] == [3000, 2250]
        assert record["dji"]["rtk"]["flag"] == 50
        assert record["focal"]["pxFrom"] == "dji-calibrated"
        assert record["focal"]["px"] == pytest.approx(3713.29 * 3000 / 5280 * 1600 / 3000)
        assert record["gps"]["altitudeDatum"] == "msl-unverified"
        assert record["mask"] == f"{record['name']}.png"
    # The shrunk frame kept its EXIF but not the XMP: the original had to be read.
    shrunk = (workdir.out_dir("normalize") / "frames" / "frame_0000.jpg").read_bytes()
    assert b"drone-dji" not in shrunk
    assert b"drone-dji" in (upload / "DJI_0000.JPG").read_bytes()
    source = _out(workdir, "source_meta.json")
    assert source["platform"]["value"] == "rtk-drone"
    assert frames["platform"] == source["platform"]
    assert source["focal"]["px"] == pytest.approx(3713.29 * 1600 / 5280, abs=1e-4)
    # DJI EXIF carries no offset, so there is no UTC time and nothing claims one.
    assert source["capturedAtUtc"] is None
    assert source["capturedAt"] == "2026-10-01T10:20:30"
    assert frames["timeRange"]["firstUtc"] is None


def test_records_are_keyed_by_the_renumbered_frame_name(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    for index in range(6):
        iphone_photo(upload / f"IMG_{index:04d}.JPG", index, size=(320, 240))

    workdir = run_normalize(tmp_path, {"keep": 3, "select": "sharpness-windowed"}, upload)

    frames = _out(workdir, "frames_meta.json")["frames"]
    assert [r["name"] for r in frames] == ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
    originals = [r["original"] for r in frames]
    assert originals == sorted(originals)
    assert len(set(originals)) == 3
    # Each record is its own original's: the time in it is that photo's.
    for record in frames:
        index = int(record["original"][4:8])
        assert record["time"]["utc"] == f"2025-06-14T09:41:{index:02d}.123Z"


def test_an_iphone_clip_gives_each_frame_its_time_in_the_clip(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    iphone_clip(upload / "IMG_4417.MOV")

    workdir = run_normalize(tmp_path, {"select": "all", "fps": 5, "keep": 400}, upload)

    frames = _out(workdir, "frames_meta.json")
    records = frames["frames"]
    assert len(records) >= 4
    for record in records:
        assert record["offsetS"] == pytest.approx(record["candidateIndex"] / 5)
        assert record["original"] == "IMG_4417.MOV"
        assert record["gps"] is None
    assert records[1]["time"] == {
        "local": "2025-06-14T10:41:02.200",
        "offset": "+01:00",
        "utc": "2025-06-14T09:41:02.200Z",
        "source": "quicktime-creationdate",
    }
    source = _out(workdir, "source_meta.json")
    assert source["location"]["accuracyM"] == pytest.approx(4.766775)
    assert source["platform"]["value"] == "phone-video"
    assert source["platform"]["source"] == "phone-quicktime"
    assert source["capturedAtUtc"] == "2025-06-14T09:41:02.000Z"
    assert frames["timeRange"]["source"] == "quicktime-creationdate"


def test_the_platform_param_and_the_phone_hint_reach_the_record(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    make_clip(upload / "clip.mp4", frames=4, fps=10)

    hinted = run_normalize(tmp_path / "a", {"select": "all", "platform_hint": "phone"}, upload)
    assert _out(hinted, "source_meta.json")["platform"]["value"] == "phone-video"
    assert _out(hinted, "source_meta.json")["platform"]["source"] == "phone-key"

    asked = run_normalize(tmp_path / "b", {"select": "all", "platform": "drone"}, upload)
    assert _out(asked, "frames_meta.json")["platform"] == {
        "value": "drone",
        "source": "param",
        "evidence": "asked for by the recipe or the run",
    }
    bare = run_normalize(tmp_path / "c", {"select": "all"}, upload)
    assert _out(bare, "source_meta.json")["platform"]["value"] == "unknown"
    step = json.loads((bare.stage_dir("normalize") / "step.json").read_text(encoding="utf-8"))
    assert step["metrics"]["platform"] == "unknown"


def test_a_platform_outside_the_vocabulary_fails_the_stage(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    make_clip(upload / "clip.mp4", frames=4, fps=10)

    with pytest.raises(Exception, match="is not one of rtk-drone"):
        run_normalize(tmp_path, {"select": "all", "platform": "satellite"}, upload)


def test_an_unreadable_original_is_a_thin_record_not_a_failure(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    iphone_photo(upload / "IMG_0000.JPG", 0, size=(320, 240))
    (upload / "IMG_0001.JPG").write_bytes(b"\xff\xd8 not a jpeg")
    iphone_photo(upload / "IMG_0002.JPG", 2, size=(320, 240))

    workdir = run_normalize(tmp_path, {"select": "all", "keep": 3}, upload)

    records = _out(workdir, "frames_meta.json")["frames"]
    assert [r["original"] for r in records] == ["IMG_0000.JPG", "IMG_0001.JPG", "IMG_0002.JPG"]
    assert records[1]["unreadable"] is not None
    assert records[1]["camera"] == {"make": None, "model": None, "lens": None}
    assert records[0]["unreadable"] is None and records[2]["camera"]["make"] == "Apple"
