"""What `normalize` now says about a capture: three synthetic uploads, described.

Builds, in `--out`:

* (a) six DJI M3E-style JPEGs at 3000x2250 -- five RTK-fixed (`RtkFlag` 50), one single
  (16), so a `drone`, not the survey tier, which needs every frame fixed; EXIF with no
  offset; GPS with a horizontal error; two exposure times;
* (b) four iPhone-style JPEGs with `OffsetTimeOriginal` and `GPSHPositioningError`;
* (c) a 10-bit HLG HEVC `.mov` with the iPhone's QuickTime keys: ISO 6709 location,
  horizontal accuracy 4.766775 m, creation date `2025-06-14T10:41:02+0100`, make, model;

runs `normalize` (`ffmpeg_frames`, the recipe's frame stage) on each under LocalRunner,
and prints the platform detection table (`tests/test_capture_platform.py`'s cases, then
the three runs), and for each run `frames_meta.json`'s platform, focal, time range and
first two frame records, and `source_meta.json`'s location, HDR decision and exposure.

**Every input is synthetic** (`tests/drone_frames.py` says from which sources the DJI
fields come). This shows the plumbing; it is not evidence that a real Mavic 3 Enterprise
writes these fields this way.

    cd tools/pipeline && PYTHONPATH=. uv run --no-sync python experiments/ingest_demo.py \\
        --out <dir> | tee <dir>/report.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tests"))

import imageio_ffmpeg  # noqa: E402

import capture_platform  # noqa: E402
from drone_frames import dji_m3e_photo, iphone_photo  # noqa: E402

__all__ = ["build_uploads", "main"]


def _hlg_iphone_clip(path: Path) -> Path:
    from test_normalize import make_hdr_clip

    raw = make_hdr_clip(path.parent / "raw" / "raw.mov", frames=20)
    subprocess.run(  # noqa: S603 - a fixed argv with the wheel's ffmpeg
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


def build_uploads(out: Path) -> dict[str, Path]:
    """The three uploads, each in its own folder under `out`."""
    drone = out / "uploads" / "drone"
    for index in range(6):
        dji_m3e_photo(
            drone / f"DJI_{index:04d}.JPG",
            index,
            rtk_flag=16 if index == 5 else 50,
            size=(3000, 2250),
            exposure_s=1 / 1000 if index < 3 else 1 / 500,
            h_error=0.03,
        )
    phone = out / "uploads" / "phone-photos"
    for index in range(4):
        iphone_photo(phone / f"IMG_{index:04d}.JPG", index, size=(1600, 1200))
    clip = out / "uploads" / "phone-video"
    clip.mkdir(parents=True, exist_ok=True)
    _hlg_iphone_clip(clip / "IMG_4417.MOV")
    return {"drone": drone, "phone-photos": phone, "phone-video": clip}


PARAMS: dict[str, dict[str, object]] = {
    "drone": {"select": "all", "keep": 6, "max_side": 1600},
    "phone-photos": {"select": "all", "keep": 4, "max_side": 800},
    "phone-video": {"select": "all", "fps": 4, "keep": 400},
}


def _run(out: Path, name: str, upload: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from test_normalize import run_normalize

    workdir = run_normalize(out / "runs" / name, PARAMS[name], upload)
    folder = workdir.out_dir("normalize")
    frames = json.loads((folder / "frames_meta.json").read_text(encoding="utf-8"))
    source = json.loads((folder / "source_meta.json").read_text(encoding="utf-8"))
    return frames, source


def _table(runs: dict[str, dict[str, Any]]) -> list[str]:
    from test_capture_platform import TABLE

    lines = [
        "PLATFORM DETECTION TABLE (tests/test_capture_platform.py, then the three runs)",
        f"{'case':<34} {'value':<12} {'source':<16} {'basis':<9} evidence",
    ]
    for case, fmt, records, video, requested, hint, *_expected in TABLE:
        found = capture_platform.detect(
            fmt=fmt, records=records, video=video, requested=requested, hint=hint
        )
        lines.append(
            f"{case:<34} {found.value:<12} {found.source:<16} {found.platform_evidence:<9} "
            f"{found.evidence}"
        )
    for name, frames in runs.items():
        platform = frames["platform"]
        lines.append(
            f"{'run: ' + name:<34} {platform['value']:<12} {platform['source']:<16} "
            f"{platform['platformEvidence']:<9} {platform['evidence']}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    uploads = build_uploads(out)
    results = {name: _run(out, name, upload) for name, upload in uploads.items()}
    lines = _table({name: frames for name, (frames, _source) in results.items()})
    for name, (frames, source) in results.items():
        lines += [
            "",
            f"=== {name}: {len(frames['frames'])} kept frame(s), format {frames['format']} ===",
            "frames_meta.platform  " + json.dumps(frames["platform"], sort_keys=True),
            "frames_meta.focal     " + json.dumps(frames["focal"], sort_keys=True),
            "frames_meta.timeRange " + json.dumps(frames["timeRange"], sort_keys=True),
            "source_meta.capturedAt / capturedAtUtc  "
            + json.dumps([source["capturedAt"], source["capturedAtUtc"]]),
            "source_meta.location  " + json.dumps(source["location"], sort_keys=True),
            "source_meta.video.hdr "
            + json.dumps((source["video"] or {}).get("hdr"), sort_keys=True),
            "source_meta.exposure  " + json.dumps(source["exposure"], sort_keys=True),
        ]
        for record in frames["frames"][:2]:
            lines.append(
                f"frames[{record['name']}] " + json.dumps(record, indent=1, sort_keys=True)
            )
    print("\n".join(lines))  # noqa: T201 - the report is the output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
