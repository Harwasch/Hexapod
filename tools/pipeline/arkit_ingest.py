"""Bundle-only branches of the existing normalize and pose stages."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image

import arkit_refine
import capture_bundle as bundle
import colmap4
import frame_meta
import sfm
from contracts import StageContext, StageOutcome
from global_sfm import KEEP

MIN_TRACKED_FRAMES = 10


def normalize(ctx: StageContext) -> StageOutcome:
    upload = ctx.input("upload")
    archives = list(upload.iterdir()) if upload.is_dir() else [upload]
    if len(archives) != 1 or archives[0].suffix != ".hexcap":
        raise ValueError("bundle input must be exactly one .hexcap archive")
    directory = ctx.work_dir / "capture-bundle"
    if directory.exists():
        shutil.rmtree(directory)
    capture = bundle.unpack(archives[0], directory)
    frames = ctx.output("frames")
    # No rotation, resizing, fps filter, deduplication or quality-based selection.
    ctx.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-threads",
            "2",
            "-noautorotate",
            "-i",
            str(directory / "rgb.mov"),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-start_number",
            "0",
            "-threads",
            "2",
            str(frames / "frame_%08d.png"),
        ]
    )
    images = sorted(frames.glob("*.png"))
    if len(images) != len(capture.frames):
        raise ValueError("decoded video count differs from the frame log")
    records = []
    for path, record in zip(images, capture.frames, strict=True):
        with Image.open(path) as image:
            if image.size != (record["imageWidth"], record["imageHeight"]):
                raise ValueError("decoded RGB dimensions differ from the frame log")
        records.append(
            frame_meta.FrameRecord(
                name=path.name,
                original="rgb.mov",
                original_size=(record["imageWidth"], record["imageHeight"]),
                frame_size=(record["imageWidth"], record["imageHeight"]),
            ).to_dict()
        )
    platform = {
        "value": "phone-video",
        "source": "capture-bundle",
        "evidence": "ARKit",
        "platformEvidence": "claimed",
    }
    source = {
        "format": "hex-capture-v1",
        "frames": len(images),
        "device": capture.manifest["device"],
        "capturedAt": capture.manifest["startedAt"],
        "platform": platform,
        "location": capture.location,
        "selection": "all",
        "sensor": "ARKit LiDAR",
    }
    ctx.output("source_meta.json").write_text(json.dumps(source) + "\n")
    ctx.output("frames_meta.json").write_text(
        json.dumps(
            {
                "schema": "frames_meta/1",
                "format": "hex-capture-v1",
                "platform": platform,
                "frames": records,
                "arkit": capture.frames,
                "locations": capture.locations,
            }
        )
        + "\n"
    )
    return StageOutcome(
        metrics={"frames": len(images), "select": "all"}, summary="All bundle frames"
    )


def seed_model(database: Path, frames: list[dict[str, Any]], output: Path) -> None:
    """Keep COLMAP's assigned image IDs; give every frame its measured calibration."""
    output.mkdir(parents=True, exist_ok=True)
    cameras, images = [], []
    with sqlite3.connect(database) as db:
        rows = db.execute("SELECT image_id, name FROM images ORDER BY name").fetchall()
        if [row[1] for row in rows] != [f"frame_{f['index']:08d}.png" for f in frames]:
            raise ValueError("COLMAP database images do not match the frame log")
        for (image_id, name), frame in zip(rows, frames, strict=True):
            camera_id = image_id
            k = bundle.matrix(frame["intrinsics"], 3)
            params = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype="<f8")
            w, h = frame["imageWidth"], frame["imageHeight"]
            db.execute(
                "INSERT OR REPLACE INTO cameras "
                "(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?,1,?,?,?,1)",
                (camera_id, w, h, params.tobytes()),
            )
            db.execute("UPDATE images SET camera_id=? WHERE image_id=?", (camera_id, image_id))
            rotation, translation = bundle.world_to_camera(bundle.matrix(frame["transform"], 4))
            pose = np.concatenate([bundle.quaternion(rotation), translation])
            cameras.append(f"{camera_id} PINHOLE {w} {h} " + " ".join(map(str, params)))
            images.append(f"{image_id} " + " ".join(map(str, pose)) + f" {camera_id} {name}\n")
    (output / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (output / "images.txt").write_text("\n".join(images) + "\n")
    (output / "points3D.txt").write_text("")


def pose(ctx: StageContext) -> StageOutcome:
    metadata = json.loads(ctx.input("frames_meta.json").read_text())
    if metadata.get("format") != "hex-capture-v1":
        raise ValueError("ARKit pose input is not a capture bundle")
    records = metadata["arkit"]
    previous = -1.0
    for i, record in enumerate(records):
        previous = bundle.validate_frame(record, i, previous)
    tracked = [record for record in records if record["trackingState"] == "normal"]
    excluded = dict(Counter(r["trackingState"] for r in records if r["trackingState"] != "normal"))
    minimum = max(2, int(ctx.param("min_tracked_frames", MIN_TRACKED_FRAMES)))
    if len(tracked) < minimum:
        raise ValueError(
            f"capture bundle has {len(tracked)} of {len(records)} frames with normal "
            f"ARKit tracking; at least {minimum} are needed"
        )
    tracking = {
        "total": len(records),
        "normal": len(tracked),
        "excluded": len(records) - len(tracked),
        "excludedByState": excluded,
    }
    image_list = ctx.work_dir / "arkit-images.txt"
    image_list.write_text("".join(f"frame_{r['index']:08d}.png\n" for r in tracked))
    frames = ctx.input("frames")
    database = ctx.work_dir / "arkit.db"
    database.unlink(missing_ok=True)
    seed = ctx.work_dir / "arkit-seed"
    out = ctx.output("poses")
    argv = sfm.feature_extractor_argv(
        database,
        frames,
        camera_model="PINHOLE",
        single_camera=False,
        max_features=int(ctx.param("max_features", 4096)),
        max_image_size=int(ctx.param("max_image_size", 1600)),
        first_octave=0,
        num_threads=4,
    )
    argv += ["--image_list_path", str(image_list)]
    ctx.run(argv)
    seed_model(database, tracked, seed)
    # Known metric poses remove the need for incremental/backbone pose estimation.
    # Only normal-tracking images entered the database, so matching cannot include others.
    ctx.run(
        sfm.matcher_argv(
            database,
            matcher="sequential",
            overlap=10,
            num_threads=4,
            vocab_tree=sfm.vocab_tree_path(),
        )
    )
    argv = sfm.point_triangulator_argv(database, frames, seed, out, 4)
    argv += ["--Mapper.fix_existing_images", "1"]
    ctx.run(argv)
    model = sfm.read_model(out)
    if model.registered != len(tracked) or model.points3d == 0:
        raise ValueError("ARKit triangulation did not retain every pose with usable points")
    # Always check the immutable fixed baseline before attempting free-pose BA.
    arkit_refine.verify(model, tracked, None)
    refinement = arkit_refine.empty_report("disabled")
    if bool(ctx.param("arkit_refine", True)):
        work = ctx.work_dir / "arkit-refine"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        candidate, report_path = work / "model", work / "report.json"
        command = [
            colmap4.python(),
            str(Path(arkit_refine.__file__).resolve()),
            "--source",
            str(out),
            "--database",
            str(database),
            "--frames",
            str(frames),
            "--output",
            str(candidate),
            "--report",
            str(report_path),
        ]
        if not bool(ctx.param("arkit_shared_lens", True)):
            command.append("--no-lens")
        try:
            ctx.run(command)
            refinement = json.loads(report_path.read_text())
            if refinement.get("status") == "accepted":
                refined = sfm.read_model(candidate)
                arkit_refine.verify(refined, tracked, refinement)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            refinement.update(status="fallback", fallbackReason=str(error))
        if refinement["status"] == "accepted":
            # An IO failure during publication fails the stage; never label a partly
            # copied candidate as the fixed fallback.
            for filename in KEEP:
                shutil.copy2(candidate / filename, out / filename)
            model = sfm.read_model(out)
        else:
            arkit_refine.verify(model, tracked, None)
    if refinement.get("reprojectionBeforePx") is None:
        refinement["reprojectionBeforePx"] = model.mean_reprojection_error
    (out / "poses.json").write_text(
        json.dumps(
            {
                "source": "hex-capture-v1",
                "registered": model.registered,
                "tracking": tracking,
                "refinement": refinement,
                "points3D": model.points3d,
                "cameras": [camera.to_dict() for camera in model.cameras],
                "images": [image.to_dict() for image in model.images],
                "scale": {
                    "metric": True,
                    "source": "arkit",
                    "note": "ARKit metres; device accuracy unverified",
                },
                "upEstimate": {"up": [0, 1, 0], "source": "arkit-gravity"},
            }
        )
        + "\n"
    )
    return StageOutcome(
        metrics={
            "registered": model.registered,
            "points3D": model.points3d,
            "excluded": len(records) - len(tracked),
            "excludedByState": json.dumps(excluded, sort_keys=True),
            "refinement": json.dumps(
                {key: value for key, value in refinement.items() if key != "corrections"},
                sort_keys=True,
            ),
        },
        summary="ARKit metric poses; refinement " + refinement["status"],
    )
