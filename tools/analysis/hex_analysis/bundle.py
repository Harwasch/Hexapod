"""Bounded bundle access for the remote registry, reusing capture validation."""

import base64
import io
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hex_analysis.lifting import F64
from hex_analysis.registry import Frame


def read_frames(
    directory: Path, records: list[dict[str, Any]], refs: list[dict[str, Any]]
) -> list[Frame]:
    if len(refs) > 8:
        raise ValueError("at most eight keyframes per analysis")
    result: list[Frame] = []
    for ref in refs:
        index = ref["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(records)
            or ref["pose"] != f"frames.jsonl#{index}"
        ):
            raise ValueError("keyframe pose/index mismatch")
        record = records[index]
        width, height = record["imageWidth"], record["imageHeight"]
        if not (0 < width <= 8192 and 0 < height <= 8192 and width * height <= 48_000_000):
            raise ValueError("keyframe dimensions exceed analysis limits")
        depth_record = record.get("depth")
        if ref.get("depth") is not None:
            if depth_record is None or ref["depth"] != depth_record["file"]:
                raise ValueError("depth reference does not belong to the frame")
            if depth_record["height"] * depth_record["width"] > 4_000_000:
                raise ValueError("depth dimensions exceed analysis limits")
        # v1 contains video, not independently uploaded stills. Only this deterministic
        # decoded frame name is accepted, so references cannot escape the validated bundle.
        if ref["image"] != f"frame-{index:06d}.jpg":
            raise ValueError("keyframe image must name its decoded video frame")
        output = directory / ref["image"]
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("frame decoding needs ffmpeg")
        subprocess.run(  # noqa: S603 - fixed executable/argv, no shell or user paths
            [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(directory / "rgb.mov"),
                "-vf",
                f"select=eq(n\\,{index})",
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-y",
                str(output),
            ],
            check=True,
            timeout=90,
            capture_output=True,
        )
        with Image.open(output) as image:
            image.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG")
        depth: F64 | None = None
        if ref.get("depth") is not None:
            assert depth_record is not None
            depth = (
                np.fromfile(directory / depth_record["file"], dtype="<f2")
                .astype(np.float64)
                .reshape(depth_record["height"], depth_record["width"])
            )
        result.append(
            Frame(
                index,
                buffer.getvalue(),
                depth,
                np.asarray(record["intrinsics"], dtype=np.float64).reshape(3, 3),
                np.asarray(record["transform"], dtype=np.float64).reshape(4, 4),
                (record["imageWidth"], record["imageHeight"]),
            )
        )
    return result


def context(frames: list[Frame]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame in frames:
        summary: dict[str, Any] = {"keyframe": frame.index, "depth": None}
        if frame.depth is not None:
            valid = frame.depth[np.isfinite(frame.depth) & (frame.depth > 0)]
            summary["depth"] = {
                "validPixels": len(valid),
                "totalPixels": frame.depth.size,
                "minM": float(valid.min()) if len(valid) else None,
                "maxM": float(valid.max()) if len(valid) else None,
            }
        import json

        content.append({"type": "text", "text": json.dumps(summary)})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frame.image).decode("ascii"),
                },
            }
        )
    return content
