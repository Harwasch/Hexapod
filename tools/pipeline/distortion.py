"""Explicit capture-bundle rectification, shared by training and projection consumers.

OpenCV alpha=0, cropped ROI, with the crop origin subtracted from the principal
point. Raw pose artifacts retain OPENCV; the training copy contains PINHOLE and
rectified observations/pixels, so gsplat performs no second undistortion.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import sfm


def enabled(directory: Path) -> bool:
    path = directory / "poses.json"
    if not path.is_file():
        return False
    document = json.loads(path.read_text())
    report = document.get("refinement", {})
    return bool(
        document.get("source") == "hex-capture-v1"
        and report.get("status") == "accepted"
        and report.get("lensEnabled")
    )


def geometry(camera: sfm.Camera) -> tuple[sfm.Camera, Any, Any, tuple[int, int, int, int]]:
    import cv2  # type: ignore[import-not-found,unused-ignore]

    cv2.setNumThreads(4)
    fx, fy, cx, cy, *params = camera.params
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    new, roi = cv2.getOptimalNewCameraMatrix(k, np.array(params), (camera.width, camera.height), 0)
    x, y, width, height = map(int, roi)
    if width <= 0 or height <= 0 or not np.isfinite(new).all():
        raise ValueError("OPENCV rectification produced an empty or invalid ROI")
    new[0, 2] -= x
    new[1, 2] -= y
    pinhole = replace(
        camera,
        model="PINHOLE",
        width=width,
        height=height,
        params=(float(new[0, 0]), float(new[1, 1]), float(new[0, 2]), float(new[1, 2])),
        param_names=("fx", "fy", "cx", "cy"),
    )
    return pinhole, k, new, (x, y, width, height)


def projection_model(model: sfm.Model, directory: Path) -> sfm.Model:
    if not enabled(directory):
        return model
    return replace(
        model, cameras=tuple(geometry(c)[0] if c.model == "OPENCV" else c for c in model.cameras)
    )


def rectify_dataset(root: Path) -> None:
    """Only mutate private dataset copies; unlink hard-linked images before saving."""
    sparse = root / "sparse" / "0"
    if not enabled(sparse):
        return
    import cv2  # type: ignore[import-not-found,unused-ignore]
    from PIL import Image

    model = sfm.read_model(sparse)
    stamps = {name: (sparse / name).stat() for name in ("cameras.bin", "images.bin")}
    opencv = [c for c in model.cameras if c.model == "OPENCV"]
    if not opencv:
        return
    names = {image.name for image in model.images}
    for path in (root / "images").iterdir():
        if path.is_file() and path.name not in names:
            path.unlink()
    # Rectify at the already-resized training dimensions, including rounding in each axis.
    converted = {}
    maps = {}
    for camera in opencv:
        name = next(i.name for i in model.images if i.camera_id == camera.id)
        with Image.open(root / "images" / name) as image:
            width, height = image.size
        sx, sy = width / camera.width, height / camera.height
        params = list(camera.params)
        params[:4] = [params[0] * sx, params[1] * sy, params[2] * sx, params[3] * sy]
        resized = replace(camera, width=width, height=height, params=tuple(params))
        pinhole, k, new, _ = geometry(resized)
        distortion = np.array(params[4:])
        maps[camera.id] = cv2.initUndistortRectifyMap(
            k, distortion, None, new, (pinhole.width, pinhole.height), cv2.CV_32FC1
        )
        converted[camera.id] = (pinhole, k, new, distortion, sx, sy)
    for item in model.images:
        if item.camera_id not in converted:
            continue
        path = root / "images" / item.name
        stamp = path.stat()
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        pixels = cv2.remap(pixels, *maps[item.camera_id], cv2.INTER_LINEAR)
        path.unlink()
        # Pillow's JPEG default (quality 75, 4:2:0) would visibly soften 4K bundle frames.
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            Image.fromarray(pixels).save(path, quality=95, subsampling=0)
        else:
            Image.fromarray(pixels).save(path)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.cameras)))
        for camera in model.cameras:
            c = converted[camera.id][0] if camera.id in converted else camera
            model_id = next(i for i, (name, _) in sfm.CAMERA_MODELS.items() if name == c.model)
            handle.write(struct.pack("<IiQQ", c.id, model_id, c.width, c.height))
            handle.write(struct.pack(f"<{len(c.params)}d", *c.params))
    # Keep feature indices and tracks stable, but express every observation in the new pixels.
    path = sparse / "images.bin"
    raw = bytearray(path.read_bytes())
    offset = 8
    for _ in range(struct.unpack_from("<Q", raw)[0]):
        camera_id = struct.unpack_from("<I", raw, offset + 60)[0]
        offset = raw.index(0, offset + 64) + 1
        count = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        if camera_id in converted and count:
            _, k, new, coefficients, sx, sy = converted[camera_id]
            xy = np.array([struct.unpack_from("<dd", raw, offset + 24 * i) for i in range(count)])
            xy *= [sx, sy]
            xy = cv2.undistortPoints(xy.reshape(-1, 1, 2), k, coefficients, P=new).reshape(-1, 2)
            for i, pair in enumerate(xy):
                struct.pack_into("<dd", raw, offset + 24 * i, *pair)
        offset += 24 * count
    path.write_bytes(raw)
    (sparse / "rectification.json").write_text(
        json.dumps(
            {
                "source": "OPENCV",
                "method": "opencv-alpha0-cropped",
                "cameras": [c[0].to_dict() for c in converted.values()],
            }
        )
        + "\n"
    )
    # Rebuilding deterministic derived files must preserve checkpoint identity.
    # The input camera timestamp also identifies the derived rectification manifest.
    stamps["rectification.json"] = stamps["cameras.bin"]
    for name, stamp in stamps.items():
        os.utime(sparse / name, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
