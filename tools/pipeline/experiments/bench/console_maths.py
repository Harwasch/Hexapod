"""CPU protocol for the live console's initial orbit and retained preview inventory."""

from __future__ import annotations

import base64
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

import live
import sfm
import splat_io
from captures_bridge import unpack_spz

VIEWS = ("train-10", "train-50", "train-90", "orbit")
PNG_NAMES = tuple(
    [f"{label}-{view}.png" for view in VIEWS for label in ("trained", "snapshot", "comparison")]
    + ["baseline-orbit.png", "baseline-comparison-orbit.png"]
)


def degree_of(names: Any) -> int:
    rest = {name for name in names if name.startswith("f_rest_")}
    for degree in range(4):
        if rest == {f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))}:
            return degree
    raise ValueError("incomplete or unsupported SH coefficients")


def artifact_info(path: Path) -> dict[str, Any]:
    if path.suffix == ".spz":
        columns = unpack_spz(path.read_bytes(), sh=True)
        count, degree = len(columns["x"]), degree_of(columns)
    else:
        layout = splat_io.read_layout(path)
        count, degree = layout.count, degree_of(layout.properties)
    size = path.stat().st_size
    return {
        "path": str(path),
        "count": count,
        "bytes": size,
        "shDegree": degree,
        "bytesPerSplat": size / count if count else None,
    }


def live_inputs(run: Path) -> tuple[dict[str, Any], Path]:
    """Use the logged payload, never recompute a different sparse sample/quantization."""
    directories = (run / "live", run / "train/checkpoint/live")
    logs = [*(d / "train.log" for d in directories), run / "train/log.txt"]
    log = next((p for p in logs if p.is_file()), None)
    if log is None:
        raise ValueError(f"{run.name}: missing live/train.log; exact console framing unavailable")
    records = live.records(log.read_text())
    camera = next((v for k, v in reversed(records) if k == "cameras"), None)
    if camera is None:
        raise ValueError("missing logged live-cameras payload; cannot reconstruct default orbit")
    latest = next((v for k, v in reversed(records) if k == "splat"), None)
    if latest is not None:
        name = Path(latest["key"]).name
        if not re.fullmatch(r"splat_\d+_\d+\.spz", name):
            raise ValueError("invalid logged console snapshot name")
        paths = [d / name for d in directories if (d / name).is_file()]
    else:
        paths = [
            p
            for d in directories
            for p in d.glob("splat_*.spz")
            if re.fullmatch(r"splat_\d+_\d+\.spz", p.name)
        ]
        paths.sort(key=lambda p: tuple(map(int, p.stem.split("_")[1:])), reverse=True)
    if not paths:
        raise ValueError(f"{run.name}: last console SPZ missing from live/checkpoint copies")
    return camera, paths[0]


def orbit_camera(payload: dict[str, Any], viewport: list[int]) -> dict[str, Any]:
    """Port liveData + live.ts drawCameras/orient/frameView, before auto-rotation.

    Browser viewport sets the initial fit. The subsequent console resize changes K
    only (16:10 desktop, 4:3 <=620px); diagnostic pixels use that CSS aspect ratio.
    """
    origin = np.asarray(payload["origin"], dtype=np.float64)

    def decode(key: str, stride: int, count_key: str) -> np.ndarray:
        raw = base64.b64decode(payload[key], validate=True)
        count = min(int(payload[count_key]), len(raw) // stride)
        packed = np.ndarray((count, 3), "<i2", buffer=raw, strides=(stride, 2))
        # The JS assignment rounds each decoded coordinate to Float32 first.
        return np.asarray(
            origin + packed.astype(float) / 32767 * payload["scale"], dtype=np.float32
        )

    cameras, points = decode("cameras", 12, "cameraCount"), decode("points", 9, "pointCount")
    anchor = points if len(points) else cameras
    centre = np.sort(anchor.astype(float), axis=0)[len(anchor) // 2] if len(anchor) else origin
    both = np.concatenate((points, cameras)).astype(float)
    radius = (
        float(np.sort(np.linalg.norm(both - centre, axis=1))[int(len(both) * 0.9)])
        if len(both) >= 3
        else float(payload["scale"]) * 0.8
    )
    size = max(radius, 1e-3)
    up = np.asarray(payload.get("up") or [0, -1, 0], dtype=float)
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, -1.0, 0.0])
    # THREE.Quaternion.setFromUnitVectors(up, +Y), including its antiparallel branch.
    real = 1 + up[1]
    if real < 1e-8:  # THREE r186 Quaternion.setFromUnitVectors, not a fitted tolerance.
        real = 0.0
        imaginary = np.array(
            [-up[1], up[0], 0.0] if abs(up[0]) > abs(up[2]) else [0.0, -up[2], up[1]]
        )
    else:
        imaginary = np.cross(up, [0.0, 1.0, 0.0])
    orient = sfm.quat_to_matrix([real, *imaginary])
    half_v = math.radians(25)
    half_h = math.atan(math.tan(half_v) * viewport[0] / viewport[1])
    distance = size / math.tan(min(half_v, half_h)) * 1.1
    eye_display = np.array([0.0, 0.5, 1.0]) / math.sqrt(1.25) * distance
    forward = -eye_display / distance
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward)) @ orient
    eye = centre + orient.T @ eye_display
    view = np.eye(4)
    view[:3, :3], view[:3, 3] = rotation, -rotation @ eye
    width, height = (800, 600) if viewport[0] <= 620 else (800, 500)
    focal = height / (2 * math.tan(half_v))
    return {
        "viewmat": view.tolist(),
        "K": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        "width": width,
        "height": height,
        "near": size / 1000,
        "far": size * 200,
        "eye": eye.tolist(),
        "target": centre.tolist(),
        "up": up.tolist(),
        "radius": radius,
        "distance": distance,
        "viewport": viewport,
        "sceneToDisplayRotation": orient.tolist(),
        "verticalFovDegrees": 50,
        "poseSource": "liveData.ts decode/medianCentre/upOf; live.ts drawCameras/orient/frameView",
        "projectionSource": "view.css .lc-scene aspect ratio; fixed diagnostic resolution",
        "moment": "fresh load, first controls lookAt, before auto-rotation or user interaction",
    }


def psnr(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    mse = float(np.mean((prediction.astype(float) - reference.astype(float)) ** 2))
    return -10 * math.log10(mse) if mse else None  # null is exact equality (+infinity)
