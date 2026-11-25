"""COLMAP metadata only; build the trainer's pixel adapter for ONE test view at a time.

No SfM points/tracks, training RGB, dense remapping grids or tensor libraries are
needed for training-camera features. Image decode/resize/remap remains the trainer's
own path, on a tiny disposable single-view model with an empty point cloud.
"""

from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import sfm

from .geometry import Array
from .renderers import Camera


@dataclass
class Views:
    cameras: dict[int, sfm.Camera]
    images: dict[str, sfm.Image]


def read_views(poses: Path) -> Views:
    # The production reader for cameras is small. Deliberately do NOT read_model:
    # its point-cloud pass is unrelated to calibration; images' observations are skipped.
    cameras = {c.id: c for c in sfm._read_cameras(poses / "cameras.bin")}
    images = {}
    path = poses / "images.bin"
    end = path.stat().st_size
    with path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            image_id, *pose, camera_id = struct.unpack("<IdddddddI", stream.read(64))
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("truncated COLMAP image name")
                if char == b"\x00":
                    break
                name.extend(char)
            decoded = name.decode("utf-8")
            if Path(decoded).is_absolute() or ".." in Path(decoded).parts:
                raise ValueError("unsafe COLMAP frame path")
            observations = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell() + 24 * observations
            if offset > end:
                raise ValueError("truncated COLMAP observations")
            stream.seek(offset)
            if decoded in images or camera_id not in cameras:
                raise ValueError("duplicate image or missing calibration")
            images[decoded] = sfm.Image(
                image_id, decoded, tuple(pose[:4]), tuple(pose[4:]), camera_id, 0, 0
            )
    if not images:
        raise ValueError("empty COLMAP model")
    return Views(cameras, dict(sorted(images.items())))


def raw_intrinsics(camera: sfm.Camera) -> tuple[Array, Array]:
    p = camera.params
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
        coefficients = {
            "SIMPLE_PINHOLE": (),
            "SIMPLE_RADIAL": ((*p[3:4], 0.0, 0.0, 0.0)),
            "RADIAL": ((*p[3:5], 0.0, 0.0)),
        }[camera.model]
    elif camera.model in {"PINHOLE", "OPENCV"}:
        fx, fy, cx, cy = p[:4]
        coefficients = p[4:8] if camera.model == "OPENCV" else ()
    else:
        raise ValueError(f"unsupported metadata camera model: {camera.model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float), np.array(
        coefficients, dtype=np.float32
    )


def image_ratio(
    views: Views, frames: Path, max_side: int | None, *, reference: str | None = None
) -> tuple[float, float]:
    """Mirror Parser's first-image dimension scaling; read only its header, not pixels."""
    from PIL import Image

    first = views.images[reference] if reference else next(iter(views.images.values()))
    calibration = views.cameras[first.camera_id]
    with Image.open(frames / first.name) as image:
        width, height = image.size
    if max_side is not None and max(width, height) > max_side:
        factor = max_side / max(width, height)
        width, height = max(1, round(width * factor)), max(1, round(height * factor))
    return width / calibration.width, height / calibration.height


def metadata_cameras(
    views: Views,
    names: list[str],
    ratio: tuple[float, float],
    metadata: dict[str, dict[str, Any]],
    scale: float,
    *,
    explicit_rectification: bool = False,
) -> list[Camera]:
    """Match trainer K/ROI without allocating per-pixel undistortion maps.

    Perspective Parser's historical crop convention is preserved: it keeps the
    getOptimalNewCameraMatrix principal point when cropping its ROI. The production
    explicit-rectification path instead subtracts the ROI origin via distortion.geometry.
    """
    calibrations = {}
    for cid in {views.images[n].camera_id for n in names}:
        source = views.cameras[cid]
        k, coefficients = raw_intrinsics(source)
        sx, sy = ratio
        k[0] *= sx
        k[1] *= sy
        width, height = int(source.width * sx), int(source.height * sy)
        if explicit_rectification and source.model == "OPENCV":
            import distortion

            p = tuple(
                [
                    source.params[0] * sx,
                    source.params[1] * sy,
                    source.params[2] * sx,
                    source.params[3] * sy,
                    *source.params[4:],
                ]
            )
            converted, _, k, _ = distortion.geometry(
                replace(source, width=width, height=height, params=p)
            )
            width, height = converted.width, converted.height
        elif len(coefficients):
            import cv2  # type: ignore[import-not-found,unused-ignore]

            cv2.setNumThreads(4)
            k, roi = cv2.getOptimalNewCameraMatrix(k, coefficients, (width, height), 0)
            width, height = int(roi[2]), int(roi[3])
        if width <= 0 or height <= 0:
            raise ValueError("empty undistortion ROI")
        calibrations[cid] = (k, width, height)
    result = []
    for name in names:
        item = views.images[name]
        pose = np.eye(4)
        pose[:3, :3] = item.rotation.T
        pose[:3, 3] = item.centre * scale
        k, width, height = calibrations[item.camera_id]
        meta = metadata.get(name, {})
        result.append(
            Camera(
                name,
                pose,
                k.copy(),
                width,
                height,
                float(meta["blur"]) if meta.get("blur") is not None else float("nan"),
                float(meta["exposure"]) if meta.get("exposure") is not None else float("nan"),
            )
        )
    return result


def single_view_source(views: Views, name: str, frozen: Path, root: Path) -> None:
    """Write metadata for one held-out view, no observations/points; link only its RGB."""
    item = views.images[name]
    calibration = views.cameras[item.camera_id]
    poses, frames = root / "poses", root / "frames"
    poses.mkdir(parents=True)
    (frames / name).parent.mkdir(parents=True, exist_ok=True)
    (frames / name).symlink_to(frozen / "frames" / name)
    model_id = next(i for i, (model, _) in sfm.CAMERA_MODELS.items() if model == calibration.model)
    with (poses / "cameras.bin").open("wb") as stream:
        stream.write(
            struct.pack(
                "<QIiQQ", 1, calibration.id, model_id, calibration.width, calibration.height
            )
        )
        stream.write(struct.pack(f"<{len(calibration.params)}d", *calibration.params))
    with (poses / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<QIdddddddI", 1, item.id, *item.qvec, *item.tvec, item.camera_id))
        stream.write(name.encode("utf-8") + b"\x00" + struct.pack("<Q", 0))
    (poses / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    if (frozen / "poses/poses.json").is_file():
        shutil.copyfile(frozen / "poses/poses.json", poses / "poses.json")
