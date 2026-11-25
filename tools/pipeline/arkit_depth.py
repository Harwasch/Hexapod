"""Measured bundle depth, in native camera pixels and ARKit metres.

Only private training copies receive seeds/priors. The final raw model supplies both
extrinsics and OPENCV (or fallback ARKit PINHOLE) calibration. Depth is camera Z,
not range along a unit ray. No video display rotation belongs in this path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import init_seed
import sfm

DIRECTORY = "arkit-depth"
PRIORS = "arkit-depth-priors"
IDENTITY = "arkit-depth.sha256"


def export(frames: Path, source: Path, records: list[dict[str, Any]]) -> None:
    """Carry smoothed measurements inside the recursive frames artifact across runners."""
    destination = frames / DIRECTORY
    destination.mkdir()
    manifest = {}
    for record in records:
        depth = record.get("smoothedDepth")
        if depth is None:
            continue
        name = f"frame_{record['index']:08d}.png"
        manifest[name] = depth
        for key in ("file", "confidence"):
            shutil.copy2(source / depth[key], destination / depth[key])
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def subset(source: Path, target: Path, names: set[str]) -> None:
    """Physically exclude external holdouts before the training runner sees depth."""
    source = source / DIRECTORY
    if not source.is_dir():
        return
    manifest = json.loads((source / "manifest.json").read_text())
    selected = {name: value for name, value in manifest.items() if name in names}
    target = target / DIRECTORY
    target.mkdir()
    for value in selected.values():
        for key in ("file", "confidence"):
            shutil.copy2(source / value[key], target / value[key])
    (target / "manifest.json").write_text(json.dumps(selected, sort_keys=True) + "\n")


def calibration(camera: sfm.Camera, width: int, height: int) -> tuple[Any, Any]:
    if camera.model not in {"PINHOLE", "OPENCV"}:
        raise ValueError("bundle depth needs a raw PINHOLE or OPENCV camera")
    fx, fy, cx, cy, *coefficients = camera.params
    sx, sy = width / camera.width, height / camera.height
    k = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
    return k, np.array(coefficients or [0, 0, 0, 0], dtype=float)


def valid_depth(depth: Any, confidence: Any) -> Any:
    return (confidence == 2) & np.isfinite(depth) & (depth > 0)


def backproject(
    depth: Any, confidence: Any, camera: sfm.Camera, pose: sfm.Image
) -> tuple[Any, Any]:
    """Return world XYZ and native depth XY of valid samples, using FINAL extrinsics."""
    import cv2  # type: ignore[import-not-found,unused-ignore]

    cv2.setNumThreads(4)
    height, width = depth.shape
    y, x = np.nonzero(valid_depth(depth, confidence))
    xy = np.column_stack([x, y]).astype(float)
    if not len(xy):
        return np.empty((0, 3)), xy
    k, coefficients = calibration(camera, width, height)
    rays = cv2.undistortPoints(xy.reshape(-1, 1, 2), k, coefficients).reshape(-1, 2)
    camera_xyz = np.column_stack([rays, np.ones(len(rays))]) * depth[y, x, None]
    # Row-vector inverse of p_cam = R_wc p_world + t_wc; no second ARKit axis flip.
    return (camera_xyz - np.asarray(pose.tvec)) @ pose.rotation, xy


def resample(depth: Any, confidence: Any, raw: sfm.Camera, target: sfm.Camera) -> Any:
    """Nearest confidence-2 Z on a <=256px grid spanning rectified training pixels.

    Sample centres match torch interpolate(align_corners=False). Nearest sampling
    never blends across confidence boundaries or foreground/background edges.
    Rectification rotates no rays: optical Z is the same before/after the crop.
    """
    import cv2  # type: ignore[import-not-found,unused-ignore]

    cv2.setNumThreads(4)
    scale = min(1.0, 256 / max(target.width, target.height))
    width, height = max(1, round(target.width * scale)), max(1, round(target.height * scale))
    y, x = np.indices((height, width), dtype=float)
    fx, fy, cx, cy = target.params[:4]
    rays = np.stack(
        [
            ((x + 0.5) * target.width / width - 0.5 - cx) / fx,
            ((y + 0.5) * target.height / height - 0.5 - cy) / fy,
            np.ones_like(x),
        ],
        axis=-1,
    )
    k, coefficients = calibration(raw, depth.shape[1], depth.shape[0])
    xy, _ = cv2.projectPoints(rays.reshape(-1, 3), np.zeros(3), np.zeros(3), k, coefficients)
    xy = xy.reshape(height, width, 2).astype(np.float32)
    clean = np.where(valid_depth(depth, confidence), depth, 0).astype(np.float32)
    return cv2.remap(
        clean,
        xy[..., 0],
        xy[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def training_names(names: list[str], every: int) -> list[str]:
    if every < 0 or every == 1:
        raise ValueError("test_every must be 0 or >=2")
    return [name for i, name in enumerate(sorted(names)) if not every or i % every]


def options(frames: Path, params: Mapping[str, Any]) -> tuple[bool, bool]:
    bundle = (frames / DIRECTORY / "manifest.json").is_file()
    return bundle and bool(params.get("arkit_depth_init", True)), bundle and bool(
        params.get("arkit_depth_loss", False)
    )


def prepare(
    frames: Path,
    poses: Path,
    dataset: Path,
    params: Mapping[str, Any],
    *,
    cap_max: int | None = None,
) -> dict[str, Any] | None:
    """Bounded streaming seed fusion and low-res, on-disk train-only supervision."""
    use_init, use_loss = options(frames, params)
    prior_dir = dataset / PRIORS
    (dataset / IDENTITY).unlink(missing_ok=True)
    if prior_dir.exists():
        shutil.rmtree(prior_dir)
    if not (use_init or use_loss):
        return None
    stride = int(params.get("arkit_depth_stride", 8))
    cap = int(params.get("arkit_depth_max_points", 250_000))
    voxel = float(params.get("arkit_depth_voxel_m", 0.02))
    if stride < 1 or cap < 1 or not math.isfinite(voxel) or voxel <= 0:
        raise ValueError("ARKit depth stride/cap/voxel must be positive")
    if int(params.get("data_factor", 1)) != 1:
        raise ValueError("ARKit depth requires data_factor=1; use train_max_side")
    if use_loss and int(params.get("batch_size", 1)) != 1:
        raise ValueError("ARKit depth loss requires batch_size=1")
    source = frames / DIRECTORY
    records = json.loads((source / "manifest.json").read_text())
    model = sfm.read_model(poses)
    if cap_max is not None:
        cap = min(cap, max(0, cap_max - model.points3d))
    cameras = {c.id: c for c in model.cameras}
    target_model = sfm.read_model(dataset / "sparse" / "0")
    targets = {c.id: c for c in target_model.cameras}
    images = {i.name: i for i in model.images}
    names = training_names(list(images), int(params.get("test_every", 0)))
    selected = set(names[::stride]) if use_init and cap else set()
    # At most cap voxel representatives, plus one 256x192 input frame in memory.
    voxels: dict[tuple[int, ...], tuple[Any, Any]] = {}
    fingerprint = hashlib.sha256()
    sampled = valid = seeded_frames = prior_frames = 0
    seed_stamp = (dataset / "sparse" / "0" / "points3D.bin").stat().st_mtime_ns
    if use_loss:
        prior_dir.mkdir()
    for name in names:
        if name not in selected and not use_loss:
            continue
        record = records.get(name)
        if record is None:
            continue
        shape = (record["height"], record["width"])
        depth = np.fromfile(source / record["file"], dtype="<f2").reshape(shape).astype(np.float32)
        confidence = np.fromfile(source / record["confidence"], dtype=np.uint8).reshape(shape)
        stamp_ns = max((source / record[k]).stat().st_mtime_ns for k in ("file", "confidence"))
        seed_stamp = max(seed_stamp, stamp_ns)
        pose = images[name]
        raw = cameras[pose.camera_id]
        sampled += 1
        valid += int(valid_depth(depth, confidence).sum())
        if name in selected:
            xyz, xy = backproject(depth, confidence, raw, pose)
            with Image.open(frames / name) as image:
                rgb_image = np.asarray(image.convert("RGB"))
            px = np.minimum(
                (xy[:, 0] * rgb_image.shape[1] / shape[1]).astype(int), rgb_image.shape[1] - 1
            )
            py = np.minimum(
                (xy[:, 1] * rgb_image.shape[0] / shape[0]).astype(int), rgb_image.shape[0] - 1
            )
            rgb = rgb_image[py, px]
            cells = np.floor(xyz / voxel).astype(np.int64)
            _, first = np.unique(cells, axis=0, return_index=True)
            # Even coverage within each frame before reaching the cap; fixed RNG for resume.
            for index in np.random.default_rng(0).permutation(first):
                key = tuple(int(v) for v in cells[index])
                if key not in voxels and len(voxels) < cap:
                    voxels[key] = (xyz[index], rgb[index])
            seeded_frames += 1
        if use_loss:
            target = targets[pose.camera_id]
            # Fallback PINHOLE datasets retain native camera dimensions when RGB shrinks.
            with Image.open(dataset / "images" / name) as image:
                width, height = image.size
            sx, sy = width / target.width, height / target.height
            p = target.params
            target = replace(
                target,
                width=width,
                height=height,
                params=(p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy),
            )
            path = prior_dir / (name + ".npy")
            prior = resample(depth, confidence, raw, target)
            np.save(path, prior, allow_pickle=False)
            fingerprint.update(name.encode())
            fingerprint.update(prior.tobytes())
            # Derived deterministic bytes keep the source timestamp for resume identity.
            os.utime(path, ns=(stamp_ns, stamp_ns))
            prior_frames += 1
    seeded = None
    if voxels:
        values = list(voxels.values())
        seed = init_seed.Seed(
            np.asarray([v[0] for v in values], dtype=np.float32),
            np.asarray([v[1] for v in values], dtype=np.uint8),
            np.ones(len(values), dtype=np.float32),
            {},
        )
        fingerprint.update(seed.xyz.tobytes())
        fingerprint.update(seed.rgb.tobytes())
        seeded = init_seed.apply(dataset / "sparse" / "0", seed, None, budget=cap)
        os.utime(dataset / "sparse" / "0" / "points3D.bin", ns=(seed_stamp, seed_stamp))
    # A content identity is needed: seed count/file size and source mtimes can stay
    # unchanged when measurements change. Hash already-materialized outputs, not RGB.
    (dataset / IDENTITY).write_text(fingerprint.hexdigest() + "\n")
    os.utime(dataset / IDENTITY, ns=(seed_stamp, seed_stamp))
    return {
        "initEnabled": use_init,
        "lossEnabled": use_loss,
        "trainFrames": len(names),
        "framesRead": sampled,
        "seedFrames": seeded_frames,
        "priorFrames": prior_frames,
        "confidence2Samples": valid,
        "seedPoints": 0 if seeded is None else seeded.seeded,
        "sfmPoints": model.points3d,
        "initialPoints": model.points3d + (0 if seeded is None else seeded.seeded),
        "voxelM": voxel,
        "cap": cap,
        "stride": stride,
    }


def loss_args(frames: Path, params: Mapping[str, Any]) -> list[str]:
    if not options(frames, params)[1]:
        return []
    weight = float(params.get("arkit_depth_weight", 0.01))
    warmup = int(params.get("arkit_depth_warmup", 1000))
    if not math.isfinite(weight) or weight <= 0 or warmup < 0:
        raise ValueError("ARKit depth weight must be positive and warmup nonnegative")
    return [
        "--arkit-depth-loss",
        "--arkit-depth-weight",
        str(weight),
        "--arkit-depth-warmup",
        str(warmup),
    ]
