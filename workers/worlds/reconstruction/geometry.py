"""Portable exports of actual model predictions; no synthetic fallback geometry."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAX_POINTS = 1_000_000


def selected_indices(mask, limit=MAX_POINTS):
    indices = np.flatnonzero(mask)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]
    return indices


def write_ply(path: Path, fields: dict[str, np.ndarray], *, byte_fields=()):
    count = len(next(iter(fields.values())))
    if not count:
        raise ValueError("No finite geometry was predicted")
    dtype = [(name, "u1" if name in byte_fields else "<f4") for name in fields]
    values = np.empty(count, dtype=dtype)
    for name, value in fields.items():
        values[name] = value
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n"
    header += "".join(
        f"property {'uchar' if name in byte_fields else 'float'} {name}\n"
        for name in fields
    )
    with path.open("wb") as stream:
        stream.write((header + "end_header\n").encode("ascii"))
        stream.write(values.tobytes())
    return count


def export_gaussians(path, means, scales, quaternions, sh, opacity, limit=MAX_POINTS):
    means = np.asarray(means).reshape(-1, 3)
    scales = np.asarray(scales).reshape(-1, 3)
    quaternions = np.asarray(quaternions).reshape(-1, 4)
    sh = np.asarray(sh).reshape(-1, 3)
    opacity = np.asarray(opacity).reshape(-1)
    if not all(len(item) == len(means) for item in (scales, quaternions, sh, opacity)):
        raise ValueError("Inconsistent Gaussian prediction shapes")
    norm = np.linalg.norm(quaternions, axis=1)
    mask = np.isfinite(np.column_stack((means, scales, quaternions, sh, opacity))).all(
        axis=1
    )
    mask &= (scales > 0).all(axis=1) & (opacity > 0.01) & (norm > 1e-8)
    idx = selected_indices(mask, limit)
    means, scales, quaternions, sh, opacity = (
        item[idx] for item in (means, scales, quaternions, sh, opacity)
    )
    quaternions = quaternions / norm[idx, None]
    fields = {key: means[:, i] for i, key in enumerate(("x", "y", "z"))}
    fields.update(
        {key: np.zeros(len(idx), dtype=np.float32) for key in ("nx", "ny", "nz")}
    )
    fields.update({f"f_dc_{i}": sh[:, i] for i in range(3)})
    # Spark / standard 3DGS PLY expects logits and log scales, not activated values.
    alpha = np.clip(opacity, 1e-6, 1 - 1e-6)
    fields["opacity"] = np.log(alpha / (1 - alpha))
    fields.update({f"scale_{i}": np.log(scales[:, i]) for i in range(3)})
    fields.update({f"rot_{i}": quaternions[:, i] for i in range(4)})
    return write_ply(path, fields)


def export_points(path, points, colors, confidence, limit=MAX_POINTS):
    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    confidence = np.asarray(confidence).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    if not valid.any():
        raise ValueError("No finite points were predicted")
    # Confidence is a model score, not a calibrated probability.
    cutoff = np.quantile(confidence[valid], 0.2)
    indices = selected_indices(valid & (confidence >= cutoff), limit)
    rgb = np.clip(colors[indices] * 255, 0, 255).astype(np.uint8)
    fields = {key: points[indices, i] for i, key in enumerate(("x", "y", "z"))}
    fields.update({key: rgb[:, i] for i, key in enumerate(("red", "green", "blue"))})
    return write_ply(path, fields, byte_fields=("red", "green", "blue"))


def points_from_depth(depth, poses, intrinsics):
    """Unproject camera-Z depth using the upstream OpenCV camera-to-world convention."""
    depth = np.asarray(depth)
    if depth.ndim == 4:
        depth = depth[..., 0]
    _, height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32)
    frames = []
    for frame, pose, intrinsic in zip(depth, poses, intrinsics, strict=True):
        rays = pixels @ np.linalg.inv(intrinsic).T
        camera = rays * frame[..., None]
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        world[~np.isfinite(frame) | (frame <= 0)] = np.nan
        frames.append(world)
    return np.stack(frames)


def export_depth_mesh(path, points, colors, confidence, stride=8):
    """Triangulate observed depth surfaces. This is not a watertight or fused mesh."""
    p = np.asarray(points)[:, ::stride, ::stride]
    c = np.asarray(colors)[:, ::stride, ::stride]
    score = np.asarray(confidence)[:, ::stride, ::stride]
    vertices, rgb, triangles = [], [], []
    offset = 0
    for frame, color, conf in zip(p, c, score, strict=True):
        h, w, _ = frame.shape
        if h < 2 or w < 2:
            continue
        ids = np.arange(h * w).reshape(h, w)
        faces = np.concatenate(
            (
                np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1]), axis=-1).reshape(
                    -1, 3
                ),
                np.stack((ids[1:, 1:], ids[1:, :-1], ids[:-1, 1:]), axis=-1).reshape(
                    -1, 3
                ),
            )
        )
        flat = frame.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=1) & np.isfinite(conf.reshape(-1))
        if not finite.any():
            continue
        cutoff = np.quantile(conf.reshape(-1)[finite], 0.2)
        keep = finite & (conf.reshape(-1) >= cutoff)
        valid = keep[faces].all(axis=1)
        distances = np.linalg.norm(
            flat[faces[:, [0, 1, 2]]] - flat[faces[:, [1, 2, 0]]], axis=-1
        )
        positive = distances[np.isfinite(distances) & (distances > 0)]
        if not len(positive):
            continue
        valid &= (distances < np.median(positive) * 4).all(axis=1)
        faces = faces[valid]
        if not len(faces):
            continue
        used, inverse = np.unique(faces, return_inverse=True)
        vertices.append(flat[used])
        rgb.append(np.clip(color.reshape(-1, 3)[used], 0, 1))
        triangles.append(inverse.reshape(-1, 3) + offset)
        offset += len(used)
    if not triangles:
        return 0
    vertices = np.concatenate(vertices).astype("<f4")
    rgb = np.concatenate(rgb).astype("<f4")
    triangles = np.concatenate(triangles).astype("<u4")
    arrays = (vertices, rgb, triangles)
    offsets = [0, vertices.nbytes, vertices.nbytes + rgb.nbytes]
    payload = b"".join(array.tobytes() for array in arrays)
    document = {
        "asset": {
            "version": "2.0",
            "generator": "Worlds predicted-depth mesh (unfused)",
        },
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes}
            for offset, array in zip(offsets, arrays, strict=True)
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5126, "count": len(rgb), "type": "VEC3"},
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": triangles.size,
                "type": "SCALAR",
            },
        ],
        "materials": [{"doubleSided": True, "extensions": {"KHR_materials_unlit": {}}}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    payload += b"\0" * (-len(payload) % 4)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(payload))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(payload), 0x004E4942)
        + payload
    )
    return len(triangles)


def geometry_diagnostics(points, confidence, depth, poses, intrinsics):
    points, confidence, depth = map(np.asarray, (points, confidence, depth))
    depth = depth.squeeze(-1) if depth.ndim == 4 else depth
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    result = {
        "frameCount": len(points),
        "finitePointFraction": float(valid.mean()),
        "scope": "prediction-self-consistency-not-ground-truth-accuracy",
        "cameraScale": "model-relative-units",
    }
    if np.isfinite(poses).all():
        result["cameraSpread"] = float(np.linalg.norm(np.ptp(poses[:, :3, 3], axis=0)))
    scores = confidence[np.isfinite(confidence)]
    if len(scores):
        result["medianModelConfidence"] = float(np.median(scores))
    residuals, overlaps = [], []
    for index in range(len(points) - 1):
        samples = points[index].reshape(-1, 3)[
            :: max(1, points[index].size // (3 * 4096))
        ]
        next_pose = poses[index + 1]
        camera = (samples - next_pose[:3, 3]) @ next_pose[:3, :3]
        projection = camera @ intrinsics[index + 1].T
        z = camera[:, 2]
        xy = projection[:, :2] / np.maximum(projection[:, 2:3], 1e-8)
        h, w = depth[index + 1].shape
        inside = np.isfinite(xy).all(axis=1) & np.isfinite(z) & (z > 1e-8)
        inside &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        overlaps.append(float(inside.mean()))
        coordinates = xy[inside].astype(np.int64)
        observed = depth[index + 1][coordinates[:, 1], coordinates[:, 0]]
        keep = np.isfinite(observed) & (observed > 1e-8)
        if keep.any():
            residuals.extend(
                (np.abs(z[inside][keep] - observed[keep]) / observed[keep]).tolist()
            )
    if overlaps:
        result["adjacentProjectedOverlap"] = float(np.mean(overlaps))
    if residuals:
        result["medianRelativeDepthResidual"] = float(np.median(residuals))
    return result
