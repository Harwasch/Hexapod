"""A synthetic orbit capture: a COLMAP model on disk and a splat, with known answers.

A table-top object at `TARGET`, ringed by cameras looking at it, standing on a ground
plane that runs well past the ring -- which is the shape of every phone orbit, and the
shape of the problem the quality bar exists for: the object is seen from everywhere,
and the ground beyond the ring (the "fringe") only from the far side, at a glance.

World frame is z up, which COLMAP's is not; nothing here depends on that, because the
quality stage reads up from how the cameras were held (`sfm.camera_up`), as it would on
a real capture.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import gaussians

TARGET = np.array([0.0, 0.0, 0.4])
#: COLMAP's PINHOLE: fx, fy, cx, cy.
PINHOLE = 1


def look_at(centre: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """World-to-camera rotation for a COLMAP camera (x right, y down, z forward)."""
    world_up = np.array([0.0, 0.0, 1.0]) if up is None else up
    forward = target - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def ring(
    count: int, *, radius: float = 1.5, height: float = 1.0, target: np.ndarray = TARGET
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`count` cameras evenly round a circle, each looking at `target`: (R, centre)."""
    out = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        centre = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
        out.append((look_at(centre, target), centre))
    return out


def _quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    q = gaussians.matrix_to_quat(rotation)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def write_model(
    directory: Path,
    cameras: list[tuple[np.ndarray, np.ndarray]],
    *,
    width: int = 1600,
    height: int = 1200,
    focal: float = 1300.0,
) -> Path:
    """`cameras.bin`, `images.bin` and an empty `points3D.bin`, as COLMAP writes them."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<IiQQ", 1, PINHOLE, width, height))
        handle.write(struct.pack("<4d", focal, focal, width / 2, height / 2))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for index, (rotation, centre) in enumerate(cameras):
            t = -rotation @ centre
            handle.write(struct.pack("<IdddddddI", index + 1, *_quat(rotation), *t, 1))
            handle.write(f"frame_{index:04d}.jpg".encode() + b"\0")
            handle.write(struct.pack("<Q", 0))
    (directory / "points3D.bin").write_bytes(struct.pack("<Q", 0))
    return directory


def scene(
    count: int, *, fringe_share: float = 0.3, seed: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A splat's columns, and which gaussians are the fringe (ground past the ring).

    The object is a table top (a disk of radius 0.4 at z 0.4) with a spool on it (a
    cylinder of radius 0.12 up to z 0.7); the ground is z 0 out to 4 m.
    """
    rng = np.random.default_rng(seed)
    fringe = int(count * fringe_share)
    table = int((count - fringe) * 0.7)
    spool = count - fringe - table
    r = 0.4 * np.sqrt(rng.random(table))
    a = 2 * np.pi * rng.random(table)
    table_xyz = np.stack([r * np.cos(a), r * np.sin(a), np.full(table, 0.4)], axis=1)
    a = 2 * np.pi * rng.random(spool)
    spool_xyz = np.stack(
        [0.12 * np.cos(a), 0.12 * np.sin(a), 0.4 + 0.3 * rng.random(spool)], axis=1
    )
    r = 1.8 + 2.2 * np.sqrt(rng.random(fringe))
    a = 2 * np.pi * rng.random(fringe)
    ground_xyz = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(fringe)], axis=1)
    xyz = np.concatenate([table_xyz, spool_xyz, ground_xyz]).astype(np.float32)
    n = xyz.shape[0]
    columns = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "f_dc_0": np.zeros(n, np.float32),
        "f_dc_1": np.zeros(n, np.float32),
        "f_dc_2": np.zeros(n, np.float32),
        # alpha ~0.88: opaque enough to occlude.
        "opacity": np.full(n, 2.0, np.float32),
        "scale_0": np.full(n, -5.0, np.float32),
        "scale_1": np.full(n, -5.0, np.float32),
        "scale_2": np.full(n, -5.0, np.float32),
        "rot_0": np.ones(n, np.float32),
        "rot_1": np.zeros(n, np.float32),
        "rot_2": np.zeros(n, np.float32),
        "rot_3": np.zeros(n, np.float32),
    }
    is_fringe = np.zeros(n, dtype=bool)
    is_fringe[table + spool :] = True
    return columns, is_fringe
