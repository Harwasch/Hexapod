"""Tiny synthetic frozen inputs and plainly labelled, fabricated evaluation metrics."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

from PIL import Image

from .scene import seal, write_json


def synthetic_scene(root: Path, scene: str, commit: str) -> dict[str, Any]:
    root.mkdir(parents=True)
    frames, poses = root / "frames", root / "poses"
    frames.mkdir()
    poses.mkdir()
    names = [f"frame_{i:03d}.png" for i in range(16)]
    for index, name in enumerate(names):
        Image.new("RGB", (64, 48), (index * 12, 60, 120)).save(frames / name)
    (poses / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 64, 48, 50, 50, 32, 24))
    with (poses / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(names)))
        for index, name in enumerate(names):
            angle = index * 2 * math.pi / len(names)
            handle.write(
                struct.pack("<I7dI", index + 1, 1, 0, 0, 0, math.cos(angle), math.sin(angle), 4, 1)
            )
            handle.write(name.encode() + b"\0")
            handle.write(struct.pack("<Q", 8))
            for point in range(8):
                handle.write(struct.pack("<2dQ", 20 + point, 24, point + 1))
    with (poses / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 8))
        for point in range(8):
            handle.write(
                struct.pack("<Q3d3BdQ", point + 1, point / 10, 0, 0, 100, 100, 100, 0.1, len(names))
            )
            for index in range(len(names)):
                handle.write(struct.pack("<II", index + 1, point))
    write_json(root / "frames_meta.json", {"rehearsal": True, "frames": names})
    return seal(root, scene, "synthetic://rehearsal", "root", commit)


def stand_in_params() -> dict[str, Any]:
    import sys

    pipeline = Path(__file__).resolve().parents[2]
    return {
        "trainer": str(pipeline / "tests/gsplat_stand_in.py"),
        "python": sys.executable,
        "iterations": 300,
        "extra_args": ["--gaussians", "64", "--sh_degree", "3"],
        "converge": False,
        "holdout_error": False,
    }


def evaluation(names: list[str]) -> dict[str, Any]:
    metrics = {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2, "psnrFullSh": 21.0, "psnrFullShGap": 1.0}
    return {
        **metrics,
        "frames": len(names),
        "perFrame": [{"name": n, **metrics} for n in names],
        "regions": None,
        "regionsReason": "Synthetic rehearsal has no masks.",
        "rehearsal": True,
    }
