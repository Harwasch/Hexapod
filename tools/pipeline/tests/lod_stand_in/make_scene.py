"""A tiny capture for `lod_optimise.py`, made with the stand-in rasteriser beside it.

Writes, under the directory given:

* `canonical.ply` -- a splat in the placed frame: a textured ground patch and a sphere;
* `placement.json` -- a similarity that is not the identity (scale 2, 30 degrees about z,
  a translation), as `place` writes it;
* `data/cameras.npz` -- an orbit of cameras in the *COLMAP* frame (the placement undone),
  with the "photos" each saw: the whole splat rendered by the stand-in over a grey
  background, since a real photo has something behind what the splat kept. The optimiser must
  put the cameras back through `placement.json` to see the splat where the photos did.

Run with an interpreter that has torch: `python make_scene.py <dir> [count] [views]`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from gsplat import rasterization  # noqa: E402

import lod_maths  # noqa: E402

WIDTH, HEIGHT, FOCAL = 64, 48, 56.0
BACKGROUND = 0.5
PROPERTIES = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
PROPERTIES += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
PLACEMENT_SCALE = 2.0
PLACEMENT_ANGLE = math.radians(30.0)
PLACEMENT_T = np.array([3.0, -1.0, 0.5])


def placement_rotation() -> np.ndarray:
    c, s = math.cos(PLACEMENT_ANGLE), math.sin(PLACEMENT_ANGLE)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def splat(count: int, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ground = count * 2 // 3
    xy = rng.uniform(-2.0, 2.0, size=(ground, 2))
    xyz = np.c_[xy, rng.normal(0.0, 0.01, ground)]
    ball = rng.normal(size=(count - ground, 3))
    ball = ball / np.linalg.norm(ball, axis=1, keepdims=True) * 0.6 + [0.5, 0.3, 0.7]
    xyz = np.r_[xyz, ball]
    # A checker on the ground and a gradient on the ball: detail a merge blurs away.
    checker = ((np.floor(xyz[:, 0] * 2) + np.floor(xyz[:, 1] * 2)) % 2) * 2.0 - 1.0
    colour = np.c_[checker, 0.4 * xyz[:, 2], -checker * 0.5]
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    columns = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    columns.update({f"f_dc_{i}": colour[:, i] for i in range(3)})
    columns["opacity"] = rng.uniform(1.0, 4.0, count)
    columns.update({f"scale_{i}": rng.normal(-3.3, 0.2, count) for i in range(3)})
    columns.update({f"rot_{i}": quat[:, i] for i in range(4)})
    return {name: np.asarray(value, dtype=np.float32) for name, value in columns.items()}


def write_ply(path: Path, columns: dict[str, np.ndarray]) -> None:
    count = columns["x"].shape[0]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in PROPERTIES] + ["end_header"]
    rows = np.empty(count, dtype=[(name, "<f4") for name in PROPERTIES])
    for name in PROPERTIES:
        rows[name] = columns[name]
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())


def main() -> int:
    out = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    views = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out.mkdir(parents=True, exist_ok=True)
    columns = splat(count)
    write_ply(out / "canonical.ply", columns)
    rotation = placement_rotation()
    (out / "placement.json").write_text(
        json.dumps(
            {
                "scale": PLACEMENT_SCALE,
                "rotation": rotation.tolist(),
                "translation": PLACEMENT_T.tolist(),
            }
        ),
        encoding="utf-8",
    )
    k = np.array([[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]])
    means = torch.tensor(np.stack([columns[a] for a in "xyz"], 1))
    quats = torch.tensor(np.stack([columns[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(torch.tensor(np.stack([columns[f"scale_{i}"] for i in range(3)], 1)))
    opacities = torch.sigmoid(torch.tensor(columns["opacity"]))
    colours = torch.tensor(np.stack([columns[f"f_dc_{i}"] for i in range(3)], 1))[:, None, :]
    names, c2ws, images = [], [], []
    for index in range(views):
        azimuth = 2 * math.pi * index / views
        radius = 4.0 + 1.5 * (index % 3)
        rise = 2.0 + 0.3 * (index % 2)
        eye = np.array([radius * math.cos(azimuth), radius * math.sin(azimuth), rise])
        view = lod_maths.look_at(eye, np.array([0.0, 0.0, 0.3]))
        with torch.no_grad():
            image, alpha, _ = rasterization(
                means, quats, scales, opacities, colours,
                torch.tensor(view, dtype=torch.float32)[None],
                torch.tensor(k, dtype=torch.float32)[None],
                WIDTH, HEIGHT, sh_degree=0, packed=False,
            )  # fmt: skip
        # A photo has something behind the splat: a grey wall, where the render has black.
        photo = image[0] + (1.0 - alpha[0]) * BACKGROUND
        images.append((photo.clamp(0, 1).numpy() * 255).round().astype(np.uint8))
        placed = np.linalg.inv(view)
        # Undo the placement: x = R^T (x' - t) / s for the centre, R^T for the axes.
        c2w = np.eye(4)
        c2w[:3, :3] = rotation.T @ placed[:3, :3]
        c2w[:3, 3] = rotation.T @ (placed[:3, 3] - PLACEMENT_T) / PLACEMENT_SCALE
        c2ws.append(c2w)
        names.append(f"frame_{index:04d}.jpg")
    (out / "data").mkdir(exist_ok=True)
    np.savez(
        out / "data" / "cameras.npz",
        names=np.array(names),
        camtoworlds=np.array(c2ws),
        K=k,
        images=np.array(images),
    )
    sys.stdout.write(f"scene: {count} gaussians, {views} views in {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
