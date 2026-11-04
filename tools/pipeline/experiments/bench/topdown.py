"""Optional one-shot GPU overview, isolated so errors/timeouts cannot lose test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from holdout_error import read_ply, sh_rest, shipped_sh


def render(ply: Path, camera: dict[str, Any], output: Path, degree: int, antialiased: bool) -> None:
    import torch
    from gsplat import rasterization
    from PIL import Image

    columns = read_ply(ply)

    def tensor(keys: list[str]) -> Any:
        return torch.from_numpy(
            np.stack([columns[k] for k in keys], axis=1).astype(np.float32)
        ).cuda()

    dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None]
    coefficients, used = shipped_sh(dc, sh_rest(columns), degree)
    with torch.no_grad():
        rgb, _, _ = rasterization(
            means=tensor(["x", "y", "z"]),
            quats=tensor([f"rot_{i}" for i in range(4)]),
            scales=tensor([f"scale_{i}" for i in range(3)]).exp(),
            opacities=tensor(["opacity"])[:, 0].sigmoid(),
            colors=torch.from_numpy(coefficients).cuda(),
            viewmats=torch.tensor([camera["viewmat"]], dtype=torch.float32, device="cuda"),
            Ks=torch.tensor([camera["K"]], dtype=torch.float32, device="cuda"),
            width=camera["width"],
            height=camera["height"],
            near_plane=camera["near"],
            far_plane=camera["far"],
            sh_degree=used,
            packed=False,
            rasterize_mode="antialiased" if antialiased else "classic",
        )
        pixels = np.rint(rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("ply", "camera", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--antialiased", action="store_true")
    args = parser.parse_args()
    render(args.ply, json.loads(args.camera.read_text()), args.out, args.degree, args.antialiased)


if __name__ == "__main__":
    main()
