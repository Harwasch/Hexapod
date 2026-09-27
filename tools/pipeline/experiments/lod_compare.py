"""Merged against optimised parents, rendered on the GPU: Phase 1's comparison, again.

Phase 1 (scratchpad lodcmp/compare.py + lpips_eval.py, on the 640k "Train" splat) found
H3DGS-merged parents ahead of the old thinned ones on PSNR, SSIM and holes, and level with
or behind them on LPIPS at the switch distance (tau 8-16 px: 0.138 vs 0.138, 0.353 vs
0.343, 0.169 vs 0.124) -- merged parents are blurry. `optimise_lod` optimises them against
the photos (H3DGS Sec. 5.1). This script repeats Phase 1's two tests on two tilesets of
the same `canonical.ply`, packed with and without the optimised parents, with gsplat's CUDA
rasteriser instead of the numpy one and LPIPS in the same pass:

1. **root at its switch distance**: each tileset's root drawn alone, from where its
   geometric error projects to tau px (tau = 8 and 16), four azimuths 25 degrees up,
   against every leaf drawn from there;
2. **what Cesium draws**: the REPLACE cut at maximumScreenSpaceError tau (`lod_maths.cut`,
   Cesium's rule) from 4, 2, 1 and 0.5 times the scan's 90th-percentile radius, against
   every leaf.

Each row: PSNR, SSIM, LPIPS (AlexNet, torchmetrics) and holes -- the share of pixels the
full render covers (alpha > 0.5) that the LoD view leaves see-through. The root at
tau = 16 is the number to compare with Phase 1's.

**The experiment, on the spool capture** (job 15b70036's video; any photo-reconstruct run
on recipe version 10 has what it needs):

1. Process it on recipe 10 -- `optimise_lod` is on by default. The run's own
   `optimise_lod` step already reports, in its metrics and in
   `stages/optimise_lod/out/lod_parents/summary.json`: held-out loss/PSNR/SSIM/LPIPS at
   the cut before and after (taus 4, 8, 16, 32; frames never trained on), the per-tile
   switch-distance PSNR/SSIM/LPIPS/holes of merged and optimised parents against their
   own subtrees, iterations, it/s and seconds on the L4, and whether they were kept.
2. For Phase 1's protocol exactly, on a GPU box with the training image
   (`modal shell` into `run_stage_l4`, or any CUDA machine with gsplat 1.5.3), with the
   run's `stages/place/out/canonical.ply` and
   `stages/optimise_lod/out/lod_parents/lod_parents.npz`:

       python tools/captures/splat_tiles.py canonical.ply merged --lat 0 --lon 0
       python tools/captures/splat_tiles.py canonical.ply optimised --lat 0 --lon 0 \\
           --parents lod_parents.npz
       $GSPLAT_PYTHON tools/pipeline/experiments/lod_compare.py merged optimised \\
           --out lod_compare.json

   (`--tile-gaussians` and `--opacity-min` at the recipe's 100000 and 0.02, the defaults;
   the packer refuses the parents for any other packing.)
3. **Decide**: keep `optimise_lod` on if the optimised root's LPIPS at tau 8 and 16 is
   below the merged root's by more than the spread across the four azimuths, holes do
   not rise by more than a point, and the held-out loss at the cut fell (the stage's own
   acceptance). If LPIPS does not move, the cost -- see the stage's `loopSeconds` -- buys
   nothing and `enabled: false` is the recipe's answer.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "captures"))

import lod_maths  # noqa: E402
import splat_tiles  # noqa: E402

WIDTH, HEIGHT, FOV_DEG = 480, 360, 50.0
SWITCH_TAUS = (8.0, 16.0)
CUT_TAU = 16.0
AZIMUTHS_DEG = (20.0, 110.0, 200.0, 290.0)
ELEVATION_DEG = 25.0
CUT_SCALES = (4.0, 2.0, 1.0, 0.5)


def decode(path: Path) -> dict[str, np.ndarray]:
    """One tile's gaussians: xyz, sh0, opacity (0-1), scales (linear), quats (w first)."""
    raw = path.read_bytes()
    offset, chunks = 12, {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunks[kind] = raw[offset + 8 : offset + 8 + length]
        offset += 8 + length
    gltf = json.loads(chunks[0x4E4F534A])
    view = gltf["bufferViews"][0]
    start = view.get("byteOffset", 0)
    d = splat_tiles.unpack_spz(chunks[0x004E4942][start : start + view["byteLength"]])
    return {
        "xyz": np.stack([d["x"], d["y"], d["z"]], 1),
        "sh0": np.stack([d[f"f_dc_{i}"] for i in range(3)], 1),
        "opacity": splat_tiles.sigmoid(d["opacity"].astype(np.float64)).astype(np.float32),
        "scales": np.exp(np.stack([d[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32),
        "quats": np.stack([d[f"rot_{i}"] for i in range(4)], 1),
    }


class Tileset:
    """A packed `splat/` folder: its tree as `lod_maths.Tree`, and each tile's content."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        document = json.loads((folder / "tileset.json").read_text(encoding="utf-8"))
        tiles: list[dict[str, Any]] = []
        children: list[tuple[int, ...]] = []

        def visit(tile: dict[str, Any]) -> int:
            index = len(tiles)
            tiles.append(tile)
            children.append(())
            kids = tuple(visit(child) for child in tile.get("children", []))
            children[index] = kids
            return index

        visit(document["root"])
        boxes = np.array([tile["boundingVolume"]["box"] for tile in tiles], dtype=np.float64)
        centre = boxes[:, :3]
        half = np.stack([boxes[:, 3], boxes[:, 7], boxes[:, 11]], 1)
        self.uris = [str(tile["content"]["uri"]) for tile in tiles]
        self.tree = lod_maths.Tree(
            uris=tuple(self.uris),
            low=centre - half,
            high=centre + half,
            error=np.array([float(t["geometricError"]) for t in tiles]),
            children=tuple(children),
            rows=np.zeros((len(tiles), 2), dtype=np.int64),
        )
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def content(self, tiles: list[int]) -> dict[str, np.ndarray]:
        for tile in tiles:
            if tile not in self.cache:
                self.cache[tile] = decode(self.folder / self.uris[tile])
        parts = [self.cache[tile] for tile in tiles]
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("merged", type=Path, help="splat/ packed without --parents")
    parser.add_argument("optimised", type=Path, help="splat/ packed with --parents")
    parser.add_argument("--out", type=Path, default=Path("lod_compare.json"))
    parser.add_argument("--device", default="cuda")
    # Phase 1's 480 x 360; smaller only to check this script on a CPU stand-in.
    parser.add_argument("--size", type=int, nargs=2, default=[WIDTH, HEIGHT])
    args = parser.parse_args(argv)
    width, height = args.size
    focal = 0.5 * height / math.tan(math.radians(FOV_DEG) / 2)

    import torch
    from gsplat import rasterization
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device(args.device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)

    def render(g: dict[str, np.ndarray], viewmat: np.ndarray) -> tuple[Any, Any]:
        def t(a: np.ndarray) -> Any:
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=t(g["xyz"]),
                quats=t(g["quats"]),
                scales=t(g["scales"]),
                opacities=t(g["opacity"]),
                colors=t(g["sh0"])[:, None, :],
                viewmats=t(viewmat)[None],
                Ks=t(k)[None],
                width=width,
                height=height,
                sh_degree=0,
                packed=False,
                rasterize_mode="classic",
            )
        return colors[0].clamp(0, 1), alphas[0, ..., 0]

    def measure(image: Any, alpha: Any, reference: Any, reference_alpha: Any) -> dict[str, float]:
        mse = float(((image - reference) ** 2).mean())
        covered = reference_alpha > 0.5
        holes = float(((alpha < 0.5) & covered).sum()) / max(float(covered.sum()), 1.0)
        with torch.no_grad():
            score = float(lpips(image.permute(2, 0, 1)[None], reference.permute(2, 0, 1)[None]))
        return {
            "psnr": 10 * math.log10(1.0 / max(mse, 1e-12)),
            "ssim": _ssim(torch, image, reference),
            "lpips": score,
            "holes": holes,
        }

    sets = {"merged": Tileset(args.merged), "optimised": Tileset(args.optimised)}
    base = sets["merged"]
    if base.tree.error.tolist() != sets["optimised"].tree.error.tolist():
        sys.stderr.write("lod_compare: the two tilesets are different trees\n")
        return 2
    leaves = base.content(list(base.tree.leaves))
    target = np.median(leaves["xyz"].astype(np.float64), axis=0)
    elevation = math.radians(ELEVATION_DEG)

    def direction(azimuth_deg: float) -> np.ndarray:
        a = math.radians(azimuth_deg)
        return np.array(
            [
                math.cos(elevation) * math.cos(a),
                math.cos(elevation) * math.sin(a),
                math.sin(elevation),
            ]
        )

    rows: list[dict[str, Any]] = []
    for tau in SWITCH_TAUS:
        distance = float(base.tree.error[0]) * focal / tau
        for azimuth in AZIMUTHS_DEG:
            eye = lod_maths.orbit_eye(
                target, direction(azimuth), base.tree.low[0], base.tree.high[0], distance
            )
            view = lod_maths.look_at(eye, target)
            reference, reference_alpha = render(leaves, view)
            for name, tileset in sets.items():
                image, alpha = render(tileset.content([0]), view)
                rows.append(
                    {"test": "root-at-switch", "tau": tau, "azimuth": azimuth, "tileset": name,
                     **measure(image, alpha, reference, reference_alpha)}
                )  # fmt: skip
    extent = float(np.percentile(np.linalg.norm(leaves["xyz"] - target, axis=1), 90))
    for scale in CUT_SCALES:
        for azimuth in AZIMUTHS_DEG[:2]:
            eye = target + extent * scale * direction(azimuth)
            view = lod_maths.look_at(eye, target)
            reference, reference_alpha = render(leaves, view)
            for name, tileset in sets.items():
                chosen = lod_maths.cut(tileset.tree, eye, focal, CUT_TAU)
                content = tileset.content(chosen)
                image, alpha = render(content, view)
                rows.append(
                    {"test": "cesium-cut", "scale": scale, "azimuth": azimuth, "tileset": name,
                     "tiles": len(chosen), "splats": int(content["xyz"].shape[0]),
                     **measure(image, alpha, reference, reference_alpha)}
                )  # fmt: skip
    args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for line in summarise(rows):
        sys.stdout.write(line + "\n")
    return 0


def summarise(rows: list[dict[str, Any]]) -> list[str]:
    """One line per (test, tau or scale, tileset): means over azimuths, and LPIPS's spread."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["test"], row.get("tau"), row.get("scale"), row["tileset"])
        groups.setdefault(key, []).append(row)
    lines = []
    for (test, tau, scale, name), members in groups.items():
        where = f"tau {tau:g}" if tau is not None else f"x{scale:g} radius"
        spread = statistics.pstdev(m["lpips"] for m in members) if len(members) > 1 else 0.0
        lines.append(
            f"{test:15s} {where:12s} {name:9s} n={len(members)} "
            f"lpips {statistics.mean(m['lpips'] for m in members):.3f} (sd {spread:.3f}) "
            f"psnr {statistics.mean(m['psnr'] for m in members):.2f} "
            f"ssim {statistics.mean(m['ssim'] for m in members):.3f} "
            f"holes {statistics.mean(m['holes'] for m in members):.3f}"
        )
    return lines


def _ssim(torch: Any, a: Any, b: Any) -> float:
    """Windowed SSIM (11 x 11 gaussian, sigma 1.5), over RGB, as Phase 1 computed it."""
    import torch.nn.functional as functional

    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5.0
    window = torch.exp(-(coords**2) / (2 * 1.5**2))
    window = window / window.sum()
    kernel = (window[:, None] * window[None, :])[None, None].repeat(3, 1, 1, 1)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]

    def blur(z: Any) -> Any:
        return functional.conv2d(z, kernel, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    top = (2 * mx * my + c1) * (2 * sxy + c2)
    return float((top / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean())


if __name__ == "__main__":
    raise SystemExit(main())
