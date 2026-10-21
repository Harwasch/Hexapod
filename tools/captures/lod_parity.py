"""Parents against their leaves, on the CPU: how much of what a parent tile stands in for it
still draws, seen from where the viewer switches to it.

For every parent tile of a `splat_tiles.py` tileset, its subtree's leaves (every original
gaussian under it) and the parent alone are rendered with `splat_render.render` (numpy,
seed 0) from where the parent's geometric error projects to `tau` pixels -- the distance
at which CesiumJS's REPLACE traversal stops drawing the children and draws the parent,
geometricError * focal / tau to the tile's box -- along `--azimuths` directions
`--elevation-deg` up, looking at the box centre. Per view:

* **coverage**: sum of the parent's alpha over the leaves' (1 is the same ink);
* **holes**: of the pixels the leaves cover (alpha > 0.5), the share the parent leaves
  see-through (alpha < 0.5);
* **psnr**: of RGB over the pixels the leaves cover.

Positions are in the root's frame: each tile's translation-only transform is composed
(`splat_tiles.tile_offsets`, ADR 0011), for boxes and gaussians alike, so a far site
(`data/tiles/synthetic-far`) is measured where it is drawn.

These are reported, not gated: no threshold has a recorded measurement behind it.

Usage:
    python lod_parity.py TILESET_DIR [--tau 16] [--azimuths 4] [--elevation-deg 25]
        [--size 256] [--out report.json] [--png sheet.png]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import splat_render
import splat_tiles
from rig_tiles import glb_spz

FOV_DEG = 50.0


def tile_splats(
    tileset_dir: Path, uri: str, offsets: splat_tiles.TileOffsets
) -> splat_render.Splats:
    """One tile's gaussians, in the root's frame."""
    columns = splat_tiles.unpack_spz(glb_spz(tileset_dir / uri))
    return splat_render.splats_from_columns(columns, offsets.get(uri))


def tiles_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    """`tileset.json`'s tiles depth first, boxes in the root's frame (`_walk`)."""
    return _walk(document["root"], splat_tiles.tile_offsets(document))


def _walk(tile: dict[str, Any], offsets: splat_tiles.TileOffsets) -> list[dict[str, Any]]:
    """Every tile depth first, each as {uri, error, low, high, children (indices)}, boxes in
    the root's frame."""
    out: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> int:
        index = len(out)
        uri = str(node["content"]["uri"])
        box = np.asarray(node["boundingVolume"]["box"], dtype=np.float64)
        half = np.abs(box[3:].reshape(3, 3)).sum(axis=0)
        centre = box[:3] + np.asarray(offsets.get(uri, np.zeros(3)), dtype=np.float64)
        out.append(
            {
                "uri": uri,
                "error": float(node["geometricError"]),
                "low": centre - half,
                "high": centre + half,
                "children": [],
            }
        )
        out[index]["children"] = [visit(child) for child in node.get("children", [])]
        return index

    visit(tile)
    return out


def _leaves_under(tiles: list[dict[str, Any]], t: int) -> list[int]:
    kids = tiles[t]["children"]
    if not kids:
        return [t]
    return [leaf for kid in kids for leaf in _leaves_under(tiles, kid)]


def _box_distance(eye: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(low - eye, eye - high), 0.0)))


def _eye(
    target: np.ndarray, direction: np.ndarray, low: np.ndarray, high: np.ndarray, d: float
) -> np.ndarray:
    """The point along `direction` from `target` at box distance `d` (bisection)."""
    unit = direction / np.linalg.norm(direction)
    lo, hi = 0.0, max(1.0, 4.0 * d + 4.0 * float(np.linalg.norm(high - low)))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _box_distance(target + mid * unit, low, high) < d:
            lo = mid
        else:
            hi = mid
    return target + hi * unit


def _measure(leaves: splat_render.Frame, parent: splat_render.Frame) -> dict[str, Any]:
    covered = leaves.alpha > 0.5
    total = float(leaves.alpha.sum())
    coverage = float(parent.alpha.sum()) / total if total > 0 else None
    count = int(covered.sum())
    holes = float((covered & (parent.alpha < 0.5)).sum()) / count if count else None
    psnr = None
    if count:
        diff = leaves.rgb[covered].astype(np.float64) - parent.rgb[covered].astype(np.float64)
        mse = float((diff**2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return {"coverage": coverage, "holes": holes, "psnr": psnr}


def _mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("coverage", "holes", "psnr"):
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def parity(
    tileset_dir: Path,
    *,
    tau: float = 16.0,
    azimuths: int = 4,
    elevation_deg: float = 25.0,
    size: int = 256,
) -> tuple[dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    """The report, and the root's (leaves, parent) RGB images per azimuth."""
    document = json.loads((tileset_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = splat_tiles.tile_offsets(document)
    tiles = _walk(document["root"], offsets)
    cache: dict[int, splat_render.Splats] = {}

    def content(t: int) -> splat_render.Splats:
        if t not in cache:
            cache[t] = tile_splats(tileset_dir, tiles[t]["uri"], offsets)
        return cache[t]

    focal = 0.5 * size / math.tan(math.radians(FOV_DEG) / 2)
    elevation = math.radians(elevation_deg)
    rows: list[dict[str, Any]] = []
    sheet: list[tuple[np.ndarray, np.ndarray]] = []
    for t, tile in enumerate(tiles):
        if not tile["children"]:
            continue
        under = _leaves_under(tiles, t)
        leaves = splat_render.Splats.concat([content(leaf) for leaf in under])
        parent = content(t)
        low, high = tile["low"], tile["high"]
        target = 0.5 * (low + high)
        switch = tile["error"] * focal / tau
        views = []
        for i in range(azimuths):
            azimuth = 360.0 * i / azimuths
            a = math.radians(azimuth)
            direction = np.array(
                [
                    math.cos(elevation) * math.cos(a),
                    math.cos(elevation) * math.sin(a),
                    math.sin(elevation),
                ]
            )
            eye = _eye(target, direction, low, high, switch)
            camera = splat_render.Camera.look_at(
                eye, target, fov_deg=FOV_DEG, width=size, height=size
            )
            drawn_leaves = splat_render.render(leaves, camera, seed=0)
            drawn_parent = splat_render.render(parent, camera, seed=0)
            views.append({"azimuthDeg": azimuth, **_measure(drawn_leaves, drawn_parent)})
            if t == 0:
                sheet.append((drawn_leaves.rgb, drawn_parent.rgb))
        rows.append(
            {
                "uri": tile["uri"],
                "geometricError": tile["error"],
                "gaussians": len(parent),
                "leafGaussians": len(leaves),
                "views": views,
                "mean": _mean(views),
            }
        )
    report = {
        "tileset": str(tileset_dir),
        "tau": tau,
        "size": size,
        "tiles": rows,
        "mean": _mean([row["mean"] for row in rows]),
    }
    return report, sheet


def write_sheet(path: Path, sheet: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Rows of [leaves | parent], one per azimuth, for the root."""
    from PIL import Image

    rows = [np.concatenate([left, right], axis=1) for left, right in sheet]
    image = np.concatenate(rows, axis=0) if rows else np.zeros((1, 1, 3))
    pixels = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("tileset_dir", type=Path, help="a folder holding tileset.json")
    parser.add_argument("--tau", type=float, default=16.0)
    parser.add_argument("--azimuths", type=int, default=4)
    parser.add_argument("--elevation-deg", dest="elevation_deg", type=float, default=25.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args(argv)
    report, sheet = parity(
        args.tileset_dir,
        tau=args.tau,
        azimuths=args.azimuths,
        elevation_deg=args.elevation_deg,
        size=args.size,
    )
    text = json.dumps(report, indent=1) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.png is not None:
        write_sheet(args.png, sheet)
    for row in report["tiles"]:
        m = row["mean"]
        print(
            f"{row['uri']:24s} parent {row['gaussians']:6d} leaves {row['leafGaussians']:7d} "
            f"coverage {_fmt(m['coverage'], 3)} holes {_fmt(m['holes'], 3)} "
            f"psnr {_fmt(m['psnr'], 2)}"
        )
    return 0


def _fmt(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
