"""Georeferenced point cloud (LAS/LAZ) to a 3D Tiles tileset of pnts tiles.

ODM writes `odm_georeferencing/odm_georeferenced_model.laz` in a UTM zone. This reprojects
it into the same local east-north-up frame the mesh and splat tilesets use (around the
OpenSfM reference latitude/longitude at height 0), builds a quadtree where every node holds
a random sample of the points under it (refine ADD), and writes quantized pnts tiles.

Usage:
    python pointcloud_tiles.py model.laz out_dir --lat 46.84 --lon -91.99
        [--height-offset 0] [--max-points 1000000] [--points-per-tile 120000]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer

from splat_tiles import WGS84_A, WGS84_E2, enu_to_ecef

PNTS_HEADER = struct.Struct("<4sIIIIII")


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, height: np.ndarray) -> np.ndarray:
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    return np.stack(
        [
            (n + height) * cos_lat * np.cos(lon),
            (n + height) * cos_lat * np.sin(lon),
            (n * (1 - WGS84_E2) + height) * sin_lat,
        ],
        axis=1,
    )


def to_enu(
    lat_deg: np.ndarray, lon_deg: np.ndarray, height: np.ndarray, ref: tuple[float, float]
) -> np.ndarray:
    """East-north-up metres around the reference point (height 0), matching enu_to_ecef."""
    ecef = geodetic_to_ecef(np.radians(lat_deg), np.radians(lon_deg), height)
    matrix = np.array(enu_to_ecef(ref[0], ref[1], 0.0)).reshape(4, 4).T  # row-major now
    origin = matrix[:3, 3]
    rotation = matrix[:3, :3]
    return (ecef - origin) @ rotation  # rotation is orthonormal: transpose == inverse


def read_points(
    path: Path, height_offset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = laspy.read(str(path))
    crs = cloud.header.parse_crs()
    if crs is None:
        raise SystemExit("point cloud has no CRS")
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(np.asarray(cloud.x), np.asarray(cloud.y))
    height = np.asarray(cloud.z) + height_offset
    rgb = np.stack([cloud.red, cloud.green, cloud.blue], axis=1).astype(np.float64)
    if rgb.max() > 255:
        rgb = rgb / 257.0
    return np.asarray(lat), np.asarray(lon), height, np.clip(rgb, 0, 255).astype(np.uint8)


@dataclass
class Node:
    center: np.ndarray
    half: float
    depth: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    children: list[Node] = field(default_factory=list)
    name: str = "r"


def build_tree(xyz: np.ndarray, rgb: np.ndarray, per_tile: int, rng: np.random.Generator) -> Node:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = (lo + hi) / 2
    half = float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2) + 0.5
    root = Node(center=center, half=half, depth=0)
    _fill(root, xyz, rgb, per_tile, rng)
    return root


def _fill(
    node: Node, xyz: np.ndarray, rgb: np.ndarray, per_tile: int, rng: np.random.Generator
) -> None:
    if xyz.shape[0] <= per_tile or node.depth >= 12:
        node.points, node.colors = xyz, rgb
        return
    pick = rng.choice(xyz.shape[0], per_tile, replace=False)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[pick] = True
    node.points, node.colors = xyz[mask], rgb[mask]
    rest, rest_rgb = xyz[~mask], rgb[~mask]
    east = rest[:, 0] >= node.center[0]
    north = rest[:, 1] >= node.center[1]
    quarter = node.half / 2
    for index, (ex, ny) in enumerate([(False, False), (True, False), (False, True), (True, True)]):
        sel = (east == ex) & (north == ny)
        if not sel.any():
            continue
        child_center = node.center + np.array(
            [quarter if ex else -quarter, quarter if ny else -quarter, 0.0]
        )
        child = Node(
            center=child_center, half=quarter, depth=node.depth + 1, name=f"{node.name}{index}"
        )
        _fill(child, rest[sel], rest_rgb[sel], per_tile, rng)
        node.children.append(child)


def write_pnts(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    lo = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - lo, 1e-3)
    quantized = np.round((xyz - lo) / span * 65535).astype(np.uint16)
    positions = quantized.astype("<u2").tobytes()
    colors = rgb.astype(np.uint8).tobytes()
    table = {
        "POINTS_LENGTH": int(xyz.shape[0]),
        "QUANTIZED_VOLUME_OFFSET": lo.tolist(),
        "QUANTIZED_VOLUME_SCALE": span.tolist(),
        "POSITION_QUANTIZED": {"byteOffset": 0},
        "RGB": {"byteOffset": len(positions)},
    }
    table_json = json.dumps(table, separators=(",", ":")).encode()
    table_json += b" " * ((8 - (PNTS_HEADER.size + len(table_json)) % 8) % 8)
    binary = positions + colors
    binary += b"\x00" * ((8 - len(binary) % 8) % 8)
    total = PNTS_HEADER.size + len(table_json) + len(binary)
    path.write_bytes(
        PNTS_HEADER.pack(b"pnts", 1, total, len(table_json), len(binary), 0, 0)
        + table_json
        + binary
    )


def node_json(node: Node, out_dir: Path, spacing: float) -> dict:
    all_points = node.points
    lo = np.minimum(all_points.min(axis=0), node.center - [node.half, node.half, 0])
    hi = np.maximum(all_points.max(axis=0), node.center + [node.half, node.half, 0])
    for child in node.children:
        # Children extend the vertical range; the quadtree only splits horizontally.
        lo[2] = min(lo[2], _zmin(child))
        hi[2] = max(hi[2], _zmax(child))
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 + 0.25
    uri = f"tiles/{node.name}.pnts"
    write_pnts(out_dir / uri, node.points, node.colors)
    # Geometric error: the point spacing this node's sample achieves. ADD refinement keeps
    # the coarser samples on screen while finer ones stream in.
    error = max(spacing, node.half * 2 / math.sqrt(max(node.points.shape[0], 1)) * 4)
    return {
        "boundingVolume": {"box": [*mid.tolist(), half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]},
        "geometricError": 0.0 if not node.children else error,
        "refine": "ADD",
        "content": {"uri": uri},
        "children": [node_json(child, out_dir, spacing) for child in node.children],
    }


def _zmin(node: Node) -> float:
    return min([float(node.points[:, 2].min())] + [_zmin(c) for c in node.children])


def _zmax(node: Node) -> float:
    return max([float(node.points[:, 2].max())] + [_zmax(c) for c in node.children])


def convert(
    laz: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    height_offset: float,
    max_points: int,
    points_per_tile: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(7)
    plat, plon, height, rgb = read_points(laz, height_offset)
    if plat.shape[0] > max_points:
        keep = rng.choice(plat.shape[0], max_points, replace=False)
        plat, plon, height, rgb = plat[keep], plon[keep], height[keep], rgb[keep]
    xyz = to_enu(plat, plon, height, (lat, lon))
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
    root = build_tree(xyz, rgb, points_per_tile, rng)
    area = float(np.prod(xyz.max(axis=0)[:2] - xyz.min(axis=0)[:2]))
    spacing = math.sqrt(area / max(xyz.shape[0], 1))
    root_json = node_json(root, out_dir, spacing)
    root_json["transform"] = enu_to_ecef(lat, lon, 0.0)
    tileset = {
        "asset": {"version": "1.0"},
        "geometricError": root_json["geometricError"] * 4,
        "root": root_json,
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    tiles = sum(1 for _ in (out_dir / "tiles").glob("*.pnts"))
    return {
        "points": int(xyz.shape[0]),
        "tiles": tiles,
        "spacing_m": round(spacing, 3),
        "ground_z_min_m": float(np.percentile(xyz[:, 2], 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("laz", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--points-per-tile", type=int, default=120_000)
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.laz,
                args.out_dir,
                args.lat,
                args.lon,
                args.height_offset,
                args.max_points,
                args.points_per_tile,
            )
        )
    )


if __name__ == "__main__":
    main()
