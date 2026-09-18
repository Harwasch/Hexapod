"""Assemble one capture site from ODM and OpenSplat outputs into data/tiles/<slug>/.

Reads an ODM project folder (mesh 3D Tiles, georeferenced point cloud, OpenSfM reference)
and, optionally, a Gaussian splat PLY trained on the same OpenSfM reconstruction, and writes:

    data/tiles/<slug>/mesh/        ODM's b3dm tiles with textures shrunk, boxes recomputed,
                                   and the glTF marked z-up so the mesh stands on the globe
    data/tiles/<slug>/pointcloud/  quantized pnts quadtree from the LAZ
    data/tiles/<slug>/splat/       one SPZ-compressed splat tile
    data/tiles/captures.json       manifest entry the API seeds a site from

All three share one local east-north-up frame around the OpenSfM reference point, lifted by
--height-offset so the model's ground meets the globe's terrain (see ground_samples.py).

Usage:
    python build_site.py <odm_project> <slug> --tiles-dir ../../data/tiles --name "..."
        --attribution "..." --license-name "..." [--license-url ...] [--source-url ...]
        [--captured 2016-08] [--splat splat.ply] [--height-offset -2.1] [--gsd 0.02]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import median

import numpy as np
from pyproj import CRS, Transformer

import pointcloud_tiles
import splat_tiles
from shrink_b3dm import B3DM_HEADER, join_glb, shrink_glb, split_b3dm, split_glb

Z_UP_TO_Y_UP = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def rewrite_b3dm(raw: bytes, max_edge: int, quality: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Shrink textures and mark the glTF z-up. Returns the tile bytes and its position range."""
    tables, glb = split_b3dm(raw)
    magic, version, _length, ftj, ftb, btj, btb = B3DM_HEADER.unpack_from(raw)
    gltf, binary = split_glb(shrink_glb(glb, max_edge, quality))
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
            lo = np.minimum(lo, accessor["min"])
            hi = np.maximum(hi, accessor["max"])
    # ODM writes east-north-up vertices but says nothing about the up axis, and Cesium turns
    # glTF y-up into z-up, so it lands on its side. Undo that in advance on every mesh node.
    for node in gltf.get("nodes", []):
        if (
            "mesh" in node
            and "matrix" not in node
            and not any(k in node for k in ("translation", "rotation", "scale"))
        ):
            node["matrix"] = Z_UP_TO_Y_UP
    new_glb = join_glb(gltf, binary)
    total = B3DM_HEADER.size + len(tables) + len(new_glb)
    return B3DM_HEADER.pack(magic, version, total, ftj, ftb, btj, btb) + tables + new_glb, lo, hi


def box_from_range(lo: np.ndarray, hi: np.ndarray, pad: float = 0.5) -> list[float]:
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 + pad
    return [*mid.tolist(), half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]


def build_mesh(
    src: Path, dst: Path, lat: float, lon: float, height_offset: float, max_edge: int, quality: int
) -> dict:
    tileset = json.loads((src / "tileset.json").read_text(encoding="utf-8"))
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    stats = {"tiles": 0, "bytes_before": 0, "bytes_after": 0}

    def visit(tile: dict) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        content = tile.get("content")
        if content and content.get("uri"):
            raw = (src / content["uri"]).read_bytes()
            out, tile_lo, tile_hi = rewrite_b3dm(raw, max_edge, quality)
            target = dst / content["uri"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(out)
            stats["tiles"] += 1
            stats["bytes_before"] += len(raw)
            stats["bytes_after"] += len(out)
            lo, hi = np.minimum(lo, tile_lo), np.maximum(hi, tile_hi)
        children = [child for child in tile.get("children") or [] if child]
        for child in children:
            child_lo, child_hi = visit(child)
            lo, hi = np.minimum(lo, child_lo), np.maximum(hi, child_hi)
        tile["children"] = children
        tile["boundingVolume"] = {"box": box_from_range(lo, hi)}
        tile.pop("transform", None)
        if not content:
            tile.pop("content", None)
        return lo, hi

    lo, hi = visit(tileset["root"])
    tileset["root"]["transform"] = splat_tiles.enu_to_ecef(lat, lon, height_offset)
    tileset["asset"] = {"version": "1.1", "generator": "twin build_site"}
    (dst / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    stats["extent_m"] = float(max(hi - lo))
    return stats


def boundary_from_cloud(laz: Path, vertices: int = 24) -> tuple[list[list[float]], list[float]]:
    """Convex hull of the point cloud in longitude/latitude, thinned to a few vertices."""
    import laspy

    cloud = laspy.read(str(laz))
    x, y = np.asarray(cloud.x), np.asarray(cloud.y)
    rng = np.random.default_rng(1)
    pick = rng.choice(x.size, min(x.size, 200_000), replace=False)
    transformer = Transformer.from_crs(
        cloud.header.parse_crs(), CRS.from_epsg(4326), always_xy=True
    )
    lon, lat = transformer.transform(x[pick], y[pick])
    pts = np.stack([lon, lat], axis=1)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) > vertices:
        step = len(hull) / vertices
        hull = [hull[int(i * step)] for i in range(vertices)]
    ring = [[round(float(p[0]), 7), round(float(p[1]), 7)] for p in hull]
    ring.append(ring[0])
    center = [round(float(lon.mean()), 7), round(float(lat.mean()), 7)]
    return ring, center


def camera_heights(project: Path) -> list[float]:
    """Camera centre heights from OpenSfM's topocentric reconstruction (same frame as the LAZ
    when the reference altitude is 0): centre = -R^T t for each shot's angle-axis rotation."""
    path = project / "opensfm" / "reconstruction.topocentric.json"
    if not path.is_file():
        return []
    heights: list[float] = []
    for reconstruction in json.loads(path.read_text(encoding="utf-8")):
        for shot in reconstruction.get("shots", {}).values():
            axis = np.asarray(shot["rotation"], dtype=float)
            angle = float(np.linalg.norm(axis))
            if angle < 1e-12:
                rotation = np.eye(3)
            else:
                k = axis / angle
                cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                rotation = (
                    np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
                )
            centre = -rotation.T @ np.asarray(shot["translation"], dtype=float)
            heights.append(float(centre[2]))
    return heights


def estimate_gsd(project: Path, ground_height: float) -> float | None:
    """Ground sample distance: the cameras' median height above the model's ground (from the
    reconstruction, not EXIF, whose altitudes are often relative or wrong) over the focal
    length in pixels. An estimate; ODM's report is more careful."""
    images = project / "images.json"
    if not images.is_file():
        return None
    rows = json.loads(images.read_text(encoding="utf-8"))
    focal_px = [
        row["focal_ratio"] * row["width"]
        for row in rows
        if row.get("focal_ratio") and row.get("width")
    ]
    heights = camera_heights(project)
    if not focal_px or not heights:
        return None
    above = median(heights) - ground_height
    if above < 5:
        return None
    return round(above / median(focal_px), 4)


def upsert_manifest(tiles_dir: Path, entry: dict) -> None:
    manifest = tiles_dir / "captures.json"
    data = (
        json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {"captures": []}
    )
    data["captures"] = [c for c in data["captures"] if c["slug"] != entry["slug"]] + [entry]
    manifest.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("slug")
    parser.add_argument(
        "--tiles-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "tiles"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--attribution-url")
    parser.add_argument("--license-name", required=True)
    parser.add_argument("--license-url")
    parser.add_argument("--source-url")
    parser.add_argument("--captured")
    parser.add_argument(
        "--splat", type=Path, help="OpenSplat PLY trained on this project's opensfm/"
    )
    parser.add_argument("--splat-iterations", type=int)
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument(
        "--gsd", type=float, help="metres per pixel; estimated from EXIF when omitted"
    )
    parser.add_argument("--max-edge", type=int, default=2048)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--max-gaussians", type=int, default=400_000)
    args = parser.parse_args()

    project: Path = args.project
    reference = json.loads((project / "opensfm" / "reference_lla.json").read_text(encoding="utf-8"))
    lat, lon = float(reference["latitude"]), float(reference["longitude"])
    site_dir: Path = args.tiles_dir / args.slug
    laz = project / "odm_georeferencing" / "odm_georeferenced_model.laz"

    report: dict = {"slug": args.slug, "reference": [lon, lat], "height_offset": args.height_offset}
    report["mesh"] = build_mesh(
        project / "3d_tiles" / "model",
        site_dir / "mesh",
        lat,
        lon,
        args.height_offset,
        args.max_edge,
        args.quality,
    )
    if (site_dir / "pointcloud").exists():
        shutil.rmtree(site_dir / "pointcloud")
    report["pointcloud"] = pointcloud_tiles.convert(
        laz, site_dir / "pointcloud", lat, lon, args.height_offset, args.max_points, 120_000
    )
    if args.splat and args.splat.is_file():
        if (site_dir / "splat").exists():
            shutil.rmtree(site_dir / "splat")
        report["splat"] = splat_tiles.convert(
            args.splat,
            site_dir / "splat",
            lat,
            lon,
            args.height_offset,
            args.max_gaussians,
            0.02,
            2.0,
        )

    ring, center = boundary_from_cloud(laz)
    ground = float(report["pointcloud"]["ground_z_min_m"])
    gsd = args.gsd if args.gsd is not None else estimate_gsd(project, ground - args.height_offset)
    images = len(list((project / "images").iterdir())) if (project / "images").is_dir() else None
    gsd_text = (
        f"about {gsd * 100:.1f} cm per pixel, estimated from the cameras' EXIF" if gsd else None
    )
    spacing = report["pointcloud"]["spacing_m"]
    assets = [
        {
            "representation": "mesh",
            "name": "Textured mesh (ODM)",
            "path": "mesh/tileset.json",
            "ground_sample_distance_m": gsd,
            "description": gsd_text,
            "maximum_screen_space_error": 2,
            "default": True,
        },
        {
            "representation": "point-cloud",
            "name": "Point cloud (ODM)",
            "path": "pointcloud/tileset.json",
            "point_spacing_m": spacing,
            "description": f"{report['pointcloud']['points']:,} points, about {spacing * 100:.0f} cm apart",
            "maximum_screen_space_error": 4,
        },
    ]
    if "splat" in report:
        iterations = f", {args.splat_iterations:,} training steps" if args.splat_iterations else ""
        assets.append(
            {
                "representation": "gaussian-splat",
                "name": "Gaussian splat (OpenSplat)",
                "path": "splat/tileset.json",
                "ground_sample_distance_m": gsd,
                "description": f"{report['splat']['gaussians']:,} gaussians{iterations}",
                "maximum_screen_space_error": 2,
            }
        )
    pipeline = "OpenDroneMap (mesh, point cloud)" + (
        " and OpenSplat (gaussian splat)" if "splat" in report else ""
    )
    entry = {
        "slug": args.slug,
        "name": args.name,
        "description": args.description,
        "boundary": ring,
        "center": [center[0], center[1], round(ground + args.height_offset, 1)],
        "attribution": args.attribution,
        "attribution_url": args.attribution_url,
        "license_name": args.license_name,
        "license_url": args.license_url,
        "source_url": args.source_url,
        "captured": args.captured,
        "pipeline": pipeline,
        "images": images,
        "assets": assets,
    }
    upsert_manifest(args.tiles_dir, entry)
    report["gsd_m"] = gsd
    report["images"] = images
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
