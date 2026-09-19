"""Gaussian splat (3DGS PLY) to a 3D Tiles tileset CesiumJS renders.

Reads the PLY that OpenSplat (or any 3DGS trainer) writes, in a local east-north-up frame
around a reference latitude/longitude/height, and writes one glTF tile using the
KHR_gaussian_splatting extension with SPZ (v2) compressed data, which is the form CesiumJS
1.145 loads, plus a tileset.json whose root
transform places the local frame on the globe. The glTF node carries the z-up to y-up swap
the same way Cesium's own sample tilesets do.

Usage:
    python -m app.scripts.splat_tiles splat.ply out_dir --lat 46.84 --lon -91.99 --height 0
        [--max-gaussians 400000] [--opacity-min 0.02] [--geometric-error 2]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Binary little-endian 3DGS PLY: one vertex element with float32 properties."""
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        count = 0
        names: list[str] = []
        types: list[str] = []
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts[0] == "property" and count:
                types.append(parts[1])
                names.append(parts[2])
        if "binary_little_endian" not in " ".join(header):
            raise SystemExit("only binary little-endian PLY is supported")
        fmt = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "int": "i4"}
        dtype = np.dtype([(n, "<" + fmt[t]) for n, t in zip(names, types, strict=True)])
        data = np.frombuffer(handle.read(dtype.itemsize * count), dtype=dtype, count=count)
    return {name: data[name].astype(np.float32) for name in names}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def enu_to_ecef(lat_deg: float, lon_deg: float, height: float) -> list[float]:
    """Column-major 4x4 east-north-up to earth-fixed frame at the reference point."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height) * cos_lat * cos_lon
    y = (n + height) * cos_lat * sin_lon
    z = (n * (1 - WGS84_E2) + height) * sin_lat
    east = [-sin_lon, cos_lon, 0.0]
    north = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]
    up = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    return [*east, 0.0, *north, 0.0, *up, 0.0, x, y, z, 1.0]


SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_VERSION = 2
SPZ_FRACTIONAL_BITS = 12
SPZ_COLOR_SCALE = 0.15


def pack_spz(
    positions: np.ndarray,
    sh0: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    quat_xyzw: np.ndarray,
) -> bytes:
    """SPZ version 2 (nianticlabs/spz): 24-bit fixed-point positions, byte alphas, colours,
    log-scales and the xyz of a w-positive quaternion, gzip-compressed after a 16-byte header."""
    count = positions.shape[0]
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, count, 0, SPZ_FRACTIONAL_BITS, 0, 0)
    fixed = np.round(positions * (1 << SPZ_FRACTIONAL_BITS)).astype(np.int32).reshape(-1)
    fixed_u = fixed.astype(np.uint32)
    pos_bytes = np.stack(
        [fixed_u & 0xFF, (fixed_u >> 8) & 0xFF, (fixed_u >> 16) & 0xFF], axis=1
    ).astype(np.uint8)
    alphas = np.clip(np.round(sigmoid(opacity_logit) * 255.0), 0, 255).astype(np.uint8)
    colors = np.clip(np.round(sh0 * (SPZ_COLOR_SCALE * 255.0) + 127.5), 0, 255).astype(np.uint8)
    scales = np.clip(np.round((log_scales + 10.0) * 16.0), 0, 255).astype(np.uint8)
    q = quat_xyzw / np.maximum(np.linalg.norm(quat_xyzw, axis=1, keepdims=True), 1e-9)
    q = np.where(q[:, 3:4] < 0, -q, q)
    rotations = np.clip(np.round(q[:, :3] * 127.5 + 127.5), 0, 255).astype(np.uint8)
    raw = (
        header
        + pos_bytes.tobytes()
        + alphas.tobytes()
        + colors.reshape(-1).tobytes()
        + scales.reshape(-1).tobytes()
        + rotations.reshape(-1).tobytes()
    )
    # mtime pinned so the same splat packs to the same bytes (git sees no change).
    return gzip.compress(raw, compresslevel=6, mtime=0)


def build_glb(count: int, pmin: list[float], pmax: list[float], spz: bytes) -> bytes:
    """One point primitive whose attributes live in the SPZ block, as Cesium's tilers write."""
    padded = spz + b"\x00" * ((4 - len(spz) % 4) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "twin splat_tiles"},
        "extensionsUsed": [
            "KHR_materials_unlit",
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "extensionsRequired": [
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # Data is z-up (east, north, up); glTF is y-up. Same swap as Cesium's own splat tiles.
        "nodes": [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "material": 0,
                        "attributes": {
                            "POSITION": 0,
                            "COLOR_0": 1,
                            "KHR_gaussian_splatting:SCALE": 2,
                            "KHR_gaussian_splatting:ROTATION": 3,
                        },
                        "extensions": {
                            "KHR_gaussian_splatting": {
                                "extensions": {
                                    "KHR_gaussian_splatting_compression_spz_2": {"bufferView": 0}
                                }
                            }
                        },
                    }
                ]
            }
        ],
        "materials": [{"extensions": {"KHR_materials_unlit": {}}}],
        "accessors": [
            {"componentType": 5126, "count": count, "type": "VEC3", "min": pmin, "max": pmax},
            {"componentType": 5121, "normalized": True, "count": count, "type": "VEC4"},
            {"componentType": 5126, "count": count, "type": "VEC3"},
            {"componentType": 5126, "count": count, "type": "VEC4"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(spz)}],
        "buffers": [{"byteLength": len(padded)}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(padded)
    return (
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(padded), 0x004E4942)
        + padded
    )


def convert(
    ply: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    height: float,
    max_gaussians: int,
    opacity_min: float,
    geometric_error: float,
) -> dict[str, float | int]:
    data = read_ply(ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    opacity = sigmoid(data["opacity"])
    # A trainer can leave a few NaN gaussians behind; one of them poisons every statistic.
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    xyz = np.where(finite[:, None], xyz, 0.0)
    keep = finite & (opacity >= opacity_min)
    # Outliers far from the bulk (sky floaters) are dropped by a robust radius.
    center = np.median(xyz[keep], axis=0)
    radius = np.linalg.norm(xyz[keep] - center, axis=1)
    keep &= np.linalg.norm(xyz - center, axis=1) <= np.percentile(radius, 99.5) * 1.5
    if keep.sum() > max_gaussians:
        # Keep the most opaque ones: they carry the surface.
        order = np.argsort(-opacity * keep)
        mask = np.zeros_like(keep)
        mask[order[:max_gaussians]] = True
        keep = mask
    xyz = xyz[keep]
    sh0 = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)[keep]
    log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)[keep]
    quat_wxyz = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1)[keep]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    spz = pack_spz(xyz, sh0, data["opacity"][keep], log_scales, quat_xyzw)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splat.glb").write_bytes(
        build_glb(int(xyz.shape[0]), xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist(), spz)
    )
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    mid = (pmin + pmax) / 2
    half = (pmax - pmin) / 2 + 1.0
    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": geometric_error * 8,
        "extensionsUsed": ["3DTILES_content_gltf"],
        "extensions": {
            "3DTILES_content_gltf": {
                "extensionsUsed": [
                    "KHR_gaussian_splatting",
                    "KHR_gaussian_splatting_compression_spz_2",
                ],
                "extensionsRequired": [
                    "KHR_gaussian_splatting",
                    "KHR_gaussian_splatting_compression_spz_2",
                ],
            }
        },
        "root": {
            "transform": enu_to_ecef(lat, lon, height),
            "boundingVolume": {
                "box": [
                    float(mid[0]),
                    float(mid[1]),
                    float(mid[2]),
                    float(half[0]),
                    0,
                    0,
                    0,
                    float(half[1]),
                    0,
                    0,
                    0,
                    float(half[2]),
                ]
            },
            "geometricError": geometric_error,
            "refine": "ADD",
            "content": {"uri": "splat.glb"},
        },
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return {
        "gaussians": int(xyz.shape[0]),
        "dropped": int((~keep).sum()),
        "extent_m": float(max(pmax - pmin)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--height", type=float, default=0.0)
    parser.add_argument("--max-gaussians", type=int, default=400_000)
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--geometric-error", type=float, default=2.0)
    args = parser.parse_args()
    stats = convert(
        args.ply,
        args.out_dir,
        args.lat,
        args.lon,
        args.height,
        args.max_gaussians,
        args.opacity_min,
        args.geometric_error,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
