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


class SplatFormatError(ValueError):
    """A splat file this reader cannot honestly turn into gaussians, and why.

    A `ValueError`, not a `SystemExit`: this module is imported as a library by the
    pipeline (`tools/pipeline/captures_bridge.py`), and a `SystemExit` raised inside a
    stage escapes every `except Exception` on the way out — the run would die without the
    stage that failed ever being named.
    """


#: Every scalar type a PLY header may name, and the numpy type it is.
#:
#: The short list this used to carry (`float/float32/double/uchar/int`) is why a
#: PlayCanvas/SuperSplat compressed export died on an unhandled `KeyError: 'uint'`.
PLY_SCALARS = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_BYTE_ORDER = {"binary_little_endian": "<", "binary_big_endian": ">"}


class _Element:
    """One `element` block of a PLY header: its name, its count, its properties."""

    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.count = count
        self.types: list[str] = []
        self.names: list[str] = []
        self.lists: list[str] = []

    def dtype(self, order: str) -> np.dtype:
        return np.dtype(
            [(n, order + PLY_SCALARS[t]) for n, t in zip(self.names, self.types, strict=True)]
        )


def _parse_header(path: Path, lines: list[str]) -> tuple[str, list[_Element]]:
    """Byte order and the element blocks, in file order.

    Properties belong to the element that most recently opened — which is the whole of
    A0 #3's second half. Collecting them into one flat list meant a mesh PLY's trailing
    `element face` contributed `property list uchar int vertex_indices` to the *vertex*
    dtype, and the vertex block was then read at the wrong stride: no exception, just
    `|x| max = 1.7e38` and a `ValueError: zero-size array to reduction` twenty lines
    later, in `convert`.
    """
    if not lines or lines[0].strip() != "ply":
        raise SplatFormatError(f"{path.name} does not start with 'ply'; it is not a PLY file")
    order = ""
    elements: list[_Element] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1] if len(parts) > 1 else ""
            if fmt == "ascii":
                raise SplatFormatError(
                    f"{path.name} is an ASCII PLY, and only binary PLY is supported. "
                    f"Re-export it as binary (in MeshLab: 'Binary encoding'), or convert "
                    f"it with a tool that writes binary_little_endian"
                )
            order = _BYTE_ORDER.get(fmt, "")
            if not order:
                raise SplatFormatError(
                    f"{path.name} declares format {fmt!r}; expected binary_little_endian, "
                    f"binary_big_endian or ascii"
                )
        elif parts[0] == "element":
            if len(parts) < 3 or not parts[2].lstrip("-").isdigit():
                raise SplatFormatError(f"{path.name}: malformed element line {line!r}")
            elements.append(_Element(parts[1], int(parts[2])))
        elif parts[0] == "property":
            if not elements:
                raise SplatFormatError(f"{path.name}: property before any element: {line!r}")
            if parts[1] == "list":
                elements[-1].lists.append(parts[-1])
                continue
            if parts[1] not in PLY_SCALARS:
                raise SplatFormatError(
                    f"{path.name}: property {parts[-1]!r} of element "
                    f"{elements[-1].name!r} has type {parts[1]!r}, which is not a PLY "
                    f"scalar type. Known types: {', '.join(sorted(PLY_SCALARS))}"
                )
            elements[-1].types.append(parts[1])
            elements[-1].names.append(parts[-1])
    if not order:
        raise SplatFormatError(f"{path.name}: the header has no `format` line")
    return order, elements


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Binary PLY, either byte order: the `vertex` element's properties as float32 arrays.

    Elements before `vertex` are skipped by their own stride, so a compressed export
    (PlayCanvas/SuperSplat write `element chunk` first) reads the vertices it means to;
    elements after it are never read at all, so a mesh PLY's `element face` cannot leak
    into the vertex dtype.
    """
    with path.open("rb") as handle:
        lines: list[str] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise SplatFormatError(f"{path.name}: the header has no `end_header` line")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            lines.append(line)
            if len(lines) > 10000:
                raise SplatFormatError(f"{path.name}: no `end_header` in the first 10000 lines")
        order, elements = _parse_header(path, lines)
        vertex = next((element for element in elements if element.name == "vertex"), None)
        if vertex is None:
            found = ", ".join(element.name for element in elements) or "none"
            raise SplatFormatError(
                f"{path.name} has no `element vertex`; its elements are: {found}"
            )
        if vertex.lists:
            raise SplatFormatError(
                f"{path.name}: the vertex element has list properties "
                f"({', '.join(vertex.lists)}), which a gaussian splat never has"
            )
        if not vertex.names:
            raise SplatFormatError(f"{path.name}: the vertex element declares no properties")
        for earlier in elements:
            if earlier is vertex:
                break
            if earlier.lists:
                raise SplatFormatError(
                    f"{path.name}: element {earlier.name!r} comes before the vertex element "
                    f"and has a list property ({', '.join(earlier.lists)}), so the vertex "
                    f"data cannot be located without parsing it"
                )
            handle.seek(earlier.dtype(order).itemsize * earlier.count, 1)
        dtype = vertex.dtype(order)
        wanted = dtype.itemsize * vertex.count
        payload = handle.read(wanted)
        if len(payload) < wanted:
            raise SplatFormatError(
                f"{path.name} is truncated: the header declares {vertex.count} vertices "
                f"({wanted} bytes of vertex data) and only {len(payload)} bytes follow it"
            )
        data = np.frombuffer(payload, dtype=dtype, count=vertex.count)
    return {name: data[name].astype(np.float32) for name in vertex.names}


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


#: Alpha 0 and 255 are +/-inf as a logit, so the probability is clamped before the log.
#:
#: 0.25/255 is not an arbitrary epsilon: it is a quarter of a byte step, which is the
#: largest clamp that still rounds back to the byte it came from (`0.25 -> 0`,
#: `254.75 -> 255`). A half-step clamp would send 255 back as 254.
SPZ_ALPHA_EPS = 0.25 / 255.0

#: Position (3 x 24-bit), alpha, colour, log-scale and rotation bytes, per gaussian, in
#: version 2. Version 3 stores a fourth rotation byte (see `_smallest_three`).
SPZ_BYTES_PER_GAUSSIAN = 19

#: The gzip-framed versions this reads. 2 stores a quaternion's first three components as
#: bytes; 3 stores its smallest three in 10 bits each, plus the index of the largest.
#: Version 1 (float16 positions) was never released, per nianticlabs/spz's own loader,
#: and version 4 is a different container (a plaintext `NGSP` header and ZSTD streams).
SPZ_READABLE_VERSIONS = (2, 3)


def unpack_spz(blob: bytes) -> dict[str, np.ndarray]:
    """The inverse of `pack_spz`: a `.spz` file back to the arrays `convert` consumes.

    Scaniverse exports `.spz` natively and it is the format this module already writes,
    so this is the whole of what Lane 1 needs to ingest a phone scan. Lossless where the
    packing was (A0 measured position error exactly 0 round-tripping the committed
    fixture, because `synthetic_tree.py` snaps positions to the 1/4096 grid), lossy where
    it was not: `unpack -> pack` differs by one byte in 228,016 at a rotation rounding
    boundary, so no SPZ round trip belongs inside a byte-identity gate.

    Versions 2 and 3 both occur in the wild: of five public Scaniverse share-page files
    read on 2026-09-23, three were version 2 and two version 3, and Spark's sample set
    has one version 3 among twenty-eight. The layout of the rotation bytes is the only
    difference, and it is read here exactly as nianticlabs/spz's
    `unpackQuaternionSmallestThree` does. Spherical-harmonic bytes after the rotations
    are not read -- `canonical.ply` carries the DC term only.

    Nothing here converts axes. SPZ's specification says the stream is right/up/back
    unless an extension says otherwise, and every public sample checked disagreed with
    it in one direction or the other -- which is why the pipeline's `ingest_splat` owns
    the up axis, and why this reader returns the stored coordinates untouched.
    """
    if blob[:4] == b"NGSP":
        raise SplatFormatError(
            "SPZ version 4 (a plaintext NGSP header over ZSTD-compressed streams) is not "
            "supported; this reads the gzip-framed versions 2 and 3. Re-export as an "
            "earlier SPZ version or as a 3DGS .ply"
        )
    try:
        raw = gzip.decompress(blob)
    except (OSError, EOFError) as error:
        raise SplatFormatError(f"not an SPZ file: it does not gunzip ({error})") from error
    if len(raw) < 16:
        raise SplatFormatError("not an SPZ file: fewer than 16 bytes after decompression")
    magic, version, count, _sh, fractional_bits, _flags, _reserved = struct.unpack_from(
        "<IIIBBBB", raw, 0
    )
    if magic != SPZ_MAGIC:
        raise SplatFormatError(
            f"not an SPZ file: magic is 0x{magic:08X}, expected 0x{SPZ_MAGIC:08X}"
        )
    if version not in SPZ_READABLE_VERSIONS:
        raise SplatFormatError(
            f"SPZ version {version} is not supported; this reads versions "
            f"{', '.join(str(v) for v in SPZ_READABLE_VERSIONS)}"
        )
    rotation_bytes = 3 if version == 2 else 4
    per_gaussian = SPZ_BYTES_PER_GAUSSIAN - 3 + rotation_bytes
    # 9 bytes of position (three 24-bit components), then one alpha, three colours, three
    # log-scales and three (or four) rotation bytes, after a 16-byte header. The
    # committed fixture is 16 + 19 * 12000 = 228,016 bytes, which is the denominator in
    # A0 #4's "one byte in 228,016".
    wanted = 16 + per_gaussian * count
    if len(raw) < wanted:
        raise SplatFormatError(
            f"SPZ is truncated: {count} gaussians need {wanted} bytes and it has {len(raw)}"
        )
    body = np.frombuffer(raw, dtype=np.uint8, count=per_gaussian * count, offset=16)
    end = 9 * count
    packed = body[:end].reshape(count, 3, 3).astype(np.uint32)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    # 24-bit two's complement, the same truncation `pack_spz` relies on going the other way.
    signed = (fixed.astype(np.int64) ^ 0x800000) - 0x800000
    xyz = (signed / float(1 << fractional_bits)).astype(np.float32)
    alphas = body[end : end + count].astype(np.float32) / 255.0
    colors = body[end + count : end + 4 * count].reshape(count, 3).astype(np.float32)
    scales = body[end + 4 * count : end + 7 * count].reshape(count, 3).astype(np.float32)
    rotations = body[end + 7 * count : end + (7 + rotation_bytes) * count].reshape(
        count, rotation_bytes
    )
    sh0 = (colors - 127.5) / (SPZ_COLOR_SCALE * 255.0)
    log_scales = scales / 16.0 - 10.0
    if version == 2:
        quat_xyz = (rotations.astype(np.float32) - 127.5) / 127.5
        quat_w = np.sqrt(np.clip(1.0 - (quat_xyz**2).sum(axis=1), 0.0, 1.0))
    else:
        quat_xyzw = _smallest_three(rotations)
        quat_xyz, quat_w = quat_xyzw[:, :3], quat_xyzw[:, 3]
    alpha = np.clip(alphas, SPZ_ALPHA_EPS, 1.0 - SPZ_ALPHA_EPS)
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    columns = {f"f_dc_{i}": sh0[:, i] for i in range(3)}
    columns.update({f"scale_{i}": log_scales[:, i] for i in range(3)})
    columns.update({f"rot_{i + 1}": quat_xyz[:, i].astype(np.float32) for i in range(3)})
    return {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "opacity": opacity,
        "rot_0": quat_w.astype(np.float32),
        **columns,
    }


def _smallest_three(rotations: np.ndarray) -> np.ndarray:
    """SPZ version 3's rotation bytes to x, y, z, w quaternions.

    A little-endian u32 per gaussian: the top two bits index the largest component, and
    the three others follow as 10-bit fields -- nine bits of magnitude scaled by
    1/sqrt(2) and a sign bit -- packed so that the highest-indexed component sits in the
    lowest bits. The largest is rebuilt from unit length and is always non-negative,
    which is the sign convention the writer normalised to.
    """
    words = rotations.astype(np.uint32)
    comp = words[:, 0] | (words[:, 1] << 8) | (words[:, 2] << 16) | (words[:, 3] << 24)
    largest = (comp >> 30).astype(np.int64)
    out = np.zeros((comp.shape[0], 4), dtype=np.float64)
    mask = np.uint32((1 << 9) - 1)
    remaining = comp.copy()
    squares = np.zeros(comp.shape[0], dtype=np.float64)
    for index in range(3, -1, -1):
        present = largest != index
        magnitude = (remaining & mask).astype(np.float64)
        negative = ((remaining >> 9) & 1).astype(bool)
        value = math.sqrt(0.5) * magnitude / float(mask)
        value = np.where(negative, -value, value)
        out[present, index] = value[present]
        squares += np.where(present, value * value, 0.0)
        remaining = np.where(present, remaining >> 10, remaining)
    out[np.arange(comp.shape[0]), largest] = np.sqrt(np.clip(1.0 - squares, 0.0, 1.0))
    return out.astype(np.float32)


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
