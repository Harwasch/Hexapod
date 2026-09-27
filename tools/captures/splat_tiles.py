"""Gaussian splat (3DGS PLY) to a 3D Tiles tileset CesiumJS renders.

Reads the PLY that OpenSplat (or any 3DGS trainer) writes, in a local east-north-up frame
around a reference latitude/longitude/height, and writes a level-of-detail hierarchy of
glTF tiles using the KHR_gaussian_splatting extension with SPZ (v2) compressed data, which
is the form CesiumJS 1.145 loads, plus a tileset.json whose root transform places the local
frame on the globe. Every gaussian that passes the opacity and floater filters is written
exactly once: an adaptive octree splits wherever a tile would hold more than
`TILE_GAUSSIANS`, each parent keeps an even subset for the coarse view, and refinement is
ADD (see `convert` for why). A scan within one tile's budget is one tile, as before. The
glTF node carries the z-up to y-up swap the same way Cesium's own sample tilesets do.

Usage:
    python splat_tiles.py splat.ply out_dir --lat 46.84 --lon -91.99 --height 0
        [--opacity-min 0.02] [--tile-gaussians 100000]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
from dataclasses import dataclass, field
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


#: Most gaussians one tile holds; the hierarchy's one tuning knob.
#:
#: A tile is the unit CesiumJS 1.145 refines in, and every selection change costs it a pass
#: over *all* selected tiles: `GaussianSplatPrimitive.update` re-aggregates their positions,
#: scales, rotations and colours into one attribute texture and re-sorts the lot
#: (`@cesium/engine` Source/Scene/GaussianSplatPrimitive.js, "aggregateAttributeValues").
#: Smaller tiles refine more finely and cost more rebuilds and requests; bigger ones make
#: the coarsest view (the root) heavier and the steps between levels coarser. 100k is
#: ~1.4 MB of SPZ (the committed tree packs at 14.5 bytes a gaussian), one second's
#: download on a middling phone connection, and 5M gaussians become ~60 tiles.
TILE_GAUSSIANS = 100_000

#: The octree is cut on an integer grid 2^21 cells a side, so a cell key at any level fits
#: one int64 (3 x 21 bits) and a 1 km scene still resolves half a millimetre. It is also the
#: depth limit: a tile at depth 21 is a leaf whatever it holds, which only coincident
#: gaussians (the same position repeated past the budget) can make it exceed.
GRID_BITS = 21

#: The log-scales SPZ can carry (`pack_spz` stores `(s + 10) * 16` in a byte), and so the
#: range the tiler reasons about sizes in. A garbage scale of +40 must not become exp(40).
SPZ_LOG_SCALE_RANGE = (-10.0, 255 / 16 - 10.0)


@dataclass
class Tile:
    """One node of the hierarchy: the gaussians its own content holds, and its children.

    Refinement is ADD, so `members` of all tiles partition the kept gaussians: a child
    holds none of its ancestors' gaussians and a parent is still drawn under its loaded
    children. `low`/`high` bound the centres of the whole subtree, padded by how far its
    gaussians reach, and enclose every child's box.
    """

    path: tuple[str, ...]
    members: np.ndarray
    geometric_error: float
    low: np.ndarray
    high: np.ndarray
    children: list[Tile] = field(default_factory=list)

    @property
    def uri(self) -> str:
        # The root keeps the single-tile layout's name: `splat.glb` next to `tileset.json`
        # is what SPLAT_TILES.required_members pins, what every tileset published before
        # the hierarchy holds, and what a small scan (one tile) still is, byte for byte.
        # Below it, the octant path, one segment a level ("splat_3-5.glb" is octant 5 of
        # octant 3); a segment of several digits is those octants' leftovers packed together.
        return f"splat_{'-'.join(self.path)}.glb" if self.path else "splat.glb"

    def walk(self) -> list[Tile]:
        """This tile and every descendant, parents before children."""
        out = [self]
        for child in self.children:
            out.extend(child.walk())
        return out


def _spread_bits(values: np.ndarray) -> np.ndarray:
    """Each of 21 bits moved to every third place: one axis of a 63-bit Morton code."""
    v = values.astype(np.uint64) & np.uint64(0x1FFFFF)
    for shift, mask in (
        (32, 0x1F00000000FFFF),
        (16, 0x1F0000FF0000FF),
        (8, 0x100F00F00F00F00F),
        (4, 0x10C30C30C30C30C3),
        (2, 0x1249249249249249),
    ):
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


def _split_levels(codes: np.ndarray) -> np.ndarray:
    """For each adjacent pair of sorted Morton codes, the first octree level that parts them.

    Level L has 2^L cells a side and a cell is `code >> 3 * (GRID_BITS - L)`, so two codes
    share a cell down to the level just above their highest differing bit. Coincident
    codes never part: they get GRID_BITS + 1, past the last level.
    """
    diff = codes[1:] ^ codes[:-1]
    # The highest set bit, from the float exponent, corrected where rounding carried
    # 2^k - 1 up to 2^k (a float64 has 53 bits of mantissa and these have up to 63).
    _, exponent = np.frexp(diff.astype(np.float64))
    high = exponent.astype(np.int64) - 1
    carried = (diff >> np.clip(high, 0, 63).astype(np.uint64)) == 0
    high = np.where(carried & (diff != 0), high - 1, high)
    return np.where(diff == 0, GRID_BITS + 1, GRID_BITS - high // 3)


def _heaviest_per_run(starts: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Positions of the heaviest element of each run (runs begin at `starts`); ties first."""
    peak = np.maximum.reduceat(weight, starts)
    run = np.zeros(weight.size, dtype=np.int64)
    run[starts[1:]] = 1
    run = np.cumsum(run)
    best = np.flatnonzero(weight == peak[run])
    first = np.ones(best.size, dtype=bool)
    first[1:] = run[best[1:]] != run[best[:-1]]
    return best[first]


def _representatives(
    codes: np.ndarray, weight: np.ndarray, depth: int, budget: int
) -> tuple[np.ndarray, int]:
    """At most `budget` of a tile's gaussians, spread evenly over where they are, and the
    octree level they were spread at. `codes` are the tile's Morton codes, sorted.

    The level is the finest at which the occupied cells still fit the budget (occupancy
    only grows with level, and at the tile's own level it is one), and each occupied cell
    gives its heaviest gaussian: most opaque times broadest face, the one that stands for
    the most of what the cell looks like from afar. What is left of the budget goes to the
    heaviest gaussians of the next level's still-empty cells, so a tile is full rather than
    stopping at the last power of two.

    In Morton order every cell at every level is a contiguous run, so this is two linear
    passes and no sort -- which is what makes 5M gaussians take seconds, not minutes.
    """
    parts = _split_levels(codes)
    counts = np.bincount(parts, minlength=GRID_BITS + 2)
    occupancy = 1 + np.cumsum(counts)  # occupancy[L]: cells occupied at level L
    level = max(
        (lvl for lvl in range(depth, GRID_BITS + 1) if occupancy[lvl] <= budget), default=depth
    )
    starts = np.concatenate([[0], np.flatnonzero(parts <= level) + 1])
    chosen = _heaviest_per_run(starts, weight)
    spare = budget - chosen.size
    if spare > 0 and level < GRID_BITS:
        finer = np.concatenate([[0], np.flatnonzero(parts <= level + 1) + 1])
        run = np.zeros(codes.size, dtype=np.int64)
        run[finer[1:]] = 1
        run = np.cumsum(run)
        open_run = np.ones(finer.size, dtype=bool)
        open_run[run[chosen]] = False
        extra = _heaviest_per_run(finer, weight)
        extra = extra[open_run[run[extra]]]
        pick = extra[np.argsort(-weight[extra], kind="stable")[:spare]]
        chosen = np.sort(np.concatenate([chosen, pick]))
    return chosen, level


def build_hierarchy(
    xyz: np.ndarray,
    log_scales: np.ndarray,
    opacity: np.ndarray,
    tile_gaussians: int | None = TILE_GAUSSIANS,
) -> Tile:
    """An adaptive octree over the gaussians, each tile holding at most `tile_gaussians`.

    Only a tile holding more than the budget splits: it keeps an even, heaviest-first
    `budget` of its gaussians as its own content (see `_representatives`) and hands the
    rest to its occupied octants, so empty space gets no tiles and dense places get deep
    ones. Refinement is ADD: nothing is duplicated and nothing is lost -- the members of
    all tiles are exactly the input rows, each once.

    Geometric error, which Cesium projects to pixels and compares with
    `maximumScreenSpaceError`: a leaf is the data at full resolution, so 0. An inner tile's
    is the cell size its representatives were spread at -- within that distance it (with
    its ancestors) has a gaussian wherever the data has one, which is how coarse it is. It
    never grows with depth, with no clamp to make it so: a child's gaussians are a subset
    of its parent's on the same aligned grid, so its occupancy at any level is no greater
    and the level it can afford is no coarser.

    `tile_gaussians=None` writes one tile holding everything: the Living Survey deformer
    needs it (splat indices are only stable while tile selection is; see
    apps/web/src/cesium/splatInternals.ts `isSingleTile`).
    """
    count = xyz.shape[0]
    if count == 0:
        raise SplatFormatError("there are no gaussians to tile")
    if tile_gaussians is not None and tile_gaussians < 1:
        raise ValueError(f"tile_gaussians must be at least 1, not {tile_gaussians}")
    centres = xyz.astype(np.float64)
    sizes = np.sort(np.clip(log_scales.astype(np.float64), *SPZ_LOG_SCALE_RANGE), axis=1)
    # Three standard deviations along the longest axis: where a gaussian stops showing.
    reach = 3.0 * np.exp(sizes[:, 2])
    origin = centres.min(axis=0)
    edge = max(float((centres.max(axis=0) - origin).max()), 1e-3)
    side = 1 << GRID_BITS
    grid = np.clip(np.floor((centres - origin) / edge * side), 0, side - 1).astype(np.int64)
    morton = (
        (_spread_bits(grid[:, 0]) << np.uint64(2))
        | (_spread_bits(grid[:, 1]) << np.uint64(1))
        | _spread_bits(grid[:, 2])
    )
    # Everything below works in Morton order; `order` maps back to PLY rows. Stable, so
    # coincident gaussians keep their PLY order and the output is deterministic.
    order = np.argsort(morton, kind="stable")
    codes = morton[order]
    weight = (opacity.astype(np.float64) * np.exp(sizes[:, 1] + sizes[:, 2]))[order]
    reach = reach[order]
    centres = centres[order]

    def split(index: np.ndarray, depth: int, path: tuple[str, ...]) -> Tile:
        # The 99th percentile of reach, not the maximum: one 10 m background gaussian must
        # not make every box it falls in 20 m wider than the data. Centres are always inside.
        pad = float(np.percentile(reach[index], 99))
        low = centres[index].min(axis=0) - pad
        high = centres[index].max(axis=0) + pad
        if tile_gaussians is None or index.size <= tile_gaussians or depth >= GRID_BITS:
            return Tile(path, np.sort(order[index]), 0.0, low, high)
        chosen, level = _representatives(codes[index], weight[index], depth, tile_gaussians)
        keep = np.ones(index.size, dtype=bool)
        keep[chosen] = False
        rest = index[keep]
        # Morton order makes each octant a contiguous run, in octant order.
        octant = (codes[rest] >> np.uint64(3 * (GRID_BITS - depth - 1))) & np.uint64(7)
        bounds = np.searchsorted(octant, np.arange(9, dtype=np.uint64))
        children: list[Tile] = []
        # Octants that would be leaves are packed together, in octant order, up to the
        # budget: an octree's leftovers are otherwise a spray of tiles holding a handful of
        # gaussians each (45 tiles, a third of them under 30, for the committed tree at a
        # budget of 1000), and each tile costs a request and, in Cesium, a re-aggregation
        # of everything selected. A packed leaf's box is its own gaussians', so it may
        # overlap a sibling's, which 3D Tiles allows.
        packed: list[tuple[str, np.ndarray]] = []

        def flush() -> None:
            if packed:
                rows = np.concatenate([rows for _, rows in packed])
                digits = "".join(value for value, _ in packed)
                children.append(split(rows, depth + 1, (*path, digits)))
                packed.clear()

        for value in range(8):
            rows = rest[bounds[value] : bounds[value + 1]]
            if rows.size == 0:
                continue
            if rows.size > tile_gaussians:
                children.append(split(rows, depth + 1, (*path, str(value))))
                continue
            if sum(held.size for _, held in packed) + rows.size > tile_gaussians:
                flush()
            packed.append((str(value), rows))
        flush()
        for child in children:
            low = np.minimum(low, child.low)
            high = np.maximum(high, child.high)
        members = np.sort(order[index[chosen]])
        return Tile(path, members, edge / (1 << level), low, high, children)

    return split(np.arange(count), 0, ())


def _box(low: np.ndarray, high: np.ndarray) -> list[float]:
    mid = (low + high) / 2
    half = (high - low) / 2
    return [
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


def convert(
    ply: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    height: float,
    opacity_min: float = 0.02,
    tile_gaussians: int | None = TILE_GAUSSIANS,
) -> dict[str, float | int]:
    """Every gaussian that passes the filters, as a level-of-detail tileset.

    There is no top-N cut any more. This used to keep the `max_gaussians` most opaque (400k
    by default) in one tile and drop the rest, so every trained scene lost detail at
    delivery and bigger scenes lost more. Now nothing that passes `opacity_min` and the
    floater radius is dropped, and how much of it is *drawn* is the viewer's budget
    (CesiumJS: `maximumScreenSpaceError`; the Spark viewer: a gaussian budget over the
    tiles, coarsest first) -- decided on the device that pays for it.

    No safety ceiling replaces the cut, because nothing downstream needs one: a viewer
    loads tiles, never the whole set; Cesium's only hard limit is its attribute texture
    (maximumTextureSize^2 / 2 splats *selected at once*, 8.4M at 4096 -- and it raises its
    own screen-space error when it is hit); and packing is linear in memory and near-linear
    in time (5M synthetic gaussians: 30 s and 1.6 GB peak, into 67 tiles and 77 MB).

    Refinement is ADD, and deliberately. CesiumJS 1.145 draws a splat tileset as one
    `GaussianSplatPrimitive` that aggregates `tileset._selectedTiles` into one texture and
    sorts them together (Source/Scene/GaussianSplatPrimitive.js, `update`), so splats from
    different tiles blend correctly with no seams between tiles. Its base traversal
    (Source/Scene/Cesium3DTilesetBaseTraversal.js, `executeTraversal`) *selects* an ADD tile
    whenever it is visited, alongside its children, and a REPLACE tile only until all its
    children have loaded. ADD therefore needs no coarse copy of anything: a parent's
    gaussians are real gaussians of the scene that stay drawn beneath the children's, the
    tiles partition the data, and the full-detail view is exactly the input. REPLACE would
    need parents made of merged, enlarged gaussians (a second, lossy representation of the
    same scene stored alongside the first) for no rendering benefit here. Two further
    facts from that source this layout honours: `transformTile` bakes each tile through
    its own `computedTransform` but the spherical-harmonic frame is taken from the *first*
    selected tile ("All tiles in a typical GS tileset share the same root coordinate
    frame"), so only the root carries a transform and every GLB has the same node matrix;
    and site tilesets keep `skipLevelOfDetail` off for splats (apps/web providers/tiles.ts),
    which is the traversal quoted above.
    """
    data = read_ply(ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    opacity = sigmoid(data["opacity"])
    # A trainer can leave a few NaN gaussians behind; one of them poisons every statistic.
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    xyz = np.where(finite[:, None], xyz, 0.0)
    keep = finite & (opacity >= opacity_min)
    if not keep.any():
        raise SplatFormatError(
            f"{ply.name}: no gaussian is finite and at least {opacity_min} opaque, so there "
            f"is nothing to tile"
        )
    # Outliers far from the bulk (sky floaters) are dropped by a robust radius.
    center = np.median(xyz[keep], axis=0)
    radius = np.linalg.norm(xyz[keep] - center, axis=1)
    keep &= np.linalg.norm(xyz - center, axis=1) <= np.percentile(radius, 99.5) * 1.5
    # PLY order is kept, within each tile too: the synthetic tree's ground-truth labels
    # and the Living Survey rig index the single-tile GLB by it.
    xyz = xyz[keep]
    sh0 = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)[keep]
    log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)[keep]
    quat_wxyz = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1)[keep]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    opacity_logit = data["opacity"][keep]

    root = build_hierarchy(xyz, log_scales, opacity[keep], tile_gaussians)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(tile: Tile) -> dict[str, object]:
        rows = tile.members
        spz = pack_spz(xyz[rows], sh0[rows], opacity_logit[rows], log_scales[rows], quat_xyzw[rows])
        (out_dir / tile.uri).write_bytes(
            build_glb(
                int(rows.size), xyz[rows].min(axis=0).tolist(), xyz[rows].max(axis=0).tolist(), spz
            )
        )
        node: dict[str, object] = {
            "boundingVolume": {"box": _box(tile.low, tile.high)},
            "geometricError": tile.geometric_error,
            "content": {"uri": tile.uri},
            # How many gaussians the content holds, so a viewer can spend a budget on the
            # tiles before fetching any of them (apps/web src/view/tiles.ts).
            "extras": {"gaussians": int(rows.size)},
        }
        if tile.children:
            node["children"] = [write(child) for child in tile.children]
        return node

    root_node = write(root)
    tiles = root.walk()
    size = float((root.high - root.low).max())
    tileset = {
        "asset": {"version": "1.1"},
        # The error of drawing none of it is the size of the thing itself: Cesium skips a
        # tileset whose error projects under `maximumScreenSpaceError`, so a scan appears
        # once it would span about that many pixels, whatever its size.
        "geometricError": size,
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
        # Refinement is stated once, on the root; every tile below inherits it.
        "root": {"transform": enu_to_ecef(lat, lon, height), "refine": "ADD", **root_node},
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    return {
        "gaussians": int(xyz.shape[0]),
        "dropped": int((~keep).sum()),
        "extent_m": float(max(pmax - pmin)),
        "tiles": len(tiles),
        "depth": max(len(tile.path) for tile in tiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--height", type=float, default=0.0)
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument(
        "--tile-gaussians",
        type=int,
        default=TILE_GAUSSIANS,
        help="most gaussians per tile; 0 writes a single tile holding everything",
    )
    args = parser.parse_args()
    stats = convert(
        args.ply,
        args.out_dir,
        args.lat,
        args.lon,
        args.height,
        args.opacity_min,
        args.tile_gaussians or None,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
