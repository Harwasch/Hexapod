"""Gaussian splat (3DGS PLY) to a 3D Tiles tileset CesiumJS renders.

Reads the PLY that OpenSplat (or any 3DGS trainer) writes, in a local east-north-up frame
around a reference latitude/longitude/height, and writes a level-of-detail hierarchy of
glTF tiles using the KHR_gaussian_splatting extension with SPZ (v2) compressed data, which
is the form CesiumJS 1.145 loads, plus a tileset.json whose root transform places the local
frame on the globe. Every gaussian that passes the opacity and floater filters is written
exactly once, in a leaf: an adaptive octree splits wherever a tile would hold more than
`TILE_GAUSSIANS`, each parent holds its subtree *merged* -- one gaussian per occupied cell,
by Hierarchical 3DGS's moment matching -- and refinement is REPLACE (see `convert` for
why). A scan within one tile's budget is one tile, as before. The packer reads the PLY in
windows and sorts through disk, so its memory does not grow with the file. The glTF node
carries the z-up to y-up swap the same way Cesium's own sample tilesets do.

Usage:
    python splat_tiles.py splat.ply out_dir --lat 46.84 --lon -91.99 --height 0
        [--opacity-min 0.02] [--tile-gaussians 100000] [--parents lod_parents.npz]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


@dataclass(frozen=True)
class PlyLayout:
    """Where a binary PLY's vertex block is and how its rows are laid out.

    Enough to read the block in windows (`iter_ply_rows`) without ever holding the file:
    a trained 8M-gaussian PLY with its SH rest is 2 GB, and the packer's memory must not
    grow with it (see `convert`).
    """

    path: Path
    dtype: np.dtype
    offset: int
    count: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.dtype.names or ())


def ply_layout(path: Path) -> PlyLayout:
    """The vertex block of a binary PLY, either byte order, located from its header.

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
        offset = handle.tell()
    order, elements = _parse_header(path, lines)
    vertex = next((element for element in elements if element.name == "vertex"), None)
    if vertex is None:
        found = ", ".join(element.name for element in elements) or "none"
        raise SplatFormatError(f"{path.name} has no `element vertex`; its elements are: {found}")
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
        offset += earlier.dtype(order).itemsize * earlier.count
    dtype = vertex.dtype(order)
    wanted = dtype.itemsize * vertex.count
    available = path.stat().st_size - offset
    if available < wanted:
        raise SplatFormatError(
            f"{path.name} is truncated: the header declares {vertex.count} vertices "
            f"({wanted} bytes of vertex data) and only {max(available, 0)} bytes follow it"
        )
    return PlyLayout(path, dtype, offset, vertex.count)


#: How much of a PLY is mapped at once when it is read in windows. 64 MiB is ~270k rows of
#: a full 3DGS PLY (62 floats) and ~1.2M of a `canonical.ply` (14): big enough that numpy's
#: per-call overheads vanish, small next to the 1.5 GB the packer may use.
CHUNK_BYTES = 64 << 20


def iter_ply_rows(
    layout: PlyLayout, columns: tuple[str, ...], chunk_bytes: int | None = None
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """`(first row, {column: float32 array})` for consecutive windows of the vertex block.

    Each window is its own `np.memmap`, dropped before the next is opened. Mapping the
    whole file at once would put all of it in the address space, and every page read in
    the resident set -- which is what reading in windows avoids.
    """
    missing = [name for name in columns if name not in layout.names]
    if missing:
        raise SplatFormatError(
            f"{layout.path.name} lacks {', '.join(missing)}; a 3DGS splat PLY carries "
            f"x/y/z, f_dc_0-2, opacity, scale_0-2 and rot_0-3"
        )
    rows = max(1, (chunk_bytes or CHUNK_BYTES) // layout.dtype.itemsize)
    for start in range(0, layout.count, rows):
        count = min(rows, layout.count - start)
        window = np.memmap(
            layout.path,
            dtype=layout.dtype,
            mode="r",
            offset=layout.offset + start * layout.dtype.itemsize,
            shape=(count,),
        )
        chunk = {name: window[name].astype(np.float32) for name in columns}
        del window
        yield start, chunk


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Binary PLY, either byte order: the `vertex` element's properties as float32 arrays.

    All of it, in memory, for the callers that want the whole splat at once (the skeleton,
    the pipeline's ingest). `convert` reads in windows instead (`iter_ply_rows`).
    """
    layout = ply_layout(path)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in layout.names}
    for _, chunk in iter_ply_rows(layout, layout.names):
        for name, column in chunk.items():
            parts[name].append(column)
    return {
        name: np.concatenate(columns) if columns else np.zeros(0, np.float32)
        for name, columns in parts.items()
    }


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
#: download on a middling phone connection, and 5M gaussians become ~60 leaves.
TILE_GAUSSIANS = 100_000

#: The octree is cut on an integer grid 2^21 cells a side, so a cell key at any level fits
#: one int64 (3 x 21 bits) and a 1 km scene still resolves half a millimetre. It is also the
#: depth limit: a tile at depth 21 is a leaf whatever it holds, which only coincident
#: gaussians (the same position repeated past the budget) can make it exceed.
GRID_BITS = 21

#: The log-scales SPZ can carry (`pack_spz` stores `(s + 10) * 16` in a byte), and so the
#: range the tiler reasons about sizes in. A garbage scale of +40 must not become exp(40).
SPZ_LOG_SCALE_RANGE = (-10.0, 255 / 16 - 10.0)

#: The most a merged gaussian may be opaque, and so where H3DGS's "falloff" is cut.
#:
#: Merging sets a parent's opacity to `sum(w_i) / S_p` (H3DGS 2406.12080 Sec. 4.1, Eqs. 8-9;
#: gaussian-hierarchy ClusterMerger.cpp: `clustered.opacity = weight_sum /
#: ellipseSurface(clustered.scale)`), which the paper calls *falloff* because it exceeds 1
#: wherever the children overlapped; its own rasteriser then clamps each fragment's alpha
#: at 0.99 (hierarchy-rasterizer forward.cu: `min(0.99f, con_o.w * exp(power))`). SPZ and
#: KHR_gaussian_splatting store an opacity in [0, 1] (a sigmoid, then a byte), so the cut
#: has to happen here instead: at 0.99, the same ceiling that rasteriser applies.
MERGED_OPACITY_MAX = 0.99

#: How far a merged gaussian may be widened to keep what the opacity cut would lose.
#:
#: Cutting falloff F > 1 to 0.99 alone makes a merged surface see-through between its
#: gaussians -- two neighbours a cell apart each reach ~0.23 alpha at the midpoint, 0.41
#: together, where the children they replace were opaque. So the product H3DGS conserves
#: when it merges, opacity x surface (its weights w = o * S, Eq. 8), is conserved here too:
#: the surface grows by F / 0.99, the scales by the square root of that. That is the rule
#: Spark's own LoD applies when `lodInflate` is on ("inflate LoD splats to ensure opacity
#: stays <= 1.0"; spark src/shaders/splatVertex.glsl), and what Cesium ion's own splat
#: tilesets show (coarse splats enlarged: p99 35 m against 17.7 m a level down). Capped at
#: 2x: F far above 4 means many *stacked* layers, and widening a node past its cell smears
#: colour across neighbours instead of filling gaps.
MERGED_INFLATE_MAX = 2.0

#: Records of the Morton-ordered working copy `convert` spills to disk: what a tile needs of
#: a gaussian, float32 as the PLY had it, plus the row it came from (tiles keep PLY order).
RECORD = np.dtype(
    [
        ("row", "<i8"),
        ("xyz", "<f4", (3,)),
        ("f_dc", "<f4", (3,)),
        ("opacity", "<f4"),
        ("scale", "<f4", (3,)),
        ("rot", "<f4", (4,)),
    ]
)
_BUCKET = np.dtype([("pos", "<i8"), *[(name, RECORD.fields[name][0]) for name in RECORD.names]])

#: Records one bucket of the external sort holds, and so what putting one in order costs:
#: 512k records are 36 MB read back and 32 MB of sorted block. A scan below this is one
#: bucket; 8M gaussians are 16.
BUCKET_RECORDS = 1 << 19

PLY_COLUMNS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


@dataclass
class Tile:
    """One node of the hierarchy.

    Refinement is REPLACE (see `convert`): a leaf holds original gaussians, and the leaves
    between them hold every kept gaussian exactly once; a parent holds *merged* gaussians,
    one per occupied cell of its grid at `level`, standing in for everything below it until
    all its children have loaded. `ranges` are the tile's gaussians as half-open runs of
    positions in Morton order: one run for a cell, several for a leaf packed from small
    octants. `low`/`high` bound what the tile and its subtree draw.
    """

    path: tuple[str, ...]
    ranges: list[tuple[int, int]]
    depth: int
    level: int = -1
    children: list[Tile] = field(default_factory=list)
    count: int = 0
    geometric_error: float = 0.0
    low: np.ndarray = field(default_factory=lambda: np.zeros(3))
    high: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def uri(self) -> str:
        # The root keeps the single-tile layout's name: `splat.glb` next to `tileset.json`
        # is what SPLAT_TILES.required_members pins, what every tileset published before
        # the hierarchy holds, and what a small scan (one tile) still is, byte for byte.
        # Below it, the octant path, one segment a level ("splat_3-5.glb" is octant 5 of
        # octant 3); a segment of several digits is those octants' leftovers packed together.
        return f"splat_{'-'.join(self.path)}.glb" if self.path else "splat.glb"

    @property
    def size(self) -> int:
        """Gaussians under this tile: its leaves' originals."""
        return sum(stop - start for start, stop in self.ranges)

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


def morton_codes(centres: np.ndarray, origin: np.ndarray, edge: float) -> np.ndarray:
    """63-bit Morton codes of centres on the 2^21 grid over the cube [origin, origin + edge]."""
    side = 1 << GRID_BITS
    grid = np.clip(
        np.floor((centres.astype(np.float64) - origin) / edge * side), 0, side - 1
    ).astype(np.int64)
    return (
        (_spread_bits(grid[:, 0]) << np.uint64(2))
        | (_spread_bits(grid[:, 1]) << np.uint64(1))
        | _spread_bits(grid[:, 2])
    )


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


def _occupancy(codes: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """`occupancy[L]`: how many cells of level L the sorted `codes` occupy, for every L.

    Counted in chunks (overlapping by one code, so no adjacent pair is missed), because the
    root's codes are every gaussian of the scan and `_split_levels` makes several
    temporaries the size of its input.
    """
    counts = np.zeros(GRID_BITS + 2, dtype=np.int64)
    for start in range(0, max(codes.size - 1, 0), chunk):
        parts = _split_levels(codes[start : start + chunk + 1])
        counts += np.bincount(parts, minlength=GRID_BITS + 2)
    return 1 + np.cumsum(counts)


def merge_level(codes: np.ndarray, depth: int, budget: int) -> tuple[int, int]:
    """The finest grid level at which a tile's occupied cells fit `budget`, and how many.

    That is the level its merged content is built at: one merged gaussian per occupied
    cell. Occupancy only grows with level, and at the tile's own level it is one.
    """
    occupancy = _occupancy(codes)
    level = max(
        (level for level in range(depth, GRID_BITS + 1) if occupancy[level] <= budget),
        default=depth,
    )
    return level, int(occupancy[level])


#: How many times more a parent's children hold than the parent: the octree's own 8.
#:
#: With it, each level of parents holds at most an eighth of the level below, so all the
#: parents together add at most 1/8 + 1/64 + ... = 1/7 (~14%) to the leaves -- the
#: 1/(r - 1) of an r-way hierarchy -- whatever the data. Capping a parent by the tile budget
#: alone does not: a parent just over the budget splits into a few children and could
#: still hold nearly a budget itself, and a scan of surfaces (four occupied octants of
#: eight, not eight) shrinks by four a level, not eight. Measured on the lumpy test scene,
#: the budget alone cost 35%.
PARENT_RATIO = 8


def plan_tree(codes: np.ndarray, tile_gaussians: int | None = TILE_GAUSSIANS) -> Tile:
    """The adaptive octree over Morton-sorted `codes`, each leaf holding at most the budget.

    Only a tile holding more than the budget splits, into its occupied octants, so empty
    space gets no tiles and dense places get deep ones. With REPLACE refinement a parent
    keeps none of its gaussians: all of them go down, and the leaves partition the input
    exactly. The plan needs only the codes (8 bytes a gaussian); the gaussians themselves
    are read tile by tile when the tiles are written.

    `tile_gaussians=None` plans one tile holding everything: the Living Survey deformer
    needs it (splat indices are only stable while tile selection is; see
    apps/web/src/cesium/splatInternals.ts `isSingleTile`).
    """
    count = int(codes.size)
    if count == 0:
        raise SplatFormatError("there are no gaussians to tile")
    if tile_gaussians is not None and tile_gaussians < 1:
        raise ValueError(f"tile_gaussians must be at least 1, not {tile_gaussians}")

    def split(start: int, stop: int, depth: int, path: tuple[str, ...]) -> Tile:
        if tile_gaussians is None or stop - start <= tile_gaussians or depth >= GRID_BITS:
            return Tile(path, [(start, stop)], depth, count=stop - start)
        cell = codes[start:stop]
        # Morton order makes each octant a contiguous run, in octant order: its bounds are
        # where the codes cross the octant's first code, found without touching the rest.
        shift = np.uint64(3 * (GRID_BITS - depth - 1))
        prefix = (int(cell[0]) >> (3 * (GRID_BITS - depth))) << 3
        firsts = np.array([(prefix + value) << int(shift) for value in range(9)], np.uint64)
        bounds = start + np.searchsorted(cell, firsts)
        bounds[8] = stop
        children: list[Tile] = []
        # Octants that would be leaves are packed together, in octant order, up to the
        # budget: an octree's leftovers are otherwise a spray of tiles holding a handful of
        # gaussians each (45 tiles, a third of them under 30, for the committed tree at a
        # budget of 1000), and each tile costs a request and, in Cesium, a re-aggregation
        # of everything selected. A packed leaf's box is its own gaussians', so it may
        # overlap a sibling's, which 3D Tiles allows.
        packed: list[tuple[str, tuple[int, int]]] = []

        def flush() -> None:
            if packed:
                digits = "".join(value for value, _ in packed)
                runs = [run for _, run in packed]
                held = sum(b - a for a, b in runs)
                children.append(Tile((*path, digits), runs, depth + 1, count=held))
                packed.clear()

        for value in range(8):
            low, high = int(bounds[value]), int(bounds[value + 1])
            if high == low:
                continue
            if high - low > tile_gaussians:
                children.append(split(low, high, depth + 1, (*path, str(value))))
                continue
            if sum(b - a for _, (a, b) in packed) + high - low > tile_gaussians:
                flush()
            packed.append((str(value), (low, high)))
        flush()
        # The children are planned first because the parent's allowance is theirs: the
        # budget, and at most an eighth of what its children hold (`PARENT_RATIO`).
        allowance = min(tile_gaussians, max(1, sum(c.count for c in children) // PARENT_RATIO))
        level, held = merge_level(cell, depth, allowance)
        # Never finer than a child's own merged level: the parent is built from its
        # children's cells (`write_tiles`), which exist at that level and coarser, and a
        # parent at least as coarse as each child keeps errors from growing with depth.
        finest = min((c.level for c in children if c.children), default=level)
        if finest < level:
            level = finest
            held = int(_occupancy(cell)[level])
        return Tile(path, [(start, stop)], depth, level, children, count=held)

    return split(0, count, 0, ())


# ------------------------------------------------------------------------ merging (H3DGS)


@dataclass
class Moments:
    """Gaussians, or merged cells of gaussians, as the moments H3DGS merges.

    `weight` is H3DGS's unnormalised w = opacity x surface (s0 s1 + s0 s2 + s1 s2) -- for a
    merged cell, the sum of its members' -- `mean` and `cov` (xx, xy, xz, yy, yz, zz) the
    weighted first and central second moments, `colour` the weighted SH DC term. `key` is
    each row's cell at the level it was merged to, ascending.

    The weights add, so merging merged cells is merging their gaussians: a cell's weight is
    exactly its members' sum (the opacity cut in `to_gaussians` touches only what is
    *written*, never these), and Eqs. 3-4 are the law of total covariance. So each parent
    is built from its children's cells rather than re-reading every gaussian below it,
    the way ClusterMerger.cpp's `mergeRec` merges each child's `merged[0]` -- and the
    result is the same as merging the leaves directly (tests/test_splat_tiles_lod.py checks
    it against a brute-force merge).
    """

    key: np.ndarray
    weight: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    colour: np.ndarray

    def __len__(self) -> int:
        return int(self.key.size)

    @staticmethod
    def concat(parts: list[Moments]) -> Moments:
        return Moments(
            *(np.concatenate([getattr(part, name) for part in parts]) for name in _MOMENT_FIELDS)
        )

    def take(self, index: np.ndarray) -> Moments:
        return Moments(*(getattr(self, name)[index] for name in _MOMENT_FIELDS))


_MOMENT_FIELDS = ("key", "weight", "mean", "cov", "colour")


def _surface(scales: np.ndarray) -> np.ndarray:
    """H3DGS's surface proxy, s0 s1 + s0 s2 + s1 s2 (ClusterMerger.cpp `ellipseSurface`)."""
    return scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2]


def _rotation_matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotations whose columns are each gaussian's axes, from w-first quaternions.

    The 3DGS convention (and gaussian-hierarchy common.h `matrixFromQuat`): covariance is
    R S S^T R^T with R from the normalised quaternion.
    """
    q = quat_wxyz.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ],
        axis=1,
    )


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    """w-first unit quaternions of proper rotation matrices (Shepperd's method, vectorised).

    The branch is chosen per matrix by the largest of trace, R00, R11, R22, so the square
    root is never taken of a small number.
    """
    r = rotations
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    pick = np.argmax(np.stack([trace, r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], 1), axis=1)
    out = np.empty((r.shape[0], 4))
    for case in range(4):
        m = r[pick == case]
        if case == 0:
            s = np.sqrt(np.maximum(1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], 1e-30)) * 2
            q = [s / 4, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s,
                 (m[:, 1, 0] - m[:, 0, 1]) / s]  # fmt: skip
        elif case == 1:
            s = np.sqrt(np.maximum(1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1e-30)) * 2
            q = [(m[:, 2, 1] - m[:, 1, 2]) / s, s / 4, (m[:, 0, 1] + m[:, 1, 0]) / s,
                 (m[:, 0, 2] + m[:, 2, 0]) / s]  # fmt: skip
        elif case == 2:
            s = np.sqrt(np.maximum(1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1e-30)) * 2
            q = [(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, s / 4,
                 (m[:, 1, 2] + m[:, 2, 1]) / s]  # fmt: skip
        else:
            s = np.sqrt(np.maximum(1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], 1e-30)) * 2
            q = [(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s,
                 (m[:, 1, 2] + m[:, 2, 1]) / s, s / 4]  # fmt: skip
        out[pick == case] = np.stack(q, 1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def gaussian_moments(records: np.ndarray, keys: np.ndarray) -> Moments:
    """Leaf gaussians (`RECORD` rows) as moments, each in its own row.

    Scales are clipped to what SPZ can store first, so a parent merges exactly the sizes
    its children are drawn at.
    """
    scales = np.exp(np.clip(records["scale"].astype(np.float64), *SPZ_LOG_SCALE_RANGE))
    opacity = sigmoid(records["opacity"].astype(np.float64))
    # (R S)(R S)^T, batched: a matmul, which is ten times faster here than the einsum.
    axes = _rotation_matrices(records["rot"]) * scales[:, None, :]
    cov3 = axes @ axes.transpose(0, 2, 1)
    return Moments(
        key=keys.astype(np.int64),
        # A floor, not a filter: a zero weight (a gaussian SPZ would round to nothing)
        # must not make a cell of such gaussians divide by zero.
        weight=np.maximum(opacity * _surface(scales), 1e-30),
        mean=records["xyz"].astype(np.float64),
        cov=cov3[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        colour=records["f_dc"].astype(np.float64),
    )


def merge_cells(moments: Moments) -> Moments:
    """One row per distinct key: H3DGS moment matching over each key's rows.

    With w_i normalised within the cell (ClusterMerger.cpp, `weights[i] / weight_sum`):
    mean mu_p = sum w_i mu_i (Eq. 3), covariance Sigma_p = sum w_i (Sigma_i + (mu_i - mu_p)
    (mu_i - mu_p)^T) (Eq. 4), SH (here the DC term) sum w_i c_i -- and the unnormalised
    weight sum is kept, so the result merges again exactly. Rows must be sorted by key;
    each cell is a contiguous run, so this is one `reduceat` per moment and no loop.
    """
    if len(moments) == 0:
        return moments
    key = moments.key
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    run = np.cumsum(np.r_[False, key[1:] != key[:-1]])
    w = moments.weight
    total = np.add.reduceat(w, starts)
    mean = np.add.reduceat(w[:, None] * moments.mean, starts) / total[:, None]
    colour = np.add.reduceat(w[:, None] * moments.colour, starts) / total[:, None]
    # Deviations from the cell's own mean, not raw second moments: E[xx^T] - mu mu^T would
    # cancel catastrophically for centimetre cells a hundred metres from the origin.
    d = moments.mean - mean[run]
    spread = d[:, [0, 0, 0, 1, 1, 2]] * d[:, [0, 1, 2, 1, 2, 2]]
    cov = np.add.reduceat(w[:, None] * (moments.cov + spread), starts) / total[:, None]
    return Moments(key[starts], total, mean, cov, colour)


def coarsen(moments: Moments, levels: int) -> Moments:
    """Cells `levels` levels coarser: keys shifted up an octant per level, then merged."""
    if levels <= 0:
        return moments
    shifted = Moments(
        moments.key >> (3 * levels), moments.weight, moments.mean, moments.cov, moments.colour
    )
    return merge_cells(shifted)


@dataclass
class Gaussians:
    """What a tile writes: arrays in `pack_spz`'s terms."""

    xyz: np.ndarray
    sh0: np.ndarray
    opacity_logit: np.ndarray
    log_scales: np.ndarray
    quat_xyzw: np.ndarray

    @property
    def reach(self) -> np.ndarray:
        """Three standard deviations along each gaussian's longest axis, clipped as SPZ is."""
        top = np.clip(self.log_scales.astype(np.float64), *SPZ_LOG_SCALE_RANGE).max(axis=1)
        return 3.0 * np.exp(top)


def to_gaussians(moments: Moments) -> Gaussians:
    """Merged cells as gaussians: axes and scales from the covariance, opacity from falloff.

    The covariance's eigenvectors are the axes and the square roots of its eigenvalues the
    scales (ClusterMerger.cpp: `SelfAdjointEigenSolver`, then `sqrt(eigenvalues)`), with the
    third axis flipped where needed to make a rotation, not a reflection, and eigenvalues
    kept off zero as its "Working hard..." loop does. Opacity is H3DGS's falloff,
    `sum(w_i) / S_p`, cut to `MERGED_OPACITY_MAX` with the surface widened to keep
    opacity x surface (see `MERGED_INFLATE_MAX` for why and by how much).
    """
    c = moments.cov
    matrices = np.stack(
        [
            np.stack([c[:, 0], c[:, 1], c[:, 2]], 1),
            np.stack([c[:, 1], c[:, 3], c[:, 4]], 1),
            np.stack([c[:, 2], c[:, 4], c[:, 5]], 1),
        ],
        axis=1,
    )
    values, vectors = np.linalg.eigh(matrices)
    floor = np.maximum(values[:, 2:3] * 1e-8, np.exp(2 * SPZ_LOG_SCALE_RANGE[0]))
    values = np.maximum(values, floor)
    flip = np.linalg.det(vectors) < 0
    vectors[flip, :, 2] *= -1
    scales = np.sqrt(values)
    falloff = moments.weight / _surface(scales)
    inflate = np.clip(np.sqrt(falloff / MERGED_OPACITY_MAX), 1.0, MERGED_INFLATE_MAX)
    scales = scales * inflate[:, None]
    opacity = np.clip(falloff / inflate**2, 1e-6, MERGED_OPACITY_MAX)
    quat = _quaternions(vectors)
    return Gaussians(
        xyz=moments.mean.astype(np.float32),
        sh0=moments.colour.astype(np.float32),
        opacity_logit=np.log(opacity / (1 - opacity)).astype(np.float32),
        log_scales=np.log(scales).astype(np.float32),
        quat_xyzw=quat[:, [1, 2, 3, 0]].astype(np.float32),
    )


def records_gaussians(records: np.ndarray) -> Gaussians:
    """Original gaussians, as the PLY had them."""
    return Gaussians(
        xyz=records["xyz"],
        sh0=records["f_dc"],
        opacity_logit=records["opacity"],
        log_scales=records["scale"],
        quat_xyzw=records["rot"][:, [1, 2, 3, 0]],
    )


# ------------------------------------------------------------------ out-of-core packaging


class SortedStore:
    """The kept gaussians as `RECORD`s in Morton order, in a file, read back by position.

    Built by a distribution sort (`build`): the PLY is read once in windows, each record
    appended to the bucket its sorted position falls in, and each bucket -- at most
    `BUCKET_RECORDS` long -- is put in order in memory and appended to the store. Nothing
    the size of the scan is held but the codes and positions (16 bytes a gaussian), and
    every read and write is sequential.
    """

    def __init__(self, path: Path, count: int) -> None:
        self.path = path
        self.count = count

    def read(self, start: int, stop: int) -> np.ndarray:
        with self.path.open("rb") as handle:
            handle.seek(start * RECORD.itemsize)
            return np.fromfile(handle, dtype=RECORD, count=stop - start)

    def read_ranges(self, ranges: list[tuple[int, int]]) -> np.ndarray:
        return np.concatenate([self.read(start, stop) for start, stop in ranges])


def _filter_rows(layout: PlyLayout, opacity_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows that are finite, at least `opacity_min` opaque and not floaters, and their centres.

    Floaters are dropped by a robust radius around the median centre -- the same arithmetic
    on the same float32 values as when the packer held the whole PLY, so the same rows go.
    """
    keep = np.zeros(layout.count, dtype=bool)
    centres = np.empty((layout.count, 3), dtype=np.float32)
    held = 0
    for start, chunk in iter_ply_rows(layout, ("x", "y", "z", "opacity")):
        xyz = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1)
        opacity = sigmoid(chunk["opacity"])
        # A trainer can leave a few NaN gaussians behind; one of them poisons every statistic.
        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
        good = finite & (opacity >= opacity_min)
        keep[start : start + good.size] = good
        centres[held : held + int(good.sum())] = xyz[good]
        held += int(good.sum())
    if held == 0:
        raise SplatFormatError(
            f"{layout.path.name}: no gaussian is finite and at least {opacity_min} opaque, "
            f"so there is nothing to tile"
        )
    centres = centres[:held]
    # Outliers far from the bulk (sky floaters) are dropped by a robust radius.
    center = np.median(centres, axis=0)
    radius = np.linalg.norm(centres - center, axis=1)
    near = radius <= np.percentile(radius, 99.5) * 1.5
    del radius
    keep[np.flatnonzero(keep)[~near]] = False
    return keep, centres[near]


def _sort_to_disk(layout: PlyLayout, keep: np.ndarray, rank: np.ndarray, work: Path) -> SortedStore:
    """Every kept row as a `RECORD` at its Morton position, in a file under `work`.

    `rank[k]` is the sorted position of the k-th kept row. Buckets are ranges of positions,
    so a bucket in order is a stretch of the store in order.
    """
    count = int(rank.size)
    buckets = max(1, -(-count // BUCKET_RECORDS))
    paths = [work / f"bucket-{index}.bin" for index in range(buckets)]
    handles = [path.open("wb") for path in paths]
    try:
        seen = 0
        for start, chunk in iter_ply_rows(layout, PLY_COLUMNS):
            good = keep[start : start + chunk["x"].size]
            rows = start + np.flatnonzero(good)
            positions = rank[seen : seen + rows.size]
            seen += rows.size
            out = np.empty(rows.size, dtype=_BUCKET)
            out["pos"] = positions
            out["row"] = rows
            out["xyz"] = np.stack([chunk[n][good] for n in ("x", "y", "z")], axis=1)
            out["f_dc"] = np.stack([chunk[f"f_dc_{i}"][good] for i in range(3)], axis=1)
            out["opacity"] = chunk["opacity"][good]
            out["scale"] = np.stack([chunk[f"scale_{i}"][good] for i in range(3)], axis=1)
            out["rot"] = np.stack([chunk[f"rot_{i}"][good] for i in range(4)], axis=1)
            which = positions // BUCKET_RECORDS
            order = np.argsort(which, kind="stable")
            edges = np.searchsorted(which[order], np.arange(buckets + 1))
            for index in range(buckets):
                part = out[order[edges[index] : edges[index + 1]]]
                if part.size:
                    handles[index].write(part.tobytes())
    finally:
        for handle in handles:
            handle.close()
    store = SortedStore(work / "sorted.bin", count)
    with store.path.open("wb") as sink:
        for index, path in enumerate(paths):
            part = np.fromfile(path, dtype=_BUCKET)
            path.unlink()
            block = np.empty(part.size, dtype=RECORD)
            slots = part["pos"] - index * BUCKET_RECORDS
            for name in RECORD.names:
                block[name][slots] = part[name]
            sink.write(block.tobytes())
    return store


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


def _write_tile(out_dir: Path, tile: Tile, gaussians: Gaussians) -> None:
    xyz = gaussians.xyz
    spz = pack_spz(
        xyz, gaussians.sh0, gaussians.opacity_logit, gaussians.log_scales, gaussians.quat_xyzw
    )
    (out_dir / tile.uri).write_bytes(
        build_glb(int(xyz.shape[0]), xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist(), spz)
    )
    tile.count = int(xyz.shape[0])


def _bounds(xyz: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres' box padded by the 99th percentile of reach, not the maximum: one 10 m
    background gaussian must not make every box it falls in 20 m wider than the data."""
    pad = float(np.percentile(reach, 99))
    centres = xyz.astype(np.float64)
    return centres.min(axis=0) - pad, centres.max(axis=0) + pad


#: What `build_tiles` hands each tile to: the tile (its box, error and count filled in),
#: what it draws, and -- for a parent -- the cell key of each of its merged gaussians, in
#: the order they are drawn (None for a leaf).
Emit = Callable[["Tile", "Gaussians", "np.ndarray | None"], None]


def build_tiles(root: Tile, codes: np.ndarray, store: SortedStore, edge: float, emit: Emit) -> None:
    """Build every tile's content, children before parents, and fill in counts, errors, boxes.

    A leaf is its originals, in PLY order (the synthetic tree's labels and the Living
    Survey rig index the single-tile GLB by it). A parent is its children's cells merged
    to its own level (`Moments`), which each child hands up coarsened to that level -- so
    every original is read once, by its leaf, and each merged cell is merged once more per
    level above it. `emit` receives each tile's content as it is finished: `write_tiles`
    writes it as a GLB; the GPU stage that optimises the parents (tools/pipeline
    lod_optimise.py) keeps it, so both see the one tree this function builds.

    Geometric error, which Cesium projects to pixels and refines on: a leaf is the data at
    full resolution, so 0. A parent's is what the merge gives away: detail finer than the
    spacing of its merged gaussians (the cell size at its level) or than their own width
    (the median over its gaussians of twice the longest standard deviation, after
    widening), whichever is larger. The median, so a handful of huge background gaussians
    cannot make a whole tile refine early. A parent's level is never finer than a child's
    (`plan_tree`), so its cells are at least as large, and an error above the parent's is
    clamped to it on the way out (`_clamp_errors`): errors never grow with depth.
    """

    def build(tile: Tile, parent_level: int) -> Moments | None:
        if not tile.children:
            records = store.read_ranges(tile.ranges)
            keys = np.concatenate([codes[a:b] for a, b in tile.ranges])
            order = np.argsort(records["row"], kind="stable")
            gaussians = records_gaussians(records[order])
            tile.count = int(gaussians.xyz.shape[0])
            tile.low, tile.high = _bounds(gaussians.xyz, gaussians.reach)
            tile.geometric_error = 0.0
            emit(tile, gaussians, None)
            if parent_level < 0:
                return None
            shift = np.uint64(3 * (GRID_BITS - parent_level))
            return merge_cells(gaussian_moments(records, (keys >> shift).astype(np.int64)))
        parts = []
        for child in tile.children:
            part = build(child, tile.level)
            assert part is not None
            parts.append(part)
        merged = Moments.concat(parts)
        merged = merge_cells(merged.take(np.argsort(merged.key, kind="stable")))
        gaussians = to_gaussians(merged)
        tile.count = int(gaussians.xyz.shape[0])
        cell = edge / (1 << tile.level)
        width = 2.0 * float(np.median(np.exp(gaussians.log_scales.astype(np.float64).max(axis=1))))
        tile.geometric_error = max(cell, width)
        low, high = _bounds(gaussians.xyz, gaussians.reach)
        for child in tile.children:
            low = np.minimum(low, child.low)
            high = np.maximum(high, child.high)
        tile.low, tile.high = low, high
        emit(tile, gaussians, merged.key)
        return coarsen(merged, tile.level - parent_level) if parent_level >= 0 else None

    build(root, -1)
    _clamp_errors(root, math.inf)


def write_tiles(
    root: Tile,
    codes: np.ndarray,
    store: SortedStore,
    out_dir: Path,
    edge: float,
    parents: ParentOverrides | None = None,
) -> int:
    """Write every tile's GLB (`build_tiles`), a parent's from `parents` where it has one.

    An optimised parent replaces what is *drawn* and nothing else: the tile's box, its
    geometric error and the moments handed up to its own parent are the merge's, so the
    tree -- and so the cut a viewer takes through it -- is the one the parents were
    optimised in (`ParentOverrides`). Returns how many parent gaussians were replaced.
    """
    replaced = 0

    def emit(tile: Tile, gaussians: Gaussians, keys: np.ndarray | None) -> None:
        nonlocal replaced
        if keys is not None and parents is not None:
            better = parents.take(tile.uri, keys)
            if better is not None:
                gaussians = better
                replaced += int(keys.size)
        _write_tile(out_dir, tile, gaussians)

    build_tiles(root, codes, store, edge, emit)
    return replaced


def _clamp_errors(tile: Tile, ceiling: float) -> None:
    tile.geometric_error = min(tile.geometric_error, ceiling)
    for child in tile.children:
        _clamp_errors(child, tile.geometric_error)


@dataclass
class Prepared:
    """A PLY filtered, Morton-sorted to disk and planned: everything `build_tiles` needs.

    `store` is readable only inside `prepare`'s `with` block, which owns its working files.
    """

    layout: PlyLayout
    count: int
    pmin: np.ndarray
    pmax: np.ndarray
    codes: np.ndarray
    root: Tile
    store: SortedStore
    edge: float


@contextmanager
def prepare(
    ply: Path, opacity_min: float, tile_gaussians: int | None, work_dir: Path
) -> Iterator[Prepared]:
    """Filter, sort and plan `ply` exactly as `convert` does, the sorted copy in `work_dir`.

    Deterministic: the same PLY and parameters give the same tree, cell for cell -- which
    is what lets a parent be optimised on one machine and written on another
    (`ParentOverrides`).
    """
    layout = ply_layout(ply)
    keep, centres = _filter_rows(layout, opacity_min)
    count = int(centres.shape[0])
    pmin = centres.min(axis=0)
    pmax = centres.max(axis=0)
    origin = pmin.astype(np.float64)
    edge = max(float((pmax.astype(np.float64) - origin).max()), 1e-3)
    step = 1 << 20
    codes = np.concatenate(
        [morton_codes(centres[at : at + step], origin, edge) for at in range(0, count, step)]
    )
    del centres
    # Stable, so coincident gaussians keep their PLY order and the output is deterministic.
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    rank = np.empty(count, dtype=np.int64)
    rank[order] = np.arange(count, dtype=np.int64)
    del order
    root = plan_tree(codes, tile_gaussians)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".splat_tiles-", dir=work_dir) as scratch:
        store = _sort_to_disk(layout, keep, rank, Path(scratch))
        del rank
        yield Prepared(layout, count, pmin, pmax, codes, root, store, edge)


def hierarchy(
    ply: Path, opacity_min: float, tile_gaussians: int | None, work_dir: Path, emit: Emit
) -> Tile:
    """The tree `convert` would write for `ply`, each tile's content handed to `emit`
    instead of written (`build_tiles`). Returns the root, boxes and errors filled in."""
    with prepare(ply, opacity_min, tile_gaussians, work_dir) as tree:
        build_tiles(tree.root, tree.codes, tree.store, tree.edge, emit)
    return tree.root


# ------------------------------------------------------------- optimised parents (H3DGS)

#: The version of the file `ParentOverrides.save` writes.
PARENTS_FORMAT = 1


class ParentOverrideError(ValueError):
    """Optimised parents that do not belong to the tree being written, and why."""


def file_sha256(path: Path) -> str:
    """The file's sha256, read in `CHUNK_BYTES` blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ParentOverrides:
    """Parent gaussians optimised against the photos (Hierarchical 3DGS Sec. 5.1), written
    in place of the merge's, tile by tile.

    Keyed by tile (`Tile.uri`, the octant path) and, within it, by each merged gaussian's
    cell key at the tile's level -- the `keys` `build_tiles` emits, in the order the tile
    draws them. A cell key alone would not do: a parent may sit at the same level as a
    child, whose cells then share keys. The file also carries what the tree was built
    from -- the PLY's sha256 and the two packing parameters -- because the tree is a
    function of exactly those (`prepare`), and parents optimised for any other tree would
    be gaussians in the wrong places. `convert` refuses a mismatch by name
    (`ParentOverrideError`), and the pipeline's `package` then packs the merged parents.
    """

    source_sha256: str
    opacity_min: float
    tile_gaussians: int | None
    uris: list[str]
    offsets: np.ndarray
    keys: np.ndarray
    gaussians: Gaussians

    def mismatch(self, ply: Path, opacity_min: float, tile_gaussians: int | None) -> str:
        """Why these parents do not belong to this packing, or "" when they do."""
        if tile_gaussians != self.tile_gaussians:
            return (
                f"the parents were optimised for tiles of {self.tile_gaussians} gaussians, "
                f"and this packing uses {tile_gaussians}"
            )
        if not math.isclose(opacity_min, self.opacity_min, rel_tol=0.0, abs_tol=1e-9):
            return (
                f"the parents were optimised with opacity_min {self.opacity_min:g}, and "
                f"this packing uses {opacity_min:g}"
            )
        sha = file_sha256(ply)
        if sha != self.source_sha256:
            return (
                f"the parents were optimised on a PLY whose sha256 is "
                f"{self.source_sha256[:12]}..., and {ply.name}'s is {sha[:12]}..."
            )
        return ""

    def take(self, uri: str, keys: np.ndarray) -> Gaussians | None:
        """The optimised gaussians of tile `uri`, if it has any; its cells must be `keys`."""
        if uri not in self.uris:
            return None
        index = self.uris.index(uri)
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        if stop - start != keys.size or not np.array_equal(self.keys[start:stop], keys):
            raise ParentOverrideError(
                f"tile {uri}: the optimised parents are {stop - start} cells, and the tree "
                f"being written has {keys.size} there, or different ones"
            )
        g = self.gaussians
        return Gaussians(
            xyz=g.xyz[start:stop],
            sh0=g.sh0[start:stop],
            opacity_logit=g.opacity_logit[start:stop],
            log_scales=g.log_scales[start:stop],
            quat_xyzw=g.quat_xyzw[start:stop],
        )

    def save(self, path: Path) -> None:
        g = self.gaussians
        with path.open("wb") as handle:
            np.savez(
                handle,
                format=np.int64(PARENTS_FORMAT),
                source_sha256=np.str_(self.source_sha256),
                opacity_min=np.float64(self.opacity_min),
                tile_gaussians=np.int64(self.tile_gaussians or 0),
                uris=np.array(self.uris, dtype=np.str_),
                offsets=np.asarray(self.offsets, dtype=np.int64),
                keys=np.asarray(self.keys, dtype=np.int64),
                xyz=np.asarray(g.xyz, dtype=np.float32),
                sh0=np.asarray(g.sh0, dtype=np.float32),
                opacity_logit=np.asarray(g.opacity_logit, dtype=np.float32),
                log_scales=np.asarray(g.log_scales, dtype=np.float32),
                quat_xyzw=np.asarray(g.quat_xyzw, dtype=np.float32),
            )

    @staticmethod
    def load(path: Path) -> ParentOverrides:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format"])
            if version != PARENTS_FORMAT:
                raise ParentOverrideError(
                    f"{path.name} is parent-override format {version}; this reads {PARENTS_FORMAT}"
                )
            uris = [str(uri) for uri in data["uris"]]
            offsets = data["offsets"].astype(np.int64)
            if offsets.shape != (len(uris) + 1,) or (offsets.size and offsets[0] != 0):
                raise ParentOverrideError(f"{path.name}: its offsets do not index its tiles")
            count = int(offsets[-1])
            names = ("keys", "xyz", "sh0", "opacity_logit", "log_scales", "quat_xyzw")
            arrays = {name: data[name] for name in names}
            for name, array in arrays.items():
                if array.shape[0] != count or not np.isfinite(array).all():
                    raise ParentOverrideError(
                        f"{path.name}: {name} has {array.shape[0]} rows for {count} parents, "
                        f"or a value that is not finite"
                    )
            budget = int(data["tile_gaussians"])
            return ParentOverrides(
                source_sha256=str(data["source_sha256"]),
                opacity_min=float(data["opacity_min"]),
                tile_gaussians=budget or None,
                uris=uris,
                offsets=offsets,
                keys=arrays["keys"].astype(np.int64),
                gaussians=Gaussians(
                    xyz=arrays["xyz"].astype(np.float32),
                    sh0=arrays["sh0"].astype(np.float32),
                    opacity_logit=arrays["opacity_logit"].astype(np.float32),
                    log_scales=arrays["log_scales"].astype(np.float32),
                    quat_xyzw=arrays["quat_xyzw"].astype(np.float32),
                ),
            )


def convert(
    ply: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    height: float,
    opacity_min: float = 0.02,
    tile_gaussians: int | None = TILE_GAUSSIANS,
    work_dir: Path | None = None,
    parents: ParentOverrides | None = None,
) -> dict[str, float | int]:
    """Every gaussian that passes the filters, as a level-of-detail tileset.

    There is no top-N cut. Nothing that passes `opacity_min` and the floater radius is
    dropped, and how much of it is *drawn* is the viewer's budget (CesiumJS:
    `maximumScreenSpaceError`, and a splat-count cap in SiteManager.ts; the Spark viewer:
    Spark's own LoD at the phone's Detail count) -- decided on the device that pays for it.

    **Refinement is REPLACE, with merged parents.** Each leaf holds original gaussians and
    the leaves hold every kept gaussian exactly once; each parent holds one gaussian per
    occupied cell of its grid, merged from everything under it by H3DGS's moment matching
    (`merge_cells`, `to_gaussians`). What this replaced was ADD with *thinned* parents --
    the heaviest original per cell, unenlarged -- which is the prune-only baseline LapisGS
    (2408.14823, Tab. 2) measures worst of all its coarse levels (SSIM 0.548, LPIPS 0.314,
    against 0.957 / 0.052 for its best), and which every shipping streamer avoids: Cesium
    ion's own splat tilesets are REPLACE with enlarged coarse splats, Spark and PlayCanvas
    merge. A parent is drawn only until all its children have loaded (CesiumJS 1.145
    Source/Scene/Cesium3DTilesetBaseTraversal.js, `updateAndPushChildren`: "For traditional
    replacement refinement only refine if all children are loaded", non-visible children
    included), so a REPLACE tileset has no holes while it streams. Storage grows by the
    parents, about 1/(8 - 1) of the leaves for an octree (`stats["parent_gaussians"]`).

    Two further facts from the Cesium source this layout honours: `transformTile` bakes
    each tile through its own `computedTransform` but the spherical-harmonic frame is taken
    from the *first* selected tile ("All tiles in a typical GS tileset share the same root
    coordinate frame"), so only the root carries a transform and every GLB has the same
    node matrix; and site tilesets keep `skipLevelOfDetail` off for splats (apps/web
    providers/tiles.ts), which is the traversal quoted above.

    **Out of core.** The PLY is read in windows (`iter_ply_rows`), twice: once for the
    filters and the Morton codes, once to spill every kept gaussian, in Morton order, to a
    working file (`SortedStore`, a distribution sort through buckets on disk). Tiles are
    then written one at a time from that file. What is held for the whole scan is 12 bytes
    a gaussian of centres during the first pass, then its codes and sorted positions (16):
    8M gaussians (a 2 GB trained PLY) package at a 380 MB peak, against the 1.5 GB the
    plan allows (tests/test_splat_tiles_memory.py). The
    working files go in `work_dir` (by default a temporary directory beside `out_dir`, on
    the same disk rather than in a RAM-backed /tmp) and are removed afterwards.

    **Optimised parents.** `parents`, when given, are parent gaussians optimised against
    the capture's photos on the GPU (tools/pipeline lod_optimise.py, Hierarchical 3DGS
    Sec. 5.1) for exactly this PLY and these parameters; each parent tile draws them in
    place of its merge, and nothing else changes (`write_tiles`). A set built for another
    PLY or other parameters is refused before anything is written (`ParentOverrideError`).
    """
    if parents is not None:
        why = parents.mismatch(ply, opacity_min, tile_gaussians)
        if why:
            raise ParentOverrideError(why)
    out_dir.mkdir(parents=True, exist_ok=True)
    with prepare(ply, opacity_min, tile_gaussians, work_dir or out_dir.parent) as tree:
        replaced = write_tiles(tree.root, tree.codes, tree.store, out_dir, tree.edge, parents)
    root, layout, count, pmin, pmax = tree.root, tree.layout, tree.count, tree.pmin, tree.pmax

    def node(tile: Tile) -> dict[str, object]:
        entry: dict[str, object] = {
            "boundingVolume": {"box": _box(tile.low, tile.high)},
            "geometricError": tile.geometric_error,
            "content": {"uri": tile.uri},
            # How many gaussians the content holds, so a viewer can spend a budget on the
            # tiles before fetching any of them (apps/web src/view/tiles.ts, and the
            # globe's splat-count cap in src/cesium/SiteManager.ts).
            "extras": {"gaussians": tile.count},
        }
        if tile.children:
            entry["children"] = [node(child) for child in tile.children]
        return entry

    tiles = root.walk()
    size = float((root.high - root.low).max())
    tileset = {
        "asset": {"version": "1.1"},
        # The error of drawing none of it is the size of the thing itself: Cesium skips a
        # tileset whose error projects under `maximumScreenSpaceError`, so a scan appears
        # once it would span about that many pixels, whatever its size.
        "geometricError": max(size, 2 * root.geometric_error),
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
        "root": {"transform": enu_to_ecef(lat, lon, height), "refine": "REPLACE", **node(root)},
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    leaves = sum(tile.count for tile in tiles if not tile.children)
    parent_count = sum(tile.count for tile in tiles if tile.children)
    return {
        "gaussians": count,
        "dropped": int(layout.count - count),
        "extent_m": float(max(pmax - pmin)),
        "tiles": len(tiles),
        "depth": max(len(tile.path) for tile in tiles),
        # What the coarse levels cost on top of the scan itself: the merged parents.
        "parent_gaussians": parent_count,
        "storage_overhead": round(parent_count / max(leaves, 1), 4),
        # How many of those were optimised against the photos (`ParentOverrides`); 0
        # when none were given.
        "optimised_parent_gaussians": replaced,
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
    parser.add_argument(
        "--parents",
        type=Path,
        default=None,
        help="optimised parents (lod_parents.npz) to draw in place of the merged ones",
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
        parents=ParentOverrides.load(args.parents) if args.parents else None,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
