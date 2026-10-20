"""Gaussian splat (3DGS PLY) to a 3D Tiles tileset CesiumJS renders.

Reads the PLY that OpenSplat (or any 3DGS trainer) writes, in a local east-north-up frame
around a reference latitude/longitude/height, and writes a level-of-detail hierarchy of
glTF tiles using the KHR_gaussian_splatting extension with SPZ (v2) compressed data, which
is the form CesiumJS 1.145 loads, plus a tileset.json whose root transform places the local
frame on the globe. Every gaussian that passes the opacity and floater filters is written
exactly once, in a leaf: an adaptive octree splits wherever a tile would hold more than
`TILE_GAUSSIANS`, each parent holds its subtree *merged* -- one gaussian per occupied cell,
by moment matching (`splat_merge`) -- and refinement is REPLACE (see `convert` for why). A
scan within one tile's budget is one tile, as before. The packer reads the PLY in windows
and sorts through disk, so its memory does not grow with the file. The glTF node
carries the z-up to y-up swap the same way Cesium's own sample tilesets do. A trained PLY's
spherical harmonics (`f_rest_*`, up to degree 3) ride along in every tile, so the view-
dependent colour the trainer learned is drawn (see `convert`). Beside the tiles goes
`collision.bin`, the scan's solid cells for the web clients' camera collision and picking,
declared on the root tile (`COLLISION_FORMAT`), and `viewcones.bin`, the directions each
part of it was seen from (`view_cones`), declared beside it. When the run measured one,
the epoch's confidence grid (`cell_grid`, `confidence.bin`) is declared there too, and each
tile says how confident its least confident cell is (`extras.confidenceMin`).

Usage:
    python splat_tiles.py splat.ply out_dir --lat 46.84 --lon -91.99 --height 0
        [--opacity-min 0.02] [--tile-gaussians 100000] [--parents lod_parents.npz]
        [--sh-degree 0-3] [--confidence confidence/confidence.json]
    python splat_tiles.py collision splat.ply out_dir [--tileset out_dir/tileset.json]
        [--opacity-min 0.02]
    python splat_tiles.py viewcones splat.ply|tileset.json out_dir [--tileset tileset.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

import cell_grid
import splat_merge
import view_cones
from splat_merge import Moments, coarsen, merge_cells

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
#: The largest magnitude a 24-bit two's-complement position component stores. At
#: `SPZ_FRACTIONAL_BITS` that is 8,388,607 / 4096 = 2047.99976 m either side of the tile's
#: frame; `pack_spz` refuses anything past it rather than letting the top bits fall away.
SPZ_FIXED_MAX = (1 << 23) - 1
#: The most negative one, a step further out than `SPZ_FIXED_MAX`: -2048 m exactly at 12
#: bits. The range checks take it; `fractional_bits_for` sizes a tile by magnitude alone,
#: so it keeps a tile clear of it on both sides and never relies on the extra step.
SPZ_FIXED_MIN = -(1 << 23)
#: The fewest fractional bits a header can declare: whole metres, +/-8,388 km.
SPZ_MIN_FRACTIONAL_BITS = 0
#: How far from its frame a tile's centres may reach before the tile is given an origin of
#: its own (`tile_frame`). Not a calibration: `SPZ_FIXED_MAX` steps of 2^-12 m is
#: 2047.99976 m, and the metre short of it keeps a whole-metre origin and float32 rounding
#: of a centre clear of the edge.
TILE_FIT_M = 2047.0

# ------------------------------------------------------------- spherical harmonics (SH)

#: SH coefficients per colour channel above the DC term, by degree: 0, 3, 8, 15
#: (nianticlabs/spz `dimForDegree`). Degree 3 is the most any viewer here draws: CesiumJS
#: 1.145 evaluates bands 1-3 (PrimitiveGaussianSplatVS.glsl `evaluateSH`), Spark 2.2 the
#: same, and nianticlabs/spz warns that degree 4 in a version-2 file is unreadable by the
#: loaders these tiles are for. A PLY with a fourth band packs its first three.
SH_DIMS = (0, 3, 8, 15)
SH_MAX_DEGREE = 3

#: Bits kept of each SH byte, as nianticlabs/spz packs by default (`DEFAULT_SH1_BITS`,
#: `DEFAULT_SH_REST_BITS`): 5 for the degree-1 coefficients, 4 for degrees 2 and 3. The
#: dropped low bits are zeros a reader never needs to know about -- the byte still means
#: `(b - 128) / 128` -- and zeros are what gzip compresses away.
SPZ_SH1_BITS = 5
SPZ_SH_REST_BITS = 4


def sh_degree_for_dim(dim: int) -> int:
    """nianticlabs/spz `degreeForDim`, stopped at `SH_MAX_DEGREE`: the degree whose bands
    fit in `dim` coefficients a channel (a PLY with 10 a channel is degree 2, its extra
    two ignored, exactly as spz's own PLY loader reads it)."""
    return max(degree for degree, need in enumerate(SH_DIMS) if need <= dim)


def ply_sh_degree(layout: PlyLayout) -> int:
    """The SH degree a trained PLY carries in its `f_rest_*` properties; 0 without them.

    3DGS trainers (the paper's own, arXiv 2308.04079; gsplat's exporter; OpenSplat) write
    the rest of the SH as `f_rest_0..f_rest_{3K-1}`, channel-major: all K of red, then
    green, then blue.
    """
    rest = [name for name in layout.names if name.startswith("f_rest_")]
    if not rest:
        return 0
    if sorted(rest) != sorted(f"f_rest_{i}" for i in range(len(rest))) or len(rest) % 3:
        raise SplatFormatError(
            f"{layout.path.name}: its {len(rest)} f_rest_* properties are not f_rest_0.."
            f"f_rest_{{3K-1}} (K coefficients for each of red, green and blue)"
        )
    return sh_degree_for_dim(len(rest) // 3)


def ply_sh_columns(layout: PlyLayout, degree: int) -> tuple[str, ...]:
    """The `f_rest_*` names of the first `degree` bands, in SPZ's order: coefficient-major,
    colour innermost (`sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, ...`), so the columns read in this
    order reshape straight to (n, coefficients, 3). The PLY is channel-major, K a channel
    (`ply_sh_degree`); nianticlabs/spz `loadSplatFromPly` reorders the same way."""
    stride = sum(1 for name in layout.names if name.startswith("f_rest_")) // 3
    return tuple(f"f_rest_{c * stride + j}" for j in range(SH_DIMS[degree]) for c in range(3))


def _round_half_away(values: np.ndarray) -> np.ndarray:
    """C's `std::round`: halves away from zero (numpy's `round` sends them to even)."""
    return np.trunc(values + np.copysign(0.5, values))


def quantize_sh(sh_rest: np.ndarray) -> np.ndarray:
    """SH coefficients (n, K, 3) to SPZ bytes, exactly as nianticlabs/spz `quantizeSH` does.

    `q = round(x * 128) + 128`, then rounded to the centre of its bucket of `2^(8 - bits)`
    (`(q + bucket / 2) / bucket * bucket`, integers) and clamped to a byte -- so 0 always
    lands on 128, and the top bucket clamps to 255. Degree-1 coefficients (the first three)
    keep `SPZ_SH1_BITS`, the rest `SPZ_SH_REST_BITS`. A coefficient that is not finite is
    written as 0 (spz's own cast would be undefined).
    """
    return _quantize_sh(sh_rest, _sh_buckets(sh_rest.shape[1])[None, :, None])


def _sh_buckets(dim: int) -> np.ndarray:
    """The bucket of each of `dim` coefficients: `2^(8 - bits)`, degree 1 first."""
    bucket = np.full(dim, 1 << (8 - SPZ_SH_REST_BITS), dtype=np.int16)
    bucket[: SH_DIMS[1]] = 1 << (8 - SPZ_SH1_BITS)
    return bucket


def _quantize_sh(values: np.ndarray, bucket: int | np.ndarray) -> np.ndarray:
    x = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Past +/-2 every coefficient clamps to 0 or 255 anyway; held to +/-4, x * 128 and its
    # rounding are exact in float32 and the rest fits an int16.
    q = _round_half_away(np.clip(x, -4.0, 4.0) * np.float32(128.0)).astype(np.int16) + 128
    return np.clip((q + bucket // 2) // bucket * bucket, 0, 255).astype(np.uint8)


def unquantize_sh(sh_bytes: np.ndarray) -> np.ndarray:
    """SPZ SH bytes back to coefficients, as nianticlabs/spz `unquantizeSH`: `(b - 128) / 128`.

    `quantize_sh` of the result gives the same bytes back: every byte it writes is a bucket
    centre (or the 255 its top bucket clamps to), which re-quantises to itself.
    """
    return (sh_bytes.astype(np.float32) - 128.0) / 128.0


def pack_spz(
    positions: np.ndarray,
    sh0: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    quat_xyzw: np.ndarray,
    sh_rest: np.ndarray | None = None,
    fractional_bits: int = SPZ_FRACTIONAL_BITS,
) -> bytes:
    """SPZ version 2 (nianticlabs/spz): 24-bit fixed-point positions, byte alphas, colours,
    log-scales and the xyz of a w-positive quaternion, gzip-compressed after a 16-byte header.

    `sh_rest`, (n, K, 3) with K one of 3/8/15, adds SH degree 1-3: the header's degree
    byte says which, and a byte a coefficient follows the rotations, coefficient-major and
    colour innermost (`quantize_sh`). Without it the output is what it always was, degree 0.

    Positions are stored at `fractional_bits` (the header byte says how many), so they
    span `SPZ_FIXED_MAX * 2^-bits` m either side of the tile's frame: +/-2047.99976 m at
    the default 12. Nothing wraps: a position past that range, or one that is not finite,
    is refused (`SplatFormatError`), and a tile that must hold one is given an origin of
    its own (`tile_frame`). Until 2026-10 the top bits were masked away silently, so 3000 m
    read back as -1096 m.
    """
    count = positions.shape[0]
    degree = 0 if sh_rest is None else SH_DIMS.index(sh_rest.shape[1])
    if sh_rest is not None and sh_rest.shape != (count, SH_DIMS[degree], 3):
        raise ValueError(f"sh_rest must be (n, K, 3) for n = {count}, not {sh_rest.shape}")
    if not SPZ_MIN_FRACTIONAL_BITS <= fractional_bits <= 23:
        raise ValueError(
            f"fractional_bits must be {SPZ_MIN_FRACTIONAL_BITS} to 23 (a 24-bit field), "
            f"not {fractional_bits}"
        )
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, count, degree, fractional_bits, 0, 0)
    # Positions times an int, exactly as before the range check existed: for float32 input
    # that stays float32 arithmetic, so every tileset that fits packs to the same bytes.
    scaled = np.round(positions * (1 << fractional_bits))
    # The range and finiteness checks read only the extremes: a NaN propagates through
    # `max`/`min`, and an infinity is an extreme. No (n, 3) temporaries -- `np.abs` and
    # `np.isfinite` of the whole array put the packer's peak memory past its bound.
    high = float(scaled.max()) if scaled.size else 0.0
    low = float(scaled.min()) if scaled.size else 0.0
    if not (math.isfinite(high) and math.isfinite(low)):
        raise SplatFormatError("SPZ positions must be finite; a NaN or infinity cannot be packed")
    if high > SPZ_FIXED_MAX or low < SPZ_FIXED_MIN:
        reach = float(np.abs(positions).max())
        step = float(1 << fractional_bits)
        raise SplatFormatError(
            f"an SPZ position reaches {reach:.3f} m from the tile's frame, past the "
            f"{SPZ_FIXED_MIN / step:.5f} to +{SPZ_FIXED_MAX / step:.5f} m that 24-bit "
            f"fixed point holds at {fractional_bits} fractional bits; give the tile an "
            "origin (tile_frame) or fewer fractional bits"
        )
    fixed = scaled.astype(np.int32).reshape(-1)
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
        + (b"" if sh_rest is None else quantize_sh(sh_rest).tobytes())
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
#: version 2. Version 3 stores a fourth rotation byte (see `_smallest_three`). SH bytes,
#: when there are any, follow all of these: 3 x `SH_DIMS[degree]` a gaussian.
SPZ_BYTES_PER_GAUSSIAN = 19

#: The gzip-framed versions this reads. 2 stores a quaternion's first three components as
#: bytes; 3 stores its smallest three in 10 bits each, plus the index of the largest.
#: Version 1 (float16 positions) was never released, per nianticlabs/spz's own loader,
#: and version 4 is a different container (a plaintext `NGSP` header and ZSTD streams).
SPZ_READABLE_VERSIONS = (2, 3)


def unpack_spz(blob: bytes, sh: bool = False) -> dict[str, np.ndarray]:
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
    are read only when `sh` is asked for -- `canonical.ply` carries the DC term only --
    and come back as the PLY's `f_rest_*` (channel-major, `(b - 128) / 128` as spz's
    `unquantizeSH`), so `unpack -> PLY -> pack` carries them through.

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
    magic, version, count, sh_degree, fractional_bits, _flags, _reserved = struct.unpack_from(
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
    if sh and sh_degree > SH_MAX_DEGREE:
        raise SplatFormatError(f"SPZ SH degree {sh_degree} is not supported; this reads 0-3")
    sh_bytes = 3 * SH_DIMS[sh_degree] * count if sh else 0
    wanted = 16 + per_gaussian * count + sh_bytes
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
    if sh_bytes:
        dim = SH_DIMS[sh_degree]
        start = 16 + per_gaussian * count
        coefficients = np.frombuffer(raw, dtype=np.uint8, count=sh_bytes, offset=start)
        rest = (coefficients.reshape(count, dim, 3).astype(np.float32) - 128.0) / 128.0
        # Back to the PLY's channel-major f_rest_{c * K + j}.
        columns.update(
            {f"f_rest_{c * dim + j}": rest[:, j, c] for c in range(3) for j in range(dim)}
        )
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


def sh_attributes(degree: int) -> list[str]:
    """The glTF attribute names of SH bands 1..`degree`, as KHR_gaussian_splatting names
    them: `KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{n}`, n = 0..2l.

    They are how CesiumJS 1.145 learns a tile's degree -- not from the SPZ header, which it
    only reads once the tile is decoded: `GaussianSplat3DTileContent`
    `degreeAndCoefFromAttributes` *counts* the attributes whose name holds `SH_DEGREE_`
    (3, 8 or 15 give degree 1, 2 or 3; any other count is degree 0), and each one is filled
    from the decoded SPZ at `base[l - 1] + 3n` (`GltfVertexBufferLoader` `processSpz`). So
    there is no `SH_DEGREE_0_COEF_0` here -- the DC term is `COLOR_0` -- because a fourth,
    ninth or sixteenth such attribute would turn SH off.
    """
    return [
        f"KHR_gaussian_splatting:SH_DEGREE_{band}_COEF_{n}"
        for band in range(1, degree + 1)
        for n in range(2 * band + 1)
    ]


def build_glb(
    count: int, pmin: list[float], pmax: list[float], spz: bytes, sh_degree: int = 0
) -> bytes:
    """One point primitive whose attributes live in the SPZ block, as Cesium's tilers write.

    `sh_degree` declares the SPZ's SH bands as attributes (`sh_attributes`), each a float
    VEC3 accessor with no buffer view of its own, like every other attribute here. Degree
    0 writes exactly the JSON it always did.
    """
    padded = spz + b"\x00" * ((4 - len(spz) % 4) % 4)
    names = sh_attributes(sh_degree)
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
                            **{name: 4 + index for index, name in enumerate(names)},
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
            *({"componentType": 5126, "count": count, "type": "VEC3"} for _ in names),
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


#: Merged parents carry their merged SH bands -- the same weighted mean as the DC colour
#: (`merge_cells`), of the bands their members are drawn with -- rather than zeros.
#:
#: Whether a parent has SH *at all* is not a choice: CesiumJS 1.145 draws a tileset as one
#: primitive whose SH degree is the first selected tile's, and it concatenates the SH of
#: only the tiles that have any (GaussianSplatPrimitive.js `aggregateShData`), so a parent
#: of degree 0 drawn beside degree-3 leaves would shift every later tile's coefficients onto
#: the wrong splats, or switch SH off for the whole view. Every tile of a tileset has the
#: leaves' degree. The choice is what the parent's coefficients are, and zeros are nearly
#: free (a constant byte 128 gzips to nothing) where the merge's cost something -- measured
#: on nianticlabs/spz's two SH-3 samples, packed at the default budget:
#:
#: | sample (gaussians, parents)  | parents, zero SH | parents, merged SH | whole tileset |
#: | ---------------------------- | ---------------- | ------------------ | ------------- |
#: | hornedlizard (786k, 49k)     | 0.79 MB          | 1.12 MB            | 18.9 MB       |
#: | racoonfamily (932k, 64k)     | 1.00 MB          | 1.52 MB            | 25.2 MB       |
#:
#: 2% of the tileset, for parents that change colour with the view as their children do.
#: SH is linear in its coefficients, so a merged parent's colour from any direction is the
#: weighted mean of its children's from that direction; with zeros it is their mean over
#: all directions, and every refinement from parent to children would visibly shift colour
#: wherever the capture has view-dependent shine.
PARENTS_CARRY_SH = True

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


def record_dtype(sh_degree: int) -> np.dtype:
    """`RECORD`, plus the SH bands above the DC term when the tiles carry them: (K, 3) a
    gaussian, in SPZ's coefficient-major order (`ply_sh_columns`), already quantised to the
    bytes SPZ will hold (`quantize_sh`).

    Bytes, not the PLY's float32s, because they are all a leaf ever writes -- quantising is
    per coefficient, so a leaf's SPZ is the same either way -- and a parent is better merged
    from the SH its children are *drawn* with, as their sizes are (`gaussian_moments` clips
    them to what SPZ holds). And a quarter the size: degree 3 makes a record 109 bytes
    rather than 64, where float32s would make it 244, and the sorted working copy of an 8M
    trained splat 0.9 GB rather than 2 GB, on a worker volume that holds the PLY beside it.
    """
    if sh_degree == 0:
        return RECORD
    return np.dtype([*RECORD.descr, ("sh", "u1", (SH_DIMS[sh_degree], 3))])


def _bucket_dtype(record: np.dtype) -> np.dtype:
    return np.dtype([("pos", "<i8"), *[(name, record.fields[name][0]) for name in record.names]])


#: Records one bucket of the external sort holds -- of `RECORD`s; a record carrying SH is
#: bigger and a bucket holds proportionally fewer (`_bucket_records`), so what putting one
#: in order costs stays the same: 512k records are 36 MB read back and 32 MB of sorted block.
#: A scan below this is one bucket; 8M gaussians are 16 (26 with SH degree 3).
BUCKET_RECORDS = 1 << 19


def _bucket_records(record: np.dtype) -> int:
    return max(1, BUCKET_RECORDS * RECORD.itemsize // record.itemsize)


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
    octants. `low`/`high` bound what the tile and its subtree draw, in the site frame (the
    PLY's), whatever frame the tile is written in.

    `origin` is the tile's frame in the site frame, whole metres: zero for every tile whose
    content fits SPZ's range as it is (every tileset of a scan under ~4 km), else the
    centre of its content (`tile_frame`); the tileset states it as a translation-only
    `transform` relative to the parent's origin. `fractional_bits` is what its SPZ header
    declares: 12, or fewer for a tile whose own content spans more than 4 km. Both are set
    when the tile is written (`_write_tile`).
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
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fractional_bits: int = SPZ_FRACTIONAL_BITS
    #: The lowest confidence score (1..255) of any scored cell its gaussians are in -- a
    #: leaf's own, a parent's the least of its children's -- when a confidence grid was
    #: given and any of them is scored; None otherwise (`write_tiles`).
    confidence_min: int | None = None

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


# --------------------------------------------------------------- merging (splat_merge.py)


@dataclass
class Gaussians:
    """What a tile writes: arrays in `pack_spz`'s terms."""

    xyz: np.ndarray
    sh0: np.ndarray
    opacity_logit: np.ndarray
    log_scales: np.ndarray
    quat_xyzw: np.ndarray
    #: SH bands above the DC term, (n, K, 3) coefficient-major; None for degree 0.
    sh_rest: np.ndarray | None = None

    @property
    def reach(self) -> np.ndarray:
        """Three standard deviations along each gaussian's longest axis, clipped as SPZ is."""
        top = np.clip(self.log_scales.astype(np.float64), *SPZ_LOG_SCALE_RANGE).max(axis=1)
        return 3.0 * np.exp(top)


def gaussian_moments(records: np.ndarray, keys: np.ndarray) -> Moments:
    """Leaves as moments (`splat_merge.leaf_moments`), one row per record in order, their
    sizes clipped to what SPZ holds and their SH the bytes SPZ will hold, unquantised."""
    n = int(records.shape[0])
    if "sh" in (records.dtype.names or ()):
        sh_flat = unquantize_sh(records["sh"]).reshape(n, -1)
    else:
        sh_flat = np.zeros((n, 0), dtype=np.float32)
    return splat_merge.leaf_moments(
        keys.astype(np.int64),
        records["xyz"],
        records["opacity"],
        records["scale"],
        records["rot"],  # w first, as the PLY's rot_0..3
        records["f_dc"],
        sh_flat,
        log_scale_range=SPZ_LOG_SCALE_RANGE,
    )


def to_gaussians(moments: Moments) -> Gaussians:
    """Merged cells as gaussians (`splat_merge.to_splats`)."""
    merged = splat_merge.to_splats(moments, log_scale_range=SPZ_LOG_SCALE_RANGE)
    return Gaussians(
        xyz=merged.xyz,
        sh0=merged.sh0,
        opacity_logit=merged.opacity_logit,
        log_scales=merged.log_scales,
        quat_xyzw=merged.quat_xyzw,
        sh_rest=merged.sh_rest,
    )


def records_gaussians(records: np.ndarray) -> Gaussians:
    """Original gaussians, as the PLY had them."""
    return Gaussians(
        xyz=records["xyz"],
        sh0=records["f_dc"],
        opacity_logit=records["opacity"],
        log_scales=records["scale"],
        quat_xyzw=records["rot"][:, [1, 2, 3, 0]],
        sh_rest=unquantize_sh(records["sh"]) if "sh" in (records.dtype.names or ()) else None,
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

    def __init__(self, path: Path, count: int, dtype: np.dtype = RECORD) -> None:
        self.path = path
        self.count = count
        self.dtype = dtype

    def read(self, start: int, stop: int) -> np.ndarray:
        with self.path.open("rb") as handle:
            handle.seek(start * self.dtype.itemsize)
            return np.fromfile(handle, dtype=self.dtype, count=stop - start)

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


def _sort_to_disk(
    layout: PlyLayout, keep: np.ndarray, rank: np.ndarray, work: Path, sh_degree: int = 0
) -> SortedStore:
    """Every kept row as a record (`record_dtype`) at its Morton position, in a file under
    `work`, with its first `sh_degree` SH bands.

    `rank[k]` is the sorted position of the k-th kept row. Buckets are ranges of positions,
    so a bucket in order is a stretch of the store in order.
    """
    count = int(rank.size)
    record = record_dtype(sh_degree)
    bucket_dtype = _bucket_dtype(record)
    per_bucket = _bucket_records(record)
    sh_columns = ply_sh_columns(layout, sh_degree)
    buckets = max(1, -(-count // per_bucket))
    paths = [work / f"bucket-{index}.bin" for index in range(buckets)]
    handles = [path.open("wb") for path in paths]
    try:
        seen = 0
        # Windows narrower when SH is read too, so the float32 columns held at once stay
        # what the fourteen without it are: 59 columns of a window, not of 64 MiB of rows.
        window = max(1, CHUNK_BYTES * len(PLY_COLUMNS) // len(PLY_COLUMNS + sh_columns))
        for start, chunk in iter_ply_rows(layout, PLY_COLUMNS + sh_columns, window):
            good = keep[start : start + chunk["x"].size]
            rows = start + np.flatnonzero(good)
            positions = rank[seen : seen + rows.size]
            seen += rows.size
            out = np.empty(rows.size, dtype=bucket_dtype)
            out["pos"] = positions
            out["row"] = rows
            out["xyz"] = np.stack([chunk[n][good] for n in ("x", "y", "z")], axis=1)
            out["f_dc"] = np.stack([chunk[f"f_dc_{i}"][good] for i in range(3)], axis=1)
            out["opacity"] = chunk["opacity"][good]
            out["scale"] = np.stack([chunk[f"scale_{i}"][good] for i in range(3)], axis=1)
            out["rot"] = np.stack([chunk[f"rot_{i}"][good] for i in range(4)], axis=1)
            sh_buckets = _sh_buckets(SH_DIMS[sh_degree])
            for index, name in enumerate(sh_columns):
                j, c = divmod(index, 3)
                out["sh"][:, j, c] = _quantize_sh(chunk[name][good], sh_buckets[j])
            which = positions // per_bucket
            order = np.argsort(which, kind="stable")
            edges = np.searchsorted(which[order], np.arange(buckets + 1))
            for index in range(buckets):
                part = out[order[edges[index] : edges[index + 1]]]
                if part.size:
                    part.tofile(handles[index])
            # Freed before the next window is read, not when its names are rebound: a
            # window's arrays and the next one's never coexist.
            del out, part, order, which, edges, positions, rows, good, chunk
    finally:
        for handle in handles:
            handle.close()
    store = SortedStore(work / "sorted.bin", count, record)
    with store.path.open("wb") as sink:
        for index, path in enumerate(paths):
            part = np.fromfile(path, dtype=bucket_dtype)
            path.unlink()
            block = np.empty(part.size, dtype=record)
            slots = part["pos"] - index * per_bucket
            for name in record.names:
                block[name][slots] = part[name]
            block.tofile(sink)
            # The bucket and its block go before the next bucket is read: two of each
            # alive at once was the packer's peak.
            del part, block, slots
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


def fractional_bits_for(reach_m: float) -> int:
    """The most fractional bits (12 at most) at which 24-bit fixed point still holds a
    position `reach_m` from the tile's frame: 12 to 2047.99 m, 11 to 4095.99 m, and so on
    down to whole metres. Each step down halves the precision and doubles the range."""
    bits = SPZ_FRACTIONAL_BITS
    while round(reach_m * (1 << bits)) > SPZ_FIXED_MAX and bits > SPZ_MIN_FRACTIONAL_BITS:
        bits -= 1
    if round(reach_m * (1 << bits)) > SPZ_FIXED_MAX:
        raise SplatFormatError(
            f"a tile reaches {reach_m:.0f} m from its own centre, past what SPZ's 24-bit "
            f"positions hold even in whole metres (+/-{SPZ_FIXED_MAX} m)"
        )
    return bits


def tile_frame(xyz: np.ndarray, may_move: bool) -> tuple[np.ndarray, int]:
    """The frame a tile's content is written in: its origin in the site frame (whole metres)
    and the fractional bits its SPZ declares.

    A tile whose centres all lie within `TILE_FIT_M` of the site origin stays there, at 12
    bits -- every scan under ~4 km across, so every tileset packed before origins existed
    packs to the same bytes. Otherwise a tile that `may_move` is moved to the centre of its
    centres' box, snapped to whole metres (so `local + origin` is exact on SPZ's grid), and
    keeps 12 bits unless its own half-extent passes the range too. The root may not move:
    its frame is the tileset's, the one `root.transform` places on the globe and every
    root-level extra (collision grid, view cones, instance bounds) is in; it takes fewer
    bits instead (docs/DECISIONS/0011-splat-tile-origins.md, R1).

    The rule reads the tile's own content in the site frame, not its parent's frame,
    because children are written before their parents (`build_tiles`): a parent's origin
    does not exist yet when its child is quantised. The tileset still states each origin
    relative to the parent's (`convert`).
    """
    if xyz.size == 0:
        return np.zeros(3), SPZ_FRACTIONAL_BITS
    centres = xyz.astype(np.float64)
    return _frame(centres.min(axis=0), centres.max(axis=0), may_move)


def _frame(low: np.ndarray, high: np.ndarray, may_move: bool) -> tuple[np.ndarray, int]:
    """`tile_frame` from the centres' box alone: the farthest a centre reaches from any
    point is from a corner, and rounding is monotone, so `|c - o|.max()` over the centres is
    `max(|low - o|, |high - o|)` exactly. `_bounds` reads it to know the tile's quantum."""
    zero = np.zeros(3)
    reach = float(max(np.abs(low).max(), np.abs(high).max()))
    if reach <= TILE_FIT_M:
        return zero, SPZ_FRACTIONAL_BITS
    if not may_move:
        return zero, fractional_bits_for(reach)
    # `+ 0.0` turns a -0.0 from rounding into 0.0, so no transform carries a negative zero.
    origin = np.round((low + high) / 2) + 0.0
    return origin, fractional_bits_for(
        float(max(np.abs(low - origin).max(), np.abs(high - origin).max()))
    )


def _write_tile(out_dir: Path, tile: Tile, gaussians: Gaussians) -> None:
    xyz = gaussians.xyz
    g = gaussians
    tile.origin, tile.fractional_bits = tile_frame(xyz, may_move=bool(tile.path))
    # A tile in the site frame packs its positions exactly as it always did (no float64
    # round trip), so every tileset that fits is the bytes it was.
    local = xyz if not tile.origin.any() else xyz.astype(np.float64) - tile.origin
    spz = pack_spz(
        local,
        g.sh0,
        g.opacity_logit,
        g.log_scales,
        g.quat_xyzw,
        g.sh_rest,
        fractional_bits=tile.fractional_bits,
    )
    degree = 0 if g.sh_rest is None else SH_DIMS.index(g.sh_rest.shape[1])
    # The accessor's bounds are of what the SPZ stores: the tile-local positions.
    bounds = local if not tile.origin.any() else local.astype(np.float32)
    (out_dir / tile.uri).write_bytes(
        build_glb(
            int(xyz.shape[0]),
            bounds.min(axis=0).tolist(),
            bounds.max(axis=0).tolist(),
            spz,
            degree,
        )
    )
    tile.count = int(xyz.shape[0])


def _bounds(xyz: np.ndarray, reach: np.ndarray, may_move: bool) -> tuple[np.ndarray, np.ndarray]:
    """Centres' box padded by the 99th percentile of reach, not the maximum: one 10 m
    background gaussian must not make every box it falls in 20 m wider than the data.

    The pad is never less than half the tile's SPZ quantum, which is as far as packing
    moves a centre, so the box holds every centre as decoded, not only as given. At 12
    fractional bits that is 2^-13 = 1.2e-4 m, under the smallest reach SPZ stores (3 e^-10
    = 1.4e-4 m), so a 12-bit tile's box is the one it always had; at 11 bits and fewer
    (a tile past 2048 m) it can be the larger. `may_move` is the tile's, as `tile_frame`
    reads it, so the quantum is the one `_write_tile` packs at."""
    centres = xyz.astype(np.float64)
    low, high = centres.min(axis=0), centres.max(axis=0)
    _, bits = _frame(low, high, may_move)
    pad = max(float(np.percentile(reach, 99)), 2.0 ** -(bits + 1))
    return low - pad, high + pad


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
    writes it as a GLB; `hierarchy` hands it to its caller, so both see the one tree this
    function builds.

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
            tile.low, tile.high = _bounds(gaussians.xyz, gaussians.reach, bool(tile.path))
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
        if gaussians.sh_rest is not None and not PARENTS_CARRY_SH:
            gaussians.sh_rest = np.zeros_like(gaussians.sh_rest)
        tile.count = int(gaussians.xyz.shape[0])
        cell = edge / (1 << tile.level)
        width = 2.0 * float(np.median(np.exp(gaussians.log_scales.astype(np.float64).max(axis=1))))
        tile.geometric_error = max(cell, width)
        low, high = _bounds(gaussians.xyz, gaussians.reach, bool(tile.path))
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
    confidence: cell_grid.CellGrid | None = None,
) -> int:
    """Write every tile's GLB (`build_tiles`), a parent's from `parents` where it has one.

    An optimised parent replaces what is *drawn* and nothing else: the tile's box, its
    geometric error and the moments handed up to its own parent are the merge's, so the
    tree -- and so the cut a viewer takes through it -- is the one the parents were
    optimised in (`ParentOverrides`). Returns how many parent gaussians were replaced.

    With `confidence` (the epoch's grid, in this tileset's frame), each tile's
    `confidence_min` is filled in: a leaf's is the lowest score of the scored cells its
    gaussians fall in, at the grid's finest level; a parent's the lowest of its
    children's, which is the same thing over its subtree -- a parent draws its subtree
    merged, so it is never more confident than the weakest part of what it stands for.
    Cells with no evidence (score 0) say nothing and are skipped.
    """
    replaced = 0
    score = None if confidence is None else confidence.channel("score")

    def emit(tile: Tile, gaussians: Gaussians, keys: np.ndarray | None) -> None:
        nonlocal replaced
        if confidence is not None and score is not None:
            if keys is None:
                found = cell_grid.lookup(confidence, gaussians.xyz)[:, score]
                scored = found[found > 0]
                tile.confidence_min = int(scored.min()) if scored.size else None
            else:
                below = [c.confidence_min for c in tile.children if c.confidence_min is not None]
                tile.confidence_min = min(below) if below else None
        if keys is not None and parents is not None:
            better = parents.take(tile.uri, keys)
            if better is not None:
                # Optimised parents carry DC colour only (`ParentOverrides`); the higher
                # bands stay the merge's, cell for cell (`take` checked the cells).
                gaussians = replace(better, sh_rest=gaussians.sh_rest)
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
    sh_degree: int = 0
    #: The PLY rows kept (`_filter_rows`): what the leaves hold, and the collision grid is of.
    keep: np.ndarray | None = None


@contextmanager
def prepare(
    ply: Path,
    opacity_min: float,
    tile_gaussians: int | None,
    work_dir: Path,
    sh_degree: int = 0,
) -> Iterator[Prepared]:
    """Filter, sort and plan `ply` exactly as `convert` does, the sorted copy in `work_dir`
    carrying the PLY's first `sh_degree` SH bands (which it must have).

    Deterministic: the same PLY and parameters give the same tree, cell for cell -- which
    is what lets a parent be optimised on one machine and written on another
    (`ParentOverrides`). SH plays no part in the tree: it is carried, never planned on.
    """
    layout = ply_layout(ply)
    if not 0 <= sh_degree <= ply_sh_degree(layout):
        raise SplatFormatError(
            f"{ply.name} carries SH degree {ply_sh_degree(layout)}; {sh_degree} was asked for"
        )
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
        store = _sort_to_disk(layout, keep, rank, Path(scratch), sh_degree)
        del rank
        yield Prepared(layout, count, pmin, pmax, codes, root, store, edge, sh_degree, keep)


def hierarchy(
    ply: Path, opacity_min: float, tile_gaussians: int | None, work_dir: Path, emit: Emit
) -> Tile:
    """The tree `convert` would write for `ply`, each tile's content handed to `emit`
    instead of written (`build_tiles`). Returns the root, boxes and errors filled in."""
    with prepare(ply, opacity_min, tile_gaussians, work_dir) as tree:
        build_tiles(tree.root, tree.codes, tree.store, tree.edge, emit)
    return tree.root


# --------- optimised parents (a file format; nothing in this repo writes one since recipe 11)

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
    """Parent gaussians optimised elsewhere against the photos, written in place of the
    merge's, tile by tile.

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


# ------------------------------------------------------------------------- collision grid

#: What the web clients collide with and cast rays against, precomputed here so they load
#: it instead of building one on the main thread (a Chrome trace of the globe put that work
#: at ~20% of main-thread time). A splat writes no depth, so neither CesiumJS nor Spark can
#: pick it; its own centres are the geometry (apps/web src/lib/occupancy.ts says why).
#:
#: **Format `hexapod.collision` v1.** `collision.bin` beside `tileset.json`, gzip (mtime 0).
#: Decompressed: one 76-byte record per brick of 8 x 8 x 8 cells that holds any solid cell,
#: sorted by (bz, by, bx): int32 bx, by, bz, little-endian, then a 64-byte bitmask. A point
#: p is in cell `i = floor((p - origin) / cell)` per axis, of brick `i >> 3`, at local
#: `(lx, ly, lz) = i & 7`, bit `n = lx + 8 ly + 64 lz`, set when `mask[n >> 3] >> (n & 7) & 1`.
#: The root tile's `extras.collision` carries `cell`, `origin` and the counts (`collision_extras`),
#: all in the tileset's local east/north/up metres -- the frame the SPZ positions are in.
COLLISION_FORMAT = "hexapod.collision"
COLLISION_VERSION = 1
COLLISION_URI = "collision.bin"
COLLISION_BRICK = 8
COLLISION_RECORD = np.dtype([("brick", "<i4", (3,)), ("mask", "u1", (COLLISION_BRICK**3 // 8,))])

#: The runtime grid's two constants (apps/web src/lib/occupancy.ts `SOLID`, `MIN_OPACITY`):
#: a cell is solid when the opacities of the splats centred in it sum to `COLLISION_SOLID`,
#: counting only splats at least `COLLISION_MIN_OPACITY` opaque. One opaque splat is
#: enough, a faint floater is not, and a surface of many translucent ones adds up.
COLLISION_SOLID = 0.6
COLLISION_MIN_OPACITY = 0.2
COLLISION_RULE = (
    f"solid where the opacities of the splats centred in the cell, counting splats at least "
    f"{COLLISION_MIN_OPACITY:g} opaque, sum to at least {COLLISION_SOLID:g}"
)

#: Opacities are summed in fixed point, 2^-32 a step, so a cell's sum is exact whatever
#: order it is added in: the same PLY read in other windows (`CHUNK_BYTES`) must give the
#: same bytes, and a float sum a hair either side of `COLLISION_SOLID` would not.
_COLLISION_FIXED = float(1 << 32)
_COLLISION_SOLID_FIXED = math.ceil(COLLISION_SOLID * _COLLISION_FIXED)

#: The cell is sized from the data as the runtime sizes a tile's (occupancy.ts `cellFor`):
#: the smallest power of two metres, from 2^-8 (4 mm) up, at which the splats that count
#: average `COLLISION_SPLATS_PER_CELL` to an occupied cell -- dense enough that neighbouring
#: cells on a surface are all occupied, so a ray cannot pass between specks -- and 2^5 m
#: (32 m) when none is.
COLLISION_SPLATS_PER_CELL = 3
COLLISION_CELL_EXPONENTS = (-8, 5)

#: ...and never finer than this many cells along the site's longest axis (rounded up to a
#: power of two). The solid cells of a surface grow as its area over the cell squared, so a
#: whole site at the millimetres its densest patch could carry would be hundreds of millions
#: of cells; 4096 a side keeps a site to a few MB wherever it is dense. Fort Clatsop (22.6M
#: gaussians over ~150 m) packs at 2^-5 m: 3.69M solid cells in 501k bricks, 4.9 MB of
#: gzip. A 16 m tree is floored at 4 mm and its density decides (the synthetic tree:
#: 12.5 cm). The camera stays 1.5 cells off a surface (occupancy.ts `CLEARANCE_CELLS`), so
#: a coarser cell is a slightly wider berth, not a hole.
COLLISION_MAX_CELLS = 4096

#: The site's extent for `COLLISION_MAX_CELLS`: the largest axis of the kept splats' box
#: between these percentiles, so a few far splats the floater radius let through do not
#: coarsen the whole grid. Measured on a strided sample of at most ~2x `COLLISION_SAMPLE`.
COLLISION_BOUNDS_PERCENTILES = (0.5, 99.5)
COLLISION_SAMPLE = 1 << 18

#: Cell indices are packed 21 bits an axis while the grid is built (2^21 cells of 4 mm is
#: 8 km); bricks, 18 bits an axis, then a cell's 9 bits, sort in the file's order.
_AXIS_BITS = 21
_AXIS_MASK = (1 << _AXIS_BITS) - 1
_BRICK_BITS = _AXIS_BITS - 3


@dataclass(frozen=True)
class CollisionGrid:
    """Solid cells of a scan: `cells` (n, 3) int64 indices, in the file's order."""

    cell: float
    origin: tuple[float, float, float]
    cells: np.ndarray
    #: Splats that counted: kept, and at least `COLLISION_MIN_OPACITY` opaque.
    splats: int = 0

    @property
    def solid(self) -> set[tuple[int, int, int]]:
        return {(int(x), int(y), int(z)) for x, y, z in self.cells}

    def cell_of(self, points: np.ndarray) -> np.ndarray:
        """The cell index of each point (n, 3), by the format's own rule."""
        origin = np.asarray(self.origin, dtype=np.float64)
        return np.floor((np.asarray(points, np.float64) - origin) / self.cell).astype(np.int64)


def _power_of_two_at_least(value: float) -> float:
    """The smallest power of two >= `value` (> 0), exactly."""
    mantissa, exponent = math.frexp(value)
    return math.ldexp(1.0, exponent - 1 if mantissa == 0.5 else exponent)


def _pack_cells(index: np.ndarray) -> np.ndarray:
    return (index[:, 2] << (2 * _AXIS_BITS)) | (index[:, 1] << _AXIS_BITS) | index[:, 0]


def _unpack_cells(keys: np.ndarray) -> np.ndarray:
    return np.stack(
        [keys & _AXIS_MASK, (keys >> _AXIS_BITS) & _AXIS_MASK, keys >> (2 * _AXIS_BITS)], axis=1
    )


def _coarsen_cells(keys: np.ndarray, shift: int, offset: np.ndarray | None = None) -> np.ndarray:
    """Packed cells `shift` sizes coarser (a cell is 2^shift of them a side), less `offset`."""
    out = np.zeros_like(keys)
    for axis in range(3):
        index = ((keys >> (axis * _AXIS_BITS)) & _AXIS_MASK) >> shift
        if offset is not None:
            index -= int(offset[axis])
        out |= index << (axis * _AXIS_BITS)
    return out


def _reduce_cells(keys: np.ndarray, sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per distinct key, ascending, its fixed-point opacities added.

    Hand it arrays nothing else holds (fresh concatenations, say): each is dropped as soon
    as its sorted copy exists, which is what keeps a merge of millions of cells small.
    """
    if keys.size == 0:
        return keys, sums
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    sums = sums[order]
    del order
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[starts], np.add.reduceat(sums, starts)


def collision_grid(layout: PlyLayout, keep: np.ndarray) -> CollisionGrid:
    """The solid cells of the rows `keep` selects -- the gaussians the leaves hold.

    Two passes over the PLY in windows (`iter_ply_rows`), holding one row per occupied
    cell and never the scan: the first finds the site's extent (`COLLISION_MAX_CELLS`) and
    the lowest counted centre; the second adds every counted splat into its cell at the
    finest size the extent allows. The cell size (`COLLISION_SPLATS_PER_CELL`) is then
    counted exactly at each size from those cells -- each is a union of finer ones, so no
    further pass is needed -- rather than estimated from a sample: a sample of 1 in k
    splats occupies up to k times too few cells at sizes the data does not fill, which is
    why a stride-scaled estimate always reads the finest size once k reaches the target.
    The opacity sums coarsen the same way, exactly (`_COLLISION_FIXED`). Fort Clatsop's
    22.6M gaussians (a 1.27 GB PLY) are 5.6M occupied cells at the finest size: 0.54 GB
    at the peak and 7 s, under the 1.1 GB `_filter_rows` already takes of the same scan
    (so `convert`'s peak is unchanged), and ~19 s more to write the file (`encode_collision`).
    """
    stride = max(1, layout.count // COLLISION_SAMPLE)
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    sample: list[np.ndarray] = []
    counted = 0
    columns = ("x", "y", "z", "opacity")
    for start, chunk in iter_ply_rows(layout, columns):
        size = chunk["x"].size
        kept = keep[start : start + size]
        xyz = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1).astype(np.float64)
        sample.append(xyz[kept & (np.arange(start, start + size) % stride == 0)])
        solid = kept & (sigmoid(chunk["opacity"].astype(np.float64)) >= COLLISION_MIN_OPACITY)
        if solid.any():
            low = np.minimum(low, xyz[solid].min(axis=0))
            high = np.maximum(high, xyz[solid].max(axis=0))
            counted += int(solid.sum())
    top = math.ldexp(1.0, COLLISION_CELL_EXPONENTS[1])
    kept_sample = np.concatenate(sample) if sample else np.zeros((0, 3))
    extent = 0.0
    if kept_sample.shape[0]:
        lo, hi = np.percentile(kept_sample, COLLISION_BOUNDS_PERCENTILES, axis=0)
        extent = float((hi - lo).max())
    del sample, kept_sample
    floor = _power_of_two_at_least(extent / COLLISION_MAX_CELLS) if extent > 0 else 0.0
    fine = max(math.ldexp(1.0, COLLISION_CELL_EXPONENTS[0]), floor)
    if counted == 0:
        return CollisionGrid(max(top, fine), (0.0, 0.0, 0.0), np.zeros((0, 3), np.int64), 0)
    # Every candidate cell is a power of two no larger than max(32 m, fine), so an origin on
    # that many bricks is on a brick boundary of every one of them.
    align = COLLISION_BRICK * max(top, fine)
    base = np.floor(low / align) * align
    while float((high - base).max()) / fine >= _AXIS_MASK:
        fine *= 2  # Kilometres of 4 mm cells: only a degenerate scan gets here.
        align = COLLISION_BRICK * max(top, fine)
        base = np.floor(low / align) * align

    keys = np.zeros(0, np.int64)
    sums = np.zeros(0, np.int64)
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    waiting = 0
    for start, chunk in iter_ply_rows(layout, columns):
        kept = keep[start : start + chunk["x"].size]
        alpha = sigmoid(chunk["opacity"].astype(np.float64))
        solid = kept & (alpha >= COLLISION_MIN_OPACITY)
        xyz = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1)[solid].astype(np.float64)
        index = np.floor((xyz - base) / fine).astype(np.int64)
        pending.append(
            _reduce_cells(
                _pack_cells(index), np.round(alpha[solid] * _COLLISION_FIXED).astype(np.int64)
            )
        )
        waiting += pending[-1][0].size
        # Folded in whenever what waits reaches half what is held: each merge sorts at
        # most 1.5x the cells, and the cells are held about twice over at the most.
        if waiting > max(keys.size // 2, 1 << 21):
            parts = [(keys, sums), *pending]
            keys = sums = np.zeros(0, np.int64)
            pending, waiting = [], 0
            keys, sums = _reduce_cells(
                np.concatenate([part[0] for part in parts]),
                np.concatenate([part[1] for part in parts]),
            )
            del parts
    if pending:
        parts = [(keys, sums), *pending]
        del pending
        keys, sums = _reduce_cells(
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
        )
        del parts

    # Sizes finer than `fine` are not counted: the answer is at least `fine` anyway, and
    # splats per occupied cell only grow with the cell, so the first size that reaches the
    # target is the one the runtime's search from 2^-8 would find, floored.
    cell = max(top, fine)
    occupied = keys
    exponent = round(math.log2(fine))
    for level in range(exponent, COLLISION_CELL_EXPONENTS[1]):
        if counted / occupied.size >= COLLISION_SPLATS_PER_CELL:
            cell = math.ldexp(1.0, level)
            break
        occupied = np.unique(_coarsen_cells(occupied, 1))
    del occupied
    shift = round(math.log2(cell / fine))
    origin = np.floor(low / (COLLISION_BRICK * cell)) * (COLLISION_BRICK * cell)
    offset = np.round((origin - base) / cell).astype(np.int64)
    if shift or offset.any():
        keys, sums = _reduce_cells(_coarsen_cells(keys, shift, offset), sums)
    solid_cells = _unpack_cells(keys[sums >= _COLLISION_SOLID_FIXED])
    del keys, sums
    corner = (float(origin[0]), float(origin[1]), float(origin[2]))
    return CollisionGrid(cell, corner, _brick_order(solid_cells), counted)


def _brick_keys(cells: np.ndarray) -> np.ndarray:
    """Each cell's place in the file: brick (bz, by, bx), then its bit."""
    brick = cells >> 3
    local = cells & 7
    bit = local[:, 0] + 8 * local[:, 1] + 64 * local[:, 2]
    packed = (brick[:, 2] << (2 * _BRICK_BITS)) | (brick[:, 1] << _BRICK_BITS) | brick[:, 0]
    return (packed << 9) | bit


def _brick_order(cells: np.ndarray) -> np.ndarray:
    return cells[np.argsort(_brick_keys(cells), kind="stable")]


def encode_collision(cells: np.ndarray) -> tuple[bytes, int]:
    """Solid cell indices (n, 3), all >= 0, as `collision.bin`, and how many bricks it holds."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    if cells.size and (cells.min() < 0 or cells.max() > _AXIS_MASK):
        raise ValueError(f"cell indices must be 0..{_AXIS_MASK}; origin is the grid's min corner")
    keys = np.unique(_brick_keys(cells))
    if keys.size == 0:
        return gzip.compress(b"", compresslevel=9, mtime=0), 0
    brick_keys = keys >> 9
    bit = keys & 511
    starts = np.flatnonzero(np.r_[True, brick_keys[1:] != brick_keys[:-1]])
    records = np.zeros(starts.size, dtype=COLLISION_RECORD)
    packed = brick_keys[starts]
    mask = (1 << _BRICK_BITS) - 1
    records["brick"] = np.stack(
        [packed & mask, (packed >> _BRICK_BITS) & mask, packed >> (2 * _BRICK_BITS)], axis=1
    ).astype(np.int32)
    # Each byte of a mask gathers distinct bits, so their sum is their OR.
    byte = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, keys.size])) * 64 + (bit >> 3)
    firsts = np.flatnonzero(np.r_[True, byte[1:] != byte[:-1]])
    flat = records["mask"].reshape(-1)
    flat[byte[firsts]] = np.add.reduceat((1 << (bit & 7)).astype(np.uint8), firsts)
    records["mask"] = flat.reshape(-1, 64)
    # Level 9: every viewer downloads this and the packer writes it once. On Fort Clatsop's
    # 38 MB of records, 4.86 MB in 15 s against 5.14 MB in 1.4 s at level 6.
    return gzip.compress(records.tobytes(), compresslevel=9, mtime=0), int(starts.size)


def decode_collision(blob: bytes) -> np.ndarray:
    """`collision.bin` back to solid cell indices (n, 3), in the file's order."""
    raw = gzip.decompress(blob)
    if len(raw) % COLLISION_RECORD.itemsize:
        raise SplatFormatError(
            f"collision data is {len(raw)} bytes, not a whole number of "
            f"{COLLISION_RECORD.itemsize}-byte bricks"
        )
    records = np.frombuffer(raw, dtype=COLLISION_RECORD)
    bits = np.unpackbits(records["mask"], axis=1, bitorder="little")
    brick, n = np.nonzero(bits)
    local = np.stack([n & 7, (n >> 3) & 7, n >> 6], axis=1)
    return records["brick"][brick].astype(np.int64) * COLLISION_BRICK + local


def collision_extras(grid: CollisionGrid, bricks: int) -> dict[str, object]:
    """The root tile's `extras.collision`: how to read `collision.bin`."""
    return {
        "format": COLLISION_FORMAT,
        "version": COLLISION_VERSION,
        "uri": COLLISION_URI,
        "cell": grid.cell,
        "origin": list(grid.origin),
        "brick": COLLISION_BRICK,
        "bricks": bricks,
        "solidCells": int(grid.cells.shape[0]),
        "rule": COLLISION_RULE,
    }


def write_collision(grid: CollisionGrid, out_dir: Path) -> dict[str, object]:
    """`collision.bin` in `out_dir`; returns the `extras.collision` that describes it."""
    blob, bricks = encode_collision(grid.cells)
    (out_dir / COLLISION_URI).write_bytes(blob)
    return collision_extras(grid, bricks)


def read_collision(path: Path) -> CollisionGrid:
    """A `collision.bin`, with the cell and origin its `tileset.json` (beside it) declares.

    `path` may be the `collision.bin`, the `tileset.json` or the directory holding both.
    The counts the tileset states are checked against the file.
    """
    folder = path if path.is_dir() else path.parent
    tileset = json.loads((folder / "tileset.json").read_text(encoding="utf-8"))
    extras = tileset["root"].get("extras", {}).get("collision")
    if not extras or extras.get("format") != COLLISION_FORMAT:
        raise SplatFormatError(f"{folder / 'tileset.json'} declares no {COLLISION_FORMAT}")
    if extras.get("version") != COLLISION_VERSION or extras.get("brick") != COLLISION_BRICK:
        raise SplatFormatError(
            f"{COLLISION_FORMAT} version {extras.get('version')} with bricks of "
            f"{extras.get('brick')}; this reads version {COLLISION_VERSION}, {COLLISION_BRICK}"
        )
    blob = (folder / extras["uri"]).read_bytes()
    cells = decode_collision(blob)
    bricks = len(gzip.decompress(blob)) // COLLISION_RECORD.itemsize
    if bricks != extras["bricks"] or cells.shape[0] != extras["solidCells"]:
        raise SplatFormatError(
            f"{extras['uri']} holds {bricks} bricks and {cells.shape[0]} solid cells; "
            f"the tileset says {extras['bricks']} and {extras['solidCells']}"
        )
    x, y, z = (float(value) for value in extras["origin"])
    return CollisionGrid(float(extras["cell"]), (x, y, z), cells)


def build_collision(
    ply: Path, out_dir: Path, opacity_min: float = 0.02, tileset: Path | None = None
) -> dict[str, object]:
    """`collision.bin` for a scan already packed, and its `extras.collision`.

    The rows are the ones `convert` keeps (`_filter_rows` with the same `opacity_min`), so
    this is the file `convert` would have written beside the tiles -- which is how a
    tileset published before the grid existed gets one. With `tileset`, that
    `tileset.json` gains the `extras.collision` too, after checking its leaves hold exactly
    as many gaussians as this PLY keeps: a grid built from another PLY, or with another
    `opacity_min`, would be solid where nothing is drawn.
    """
    layout = ply_layout(ply)
    keep, _ = _filter_rows(layout, opacity_min)
    document = None
    if tileset is not None:
        document = json.loads(tileset.read_text(encoding="utf-8"))
        leaves, stack = 0, [document["root"]]
        while stack:
            tile = stack.pop()
            if tile.get("children"):
                stack.extend(tile["children"])
            else:
                leaves += int(tile.get("extras", {}).get("gaussians", -1))
        if leaves != int(keep.sum()):
            raise SplatFormatError(
                f"{tileset} holds {leaves} gaussians in its leaves and {ply.name} keeps "
                f"{int(keep.sum())} at opacity_min {opacity_min:g}: not the PLY it was packed from"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = write_collision(collision_grid(layout, keep), out_dir)
    if tileset is not None and document is not None:
        document["root"].setdefault("extras", {})["collision"] = extras
        tileset.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return extras


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
    sh_degree: int | None = None,
    confidence: Path | None = None,
) -> dict[str, float | int]:
    """Every gaussian that passes the filters, as a level-of-detail tileset.

    There is no top-N cut. Nothing that passes `opacity_min` and the floater radius is
    dropped, and how much of it is *drawn* is the viewer's budget (CesiumJS:
    `maximumScreenSpaceError`, and a splat-count cap in SiteManager.ts; the Spark viewer:
    Spark's own LoD at the phone's Detail count) -- decided on the device that pays for it.

    **Refinement is REPLACE, with merged parents.** Each leaf holds original gaussians and
    the leaves hold every kept gaussian exactly once; each parent holds one gaussian per
    occupied cell of its grid, merged by moment matching (`splat_merge`: the law of total
    covariance over opacity x surface-area weights). What this replaced was ADD with *thinned* parents --
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
    coordinate frame"), so every GLB has the same node matrix and no tile below the root
    carries anything but a translation -- `getRotation` of a computed transform ignores its
    translation, so every tile shares the root's SH frame; and site tilesets keep
    `skipLevelOfDetail` off for splats (apps/web providers/tiles.ts), which is the
    traversal quoted above.

    **Per-tile origins.** SPZ stores positions as 24-bit fixed point: +/-2047.99976 m from
    the tile's frame at 12 fractional bits. A tile whose content reaches past that is
    written in a frame of its own (`tile_frame`): its centre, in whole metres, stated as a
    translation-only `transform` relative to its parent's frame. A scan under ~4 km across
    has no such tile, and its tileset is the bytes it always was. The root never moves --
    its frame is the tileset's, which the root transform places on the globe and every
    root extra (collision grid, view cones, instance bounds) is expressed in -- so a root
    whose own content passes the range is written at fewer fractional bits instead, as is
    any tile whose own half-extent does (docs/DECISIONS/0011-splat-tile-origins.md).
    Readers compose the translations (`tile_offsets`) and checksum site-frame positions
    (`site_positions`).

    **Out of core.** The PLY is read in windows (`iter_ply_rows`), twice: once for the
    filters and the Morton codes, once to spill every kept gaussian, in Morton order, to a
    working file (`SortedStore`, a distribution sort through buckets on disk). Tiles are
    then written one at a time from that file. What is held for the whole scan is 12 bytes
    a gaussian of centres during the first pass, then its codes and sorted positions (16):
    8M gaussians (a 2 GB trained PLY) package at a 380 MB peak, against the 1.5 GB the
    plan allows (tests/test_splat_tiles_memory.py). The
    working files go in `work_dir` (by default a temporary directory beside `out_dir`, on
    the same disk rather than in a RAM-backed /tmp) and are removed afterwards.

    **Spherical harmonics.** Every tile carries the SH bands the PLY has (`f_rest_*`,
    degree 1-3), or the first `sh_degree` of them: SPZ bytes quantised exactly as
    nianticlabs/spz does (`quantize_sh`), declared to glTF as `SH_DEGREE_l_COEF_n`
    attributes (`sh_attributes`), which is how CesiumJS finds them. Leaves carry the
    originals, parents the merge's (`PARENTS_CARRY_SH`), and every tile the same degree,
    which CesiumJS requires. They are evaluated in the PLY's own frame -- the frame the SPZ
    positions are in, which Cesium maps the view direction back into
    (GaussianSplatPrimitive.js `_shInverseRotation`) -- so a PLY that was rotated after
    training must have had its SH rotated with it. Measured on nianticlabs/spz's SH-3
    samples (786k and 932k gaussians), degree 3 makes the tileset 1.6-1.7x the bytes of
    degree 0 (degree 1: 1.17-1.20x, degree 2: 1.36-1.44x), not the 3.4x of the raw bytes:
    quantised SH is mostly near zero and compresses well. Packing takes ~3x as long
    (786k gaussians: 3.0 s at degree 0, 8.4 s at 3), most of it gzip. A PLY without `f_rest_*` packs
    to exactly the bytes it always did -- which is every pipeline `canonical.ply` by
    default: the pipeline ships `ship_sh_degree` bands (0 unless a run asks), and when it
    does, its normalise/place stages turn them with the splat (tools/pipeline
    harmonics.py), so a canonical.ply's SH is already in the frame its positions are.

    **Optimised parents.** `parents`, when given, are parent gaussians optimised elsewhere
    against the capture's photos (no stage in this repository writes them since recipe 11)
    for exactly this PLY and these parameters; each parent tile draws them in
    place of its merge, and nothing else changes (`write_tiles`). A set built for another
    PLY or other parameters is refused before anything is written (`ParentOverrideError`).

    **Collision grid.** Beside the tiles, `collision.bin`: every leaf gaussian's centre
    added into a voxel grid, and the cells that come out solid by the web runtime's own
    rule, in the format `COLLISION_FORMAT` describes and declared on the root tile's
    `extras.collision`. The web clients load it instead of building one per tile on the
    main thread. Two more windowed passes over the PLY once the tiles are written, holding
    a row per occupied cell (`collision_grid`); `build_collision` writes the same file for
    a tileset packed before it existed.

    **View cones.** Beside them, `viewcones.bin`: a coarse grid of the directions each part
    of the scan was seen from, estimated from where its detail is finest (`view_cones`),
    declared on the root tile's `extras.viewCones`. The globe fades a splat seen from
    outside its cone, so a scan walked inside a site does not show the overview the backs
    of what it only saw from within. Two more windowed passes; `view_cones.py` writes the
    same file for a tileset packed before it existed (from the PLY, or from its leaves).

    **Confidence.** `confidence`, when given, is the epoch's confidence grid descriptor
    (tools/pipeline confidence.py's `confidence.json`, a `cell_grid` in this tileset's
    frame, with the file it names beside it). Its file is copied in beside the tiles, the
    descriptor is declared on the root tile's `extras.confidence` -- after the collision
    grid, so a backfill appending it last writes these bytes -- and every tile states its
    least confident scored cell as `extras.confidenceMin`, after `gaussians` (`write_tiles`):
    a viewer can find the weak tiles from `tileset.json` alone. Opt-in: without it every
    byte written is what it was before the grid existed.
    """
    if sh_degree is not None and not 0 <= sh_degree <= SH_MAX_DEGREE:
        raise ValueError(f"sh_degree caps the SH bands at 0 to {SH_MAX_DEGREE}, not {sh_degree}")
    if parents is not None:
        why = parents.mismatch(ply, opacity_min, tile_gaussians)
        if why:
            raise ParentOverrideError(why)
    scores, scores_blob, scores_extras = (
        (None, b"", None) if confidence is None else _confidence(confidence)
    )
    carried = ply_sh_degree(ply_layout(ply))
    degree = carried if sh_degree is None else min(sh_degree, carried)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir or out_dir.parent
    with prepare(ply, opacity_min, tile_gaussians, work, degree) as tree:
        replaced = write_tiles(
            tree.root, tree.codes, tree.store, out_dir, tree.edge, parents, confidence=scores
        )
    root, layout, count, pmin, pmax = tree.root, tree.layout, tree.count, tree.pmin, tree.pmax
    keep = tree.keep
    assert keep is not None
    del tree  # the Morton codes, 8 bytes a gaussian, are not needed past the tiles
    grid = collision_grid(layout, keep)
    collision = write_collision(grid, out_dir)
    cones = view_cones.cone_grid(layout, keep)
    del keep
    seen_from = view_cones.write_view_cones(cones, out_dir)

    def node(tile: Tile, parent_origin: np.ndarray) -> dict[str, object]:
        entry: dict[str, object] = {}
        if not np.array_equal(tile.origin, parent_origin):
            # 3D Tiles 1.1 column-major, translation only, relative to the parent's frame;
            # whole metres, so `+ 0.0` (no negative zero) is all the cleaning it needs.
            d = [float(v) + 0.0 for v in tile.origin - parent_origin]
            entry["transform"] = [
                *(1.0, 0.0, 0.0, 0.0),
                *(0.0, 1.0, 0.0, 0.0),
                *(0.0, 0.0, 1.0, 0.0),
                *(d[0], d[1], d[2], 1.0),
            ]
        entry |= {
            # In the tile's own frame, as 3D Tiles reads a bounding volume.
            "boundingVolume": {"box": _box(tile.low - tile.origin, tile.high - tile.origin)},
            "geometricError": tile.geometric_error,
            "content": {"uri": tile.uri},
            # How many gaussians the content holds, so a viewer can spend a budget on the
            # tiles before fetching any of them (apps/web src/view/tiles.ts, and the
            # globe's splat-count cap in src/cesium/SiteManager.ts).
            "extras": {"gaussians": tile.count},
        }
        if tile.confidence_min is not None:
            # The least confident scored cell under it, when a grid was given (`write_tiles`).
            cast(dict[str, object], entry["extras"])["confidenceMin"] = tile.confidence_min
        if tile.children:
            entry["children"] = [node(child, tile.origin) for child in tile.children]
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
        # The root's origin is always zero (`tile_frame`), so its entry has no transform of
        # its own to collide with this one.
        "root": {
            "transform": enu_to_ecef(lat, lon, height),
            "refine": "REPLACE",
            **node(root, np.zeros(3)),
        },
    }
    # What the web clients collide with: the leaves' gaussians as solid cells, precomputed
    # (see `COLLISION_FORMAT`). On the root, so a viewer has it before any tile loads.
    # From where each part of the scan was seen, so a viewer can fade what it never saw
    # (`view_cones`). On the root too: the shader needs it from the first tile. Before the
    # collision grid, so a backfill of either one (each appended last) writes these bytes.
    tileset["root"]["extras"]["viewCones"] = seen_from
    tileset["root"]["extras"]["collision"] = collision
    if scores_extras is not None:
        # How well each part of this epoch was seen, per cell (`cell_grid`): after the
        # collision grid, so a backfill that appends it writes these bytes.
        (out_dir / str(scores_extras["uri"])).write_bytes(scores_blob)
        tileset["root"]["extras"]["confidence"] = scores_extras
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
        # The SH degree every tile carries: the PLY's, or `sh_degree` if that is lower.
        "sh_degree": degree,
        # The collision grid (`collision.bin`): its cell (m), bricks and solid cells.
        "collision_cell_m": grid.cell,
        "collision_bricks": cast(int, collision["bricks"]),
        "collision_solid_cells": int(grid.cells.shape[0]),
        # The view-cone grid (`viewcones.bin`): its cell (m), and the share of gaussians a
        # viewer fades when seen from where the capture never looked.
        "view_cones_cell_m": cones.cell,
        "view_cones_directional_share": cast(float, cones.stats.get("directionalShare", 0.0)),
        # Tiles written in a frame of their own (`tile_frame`), and the fewest fractional
        # bits any tile's SPZ declares: 0 and 12 for every scan under ~4 km across.
        "offset_tiles": sum(1 for tile in tiles if tile.origin.any()),
        "min_fractional_bits": min(tile.fractional_bits for tile in tiles),
    } | (
        {}
        if scores is None
        else {
            # The confidence grid's finest level, and the root's least confident cell (0:
            # no tile holds a scored cell).
            "confidence_level": scores.level,
            "confidence_min": root.confidence_min or 0,
        }
    )


#: Files `convert` writes beside the tiles, which a declared grid's `uri` may not name.
_RESERVED = ("tileset.json", COLLISION_URI, view_cones.URI)


def _confidence(path: Path) -> tuple[cell_grid.CellGrid, bytes, dict[str, Any]]:
    """The confidence grid `path` (its descriptor) describes, its file's bytes, and the
    descriptor -- checked (`cell_grid.decode`: format, sizes, sha256) before anything is
    written, and refused unless it is a confidence grid with a plain file name."""
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict):
        raise cell_grid.CellGridError(f"{path.name} is not a cell grid descriptor")
    uri = str(descriptor.get("uri") or "")
    if (
        not uri
        or "/" in uri
        or "\\" in uri
        or uri.startswith(".")
        or uri in _RESERVED
        or (uri.startswith("splat") and uri.endswith(".glb"))
    ):
        raise cell_grid.CellGridError(f"{path.name}: {uri!r} is not a file name a tileset can hold")
    blob = (path.parent / uri).read_bytes()
    grid = cell_grid.decode(descriptor, blob)
    if grid.kind != "confidence":
        raise cell_grid.CellGridError(f"{path.name} is a {grid.kind!r} grid, not a confidence grid")
    return grid, blob, descriptor


def _translation(transform: object, uri: str) -> np.ndarray:
    """The translation of a tile's 3D Tiles `transform`, refusing anything else."""
    if (
        not isinstance(transform, list)
        or len(transform) != 16
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in transform
        )
    ):
        raise SplatFormatError(f"{uri}: its tile transform is not 16 finite numbers")
    m = np.asarray(transform, dtype=np.float64).reshape(4, 4).T  # column-major
    if not (np.array_equal(m[:3, :3], np.eye(3)) and not m[3, :3].any() and float(m[3, 3]) == 1.0):
        raise SplatFormatError(
            f"{uri}: its tile transform is not a pure translation; a splat tile may only be "
            "moved, never turned or scaled (every tile shares the root's SH frame)"
        )
    return m[:3, 3].copy()


class TileOffsets(dict[str, np.ndarray]):
    """`tile_offsets`' result: keyed by content uri without a leading `./`, and looked up
    the same way, so a reader that asks with the uri as the tileset spells it
    (`rig_tiles.tile_uris`: `./splat_0.glb` or `splat_0.glb`) gets that tile's offset and
    never the site frame by default."""

    def __getitem__(self, uri: str) -> np.ndarray:
        return super().__getitem__(uri.removeprefix("./"))

    def __contains__(self, uri: object) -> bool:
        return super().__contains__(uri.removeprefix("./") if isinstance(uri, str) else uri)

    def get(self, uri: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(uri.removeprefix("./"), default)


def tile_offsets(tileset: Mapping[str, Any]) -> TileOffsets:
    """Each content uri's frame in the root's frame: the float64 sum of the translation-only
    `transform`s from the root down to its tile, the root's own (the globe placement)
    excluded. Zero for every tile of a tileset that fits SPZ's range (`tile_frame`).

    Reads `content.uri` and 3D Tiles 1.1 `contents[].uri`, a leading `./` removed (as
    apps/api services/sidecars.py `content_uris` does) from the keys and from every lookup
    (`TileOffsets`). A child transform that is anything but a translation is refused
    (`SplatFormatError`): no reader here composes rotations.
    """
    out = TileOffsets()
    root = tileset.get("root")
    stack: list[tuple[Any, np.ndarray]] = [(root, np.zeros(3))]
    while stack:
        tile, offset = stack.pop()
        if not isinstance(tile, dict):
            continue
        if tile is not root and tile.get("transform") is not None:
            name = (tile.get("content") or {}).get("uri", "a tile")
            offset = offset + _translation(tile["transform"], str(name))
        for content in [tile.get("content"), *(tile.get("contents") or [])]:
            uri = content.get("uri") if isinstance(content, dict) else None
            if isinstance(uri, str) and uri:
                out[uri.removeprefix("./")] = offset
        children = tile.get("children")
        if isinstance(children, list):
            stack.extend((child, offset) for child in children)
    return out


def site_positions(local: np.ndarray, offset: np.ndarray | None) -> np.ndarray:
    """A tile's decoded positions moved into the site frame (the root's): the basis every
    position checksum is defined over (rig `tileChecksums`, instances, skins, plants).

    `float32(float64(local) + offset)`, negative zero cleared. For a tile with no offset
    (every tile of a scan under ~4 km) that is `local` itself, unchanged. Otherwise it is
    exact -- equal to the float64 sum -- wherever |site| < 4096 m (an origin is whole
    metres and `local` lies on the 2^-12 grid, so the sum is on it too, within float32's
    24 bits); further out it is the correctly rounded float32 of the exact sum, which
    TypeScript's `Float32Array` of the same double gives bit for bit.
    """
    if offset is None or not np.asarray(offset).any():
        return local
    out = (local.astype(np.float64) + np.asarray(offset, dtype=np.float64)).astype("<f4")
    out[out == 0.0] = 0.0
    return out


def collision_main(argv: list[str]) -> None:
    """`collision`: the grid alone, for a tileset packed (and published) before it existed."""
    parser = argparse.ArgumentParser(
        prog="splat_tiles.py collision",
        description="Write collision.bin for a splat PLY, and optionally declare it on the "
        "tileset.json packed from that PLY (the one convert would have written).",
    )
    parser.add_argument("ply", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--tileset",
        type=Path,
        default=None,
        help="tileset.json to add root.extras.collision to (checked against the PLY first)",
    )
    parser.add_argument("--opacity-min", type=float, default=0.02)
    args = parser.parse_args(argv)
    extras = build_collision(args.ply, args.out_dir, args.opacity_min, args.tileset)
    size = (args.out_dir / COLLISION_URI).stat().st_size
    print(json.dumps({**extras, "bytes": size}))


def main() -> None:
    if sys.argv[1:2] == ["collision"]:
        collision_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["viewcones"]:
        raise SystemExit(view_cones.main(sys.argv[2:]))
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
    parser.add_argument(
        "--sh-degree",
        type=int,
        choices=range(SH_MAX_DEGREE + 1),
        default=None,
        help="most SH bands the tiles carry (0-3); default: all the PLY's f_rest_* has",
    )
    parser.add_argument(
        "--confidence",
        type=Path,
        default=None,
        help="the epoch's confidence grid descriptor (confidence.json, its .bin beside it)",
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
        sh_degree=args.sh_degree,
        confidence=args.confidence,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
