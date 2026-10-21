"""What a published fill layer may be: exactly what the tiler writes (ADR 0016 § Fill).

The publish pre-check (`rescan_gate.layer_positions`) reads a tile's SPZ block and each
tile's translation, which is the whole placement only on the layout
`splat_tiles.write_tiles` and `splat_tiles.build_glb` write. Cesium honours more than that
-- `content.url`, multiple contents, a second (data-URI) buffer, node transforms,
extensions -- so a layer that uses any of it is refused, never read (Codex R4 on #60: no
denylist). A transcription of the API gate's allowlist (apps/api app/worker/positions.py,
`packer_placements` and `packer_spz`), held to the same cases by name
(tests/test_layer_allowlist.py in both projects).
"""

from __future__ import annotations

import json
import math
import re
import struct
import zlib
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from splat_tiles import SPZ_BYTES_PER_GAUSSIAN, SPZ_MAGIC, SplatFormatError, tile_placements


class LayoutError(ValueError):
    """A layer tileset or tile GLB not laid out as the tiler writes one; the message says why."""


#: The top-level keys of a layer's tileset, and of each tile and its content, as the tiler
#: writes them.
TILESET_KEYS = frozenset({"asset", "geometricError", "extensionsUsed", "extensions", "root"})
TILE_KEYS = frozenset(
    {"transform", "boundingVolume", "geometricError", "refine", "content", "extras", "children"}
)
CONTENT_KEYS = frozenset({"uri"})
#: The one tileset extension the tiler declares, and what it says.
CONTENT_GLTF = "3DTILES_content_gltf"
SPLAT_EXTENSIONS = frozenset({"KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"})
#: A content uri: a plain relative path of plain names (sidecars.check_relative's rule) to a
#: `.glb`, so no `?`, `#`, `%`, `..`, `./`, scheme, absolute path or backslash.
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_DEPTH = 6

#: `build_glb`'s glTF: its top-level keys, extensions, scene, node and primitive.
GLB_KEYS = frozenset(
    {
        "asset",
        "extensionsUsed",
        "extensionsRequired",
        "scene",
        "scenes",
        "nodes",
        "meshes",
        "materials",
        "accessors",
        "bufferViews",
        "buffers",
    }
)
GLB_USED = frozenset({"KHR_materials_unlit", *SPLAT_EXTENSIONS})
PACKER_NODE: dict[str, Any] = {
    "mesh": 0,
    "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
}
PACKER_MATERIALS: list[dict[str, Any]] = [{"extensions": {"KHR_materials_unlit": {}}}]
PACKER_SPLAT: dict[str, Any] = {
    "KHR_gaussian_splatting": {
        "extensions": {"KHR_gaussian_splatting_compression_spz_2": {"bufferView": 0}}
    }
}
PRIMITIVE_KEYS = frozenset({"mode", "material", "attributes", "extensions"})
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942


def _same(a: object, b: object) -> bool:
    """JSON equality that tells `true` from `1` (Python does not)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    return type(a) is type(b) and a == b


def _index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _names(value: object, expected: frozenset[str], what: str) -> None:
    if (
        not isinstance(value, list)
        or not all(isinstance(v, str) for v in value)
        or len(set(value)) != len(value)
        or set(value) != expected
    ):
        raise LayoutError(f"{what} {value!r} is not the packer's {sorted(expected)}")


def _keys(value: object, allowed: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LayoutError(f"{what} is not an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise LayoutError(f"{what} has {unknown}, which the packer never writes")
    return value


def content_uri(uri: object) -> str:
    """`uri` if it is a plain relative path to a `.glb` beside the tileset, else
    `LayoutError`."""
    parts = uri.split("/") if isinstance(uri, str) else []
    if (
        not parts
        or len(parts) > MAX_DEPTH
        or not all(_SEGMENT.fullmatch(part) for part in parts)
        or not parts[-1].endswith(".glb")
    ):
        raise LayoutError(
            f"content uri {uri!r} is not a plain relative path to a .glb (letters, digits, "
            f"'.', '_' and '-' in at most {MAX_DEPTH} segments, none starting with '.')"
        )
    return str(uri)


def packer_placements(document: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """`tile_placements` of a tileset laid out as the tiler writes one, else
    `LayoutError`: the tiler's keys only, at the top, on each tile and on each content;
    content named by `uri` alone (never `url`, `contents` or an extension), each a plain
    relative `.glb`; a box for each bounding volume; child transforms translations."""
    _keys(document, TILESET_KEYS, "the tileset")
    if not _same(document.get("asset"), {"version": "1.1"}):
        raise LayoutError(f"tileset asset {document.get('asset')!r} is not 3D Tiles 1.1")
    if "extensionsUsed" in document or "extensions" in document:
        _names(document.get("extensionsUsed"), frozenset({CONTENT_GLTF}), "extensionsUsed")
        extensions = _keys(document.get("extensions"), frozenset({CONTENT_GLTF}), "extensions")
        declared = _keys(
            extensions.get(CONTENT_GLTF),
            frozenset({"extensionsUsed", "extensionsRequired"}),
            CONTENT_GLTF,
        )
        _names(declared.get("extensionsUsed"), SPLAT_EXTENSIONS, f"{CONTENT_GLTF} used")
        _names(declared.get("extensionsRequired"), SPLAT_EXTENSIONS, f"{CONTENT_GLTF} required")
    _error(document.get("geometricError"))
    root = document.get("root")
    stack: list[tuple[object, tuple[float, float, float]]] = [(root, (0.0, 0.0, 0.0))]
    while stack:
        item, offset = stack.pop()
        tile = _keys(item, TILE_KEYS, "a tile")
        _error(tile.get("geometricError"))
        if tile is not root and "transform" in tile:
            offset = _whole_metres(tile["transform"], offset)
        volume = _keys(tile.get("boundingVolume"), frozenset({"box"}), "a bounding volume")
        _numbers(volume.get("box"), 12, "a bounding box")
        if tile.get("refine", "REPLACE") not in ("REPLACE", "ADD"):
            raise LayoutError(f"refine {tile.get('refine')!r} is not REPLACE or ADD")
        if "content" in tile:
            content_uri(_keys(tile["content"], CONTENT_KEYS, "a tile's content").get("uri"))
        children = tile.get("children", [])
        if not isinstance(children, list):
            raise LayoutError("a tile's children are not a list")
        stack.extend((child, offset) for child in children)
    try:
        return [(uri, offset) for uri, offset in tile_placements(document)]
    except SplatFormatError as error:
        raise LayoutError(str(error)) from error


#: How far from the site frame a tile's origin, or any one translation, may be: SPZ's
#: whole-metre reach (2^23 m, `splat_tiles.SPZ_MIN_FRACTIONAL_BITS`). No tile of one site
#: is farther, and whole metres under 2^24 add exactly in float32 as in float64.
MAX_ORIGIN_M = float(1 << 23)


def _error(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise LayoutError(f"geometricError {value!r} is not a finite number >= 0")


def _whole_metres(
    transform: object, offset: tuple[float, float, float]
) -> tuple[float, float, float]:
    """`offset` moved by a child tile's `transform`, which must be what the tiler writes:
    a pure translation (the rest is `_translation`'s to refuse) by whole metres, each
    component and each composed offset within `MAX_ORIGIN_M`. Whole metres sum exactly,
    so the gate's offsets are Cesium's Matrix4 products and float32's, in any order, with
    nothing to cancel (+1e300 then -1e300 hid a 100 m root in Cesium's arithmetic)."""
    _numbers(transform, 16, "a tile transform")
    step = transform[12:15]  # type: ignore[index]
    moved = tuple(o + t for o, t in zip(offset, step, strict=True))
    if not all(float(v).is_integer() and abs(v) <= MAX_ORIGIN_M for v in (*step, *moved)):
        raise LayoutError(
            f"a tile translation {step!r} (composed {moved!r}) is not whole metres within "
            f"{MAX_ORIGIN_M:g} m of the site frame, as the tiler writes"
        )
    return moved


def _numbers(value: object, count: int, what: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise LayoutError(f"{what} is not {count} finite numbers")


def packer_spz(data: bytes) -> bytes:
    """The SPZ block of a GLB laid out exactly as `splat_tiles.build_glb` writes one, else
    `LayoutError`: a JSON chunk then a BIN chunk and nothing after; one buffer, the BIN
    chunk itself (no `uri`: no data-URI or external buffer); one buffer view in it; the
    packer's extensions exactly, used and required; one scene of one node, that node
    exactly `PACKER_NODE`; one mesh of one point primitive whose splats are that view's SPZ
    (no indices, morph targets, other attributes); accessors with no data of their own (no
    buffer view, no sparse); no skins, animations, instancing or anything else."""
    if len(data) < 28:
        raise LayoutError("not a GLB: shorter than its header")
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise LayoutError("not a whole binary glTF 2.0")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    bin_at = 20 + json_length
    if json_type != _JSON_CHUNK or bin_at + 8 > len(data):
        raise LayoutError("not a GLB of a JSON chunk then a BIN chunk")
    bin_length, bin_type = struct.unpack_from("<II", data, bin_at)
    if bin_type != _BIN_CHUNK or bin_at + 8 + bin_length != len(data):
        raise LayoutError("not a GLB of a JSON chunk then a BIN chunk, and nothing after")
    try:
        gltf = json.loads(data[20:bin_at])
    except ValueError as error:
        raise LayoutError(f"not a splat GLB: {error!r}") from error
    _keys(gltf, GLB_KEYS, "the glTF")
    asset = _keys(gltf.get("asset"), frozenset({"version", "generator"}), "the glTF asset")
    if asset.get("version") != "2.0":
        raise LayoutError(f"glTF version {asset.get('version')!r} is not 2.0")
    _names(gltf.get("extensionsUsed"), GLB_USED, "extensionsUsed")
    _names(gltf.get("extensionsRequired"), SPLAT_EXTENSIONS, "extensionsRequired")
    if not (
        _same(gltf.get("scene"), 0)
        and _same(gltf.get("scenes"), [{"nodes": [0]}])
        and _same(gltf.get("nodes"), [PACKER_NODE])
    ):
        raise LayoutError("not build_glb's one scene of one node with its matrix")
    if "materials" in gltf and not _same(gltf["materials"], PACKER_MATERIALS):
        raise LayoutError("not build_glb's one unlit material")
    meshes = gltf.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise LayoutError("not one mesh")
    primitives = _keys(meshes[0], frozenset({"primitives"}), "the mesh").get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise LayoutError("not one primitive")
    primitive = _keys(primitives[0], PRIMITIVE_KEYS, "the primitive")
    if not _same(primitive.get("extensions"), PACKER_SPLAT):
        raise LayoutError("the primitive's splats are not build_glb's SPZ in view 0")
    if not _same(primitive.get("mode", 0), 0):
        raise LayoutError("the primitive is not points")
    if "material" in primitive and not ("materials" in gltf and _same(primitive["material"], 0)):
        raise LayoutError("the primitive's material is not the unlit one")
    buffers = gltf.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise LayoutError("not one buffer")
    size = _keys(buffers[0], frozenset({"byteLength"}), "the buffer").get("byteLength")
    if not _index(size) or size > bin_length:
        raise LayoutError("the buffer is not the BIN chunk")
    views = gltf.get("bufferViews")
    if not isinstance(views, list) or len(views) != 1:
        raise LayoutError("not one buffer view")
    view = _keys(views[0], frozenset({"buffer", "byteOffset", "byteLength"}), "the view")
    offset, length = view.get("byteOffset", 0), view.get("byteLength")
    if not (_same(view.get("buffer"), 0) and _index(offset) and _index(length)):
        raise LayoutError("the buffer view is not one into the BIN chunk")
    if offset + length > size:
        raise LayoutError("the SPZ block runs past the buffer")
    start = bin_at + 8 + offset
    block = data[start : start + length]
    _packer_accessors(gltf.get("accessors"), primitive.get("attributes"), packer_spz_header(block))
    return block


def _sh_attributes(degree: int) -> list[str]:
    """`splat_tiles.sh_attributes`: the SH accessors' names, band by band."""
    return [
        f"KHR_gaussian_splatting:SH_DEGREE_{band}_COEF_{n}"
        for band in range(1, degree + 1)
        for n in range(2 * band + 1)
    ]


def _packer_accessors(accessors: object, attributes: object, spz: SpzHeader) -> None:
    """Refuses (`LayoutError`) accessors and attributes that are not `build_glb`'s
    value for value for this SPZ: POSITION, COLOR_0, SCALE, ROTATION and the SH degree's
    accessors, in that order and nothing else, each its exact componentType, type and
    `normalized`, its count the SPZ's; POSITION's bounds those of the stored positions
    (within the one step packing moves them). The SPZ path keeps these: a POSITION read as
    unsigned ints draws the float bytes of 0.5 at 1056964608."""
    n = spz.gaussians
    names = [
        "POSITION",
        "COLOR_0",
        "KHR_gaussian_splatting:SCALE",
        "KHR_gaussian_splatting:ROTATION",
        *_sh_attributes(spz.degree),
    ]
    if not _same(attributes, {name: index for index, name in enumerate(names)}):
        raise LayoutError(f"attributes {attributes!r} are not build_glb's for this SPZ")
    expected: list[dict[str, Any]] = [
        {"componentType": 5126, "count": n, "type": "VEC3"},
        {"componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
        {"componentType": 5126, "count": n, "type": "VEC3"},
        {"componentType": 5126, "count": n, "type": "VEC4"},
        *({"componentType": 5126, "count": n, "type": "VEC3"} for _ in names[4:]),
    ]
    if not isinstance(accessors, list) or len(accessors) != len(expected):
        raise LayoutError("the accessors are not build_glb's")
    position = dict(accessors[0]) if isinstance(accessors[0], dict) else {}
    low, high = position.pop("min", None), position.pop("max", None)
    if not all(_same(a, e) for a, e in zip([position, *accessors[1:]], expected, strict=True)):
        raise LayoutError("an accessor's type, count or normalisation is not build_glb's")
    _numbers(low, 3, "POSITION's min")
    _numbers(high, 3, "POSITION's max")
    step = math.ldexp(1.0, spz.bits)
    for bound, stored in ((low, spz.low), (high, spz.high)):
        if any(abs(b * step - s) > 1.0 for b, s in zip(bound, stored, strict=True)):
            raise LayoutError("POSITION's bounds are not those of the stored positions")


#: What `splat_tiles.pack_spz` writes in an SPZ header: version 2, no flags, nothing
#: reserved, SH degree 0 to 3; the fractional bits by the tiler's rule (`_packer_bits`).
PACKER_SPZ_VERSION = 2
PACKER_FRACTIONAL_BITS = 12
SH_DIMS = (0, 3, 8, 15)
#: The largest magnitude 24-bit two's complement fixed point holds (`splat_tiles`).
SPZ_FIXED_MAX = (1 << 23) - 1
#: A tile's SPZ, decompressed, past this is no tile the packer writes (a million
#: gaussians at SH degree 3 are 64 MB): refused before it is inflated.
MAX_SPZ_RAW_BYTES = 256 * 1024 * 1024


class SpzHeader(NamedTuple):
    """What `packer_spz_header` read: the gaussians' count, SH degree and fractional bits, and the
    stored positions' per-axis extremes in steps."""

    gaussians: int
    degree: int
    bits: int
    low: tuple[int, int, int]
    high: tuple[int, int, int]


def packer_spz_header(block: bytes) -> SpzHeader:
    """`SpzHeader` of an SPZ stream exactly as `splat_tiles.pack_spz` writes one,
    else `LayoutError`: one gzip member and nothing after it, holding a 16-byte header
    and exactly the body it declares; magic, version 2, no flags or reserved byte, SH degree
    0 to 3; and the fractional bits the tiler gives such a tile (`_packer_bits`). Python
    and Cesium's SPZ loader agree on what such a stream holds; on others they need not (at
    32 bits Python reads 2048 / 2^32 m where Cesium draws 2048 m). Inflated no further than
    the header says it reaches."""
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        head = inflater.decompress(block, 16)
        if len(head) < 16:
            raise LayoutError("not an SPZ stream: fewer than 16 bytes")
        magic, version, count, degree, bits, flags, reserved = struct.unpack("<IIIBBBB", head)
        if magic != SPZ_MAGIC or version != PACKER_SPZ_VERSION or flags or reserved:
            raise LayoutError(
                f"not the packer's SPZ header (magic {magic:#x}, version {version}, flags "
                f"{flags}, reserved {reserved})"
            )
        if degree >= len(SH_DIMS):
            raise LayoutError(f"SPZ SH degree {degree} is not 0 to 3")
        need = (SPZ_BYTES_PER_GAUSSIAN + 3 * SH_DIMS[degree]) * count
        if need > MAX_SPZ_RAW_BYTES:
            raise LayoutError(f"{count} gaussians: past what one packed tile holds")
        body = inflater.decompress(inflater.unconsumed_tail, need + 1)
    except zlib.error as error:
        raise LayoutError(f"not an SPZ stream: {error}") from error
    if len(body) != need or inflater.unconsumed_tail or not inflater.eof:
        raise LayoutError(f"the SPZ body is not the {need} bytes its header declares")
    if inflater.unused_data:
        raise LayoutError("bytes after the SPZ stream's one gzip member")
    if not count:
        raise LayoutError("an SPZ of no gaussians: the packer writes no empty tile")
    packed = np.frombuffer(body, dtype=np.uint8, count=9 * count).reshape(count, 3, 3)
    wide = packed.astype(np.int64)
    fixed = ((wide[:, :, 0] | (wide[:, :, 1] << 8) | (wide[:, :, 2] << 16)) ^ 0x800000) - 0x800000
    _packer_bits(bits, int(np.abs(fixed).max()))
    low = tuple(int(v) for v in fixed.min(axis=0))
    high = tuple(int(v) for v in fixed.max(axis=0))
    return SpzHeader(count, degree, bits, low, high)  # type: ignore[arg-type]


def _packer_bits(bits: int, largest: int) -> None:
    """The tiler's rule (`splat_tiles.fractional_bits_for`): the most fractional bits, 12 at
    most, at which the tile's farthest position component still fits 24 bits. So 12, or
    fewer only when one more would not fit: with `m` the largest stored magnitude, `bits`
    is 12 or `2 m` overflows."""
    if (
        bits > PACKER_FRACTIONAL_BITS
        or largest > SPZ_FIXED_MAX
        or (bits < PACKER_FRACTIONAL_BITS and 2 * largest <= SPZ_FIXED_MAX)
    ):
        raise LayoutError(
            f"{bits} fractional bits for positions reaching {largest} steps: not what the "
            f"tiler writes (12, or the most at which the farthest one fits 24 bits)"
        )
