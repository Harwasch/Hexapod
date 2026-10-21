"""Per-tile position checksums: what objects, skins and a plant rig are bound to.

`instances.json`, `skin.json`, the rig's `plants.json` and `rig.json`'s `tileChecksums` key
every tile by one string: FNV-1a over the tile's decoded float32 positions in the **site
frame** (the tileset root's), in its own gaussian order, plus the count
(`fnv1a32:<n>:<8 hex>`). A tile of a site past ~4 km may be written in a frame of its own,
a translation-only `transform` relative to its parent (`splat_tiles.tile_frame`); its
site positions are `float32(local + composed offset)` (`tile_offsets`). Every tile of a
smaller scan has no transform and an offset of zero, and is hashed as it is decoded.
The viewer computes the same string
for each tile it loads (`checksumPositions`, packages/world/src/rig.ts) and refuses a tile
the binding does not list, so a binding holds for new tiles exactly when every new tile's
checksum is one it lists -- whatever else in the tiles changed. A re-pack with another
spherical-harmonics degree (`ship_sh_degree`), say, rewrites every tile's bytes and leaves
every position where it was.

This is a transcription of the tools/captures code that writes those keys, and must stay
one: `rig_tiles.glb_spz` (the SPZ block of a tile's GLB), the position half of
`splat_tiles.unpack_spz` (24-bit fixed point over `fractional_bits`, SPZ versions 2 and 3),
`splat_tiles.tile_offsets` and `site_positions`, `rig_tiles.tile_positions` (little-endian
float32, negative zero cleared) and `synthetic_tree.checksum_positions`.
tests/test_tile_positions.py holds it to the vectors both languages' tests read
(data/tiles/synthetic-tree/source/checksum_vectors.json), to the committed fixture tree's
`rig.json`, which tools/captures stamped, and to the far fixture's expected.json
(data/tiles/synthetic-far), whose tiles have transforms.

numpy is the worker's dependency, not the API's (pyproject.toml); only `app/worker` imports
this. FNV-1a is a byte-serial recurrence, so the hash itself is a Python loop: about 0.1 s
for a tile of 100k gaussians (1.2 MB of positions), which is why `carry` asks for it only
when a kind bound to positions could otherwise be kept, and stops at the first tile that
rules every such kind out.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import struct
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np

#: "NGSP", read as a little-endian uint32 (`splat_tiles.SPZ_MAGIC`).
SPZ_MAGIC = 0x5053474E
#: The SPZ versions `splat_tiles.unpack_spz` reads; they differ only after the positions.
SPZ_READABLE_VERSIONS = (2, 3)
#: Bytes per gaussian in a version 2 stream (`splat_tiles.SPZ_BYTES_PER_GAUSSIAN`): nine of
#: position, one alpha, three colour, three scale, three rotation (four in version 3).
SPZ_BYTES_PER_GAUSSIAN = 19

FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193


class TileFormatError(ValueError):
    """Bytes that are not a splat tile this can read positions from."""


def glb_spz(data: bytes) -> bytes:
    """The SPZ block of a GLB written by `splat_tiles.build_glb` (`rig_tiles.glb_spz`)."""
    if len(data) < 20:
        raise TileFormatError("not a GLB: shorter than its header")
    magic, _version, _total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise TileFormatError("not a GLB")
    json_length, _json_type = struct.unpack_from("<II", data, 12)
    try:
        gltf = json.loads(data[20 : 20 + json_length])
        bin_offset = 20 + json_length + 8
        primitive = gltf["meshes"][0]["primitives"][0]
        view_index = primitive["extensions"]["KHR_gaussian_splatting"]["extensions"][
            "KHR_gaussian_splatting_compression_spz_2"
        ]["bufferView"]
        view = gltf["bufferViews"][view_index]
        start = bin_offset + int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise TileFormatError(f"not a splat GLB: {error!r}") from error
    if start + length > len(data):
        raise TileFormatError("the SPZ block runs past the end of the GLB")
    return data[start : start + length]


def spz_positions(blob: bytes) -> np.ndarray:
    """`(n, 3)` float32 positions of an SPZ stream, exactly as `splat_tiles.unpack_spz`
    decodes them: 24-bit two's complement over `2**fractional_bits`, divided in float64
    and rounded to float32."""
    try:
        raw = gzip.decompress(blob)
    except (OSError, EOFError, zlib.error) as error:
        raise TileFormatError(f"not an SPZ stream: {error}") from error
    if len(raw) < 16:
        raise TileFormatError("not an SPZ stream: fewer than 16 bytes")
    magic, version, count, _sh_degree, fractional_bits, _flags, _reserved = struct.unpack_from(
        "<IIIBBBB", raw, 0
    )
    if magic != SPZ_MAGIC or version not in SPZ_READABLE_VERSIONS:
        raise TileFormatError(f"not a readable SPZ stream (magic {magic:#x}, version {version})")
    per_gaussian = SPZ_BYTES_PER_GAUSSIAN - 3 + (3 if version == 2 else 4)
    if len(raw) < 16 + per_gaussian * count:
        raise TileFormatError(f"SPZ is truncated: {count} gaussians and {len(raw)} bytes")
    body = np.frombuffer(raw, dtype=np.uint8, count=9 * count, offset=16)
    packed = body.reshape(count, 3, 3).astype(np.uint32)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    signed = (fixed.astype(np.int64) ^ 0x800000) - 0x800000
    xyz: np.ndarray = (signed / float(1 << fractional_bits)).astype(np.float32)
    return xyz


def _translation(transform: object, uri: str) -> np.ndarray:
    """A child tile's `transform` as its translation, refusing anything else
    (`splat_tiles._translation`)."""
    if (
        not isinstance(transform, list)
        or len(transform) != 16
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in transform
        )
    ):
        raise TileFormatError(f"{uri}: its tile transform is not 16 finite numbers")
    m = np.asarray(transform, dtype=np.float64).reshape(4, 4).T  # column-major
    if not (np.array_equal(m[:3, :3], np.eye(3)) and not m[3, :3].any() and float(m[3, 3]) == 1.0):
        raise TileFormatError(f"{uri}: its tile transform is not a pure translation")
    out: np.ndarray = m[:3, 3].copy()
    return out


def tile_placements(document: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Every `(content uri, offset)` the tileset draws, one per tile that names it, in the
    order `tile_offsets` visits them: the float64 sum of the child translations from the
    root down, the root's own transform (the globe placement) excluded. `content.uri` and
    3D Tiles 1.1 `contents[].uri`, a leading `./` removed. A uri two tiles name is drawn
    twice, once at each tile's offset, so it is here twice (`tile_offsets` keeps one).
    Raises `TileFormatError` on a child transform that is not a pure translation."""
    out: list[tuple[str, np.ndarray]] = []
    root = document.get("root")
    stack: list[tuple[Any, np.ndarray]] = [(root, np.zeros(3))]
    while stack:
        tile, offset = stack.pop()
        if not isinstance(tile, dict):
            continue
        if tile is not root and tile.get("transform") is not None:
            content = tile.get("content")
            name = content.get("uri", "a tile") if isinstance(content, dict) else "a tile"
            offset = offset + _translation(tile["transform"], str(name))
        for content in [tile.get("content"), *(tile.get("contents") or [])]:
            uri = content.get("uri") if isinstance(content, dict) else None
            if isinstance(uri, str) and uri:
                out.append((uri.removeprefix("./"), offset))
        children = tile.get("children")
        if isinstance(children, list):
            stack.extend((child, offset) for child in children)
    return out


def tile_offsets(document: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Each content uri's frame in the root's frame (`splat_tiles.tile_offsets`): its
    `tile_placements` offset, the last visited when two tiles name one uri. What a
    position checksum is keyed by; a gate that must see every drawn copy reads
    `tile_placements`."""
    return dict(tile_placements(document))


def tile_positions(data: bytes, offset: np.ndarray | None) -> np.ndarray:
    """A tile's canonical positions from its GLB bytes, in the site frame
    (`rig_tiles.tile_positions`): `offset` is its entry in `tile_offsets`, required so no
    caller forgets it, None for a tile known to be in the site frame."""
    positions = spz_positions(glb_spz(data)).astype("<f4")
    # SPZ's integer round trip has no negative zero; neither does the runtime's un-bake.
    positions[positions == 0.0] = 0.0
    if offset is None or not np.asarray(offset).any():
        return positions
    # `splat_tiles.site_positions`: the float64 sum, rounded once to float32.
    moved = (positions.astype(np.float64) + np.asarray(offset, np.float64)).astype("<f4")
    moved[moved == 0.0] = 0.0
    return moved


def checksum_positions(positions: np.ndarray) -> str:
    """FNV-1a over the raw little-endian float32 bytes, plus the splat count
    (`synthetic_tree.checksum_positions`)."""
    flat = np.ascontiguousarray(positions, dtype="<f4").reshape(-1)
    digest = FNV_OFFSET
    for byte in flat.tobytes():
        digest = ((digest ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return f"fnv1a32:{flat.size // 3}:{digest:08x}"


def tile_checksum(data: bytes, offset: np.ndarray | None) -> str:
    """The position checksum a binding keys this tile (its GLB bytes, at `offset` in the
    site frame) by."""
    return checksum_positions(tile_positions(data, offset))


# --- the fill gate's allowlist: only what the packer writes --------------------------------
#
# The checksums above read a tile as the packer meant it. The fill gate cannot assume the
# packer wrote it: it must see every splat the viewer draws, where it draws it, and each
# time the gate read more of what Cesium honours (a second copy of a tile, a node matrix),
# review found something else it honours (`content.url`, 3DTILES_multiple_contents, an SPZ
# block in a second buffer). So the gate reads no general tileset or glTF: it admits exactly
# what tools/captures' tiler (`splat_tiles.write_tiles`) and `splat_tiles.build_glb` write,
# and refuses anything else (`TileFormatError`). On that layout the SPZ positions, each tile's
# composed translation and the viewer's turn of the node's matrix are the whole placement,
# so `tile_positions` is where the viewer draws every gaussian.
# tools/captures/rescan_gate.py holds the publish pre-check to the same allowlist.

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
ACCESSOR_KEYS = frozenset({"componentType", "count", "type", "min", "max", "normalized"})
_ATTRIBUTE = re.compile(
    r"POSITION|COLOR_0|KHR_gaussian_splatting:(SCALE|ROTATION|SH_DEGREE_[1-3]_COEF_[0-9]{1,2})"
)
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
        raise TileFormatError(f"{what} {value!r} is not the packer's {sorted(expected)}")


def _keys(value: object, allowed: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TileFormatError(f"{what} is not an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TileFormatError(f"{what} has {unknown}, which the packer never writes")
    return value


def content_uri(uri: object) -> str:
    """`uri` if it is a plain relative path to a `.glb` beside the tileset, else
    `TileFormatError`."""
    parts = uri.split("/") if isinstance(uri, str) else []
    if (
        not parts
        or len(parts) > MAX_DEPTH
        or not all(_SEGMENT.fullmatch(part) for part in parts)
        or not parts[-1].endswith(".glb")
    ):
        raise TileFormatError(
            f"content uri {uri!r} is not a plain relative path to a .glb (letters, digits, "
            f"'.', '_' and '-' in at most {MAX_DEPTH} segments, none starting with '.')"
        )
    return str(uri)


def packer_placements(document: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """`tile_placements` of a tileset laid out as the tiler writes one, else
    `TileFormatError`: the tiler's keys only, at the top, on each tile and on each content;
    content named by `uri` alone (never `url`, `contents` or an extension), each a plain
    relative `.glb`; a box for each bounding volume; child transforms translations."""
    _keys(document, TILESET_KEYS, "the tileset")
    if not _same(document.get("asset"), {"version": "1.1"}):
        raise TileFormatError(f"tileset asset {document.get('asset')!r} is not 3D Tiles 1.1")
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
    stack: list[object] = [document.get("root")]
    while stack:
        tile = _keys(stack.pop(), TILE_KEYS, "a tile")
        volume = _keys(tile.get("boundingVolume"), frozenset({"box"}), "a bounding volume")
        _numbers(volume.get("box"), 12, "a bounding box")
        if tile.get("refine", "REPLACE") not in ("REPLACE", "ADD"):
            raise TileFormatError(f"refine {tile.get('refine')!r} is not REPLACE or ADD")
        if "content" in tile:
            content_uri(_keys(tile["content"], CONTENT_KEYS, "a tile's content").get("uri"))
        children = tile.get("children", [])
        if not isinstance(children, list):
            raise TileFormatError("a tile's children are not a list")
        stack.extend(children)
    return tile_placements(document)


def _numbers(value: object, count: int, what: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise TileFormatError(f"{what} is not {count} finite numbers")


def packer_spz(data: bytes) -> bytes:
    """The SPZ block of a GLB laid out exactly as `splat_tiles.build_glb` writes one, else
    `TileFormatError`: a JSON chunk then a BIN chunk and nothing after; one buffer, the BIN
    chunk itself (no `uri`: no data-URI or external buffer); one buffer view in it; the
    packer's extensions exactly, used and required; one scene of one node, that node
    exactly `PACKER_NODE`; one mesh of one point primitive whose splats are that view's SPZ
    (no indices, morph targets, other attributes); accessors with no data of their own (no
    buffer view, no sparse); no skins, animations, instancing or anything else."""
    if len(data) < 28:
        raise TileFormatError("not a GLB: shorter than its header")
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise TileFormatError("not a whole binary glTF 2.0")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    bin_at = 20 + json_length
    if json_type != _JSON_CHUNK or bin_at + 8 > len(data):
        raise TileFormatError("not a GLB of a JSON chunk then a BIN chunk")
    bin_length, bin_type = struct.unpack_from("<II", data, bin_at)
    if bin_type != _BIN_CHUNK or bin_at + 8 + bin_length != len(data):
        raise TileFormatError("not a GLB of a JSON chunk then a BIN chunk, and nothing after")
    try:
        gltf = json.loads(data[20:bin_at])
    except ValueError as error:
        raise TileFormatError(f"not a splat GLB: {error!r}") from error
    _keys(gltf, GLB_KEYS, "the glTF")
    asset = _keys(gltf.get("asset"), frozenset({"version", "generator"}), "the glTF asset")
    if asset.get("version") != "2.0":
        raise TileFormatError(f"glTF version {asset.get('version')!r} is not 2.0")
    _names(gltf.get("extensionsUsed"), GLB_USED, "extensionsUsed")
    _names(gltf.get("extensionsRequired"), SPLAT_EXTENSIONS, "extensionsRequired")
    if not (
        _same(gltf.get("scene"), 0)
        and _same(gltf.get("scenes"), [{"nodes": [0]}])
        and _same(gltf.get("nodes"), [PACKER_NODE])
    ):
        raise TileFormatError("not build_glb's one scene of one node with its matrix")
    if "materials" in gltf and not _same(gltf["materials"], PACKER_MATERIALS):
        raise TileFormatError("not build_glb's one unlit material")
    meshes = gltf.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise TileFormatError("not one mesh")
    primitives = _keys(meshes[0], frozenset({"primitives"}), "the mesh").get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise TileFormatError("not one primitive")
    primitive = _keys(primitives[0], PRIMITIVE_KEYS, "the primitive")
    if not _same(primitive.get("extensions"), PACKER_SPLAT):
        raise TileFormatError("the primitive's splats are not build_glb's SPZ in view 0")
    if not _same(primitive.get("mode", 0), 0):
        raise TileFormatError("the primitive is not points")
    if "material" in primitive and not ("materials" in gltf and _same(primitive["material"], 0)):
        raise TileFormatError("the primitive's material is not the unlit one")
    accessors = gltf.get("accessors")
    if not isinstance(accessors, list):
        raise TileFormatError("no accessors")
    for accessor in accessors:
        _keys(accessor, ACCESSOR_KEYS, "an accessor")
    attributes = primitive.get("attributes")
    if (
        not isinstance(attributes, dict)
        or "POSITION" not in attributes
        or not all(_ATTRIBUTE.fullmatch(name) for name in attributes)
        or not all(_index(v) and v < len(accessors) for v in attributes.values())
    ):
        raise TileFormatError(f"attributes {attributes!r} are not build_glb's")
    buffers = gltf.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise TileFormatError("not one buffer")
    size = _keys(buffers[0], frozenset({"byteLength"}), "the buffer").get("byteLength")
    if not _index(size) or size > bin_length:
        raise TileFormatError("the buffer is not the BIN chunk")
    views = gltf.get("bufferViews")
    if not isinstance(views, list) or len(views) != 1:
        raise TileFormatError("not one buffer view")
    view = _keys(views[0], frozenset({"buffer", "byteOffset", "byteLength"}), "the view")
    offset, length = view.get("byteOffset", 0), view.get("byteLength")
    if not (_same(view.get("buffer"), 0) and _index(offset) and _index(length)):
        raise TileFormatError("the buffer view is not one into the BIN chunk")
    if offset + length > size:
        raise TileFormatError("the SPZ block runs past the buffer")
    start = bin_at + 8 + offset
    block = data[start : start + length]
    packer_spz_header(block)
    return block


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


def packer_spz_header(block: bytes) -> tuple[int, int]:
    """`(count, sh degree)` of an SPZ stream exactly as `splat_tiles.pack_spz` writes one,
    else `TileFormatError`: one gzip member and nothing after it, holding a 16-byte header
    and exactly the body it declares; magic, version 2, no flags or reserved byte, SH degree
    0 to 3; and the fractional bits the tiler gives such a tile (`_packer_bits`). Python
    and Cesium's SPZ loader agree on what such a stream holds; on others they need not (at
    32 bits Python reads 2048 / 2^32 m where Cesium draws 2048 m). Inflated no further than
    the header says it reaches."""
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        head = inflater.decompress(block, 16)
        if len(head) < 16:
            raise TileFormatError("not an SPZ stream: fewer than 16 bytes")
        magic, version, count, degree, bits, flags, reserved = struct.unpack("<IIIBBBB", head)
        if magic != SPZ_MAGIC or version != PACKER_SPZ_VERSION or flags or reserved:
            raise TileFormatError(
                f"not the packer's SPZ header (magic {magic:#x}, version {version}, flags "
                f"{flags}, reserved {reserved})"
            )
        if degree >= len(SH_DIMS):
            raise TileFormatError(f"SPZ SH degree {degree} is not 0 to 3")
        need = (SPZ_BYTES_PER_GAUSSIAN + 3 * SH_DIMS[degree]) * count
        if need > MAX_SPZ_RAW_BYTES:
            raise TileFormatError(f"{count} gaussians: past what one packed tile holds")
        body = inflater.decompress(inflater.unconsumed_tail, need + 1)
    except zlib.error as error:
        raise TileFormatError(f"not an SPZ stream: {error}") from error
    if len(body) != need or inflater.unconsumed_tail or not inflater.eof:
        raise TileFormatError(f"the SPZ body is not the {need} bytes its header declares")
    if inflater.unused_data:
        raise TileFormatError("bytes after the SPZ stream's one gzip member")
    _packer_bits(bits, body[: 9 * count], count)
    return count, degree


def _packer_bits(bits: int, positions: bytes, count: int) -> None:
    """The tiler's rule (`splat_tiles.fractional_bits_for`): the most fractional bits, 12 at
    most, at which the tile's farthest position component still fits 24 bits. So 12, or
    fewer only when one more would not fit: with `m` the largest stored magnitude, `bits`
    is 12 or `2 m` overflows."""
    if count:
        packed = np.frombuffer(positions, dtype=np.uint8).reshape(count * 3, 3).astype(np.int64)
        fixed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        largest = int(np.abs((fixed ^ 0x800000) - 0x800000).max())
    else:
        largest = 0
    if (
        bits > PACKER_FRACTIONAL_BITS
        or largest > SPZ_FIXED_MAX
        or (bits < PACKER_FRACTIONAL_BITS and 2 * largest <= SPZ_FIXED_MAX)
    ):
        raise TileFormatError(
            f"{bits} fractional bits for positions reaching {largest} steps: not what the "
            f"tiler writes (12, or the most at which the farthest one fits 24 bits)"
        )


def packer_positions(data: bytes, offset: np.ndarray | None) -> np.ndarray:
    """`tile_positions` of a GLB `packer_spz` admits: on build_glb's layout, where the viewer
    draws every gaussian (the node's turn undone by the viewer's)."""
    packer_spz(data)
    return tile_positions(data, offset)
