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


# --- what the viewer draws of a tile: the fill gate's reader -----------------------------
#
# The checksums above read the one SPZ block `splat_tiles.build_glb` writes, as the packer
# meant it. A fill gate cannot assume the packer wrote the GLB: it must see every splat any
# viewer draws, where it draws it. Cesium (GaussianSplat3DTileContent, GaussianSplatPrimitive)
# draws the first primitive of the scene's first node, placed by that node's own `matrix`
# and turned z-up (`Axis.Y_UP_TO_Z_UP`, always, whatever the tileset's `gltfUpAxis`); a glTF
# viewer that follows the spec draws every primitive of every node, through the hierarchy,
# `matrix` or TRS. `tile_draws` reads both, so a gaussian is held to the targets wherever
# either would put it, and refuses what neither can be read for: an instanced, skinned,
# animated or morphed node, a primitive that is not an SPZ splat, a node hierarchy that is
# not a forest.

#: glTF is y-up; the viewer turns a splat tile z-up: `(x, y, z) -> (x, -z, y)`.
Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
#: Nodes a splat tile may hold: the packer writes one. Each placement is a pass over its
#: primitive's gaussians, so the count bounds the work a small file can ask for.
MAX_NODES = 64


def _numbers(value: object, count: int, what: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise TileFormatError(f"{what} is not {count} finite numbers")
    return np.asarray(value, dtype=np.float64)


def _local(node: Mapping[str, Any], index: int) -> np.ndarray:
    """A node's own transform, `matrix` (column-major) or translation x rotation x scale."""
    what = f"node {index}"
    trs = [key for key in ("translation", "rotation", "scale") if key in node]
    if "matrix" in node:
        if trs:
            raise TileFormatError(f"{what} has both a matrix and {', '.join(trs)}")
        out: np.ndarray = _numbers(node["matrix"], 16, f"{what}'s matrix").reshape(4, 4).T
        return out
    t = _numbers(node.get("translation", [0.0, 0.0, 0.0]), 3, f"{what}'s translation")
    x, y, z, w = _numbers(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), 4, f"{what}'s rotation")
    scale = _numbers(node.get("scale", [1.0, 1.0, 1.0]), 3, f"{what}'s scale")
    norm = x * x + y * y + z * z + w * w
    if not abs(norm - 1.0) < 1e-6:
        raise TileFormatError(f"{what}'s rotation is not a unit quaternion")
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rotation * scale
    out[:3, 3] = t
    return out


def _glb_json(data: bytes) -> tuple[dict[str, Any], int]:
    if len(data) < 20:
        raise TileFormatError("not a GLB: shorter than its header")
    magic, _version, _total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise TileFormatError("not a GLB")
    json_length, _json_type = struct.unpack_from("<II", data, 12)
    try:
        gltf = json.loads(data[20 : 20 + json_length])
    except ValueError as error:
        raise TileFormatError(f"not a splat GLB: {error!r}") from error
    if not isinstance(gltf, dict):
        raise TileFormatError("not a splat GLB: its JSON is not an object")
    return gltf, 20 + json_length + 8


def _spz_of(gltf: Mapping[str, Any], bin_offset: int, data: bytes, primitive: object) -> bytes:
    """The SPZ block a primitive's splats are in; a `TileFormatError` for any primitive
    that is not an SPZ splat (a mesh of its own would be drawn wherever it says)."""
    try:
        if not isinstance(primitive, Mapping) or primitive.get("targets"):
            raise TypeError("a morphed primitive, or not one")
        view_index = primitive["extensions"]["KHR_gaussian_splatting"]["extensions"][
            "KHR_gaussian_splatting_compression_spz_2"
        ]["bufferView"]
        view = gltf["bufferViews"][view_index]
        start = bin_offset + int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise TileFormatError(f"a primitive that is not an SPZ splat: {error!r}") from error
    if start < bin_offset or length < 0 or start + length > len(data):
        raise TileFormatError("an SPZ block runs past the end of the GLB")
    return data[start : start + length]


def tile_draws(data: bytes) -> list[tuple[bytes, np.ndarray]]:
    """Every `(SPZ block, matrix)` a viewer may draw of this GLB: the matrix (4 x 4, into
    the tile's z-up frame) of each primitive of each node, composed from the root of its
    hierarchy, and Cesium's own reading (the module note above), each placement once.
    `TileFormatError` for a GLB the gate cannot place every splat of."""
    gltf, bin_offset = _glb_json(data)
    if gltf.get("animations") or gltf.get("skins"):
        raise TileFormatError("an animated or skinned GLB: its splats move as it plays")
    nodes = gltf.get("nodes")
    meshes = gltf.get("meshes")
    if not isinstance(nodes, list) or not nodes:
        raise TileFormatError("no node places its splats (the viewer draws the first node's)")
    if len(nodes) > MAX_NODES:
        raise TileFormatError(f"{len(nodes)} nodes, over {MAX_NODES}")
    if not isinstance(meshes, list):
        raise TileFormatError("no meshes")
    parent: dict[int, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TileFormatError(f"node {index} is not an object")
        if node.get("extensions") or "skin" in node or "weights" in node:
            raise TileFormatError(
                f"node {index} carries an extension, a skin or morph weights (instancing "
                "draws its mesh many times over)"
            )
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TileFormatError(f"node {index}'s children are not a list")
        for child in children:
            if (
                isinstance(child, bool)
                or not isinstance(child, int)
                or not 0 <= child < len(nodes)
                or child in parent
                or child == index
            ):
                raise TileFormatError(f"node {index}: child {child!r} is not a node of one parent")
            parent[child] = index
    world: dict[int, np.ndarray] = {}
    order = [index for index in range(len(nodes)) if index not in parent]
    for index in order:  # grows as children are reached: breadth first from each root
        local = _local(nodes[index], index)
        world[index] = world[parent[index]] @ local if index in parent else local
        order.extend(nodes[index].get("children", []))
    if len(world) != len(nodes):
        raise TileFormatError("the node hierarchy has a cycle")
    draws: list[tuple[int, int, np.ndarray]] = []
    placed: set[int] = set()
    for index, node in enumerate(nodes):
        mesh = node.get("mesh")
        if mesh is None:
            continue
        if isinstance(mesh, bool) or not isinstance(mesh, int) or not 0 <= mesh < len(meshes):
            raise TileFormatError(f"node {index}'s mesh {mesh!r} is not one of the GLB's")
        placed.add(mesh)
        primitives = meshes[mesh].get("primitives") if isinstance(meshes[mesh], dict) else None
        if not isinstance(primitives, list) or not primitives:
            raise TileFormatError(f"mesh {mesh} has no primitives")
        draws.extend((mesh, n, world[index]) for n in range(len(primitives)))
    if placed != set(range(len(meshes))):
        raise TileFormatError("a mesh no node places, or no mesh at all")
    # Cesium: the scene's first root node, its own `matrix` only, its first primitive. Its
    # roots are `scenes[scene].nodes` when both are given, else every node (`nodes[0]`).
    scenes, scene = gltf.get("scenes"), gltf.get("scene")
    first: object = 0
    if isinstance(scenes, list) and scene is not None:
        listed = scenes[scene] if isinstance(scene, int) and 0 <= scene < len(scenes) else None
        ids = listed.get("nodes") if isinstance(listed, dict) else None
        first = ids[0] if isinstance(ids, list) and ids else None
    if isinstance(first, bool) or not isinstance(first, int) or not 0 <= first < len(nodes):
        raise TileFormatError(f"the scene's first node {first!r} is not one of the GLB's")
    cesium = nodes[first]
    if cesium.get("mesh") is None:
        raise TileFormatError("the scene's first node has no mesh (the viewer draws its)")
    own = cesium.get("matrix")
    matrix = (
        np.eye(4) if own is None else _numbers(own, 16, "the first node's matrix").reshape(4, 4).T
    )
    draws.append((int(cesium["mesh"]), 0, matrix))
    out: list[tuple[bytes, np.ndarray]] = []
    seen: list[tuple[int, int, np.ndarray]] = []
    for mesh, n, at in draws:
        if any(m == mesh and k == n and np.array_equal(a, at) for m, k, a in seen):
            continue
        seen.append((mesh, n, at))
        spz = _spz_of(gltf, bin_offset, data, meshes[mesh]["primitives"][n])
        out.append((spz, Y_UP_TO_Z_UP @ at))
    return out


def drawn_positions(data: bytes, offset: np.ndarray | None) -> list[np.ndarray]:
    """Each of `tile_draws`' placements as `(n, 3)` float32 positions in the site frame:
    the matrix applied in float64, `offset` added, rounded once to float32 -- for the
    packer's GLB (its matrix undone by the viewer's turn) exactly `tile_positions`."""
    out: list[np.ndarray] = []
    shift = np.zeros(3) if offset is None else np.asarray(offset, np.float64)
    for spz, matrix in tile_draws(data):
        local = spz_positions(spz).astype("<f4")
        local[local == 0.0] = 0.0
        xyz = local.astype(np.float64)
        if not np.array_equal(matrix, np.eye(4)):
            if not np.array_equal(matrix[3], [0.0, 0.0, 0.0, 1.0]):
                raise TileFormatError("a node matrix with a projective row")
            xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        moved = (xyz + shift).astype("<f4")
        moved[moved == 0.0] = 0.0
        out.append(moved)
    return out
