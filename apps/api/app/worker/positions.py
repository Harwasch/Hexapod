"""Per-tile position checksums: what objects, skins and a plant rig are bound to.

`instances.json`, `skin.json`, the rig's `plants.json` and `rig.json`'s `tileChecksums` key
every tile by one string: FNV-1a over the tile's decoded float32 positions, in its own
gaussian order, plus the count (`fnv1a32:<n>:<8 hex>`). The viewer computes the same string
for each tile it loads (`checksumPositions`, packages/world/src/rig.ts) and refuses a tile
the binding does not list, so a binding holds for new tiles exactly when every new tile's
checksum is one it lists -- whatever else in the tiles changed. A re-pack with another
spherical-harmonics degree (`ship_sh_degree`), say, rewrites every tile's bytes and leaves
every position where it was.

This is a transcription of the tools/captures code that writes those keys, and must stay
one: `rig_tiles.glb_spz` (the SPZ block of a tile's GLB), the position half of
`splat_tiles.unpack_spz` (24-bit fixed point over `fractional_bits`, SPZ versions 2 and 3),
`rig_tiles.tile_positions` (little-endian float32, negative zero cleared) and
`synthetic_tree.checksum_positions`. tests/test_tile_positions.py holds it to the vectors
both languages' tests read (data/tiles/synthetic-tree/source/checksum_vectors.json) and to
the committed fixture tree's `rig.json`, which tools/captures stamped.

numpy is the worker's dependency, not the API's (pyproject.toml); only `app/worker` imports
this. FNV-1a is a byte-serial recurrence, so the hash itself is a Python loop: about 0.1 s
for a tile of 100k gaussians (1.2 MB of positions), which is why `carry` asks for it only
when a kind bound to positions could otherwise be kept, and stops at the first tile that
rules every such kind out.
"""

from __future__ import annotations

import gzip
import json
import struct
import zlib

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


def tile_positions(data: bytes) -> np.ndarray:
    """A tile's canonical positions from its GLB bytes (`rig_tiles.tile_positions`)."""
    positions = spz_positions(glb_spz(data)).astype("<f4")
    # SPZ's integer round trip has no negative zero; neither does the runtime's un-bake.
    positions[positions == 0.0] = 0.0
    return positions


def checksum_positions(positions: np.ndarray) -> str:
    """FNV-1a over the raw little-endian float32 bytes, plus the splat count
    (`synthetic_tree.checksum_positions`)."""
    flat = np.ascontiguousarray(positions, dtype="<f4").reshape(-1)
    digest = FNV_OFFSET
    for byte in flat.tobytes():
        digest = ((digest ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return f"fnv1a32:{flat.size // 3}:{digest:08x}"


def tile_checksum(data: bytes) -> str:
    """The position checksum a binding keys this tile (its GLB bytes) by."""
    return checksum_positions(tile_positions(data))
