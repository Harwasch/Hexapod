"""Stamp a motion rig with the identity of every tile of a splat tileset.

The Living Survey deformer refuses to move splats it cannot prove are the ones a rig was
built for. For a single-tile tileset that proof is ``rig.canonicalChecksum``: FNV-1a over the
tile's canonical positions (``checksumPositions`` in packages/world/src/rig.ts). A
level-of-detail tileset (``splat_tiles.convert`` with a tile budget) has no single array to
checksum -- leaves hold the capture's gaussians once, merged parents hold new ones, and the
viewer only ever has some of them loaded -- so the rig carries one checksum per tile instead:

    rig.tileChecksums = [checksum_positions(tile positions) for every tile content]

in the tile's own gaussian order, in the rig's local east-north-up frame, which is exactly
what the SPZ block stores: ``splat_tiles`` snaps every coordinate to the 1/4096 m grid and the
GLB's node matrix is the only axis swap. The runtime un-bakes each selected tile and looks its
digest up in this set (apps/web/src/cesium/splatTiles.ts).

Only identity is packaged. Which rig node a gaussian follows is *not* written here: the
runtime binds every gaussian, leaf or merged parent, to its nearest node from its own
position, once per tile load (docs/LIVING_SURVEY.md, "Multi-tile tilesets", has the costs
that decided it).

Usage:
    python rig_tiles.py tileset_dir rig.json [--out tileset_dir/rig.json]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from splat_tiles import unpack_spz
from synthetic_tree import checksum_positions


def glb_spz(path: Path) -> bytes:
    """The SPZ block of a GLB written by ``splat_tiles.build_glb``."""
    data = path.read_bytes()
    magic, _version, _total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} is not a GLB")
    json_length, _json_type = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20 : 20 + json_length])
    bin_offset = 20 + json_length + 8
    primitive = gltf["meshes"][0]["primitives"][0]
    view_index = primitive["extensions"]["KHR_gaussian_splatting"]["extensions"][
        "KHR_gaussian_splatting_compression_spz_2"
    ]["bufferView"]
    view = gltf["bufferViews"][view_index]
    start = bin_offset + view.get("byteOffset", 0)
    return data[start : start + view["byteLength"]]


def tile_positions(path: Path) -> np.ndarray:
    """A tile's canonical positions, ``(n, 3)`` float32, in its own order."""
    columns = unpack_spz(glb_spz(path))
    positions = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype("<f4")
    # SPZ's integer round trip has no negative zero; neither does the runtime's un-bake.
    positions[positions == 0.0] = 0.0
    return positions


def tile_uris(tileset: dict) -> list[str]:
    """Every content uri in the tileset, depth first from the root."""
    uris: list[str] = []

    def walk(tile: dict) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri:
            uris.append(uri)
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return uris


def tile_checksums(tileset_dir: Path) -> list[str]:
    """``checksum_positions`` of every tile's canonical positions, sorted and de-duplicated."""
    tileset = json.loads((tileset_dir / "tileset.json").read_text(encoding="utf-8"))
    return sorted(
        {checksum_positions(tile_positions(tileset_dir / uri)) for uri in tile_uris(tileset)}
    )


def stamp(rig: dict, tileset_dir: Path) -> dict:
    """The rig with ``tileChecksums`` for this tileset; nothing else changes."""
    return {**rig, "tileChecksums": tile_checksums(tileset_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tileset_dir", type=Path)
    parser.add_argument("rig", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rig = json.loads(args.rig.read_text(encoding="utf-8"))
    stamped = stamp(rig, args.tileset_dir)
    out = args.out or (args.tileset_dir / "rig.json")
    out.write_text(json.dumps(stamped, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"tiles": len(stamped["tileChecksums"]), "out": str(out)}))


if __name__ == "__main__":
    main()
