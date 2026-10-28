"""Bind CPU graph pieces to the unchanged round-3 player; output only lab sidecars.

The clip is a diagnostic of mechanically independent fragments, not verified leaves.
Display LOD bindings use nearest native-detail splat within 5 cm; analysis never uses
LOD splats. The source scan and round-3 files are read-only inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from living_articulation import segment_binding, tile_data
from living_structure import leaf_uris, nearest_segments, piece_table
from rig_tiles import tile_uris
from splat_tiles import tile_offsets
from synthetic_tree import checksum_positions


def prepare(source: Path, probe: Path, baseline: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    scan = dict(np.load(probe / "scan.npz"))
    derived = dict(np.load(probe / "derived.npz"))
    crown = derived["crown"]
    labels = np.load(probe / "labels-r0.025-c0.12-n0.8-seed17.npy")
    table = piece_table(
        scan["points"][crown], scan["scales"][crown], labels, derived["skeleton_distance"][crown]
    )
    count, size = table["count"], table["diameter"]
    viable = (count >= 8) & (size >= 0.03) & (size <= 0.30)
    identifiers = np.flatnonzero(viable)
    mapping = np.zeros(len(count), dtype=np.uint32)
    mapping[identifiers] = np.arange(len(identifiers)) + 1
    metadata = json.loads((baseline / "articulation.json").read_text())
    nodes = metadata["nodes"]
    _, branch = nearest_segments(table["centre"], nodes)
    leaves = np.zeros((len(identifiers), 9), dtype="<f4")
    rng = np.random.default_rng(891)
    leaves[:, :3] = table["centre"][identifiers]
    axis = rng.normal(size=(len(leaves), 3))
    leaves[:, 3:6] = axis / np.linalg.norm(axis, axis=1)[:, None]
    leaves[:, 6] = rng.integers(20, 61, len(leaves)) / 10
    leaves[:, 7] = rng.uniform(0, 2 * np.pi, len(leaves))
    leaves[:, 8] = rng.uniform(0.035, 0.10, len(leaves))
    leaves.tofile(out / "leaves.bin")
    shutil.copyfile(baseline / "motion.bin", out / "motion.bin")
    manifest = json.loads((source / "splat/tileset.json").read_text())
    offsets = tile_offsets(manifest)

    def prune(tile, depth=0):
        if depth >= 2:
            tile.pop("children", None)
        tile["geometricError"] = 1000 if tile.get("children") else 0
        for child in tile.get("children", []):
            prune(child, depth + 1)

    prune(manifest["root"])
    manifest["root"].setdefault("extras", {})["nativeLod"] = False
    native = cKDTree(scan["points"][crown])
    tiles, offset, matched, total = {}, 0, 0, 0
    with (out / "bindings.bin").open("wb") as stream:
        for uri in tile_uris(manifest):
            _, points = tile_data(source / "splat", uri, offsets)
            binding = segment_binding(points, nodes)
            d, nearest = native.query(points, workers=4)
            own = labels[nearest]
            use = (d <= 0.05) & (points[:, 2] >= 2) & viable[own]
            binding[use, 0] = branch[own[use]] + 1
            binding[use, 1] = branch[own[use]] + 1
            binding[use, 2] = 65535
            binding[use, 3] = mapping[own[use]]
            tiles[checksum_positions(points)] = {"offset": offset, "count": len(points)}
            stream.write(binding.astype("<u4").tobytes())
            offset += binding.nbytes
            matched += int(use.sum())
            total += len(points)
    metadata.update(
        {
            "leafCount": len(leaves),
            "tiles": tiles,
            "sourceNote": "CPU graph components, biological identity unverified; simulated motion",
            "metrics": {
                "displayLodDepth": 2,
                "displayLodPieceSplatsAcrossLods": matched,
                "displaySplatsAcrossLods": total,
                "nativePieces": len(leaves),
            },
        }
    )
    (out / "articulation.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def rewrite(tile):
        if "content" in tile:
            tile["content"]["uri"] = "/scan/" + tile["content"]["uri"]
        for child in tile.get("children", []):
            rewrite(child)

    selected = leaf_uris(manifest)
    rewrite(manifest["root"])
    (out / "tileset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "playback.json").write_text(
        json.dumps({"selectedTiles": selected, "metadata": metadata["metrics"]}, indent=2) + "\n"
    )
    print(json.dumps(metadata["metrics"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "probe", "baseline", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.probe, args.baseline, args.out)
