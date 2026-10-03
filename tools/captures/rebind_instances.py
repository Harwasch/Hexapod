"""Object ids for the coarse tiles of a segmented scan (docs/SCENE_OBJECTS.md §4, `tiles`).

The segmentation gives every leaf splat an instance id, exactly. A coarse tile's splats are
merged from many leaf splats, and the first binding rule gave a merged splat an id only when
all of them shared one. Leaf instances are small (an object's parts), so few merged splats
qualified: on the published camp 46% of the non-leaf splats of the fourth level of detail and
22% of the fifth carried id 0, which nothing hides or highlights (32% and 11% after this).

Here a merged splat takes the id most of the leaf splats nearest to it carry (`NEIGHBOURS` of
them; ties to the smaller id, so to 0 where unlabelled leaves are as many), and a splat that
sits exactly on a leaf splat takes that leaf's id. Leaf tiles keep their ids. It needs only
the published package -- the tiles and instances.json -- so a scan already published is fixed
by rewriting its instances.json:

    python rebind_instances.py TILES_DIR [--instances instances.json] [--out PATH]

`segment_scene.py` applies the same step to what it binds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

#: Leaf splats a merged splat consults.
NEIGHBOURS = 8
TILES_ENCODING = (
    "rle; a leaf splat carries its instance; a merged splat the instance most of the "
    f"{NEIGHBOURS} leaf splats nearest to it carry"
)


def decode_runs(runs: Sequence[int]) -> np.ndarray:
    """`[id, count, ...]` to one id per splat."""
    pairs = np.asarray(runs, np.int64).reshape(-1, 2)
    return np.repeat(pairs[:, 0], pairs[:, 1])


def encode_runs(values: np.ndarray) -> list[int]:
    """One id per splat to `[id, count, ...]`."""
    values = np.asarray(values, np.int64)
    if values.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    counts = np.diff(np.r_[starts, values.size])
    return np.stack([values[starts], counts], 1).reshape(-1).tolist()


def tile_tree(tileset: dict) -> list[tuple[str, bool]]:
    """Every content uri, depth first, and whether it is a leaf (no content below it)."""
    out: list[tuple[str, bool]] = []

    def has_content(tile: dict) -> bool:
        return bool(tile.get("content", {}).get("uri")) or any(
            has_content(c) for c in tile.get("children", [])
        )

    def walk(tile: dict) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri:
            out.append((uri, not any(has_content(c) for c in tile.get("children", []))))
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return out


def plurality(near: np.ndarray) -> np.ndarray:
    """Per row, the value most entries hold (ties to the smaller)."""
    ordered = np.sort(near, axis=1)
    k = ordered.shape[1]
    best = ordered[:, 0].copy()
    best_count = np.zeros(len(ordered), np.int64)
    run = np.ones(len(ordered), np.int64)
    for j in range(1, k + 1):
        ended = np.ones(len(ordered), bool) if j == k else ordered[:, j] != ordered[:, j - 1]
        better = ended & (run > best_count)
        best = np.where(better, ordered[:, j - 1], best)
        best_count = np.where(better, run, best_count)
        if j < k:
            run = np.where(ended, 1, run + 1)
    return best


def rebind(
    tiles_dir: Path,
    tiles: dict[str, list[int]],
    *,
    positions_of: Callable[[Path], np.ndarray] | None = None,
    checksum_of: Callable[[np.ndarray], str] | None = None,
    neighbours: int = NEIGHBOURS,
) -> dict[str, list[int]]:
    """`tiles` (checksum to runs) with every non-leaf tile's ids taken from the leaf splats
    nearest to its splats. Leaf tiles, and tiles `tiles` does not list, are left as they are."""
    if positions_of is None or checksum_of is None:
        from rig_tiles import tile_positions
        from synthetic_tree import checksum_positions

        positions_of = positions_of or tile_positions
        checksum_of = checksum_of or checksum_positions
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    entries = []
    for uri, leaf in tile_tree(tileset):
        at = positions_of(tiles_dir / uri)
        entries.append((checksum_of(at), at, leaf))
    leaf_positions, leaf_ids = [], []
    for checksum, at, leaf in entries:
        runs = tiles.get(checksum)
        if leaf and runs is not None:
            ids = decode_runs(runs)
            if ids.size == len(at):
                leaf_positions.append(at)
                leaf_ids.append(ids)
    out = dict(tiles)
    if not leaf_positions:
        return out
    tree = cKDTree(np.concatenate(leaf_positions).astype(np.float64))
    ids = np.concatenate(leaf_ids)
    for checksum, at, leaf in entries:
        if leaf or checksum not in tiles:
            continue
        distance, index = tree.query(at.astype(np.float64), k=neighbours)
        near = ids[np.atleast_2d(index)]
        labels = np.where(np.atleast_2d(distance)[:, 0] == 0.0, near[:, 0], plurality(near))
        out[checksum] = encode_runs(labels)
    return dict(sorted(out.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tiles", type=Path, help="the scan's tileset directory")
    parser.add_argument("--instances", type=Path, help="default: TILES/instances.json")
    parser.add_argument("--out", type=Path, help="default: overwrite --instances")
    args = parser.parse_args(argv)
    source = args.instances or args.tiles / "instances.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    before = _zero_share(document["tiles"])
    document["tiles"] = rebind(args.tiles, document["tiles"])
    document["tilesEncoding"] = TILES_ENCODING
    (args.out or source).write_text(
        json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(
        f"splats without an instance: {before:.1%} before, {_zero_share(document['tiles']):.1%} after"
    )
    return 0


def _zero_share(tiles: dict[str, list[int]]) -> float:
    total = zero = 0
    for runs in tiles.values():
        pairs = np.asarray(runs, np.int64).reshape(-1, 2)
        total += int(pairs[:, 1].sum())
        zero += int(pairs[pairs[:, 0] == 0, 1].sum())
    return zero / max(total, 1)


if __name__ == "__main__":
    raise SystemExit(main())
