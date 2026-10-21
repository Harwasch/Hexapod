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

    python rebind_instances.py TILES_DIR [--instances instances.json] [--out PATH] [--rekey]

`--rekey` follows a re-pack that kept every leaf but changed the parents (a new merge rule):
the parents' new checksums are bound from the leaves and the old keys dropped (`rekey`).

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


def _readers(
    positions_of: Callable[[Path], np.ndarray] | None,
    checksum_of: Callable[[np.ndarray], str] | None,
) -> tuple[Callable[[Path], np.ndarray], Callable[[np.ndarray], str]]:
    """The given readers, or the real ones: positions tile-local by contract (the caller
    adds each tile's offset, `site_positions`, before it checksums) and the checksum every
    binding is keyed by."""
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    return (
        positions_of or (lambda path: tile_positions(path, None)),
        checksum_of or checksum_positions,
    )


def rebind(
    tiles_dir: Path,
    tiles: dict[str, list[int]],
    *,
    positions_of: Callable[[Path], np.ndarray] | None = None,
    checksum_of: Callable[[np.ndarray], str] | None = None,
    neighbours: int = NEIGHBOURS,
) -> dict[str, list[int]]:
    """`tiles` (checksum to runs) with every non-leaf tile's ids taken from the leaf splats
    nearest to its splats. Leaf tiles, and tiles `tiles` does not list, are left as they are.

    `positions_of` returns a tile's positions as its SPZ stores them (tile-local); each is
    moved into the site frame here (`splat_tiles.site_positions`), which is what both the
    checksums and the nearest-leaf search are over."""
    from splat_tiles import site_positions, tile_offsets

    positions_of, checksum_of = _readers(positions_of, checksum_of)
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = tile_offsets(tileset)
    entries = []
    for uri, leaf in tile_tree(tileset):
        at = site_positions(positions_of(tiles_dir / uri), offsets.get(uri))
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


def rekey(
    tiles_dir: Path,
    tiles: dict[str, list[int]],
    *,
    positions_of: Callable[[Path], np.ndarray] | None = None,
    checksum_of: Callable[[np.ndarray], str] | None = None,
    neighbours: int = NEIGHBOURS,
) -> dict[str, list[int]]:
    """`tiles` re-keyed to a re-pack of the same scan that kept every leaf: each leaf tile's
    checksum must already be a key, with runs covering its gaussians (else ValueError); a
    parent whose checksum is new is added, then bound as `rebind` binds it, from the leaf
    splats nearest to its splats; and every key that is no tile's checksum is dropped.

    Checksums are over site-frame positions, each tile's transform composed exactly as
    `rebind` composes it (`splat_tiles.tile_offsets`, `site_positions`)."""
    from splat_tiles import site_positions, tile_offsets

    positions_of, checksum_of = _readers(positions_of, checksum_of)
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = tile_offsets(tileset)
    seen: list[tuple[str, int, bool]] = []
    for uri, leaf in tile_tree(tileset):
        at = site_positions(positions_of(tiles_dir / uri), offsets.get(uri))
        checksum = checksum_of(at)
        if leaf:
            runs = tiles.get(checksum)
            if runs is None or int(np.sum(np.asarray(runs, np.int64)[1::2])) != len(at):
                raise ValueError(f"rekey keeps leaves: tile {uri} is not in the instances")
        seen.append((checksum, len(at), leaf))
    seeded = dict(tiles)
    for checksum, count, leaf in seen:
        if not leaf and checksum not in seeded:
            seeded[checksum] = [0, count]
    bound = rebind(
        tiles_dir, seeded, positions_of=positions_of, checksum_of=checksum_of, neighbours=neighbours
    )
    live = {checksum for checksum, _, _ in seen}
    return {key: runs for key, runs in bound.items() if key in live}


def check_binding(tiles_dir: Path, document: dict) -> None:
    """Refuse an `instances.json` that is not bound to the tiles in `tiles_dir` (a tileset
    as published: tileset.json and every tile it names). Every tile's checksum, over its
    site-frame positions (each tile's transform composed, `splat_tiles.tile_offsets`), must
    be a key of `tiles`, and its runs must cover exactly the tile's gaussians.
    publish-instances.yml runs this before it stages a segmentation, so a run made against
    other tiles never reaches the asset. Raises ValueError.
    """
    from rig_tiles import tile_positions, tile_uris
    from splat_tiles import tile_offsets
    from synthetic_tree import checksum_positions

    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = tile_offsets(tileset)
    binding = document["tiles"]
    uris = tile_uris(tileset)
    for uri in uris:
        positions = tile_positions(tiles_dir / uri, offsets.get(uri))
        runs = binding.get(checksum_positions(positions))
        if runs is None:
            raise ValueError(f"{uri} is not in instances.json's binding")
        if int(np.sum(np.asarray(runs, np.int64)[1::2])) != positions.shape[0]:
            raise ValueError(f"{uri}'s runs do not cover its gaussians")
    if len(binding) < len(set(uris)):
        raise ValueError("the binding does not match the tiles")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tiles", type=Path, help="the scan's tileset directory")
    parser.add_argument("--instances", type=Path, help="default: TILES/instances.json")
    parser.add_argument("--out", type=Path, help="default: overwrite --instances")
    parser.add_argument(
        "--rekey",
        action="store_true",
        help="follow a re-pack that kept every leaf: bind new parent tiles, drop stale keys",
    )
    args = parser.parse_args(argv)
    source = args.instances or args.tiles / "instances.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    before = _zero_share(document["tiles"])
    document["tiles"] = (rekey if args.rekey else rebind)(args.tiles, document["tiles"])
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
