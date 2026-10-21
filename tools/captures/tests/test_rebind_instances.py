"""Coarse tiles take their ids from the leaf splats nearest to their splats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rebind_instances as ri


def test_runs_round_trip() -> None:
    values = np.array([3, 3, 0, 5, 5, 5])
    runs = ri.encode_runs(values)
    assert runs == [3, 2, 0, 1, 5, 3]
    assert ri.decode_runs(runs).tolist() == values.tolist()
    assert ri.encode_runs(np.array([], np.int64)) == []


def test_plurality_takes_the_most_common_value_ties_to_the_smaller() -> None:
    near = np.array([[4, 4, 1, 2], [7, 0, 7, 0], [9, 9, 9, 9], [3, 1, 2, 5]])
    assert ri.plurality(near).tolist() == [4, 0, 9, 1]


def test_tile_tree_marks_leaves() -> None:
    tileset = {
        "root": {
            "content": {"uri": "root.glb"},
            "children": [
                {"content": {"uri": "a.glb"}, "children": [{"content": {"uri": "a0.glb"}}]},
                {"content": {"uri": "b.glb"}, "children": [{"children": []}]},
            ],
        }
    }
    assert ri.tile_tree(tileset) == [
        ("root.glb", False),
        ("a.glb", False),
        ("a0.glb", True),
        ("b.glb", True),
    ]


def test_a_coarse_tile_takes_what_most_nearby_leaf_splats_are(tmp_path: Path) -> None:
    # Two leaf tiles: object 1 on the left (x < 0), object 2 and some unlabelled splats on the
    # right; a coarse tile over both whose merged splats no single leaf id covers.
    left = np.array([[-3 + 0.1 * i, 0, 0] for i in range(10)], np.float32)
    right = np.array([[3 + 0.1 * i, 0, 0] for i in range(10)], np.float32)
    coarse = np.array([[-2.5, 0, 0], [3.4, 0, 0], [-3.0, 0, 0], [50, 0, 0]], np.float32)
    files = {"coarse.glb": coarse, "left.glb": left, "right.glb": right}
    (tmp_path / "tileset.json").write_text(
        json.dumps(
            {
                "root": {
                    "content": {"uri": "coarse.glb"},
                    "children": [
                        {"content": {"uri": "left.glb"}},
                        {"content": {"uri": "right.glb"}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    tiles = {
        "coarse": [0, 4],
        "left": [1, 10],
        "right": [2, 7, 0, 3],
        "other": [9, 1],
    }
    out = ri.rebind(
        tmp_path,
        tiles,
        positions_of=lambda path: files[path.name],
        checksum_of=lambda at: next(k for k, v in files.items() if v is at).removesuffix(".glb"),
        neighbours=4,
    )
    # Left: object 1; right: object 2; on a leaf splat: its id; far off past the unlabelled
    # end of the right tile: what its nearest leaf splats mostly are, nothing.
    assert ri.decode_runs(out["coarse"]).tolist() == [1, 2, 1, 0]
    assert out["left"] == tiles["left"] and out["right"] == tiles["right"]
    assert out["other"] == tiles["other"]


def _two_leaf_tileset(tmp_path: Path) -> dict[str, np.ndarray]:
    left = np.array([[-3 + 0.1 * i, 0, 0] for i in range(10)], np.float32)
    right = np.array([[3 + 0.1 * i, 0, 0] for i in range(10)], np.float32)
    (tmp_path / "tileset.json").write_text(
        json.dumps(
            {
                "root": {
                    "content": {"uri": "coarse.glb"},
                    "children": [
                        {"content": {"uri": "left.glb"}},
                        {"content": {"uri": "right.glb"}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return {"left.glb": left, "right.glb": right}


def test_rekey_follows_a_repack_that_changed_only_the_parents(tmp_path: Path) -> None:
    files = _two_leaf_tileset(tmp_path)
    # The re-pack moved the parent's splats (a new merge), so its checksum is new.
    files["coarse.glb"] = np.array([[-2.5, 0, 0], [3.4, 0, 0], [-3.0, 0, 0]], np.float32)
    names = {"coarse.glb": "new-coarse", "left.glb": "left", "right.glb": "right"}

    def checksum_of(at: np.ndarray) -> str:
        return next(names[k] for k, v in files.items() if v is at)

    old = {"old-coarse": [0, 4], "left": [1, 10], "right": [2, 7, 0, 3]}
    out = ri.rekey(
        tmp_path,
        old,
        positions_of=lambda path: files[path.name],
        checksum_of=checksum_of,
        neighbours=4,
    )
    assert "old-coarse" not in out
    assert ri.decode_runs(out["new-coarse"]).tolist() == [1, 2, 1]
    assert out["left"] == old["left"] and out["right"] == old["right"]
    assert sorted(out) == ["left", "new-coarse", "right"]
    # A leaf the instances do not hold (or hold with the wrong count) is not a re-pack.
    with pytest.raises(ValueError, match="rekey keeps leaves: tile right.glb"):
        ri.rekey(
            tmp_path,
            {"left": [1, 10]},
            positions_of=lambda path: files[path.name],
            checksum_of=checksum_of,
        )
    with pytest.raises(ValueError, match="rekey keeps leaves: tile left.glb"):
        ri.rekey(
            tmp_path,
            {"left": [1, 9], "right": [2, 10]},
            positions_of=lambda path: files[path.name],
            checksum_of=checksum_of,
        )


def test_rekey_keys_a_far_tileset_in_the_site_frame() -> None:
    """synthetic-far's tiles carry their own origins. Given instances for its leaves alone,
    keyed in the site frame as every reader keys them, rekey binds every parent and the
    result is a binding `check_binding` accepts -- which local-frame keys would not be."""
    import splat_tiles
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    far = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-far" / "splat"
    tileset = json.loads((far / "tileset.json").read_text(encoding="utf-8"))
    offsets = splat_tiles.tile_offsets(tileset)
    assert any(np.any(o) for o in offsets.values())
    leaves: dict[str, list[int]] = {}
    for index, (uri, leaf) in enumerate(ri.tile_tree(tileset)):
        if leaf:
            positions = tile_positions(far / uri, offsets.get(uri))
            leaves[checksum_positions(positions)] = [index + 1, int(positions.shape[0])]
    out = ri.rekey(far, leaves)
    ri.check_binding(far, {"tiles": out})
    assert len(out) == len(ri.tile_tree(tileset))
    assert all(out[key] == runs for key, runs in leaves.items())
