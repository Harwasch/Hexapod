"""Coarse tiles take their ids from the leaf splats nearest to their splats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
