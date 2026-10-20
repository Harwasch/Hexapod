"""lod_leaves.py: the streamed-LOD backfill's leaves, each put in the site frame before
splat-transform merges them (.github/workflows/streamed-lod-backfill.yml)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import lod_leaves
import splat_tiles
import synthetic_far
from rig_tiles import glb_spz
from splat_render import save_ply

REPO_ROOT = Path(__file__).resolve().parents[3]
FAR = REPO_ROOT / "data" / "tiles" / "synthetic-far"
LOD = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"


def _document(folder: Path) -> dict:
    return json.loads((folder / "tileset.json").read_text(encoding="utf-8"))


def test_plan_gives_far_leaves_a_translate_in_splat_transforms_frame() -> None:
    expected = json.loads((FAR / "source" / "expected.json").read_text(encoding="utf-8"))
    rows = lod_leaves.plan(_document(FAR / "splat"))
    assert [row["name"] for row in rows] == sorted(row["name"] for row in rows)
    moved = [row for row in rows if row["translate"]]
    assert len(moved) >= 4
    for row in rows:
        offset = expected["tiles"][row["uri"]]["offset"]
        assert row["offset"] == offset
        if not any(offset):
            assert row["translate"] is None
            continue
        # PLY space is a half turn about z of splat-transform's: x and y negate, z does not.
        x, y, z = (float(v) for v in row["translate"].split(","))
        assert [x, y, z] == [-offset[0] + 0.0, -offset[1] + 0.0, offset[2]]
    assert any(row["translate"] == "-1950,-901,0" for row in rows)


def test_a_tileset_that_fits_plans_no_translate() -> None:
    rows = lod_leaves.plan(_document(LOD))
    assert len(rows) > 5
    assert all(row["translate"] is None and not any(row["offset"]) for row in rows)


def test_fetch_from_a_directory_writes_every_leaf(tmp_path: Path) -> None:
    summary = lod_leaves.fetch(str(FAR / "splat"), tmp_path)
    rows = summary["leaves"]
    assert sorted(p.name for p in (tmp_path / "spz").iterdir()) == sorted(
        f"{row['name']}.spz" for row in rows
    )
    for row in rows:
        assert (tmp_path / "spz" / f"{row['name']}.spz").read_bytes() == glb_spz(
            FAR / "splat" / row["uri"]
        )
    assert json.loads((tmp_path / "leaves.json").read_text(encoding="utf-8")) == summary
    assert sum(row["gaussians"] for row in rows) == len(synthetic_far.scene())
    expected = json.loads((FAR / "source" / "expected.json").read_text(encoding="utf-8"))
    box = expected["bboxSiteM"]
    assert np.min([row["min"] for row in rows], axis=0).tolist() == box["min"]
    assert np.max([row["max"] for row in rows], axis=0).tolist() == box["max"]


def test_check_accepts_a_site_frame_merge_and_refuses_an_untranslated_one(
    tmp_path: Path,
) -> None:
    lod_leaves.fetch(str(FAR / "splat"), tmp_path)
    leaves = json.loads((tmp_path / "leaves.json").read_text(encoding="utf-8"))["leaves"]
    # What splat-transform's merge writes when every translate is right: the leaves' own
    # gaussians, in the site frame -- the scene itself.
    right = tmp_path / "right.ply"
    save_ply(right, synthetic_far.scene())
    report = lod_leaves.check(tmp_path, right)
    assert report["errorM"] <= lod_leaves.CHECK_TOLERANCE_M
    assert report["gaussians"] == sum(row["gaussians"] for row in leaves)
    # And when the translates are left off: each leaf where it is stored.
    parts = []
    for row in leaves:
        columns = splat_tiles.unpack_spz((tmp_path / "spz" / f"{row['name']}.spz").read_bytes())
        parts.append(np.stack([columns["x"], columns["y"], columns["z"]], axis=1))
    stored = np.concatenate(parts).astype(np.float32)
    scene = synthetic_far.scene()
    wrong = tmp_path / "wrong.ply"
    save_ply(
        wrong,
        type(scene)(stored, scene.rotations, scene.scales, scene.colours, scene.opacities),
    )
    with pytest.raises(SystemExit, match="wrong frame"):
        lod_leaves.check(tmp_path, wrong)
