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


def _merge(out_dir: Path, path: Path, moved: dict[str, np.ndarray] | None = None) -> None:
    """What splat-transform's merge writes: every leaf's gaussians in leaves.json's order,
    each moved by its offset -- or by `moved[name]` instead, for a leaf it got wrong."""
    leaves = json.loads((out_dir / "leaves.json").read_text(encoding="utf-8"))["leaves"]
    parts = []
    for row in leaves:
        columns = splat_tiles.unpack_spz((out_dir / "spz" / f"{row['name']}.spz").read_bytes())
        xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        offset = (moved or {}).get(row["name"], np.asarray(row["offset"], np.float64))
        parts.append(xyz + offset)
    xyz = np.concatenate(parts).astype(np.float32)
    n = len(xyz)
    scene = synthetic_far.scene()
    assert n == len(scene)
    save_ply(path, type(scene)(xyz, scene.rotations, scene.scales, scene.colours, scene.opacities))


def test_check_accepts_a_site_frame_merge_and_refuses_an_untranslated_one(
    tmp_path: Path,
) -> None:
    lod_leaves.fetch(str(FAR / "splat"), tmp_path)
    leaves = json.loads((tmp_path / "leaves.json").read_text(encoding="utf-8"))["leaves"]
    assert all(row["count"] == row["gaussians"] for row in leaves)
    right = tmp_path / "right.ply"
    _merge(tmp_path, right)
    report = lod_leaves.check(tmp_path, right)
    assert report["errorM"] <= lod_leaves.CHECK_TOLERANCE_M
    assert report["gaussians"] == sum(row["gaussians"] for row in leaves)
    # And when the translates are left off: each leaf where it is stored.
    wrong = tmp_path / "wrong.ply"
    _merge(tmp_path, wrong, {row["name"]: np.zeros(3) for row in leaves})
    with pytest.raises(SystemExit, match="wrong frame"):
        lod_leaves.check(tmp_path, wrong)


@pytest.mark.parametrize("leaf", ["splat_0-4-0", "splat_0-0-4", "splat_4-4-0-4"])
def test_check_refuses_one_leaf_merged_in_the_wrong_frame(tmp_path: Path, leaf: str) -> None:
    """One leaf given the opposite sign, or no translate at all, lands inside the site's
    overall box (a box of the whole merge passed it, errorM 0); its own run does not."""
    lod_leaves.fetch(str(FAR / "splat"), tmp_path)
    leaves = json.loads((tmp_path / "leaves.json").read_text(encoding="utf-8"))["leaves"]
    offset = np.asarray(next(r["offset"] for r in leaves if r["name"] == leaf), np.float64)
    assert offset.any()
    for moved in (-offset, np.zeros(3)):
        bad = tmp_path / "bad.ply"
        _merge(tmp_path, bad, {leaf: moved})
        with pytest.raises(SystemExit, match=leaf):
            lod_leaves.check(tmp_path, bad)


def test_check_refuses_a_merge_missing_a_leaf(tmp_path: Path) -> None:
    lod_leaves.fetch(str(FAR / "splat"), tmp_path)
    full = tmp_path / "full.ply"
    _merge(tmp_path, full)
    summary = json.loads((tmp_path / "leaves.json").read_text(encoding="utf-8"))
    summary["leaves"].append({**summary["leaves"][0], "name": "extra"})
    (tmp_path / "leaves.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing"):
        lod_leaves.check(tmp_path, full)
