"""Every reader that binds by tile checksum, held to the 10.4 km fixture
(data/tiles/synthetic-far): its moved tiles are written in frames of their own, and each
reader must compose their transforms (docs/DECISIONS/0011-splat-tile-origins.md), so that
what it binds is keyed by expected.json's site-frame checksums. Each test fails when its
reader drops a tile's offset (review's mutations of 589947d0)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import synthetic_far

REPO_ROOT = Path(__file__).resolve().parents[3]
FAR = REPO_ROOT / "data" / "tiles" / "synthetic-far"
TILES = FAR / "splat"


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((FAR / "source" / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checksums(expected: dict) -> dict[str, str]:
    """Per uri, its site-frame checksum, and at least four of them moved tiles'."""
    out = {uri: row["checksum"] for uri, row in expected["tiles"].items()}
    assert sum(any(row["offset"]) for row in expected["tiles"].values()) >= 4
    return out


def _leaves(tileset: dict) -> set[str]:
    out: set[str] = set()
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        if not tile.get("children"):
            out.add(tile["content"]["uri"])
    return out


def _tileset() -> dict:
    return json.loads((TILES / "tileset.json").read_text(encoding="utf-8"))


def _instances(expected: dict, label: int = 1) -> dict:
    """An instances.json bound to the fixture: every gaussian of every tile is `label`."""
    tiles = {row["checksum"]: [label, row["gaussians"]] for row in expected["tiles"].values()}
    return {
        "format": "hexapod.instances",
        "version": 1,
        "instances": [
            {
                "id": label,
                "parent": None,
                "level": 0,
                "splats": 1,
                "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
                "tags": [{"label": "plot", "score": 1.0}],
                "category": "trees",
            }
        ],
        "tiles": tiles,
    }


def _site_leaves() -> tuple[np.ndarray, np.ndarray]:
    """The scene's gaussians (the leaves in the site frame) and an id for each: 2 within
    50 m of the north-east plot, else 1."""
    positions = synthetic_far.scene().positions.astype(np.float64)
    plot = np.asarray(synthetic_far.PLOTS["north-east"], np.float64)
    ids = np.where(np.linalg.norm(positions - plot, axis=1) < 50.0, 2, 1).astype(np.int64)
    assert (ids == 2).any()
    return positions, ids


# ---------------------------------------------------------------------- binding checks


def test_check_binding_accepts_site_frame_checksums_and_refuses_local_ones(
    expected: dict,
) -> None:
    """publish-instances.yml's check, which once called tile_positions with no offset."""
    import rebind_instances
    import splat_tiles
    from rig_tiles import glb_spz
    from synthetic_tree import checksum_positions

    rebind_instances.check_binding(TILES, _instances(expected))
    local = {}
    for uri, row in expected["tiles"].items():
        columns = splat_tiles.unpack_spz(glb_spz(TILES / uri))
        xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype("<f4")
        xyz[xyz == 0.0] = 0.0
        local[checksum_positions(xyz)] = [1, row["gaussians"]]
    with pytest.raises(ValueError, match="not in instances.json's binding"):
        rebind_instances.check_binding(TILES, {"tiles": local})
    short = _instances(expected)
    first = next(iter(short["tiles"]))
    short["tiles"][first] = [1, short["tiles"][first][1] - 1]
    with pytest.raises(ValueError, match="do not cover"):
        rebind_instances.check_binding(TILES, short)


def test_instances_report_checksums_are_the_site_frames(checksums: dict[str, str]) -> None:
    import instances_report

    assert instances_report.tile_checksums(TILES) == checksums


def test_rebind_keeps_site_frame_checksums_and_reads_far_leaves(
    expected: dict, checksums: dict[str, str]
) -> None:
    """Every leaf is instance 1 and every parent 0: rebinding takes each parent's ids from
    the leaves nearest it, which it can only do with every far tile in the site frame."""
    import rebind_instances

    leaves = _leaves(_tileset())
    tiles = {
        row["checksum"]: [1 if uri in leaves else 0, row["gaussians"]]
        for uri, row in expected["tiles"].items()
    }
    out = rebind_instances.rebind(TILES, tiles)
    assert sorted(out) == sorted(checksums.values())
    for uri, checksum in checksums.items():
        assert out[checksum] == [1, expected["tiles"][uri]["gaussians"]], uri


# ------------------------------------------------------------------------- segmentations


def test_segment_scene_binds_by_position_in_the_site_frame(
    expected: dict, checksums: dict[str, str]
) -> None:
    import rebind_instances
    import segment_scene

    positions, ids = _site_leaves()
    tiles = segment_scene.tile_binding_by_position(TILES, positions, ids)
    assert sorted(tiles) == sorted(checksums.values())
    for uri in expected["plotLeaves"]["north-east"]:
        assert (rebind_instances.decode_runs(tiles[checksums[uri]]) == 2).any(), uri
    for uri in _leaves(_tileset()):
        assert (rebind_instances.decode_runs(tiles[checksums[uri]]) > 0).all(), uri


def test_segment_ground_first_crop_binds_the_far_plot(
    expected: dict, checksums: dict[str, str]
) -> None:
    import rebind_instances
    import segment_ground_first

    positions, ids = _site_leaves()
    x, y, _ = synthetic_far.PLOTS["north-east"]
    box = (x - 60.0, y - 60.0, x + 60.0, y + 60.0)
    tiles = segment_ground_first.crop_binding(TILES, positions, ids, box)
    assert sorted(tiles) == sorted(checksums.values())
    for uri in expected["plotLeaves"]["north-east"]:
        assert (rebind_instances.decode_runs(tiles[checksums[uri]]) == 2).any(), uri


def test_scene_plants_binds_the_site_frame_checksums(
    tmp_path: Path, checksums: dict[str, str]
) -> None:
    import scene_plants
    import splat_tiles
    from splat_render import save_ply

    ply = tmp_path / "far.ply"
    save_ply(ply, synthetic_far.scene())
    labels = np.ones(len(synthetic_far.scene()), np.int64)
    document, by_uri = scene_plants.plant_binding(
        TILES,
        ply,
        labels,
        opacity_min=0.02,
        tile_gaussians=synthetic_far.TILE_GAUSSIANS,
    )
    assert by_uri == checksums
    assert sorted(document["tiles"]) == sorted(checksums.values())
    assert splat_tiles.tile_offsets(_tileset())  # the fixture really has moved tiles


# --------------------------------------------------------------------------------- skins


def test_skin_scene_reads_tiles_bound_by_site_frame_checksums(
    expected: dict, checksums: dict[str, str]
) -> None:
    import skin_scene

    tiles = skin_scene.read_tiles(TILES, _instances(expected))
    assert {t.uri: t.checksum for t in tiles} == checksums
    leaves = np.concatenate([t.positions for t in tiles if t.leaf]).astype(np.float64)
    box = expected["bboxSiteM"]
    assert leaves.min(axis=0).tolist() == box["min"]
    assert leaves.max(axis=0).tolist() == box["max"]


def test_skin_variants_reads_a_far_scan_whole_and_by_its_instances(
    tmp_path: Path, expected: dict, checksums: dict[str, str]
) -> None:
    import skin_variants

    scan = tmp_path / "scan"
    shutil.copytree(TILES, scan)
    whole, doc, _ = skin_variants.read_scan(scan, whole=True)
    assert {t.uri: t.checksum for t in whole} == checksums
    assert sorted(doc["tiles"]) == sorted(checksums.values())
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(_instances(expected)), encoding="utf-8")
    bound, _, _ = skin_variants.read_scan(scan, instances=instances)
    assert {t.uri: t.checksum for t in bound} == checksums


def test_teacher_materials_loads_a_far_scene_with_its_skins(tmp_path: Path, expected: dict) -> None:
    """The scene lands where synthetic_far put it, and a skin bound to a moved leaf's
    site-frame checksum owns that leaf's gaussians."""
    import skin_scene
    import teacher_materials

    moved = next(
        uri for uri in sorted(_leaves(_tileset())) if any(expected["tiles"][uri]["offset"])
    )
    n = expected["tiles"][moved]["gaussians"]
    skin_dir = tmp_path / "skin"
    skin_dir.mkdir()
    (skin_dir / "skin.json").write_text(
        json.dumps(
            {
                "skins": [{"id": 1, "instance": 7}],
                "tiles": {expected["tiles"][moved]["checksum"]: {"skins": [1, n], "row": 0}},
            }
        ),
        encoding="utf-8",
    )
    (skin_dir / "skin.bin").write_bytes(bytes(n * skin_scene.ROW_BYTES))
    scene = teacher_materials.load_scene(TILES, skin_dir)

    def rows(a: np.ndarray) -> np.ndarray:
        return a[np.lexsort(a.T[::-1])]

    truth = synthetic_far.scene().positions.astype(np.float64)
    loaded = scene.splats.positions.astype(np.float64)
    assert np.array_equal(rows(loaded), rows(truth))
    assert int((scene.owner == 7).sum()) == n
