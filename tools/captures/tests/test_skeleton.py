"""The skeleton extractor is the half of the Living Survey that meets a real capture.

Every other test in this repo checks a tree whose skeleton was known before a splat existed.
This one runs the extractor on that same tree and asks how much of the known answer it gets
back — the only place in the sprint where geometric extraction is scored rather than trusted.

The numbers asserted here are floors with headroom below what the extractor currently scores,
not targets. They exist so that a change which quietly makes recovery worse fails, and so that
the honest figure is written down next to the code that produces it rather than in a report
nobody re-runs.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import skeleton
import splat_tiles
import synthetic_tree

REPO_ROOT = Path(__file__).resolve().parents[3]
#: The committed 12,000-splat fixture, with ground truth beside it.
FIXTURE = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source"

#: Somewhere inside the Sheffield Park capture; the extractor needs a reference point to tile
#: against but nothing in these tests depends on where on the globe the tree lands.
LAT, LON = 28.0389, -82.6966


@pytest.fixture(scope="module")
def cloud() -> tuple[np.ndarray, np.ndarray]:
    """Positions and opacities of the fixture, as read_ply hands them over."""
    data = splat_tiles.read_ply(FIXTURE / "splat.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    return xyz, splat_tiles.sigmoid(data["opacity"])


@pytest.fixture(scope="module")
def truth() -> tuple[dict, dict]:
    return (
        json.loads((FIXTURE / "labels.json").read_text(encoding="utf-8")),
        json.loads((FIXTURE / "rig.json").read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="module")
def extracted(tmp_path_factory: pytest.TempPathFactory, truth: tuple[dict, dict]) -> dict:
    """One scored extraction run, shared by the tests that read its report."""
    out = tmp_path_factory.mktemp("skeleton")
    report = skeleton.extract(FIXTURE / "splat.ply", out, lat=LAT, lon=LON, truth=truth)
    report["out"] = out
    return report


def _rig(out: Path) -> dict:
    return json.loads((out / "source" / "rig.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- isolation


def test_isolation_keeps_almost_the_whole_tree(cloud: tuple[np.ndarray, np.ndarray]) -> None:
    """The fixture is one clean tree, so isolation should be nearly a no-op on it.

    This is the control for the noise filters: if they ever start eating structure rather than
    floaters, they eat it here first, where there is no floater to justify the loss.
    """
    xyz, opacity = cloud
    keep = skeleton.isolate_tree(xyz, opacity)
    assert keep.mean() > 0.95


def test_a_cylinder_picks_one_tree_out_of_two(cloud: tuple[np.ndarray, np.ndarray]) -> None:
    """Two copies of the fixture 20 m apart; the region must return exactly one of them."""
    xyz, opacity = cloud
    second = xyz + np.asarray([20.0, 0.0, 0.0])
    both = np.concatenate([xyz, second], axis=0)
    opacities = np.concatenate([opacity, opacity])
    keep = skeleton.isolate_tree(both, opacities, cylinder=(20.0, 0.0, 8.0))
    assert keep[: xyz.shape[0]].sum() == 0
    assert keep[xyz.shape[0] :].mean() > 0.95


def test_a_floater_is_dropped_and_does_not_stretch_the_tree(
    cloud: tuple[np.ndarray, np.ndarray],
) -> None:
    """One gaussian 40 m up is the case the density filter exists for.

    Without it the top band sits 40 m above the ground, every band below is empty, and the
    skeleton collapses to a single node — a failure that looks like a bad algorithm rather
    than like one bad splat.
    """
    xyz, opacity = cloud
    with_floater = np.concatenate([xyz, np.asarray([[0.0, 0.0, 40.0]])], axis=0)
    opacities = np.concatenate([opacity, [0.9]])
    keep = skeleton.isolate_tree(with_floater, opacities)
    assert not keep[-1]
    assert with_floater[keep, 2].max() < 10.0


def test_an_empty_region_refuses_rather_than_returning_a_rig(
    tmp_path: Path, cloud: tuple[np.ndarray, np.ndarray]
) -> None:
    del cloud
    with pytest.raises(SystemExit):
        skeleton.extract(
            FIXTURE / "splat.ply", tmp_path, lat=LAT, lon=LON, cylinder=(500.0, 500.0, 1.0)
        )


# ------------------------------------------------------------------------- rig structure


def test_extracted_rig_is_structurally_valid(extracted: dict) -> None:
    """The same checks validateRig() makes in packages/world/src/rig.ts."""
    rig = _rig(extracted["out"])
    assert skeleton.rig_issues(rig) == []
    assert rig["units"] == "meters"
    assert rig["nodes"][0]["parent"] == -1
    assert sum(1 for node in rig["nodes"] if node["parent"] == -1) == 1
    for i, node in enumerate(rig["nodes"][1:], start=1):
        assert node["parent"] < i


def test_the_rig_is_the_size_the_deformer_wants(extracted: dict) -> None:
    """A couple of hundred nodes: enough leaf clusters to rustle, few enough to assign cheaply.

    ``--max-nodes`` rose from 36 to 200 with the fixture's own node count, because a 36-node
    skeleton cannot express a crown of 108 leaf clusters and scoring it against one is scoring
    the wrong question.
    """
    rig = _rig(extracted["out"])
    assert 150 <= len(rig["nodes"]) <= 250


def test_the_trunk_is_a_chain_from_the_root(extracted: dict) -> None:
    """Trunk nodes must form one unbroken chain starting at the root.

    The chain is what the band labels mean. A trunk node hanging off a branch would give a
    twig the stiffness of a bole, and nothing downstream would notice.
    """
    rig = _rig(extracted["out"])
    trunk = [i for i, node in enumerate(rig["nodes"]) if node["band"] == "trunk"]
    assert trunk[0] == 0
    for i in trunk[1:]:
        assert rig["nodes"][i]["parent"] in trunk


def test_every_band_is_represented_and_leaves_are_terminal(extracted: dict) -> None:
    rig = _rig(extracted["out"])
    parents = {node["parent"] for node in rig["nodes"]}
    bands = {node["band"] for node in rig["nodes"]}
    assert bands == {"trunk", "branch", "leaf"}
    for i, node in enumerate(rig["nodes"]):
        if node["band"] == "leaf":
            assert i not in parents


def test_stiffness_falls_off_from_trunk_to_leaf(extracted: dict) -> None:
    rig = _rig(extracted["out"])
    by_band = {band: [] for band in ("trunk", "branch", "leaf")}
    for node in rig["nodes"]:
        by_band[node["band"]].append(node["stiffness"])
    assert min(by_band["trunk"]) > max(by_band["branch"])
    assert min(by_band["branch"]) > max(by_band["leaf"])


def test_source_note_says_what_it_was_extracted_from(extracted: dict) -> None:
    """That string is shown verbatim in the Inspector's Motion row, so it has to be true."""
    note = _rig(extracted["out"])["sourceNote"]
    assert "splat.ply" in note
    assert "skeleton.py" in note


# ------------------------------------------------------------- score against ground truth


def test_recovery_against_ground_truth(extracted: dict) -> None:
    """The honest number for geometric extraction, on the one tree where truth exists.

    Floors, with headroom below what the extractor scores today: ARI 0.459, purity 0.597,
    band 0.601, mean joint offset 0.140 m, 99 % of true joints within half a metre. None of
    these is close to 1.0 and none should be read as if it were — a 189-node skeleton inferred
    from a point cloud is not the 214-node skeleton the cloud was generated from, and where the
    canopy is dense the two genuinely disagree about which twig a leaf belongs to.

    Two of these figures moved a long way when the fixture became a real tree rather than a
    post with stubs, and in opposite directions. **Joint localisation got much better**: mean
    offset 0.14 m against 0.35 m, and 99 % of true joints recovered within half a metre against
    70 %, because the truth now has 214 joints spread through the crown instead of 33, so the
    extractor's own nodes have something near them to match. **Cluster agreement got worse**:
    ARI 0.459 against 0.672, because assigning a leaf to one of 108 clusters 20 cm apart is a
    far harder question than assigning it to one of 9 clusters a metre apart. The second number
    is the honest one to quote about a real capture, and it is the one that fell.

    The ceiling is not 1.0 either — see test_a_perfect_rig_scores_perfectly, which scores the
    true rig against itself at ARI 0.799 and purity 0.876. Against that ceiling the extractor
    recovers about seven tenths of what nearest-node assignment can express, where it recovered
    about eight tenths on the old fixture.
    """
    score = extracted["score"]
    assert score["adjustedRandIndex"] > 0.40
    assert score["bandAccuracy"] > 0.50
    assert score["purity"] > 0.50
    assert score["nodeOffsetMeanM"] < 0.25
    assert score["trueNodesWithin0_5m"] > 0.90
    assert score["isolationRetained"] > 0.95


def test_the_extracted_tree_is_the_right_size(extracted: dict) -> None:
    """5.9 m from the lowest bark splat to the top of the canopy, by construction."""
    assert 5.5 < extracted["height_m"] < 7.0


def test_scoring_refuses_labels_that_are_not_this_capture(
    tmp_path: Path, truth: tuple[dict, dict]
) -> None:
    labels, truth_rig = truth
    wrong = {**labels, "nodes": labels["nodes"][:10]}
    with pytest.raises(SystemExit):
        skeleton.extract(
            FIXTURE / "splat.ply", tmp_path, lat=LAT, lon=LON, truth=(wrong, truth_rig)
        )


def test_a_perfect_rig_scores_perfectly(truth: tuple[dict, dict]) -> None:
    """The scorer's own control: score the ground-truth rig against itself.

    Without this, every number above could be measuring a broken metric rather than a lossy
    extractor, and a floor that a bug makes easy to clear is worse than no floor.
    """
    labels, truth_rig = truth
    positions = np.frombuffer((FIXTURE / "positions.f32").read_bytes(), dtype="<f4")
    positions = positions.reshape(-1, 3).astype(np.float64)
    score = skeleton.score_against_truth(
        positions, np.asarray(labels["nodes"], dtype=np.int64), truth_rig, truth_rig
    )
    # Not 1.0, and that is the point. Purity comes out at 0.8762 — nearest-node assignment
    # against truth on the fixture's own splats, which is what the runtime does — because a
    # bark splat on the far side of a limb genuinely is nearer its neighbour's node, and three
    # leaf sleeves on one fork genuinely overlap. ARI 0.799 and band agreement 0.989 are the
    # matching ceilings. Every figure in test_recovery_against_ground_truth should be read
    # against these, not against 1.0.
    assert score["nodeOffsetMeanM"] == 0.0
    assert score["trueNodesWithin0_5m"] == 1.0
    assert score["purity"] > 0.85
    assert score["bandAccuracy"] > 0.95
    assert score["adjustedRandIndex"] > 0.75


# ---------------------------------------------------------------- density independence


def test_recovery_survives_a_quarter_density_cloud(
    cloud: tuple[np.ndarray, np.ndarray], truth: tuple[dict, dict]
) -> None:
    """Every radius is a multiple of the cloud's own spacing, so thinning must not break it.

    Fixed metre thresholds passed on the fixture they were tuned against and welded the whole
    canopy into one blob on a cloud four times denser. This is the test that caught it.
    """
    xyz, opacity = cloud
    labels, truth_rig = truth
    every_fourth = np.zeros(xyz.shape[0], dtype=bool)
    every_fourth[::4] = True
    keep = skeleton.isolate_tree(xyz[every_fourth], opacity[every_fourth])
    positions = xyz[every_fourth][keep]
    nodes = skeleton.extract_skeleton(positions)
    rig = skeleton.build_rig(nodes, "fnv1a32:0:00000000", "thinned")
    score = skeleton.score_against_truth(
        positions,
        np.asarray(labels["nodes"], dtype=np.int64)[every_fourth][keep],
        truth_rig,
        rig,
    )
    assert score["adjustedRandIndex"] > 0.30
    assert score["nodeOffsetMeanM"] < 0.8


# ---------------------------------------------------------------------------- the pipeline


def test_two_runs_produce_identical_bytes(tmp_path: Path) -> None:
    """Seedless and deterministic: every tie in clustering and pruning breaks on index."""
    first, second = tmp_path / "a", tmp_path / "b"
    skeleton.extract(FIXTURE / "splat.ply", first, lat=LAT, lon=LON)
    skeleton.extract(FIXTURE / "splat.ply", second, lat=LAT, lon=LON)
    for name in ("source/splat.ply", "source/rig.json", "splat/splat.glb", "splat/tileset.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_tileset_is_a_single_tile(extracted: dict) -> None:
    """The deformer refuses a multi-tile tileset by design, so the tiler must emit one node."""
    tileset = json.loads((extracted["out"] / "splat" / "tileset.json").read_text(encoding="utf-8"))
    assert "children" not in tileset["root"]
    assert tileset["root"]["content"]["uri"] == "splat.glb"


def _decode_glb_positions(path: Path) -> np.ndarray:
    """Positions back out of the tile the way the engine gets them: GLB -> SPZ -> fixed point."""
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    offset, chunks = 12, {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunks[kind] = raw[offset + 8 : offset + 8 + length]
        offset += 8 + length
    gltf = json.loads(chunks[0x4E4F534A])
    view = gltf["bufferViews"][0]
    spz = gzip.decompress(
        chunks[0x004E4942][view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    )
    _magic, _version, count, _flags, bits = struct.unpack_from("<IIIBB", spz, 0)
    packed = np.frombuffer(spz, dtype=np.uint8, count=count * 9, offset=16).reshape(count, 3, 3)
    fixed = (
        packed[:, :, 0].astype(np.int32)
        | (packed[:, :, 1].astype(np.int32) << 8)
        | (packed[:, :, 2].astype(np.int32) << 16)
    )
    fixed = np.where(fixed >= 1 << 23, fixed - (1 << 24), fixed)
    return (fixed / (1 << bits)).astype(np.float32)


def test_the_tile_decodes_to_exactly_the_checksummed_positions(extracted: dict) -> None:
    """The refusal contract, end to end: what the engine decodes is what the rig was built on.

    Positions are snapped to the SPZ grid before the isolated PLY is written precisely so that
    this holds for a real capture, whose coordinates are nothing like grid-aligned.
    """
    out = extracted["out"]
    decoded = _decode_glb_positions(out / "splat" / "splat.glb")
    canonical = np.frombuffer((out / "source" / "positions.f32").read_bytes(), dtype="<f4")
    assert np.array_equal(decoded, canonical.reshape(-1, 3))
    assert synthetic_tree.checksum_positions(decoded) == _rig(out)["canonicalChecksum"]
