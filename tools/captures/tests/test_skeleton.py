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


# ------------------------------------------------------------------- the woody radius

#: The motion model, transcribed from packages/world/src/modes.ts and flutter.ts. Duplicated
#: for the same reason ``rig_issues`` duplicates ``validateRig``: the consequence of a radius
#: is a frequency, and a frequency distribution that only goes wrong in the browser goes wrong
#: after the rig has been committed and served. If either side changes, this has to follow.
FREQ_SCALE_HZ, MIN_NODE_HZ, MAX_NODE_HZ, MIN_LENGTH_M = 450.0, 0.25, 30.0, 0.15
FLUTTER_RADIUS_M, FLUTTER_SHAPE_MIN = 0.015, 0.05


def _natural_hz(nodes: list[dict]) -> np.ndarray:
    """Natural frequency of every node of a rig, hertz — ``nodeNaturalHz`` over ``tipReaches``."""
    reach = np.zeros(len(nodes))
    for i in range(len(nodes) - 1, 0, -1):
        parent = nodes[i]["parent"]
        if parent < 0:
            continue
        step = float(
            np.linalg.norm(np.asarray(nodes[i]["position"]) - np.asarray(nodes[parent]["position"]))
        )
        reach[parent] = max(reach[parent], step + reach[i])
    out = []
    for i, node in enumerate(nodes):
        parent = node["parent"]
        segment = (
            0.0
            if parent < 0
            else float(
                np.linalg.norm(np.asarray(node["position"]) - np.asarray(nodes[parent]["position"]))
            )
        )
        length = max(reach[i], segment, MIN_LENGTH_M)
        radius = node["radius"] if node["radius"] > 0 else 0.01
        out.append(min(max(FREQ_SCALE_HZ * radius / (length * length), MIN_NODE_HZ), MAX_NODE_HZ))
    return np.asarray(out)


def _flutters(nodes: list[dict]) -> np.ndarray:
    """Which nodes' splats shimmer at all — ``flutterShape`` above its exact-zero cut."""
    radius = np.asarray([node["radius"] for node in nodes])
    return np.exp(-radius / FLUTTER_RADIUS_M) >= FLUTTER_SHAPE_MIN


def _cylinder(radius: float, length: float, count: int, seed: int) -> np.ndarray:
    """Splats on the *surface* of a vertical tube: what a limb looks like in a splat cloud."""
    rng = np.random.default_rng(seed)
    theta = rng.random(count) * 2 * np.pi
    # A few per cent out of round, the way bark is.
    r = radius * (0.94 + 0.12 * rng.random(count))
    return np.stack([r * np.cos(theta), r * np.sin(theta), rng.random(count) * length], axis=1)


def _halo(girth: float, length: float, count: int, seed: int) -> np.ndarray:
    """Foliage: a diffuse cloud filling the volume around the same axis, not a surface on it."""
    rng = np.random.default_rng(seed)
    theta = rng.random(count) * 2 * np.pi
    r = girth * np.sqrt(rng.random(count))
    return np.stack([r * np.cos(theta), r * np.sin(theta), rng.random(count) * length], axis=1)


UP = np.asarray([0.0, 0.0, 1.0])


def test_woody_radius_finds_a_twig_inside_its_own_foliage() -> None:
    """The case the old estimator got wrong: a thin stem wearing a blob of leaves.

    A 3 cm tube carrying a 30 cm halo of four times as many points. The foliage extent is the
    halo — correctly, that is what it measures — and the woody radius is the tube.
    """
    points = np.concatenate(
        [_cylinder(0.03, 1.0, 120, seed=1), _halo(0.30, 1.0, 480, seed=2)], axis=0
    )
    centre = points.mean(axis=0)
    assert skeleton.foliage_extent(points, centre) > 0.15
    recovered = skeleton.woody_radius(points, centre, UP, spacing=0.02)
    assert 0.021 < recovered < 0.045


def test_woody_radius_works_from_a_bole_to_a_twig_without_being_told_which() -> None:
    """One estimator, no per-case tuning, over a 30-fold range of thickness.

    The loose end is the 5 mm tube: at a 2 cm point spacing its circumference carries about
    one and a half points, so it is below the resolution limit and comes back *at* the limit
    (0.0095 m) rather than at its true value. That is the declared behaviour, not a near miss.
    """
    for true_radius in (0.005, 0.02, 0.05, 0.15):
        points = np.concatenate(
            [
                _cylinder(true_radius, 1.0, 200, seed=3),
                _halo(max(4 * true_radius, 0.2), 1.0, 400, seed=4),
            ],
            axis=0,
        )
        recovered = skeleton.woody_radius(points, points.mean(axis=0), UP, spacing=0.02)
        limit = skeleton.RING_POINTS * 0.02 / (2 * np.pi)
        assert recovered >= limit
        if true_radius >= limit:
            assert 0.6 * true_radius < recovered < 1.7 * true_radius, true_radius
        else:
            assert recovered == pytest.approx(limit)


def test_woody_radius_refuses_a_cloud_with_no_limb_in_it() -> None:
    """A ball of foliage and nothing else has no woody radius, and saying so beats inventing one.

    It comes back at the resolution limit — an upper bound on a limb too thin for this cloud to
    show — rather than at the size of the ball, which is what put 123 of 189 fixture nodes on
    the 30 Hz clamp.
    """
    points = _halo(0.4, 0.8, 500, seed=5)
    limit = skeleton.RING_POINTS * 0.02 / (2 * np.pi)
    assert skeleton.woody_radius(points, points.mean(axis=0), UP, spacing=0.02) == pytest.approx(
        limit
    )


def test_the_axis_matters_for_a_limb_that_is_not_vertical() -> None:
    """Measured across the wrong axis, a limb's length leaks into its thickness.

    This is half of why the old estimator was wrong: it took a *horizontal* spread, so a band
    cutting a leaning branch returned the branch's run through the band.
    """
    axis = np.asarray([1.0, 0.0, 0.3])
    axis = axis / np.linalg.norm(axis)
    upright = _cylinder(0.04, 2.0, 300, seed=6)
    rotated = upright @ np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )  # lay the tube along +X
    centre = rotated.mean(axis=0)
    along = skeleton.woody_radius(rotated, centre, np.asarray([1.0, 0.0, 0.0]), spacing=0.01)
    across = skeleton.woody_radius(rotated, centre, UP, spacing=0.01)
    assert 0.03 < along < 0.055
    assert across > 5 * along


def test_extracted_radii_are_woody_rather_than_foliage(extracted: dict) -> None:
    """The fixture's true radii run 8.8 mm to 15 cm. The extracted ones must live in that range.

    The old estimator returned 2.7 cm to 45 cm — the whole distribution sat above the true one,
    and the top of it was three times the trunk. The floor here is the cloud's own resolution
    limit (12.5 mm at this density), which is why the minimum is not 8.8 mm.
    """
    rig = _rig(extracted["out"])
    radii = np.asarray([node["radius"] for node in rig["nodes"]])
    assert radii.min() >= 0.010
    assert radii.max() < 0.45  # loose: this is the whole crown's worth of spurious width gone
    assert np.median(radii) < 0.02
    # The foliage extent is still measured, still much bigger, and reported separately.
    assert extracted["foliage_extent_median_m"] > 4 * extracted["radius_median_m"]
    assert extracted["radius_resolution_m"] == pytest.approx(
        skeleton.RING_POINTS * extracted["spacing_m"] / (2 * np.pi), abs=1e-4
    )


def test_the_crown_is_off_the_frequency_clamp_and_flutters(extracted: dict, truth) -> None:
    """The measured consequence, before and after, on the tree whose answer is known.

    With the old estimator: median node at the 30 Hz clamp, 123 of 189 nodes pinned there, 9
    fluttering. With this one: median 19.1 Hz, 81 pinned, 158 fluttering. The true rig, for
    scale, is 9.1 Hz with 9 of 214 pinned and 204 fluttering; the same 189-node skeleton
    carrying the *true* radii is 11.2 Hz with 58 pinned and 173 fluttering, which is the
    ceiling this skeleton can reach and what the bounds below are set against.

    The bounds are wide on purpose. They are placed where a regression to the old behaviour —
    a crown that is an order of magnitude too stiff to move — would trip them, not where this
    run's arithmetic happens to land.
    """
    del truth
    nodes = _rig(extracted["out"])["nodes"]
    hz = _natural_hz(nodes)
    clamped = int((hz >= MAX_NODE_HZ - 1e-9).sum())
    fluttering = int(_flutters(nodes).sum())
    assert 5.0 < float(np.median(hz)) < 25.0
    assert clamped < 0.55 * len(nodes)
    assert fluttering > 0.7 * len(nodes)


#: Trees for the generalisation check, all built in memory and none of them the fixture. The
#: point is that they differ in the things a radius estimator could secretly depend on: overall
#: size, absolute limb thickness, how many limbs, how much foliage and how far it hangs, and
#: point spacing. ``TWIG_RADIUS_M`` and the foliage constants are module-level knobs of the
#: generator, so they are patched rather than passed.
GENERAL_TREES: dict[str, dict] = {
    "bushy 7 m, 30k splats": {
        "rig": {
            "height_m": 7.0,
            "whorls": 6,
            "branches_per_whorl": 4,
            "secondaries_per_branch": 2,
            "twigs_per_secondary": 4,
        },
        "splats": 30000,
        "seed": 5,
    },
    "spare 5 m, thick twigs, 6k splats": {
        "rig": {
            "height_m": 5.0,
            "whorls": 2,
            "branches_per_whorl": 5,
            "secondaries_per_branch": 4,
            "twigs_per_secondary": 2,
        },
        "splats": 6000,
        "seed": 23,
        "constants": {"TWIG_RADIUS_M": 0.013},
    },
    "bare, little foliage": {
        "rig": {},
        "splats": 12000,
        "seed": 31,
        "constants": {
            "WOOD_FRACTION": 0.62,
            "TIP_FRACTION": 0.6,
            "TIP_FOLIAGE_GIRTH": 0.14,
            "INNER_FOLIAGE_GIRTH": 0.12,
        },
    },
    "dense canopy, thin wood": {
        "rig": {},
        "splats": 20000,
        "seed": 41,
        "constants": {
            "WOOD_FRACTION": 0.16,
            "TIP_FOLIAGE_GIRTH": 0.42,
            "INNER_FOLIAGE_GIRTH": 0.38,
        },
    },
    "four times the fixture's density": {"rig": {}, "splats": 48000, "seed": 17},
}


def _grow(spec: dict, monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, per-splat true node index and per-node true radius, for one generated tree."""
    for key, value in spec.get("constants", {}).items():
        monkeypatch.setattr(synthetic_tree, key, value)
    rig = synthetic_tree.synthetic_tree_rig(**spec["rig"])
    data = synthetic_tree.generate_splats(
        rig, spec["splats"], spec["seed"], spec["rig"].get("height_m", 6.0)
    )
    return (
        data["position"].astype(np.float64),
        data["node"].astype(np.int64),
        np.asarray([node["radius"] for node in rig["nodes"]]),
    )


def _recovery(spec: dict, monkeypatch: pytest.MonkeyPatch) -> tuple[float, float, float]:
    """Median recovered/true radius ratio, median absolute log2 error, and the share within 2x."""
    xyz, labels, true_radius = _grow(spec, monkeypatch)
    keep = skeleton.isolate_tree(xyz, np.full(xyz.shape[0], 0.9))
    positions, labels = xyz[keep], labels[keep]
    nodes = skeleton.extract_skeleton(positions)
    # A node stands for whatever splats fell in its cluster, so its truth is the woody radius
    # of the limb material those splats came from — the median over its own members.
    truth = np.asarray([np.median(true_radius[labels[node["members"]]]) for node in nodes])
    recovered = np.asarray([node["radius"] for node in nodes])
    error = np.log2(recovered / truth)
    return (
        float(2 ** np.median(error)),
        float(np.median(np.abs(error))),
        float(np.mean(np.abs(error) <= 1.0)),
    )


@pytest.mark.parametrize("name", list(GENERAL_TREES))
def test_the_estimator_generalises_across_trees(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Five trees that are not the fixture, and no constant moved between them.

    This is the test the step exists for. An estimator with a constant tuned until the one tree
    whose answer is known comes out right has learned that tree; these are built to differ in
    everything it could have learned — 5 to 7 m tall, twigs 8.8 mm and 13 mm, 2 to 6 whorls,
    foliage from a sixth of the splats to nearly two thirds, halos from 0.14 to 0.42 of a
    twig's length, and point spacings from 12 to 29 mm.

    Measured median ratios: 0.92, 1.05, 0.94, 1.51, 0.80. The fixture itself is 1.42. About two
    thirds of nodes land within a factor of two either way, and that looseness is real: a band
    cluster that welds half a dozen twigs together has no single woody radius to recover.

    **Two trees outside this set fail, and both are recorded rather than dropped.** An 11 m tree
    with 5 mm twigs at a 49 mm point spacing comes out 4.7x high — every twig is far below the
    cloud's resolution, so the estimator returns the resolution limit and the limit is all it
    can honestly return. A 3 m tree with 2 cm twigs, and therefore a 34 cm trunk radius on a 3 m
    stem, comes out 2.6x low: its bark surface is far too large for 12,000 splats to cover, the
    cross-sections read as haze, and they too fall back to the limit — which is much too small
    there. Neither is a tree, but both are the same failure, and it is a sampling failure.
    """
    ratio, absolute, within = _recovery(GENERAL_TREES[name], monkeypatch)
    assert 0.5 < ratio < 2.0, f"{name}: median recovered/true radius {ratio:.2f}"
    assert absolute < 1.0, f"{name}: median |log2| error {absolute:.2f}"
    assert within > 0.45, f"{name}: only {within:.0%} of nodes within a factor of two"


def test_the_estimator_is_not_sensitive_to_its_own_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Move each constant over the range a different judgement could have picked, and re-measure.

    A constant that has to be exactly right is a constant that was fitted. These are not: over
    ``RING_POINTS`` 2 to 4, ``SHELL_NEIGHBOURS`` 3 to 8, ``SHELL_DENSITY_SHARE`` 0.5 to 0.8 and
    ``SHELL_ROUNDNESS`` 0.15 to 0.6, the median recovered/true ratio on this tree moves between
    about 0.85 and 1.05 — inside the scatter of the estimate itself.

    The one edge worth naming is ``SHELL_DENSITY_SHARE`` at 0.95 and above, where the shell
    shrinks to the handful of points at the very peak of the density profile and the estimate
    falls by about a third. 0.8 is short of that edge, deliberately.
    """
    spec = GENERAL_TREES["bushy 7 m, 30k splats"]
    ratios = [_recovery(spec, monkeypatch)[0]]
    for name, values in (
        ("RING_POINTS", (2, 4)),
        ("SHELL_NEIGHBOURS", (3, 8)),
        ("SHELL_DENSITY_SHARE", (0.5, 0.8)),
        ("SHELL_ROUNDNESS", (0.15, 0.6)),
    ):
        for value in values:
            monkeypatch.setattr(skeleton, name, value)
            ratios.append(_recovery(spec, monkeypatch)[0])
            monkeypatch.undo()
    assert min(ratios) > 0.6, ratios
    assert max(ratios) < 1.6, ratios
    assert max(ratios) / min(ratios) < 2.0, ratios
