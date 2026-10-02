"""The scene step (scene_plants.py) on the synthetic yard, where every answer is known.

``synthetic_yard.py`` builds a yard whose every splat carries its class and plant; this scores
the scene step against it -- per-class IoU over splats, instance counts, heights -- and checks
the committed fixture (data/tiles/synthetic-yard/splat) is what a fresh run writes, the rules
one by one, and that the result survives the two things a real capture will do differently:
another density, and ground that is not level.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import fetch_capture
import motion_params
import scene_plants as sp
import splat_tiles
import synthetic_yard
from synthetic_tree import write_ply

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED = REPO_ROOT / "data" / "tiles" / "synthetic-yard" / "splat"
SYNTHETIC_TREE = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source"


@pytest.fixture(scope="module")
def yard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("yard")
    synthetic_yard.generate(out)
    return out


@pytest.fixture(scope="module")
def scene(yard: Path) -> dict:
    return json.loads((yard / "splat" / "scene.json").read_text(encoding="utf-8"))


def _capture(ply: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = sp.read_capture(ply, sp.PACKAGE_OPACITY_MIN)
    keep = data["keep"]
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)[keep].astype(np.float64)
    rgb = np.clip(
        0.5 + splat_tiles.SH_C0 * np.stack([data[f"f_dc_{k}"] for k in range(3)], 1)[keep], 0, 1
    )
    return xyz, rgb, keep


def _truth(yard: Path) -> dict:
    return json.loads((yard / "source" / "labels.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ the fixture


def test_the_committed_yard_is_what_a_fresh_run_writes(yard: Path) -> None:
    """Byte for byte: tiles, rig, sidecar, binding, scene report, ground and classes."""
    fresh = sorted(p.name for p in (yard / "splat").iterdir())
    committed = sorted(p.name for p in COMMITTED.iterdir())
    assert fresh == committed
    for name in committed:
        assert (yard / "splat" / name).read_bytes() == (COMMITTED / name).read_bytes(), name


def test_two_runs_write_identical_bytes(tmp_path: Path, yard: Path) -> None:
    sp.build(
        yard / "source" / "splat.ply",
        tmp_path,
        tiles_dir=yard / "splat",
        opacity_min=0.02,
        tile_gaussians=6000,
        truth=_truth(yard),
    )
    for name in ("rig.json", "motion.json", "plants.json", "scene.json", "classes.u8"):
        assert (tmp_path / name).read_bytes() == (yard / "splat" / name).read_bytes(), name


# ------------------------------------------------------------------------- the scores


def test_every_class_is_found_on_the_yard(scene: dict) -> None:
    iou = scene["score"]["classIoU"]
    for name in ("tree", "shrub", "snag", "grass/low"):
        assert iou[name] >= 0.95, (name, iou[name])
    # The ground and the building lose only their bottom centimetres to each other.
    assert iou["ground"] >= 0.9
    assert iou["other-static"] >= 0.8


def test_every_plant_is_found_once_with_its_height(scene: dict) -> None:
    score = scene["score"]
    for name, counts in score["instances"].items():
        assert counts["found"] == counts["truth"], name
    truths = [m["truth"] for m in score["matched"]]
    assert len(truths) == len(set(truths)) and None not in truths
    assert all(m["classMatches"] for m in score["matched"])
    assert score["heightErrorMaxM"] <= 0.25
    assert max(m["stemErrorM"] for m in score["matched"]) <= 0.2


def test_the_evidence_picks_the_rig(scene: dict) -> None:
    """Dense trees get the banded skeleton, the sparse one and every shrub a crown rig, snags a
    trunk: by splats per metre of height against the skeleton's own tested floor."""
    rigs = {p["id"]: (p["rig"], p["splatsPerM"]) for p in scene["plants"]}
    for plant in scene["plants"]:
        rig, density = rigs[plant["id"]]
        if plant["class"] == "tree":
            assert rig == ("skeleton" if density >= sp.SKELETON_MIN_SPLATS_PER_M else "crown")
        elif plant["class"] == "shrub":
            assert rig == "crown"
        else:
            assert rig == "trunk"
    assert sorted(r for r, _ in rigs.values()).count("crown") == 6  # 5 shrubs + the sparse tree


def test_the_skeleton_floor_is_the_fixture_it_is_tested_on() -> None:
    sidecar = json.loads((SYNTHETIC_TREE / "motion.json").read_text(encoding="utf-8"))
    labels = json.loads((SYNTHETIC_TREE / "labels.json").read_text(encoding="utf-8"))
    assert sp.SKELETON_FIXTURE_HEIGHT_M == sidecar["treeHeightM"]
    assert sp.SKELETON_FIXTURE_SPLATS == len(labels["nodes"])


def test_every_threshold_says_where_it_came_from(scene: dict) -> None:
    for name, entry in scene["thresholds"].items():
        if isinstance(entry, dict):
            assert "rule" in entry or "source" in entry, name


# ------------------------------------------------------------ what a real capture varies


def _rescore(ply: Path, labels: dict, rows: np.ndarray) -> dict:
    xyz, rgb, keep = _capture(ply)
    analysis = sp.analyse(xyz, rgb)
    kept = rows[keep]
    return sp.score(
        analysis,
        np.asarray(labels["class"])[kept],
        np.asarray(labels["instance"])[kept],
        labels["instances"],
    )


def _rewrite(yard: Path, tmp: Path, keep_rows: np.ndarray, shear: tuple[float, float]) -> Path:
    data = splat_tiles.read_ply(yard / "source" / "splat.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1)[keep_rows].astype(np.float64)
    xyz[:, 2] += shear[0] * xyz[:, 0] + shear[1] * xyz[:, 1]
    out = tmp / "variant.ply"
    write_ply(
        out,
        {
            "position": xyz.astype(np.float32),
            "rgb": np.stack([data[f"f_dc_{k}"] for k in range(3)], 1)[keep_rows] * splat_tiles.SH_C0
            + 0.5,
            "opacity_logit": data["opacity"][keep_rows],
            "log_scale": np.stack([data[f"scale_{k}"] for k in range(3)], 1)[keep_rows],
            "quat_wxyz": np.stack([data[f"rot_{k}"] for k in range(4)], 1)[keep_rows],
        },
    )
    return out


def test_half_the_density_finds_the_same_plants(yard: Path, tmp_path: Path) -> None:
    """Every radius is relative to the density where a splat stands."""
    labels = _truth(yard)
    rows = np.arange(len(labels["class"]))[::2]
    score = _rescore(_rewrite(yard, tmp_path, rows, (0.0, 0.0)), labels, rows)
    assert score["instances"]["tree"]["found"] == 3
    assert score["instances"]["snag"]["found"] == 2
    # Two of the shrubs stand 36 cm apart: four point spacings at half density, inside the
    # link radius, so they are one shrub there -- an unresolved gap is not a gap.
    assert score["instances"]["shrub"]["found"] in (4, 5)
    for name in ("tree", "snag"):
        assert score["classIoU"][name] >= 0.9, (name, score["classIoU"][name])
    # A shrub's lowest leaves, half as dense, are further from each other than from the lawn.
    assert score["classIoU"]["shrub"] >= 0.85


def test_a_sloping_yard_finds_the_same_plants(yard: Path, tmp_path: Path) -> None:
    """Ground at 8 % along one axis and 5 % along the other: the slope bound is the capture's."""
    labels = _truth(yard)
    rows = np.arange(len(labels["class"]))
    score = _rescore(_rewrite(yard, tmp_path, rows, (0.08, 0.05)), labels, rows)
    for name, counts in score["instances"].items():
        assert counts["found"] == counts["truth"], (name, counts)
    for name in ("tree", "shrub", "snag", "grass/low"):
        assert score["classIoU"][name] >= 0.9, (name, score["classIoU"][name])
    assert score["heightErrorMaxM"] <= 0.3


def test_a_sampled_analysis_labels_every_splat_as_the_full_one_does(
    yard: Path, tmp_path: Path, scene: dict
) -> None:
    """A capture over ``ANALYSIS_MAX_SPLATS`` is analysed on a uniform sample and the rest take
    their nearest analysed splat's labels: forced here with a cap of three quarters of the
    yard, the same plants are found and nearly every splat gets the full analysis's labels."""
    labels = _truth(yard)
    kept = scene["keptSplats"]
    cap = kept * 3 // 4
    report = sp.build(
        yard / "source" / "splat.ply",
        tmp_path,
        tiles_dir=yard / "splat",
        opacity_min=0.02,
        tile_gaussians=6000,
        truth=labels,
        max_analysed=cap,
    )
    assert not scene["analysis"]["subsampled"] and scene["analysedSplats"] == kept
    assert report["analysis"]["subsampled"] and report["analysedSplats"] == cap
    assert report["keptSplats"] == kept and sum(report["classCounts"].values()) == kept
    found = {name: c["found"] for name, c in report["score"]["instances"].items()}
    assert found["tree"] == 3 and found["snag"] == 2
    # The two shrubs 36 cm apart are a gap a thinner cloud may not resolve (as at half density).
    assert found["shrub"] in (4, 5)
    for name in ("tree", "snag"):
        assert report["score"]["classIoU"][name] >= 0.9, name
    # Every kept splat has a class, and it is the full analysis's for all but a few.
    fresh = np.frombuffer((tmp_path / "classes.u8").read_bytes(), dtype=np.uint8)
    full = np.frombuffer((yard / "splat" / "classes.u8").read_bytes(), dtype=np.uint8)
    keep = sp.read_capture(yard / "source" / "splat.ply", 0.02)["keep"]
    assert fresh.size == full.size == keep.size
    assert float(np.mean(fresh[keep] == full[keep])) >= 0.97
    moving = np.isin(fresh[keep], sp.PLANT_CLASSES)
    assert float(np.mean(moving == np.isin(full[keep], sp.PLANT_CLASSES))) >= 0.97
    # Every gaussian of every tile is bound, and the plants hold what the classes say.
    binding = json.loads((tmp_path / "plants.json").read_text(encoding="utf-8"))
    for checksum, runs in binding["tiles"].items():
        assert sum(runs[1::2]) == int(checksum.split(":")[1])
    assert sum(p["boundSplats"] for p in report["plants"]) == int(moving.sum())
    assert all(p["boundSplats"] >= p["splats"] for p in report["plants"])


def test_labels_reach_every_splat_from_the_nearest_analysed_one() -> None:
    rng = np.random.default_rng(8)
    xyz = rng.uniform(0, 10, (5000, 3)).astype(np.float32)
    xyz[1] = xyz[0]  # coincident: each keeps its own label when both are analysed
    sample = sp.analysis_sample(xyz.shape[0], 1000, seed=4)
    assert sample is not None and sample.size == 1000 and np.all(np.diff(sample) > 0)
    assert np.array_equal(sample, sp.analysis_sample(xyz.shape[0], 1000, seed=4))
    assert sp.analysis_sample(xyz.shape[0], 5000, seed=4) is None
    assert sp.analysis_sample(xyz.shape[0], None, seed=4) is None
    sample = np.union1d(sample, [0, 1])
    analysed = xyz[sample].astype(np.float64)
    own = np.arange(sample.size)
    (label,) = sp.propagate(analysed, xyz, sample, own)
    assert label.shape == (5000,)
    assert np.array_equal(label[sample], own)
    distance = np.linalg.norm(xyz[:, None, :] - analysed[None, :, :], axis=2)
    assert np.allclose(distance[np.arange(5000), label], distance.min(axis=1), rtol=0, atol=1e-6)


# ------------------------------------------------------------------------- the rules


def test_tree_tops_are_those_of_a_full_raster_filter_per_radius() -> None:
    """The candidate search finds exactly the tops one maximum filter per radius finds."""
    from scipy import ndimage

    def reference(chm: np.ndarray, cell: float) -> np.ndarray:
        filled = np.where(np.isfinite(chm), chm, -np.inf)
        tall = filled >= sp.FAO_TREE_MIN_M
        crown = np.where(tall, sp.popescu_wynne_crown_m(np.where(tall, filled, 0.0)), 0.0)
        radius_cells = np.ceil(crown / 2.0 / cell).astype(np.int64)
        tops = np.zeros(chm.shape, dtype=bool)
        for r in np.unique(radius_cells[tall]):
            highest = ndimage.maximum_filter(
                filled, footprint=sp._disk(int(r)), mode="constant", cval=-np.inf
            )
            tops |= tall & (radius_cells == r) & (filled >= highest)
        return tops

    rng = np.random.default_rng(12)
    for _ in range(40):
        shape = tuple(int(v) for v in rng.integers(1, 40, 2))
        chm = ndimage.gaussian_filter(rng.uniform(0, 20, shape), float(rng.uniform(0, 3)))
        chm = np.round(chm * 2) / 2  # plateaus of equal maxima
        chm[rng.random(shape) < 0.2] = np.nan
        cell = float(rng.choice([0.25, 0.5, 1.0]))
        expected = reference(chm, cell)
        found = sp._tree_tops(chm, cell)
        # One cell per plateau of tops: the same plateaus, one of each.
        labels, count = ndimage.label(expected, structure=np.ones((3, 3)))
        assert int(found.sum()) == (count if count > 1 else int(expected.sum()))
        assert np.all(expected[found])
        if count > 1:
            assert sorted(np.unique(labels[found]).tolist()) == list(range(1, count + 1))


def test_otsu_splits_two_populations_between_them() -> None:
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(0.0, 0.02, 4000), rng.normal(0.4, 0.05, 2000)])
    threshold, effectiveness = sp.otsu_threshold(values)
    assert 0.05 < threshold < 0.3
    assert effectiveness > 0.8


def test_exg_is_green_whatever_the_brightness() -> None:
    rgb = np.asarray([[0.1, 0.2, 0.08], [0.4, 0.8, 0.32], [0.34, 0.235, 0.155], [0.5, 0.5, 0.5]])
    exg, _ = sp.chromatic_indices(rgb)
    assert exg[0] == pytest.approx(exg[1])
    assert exg[0] > 0.2 and exg[2] < 0.0 and exg[3] == pytest.approx(0.0)


def test_the_slope_envelope_is_the_terrain_under_objects() -> None:
    """A tilted plane with a box and a column standing on it: the envelope is the plane."""
    cell = 0.25
    x, y = np.meshgrid(np.arange(80) * cell, np.arange(60) * cell, indexing="ij")
    plane = 0.1 * x + 0.03 * y
    lowest = plane.copy()
    lowest[20:36, 20:32] += 3.0  # a building's roof, the walls unseen
    lowest[60:63, 40:43] += 1.5  # a crown with no stem under it
    slopes = sp._neighbour_slopes(lowest, cell)
    median, sigma = sp.robust_sigma(slopes)
    envelope = sp.slope_envelope(lowest, cell, median + sp.SIGMA_CLIP * sigma)
    error = np.abs(envelope - plane)
    assert error.max() < 0.1 * cell * 16  # never higher than the slope bound allows
    assert np.median(error) == pytest.approx(0.0, abs=1e-9)


def test_crown_width_is_popescu_and_wynne() -> None:
    assert sp.popescu_wynne_crown_m(10.0) == pytest.approx(2.51503 + 0.901)


def test_a_snag_bends_by_its_frontal_area() -> None:
    assert sp.snag_bend_scale(0.4, 10.0) == pytest.approx(0.4 / (0.4 + (2.51503 + 0.901) / 2))
    assert sp.snag_bend_scale(0.0, 10.0) == 0.0
    # A thicker trunk catches more of the wind a crown would.
    assert sp.snag_bend_scale(0.8, 10.0) > sp.snag_bend_scale(0.4, 10.0)


def test_an_unrooted_basin_joins_the_rooted_one_it_borders_most() -> None:
    basins = np.asarray([[0, 0, 1, 1], [0, 0, 1, 2], [-1, 0, 2, 2]])
    target = sp._merge_unrooted(basins, np.asarray([True, False, False]))
    assert target.tolist() == [0, 0, 0]
    # Nothing rooted: the tops stand as they are.
    assert sp._merge_unrooted(basins, np.asarray([False, False, False])).tolist() == [0, 1, 2]


def test_a_crown_finds_the_trunk_under_it() -> None:
    rng = np.random.default_rng(5)
    trunk = np.column_stack(
        [rng.normal(0, 0.05, 200), rng.normal(0, 0.05, 200), rng.uniform(0, 4, 200)]
    )
    crown = np.column_stack(
        [rng.normal(0, 1.5, 800), rng.normal(0, 1.5, 800), rng.uniform(4.5, 7, 800)]
    )
    lobe = np.column_stack(
        [rng.normal(3.2, 0.2, 60), rng.normal(0, 0.2, 60), rng.uniform(5, 6, 60)]
    )
    xyz = np.concatenate([trunk, crown, lobe])
    height = xyz[:, 2]
    instances = [
        sp.Instance(members=np.arange(0, 200)),
        sp.Instance(members=np.arange(200, 1000)),
        sp.Instance(members=np.arange(1000, 1060)),
    ]
    joined = sp._assemble(instances, xyz, height)
    assert len(joined) == 1
    assert joined[0].members.size == 1060


def test_a_far_unrooted_piece_stays_its_own() -> None:
    rng = np.random.default_rng(6)
    shrub = np.column_stack(
        [rng.normal(0, 0.3, 300), rng.normal(0, 0.3, 300), rng.uniform(0, 1, 300)]
    )
    floater = np.column_stack(
        [rng.normal(20, 0.1, 30), rng.normal(0, 0.1, 30), rng.uniform(6, 6.3, 30)]
    )
    xyz = np.concatenate([shrub, floater])
    instances = [sp.Instance(members=np.arange(300)), sp.Instance(members=np.arange(300, 330))]
    assert len(sp._assemble(instances, xyz, xyz[:, 2])) == 2


def test_run_lengths_round_trip() -> None:
    values = np.asarray([0, 0, 3, 3, 3, 0, 1])
    runs = sp._rle(values)
    assert runs == [0, 2, 3, 3, 0, 1, 1, 1]
    assert sp._rle(np.zeros(0, dtype=np.int64)) == []


# ------------------------------------------------------------------ rigs and motion


def test_the_forest_rig_is_one_root_per_plant_and_a_static_anchor(yard: Path) -> None:
    rig = json.loads((yard / "splat" / "rig.json").read_text(encoding="utf-8"))
    assert sp.forest_rig_issues(rig) == []
    assert rig["nodes"][0]["id"] == sp.STATIC_NODE_ID and rig["nodes"][0]["parent"] == -1
    assert rig["binding"] == "plants.json" and rig["motion"] == "motion.json"
    roots = [i for i, n in enumerate(rig["nodes"]) if n["parent"] == -1]
    assert roots == [0] + [p["nodeStart"] for p in rig["plants"]]
    broken = json.loads(json.dumps(rig))
    second = rig["plants"][1]["nodeStart"]
    broken["nodes"][second + 1]["parent"] = 1
    assert any("not an earlier node of its plant" in i for i in sp.forest_rig_issues(broken))


def test_each_class_moves_by_its_own_rules(yard: Path) -> None:
    rig = json.loads((yard / "splat" / "rig.json").read_text(encoding="utf-8"))
    sidecar = json.loads((yard / "splat" / "motion.json").read_text(encoding="utf-8"))
    columns = sidecar["nodes"]
    assert columns["gainRad"][0] == 0 and columns["flutterM"][0] == 0
    for plant in sidecar["plants"]:
        start, end = plant["nodeStart"], plant["nodeEnd"]
        f0 = columns["frequencyHz"][start]
        # Every plant's whole-plant mode is the pendulum law at its own height.
        assert f0 == pytest.approx(motion_params.tree_frequency_hz(plant["heightM"]), abs=1e-4)
        assert columns["gainRad"][start] == 0
        assert all(columns["branch"][i] >= start for i in range(start, end))
        gains = columns["gainRad"][start:end]
        if plant["class"] == "snag":
            assert all(v == 0 for v in columns["flutterM"][start:end])
            assert all(m == 0 for m in columns["mode"][start:end])
            # Near still: its total bend a fraction of a crowned tree's.
            assert sum(gains) < 0.3 * motion_params.TREE_BEND_REF_RAD
        else:
            assert any(v > 0 for v in columns["flutterM"][start:end])
        if plant["class"] == "shrub":
            assert f0 > motion_params.tree_frequency_hz(5.0)
    assert len(sidecar["plants"]) == len(rig["plants"])


def test_the_binding_labels_every_tile_by_its_checksum(yard: Path) -> None:
    rig = json.loads((yard / "splat" / "rig.json").read_text(encoding="utf-8"))
    binding = json.loads((yard / "splat" / "plants.json").read_text(encoding="utf-8"))
    assert sorted(binding["tiles"]) == sorted(rig["tileChecksums"])
    for checksum, runs in binding["tiles"].items():
        count = int(checksum.split(":")[1])
        assert sum(runs[1::2]) == count
        assert all(0 <= v <= len(rig["plants"]) for v in runs[0::2])


def test_merged_parents_take_a_plant_only_when_every_original_is_its(yard: Path) -> None:
    """Replays the packer independently: a parent's keys, from ``splat_tiles.hierarchy``."""
    ply = yard / "source" / "splat.ply"
    classes = np.frombuffer((yard / "splat" / "classes.u8").read_bytes(), dtype=np.uint8)
    xyz, rgb, keep = _capture(ply)
    analysis = sp.analyse(xyz, rgb)
    labels_by_row = np.zeros(keep.size, dtype=np.int64)
    labels_by_row[np.flatnonzero(keep)] = analysis.plant + 1
    assert np.array_equal(classes[np.flatnonzero(keep)], analysis.klass)
    labels = sp.tile_labels(ply, labels_by_row, 0.02, 6000)
    counts: dict[str, int] = {}

    def emit(tile, gaussians, keys) -> None:
        counts[tile.uri] = int(gaussians.xyz.shape[0])

    splat_tiles.hierarchy(ply, 0.02, 6000, yard, emit)
    assert {uri: v.size for uri, v in labels.items()} == counts
    leaves = np.concatenate([v for uri, v in labels.items() if "-" in uri or uri == "splat.glb"])
    assert leaves.size > 0
    # Every plant's splats are in the leaves exactly as often as the scene step found them.
    tileset = json.loads((yard / "splat" / "tileset.json").read_text(encoding="utf-8"))

    def leaf_uris(tile: dict) -> list[str]:
        kids = tile.get("children", [])
        return [tile["content"]["uri"]] if not kids else [u for k in kids for u in leaf_uris(k)]

    in_leaves = np.concatenate([labels[uri] for uri in leaf_uris(tileset["root"])])
    for k in range(len(analysis.plants)):
        assert int((in_leaves == k + 1).sum()) == int((analysis.plant == k).sum())


def test_tiles_from_another_packing_are_refused(yard: Path, tmp_path: Path) -> None:
    labels = np.zeros(len(_truth(yard)["class"]), dtype=np.int64)
    with pytest.raises(SystemExit, match="not the ones this PLY packs to"):
        sp.plant_binding(
            yard / "splat", yard / "source" / "splat.ply", labels, opacity_min=0.02,
            tile_gaussians=4000,
        )  # fmt: skip


# ---------------------------------------------------------------------- fetching


def test_a_capture_in_unknown_units_is_refused_unless_said_otherwise() -> None:
    with pytest.raises(SystemExit, match="not known to be in metres"):
        fetch_capture.check_scale({"scaleSource": "unresolved"}, assume_metric=False)
    assert (
        fetch_capture.check_scale({"scaleSource": "unresolved"}, assume_metric=True) == "unresolved"
    )
    assert fetch_capture.check_scale({"scaleSource": "arkit"}, assume_metric=False) == "arkit"


def test_the_splat_asset_is_the_sites_gaussian_splat() -> None:
    assets = [
        {"id": "a", "siteId": "s", "representation": "mesh"},
        {"id": "b", "siteId": "s", "representation": "gaussian-splat"},
    ]
    assert fetch_capture.splat_asset(assets, "s")["id"] == "b"
    with pytest.raises(SystemExit):
        fetch_capture.splat_asset(assets, "t")


def test_math_is_the_one_in_the_docstring() -> None:
    assert sp.SKELETON_MIN_SPLATS_PER_M == pytest.approx(12000 / 4 / 6.4597)
    assert math.isclose(sp.FAO_TREE_MIN_M / sp.FAO_SHRUB_MIN_M, 10.0)
