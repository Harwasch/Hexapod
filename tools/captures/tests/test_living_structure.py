"""Probe safeguards: semantic claims must not come from forced nearest assignment."""

import numpy as np

from living_structure import (
    adjusted_rand,
    colonise,
    describe,
    graph_components,
    leaf_uris,
    load_scan,
    nearest_segments,
    representatives,
    wood_prediction,
)
from living_structure_gpu import lift_masks, merge_views
from living_structure_playback import prepare
from living_structure_render import raster_python


def test_leaf_tiles_exclude_duplicate_replacement_lods():
    doc = {
        "root": {
            "content": {"uri": "parent"},
            "children": [{"content": {"uri": "a"}}, {"content": {"uri": "b"}}],
        }
    }
    assert leaf_uris(doc) == ["a", "b"]


def test_voxel_membership_order_matches_representatives():
    p = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.031, 0, 0]])
    index, inverse = representatives(p, 0.015, 17)
    np.testing.assert_array_equal(np.floor(p[index][inverse] / 0.015), np.floor(p / 0.015))


def test_graph_cuts_colour_and_haze_bridges_but_keeps_a_surface():
    p = np.column_stack([np.arange(8) * 0.01, np.zeros((8, 2))])
    rgb = np.tile([0.1, 0.7, 0.1], (8, 1))
    rgb[4:] = [0.6, 0.1, 0.1]
    scales = np.full((8, 3), 0.003)
    quat = np.tile([0, 0, 0, 1.0], (8, 1))
    labels = graph_components(p, rgb, scales, quat, np.ones(8), 0.02, 0.2)
    assert len(np.unique(labels)) == 2
    rgb[:] = [0.1, 0.7, 0.1]
    alpha = np.ones(8)
    alpha[3:5] = 0.01
    labels = graph_components(p, rgb, scales, quat, alpha, 0.015, 0.2)
    assert labels[0] != labels[-1]
    labels = graph_components(p, rgb, scales, quat, alpha, 0.015, 0.2, alpha_min=0.001)
    assert len(np.unique(labels)) == 1


def test_nearest_binding_is_not_evidence_of_contact():
    nodes = [{"position": [0, 0, 0], "parent": -1}, {"position": [0, 0, 1], "parent": 0}]
    points = np.column_stack([np.ones(10), np.arange(10) * 0.01, np.ones(10) * 0.5])
    distance, owner = nearest_segments(points, nodes)
    np.testing.assert_allclose(distance, np.sqrt(1 + points[:, 1] ** 2))
    stats = describe(points, np.full((10, 3), 0.001), owner, distance)
    assert stats["mechanicallyBindableCrownShare"] == 1
    assert stats["plausiblePieces"] == 1
    assert stats["contactSupportedCrownShare"] == 0


def test_flat_covariance_normals_are_unoriented_but_crossings_are_cut():
    points = np.column_stack([np.arange(4) * 0.01, np.zeros((4, 2))])
    rgb = np.tile([0.1, 0.7, 0.1], (4, 1))
    scales = np.tile([0.005, 0.005, 0.0005], (4, 1))
    quat = np.tile([0, 0, 0, 1.0], (4, 1))
    quat[2:] = [0, np.sqrt(0.5), 0, np.sqrt(0.5)]
    labels = graph_components(points, rgb, scales, quat, np.ones(4), 0.015, 0.2)
    assert len(np.unique(labels)) == 2
    quat[2:] = [1, 0, 0, 0]
    labels = graph_components(points, rgb, scales, quat, np.ones(4), 0.015, 0.2)
    assert len(np.unique(labels)) == 1


def test_colonisation_is_connected_bounded_and_reproducible():
    nodes = [{"position": [0, 0, 0], "parent": -1}, {"position": [0, 0, 0.1], "parent": 0}]
    points = np.column_stack([np.zeros((20, 2)), np.linspace(0.2, 0.8, 20)])
    a = colonise(points, nodes, 17, limit=8)
    assert a == colonise(points, nodes, 17, limit=8)
    assert len(nodes) < len(a) <= 8
    assert all(-1 <= n["parent"] < i for i, n in enumerate(a))
    for i in range(len(nodes), len(a)):
        np.testing.assert_allclose(
            np.linalg.norm(np.array(a[i]["position"]) - a[a[i]["parent"]]["position"]), 0.06
        )


def test_stability_metric_ignores_label_numbers_and_detects_different_cuts():
    a = np.array([0, 0, 1, 1])
    assert adjusted_rand(a, 10 - a) == 1
    assert adjusted_rand(a, np.array([0, 1, 0, 1])) < 0


def test_wood_rule_requires_colour_and_geometry():
    rgb = np.array([[0.4, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.3, 0.2]])
    assert wood_prediction(rgb, np.full((3, 3), 0.01), np.array([0.01, 0.01, 1.0])).tolist() == [
        True,
        False,
        False,
    ]


def test_lifting_uses_alpha_votes_and_abstains_on_conflicts():
    ids = np.array([[0, 0, 1, 2]])
    masks = [
        {"area": 4, "segmentation": np.ones((1, 4), dtype=bool)},
        {"area": 1, "segmentation": np.array([[False, True, False, False]])},
    ]
    weight = np.array([[0.6, 0.6, 0.8, 0.1]])
    np.testing.assert_array_equal(lift_masks(masks, ids, weight, 3), [-1, 0, -1])
    a = np.array([0] * 10 + [1] * 10 + [-1])
    b = np.array([3] * 10 + [4] * 10 + [-1])
    merged = merge_views([a, b])
    assert merged[0] == merged[9] and merged[0] != merged[10]
    assert merged[-1] == -1


def test_raster_keeps_transmittance_and_dominant_contribution_identity():
    p = np.array([[0.5, 0.5, 1, 0, 1, 0.5], [0.5, 0.5, 1, 0, 1, 0.5]], dtype=np.float32)
    colours = np.zeros((2, 9), dtype=np.float32)
    colours[0, 0], colours[1, 1] = 1, 1
    out = np.zeros((1, 1, 9), dtype=np.float32)
    trans, best = np.ones((1, 1)), np.zeros((1, 1))
    ids, contribution = np.full((1, 1), -1), np.zeros(2)
    raster_python(2, 1, 1, p, colours, np.array([0, 1]), out, trans, best, ids, contribution)
    np.testing.assert_allclose(out[0, 0, :3], [0.5, 0.25, 0])
    np.testing.assert_allclose(trans, 0.25)
    np.testing.assert_allclose(contribution, [0.5, 0.25])
    assert ids[0, 0] == 0


def test_mask_purity_includes_visible_pixels_outside_all_masks():
    # One small mask must not claim a Gaussian whose remaining visible area is outside it.
    ids = np.zeros((1, 4), dtype=np.int32)
    masks = [{"area": 1, "segmentation": np.array([[True, False, False, False]])}]
    np.testing.assert_array_equal(lift_masks(masks, ids, np.ones((1, 4)), 1), [-1])


def test_playback_sidecar_binds_canonical_tile_and_one_branch_per_piece(tmp_path):
    import json

    from splat_tiles import build_glb, pack_spz
    from synthetic_tree import checksum_positions

    source, probe, baseline, output = (
        tmp_path / name for name in ("source", "probe", "baseline", "output")
    )
    (source / "splat").mkdir(parents=True)
    probe.mkdir()
    baseline.mkdir()
    points = np.column_stack([np.linspace(0.1, 0.16, 8), np.zeros(8), np.full(8, 2.5)])
    blob = pack_spz(
        points, np.zeros((8, 3)), np.ones(8), np.full((8, 3), -6.0), np.tile([0, 0, 0, 1.0], (8, 1))
    )
    (source / "splat/a.glb").write_bytes(
        build_glb(8, points.min(0).tolist(), points.max(0).tolist(), blob)
    )
    (source / "splat/tileset.json").write_text(json.dumps({"root": {"content": {"uri": "a.glb"}}}))
    scan = load_scan(source, probe)
    nodes = [{"position": [0, 0, 2], "parent": -1}, {"position": [0, 0, 3], "parent": 0}]
    distance, owner = nearest_segments(scan["points"], nodes)
    np.savez(
        probe / "derived.npz", crown=np.ones(8, dtype=bool), skeleton_distance=distance, owner=owner
    )
    np.save(probe / "labels-r0.025-c0.12-n0.8-seed17.npy", np.zeros(8, dtype=np.int32))
    (baseline / "articulation.json").write_text(
        json.dumps(
            {
                "nodes": nodes,
                "format": "hexapod.living-articulation",
                "version": 1,
                "motionEvidence": "simulated",
                "frequenciesHz": [0],
                "leafCount": 0,
                "tiles": {},
            }
        )
    )
    (baseline / "motion.bin").write_bytes(bytes(48))
    prepare(source, probe, baseline, output)
    header = json.loads((output / "articulation.json").read_text())
    assert header["leafCount"] == 1
    assert header["tiles"][checksum_positions(scan["points"])] == {"offset": 0, "count": 8}
    binding = np.fromfile(output / "bindings.bin", dtype="<u4").reshape(8, 4)
    np.testing.assert_array_equal(binding[:, 3], np.ones(8))
    assert len(np.unique(binding[:, 0])) == 1
    assert (output / "motion.bin").read_bytes() == bytes(48)
