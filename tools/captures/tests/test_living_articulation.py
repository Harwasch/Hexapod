import numpy as np

from living_articulation import forward_kinematics, node_spectra, segment_binding, tuft_hierarchy
from living_spectra import inverse

NODES = [
    {"position": [0, 0, 0], "parent": -1, "band": "trunk"},
    {"position": [1, 0, 0], "parent": 0, "band": "branch"},
    {"position": [2, 0, 0], "parent": 1, "band": "branch"},
]


def test_joint_rotation_inherits_parent_and_preserves_rigid_piece():
    angles = np.zeros((3, 3))
    angles[1, 2] = np.pi / 2
    rotation, shift = forward_kinematics(NODES, angles)
    points = np.array([[1, 0, 0], [2, 0, 0], [2, 0.2, 0]])
    moved = points @ rotation[2].T + shift[2]
    np.testing.assert_allclose(moved, [[0, 1, 0], [0, 2, 0], [-0.2, 2, 0]], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(moved[1] - moved[2]), 0.2)
    np.testing.assert_allclose(
        rotation @ rotation.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), rotation.shape),
        atol=1e-14,
    )


def test_binding_blends_only_at_connected_joint():
    points = np.array([[0.5, 0, 0], [1.001, 0, 0], [1.5, 0, 0]])
    binding = segment_binding(points, NODES)
    np.testing.assert_array_equal(binding[:, 0], [2, 3, 3])
    assert binding[0, 2] == binding[2, 2] == 65535
    assert 32767 <= binding[1, 2] < 34000
    assert binding[1, 1] == 2


def test_rotation_spectrum_roundtrip_and_pinned_root():
    f, c, error = node_spectra(NODES)
    t = np.arange(300) / 30
    signal = inverse(f, c, t)
    np.testing.assert_allclose(inverse(f, c, t + 10), signal, atol=1e-14)
    np.testing.assert_array_equal(signal[:, 0], 0)
    assert len(f) == 16
    assert (
        error < 1e-6
    )  # Analytic periodic forcing; integration transient after five warmup periods.
    assert np.ptp(signal[:, 1]) > 0.01


def test_tufts_are_ground_rooted_separate_chains():
    rng = np.random.default_rng(4)
    points = rng.uniform([0.01, 0.01, 0.1], [0.2, 0.2, 0.5], (100, 3))
    nodes = tuft_hierarchy(points, np.ones(100, dtype=bool), 0)
    assert len(nodes) == 4
    assert nodes[0]["position"][2] == 0
    assert [n["parent"] for n in nodes] == [-1, 0, 1, 2]


def test_baked_files_roundtrip_and_bind_actual_tile_identity(tmp_path):
    import json

    from living_articulation import bake
    from splat_tiles import build_glb, pack_spz
    from synthetic_tree import checksum_positions

    source = tmp_path / "input"
    (source / "splat").mkdir(parents=True)
    (source / "source").mkdir()
    points = np.array([[0.5, 0, 1], [1.5, 0, 1]], dtype=np.float32)
    blob = pack_spz(
        points,
        np.array([[0, 1, 0], [0, 1, 0]]),
        np.ones(2),
        np.full((2, 3), -6),
        np.tile([0, 0, 0, 1], (2, 1)),
    )
    (source / "splat/splat.glb").write_bytes(
        build_glb(2, points.min(0).tolist(), points.max(0).tolist(), blob)
    )
    (source / "splat/tileset.json").write_text(
        json.dumps({"root": {"content": {"uri": "splat.glb"}}})
    )
    (source / "source/rig.json").write_text(json.dumps({"nodes": NODES}))
    out = tmp_path / "output"
    meta = bake(source, out)
    loaded = json.loads((out / "articulation.json").read_text())
    assert loaded == meta
    assert loaded["motionEvidence"] == "simulated"
    assert loaded["tiles"][checksum_positions(points)] == {"offset": 0, "count": 2}
    c = np.fromfile(out / "motion.bin", dtype="<f4").reshape(3, 16, 3, 2)
    restored = inverse(
        np.array(meta["frequenciesHz"]), c[..., 0] + 1j * c[..., 1], np.arange(300) / 30
    )
    assert np.isfinite(restored).all()
    binding = np.fromfile(out / "bindings.bin", dtype="<u4").reshape(2, 4)
    assert (binding[:, 0] > 0).all()
    assert (binding[:, 3] > 0).all()
    leaves = np.fromfile(out / "leaves.bin", dtype="<f4").reshape(-1, 9)
    assert len(leaves) == loaded["leafCount"]
    assert ((leaves[:, 6] >= 2) & (leaves[:, 6] <= 6)).all()


def test_breeze_pins_trunk_and_ground_before_flutter():
    from living_articulation import pin_static

    nodes = [dict(NODES[0]), dict(NODES[1])]
    points = np.array([[0, 0, 2], [1, 0, -0.35], [1, 0, 1]])
    binding = np.array([[1, 0, 65535, 9], [2, 1, 65535, 9], [2, 1, 65535, 9]], dtype="<u4")
    pin_static(binding, points, nodes, -0.35)
    np.testing.assert_array_equal(binding[:2], 0)
    np.testing.assert_array_equal(binding[2], [2, 1, 65535, 9])
