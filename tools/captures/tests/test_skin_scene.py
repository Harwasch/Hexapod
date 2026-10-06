"""Skins (skin_scene.py, kaolin_rkpm.py): the basis, the format, and how a skin deforms.

The synthetic tree is the deformation case (a trunk, limbs, a crown: what bends); the yard is
the format case (tiles, instances, the committed `synthetic-yard/skin/` fixture).
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import kaolin_rkpm
import skin_scene

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "positions.f32"
YARD = ROOT / "data" / "tiles" / "synthetic-yard"
FIXTURE_ONLY = (1, 9, 10, 12)


@cache
def tree_points() -> np.ndarray:
    return np.frombuffer(TREE.read_bytes(), "<f4").reshape(-1, 3).astype(np.float64)


@cache
def tree_skin() -> skin_scene.Skin:
    return skin_scene.fit_skin(tree_points(), 1, 1)


@cache
def yard_instances() -> dict:
    return json.loads((YARD / "instances" / "instances.json").read_text(encoding="utf-8"))


@cache
def yard_tiles() -> tuple[skin_scene.TileSplats, ...]:
    return tuple(skin_scene.read_tiles(YARD / "splat", yard_instances()))


# ------------------------------------------------------------------------------- the basis


def test_rkpm_reproduces_linear_fields_and_its_gradient_is_the_derivative():
    rng = np.random.default_rng(0)
    nodes = rng.uniform(size=(60, 3))
    rkpm = kaolin_rkpm.RKPM(nodes, np.full(60, 0.3))
    x = rng.uniform(0.2, 0.8, size=(40, 3))
    phi = rkpm.phi(x)
    # First-order consistency: Σφ = 1 and Σφ·x_I = x.
    assert np.allclose(phi.sum(1), 1.0, atol=1e-8)
    assert np.allclose(phi @ nodes, x, atol=1e-8)
    h = 1e-6
    numeric = np.stack(
        [(rkpm.phi(x + h * e) - rkpm.phi(x - h * e)) / (2 * h) for e in np.eye(3)], axis=-1
    )
    assert np.allclose(rkpm.grad_phi(x), numeric, atol=1e-5 * np.abs(numeric).max())


def test_flat_objects_get_a_skin():
    # A lawn: every node coplanar, so the moment matrix is singular without the ridge.
    rng = np.random.default_rng(2)
    flat = np.c_[rng.uniform(0, 4, size=(300, 2)), np.zeros(300)]
    skin = skin_scene.fit_skin(flat, 1, 1)
    w = skin.weights(flat)
    assert np.isfinite(w).all()
    assert np.allclose(np.abs(w).max(0), 1.0)


def test_an_object_in_pieces_keeps_no_free_motion(monkeypatch):
    # A crown cut from its trunk by segmentation (a gap of 0.7 m). Each piece alone would
    # keep a rigid motion of its own: an eigenvalue of ~0 that the wind drives without bound.
    rng = np.random.default_rng(4)
    trunk = np.c_[rng.uniform(-0.12, 0.12, (1500, 2)), rng.uniform(0, 2, 1500)]
    d = rng.normal(size=(3000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    crown = d * (0.8 * rng.uniform(size=(3000, 1)) ** (1 / 3)) + (0, 0, 3.5)
    pieces = np.r_[trunk, crown]
    bridged = kaolin_rkpm.bridged_radii
    groups: list[int] = []

    def spy(nodes, radius):
        out = bridged(nodes, radius)
        groups.append(out[1])
        return out

    monkeypatch.setattr(kaolin_rkpm, "bridged_radii", spy)
    one = skin_scene.fit_skin(trunk, 1, 1)
    two = skin_scene.fit_skin(pieces, 1, 1)
    assert groups == [1, 2]
    assert one.eigenvalues[0] > 0.01 * one.eigenvalues[1]
    # Bridged, the lowest mode is the crown swaying on its trunk: soft, never free.
    assert two.eigenvalues[0] > 1e-3 * two.eigenvalues[1]
    monkeypatch.setattr(kaolin_rkpm, "bridged_radii", lambda nodes, radius: (radius, 1))
    loose = skin_scene.fit_skin(pieces, 1, 1)
    assert loose.eigenvalues[0] < 1e-6 * loose.eigenvalues[1]
    # A shape in one piece keeps its radii exactly.
    nodes = trunk[:200]
    radius = np.full(200, 0.3)
    same, count = bridged(nodes, radius)
    assert count == 1 and np.array_equal(same, radius)


def test_handle_count_scales_with_size_within_eight_to_sixteen():
    assert skin_scene.handle_count(0.5) == 8
    assert skin_scene.handle_count(2.0) == 8
    assert skin_scene.handle_count(8.0) == 12
    assert skin_scene.handle_count(32.0) == 16
    assert skin_scene.handle_count(500.0) == 16
    counts = [skin_scene.handle_count(d) for d in np.geomspace(0.1, 100, 30)]
    assert counts == sorted(counts)


def test_the_tree_skin_is_normalised_signed_and_not_a_partition_of_unity():
    skin = tree_skin()
    assert 8 <= skin.handles <= 16
    w = skin.weights(tree_points())
    assert w.shape == (len(tree_points()), skin.handles - 1)
    assert np.allclose(w.max(0), 1.0)  # each field peaks at +1
    assert (w.min(0) < -0.05).all()  # and is signed
    assert np.ptp(w.sum(1)) > 0.5  # nothing makes them sum to one
    assert (np.diff(skin.eigenvalues) >= -1e-9).all()


# ------------------------------------------------------------------------ how a skin moves


def test_the_constant_handle_moves_the_instance_rigidly():
    skin = tree_skin()
    points = tree_points()
    w = skin_scene.dequantise(skin_scene.quantise(skin.weights(points)), skin.handles - 1)
    angle = 0.3
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    r = r @ np.array([[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]])
    t = np.array([0.4, -1.2, 0.25])
    z = np.zeros((skin.handles, 3, 4))
    z[0, :, :3] = r - np.eye(3)
    z[0, :, 3] = t
    moved = skin_scene.deform(points, skin.origin, w, z)
    expected = (points - skin.origin) @ r.T + t + skin.origin
    assert np.abs(moved - expected).max() < 1e-9
    stretch = skin_scene.edge_stretch(points, moved)
    assert np.allclose(stretch, 1.0, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_small_random_handles_bend_the_tree_without_tearing_or_folding(seed):
    skin = tree_skin()
    points = tree_points()
    w = skin_scene.dequantise(skin_scene.quantise(skin.weights(points)), skin.handles - 1)
    rng = np.random.default_rng(seed)
    z = skin_scene.random_handles(rng, skin.handles, skin.scale)
    # The crown sways by 2% of the tree's half-height: a strong wind.
    z = skin_scene.scaled_to(points, skin.origin, w, z, 0.02 * skin.scale)
    moved = skin_scene.deform(points, skin.origin, w, z)
    assert np.abs(moved - points).max() > 0.019 * skin.scale
    stretch = skin_scene.edge_stretch(points, moved)
    assert np.percentile(stretch, 99) < 1.06
    assert np.percentile(stretch, 1) > 0.94
    assert stretch.max() < 1.3
    det = np.linalg.det(skin_scene.local_jacobians(points, moved))
    assert det.min() > 0.5  # no inverted neighbourhoods
    # The analytic Jacobian (with ∇w) agrees: I + Σ_j (w_j A_j + Z_j[x;1] ∇w_jᵀ) stays positive.
    sample = points[:: max(1, len(points) // 2000)]
    jac = skin_scene.skin_jacobians(skin, sample, z)
    assert np.linalg.det(jac).min() > 0.5
    # What the viewer drops (the ∇w term) is small at this amplitude.
    w = np.c_[np.ones(len(sample)), skin.weights(sample)]
    kept = np.eye(3)[None] + np.einsum("nj,jab->nab", w, z[:, :, :3])
    assert np.abs(jac - kept).max() < 0.25


def test_dense_int8_is_close_to_float_and_top_k_is_not():
    skin = tree_skin()
    report = skin_scene.sparsity_report(skin, tree_points()[::3], ks=(4,), trials=3)
    assert report["int8"]["rmsError"] < 0.015
    assert report["int8"]["stretchP99"] < report["float"]["stretchP99"] + 0.01
    # The modes are global: keeping four per splat loses a large share of the motion.
    assert report["top4-int8"]["rmsError"] > 10 * report["int8"]["rmsError"]


# ---------------------------------------------------------------------------- the format


def test_skin_owners_take_the_coarsest_moving_ancestor():
    instances = [
        {"id": 1, "parent": None, "behaviour": "static"},
        {"id": 2, "parent": 1, "behaviour": "in-place"},
        {"id": 3, "parent": 2, "behaviour": "in-place"},
        {"id": 4, "parent": None, "behaviour": "movable"},
        {"id": 5, "parent": 4, "behaviour": "static"},
        {"id": 6, "parent": 1, "behaviour": "static"},
    ]
    assert skin_scene.skin_owners(instances).tolist() == [0, 0, 2, 2, 4, 4, 0]


def test_round_trip_and_size_on_the_yard(tmp_path):
    tiles = yard_tiles()
    built = skin_scene.build(tiles, yard_instances()["instances"], only=(10, 12, 16))
    skin_scene.write_skin(tmp_path, built)
    document = json.loads((tmp_path / "skin.json").read_text(encoding="utf-8"))
    blob = (tmp_path / "skin.bin").read_bytes()
    assert document["format"] == "hexapod.skin" and document["version"] == 1
    assert [s["instance"] for s in document["skins"]] == [10, 12, 16]
    owner = skin_scene.skin_owners(yard_instances()["instances"])
    decoded = skin_scene.decode_tiles(document, blob)
    assert set(decoded) == {t.checksum for t in tiles}
    skinned = 0
    for tile in tiles:
        which, rows = decoded[tile.checksum]
        own = owner[tile.ids]
        expected = np.select([own == k for k in (10, 12, 16)], [1, 2, 3], 0)
        assert np.array_equal(which, expected)
        for s, skin in enumerate(built.skins, start=1):
            at = which == s
            want = skin_scene.quantise(skin.weights(tile.positions[at].astype(np.float64)))
            assert np.array_equal(rows[at], want)
        assert not rows[which == 0].any()
        skinned += int((which > 0).sum())
    # Size: one 16-byte row per skinned splat, nothing for the rest; the table is small.
    assert len(blob) == 16 * skinned == 16 * document["weights"]["rows"]
    assert len((tmp_path / "skin.json").read_bytes()) < 400 * len(tiles) + 2500 * 3
    for s in document["skins"]:
        assert len(s["eigenvalues"]) == len(s["support"]) == s["handles"] - 1
        m = s["handles"]
        assert (
            len(s["dynamics"]["mass"]) == len(s["dynamics"]["anchor"]["gram"]) == m * (m + 1) // 2
        )


def test_dynamics_grams_are_the_weights_moments_and_the_anchor_is_the_base():
    skin = tree_skin()
    points = tree_points()
    m = skin.handles
    w = np.c_[np.ones(len(points)), skin.weights(points)]
    assert skin.mass.shape == skin.anchor_gram.shape == (m, m)
    # M is the Gram over the fit points (the unique splat centres), w_0 = 1 on the diagonal.
    unique = np.unique(points, axis=0)
    wu = np.c_[np.ones(len(unique)), skin.weights(unique)]
    assert np.allclose(skin.mass, wu.T @ wu / len(wu), atol=1e-9)
    assert skin.mass[0, 0] == pytest.approx(1.0)
    assert np.allclose(skin.mass, skin.mass.T) and np.allclose(skin.anchor_gram, skin.anchor_gram.T)
    # Positive definite: the handles are independent fields.
    assert np.linalg.eigvalsh(skin.mass).min() > 0
    # The anchor is the lowest band: a tenth of the height above the lowest splat.
    mask, band = skin_scene.anchor_mask(unique)
    height = np.ptp(unique[:, 2])
    assert band >= 0.1 * height - 1e-12 and band == pytest.approx(skin.anchor_band)
    assert unique[mask, 2].max() <= unique[:, 2].min() + band + 1e-12
    assert skin.anchor_splats == int(mask.sum()) > 0
    # Some handle directions leave the base still while moving the crown: G y = μ M y, √μ small.
    mu = np.linalg.eigvals(np.linalg.solve(skin.mass, skin.anchor_gram)).real
    assert (np.sqrt(np.clip(mu, 0, None)) < 0.05).sum() >= 3
    assert w.shape[1] == m


def test_the_committed_yard_skin_is_current():
    committed = json.loads((YARD / "skin" / "skin.json").read_text(encoding="utf-8"))
    blob = (YARD / "skin" / "skin.bin").read_bytes()
    built = skin_scene.build(yard_tiles(), yard_instances()["instances"], only=FIXTURE_ONLY)
    assert committed["tiles"] == built.document["tiles"]
    loose = ("eigenvalues", "support", "dynamics")
    for a, b in zip(committed["skins"], built.document["skins"], strict=True):
        assert {k: v for k, v in a.items() if k not in loose} == {
            k: v for k, v in b.items() if k not in loose
        }
        assert np.allclose(a["eigenvalues"], b["eigenvalues"], rtol=1e-4)
        da, db = a["dynamics"], b["dynamics"]
        assert np.allclose(da["mass"], db["mass"], rtol=1e-3, atol=1e-5)
        assert np.allclose(da["anchor"]["gram"], db["anchor"]["gram"], rtol=1e-3, atol=1e-5)
        assert da["anchor"]["splats"] == db["anchor"]["splats"]
    # The last bit of a weight may differ between BLAS kernels; never more than one step.
    fresh = np.frombuffer(built.blob, np.int8).astype(int)
    assert len(fresh) == len(blob)
    assert np.abs(fresh - np.frombuffer(blob, np.int8)).max() <= 1
