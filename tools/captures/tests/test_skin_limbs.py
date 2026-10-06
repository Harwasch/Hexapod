"""The limbs skin (skin_methods.fit_limbs_from_rig, docs/SCENE_OBJECTS.md §9 "Limbs"): today's
rig and motion sidecar read as one handle per limb.

The Minnetonka tree's extracted rig (packages/world/fixtures/minnetonka, CC BY 4.0) is the real
case -- a zigzagging skeleton of 200 joints and 24 limbs -- with its sidecar derived here as
the TypeScript tests derive it (`deriveMotionSidecar(rig, {treeHeightM: 6})`); a straight
two-limb rig is the case where the conversion is near exact. What must hold: one handle per
oscillator, trunk first, each with its rig's numbers and the handle it hangs from; weights
that are the rig's bend profile (zero at the pivot, rising along the limb, nothing on limbs
not on a splat's chain); a displacement within a few per cent of today's rig, linearised, in
magnitude; the flutter share in the row's last byte; and the per-limb record the TypeScript
driver is held to (`packages/world/fixtures/minnetonka/limbs.json`, `limbWind.test.ts`).
"""

import json
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import motion_params
import skin_methods
import skin_scene
import skin_variants

ROOT = Path(__file__).resolve().parents[3]
MINNETONKA = ROOT / "packages" / "world" / "fixtures" / "minnetonka"
#: The tree's measured height, as living.test.ts reads the rig (not a tuned number).
MINNETONKA_HEIGHT_M = 6.0
#: Per-limb fields the TypeScript driver's `limbHandlesFromRig` is held to.
TWIN_FIELDS = (
    "key", "pivot", "parent", "level", "spanM", "frequencyHz", "damping", "tree", "direction",
    "samplePoint", "widthM", "heightM", "staticTipM", "flutterM",
)  # fmt: skip


@cache
def minnetonka() -> tuple[dict, dict]:
    rig = json.loads((MINNETONKA / "rig.json").read_text(encoding="utf-8"))
    motion = motion_params.derive_sidecar(rig, tree_height_m=MINNETONKA_HEIGHT_M)
    return rig, motion


def straight_rig() -> tuple[dict, dict]:
    """A trunk 4 m straight up and one limb 2 m straight out from its third joint."""
    nodes = [{"id": "root", "parent": -1, "position": [0.0, 0.0, 0.0], "band": "trunk"}]
    for k in range(1, 5):
        nodes.append({"id": f"t{k}", "parent": k - 1, "position": [0.0, 0.0, float(k)],
                      "band": "trunk"})  # fmt: skip
    nodes.append({"id": "t5", "parent": 4, "position": [0.0, 0.0, 5.0], "band": "branch"})
    for k in range(1, 5):
        parent = 3 if k == 1 else len(nodes) - 1
        nodes.append({"id": f"l{k}", "parent": parent, "position": [0.5 * k, 0.0, 3.0],
                      "band": "branch"})  # fmt: skip
    rig = {"canonicalChecksum": "fnv1a32:0:0", "nodes": nodes}
    return rig, motion_params.derive_sidecar(rig, tree_height_m=5.0)


def probes(rig: dict, seed: int = 0, jitter: float = 0.03) -> np.ndarray:
    """Points along every segment of the rig (a quarter, half, three quarters and the joint),
    jittered: where a capture's splats sit."""
    rng = np.random.default_rng(seed)
    joints = np.array([n["position"] for n in rig["nodes"]], np.float64)
    out = []
    for i, node in enumerate(rig["nodes"]):
        if node["parent"] < 0:
            continue
        a, b = joints[node["parent"]], joints[i]
        for f in (0.25, 0.5, 0.75, 1.0):
            out.append(a + f * (b - a))
    points = np.array(out)
    return points + rng.normal(scale=jitter, size=points.shape)


def linearised_today(limbs: skin_methods.LimbRig, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Today's displacement to first order: `Σ_b θ_b × V_b(x)` (living.ts's chain of hinges
    over the splat's four Shepard joints)."""
    joints, w = skin_methods.shepard_binding(limbs.joints, x)
    gains = np.einsum("ns,nsl->nl", w, limbs.gains[joints])
    sums = np.einsum("ns,nslc->nlc", w, limbs.pivot_sums[joints])
    v = gains[:, :, None] * x[:, None, :] - sums
    return np.cross(theta[None], v).sum(1)


def skin_displacement(skin: skin_scene.Skin, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The skin's displacement under the same bends, through its int8 rows and gains:
    `Σ_j w_j gain_j θ_j × (x − p_j)`."""
    learned = skin.weights(x)
    rows = skin_scene.dequantise(skin_scene.quantise(learned, 32), learned.shape[1])
    handles = skin.extra["limbs"]["handles"]
    gain = np.array([h["gain"] for h in handles])
    pivots = np.array([h["pivot"] for h in handles])
    arm = x[:, None, :] - pivots[None]
    return (rows[:, :, None] * gain[None, :, None] * np.cross(theta[None], arm)).sum(1)


def random_bends(limbs: skin_methods.LimbRig, rng: np.random.Generator) -> np.ndarray:
    """Bends across each limb, as the wind gives them."""
    theta = rng.normal(size=(len(limbs.oscillators), 3))
    d = limbs.directions
    return theta - (theta * d).sum(1, keepdims=True) * d


# ------------------------------------------------------------------------------ binding


def test_shepard_binding_is_the_rigs_four_nearest_joints():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0.0]])
    joints, w = skin_methods.shepard_binding(nodes, np.array([[1.0, 0, 0], [1.25, 0.1, 0]]))
    # On a joint: entirely that joint's.
    assert joints[0, 0] == 1 and w[0, 0] == 1.0 and w[0, 1:].sum() == 0
    # Off one: the four nearest, nearest first, weights in 1023ths summing to one.
    assert sorted(joints[1].tolist()) == [0, 1, 2, 3]
    assert joints[1, 0] == 1
    assert w[1].sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(w[1] * 1023, np.round(w[1] * 1023))
    # (1/d − 1/R)², R the fifth's distance: the joint at R has none.
    d = np.linalg.norm(nodes[joints[1]] - [1.25, 0.1, 0], axis=1)
    radius = np.linalg.norm(nodes[4] - [1.25, 0.1, 0])
    raw = (1 / d - 1 / radius) ** 2
    assert np.allclose(w[1], raw / raw.sum(), atol=1.5 / 1023)


# ------------------------------------------------------------------------------ handles


def test_one_handle_per_limb_trunk_first_with_the_rigs_numbers():
    rig, motion = minnetonka()
    limbs = skin_methods.limb_rig(rig, motion)
    structure = motion_params.branch_structure(rig)
    columns = motion["nodes"]
    assert len(limbs.oscillators) == 24
    assert limbs.oscillators[0] == structure["tree_branch"]
    handles = limbs.handles
    assert handles[0]["tree"] and handles[0]["parent"] == 0 and handles[0]["level"] == 0
    for j, h in enumerate(handles, start=1):
        b = h["key"]
        assert h["frequencyHz"] == columns["frequencyHz"][b]
        assert h["damping"] == columns["damping"][b]
        # The pivot is the joint the limb hangs from; the handle it hangs from carries it.
        attach = rig["nodes"][b]["parent"]
        assert h["pivot"] == pytest.approx(rig["nodes"][attach]["position"], abs=1e-6)
        assert 0 <= h["parent"] < j
        if h["parent"] > 0:
            assert columns["branch"][attach] == handles[h["parent"] - 1]["key"]
            assert h["level"] > handles[h["parent"] - 1]["level"] or h["parent"] == 1
        assert np.linalg.norm(h["direction"]) == pytest.approx(1.0, abs=1e-5)
    # A limb's frequency is its span's (Coder 2000), never below the tree's.
    for h in handles[1:]:
        assert h["frequencyHz"] == pytest.approx(
            max(handles[0]["frequencyHz"], motion_params.branch_frequency_hz(h["spanM"])),
            abs=1e-3,
        )


def test_a_forest_rig_or_too_many_limbs_is_refused():
    rig, motion = minnetonka()
    with pytest.raises(ValueError, match="forest"):
        skin_methods.limb_rig(rig, {**motion, "plants": [{"id": 1}]})


# ------------------------------------------------------------------------------ weights


def test_weights_follow_the_bend_profile_and_only_the_chain():
    rig, motion = straight_rig()
    limbs = skin_methods.limb_rig(rig, motion)
    assert len(limbs.oscillators) == 2  # trunk, limb
    # Along the limb, on its axis: no weight at its pivot, rising as s² / s = s (uniform
    # curvature: displacement ∝ s², the lever s), as the rig's gains spread its bend by length.
    s = np.linspace(0.0, 2.0, 9)
    on_limb = np.c_[s, np.zeros_like(s), np.full_like(s, 3.0)]
    raw, _ = skin_methods.limb_weights(limbs, on_limb)
    assert raw[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(raw[:, 1]) > 0)
    # A splat on a trunk joint above the limb: the trunk only (the limb is not on its chain).
    up = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    raw_up, _ = skin_methods.limb_weights(limbs, up)
    assert np.all(raw_up[:, 1] == 0) and np.all(raw_up[:, 0] > 0)
    # Between them, only what the rig's own Shepard blend lets in from the limb's joints.
    between = np.c_[np.zeros(5), np.zeros(5), np.linspace(3.6, 4.9, 5)]
    raw_between, _ = skin_methods.limb_weights(limbs, between)
    assert np.abs(raw_between[:, 1]).max() < 0.05 * raw[:, 1].max()
    # Below the first trunk joint's reach the trunk barely moves.
    raw_base, _ = skin_methods.limb_weights(limbs, np.array([[0.0, 0.0, 0.05]]))
    assert abs(raw_base[0, 0]) < 0.05 * raw_up[:, 0].max()


def test_the_skin_moves_as_the_rig_does_linearised():
    """On the straight rig nearly exactly; on the Minnetonka rig's zigzag the scalar weight per
    limb cannot follow the effective pivot off the line to the splat, so the vector error is
    larger (10-20%, 8% on the straight rig, where the trunk's pivots are off the line to a limb's
    splats) -- but the size of the displacement stays within a few per cent."""
    rng = np.random.default_rng(5)
    for (rig, motion), size_tol, vector_tol in (
        (straight_rig(), 0.05, 0.12),
        (minnetonka(), 0.05, 0.25),
    ):
        x = probes(rig)
        skin = skin_methods.fit_limbs_from_rig(x, 1, 1, rig=rig, motion=motion)
        limbs = skin_methods.limb_rig(rig, motion)
        for _ in range(3):
            theta = random_bends(limbs, rng)
            today = linearised_today(limbs, x, theta)
            ours = skin_displacement(skin, x, theta)
            rms_today = np.sqrt((today**2).sum(1).mean())
            rms_ours = np.sqrt((ours**2).sum(1).mean())
            error = np.sqrt(((ours - today) ** 2).sum(1).mean())
            assert abs(rms_ours / rms_today - 1) < size_tol
            assert error / rms_today < vector_tol


def test_a_limbs_skin_builds_with_its_block_and_flutter_byte(tmp_path):
    rig, motion = minnetonka()
    x = probes(rig, jitter=0.05).astype(np.float32)
    checksum = "fnv1a32:test"
    tile = skin_scene.TileSplats("t.glb", checksum, x, np.ones(len(x), np.int64), True)
    instances = [{"id": 1, "parent": None, "behaviour": "in-place", "category": "trees",
                  "properties": {"vegetation": 1.0, "elastic": 0.7, "rigid": 0.2},
                  "tags": [{"label": "tree"}]}]  # fmt: skip
    built = skin_scene.build(
        [tile],
        instances,
        owner=skin_scene.owners_of(instances, [1]),
        fit=skin_methods.limbs_fitter(rig, motion, instances),
        method=skin_methods.LIMBS_METHOD,
    )
    doc = built.document
    assert doc["method"]["name"] == "limbs"
    assert doc["weights"]["rowBytes"] == 32  # 25 handles: two texels a splat
    entry = doc["skins"][0]
    assert entry["handles"] == 25
    block = entry["limbs"]
    assert len(block["handles"]) == 24
    assert block["seed"] == motion["seed"]
    assert block["wind"]["lengthScaleM"] == pytest.approx(motion["wind"]["lengthScaleM"])
    assert block["flutter"]["referenceM"] == pytest.approx(0.006)
    assert all(h["gain"] > 0 for h in block["handles"])
    # The poke's: dynamics and an eigenvalue per handle (each rings alone at its frequency).
    assert len(entry["eigenvalues"]) == 24 and "dynamics" in entry
    skin_scene.write_skin(tmp_path, built)
    which, rows = skin_scene.decode_tiles(doc, (tmp_path / "skin.bin").read_bytes())[checksum]
    assert (which == 1).all()
    # Weights: bytes 0..23; nothing in 24..30; the flutter share in byte 31.
    assert np.abs(rows[:, :24]).max() == 127
    assert (rows[:, 24:31] == 0).all()
    _, leaf = skin_methods.limb_weights(skin_methods.limb_rig(rig, motion), x.astype(np.float64))
    assert np.array_equal(rows[:, 31], np.round(np.clip(leaf, 0, 1) * 127).astype(np.int8))
    assert rows[:, 31].max() > 100 and rows[:, 31].min() >= 0
    # Older skins' rows keep their last byte 0.
    plain = skin_scene.build([tile], instances, owner=skin_scene.owners_of(instances, [1]),
                             fit=skin_methods.fitter("freeform", skin_methods.size_policy(),
                                                     instances))  # fmt: skip
    plain_rows = np.frombuffer(plain.blob, np.int8).reshape(
        -1, plain.document["weights"]["rowBytes"]
    )
    assert (plain_rows[:, -1] == 0).all()


# ------------------------------------------------------------------------------ bake-off


def test_limbs_today_is_only_for_a_scan_with_a_rig():
    names = ["freeform", "limbs-today"]
    assert skin_variants.variants_for(skin_variants.BAKEOFF["camp"], names) == ["freeform"]
    tree = skin_variants.BAKEOFF["minnetonka-tree"]
    assert skin_variants.variants_for(tree, names) == names
    assert tree["rig"] == "../source/rig.json"
    assert skin_variants.VARIANTS_BY_NAME["limbs-today"].needs_rig


def test_fetch_brings_the_rig_and_the_motion_it_names(tmp_path, monkeypatch):
    base = "https://example.test/sites/tree"
    served = {
        f"{base}/splat/tileset.json": json.dumps({"root": {"children": []}}).encode(),
        f"{base}/source/rig.json": json.dumps({"nodes": [], "motion": "motion.json"}).encode(),
        f"{base}/source/motion.json": b'{"format": "hexapod.motion"}',
    }
    monkeypatch.setattr(skin_variants, "_get", lambda url: served[url])
    skin_variants.fetch(f"{base}/splat/tileset.json", tmp_path, rig="../source/rig.json")
    assert json.loads((tmp_path / "rig.json").read_text())["motion"] == "motion.json"
    assert json.loads((tmp_path / "motion.json").read_text())["format"] == "hexapod.motion"


def test_build_variants_writes_a_limbs_skin_from_the_scans_rig(tmp_path, monkeypatch):
    rig, motion = minnetonka()
    (tmp_path / "rig.json").write_text(json.dumps(rig), encoding="utf-8")
    (tmp_path / "motion.json").write_text(json.dumps(motion), encoding="utf-8")
    x = probes(rig).astype(np.float32)
    tile = skin_scene.TileSplats("t.glb", "fnv1a32:t", x, np.ones(len(x), np.int64), True)
    doc = {"instances": [{"id": 1, "parent": None, "behaviour": "in-place", "category": "trees",
                          "properties": {"vegetation": 1.0}, "tags": [{"label": "tree"}],
                          "bounds": {"min": [0, 0, 0], "max": [1, 1, 6]}}]}  # fmt: skip
    monkeypatch.setattr(skin_variants, "read_scan", lambda *a, **k: ([tile], doc, {}))
    entries, report = skin_variants.build_variants(tmp_path, ["limbs-today"], whole=True, log=False)
    assert [e["name"] for e in entries] == ["limbs-today"]
    written = json.loads(
        (tmp_path / "variants" / "skins" / "limbs-today" / "skin.json").read_text()
    )
    assert written["method"]["name"] == "limbs"
    assert "limb" in written["method"]["handlePolicy"]
    row = report["limbs-today"]["skins"][0]
    assert row["handles"] == 25 and row["limbs"] == 24 and row["class"] == "tree"
    assert row["lowestHz"][0] == pytest.approx(0.98, abs=0.01)


def test_a_scan_none_of_the_asked_variants_fits_is_skipped(tmp_path, monkeypatch):
    import attach_sidecars

    monkeypatch.setattr(
        skin_variants,
        "locate",
        lambda name: {"kind": "attach", "url": "https://x.test/t.json", "assetId": "a"},
    )

    def refuse(*args, **kwargs):
        raise AssertionError("nothing may be attached")

    monkeypatch.setattr(attach_sidecars, "attach", refuse)
    out = skin_variants.prepare_scan("camp", tmp_path, ["limbs-today"])
    plan = json.loads((out.parent / "camp.publish.json").read_text())
    assert plan["variants"] == {"skins": []} and "plant rig" in plan["skip"]
    assert skin_variants.publish(out)["skipped"] == plan["skip"]


# ------------------------------------------------------------------------------ the twin


def limb_fixture() -> dict:
    rig, motion = minnetonka()
    limbs = skin_methods.limb_rig(rig, motion)
    return {
        "about": "skin_methods.limb_rig of packages/world/fixtures/minnetonka/rig.json with "
        "motion_params.derive_sidecar(rig, tree_height_m=6): what limbWind.test.ts holds "
        "limbHandlesFromRig to (tools/captures/tests/test_skin_limbs.py writes it with "
        "UPDATE_LIMB_FIXTURE=1)",
        "handles": [{k: h[k] for k in TWIN_FIELDS} for h in limbs.handles],
    }


def test_the_committed_limb_fixture_is_current():
    path = MINNETONKA / "limbs.json"
    fresh = limb_fixture()
    if os.environ.get("UPDATE_LIMB_FIXTURE") == "1":
        path.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(fresh))
