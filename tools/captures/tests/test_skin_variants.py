"""The motion-skins bake-off (skin_methods.py, skin_variants.py): methods, handle policies, the
variant files, and how they register beside the tiles.

The yard is the format case (tiles, instances, the committed `synthetic-yard/skin-wide/`
fixture: the stiffness rule's skin, the tree and the snag at 32 handles); the synthetic tree is
the deformation case.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import skin_methods
import skin_scene
import skin_variants
import skin_wind as sw

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "positions.f32"
YARD = ROOT / "data" / "tiles" / "synthetic-yard"
YARD_OBJECTS = (1, 9, 10, 12)


@cache
def tree_points() -> np.ndarray:
    return np.frombuffer(TREE.read_bytes(), "<f4").reshape(-1, 3).astype(np.float64)


@cache
def yard_instances() -> dict:
    return json.loads((YARD / "instances" / "instances.json").read_text(encoding="utf-8"))


@cache
def yard_tiles() -> tuple[skin_scene.TileSplats, ...]:
    return tuple(skin_scene.read_tiles(YARD / "splat", yard_instances()))


@cache
def yard_materials() -> dict[int, dict]:
    doc = json.loads((YARD / "skin" / "materials.json").read_text(encoding="utf-8"))
    return {int(r["instance"]): r for r in doc["materials"]}


def anchored_share(skin: skin_scene.Skin) -> tuple[int, int]:
    """How many handle-space directions keep the base within the wind's tolerance, of all."""
    mu = np.linalg.eigvals(np.linalg.solve(skin.mass, skin.anchor_gram)).real
    return int((np.sqrt(np.clip(mu, 0, None)) <= sw.ANCHOR_TOLERANCE).sum()), skin.handles


# ------------------------------------------------------------------------------ policies


def instance(**fields) -> dict:
    base = {"id": 1, "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}, "properties": {}}
    return {**base, **fields}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (instance(category="produce", tags=[{"label": "pumpkin"}]), "firm"),
        # The spool: a fixture, rigid, whatever its names say.
        (instance(category="fixtures", tags=[{"label": "manhole"}]), "rigid"),
        (instance(category="shrubs", tags=[{"label": "bush"}]), "plant"),
        (
            instance(category="trees", bounds={"min": [0, 0, 0], "max": [3, 3, 9]}),
            "tree",
        ),
        # No category: names, then the property scores.
        (instance(tags=[{"label": "boulder"}]), "rigid"),
        (instance(properties={"vegetation": 0.9, "rigid": 0.2}), "plant"),
        (instance(properties={"rigid": 0.95, "vegetation": 0.1, "elastic": 0.1}), "rigid"),
        (instance(properties={"rigid": 0.5, "vegetation": 0.3}), "firm"),
    ],
)
def test_stiffness_class_reads_category_names_then_properties(record, expected):
    assert skin_methods.stiffness_class(record) == expected


def test_a_fitted_wave_speed_decides_the_class():
    record = instance(category="trees")
    assert skin_methods.stiffness_class(record, {"stiffness": 3.5}) == "plant"
    assert skin_methods.stiffness_class(record, {"stiffness": 12}) == "firm"
    assert skin_methods.stiffness_class(record, {"stiffness": 40}) == "rigid"


def test_the_stiffness_rule_gives_rigid_one_firm_four_plants_by_size_trees_thirty_two():
    policy = skin_methods.stiffness_policy()
    points = np.random.default_rng(0).uniform(0, 2, (100, 3))
    assert policy.handles(instance(category="fixtures"), points) == (1, "rigid")
    assert policy.handles(instance(category="produce"), points) == (4, "firm")
    size = skin_scene.handle_count(float(np.linalg.norm(np.ptp(points, 0))))
    assert policy.handles(instance(category="shrubs"), points) == (size, "plant")
    tall = instance(category="trees", bounds={"min": [0, 0, 0], "max": [3, 3, 9]})
    assert policy.handles(tall, points) == (32, "tree")
    narrow = skin_methods.stiffness_policy(wide=False)
    assert narrow.handles(tall, points) == (16, "tree")
    # Today's rule: size alone, whatever the object is.
    assert skin_methods.size_policy().handles(instance(category="fixtures"), points)[0] == size


# ------------------------------------------------------------------------------- methods


def test_pinned_modes_each_leave_the_base_still():
    points = tree_points()
    free = skin_scene.fit_skin(points, 1, 1, handles=12)
    pinned = skin_methods.fit_pinned(points, 1, 1, handles=12)
    kept_free, m = anchored_share(free)
    kept_pinned, _ = anchored_share(pinned)
    # Every learned handle of the pinned basis is a direction the wind keeps (the constant
    # handle alone moves the base); the free basis keeps fewer.
    assert kept_pinned == m - 1
    assert kept_free < kept_pinned
    # The base barely moves under any learned handle.
    mask, _ = skin_scene.anchor_mask(points)
    w = pinned.weights(points)
    assert np.abs(w[mask]).max() < 0.1 * np.abs(w).max()
    assert np.all(np.diff(pinned.eigenvalues) >= -1e-9) and pinned.eigenvalues[0] > 0


def test_tet_fem_fills_the_occupancy_and_its_weights_are_smooth():
    points = tree_points()
    skin = skin_methods.fit_tetfem(points, 1, 1, handles=10)
    assert skin.handles == 10
    mesh = skin.extra["mesh"]
    assert mesh["filled"] and mesh["tets"] == 6 * mesh["voxels"]
    w = skin.weights(points)
    assert np.isfinite(w).all() and np.abs(w).max() == pytest.approx(1.0)
    # Small random handles bend it without tearing (kNN edge stretch) or folding.
    rng = np.random.default_rng(3)
    z = skin_scene.random_handles(rng, skin.handles, skin.scale)
    z = skin_scene.scaled_to(points, skin.origin, w, z, 0.02 * skin.scale)
    moved = skin_scene.deform(points, skin.origin, w, z)
    stretch = skin_scene.edge_stretch(points, moved)
    assert np.percentile(stretch, 99) < 1.05
    # Its eigenvalues are on the same scale as the RKPM basis's (unit box, E = 1): the first
    # within a factor of a few of the pinned FreeForm's (both hold the base) on the same shape.
    pinned = skin_methods.fit_pinned(points, 1, 1, handles=10)
    assert 0.2 < skin.eigenvalues[0] / pinned.eigenvalues[0] < 5
    # Its base is held: every learned handle is a direction the wind keeps.
    assert anchored_share(skin)[0] == skin.handles - 1
    # Its mass is integrated over the volume: still a positive definite Gram with w_0 = 1.
    assert skin.mass[0, 0] == pytest.approx(1.0)
    assert np.linalg.eigvalsh(skin.mass).min() > 0
    assert skin.extra["dynamicsOver"] == "volume"


def test_a_rigid_skin_has_one_handle_and_takes_no_rows():
    tiles = yard_tiles()
    instances = yard_instances()["instances"]
    owner = skin_scene.owners_of(instances, [10, 12])

    def fit(points, index, inst, *, seed=0):
        if inst == 12:
            return skin_methods.fit_rigid(points, index, inst)
        return skin_scene.fit_skin(points, index, inst, seed=seed)

    built = skin_scene.build(tiles, instances, owner=owner, fit=fit)
    rigid = next(s for s in built.document["skins"] if s["instance"] == 12)
    assert rigid["handles"] == 1 and rigid["eigenvalues"] == [] and rigid["support"] == []
    decoded = skin_scene.decode_tiles(built.document, built.blob)
    rows = 0
    for tile in tiles:
        which, block = decoded[tile.checksum]
        rows += int((which == 1).sum())  # skin 1 is instance 10, the only one with rows
        assert not block[which == 2].any()
    assert built.document["weights"]["rows"] == rows
    assert len(built.blob) == 16 * rows


def test_owners_of_takes_the_chosen_objects_whatever_their_behaviour():
    instances = [
        {"id": 1, "parent": None, "behaviour": "static"},
        {"id": 2, "parent": 1, "behaviour": "static"},
        {"id": 3, "parent": None, "behaviour": "in-place"},
        {"id": 4, "parent": 3, "behaviour": "in-place"},
    ]
    assert skin_scene.owners_of(instances, [1]).tolist() == [0, 1, 1, 0, 0]
    assert skin_scene.owners_of(instances, [4, 3]).tolist() == [0, 0, 0, 3, 3]


# ------------------------------------------------------------------------------ variants


def test_every_variant_builds_on_the_yard_and_reads_back(tmp_path):
    tiles = yard_tiles()
    doc = yard_instances()
    by_id = {int(i["id"]): i for i in doc["instances"]}
    for variant in skin_variants.VARIANTS:
        policy = skin_variants.policy_of(variant.policy, yard_materials(), wide=True)
        built = skin_scene.build(
            tiles,
            doc["instances"],
            owner=skin_scene.owners_of(doc["instances"], YARD_OBJECTS),
            fit=skin_methods.fitter(variant.method, policy, doc["instances"]),
            method=skin_variants.METHOD_DOCS[variant.method],
            listed="skinned",
        )
        document = built.document
        assert [s["instance"] for s in document["skins"]] == list(YARD_OBJECTS)
        wide = any(s["handles"] > 16 for s in document["skins"])
        assert document["weights"]["rowBytes"] == (32 if wide else 16)
        assert len(built.blob) == document["weights"]["rowBytes"] * document["weights"]["rows"]
        for entry in document["skins"]:
            assert entry["traits"]["properties"]["vegetation"] == pytest.approx(
                by_id[entry["instance"]]["properties"]["vegetation"], abs=1e-3
            )
            m = entry["handles"]
            assert len(entry["eigenvalues"]) == len(entry["support"]) == m - 1
            assert len(entry["dynamics"]["mass"]) == m * (m + 1) // 2
            # The wind can drive every skin of a plant: it keeps some anchored modes.
            model = sw.skin_wind_model(
                sw.SkinDynamics.from_skin(entry), sw.material_prior({"vegetation": 1}, "in-place")
            )
            assert model is not None and model.modes > 0
        # Only tiles with a skinned splat are listed, and every listed tile round-trips.
        decoded = skin_scene.decode_tiles(document, built.blob)
        owner = skin_scene.owners_of(doc["instances"], YARD_OBJECTS)
        assert set(decoded) == {t.checksum for t in tiles if (owner[t.ids] > 0).any()}
        assert all((which > 0).any() for which, _ in decoded.values())
        assert built.clipped <= 2


def test_the_committed_wide_yard_skin_is_current():
    """data/tiles/synthetic-yard/skin-wide/: `skin_variants.py build ... --variants
    freeform-stiff --flat` on the yard (see the e2e poke spec)."""
    committed = json.loads((YARD / "skin-wide" / "skin.json").read_text(encoding="utf-8"))
    blob = (YARD / "skin-wide" / "skin.bin").read_bytes()
    doc = yard_instances()
    policy = skin_variants.policy_of("stiffness", yard_materials(), wide=True)
    built = skin_scene.build(
        yard_tiles(),
        doc["instances"],
        owner=skin_scene.owners_of(doc["instances"], YARD_OBJECTS),
        fit=skin_methods.fitter("freeform", policy, doc["instances"]),
        method={**skin_variants.METHOD_DOCS["freeform"], "handlePolicy": policy.about},
        listed="skinned",
    )
    assert committed["weights"] == built.document["weights"]
    assert committed["tiles"] == built.document["tiles"]
    assert [s["handles"] for s in committed["skins"]] == [32, 32, 9, 8]
    loose = ("eigenvalues", "support", "dynamics")
    for a, b in zip(committed["skins"], built.document["skins"], strict=True):
        assert {k: v for k, v in a.items() if k not in loose} == {
            k: v for k, v in b.items() if k not in loose
        }
        assert np.allclose(a["eigenvalues"], b["eigenvalues"], rtol=1e-4)
    fresh = np.frombuffer(built.blob, np.int8).astype(int)
    assert len(fresh) == len(blob)
    assert np.abs(fresh - np.frombuffer(blob, np.int8)).max() <= 1


def test_registering_replaces_by_name_and_keeps_every_other_system():
    extras = {
        "instances": {"uri": "instances.json", "count": 3},
        "variants": {
            "objects": [{"name": "ground-first", "instances": "variants/objects/g/instances.json"}],
            "skins": [
                {"name": "freeform", "skin": "old.json", "label": "old"},
                {"name": "someone-else", "skin": "x.json"},
            ],
        },
    }
    mine = {
        "skins": [
            {"name": "freeform", "skin": "variants/skins/freeform/skin.json", "label": "new"},
            {"name": "tetfem-stiff", "skin": "variants/skins/tetfem-stiff/skin.json"},
        ]
    }
    merged = skin_variants.merged_variants(extras, mine)
    assert merged["objects"] == extras["variants"]["objects"]
    assert [e["name"] for e in merged["skins"]] == ["freeform", "someone-else", "tetfem-stiff"]
    assert merged["skins"][0]["label"] == "new"
    # Nothing there before: just mine.
    assert skin_variants.merged_variants({}, mine) == {"skins": mine["skins"]}
    # Withdrawn: only the named entries go; a system left empty goes with them.
    mine_names = {"skins": ["freeform", "tetfem-stiff"]}
    left = skin_variants.merged_variants(extras, {"skins": []}, mine_names)
    assert left == {
        "objects": extras["variants"]["objects"],
        "skins": [{"name": "someone-else", "skin": "x.json"}],
    }
    only_mine = {"variants": {"skins": [{"name": "freeform", "skin": "a.json"}]}}
    assert skin_variants.merged_variants(only_mine, {"skins": []}, mine_names) == {}


def test_every_scan_kept_says_what_to_look_at_and_the_spool_is_withdrawn():
    names = [v.name for v in skin_variants.VARIANTS]
    for scan, spec in skin_variants.BAKEOFF.items():
        if spec.get("withdrawn"):
            assert "look" not in spec
            continue
        assert sorted(spec["look"]) == sorted(names), scan
        for sentence in spec["look"].values():
            assert sentence.endswith(".") and sentence.count(". ") == 0, sentence
    assert skin_variants.BAKEOFF["spool"]["withdrawn"]


# ------------------------------------------------------------------------------- publish


class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class _Bucket:
    """A stand-in for boto3's S3 client: objects by (bucket, key)."""

    def __init__(self, objects: dict) -> None:
        self.objects = objects

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = (Path(path).read_bytes(), ExtraArgs["ContentType"])

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.objects[(Bucket, Key)][0])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = (Body, ContentType)


def lay_out(tmp_path: Path, name: str, plan: dict) -> Path:
    out = tmp_path / "out" / name
    for variant in ("freeform", "tetfem-stiff"):
        folder = out / "variants" / "skins" / variant
        folder.mkdir(parents=True)
        (folder / "skin.json").write_text("{}", encoding="utf-8")
        (folder / "skin.bin").write_bytes(b"\0" * 16)
    entries = [
        {"name": v, "label": v, "about": "", "skin": f"variants/skins/{v}/skin.json"}
        for v in ("freeform", "tetfem-stiff")
    ]
    (out.parent / f"{name}.publish.json").write_text(
        json.dumps({**plan, "scan": name, "variants": {"skins": entries}}), encoding="utf-8"
    )
    return out


def test_a_site_gets_its_files_and_only_its_variants_declared(tmp_path, monkeypatch):
    import sys
    import types

    url = f"{skin_variants.PUBLIC_BASE}/sites/minnetonka-tree/splat/tileset.json"
    out = lay_out(tmp_path, "minnetonka-tree", {"kind": "site", "url": url})
    key = "sites/minnetonka-tree/splat/tileset.json"
    tileset = {
        "asset": {"version": "1.1"},
        "root": {"extras": {"gaussians": 5, "variants": {"objects": [{"name": "x"}]}}},
    }
    objects = {("public", key): (json.dumps(tileset).encode(), "application/json")}
    bucket = _Bucket(objects)
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: bucket))
    for name in (
        "OBJECT_STORAGE_ENDPOINT_URL",
        "OBJECT_STORAGE_ACCESS_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
    ):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("OBJECT_STORAGE_PUBLIC_BUCKET", "public")
    skin_variants.publish(out)
    keys = {k for _, k in objects}
    assert "sites/minnetonka-tree/splat/variants/skins/tetfem-stiff/skin.bin" in keys
    written = json.loads(objects[("public", key)][0])
    assert written["asset"] == tileset["asset"]
    assert written["root"]["extras"]["gaussians"] == 5
    variants = written["root"]["extras"]["variants"]
    assert variants["objects"] == [{"name": "x"}]
    assert [e["name"] for e in variants["skins"]] == ["freeform", "tetfem-stiff"]


def test_a_run_scan_attaches_with_the_variants_merged_at_publish_time(tmp_path, monkeypatch):
    import attach_sidecars

    asset = "29ad7e37-9ad1-42a8-a61d-a39e08ac6710"
    built_on = "https://pub.example/runs/8e1c/package/splat/tileset.json"
    out = lay_out(tmp_path, "spool", {"kind": "attach", "url": built_on, "assetId": asset})
    now = "https://pub.example/runs/8e1c/package/splat/g2/tileset.json"
    current = {"root": {"extras": {"variants": {"skins": [{"name": "other", "skin": "o.json"}]}}}}
    monkeypatch.setattr(attach_sidecars, "resolve_asset", lambda a: {"assetId": a, "url": now})
    monkeypatch.setattr(skin_variants, "_get", lambda u: json.dumps(current).encode())
    sent: dict = {}

    def attach(directory):
        sent["dir"] = directory
        return {}

    monkeypatch.setattr(attach_sidecars, "attach", attach)
    skin_variants.publish(out)
    manifest = json.loads((out / "attach.json").read_text(encoding="utf-8"))
    assert sent["dir"] == out
    assert manifest["assetId"] == asset and manifest["basedOn"] == built_on
    assert [e["name"] for e in manifest["extras"]["variants"]["skins"]] == [
        "other",
        "freeform",
        "tetfem-stiff",
    ]
    assert manifest["files"] == sorted(
        f"variants/skins/{v}/skin.{x}"
        for v in ("freeform", "tetfem-stiff")
        for x in ("bin", "json")
    )


def test_a_withdrawn_scan_is_not_fitted_and_its_entries_are_taken_off(tmp_path, monkeypatch):
    import attach_sidecars

    asset = "29ad7e37-9ad1-42a8-a61d-a39e08ac6710"
    built_on = "https://pub.example/runs/8e1c/package/splat/tileset.json"
    where = {"kind": "attach", "url": built_on, "assetId": asset}
    monkeypatch.setattr(skin_variants, "locate", lambda name: where)
    monkeypatch.setattr(skin_variants, "fetch", lambda *a, **k: pytest.fail("fetched"))
    names = [v.name for v in skin_variants.VARIANTS]
    out = skin_variants.prepare_scan("spool", tmp_path, names)
    assert out.is_dir() and not any(out.iterdir())
    plan = json.loads((out.parent / "spool.publish.json").read_text(encoding="utf-8"))
    assert plan["variants"] == {"skins": []}
    assert plan["withdraw"]["skins"] == names
    # The workflow's artifact holds no empty folder: publish makes it again.
    out.rmdir()
    mine = [{"name": n, "skin": f"variants/skins/{n}/skin.json"} for n in names]
    others = {
        "objects": [{"name": "feature-fields", "instances": "variants/objects/ff/i.json"}],
        "fill": [
            {"name": "vace-1-3b", "inferredLayers": [{"uri": "variants/fill/v/t.json"}]},
            {"name": "lama-baseline", "inferredLayers": []},
        ],
    }
    variants = {**others, "skins": [*mine, {"name": "candidate-c", "skin": "c.json"}]}
    current = {"root": {"extras": {"instances": {"uri": "i.json"}, "variants": variants}}}
    monkeypatch.setattr(attach_sidecars, "resolve_asset", lambda a: {"assetId": a, "url": "u"})
    monkeypatch.setattr(skin_variants, "_get", lambda u: json.dumps(current).encode())
    monkeypatch.setattr(attach_sidecars, "attach", lambda directory: {})
    skin_variants.publish(out)
    manifest = json.loads((out / "attach.json").read_text(encoding="utf-8"))
    # Nothing staged; only this tool's entries withdrawn, at the request (attach merges into
    # the tileset as it is then); the preview keeps everyone else's byte for byte.
    assert manifest["files"] == []
    assert manifest["withdraw"] == [{"system": "skins", "name": n} for n in names]
    preview = manifest["extras"]["variants"]
    assert json.dumps(preview["objects"]) == json.dumps(others["objects"])
    assert json.dumps(preview["fill"]) == json.dumps(others["fill"])
    assert preview["skins"] == [{"name": "candidate-c", "skin": "c.json"}]
    # Had only this tool's entries been there, nothing would be left: the key goes.
    alone = {"variants": {"skins": mine}}
    assert skin_variants.merged_variants(alone, {"skins": []}, plan["withdraw"]) == {}


def test_withdrawing_skins_keeps_objects_and_fill_byte_for_byte():
    extras = {
        "instances": {"uri": "instances.json"},
        "variants": {
            "objects": [{"name": "feature-fields", "label": "B · Feature fields", "about": "x"}],
            "fill": [
                {"name": "wan22-5b", "inferredLayers": [{"uri": "a.json", "evidence": {"k": 1}}]},
                {"name": "cosmos-p2-2b", "inferredLayers": []},
            ],
            "skins": [
                {"name": "freeform", "skin": "s.json"},
                {"name": "pinned-stiff", "skin": "p.json"},
            ],
        },
    }
    before = json.dumps(extras, sort_keys=True)
    left = skin_variants.merged_variants(extras, {}, {"skins": ["freeform", "pinned-stiff"]})
    assert json.dumps(extras, sort_keys=True) == before
    assert json.dumps(left["objects"]) == json.dumps(extras["variants"]["objects"])
    assert json.dumps(left["fill"]) == json.dumps(extras["variants"]["fill"])
    assert "skins" not in left and sorted(left) == ["fill", "objects"]
