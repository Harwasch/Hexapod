"""`split_objects` (C4) on the synthetic yard, with the CPU stand-in filler.

The fixture is the yard regenerated with the lawn under one shrub taken away -- what a
capture could never see under a crown -- packed into small tiles (merged parents and all),
with instances from its ground truth (shrubs `movable`, the rest not). Splitting that shrub
out must conserve every gaussian, keep every tile bound, leave the scene byte for byte what
it was elsewhere, put the shrub back exactly where it was when its tileset is placed at rest,
and close the hole it leaves. The committed yard (its own segmentation and skins) checks the
skin re-binding.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import segment_scene as ss
import skin_scene
import splat_tiles
import split_objects as so
import synthetic_yard
from rig_tiles import glb_spz, tile_positions, tile_uris
from splat_render import Camera, load_tileset, render
from synthetic_tree import checksum_positions, write_ply

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard"
#: The shrub at (13.3, 7.1), crown radius 0.95 m: synthetic_yard.SHRUBS[2].
SHRUB_STEM = (13.3, 7.1)
SHRUB_INDEX = 5  # trees 0-2, then the shrubs
SHRUB_ID = SHRUB_INDEX + 1
FOOTPRINT_M = 1.0
SIZE = {"width": 200, "height": 150}


def _fingerprint(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def scan(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return footprint_yard(tmp_path_factory.mktemp("footprint"))


def footprint_yard(root: Path) -> Path:
    """The yard without the lawn under the shrub, packed and bound to truth instances."""
    data, klass, inst, instances = synthetic_yard.generate_yard()
    under = (inst < 0) & (
        np.linalg.norm(data["position"][:, :2] - np.asarray(SHRUB_STEM), axis=1) < FOOTPRINT_M
    )
    data = {k: v[~under] for k, v in data.items()}
    klass, inst = klass[~under], inst[~under]
    ply = root / "splat.ply"
    write_ply(ply, data)
    tiles = root / "splat"
    splat_tiles.convert(ply, tiles, 46.1335, -123.88, 0.0, opacity_min=0.02, tile_gaussians=6000)
    ground = len(instances) + 1
    ids = np.where(inst >= 0, inst + 1, ground)
    behaviour = {"tree": "in-place", "shrub": "movable", "snag": "in-place"}
    rows = []
    for k, info in enumerate([*instances, {"class": "ground"}], start=1):
        mine = data["position"][ids == k]
        rows.append(
            {
                "id": k,
                "parent": None,
                "level": 0,
                "splats": int(mine.shape[0]),
                "bounds": {"min": mine.min(0).tolist(), "max": mine.max(0).tolist()},
                "centroid": mine.mean(0).tolist(),
                "tags": [{"label": info["class"], "score": 1.0}],
                "properties": {"movable": 1.0 if info["class"] == "shrub" else 0.0},
                "behaviour": behaviour.get(info["class"], "static"),
                "views": 1,
            }
        )
    layout = splat_tiles.ply_layout(ply)
    binding = ss.tile_binding(
        tiles,
        ply,
        np.arange(layout.count),
        ids,
        layout.count,
        opacity_min=0.02,
        tile_gaussians=6000,
    )
    doc = {
        "format": ss.FORMAT,
        "version": ss.VERSION,
        "frame": ss.FRAME,
        "embedding": {"file": "instances.emb", "model": "none", "dim": 4, "dtype": "float16"},
        "vocabulary": {"model": "none", "size": 0},
        "instances": rows,
        "tiles": binding,
        "tilesEncoding": ss.TILES_ENCODING,
    }
    (tiles / "instances.json").write_text(json.dumps(doc), encoding="utf-8")
    (tiles / "instances.emb").write_bytes(np.zeros((len(rows), 4), "<f2").tobytes())
    ss.link_instances(tiles / "tileset.json", len(rows))
    return tiles


@pytest.fixture(scope="module")
def result(scan: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[so.Split, dict]:
    before = _fingerprint(scan)
    out = tmp_path_factory.mktemp("split") / "out"
    fill = so.fill_holes("telea", views=4, **SIZE)
    return so.split(scan, out, ids=[SHRUB_ID], fill=fill), before


def _leaves(tileset: Path) -> list[so.Spz]:
    doc = json.loads(tileset.read_text(encoding="utf-8"))
    return [
        so.read_tile(tileset.parent / n["content"]["uri"]) for n, leaf in so._nodes(doc) if leaf
    ]


def _records(spz: so.Spz) -> np.ndarray:
    """One row of bytes per gaussian, sorted: a multiset to compare."""
    rows = np.concatenate(
        [
            spz.fixed.astype("<i8").view(np.uint8).reshape(len(spz), -1),
            spz.alpha[:, None],
            spz.colour,
            spz.scale,
            spz.rotation,
            spz.sh,
        ],
        axis=1,
    )
    return rows[np.lexsort(rows.T[::-1])]


def test_spz_records_round_trip_to_the_same_bytes() -> None:
    tiles = YARD / "splat"
    for uri in tile_uris(json.loads((tiles / "tileset.json").read_text(encoding="utf-8")))[:4]:
        blob = glb_spz(tiles / uri)
        spz = so.read_spz(blob)
        assert so.write_spz(spz) == blob
        assert np.array_equal(spz.positions(), tile_positions(tiles / uri))


def test_default_choice_is_the_movable_objects_of_a_useful_size(scan: Path) -> None:
    doc = json.loads((scan / "instances.json").read_text(encoding="utf-8"))
    chosen = so.choose(doc)
    shrubs = {i["id"] for i in doc["instances"] if i["tags"][0]["label"] == "shrub"}
    big = {i["id"] for i in doc["instances"] if i["id"] in shrubs and i["splats"] >= so.MIN_SPLATS}
    assert {c.instance for c in chosen} == big
    assert so.choose(doc, min_splats=10**9) == []


def test_explicit_ids_drop_what_another_choice_contains() -> None:
    doc = json.loads((YARD / "instances" / "instances.json").read_text(encoding="utf-8"))
    child = next(i["id"] for i in doc["instances"] if i["parent"] == 10)
    chosen = so.choose(doc, [child, 10])
    assert [c.instance for c in chosen] == [10]
    assert child in chosen[0].ids


def _tile(positions: np.ndarray, labels: np.ndarray) -> so.TileRef:
    n = positions.shape[0]
    spz = so.Spz(
        2,
        0,
        12,
        0,
        np.round(positions * 4096).astype(np.int64),
        *(np.zeros((n, k), np.uint8) for k in (1, 3, 3, 3, 0)),
    )
    spz.alpha = spz.alpha[:, 0]
    return so.TileRef({}, "t.glb", True, spz, "", np.asarray(labels))


def test_absorb_takes_fragments_inside_the_object_not_its_neighbours() -> None:
    rng = np.random.default_rng(0)
    body = rng.uniform(0, 1, (1000, 3))  # id 1: the object
    fragment = rng.uniform(0.2, 0.8, (100, 3))  # id 2: a piece of it under another id
    neighbour = rng.uniform(0.9, 1.9, (100, 3))  # id 3: half outside
    ground = np.column_stack([rng.uniform(-5, 5, (5000, 2)), np.zeros(5000)])  # id 4: big
    tile = _tile(
        np.concatenate([body, fragment, neighbour, ground]),
        np.repeat([1, 2, 3, 4], [1000, 100, 100, 5000]),
    )
    choice = so.Choice(1, [1], 1000, np.zeros(3), np.ones(3))
    assert so.absorb([tile], [choice]) == {1: [2]}
    assert choice.ids == [1, 2]


def test_loose_rule_takes_compact_ambivalent_objects_not_the_ground() -> None:
    def inst(movable: float, static: float, side: float, behaviour: str = "in-place") -> dict:
        return {
            "behaviour": behaviour,
            "properties": {"movable": movable, "static": static, "vegetation": 0.9},
            "bounds": {"min": [0, 0, 0], "max": [side, side, side / 2]},
        }

    assert so.loose(inst(0.47, 0.52, 1.6), 5.6)  # A6's pumpkin 3
    assert not so.loose(inst(0.40, 0.49, 5.2), 5.6)  # a ground patch: too large
    assert not so.loose(inst(0.10, 0.65, 1.5), 5.6)  # dirt: not movable
    assert so.loose(inst(0.2, 0.9, 9.0, "movable"), 5.6)  # the rule's own movable stays


def test_every_gaussian_is_kept_once_and_the_scene_is_otherwise_untouched(result) -> None:
    split, _ = result
    obj = split.objects[0]
    scan_leaves = so.Spz.concat([t.spz for t in split.tiles if t.leaf])
    after = so.Spz.concat(_leaves(split.out_dir / "tileset.json"))
    assert len(after) + len(obj.spz) == len(scan_leaves)
    assert np.array_equal(_records(so.Spz.concat([after, obj.scene_spz()])), _records(scan_leaves))
    report = split.report
    assert report["objects"][0]["splats"] == len(obj.spz) > 1000
    assert report["removed"]["leaves"] == len(obj.spz)
    assert report["removed"]["parents"] > 0
    # Per id: the scene keeps every other id's gaussians, the object holds all of its own.
    doc = json.loads((split.out_dir / "instances.json").read_text(encoding="utf-8"))
    ids_after = np.concatenate(
        [
            skin_scene.decode_runs(doc["tiles"][checksum_positions(s.positions())])
            for s in _leaves(split.out_dir / "tileset.json")
        ]
    )
    ids_before = np.concatenate([t.labels for t in split.tiles if t.leaf])
    assert not np.isin(ids_after, obj.choice.ids).any()
    assert np.isin(obj.labels, obj.choice.ids).all()
    assert np.array_equal(
        np.bincount(np.concatenate([ids_after, obj.labels]), minlength=ids_before.max() + 1),
        np.bincount(ids_before, minlength=ids_before.max() + 1),
    )


def test_every_tile_is_bound_and_untouched_tiles_keep_their_bytes(result, scan: Path) -> None:
    split, _ = result
    out = split.out_dir
    doc = json.loads((out / "instances.json").read_text(encoding="utf-8"))
    tileset = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    checksums = set()
    for uri in tile_uris(tileset):
        positions = tile_positions(out / uri)
        checksum = checksum_positions(positions)
        checksums.add(checksum)
        assert skin_scene.decode_runs(doc["tiles"][checksum]).size == positions.shape[0]
        if (scan / uri).exists():
            assert (scan / uri).read_bytes() == (out / uri).read_bytes()
    obj = split.objects[0]
    tile = out / so.OBJECTS_DIR / str(SHRUB_ID) / so.OBJECT_CONTENT
    object_checksum = checksum_positions(tile_positions(tile))
    assert object_checksum == obj.checksum
    assert np.array_equal(skin_scene.decode_runs(doc["tiles"][object_checksum]), obj.labels)
    # Nothing stale: every bound tile is drawn by the scene or the object.
    assert set(doc["tiles"]) == checksums | {object_checksum}
    assert split.report["tilesRewritten"] >= 2
    original = json.loads((scan / "instances.json").read_text(encoding="utf-8"))
    assert doc["instances"] == original["instances"]


def test_the_object_tileset_puts_it_back_where_it_stood(result) -> None:
    split, _ = result
    out = split.out_dir
    scene = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    entry = scene["root"]["extras"]["objects"][0]
    assert entry["instance"] == SHRUB_ID and entry["pose"] == so.POSE_REST
    assert entry["uri"] == f"{so.OBJECTS_DIR}/{SHRUB_ID}/tileset.json"
    obj_doc = json.loads((out / entry["uri"]).read_text(encoding="utf-8"))
    origin = np.asarray(entry["origin"])
    expected = so._matrix(scene["root"]["transform"]) @ np.block(
        [[np.eye(3), origin[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
    )
    assert np.allclose(so._matrix(obj_doc["root"]["transform"]), expected)
    extras = obj_doc["root"]["extras"]
    assert (
        extras["object"]["instance"] == SHRUB_ID and extras["object"]["origin"] == entry["origin"]
    )
    assert extras["instances"]["uri"] == "../../instances.json"
    # On the SPZ grid, so the shift is exact; its base sits at the origin.
    assert np.array_equal(np.round(origin * 4096), origin * 4096)
    local = tile_positions(out / so.OBJECTS_DIR / str(SHRUB_ID) / so.OBJECT_CONTENT)
    assert abs(float(local[:, 2].min())) < 1e-6
    assert np.allclose(np.median(local[:, :2], axis=0), 0.0, atol=0.3)


def test_scene_and_object_at_rest_render_as_the_scan(result, scan: Path) -> None:
    """The round trip: drawn together at its rest pose, scene and object are the scan."""
    split, _ = result
    out = split.out_dir
    before = load_tileset(scan / "tileset.json")
    scene = load_tileset(out / "tileset.json")
    entry = json.loads((out / "tileset.json").read_text(encoding="utf-8"))["root"]["extras"][
        "objects"
    ][0]
    obj = load_tileset(out / entry["uri"])
    obj.positions = obj.positions + np.asarray(entry["origin"])
    after = type(scene).concat([scene, obj])
    # Same gaussians; the CPU renderer's samples follow each gaussian's index, so compare in
    # one order.
    order_a = np.lexsort(before.positions.T[::-1])
    order_b = np.lexsort(after.positions.T[::-1])
    for name in ("positions", "rotations", "scales", "colours", "opacities"):
        assert np.allclose(getattr(before, name)[order_a], getattr(after, name)[order_b], atol=1e-6)
    camera = Camera.look_at([13.3, 2.0, 4.0], [13.3, 7.1, 0.5], **SIZE)
    a = render(before.take(order_a), camera).rgb
    b = render(after.take(order_b), camera).rgb
    assert np.abs(a - b).max() <= 1.0 / 255


def test_the_hole_is_filled_as_an_inferred_layer(result) -> None:
    split, _ = result
    out = split.out_dir
    report = split.report["fills"][str(SHRUB_ID)]
    assert report["lifted"] > 0
    assert all(v["accepted"] for v in report["views"])
    held = report["heldOut"]
    # The lawn under the crown was never there: the held-out view sees through where the
    # shrub stood, and the fill closes it.
    assert held["throughPx"] > 200
    assert held["coveredBefore"] < 0.3
    assert held["coveredAfter"] > 0.8
    scene = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    layer = scene["root"]["extras"]["inferredLayers"][-1]
    assert layer["uri"] == f"{so.FILLS_DIR}/{SHRUB_ID}/tileset.json"
    assert scene["root"]["extras"]["objects"][0]["fill"] == layer["uri"]
    evidence = json.loads((out / layer["uri"]).read_text(encoding="utf-8"))["root"]["extras"][
        "evidence"
    ]
    assert evidence["kind"] == "inferred" and evidence["hole"] == SHRUB_ID
    assert evidence == layer["evidence"]
    # Lifted onto the ground where it stood, not floating at the crown.
    fill = load_tileset(out / layer["uri"])
    near = np.linalg.norm(fill.positions[:, :2] - np.asarray(SHRUB_STEM), axis=1) < 1.2
    assert near.mean() > 0.5
    ground = synthetic_yard.ground_height(fill.positions[near, 0], fill.positions[near, 1])
    assert np.median(np.abs(fill.positions[near, 2] - ground)) < 0.25


def test_split_is_idempotent_and_never_writes_its_input(result, scan: Path, tmp_path: Path) -> None:
    split, before = result
    assert _fingerprint(scan) == before
    first = _fingerprint(split.out_dir)
    so.split(scan, split.out_dir, ids=[SHRUB_ID], fill=so.fill_holes("telea", views=4, **SIZE))
    assert _fingerprint(split.out_dir) == first
    with pytest.raises(ValueError, match="of its own"):
        so.split(scan, scan, ids=[SHRUB_ID])
    other = tmp_path / "published"
    shutil.copytree(scan, other)
    with pytest.raises(ValueError, match="did not write"):
        so.split(scan, other, ids=[SHRUB_ID])
    assert _fingerprint(other) == before


def test_a_split_tileset_can_be_split_again(result, tmp_path: Path) -> None:
    split, _ = result
    again = so.split(split.out_dir, tmp_path / "again", ids=[SHRUB_ID - 1])
    extras = json.loads((tmp_path / "again" / "tileset.json").read_text(encoding="utf-8"))["root"][
        "extras"
    ]
    assert [o["instance"] for o in extras["objects"]] == [SHRUB_ID, SHRUB_ID - 1]
    assert (tmp_path / "again" / so.OBJECTS_DIR / str(SHRUB_ID) / so.OBJECT_CONTENT).exists()
    assert again.report["objects"][0]["instance"] == SHRUB_ID - 1
    with pytest.raises(ValueError, match="already split"):
        so.split(split.out_dir, tmp_path / "twice", ids=[SHRUB_ID])


class _ChainedPainter:
    """A stand-in generative filler: paints what it is asked in one colour, reads a context,
    and asks to see the earlier views' fill (`chain_views`)."""

    name = "painter"
    chain_views = True

    def __init__(self) -> None:
        self.context: dict | None = None
        self.asked: list[int] = []
        self.received: list[dict] = []

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        self.asked.append(int(mask.sum()))
        self.received.append({"model": "painter"})
        return [np.where(mask[..., None], np.array([200, 40, 40], np.uint8), rgb)]


def test_a_generative_filler_is_told_the_surroundings_and_chains_its_views(
    scan: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import teacher_fill as tf

    painter = _ChainedPainter()
    monkeypatch.setattr(tf, "make_filler", lambda spec: painter)
    save = tmp_path / "strips"
    split = so.split(
        scan,
        tmp_path / "out",
        ids=[SHRUB_ID],
        fill=so.fill_holes("painter", views=4, save_dir=save, **SIZE),
    )
    report = split.report["fills"][str(SHRUB_ID)]
    # Told what is around the hole (the lawn), never to paint the shrub back.
    assert painter.context == report["context"]
    assert "ground" in report["context"]["labels"]
    assert "shrub" not in report["context"]["labels"]
    assert report["context"]["negative"] == "shrub"
    assert len(report["fillerCalls"]) == 4
    # The first view paints its whole hole; later ones only what the earlier fills, lifted
    # and re-rendered, do not already cover -- but each lifts its whole hole.
    through = [v["throughPx"] for v in report["views"]]
    assert painter.asked[0] == through[0]
    assert sum(painter.asked[1:]) < 0.8 * sum(through[1:])
    assert all(v["accepted"] for v in report["views"])
    assert report["heldOut"]["coveredAfter"] > 0.8
    # The held-out strip: as it was | without | filled | the object moved aside.
    from PIL import Image

    strip = Image.open(save / str(SHRUB_ID) / "held-out.png")
    assert strip.size == (4 * SIZE["width"], SIZE["height"])


def test_describe_surroundings_names_the_neighbours_by_footprint_not_the_object() -> None:
    import teacher_fill as tf

    def inst(k: int, label: str, low: list[float], high: list[float], splats: int) -> dict:
        return {
            "id": k,
            "splats": splats,
            "bounds": {"min": low, "max": high},
            "tags": [{"label": label, "score": 0.5}],
        }

    instances = [
        inst(1, "pumpkin", [-50, -50, 0], [50, 50, 2], 100_000),  # scene-wide, mostly elsewhere
        inst(2, "straw", [-1, -1, 0], [1, 1, 0.3], 5_000),
        inst(3, "pumpkin", [-0.5, -0.5, 0], [0.5, 0.5, 1], 20_000),  # the object
        inst(4, "dirt", [-2, -2, 0], [0, 0, 0.2], 3_000),
        inst(5, "moss", [-0.2, -0.2, 0], [0.2, 0.2, 0.1], 50),  # a fragment
        inst(6, "branch", [-1, -1, 3], [1, 1, 4], 9_000),  # above the region
    ]
    out = tf.describe_surroundings(
        instances, np.array([-1.0, -1, -0.5]), np.array([1.0, 1, 0.3]), object_ids=[3],
        exclude_ids=[5],
    )  # fmt: skip
    assert out["labels"] == ["straw", "dirt"]
    assert out["negative"] == "pumpkin"
    assert out["prompt"].startswith("Straw and dirt: a top-down close-up photograph")


def test_skins_are_rebound_with_the_tiles(tmp_path: Path) -> None:
    """The committed yard with its skins linked: splitting a skinned shrub keeps every other
    tile's skin rows and the changed tiles' rows for what stayed."""
    tiles = tmp_path / "splat"
    shutil.copytree(YARD / "splat", tiles)
    for name in ("instances.json", "instances.emb"):
        shutil.copyfile(YARD / "instances" / name, tiles / name)
    for name in ("skin.json", "skin.bin"):
        shutil.copyfile(YARD / "skin" / name, tiles / name)
    ss.link_instances(tiles / "tileset.json", 103)
    skin_scene.link_skin(tiles / "tileset.json", 4)
    out = tmp_path / "out"
    split = so.split(tiles, out, ids=[10])
    old_doc = json.loads((tiles / "skin.json").read_text(encoding="utf-8"))
    old = skin_scene.decode_tiles(old_doc, (tiles / "skin.bin").read_bytes())
    new_doc = json.loads((out / "skin.json").read_text(encoding="utf-8"))
    new = skin_scene.decode_tiles(new_doc, (out / "skin.bin").read_bytes())
    assert new_doc["weights"]["rows"] == len((out / "skin.bin").read_bytes()) // 16
    for t in split.tiles:
        keep = t.owner < 0
        if keep.all():
            assert all(np.array_equal(a, b) for a, b in zip(new[t.checksum], old[t.checksum]))
            continue
        checksum = checksum_positions(t.spz.take(np.flatnonzero(keep)).positions())
        which, rows = new[checksum]
        assert np.array_equal(which, old[t.checksum][0][keep])
        assert np.array_equal(rows, old[t.checksum][1][keep])
    # The shrub's own skin rows left with it.
    assert new_doc["weights"]["rows"] < old_doc["weights"]["rows"]
