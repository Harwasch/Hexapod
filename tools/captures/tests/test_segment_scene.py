"""`segment_scene` on the synthetic yard, with masks from its ground truth (`OracleMasks`).

The yard's every splat carries its object (`source/labels.json`), so the lifting can be
scored: rendered views are cut into the true objects (level 0) and their colour parts
(level 1), lifted back to splats, and the instances compared with the objects. The files are
checked against the v1 contract (docs/SCENE_OBJECTS.md §4) and the committed tiles.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import rig_tiles
import segment_scene as ss
import synthetic_yard
from synthetic_tree import checksum_positions, write_ply

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard"
OPACITY_MIN = 0.02
TILE_GAUSSIANS = 6000
VIEWS = 16
VOCABULARY = ["tree", "shrub", "dead tree", "house", "grass", "dirt", "path", "car"]
#: Objects scored need this many splats (the smallest yard object has 400).
MIN_OBJECT_SPLATS = 200


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The yard's source (PLY + labels): the checkout's when generated, else a fresh one
    (it is what the committed tiles were packed from, `test_scene_plants`)."""
    if (YARD / "source" / "splat.ply").exists() and (YARD / "source" / "labels.json").exists():
        return YARD / "source"
    out = tmp_path_factory.mktemp("yard") / "source"
    out.mkdir()
    data, klass, inst, instances = synthetic_yard.generate_yard()
    write_ply(out / "splat.ply", data)
    (out / "labels.json").write_text(
        json.dumps(
            {
                "classes": list(synthetic_yard.CLASSES),
                "instances": instances,
                "class": [int(v) for v in klass],
                "instance": [int(v) for v in inst],
            }
        ),
        encoding="utf-8",
    )
    return out


@pytest.fixture(scope="module")
def run(source: Path) -> dict:
    splats, rows, row_count = ss.load_source(source / "splat.ply", OPACITY_MIN)
    truth = json.loads((source / "labels.json").read_text(encoding="utf-8"))
    levels = ss.truth_levels(truth, rows, splats.colours)
    result = ss.segment(
        splats,
        None,
        ss.FakeEmbedder(),
        VOCABULARY,
        view_count=VIEWS,
        source_factory=lambda cameras: ss.OracleMasks(splats, levels, cameras),
    )
    return {
        "splats": splats,
        "rows": rows,
        "row_count": row_count,
        "levels": levels,
        "result": result,
    }


@pytest.fixture(scope="module")
def written(run: dict, source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tiles = tmp_path_factory.mktemp("tiles") / "splat"
    shutil.copytree(YARD / "splat", tiles)
    result = run["result"]
    binding = ss.tile_binding(
        tiles,
        source / "splat.ply",
        run["rows"],
        result.splat_id,
        run["row_count"],
        opacity_min=OPACITY_MIN,
        tile_gaussians=TILE_GAUSSIANS,
    )
    embedder = ss.FakeEmbedder()
    document = ss.instances_document(
        result.instances,
        binding,
        embedding_model=embedder.name,
        dim=embedder.dim,
        vocabulary_model=embedder.name,
        vocabulary_size=len(VOCABULARY),
    )
    ss.write_instances(tiles, document, result.instances)
    ss.link_instances(tiles / "tileset.json", len(result.instances))
    return tiles


def _top(parent: np.ndarray) -> np.ndarray:
    """Per id (index 0 = none), its level-0 ancestor."""
    top = np.arange(parent.size + 1)
    for k in range(1, parent.size + 1):
        a = k
        while parent[a - 1]:
            a = int(parent[a - 1])
        top[k] = a
    return top


def _ious(truth: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    """Per true object (enough splats): IoU with the predicted id covering most of it."""
    out: dict[int, float] = {}
    for obj in np.unique(truth):
        mine = truth == obj
        if mine.sum() < MIN_OBJECT_SPLATS:
            continue
        votes = np.bincount(pred[mine])
        votes[0] = 0
        best = int(votes.argmax())
        if best == 0:
            out[int(obj)] = 0.0
            continue
        theirs = pred == best
        out[int(obj)] = float((mine & theirs).sum() / (mine | theirs).sum())
    return out


# ------------------------------------------------------------------------ the models


def test_doubles_satisfy_the_protocols() -> None:
    embedder = ss.FakeEmbedder(dim=32)
    assert isinstance(embedder, ss.Embedder)
    assert not isinstance(embedder, ss.MaskSource)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((10, 20), (33, 7))]
    a = embedder.embed_images(crops)
    t = embedder.embed_texts(["tree", "car", "tree"])
    assert a.shape == (2, 32) and a.dtype == np.float32
    assert t.shape == (3, 32) and t.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(t[0], t[2])
    np.testing.assert_array_equal(a, ss.FakeEmbedder(dim=32).embed_images(crops))
    assert isinstance(ss.load_embedder("segment_scene:FakeEmbedder"), ss.Embedder)
    with pytest.raises(TypeError):
        ss.load_masks("segment_scene:FakeEmbedder")
    with pytest.raises(ValueError):
        ss.load_masks("segment_scene")


def test_supervoxels_bound_the_cells() -> None:
    pos = np.random.default_rng(1).uniform(0, 10, (50_000, 3)).astype(np.float32)
    cell, centroids, counts, edge = ss.supervoxels(pos, max_cells=2000)
    assert cell.shape == (50_000,) and cell.dtype == np.int32
    assert len(centroids) <= 2000 and counts.sum() == 50_000
    assert edge > ss.CELL_M
    # A cell's splats are within one edge of its centroid on every axis.
    assert np.all(np.abs(pos - centroids[cell]) <= edge + 1e-6)


def test_behaviour_rule() -> None:
    base = dict.fromkeys(ss.PROPERTY_PROMPTS, 0.1)
    assert ss.behaviour(base) == "static"
    assert ss.behaviour({**base, "vegetation": 0.7}) == "in-place"
    assert ss.behaviour({**base, "vegetation": 0.7, "vehicle": 0.6}) == "movable"
    assert ss.behaviour({**base, "movable": 0.6, "static": 0.7}) == "static"
    assert ss.behaviour({**base, "movable": 0.6, "static": 0.2}) == "movable"


# ---------------------------------------------------------------------------- lifting


def test_oracle_masks_cover_the_views(run: dict) -> None:
    result = run["result"]
    # The yard is wider than one view's footprint: the whole-scan views plus local ones.
    assert VIEWS < len(result.views) <= ss.MAX_VIEWS
    assert sum(np.isfinite(v.camera.far) for v in result.views) > VIEWS
    assert result.lifted.stats["levels"] == 2
    # Every view's cells voted somewhere at the object level.
    for v in result.votes:
        assert (v.masks[0] >= 0).mean() > 0.5


def test_instances_match_the_true_objects(run: dict) -> None:
    result = run["result"]
    top = _top(result.lifted.parent)[result.splat_id]
    ious = _ious(run["levels"][0], top)
    assert len(ious) >= 12
    assert np.mean(list(ious.values())) >= 0.8, ious
    # The yard's plants (trees, shrubs, snags: objects 0-9) one by one.
    plants = [iou for obj, iou in ious.items() if obj < 10]
    assert len(plants) == 10 and min(plants) >= 0.75, ious
    assert (result.splat_id > 0).mean() >= 0.9
    # Every top-level instance that holds an object is described.
    for inst in result.instances:
        if inst.level == 0 and inst.splats >= MIN_OBJECT_SPLATS:
            assert inst.tags, inst


def test_hierarchy_is_consistent(run: dict) -> None:
    result = run["result"]
    lifted = result.lifted
    n = lifted.parent.size
    assert n == len(result.instances) > 0
    for k in range(n):
        p = int(lifted.parent[k])
        if p == 0:
            assert lifted.level[k] == 0
        else:
            assert p < k + 1  # parents are numbered first
            assert lifted.level[k] == lifted.level[p - 1] + 1
    assert (lifted.level == 1).any()  # the colour parts split some objects
    # Splats of a child lie inside its parent: every splat's ancestors contain it.
    for inst in result.instances:
        if inst.parent is not None:
            parent = result.instances[inst.parent - 1]
            assert np.all(inst.bounds_min >= parent.bounds_min - 1e-6)
            assert np.all(inst.bounds_max <= parent.bounds_max + 1e-6)
            assert inst.views <= parent.views
    # Parts refine objects: a level-1 instance holds one true object only (mostly).
    ids = result.splat_id
    objects = run["levels"][0]
    for k in np.flatnonzero(lifted.level == 1)[:40] + 1:
        mine = objects[ids == k]
        if mine.size >= MIN_OBJECT_SPLATS:
            assert np.bincount(mine).max() / mine.size >= 0.8


def test_instance_records(run: dict) -> None:
    result = run["result"]
    splats = run["splats"]
    total = np.bincount(result.splat_id, minlength=len(result.instances) + 1)
    for inst in result.instances:
        assert inst.splats == total[inst.id]
        assert np.all(inst.bounds_min <= inst.centroid) and np.all(inst.centroid <= inst.bounds_max)
        assert inst.views >= 1
        if inst.tags:
            assert abs(float(np.linalg.norm(inst.embedding)) - 1) < 1e-6
            scores = [t["score"] for t in inst.tags]
            assert len(inst.tags) == ss.TAGS_TOP_K and scores == sorted(scores, reverse=True)
            assert {t["label"] for t in inst.tags} <= set(VOCABULARY)
        else:
            # Too small to describe: no embedding, no tags, its ancestor's properties.
            assert not np.any(inst.embedding)
            if inst.parent is None:
                assert not any(inst.properties.values())
            else:
                assert inst.properties == result.instances[inst.parent - 1].properties
        assert list(inst.properties) == list(ss.PROPERTY_PROMPTS)
        assert all(0 <= v <= 1 for v in inst.properties.values())
        assert inst.behaviour == ss.behaviour(inst.properties)
        own = splats.positions[result.splat_id == inst.id]
        if own.size:
            assert np.all(own.min(axis=0) >= inst.bounds_min - 1e-4)


def test_lifting_is_deterministic(run: dict) -> None:
    result = run["result"]
    again = result.relift()
    np.testing.assert_array_equal(again.cell_id, result.lifted.cell_id)
    np.testing.assert_array_equal(again.parent, result.lifted.parent)
    np.testing.assert_array_equal(again.level, result.lifted.level)
    view = result.views[3]
    redone = ss.render_view(run["splats"], view.camera, result.cell)
    np.testing.assert_array_equal(redone.rgb, view.rgb)
    np.testing.assert_array_equal(redone.cell, view.cell)


# ---------------------------------------------------------------------------- the files


def test_instances_json_follows_the_contract(written: Path, run: dict) -> None:
    document = json.loads((written / "instances.json").read_text(encoding="utf-8"))
    assert list(document) == [
        "format",
        "version",
        "frame",
        "embedding",
        "vocabulary",
        "instances",
        "tiles",
        "tilesEncoding",
    ]
    assert document["format"] == "hexapod.instances" and document["version"] == 1
    assert document["embedding"] == {
        "file": "instances.emb",
        "model": "fake-colour-histogram",
        "dim": 64,
        "dtype": "float16",
    }
    assert document["vocabulary"]["size"] == len(VOCABULARY)
    records = document["instances"]
    assert [r["id"] for r in records] == list(range(1, len(records) + 1))
    for r in records:
        assert list(r) == [
            "id",
            "parent",
            "level",
            "splats",
            "bounds",
            "centroid",
            "tags",
            "properties",
            "behaviour",
            "views",
        ]
        assert r["parent"] is None or 1 <= r["parent"] < r["id"]
        assert r["behaviour"] in ("static", "in-place", "movable")
        assert len(r["bounds"]["min"]) == len(r["centroid"]) == 3
    emb = np.frombuffer((written / "instances.emb").read_bytes(), "<f2")
    emb = emb.reshape(len(records), 64).astype(np.float32)
    described = np.array([bool(r["tags"]) for r in records])
    assert described.mean() > 0.3
    np.testing.assert_allclose(np.linalg.norm(emb[described], axis=1), 1, atol=2e-3)
    assert not emb[~described].any()
    np.testing.assert_allclose(
        emb[0], run["result"].instances[0].embedding.astype(np.float16), atol=0
    )


def test_tile_rle_covers_every_gaussian(written: Path, run: dict) -> None:
    document = json.loads((written / "instances.json").read_text(encoding="utf-8"))
    tileset = json.loads((written / "tileset.json").read_text(encoding="utf-8"))
    uris = rig_tiles.tile_uris(tileset)
    assert len(document["tiles"]) == len(uris)
    n = len(document["instances"])
    leaf_counts = np.zeros(n + 1, np.int64)
    for uri in uris:
        positions = rig_tiles.tile_positions(written / uri)
        rle = document["tiles"][checksum_positions(positions)]
        ids, counts = np.asarray(rle[0::2]), np.asarray(rle[1::2])
        assert counts.sum() == positions.shape[0], uri
        assert ids.min() >= 0 and ids.max() <= n
        assert np.all(counts > 0) and np.all(ids[1:] != ids[:-1])
        np.add.at(leaf_counts, ids, counts)
    # The leaf tiles hold every kept splat once, so each id appears at least `splats` times.
    splats = np.asarray([r["splats"] for r in document["instances"]])
    assert np.all(leaf_counts[1:] >= splats)


def test_root_extras_link_keeps_the_order(written: Path) -> None:
    before = json.loads((YARD / "splat" / "tileset.json").read_text(encoding="utf-8"))
    after = json.loads((written / "tileset.json").read_text(encoding="utf-8"))
    old = list(before["root"].get("extras", {}))
    extras = after["root"]["extras"]
    assert list(extras)[: len(old)] == old
    n = len(json.loads((written / "instances.json").read_text(encoding="utf-8"))["instances"])
    assert extras["instances"] == {"uri": "instances.json", "count": n}
    # Linking again replaces it in place.
    ss.link_instances(written / "tileset.json", 7)
    again = json.loads((written / "tileset.json").read_text(encoding="utf-8"))["root"]["extras"]
    assert list(again) == list(extras) and again["instances"]["count"] == 7
    ss.link_instances(written / "tileset.json", n)


def test_render_instances(run: dict, tmp_path: Path) -> None:
    result = run["result"]
    pixels = ss.render_instances(
        run["splats"], result.splat_id, result.views[0].camera, tmp_path / "i.png"
    )
    assert (tmp_path / "i.png").exists()
    h, w = result.views[0].rgb.shape[:2]
    assert pixels.shape == (h, 2 * w, 3)
    assert pixels[:, w:].any()


def test_binding_by_position_agrees_with_the_ply_replay(written: Path, run: dict) -> None:
    """A scan known only by its tiles binds as one known by its PLY: exactly on the leaves,
    and on nearly every merged parent gaussian."""
    from scipy.spatial import cKDTree

    from splat_render import load_tileset

    replayed = json.loads((written / "instances.json").read_text(encoding="utf-8"))["tiles"]
    leaves = load_tileset(written / "tileset.json")
    nearest = cKDTree(run["splats"].positions).query(leaves.positions)[1]
    by_position = ss.tile_binding_by_position(
        written, leaves.positions, run["result"].splat_id[nearest]
    )
    assert sorted(by_position) == sorted(replayed)

    def expand(rle: list[int]) -> np.ndarray:
        return np.repeat(np.asarray(rle[0::2]), np.asarray(rle[1::2]))

    agree = total = 0
    for checksum, rle in replayed.items():
        a, b = expand(rle), expand(by_position[checksum])
        assert a.size == b.size
        agree += int((a == b).sum())
        total += a.size
    assert agree / total > 0.95


def test_a_cached_run_resumes_with_the_same_result(run: dict, tmp_path: Path) -> None:
    """Views and masks kept in a cache are reused: a second run asks the mask source for
    nothing and lifts the same instances."""
    splats, levels = run["splats"], run["levels"]
    calls: list[int] = []

    def factory(cameras):
        oracle = ss.OracleMasks(splats, levels, cameras)

        class Counting:
            name = oracle.name

            def masks(self, rgb):
                calls.append(1)
                return oracle.masks(rgb)

        return Counting()

    cameras = [v.camera for v in run["result"].views][:4]
    args = {"cameras": cameras, "source_factory": factory, "cache": tmp_path / "cache"}
    first = ss.segment(splats, None, ss.FakeEmbedder(), VOCABULARY, **args)
    assert len(calls) == 4
    second = ss.segment(splats, None, ss.FakeEmbedder(), VOCABULARY, **args)
    assert len(calls) == 4  # nothing asked again
    assert np.array_equal(first.splat_id, second.splat_id)
    assert len(list((tmp_path / "cache").glob("view-*.npz"))) == 4
