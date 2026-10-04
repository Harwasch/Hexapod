"""`segment_scene` on the synthetic yard, with masks from its ground truth (`OracleMasks`).

The yard's every splat carries its object (`source/labels.json`), so the lifting can be
scored: rendered views are cut into the true objects (level 0) and their colour parts
(level 1), lifted back to splats, and the instances compared with the objects. The files are
checked against the v1 contract (docs/SCENE_OBJECTS.md §4) and the committed tiles.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import rig_tiles
import scene_categories
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
        workers=1,  # rendered here: the pytest process does not fork (tests/conftest.py)
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
    # The top-level instances that are described hold nearly all of the top level.
    top = _top(result.lifted.parent)[result.splat_id]
    sub = np.bincount(top, minlength=len(result.instances) + 1)[1:]
    tops = [i for i in result.instances if i.level == 0]
    described = sum(sub[i.id - 1] for i in tops if i.tags)
    assert described >= 0.95 * sum(sub[i.id - 1] for i in tops)


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
            "category",
        ]
        assert r["category"] in scene_categories.category_ids()
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


def test_check_cameras_with_and_without_local_views(run: dict) -> None:
    views = run["result"].views
    cams = ss.check_cameras(views)
    assert len(cams) == 4 and cams[0] is views[0].camera
    assert all(np.isfinite(c.far) for c in cams[1:])
    # A small scan's plan has no far planes: the others are spread through its views.
    whole = [v for v in views if not np.isfinite(v.camera.far)]
    cams = ss.check_cameras(whole)
    assert len(cams) == min(4, len(whole)) and len({id(c) for c in cams}) == len(cams)
    assert ss.check_cameras(whole[:1]) == [whole[0].camera]


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
    args = {
        "cameras": cameras,
        "source_factory": factory,
        "cache": tmp_path / "cache",
        "workers": 1,
    }
    first = ss.segment(splats, None, ss.FakeEmbedder(), VOCABULARY, **args)
    assert len(calls) == 4
    second = ss.segment(splats, None, ss.FakeEmbedder(), VOCABULARY, **args)
    assert len(calls) == 4  # nothing asked again
    assert np.array_equal(first.splat_id, second.splat_id)
    assert len(list((tmp_path / "cache").glob("view-*.npz"))) == 4


# ------------------------------------------------------------------ coverage and meaning


def test_coverage_views_aim_at_what_is_unassigned() -> None:
    """Targets are where the unassigned splats are, each with `COVERAGE_SHOTS` looking at
    it (one from eye height, looking up into a crown); nothing unassigned, no views."""
    rng = np.random.default_rng(3)
    ground = np.c_[rng.uniform(-20, 20, (4000, 2)), rng.uniform(0, 0.2, 4000)]
    crown = np.c_[rng.normal(15, 0.8, (300, 2)), rng.uniform(4, 7, 300)]
    centroids = np.concatenate([ground, crown])
    missing = np.r_[np.zeros(len(ground)), np.full(len(crown), 3.0)]
    edge = 0.2
    cameras = ss.coverage_views(centroids, missing, edge, budget=12)
    assert 0 < len(cameras) <= 12
    assert len(cameras) % len(ss.COVERAGE_SHOTS) == 0
    for camera in cameras:
        # Each looks at a target inside the crown.
        axis = camera.rotation[2]
        t = np.linalg.lstsq(axis[:, None], (crown.mean(axis=0) - camera.centre), rcond=None)[0]
        nearest = camera.centre + axis * t[0]
        assert np.min(np.linalg.norm(crown - nearest, axis=1)) < 1.5
        assert math.isfinite(camera.far) and camera.far > t[0]
    low = [c for c in cameras if c.centre[2] < 3.0]
    assert low and all(c.rotation[2, 2] > 0 for c in low)  # looking up
    assert ss.coverage_views(centroids, np.zeros(len(centroids)), edge) == []


def test_unseen_cells_take_a_label_within_their_reach() -> None:
    """A cell no view saw takes its nearest seen cell's labels within `FILL_CELLS` edges, or
    within its own reach when that is farther (a floater left out of the views)."""
    seen = np.c_[np.arange(12) * 0.1, np.zeros((12, 2))]
    centroids = np.concatenate([seen, [[2.1, 0, 0], [5.0, 0, 0]]])
    counts = np.full(len(centroids), 40)
    one = ss._Votes(
        np.arange(12, dtype=np.int32), np.ones(12, np.float32), np.zeros((1, 12), np.int32)
    )
    votes = [one] * 6
    plain = ss.lift(votes, centroids, counts, 0.1, 1)
    assert plain.cell_id[0] > 0 and plain.cell_id[12] == 0 and plain.cell_id[13] == 0
    reach = np.r_[np.zeros(12), 1.5, 1.5]
    reached = ss.lift(votes, centroids, counts, 0.1, 1, cell_reach=reach)
    assert reached.cell_id[12] == reached.cell_id[0] and reached.cell_id[13] == 0


def test_coverage_rounds_assign_more_of_the_scan(run: dict) -> None:
    """From a few views, a coverage round adds views where splats are left without an
    instance, and fewer are left."""
    splats, levels = run["splats"], run["levels"]
    cameras = [v.camera for v in run["result"].views][:3]
    args = {
        "cameras": cameras,
        "source_factory": lambda c: ss.OracleMasks(splats, levels, c),
        "cells": run["result"].cells,
        "workers": 1,
    }
    before = ss.segment(splats, None, ss.FakeEmbedder(), VOCABULARY, **args)
    after = ss.segment(
        splats, None, ss.FakeEmbedder(), VOCABULARY, coverage_rounds=1, coverage_budget=12, **args
    )
    rounds = after.lifted.stats["coverageRounds"]
    assert len(rounds) == 1 and 0 < rounds[0]["views"] <= 12
    assert len(after.views) == 3 + rounds[0]["views"]
    assert (after.splat_id == 0).mean() < (before.splat_id == 0).mean()


def test_described_instances_carry_a_category_and_portraits_are_embedded(run: dict) -> None:
    """With a renderer, each described instance's crops include portraits of its own splats;
    its category (tags and the category head) is written as given."""
    result = run["result"]
    seen: list[np.ndarray] = []

    class Recording(ss.FakeEmbedder):
        def embed_images(self, images):
            seen.extend(images)
            return super().embed_images(images)

    instances = ss.describe(
        result.lifted, run["splats"], result.cell, result.views[:6], Recording(), VOCABULARY,
        renderer=ss.CpuRenderer(), render_splats=run["splats"], render_cell=result.cell, kinds=("context", "black", "portrait"),
    )  # fmt: skip
    by_kind: dict[str, np.ndarray] = {}
    again = ss.describe(
        result.lifted, run["splats"], result.cell, result.views[:6], ss.FakeEmbedder(),
        VOCABULARY, renderer=ss.CpuRenderer(), render_splats=run["splats"],
        render_cell=result.cell, by_kind=by_kind,
    )  # fmt: skip
    assert {"context", "alone", "portrait"} <= set(by_kind)
    mixed = ss._normalise(sum(by_kind[k] for k in ss.DESCRIBE_KINDS))
    np.testing.assert_allclose(mixed, np.stack([i.embedding for i in again]), atol=1e-9)
    distributions: dict[str, np.ndarray] = {}
    variants = ss.describe_variants(ss.FakeEmbedder(), VOCABULARY, by_kind, distributions)
    rows = distributions["context/rows"]
    assert distributions["context/head"].shape == (rows.size, len(distributions["categories"]))
    assert "context+black@0.5" in variants
    # The category: each described kind's distribution, averaged, mixed with the parent's.
    described = np.array([i.category is not None for i in again])
    expected = ss._categories_by_kind(
        ss.FakeEmbedder(), {k: by_kind[k] for k in sorted(ss.DESCRIBE_KINDS)}, described,
        VOCABULARY, None, result.lifted.parent,
    )  # fmt: skip
    assert expected == [i.category for i in again]
    portraits = [i for i in seen if i.shape == (ss.PORTRAIT_PX, ss.PORTRAIT_PX, 3)]
    assert portraits and all(max(i.shape[:2]) <= ss.CROP_MAX_SIDE for i in seen)
    ids = scene_categories.category_ids()
    for inst in instances:
        assert (inst.category in ids) if inst.tags else inst.category is None
    document = ss.instances_document(
        instances, {}, embedding_model="fake", dim=64, vocabulary_model="fake",
        vocabulary_size=len(VOCABULARY),
    )  # fmt: skip
    for record, inst in zip(document["instances"], instances, strict=True):
        if inst.category is not None:
            assert record["category"] == inst.category


def test_given_categories_take_the_place_of_the_tags_vote() -> None:
    records = [
        {"id": 1, "parent": None, "tags": [{"label": "dirt", "score": 0.5}], "splats": 10},
        {"id": 2, "parent": 1, "tags": [], "splats": 5},
    ]
    labels = {"dirt": "ground"}
    assert scene_categories.instance_categories(records, labels) == {1: "ground", 2: "ground"}
    given = scene_categories.instance_categories(records, labels, given={1: "produce"})
    assert given == {1: "produce", 2: "produce"}


def test_each_batch_of_views_is_sized_from_the_reservation(
    run: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no `workers`, the render processes come from the reservation the caller passed
    (`default_workers(cpus, memory_bytes, worker_bytes)`, infra/modal/segment.py's `--cpus` /
    `--memory-gb`, and what a render of this scan may hold), asked again for each batch of
    views -- the first and every coverage round -- said, and kept in the stats. The first
    batch's are forked (one `RenderPool`) and kept: a coverage round renders in them, as many
    at a time as it is sized for, rather than forking again."""
    splats, levels = run["splats"], run["levels"]
    asked: list[tuple[object, object, object]] = []
    sizes = iter([3, 2])

    def default_workers(cpus=None, memory_bytes=None, worker_bytes=None):
        asked.append((cpus, memory_bytes, worker_bytes))
        return next(sizes)

    pools: list[int] = []
    batches: list[tuple[int, int | None]] = []

    class Pool(ss.RenderPool):
        """A pool of the processes asked for, that renders in this process (tests/conftest.py:
        the pytest process does not fork)."""

        def __init__(self, *args, workers: int = 1, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            pools.append(workers)

        def __enter__(self):
            super().__enter__()
            self.workers = pools[-1]
            return self

        def views(self, cameras, workers=None):
            batches.append((len(cameras), workers))
            return super().views(cameras, workers)

    monkeypatch.setattr(ss, "default_workers", default_workers)
    monkeypatch.setattr(ss, "RenderPool", Pool)
    said: list[str] = []
    cameras = [v.camera for v in run["result"].views][:2]
    sized = ss.segment(
        splats,
        None,
        ss.FakeEmbedder(),
        VOCABULARY,
        cameras=cameras,
        source_factory=lambda batch: ss.OracleMasks(splats, levels, batch),
        progress=said.append,
        cpus=8,
        memory_bytes=32 * float(1 << 30),
        coverage_rounds=1,
        coverage_budget=4,
    )
    assert asked == [(8, 32 * float(1 << 30), ss.RENDER_WORKER_BYTES)] * 2
    extra = len(sized.views) - len(cameras)
    assert pools == [3] and batches == [(2, 3), (extra, 2)] and extra > 0
    assert "render workers: 3" in said and "render workers: 2" in said
    assert sized.lifted.stats["renderWorkers"] == [3, 2]
    # A count given is used as it is.
    asked.clear()
    ss.segment(
        splats,
        None,
        ss.FakeEmbedder(),
        VOCABULARY,
        cameras=cameras,
        source_factory=lambda batch: ss.OracleMasks(splats, levels, batch),
        workers=1,
        cpus=8,
        memory_bytes=32 * float(1 << 30),
    )
    assert asked == []


#: Two batches through one pool of three processes, a thread started between them.
RENDER_POOL = """
import json
import os
import threading
import numpy as np
import segment_scene as ss
from splat_render import Camera, Splats

forks = []
os.register_at_fork(before=lambda: forks.append(threading.active_count()))
rng = np.random.default_rng(1)
n = 5_000
splats = Splats(
    rng.uniform(-1, 1, (n, 3)), np.tile([1.0, 0, 0, 0], (n, 1)), np.full((n, 3), 0.03),
    rng.uniform(0, 1, (n, 3)), np.full(n, 0.8),
)
cells = rng.integers(0, 40, n)
cameras = [
    Camera.look_at([3 * np.cos(a), 3 * np.sin(a), 1.0], [0, 0, 0], width=64, height=48)
    for a in np.linspace(0.0, 6.0, 7)
]
expected = [ss.render_view(splats, camera, cells) for camera in cameras]
with ss.RenderPool(splats, cells, workers=3) as pool:
    forked = len(forks)
    first = list(pool.views(cameras[:4]))
    stop = threading.Event()
    busy = threading.Thread(target=stop.wait)
    busy.start()
    second = list(pool.views(cameras[4:], workers=1))
    stop.set()
    busy.join()
same = [
    bool(np.array_equal(a.rgb, b.rgb) and np.array_equal(a.cell, b.cell))
    for a, b in zip(first + second, expected, strict=True)
]
print(json.dumps({
    "forkedOnEnter": forked, "threadsAtFork": forks, "same": same,
    "batches": [len(first), len(second)], "peak": ss._WORKER_PEAK["bytes"],
}))
"""


def test_a_runs_render_processes_are_forked_once(fresh_process: Callable[[str], dict]) -> None:
    """A `RenderPool` forks its processes as it is entered, from a process with no thread
    but the one forking, and renders every batch in them, in order, the views this process
    renders -- the second batch after this process has started a thread of its own, as the
    mask model does, and without a fork. (Run in a new interpreter: tests/conftest.py.)"""
    out = fresh_process(RENDER_POOL)
    assert out["forkedOnEnter"] == 3 and out["threadsAtFork"] == [1, 1, 1]
    assert out["batches"] == [4, 3] and all(out["same"]) and out["peak"] > 0
