"""Concept-first segmentation (concept_scene.py) on the synthetic yard, with oracle concepts
and oracle class-free masks: what the lift, the ground classes, the names and the file make
of perfect 2D answers. The real models are tested in test_concept_models.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import concept_scene as cs
import segment_ground_first as sgf
import segment_scene as ss
import synthetic_yard
from splat_render import Camera
from synthetic_tree import write_ply

OPACITY_MIN = 0.02
VIEWS = 12
#: Objects scored need this many splats (the smallest yard object has 400).
MIN_OBJECT_SPLATS = 200


def _yard(out: Path) -> Path:
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


def _run(
    splats, rows, truth: dict, *, named: tuple[str, ...] | None = None, tracks: bool = True
) -> dict:
    """The pipeline on the yard; `named`: the things the vocabulary lists (default all)."""
    ground = cs.ground_layer(splats)
    _, centroids, _, edge, _, _ = cs.ground_cells(splats.positions, ground)
    cameras = ss.plan_views(
        splats.positions, VIEWS, observers=ss.observer_points(splats), edge=edge,
        solid=centroids,
    )  # fmt: skip
    thing, obj, stuff = cs.yard_truth(truth, rows)
    concepts = list(cs.YARD_CONCEPTS)
    if named is not None:
        keep = [k for k, c in enumerate(cs.YARD_CONCEPTS) if c.kind == "stuff" or c.name in named]
        remap = np.full(len(cs.YARD_CONCEPTS), -1)
        things = [k for k in keep if cs.YARD_CONCEPTS[k].kind == "thing"]
        remap[things] = np.arange(len(things))
        thing = np.where(thing >= 0, remap[np.maximum(thing, 0)], -1)
        concepts = [cs.YARD_CONCEPTS[k] for k in keep]
    source = cs.OracleConcepts(splats, thing, obj, stuff, cameras, tracks=tracks)
    levels = ss.truth_levels(truth, rows, splats.colours)
    result = cs.segment_concepts(
        splats, cs.FixedVocabulary(concepts), source, None, ss.FakeEmbedder(),
        ["tree", "shrub", "grass", "dirt", "path", "house"],
        cameras=cameras, overview_count=4, workers=1, ground=ground,
        free_factory=lambda cams: ss.OracleMasks(splats, levels, cams),
    )  # fmt: skip
    return {"result": result, "thing": thing, "obj": obj, "stuff": stuff, "ground": ground}


@pytest.fixture(scope="module")
def yard(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = _yard(tmp_path_factory.mktemp("yard"))
    splats, rows, _ = ss.load_source(out / "splat.ply", OPACITY_MIN)
    truth = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    return {"splats": splats, "rows": rows, "truth": truth, "dir": out}


@pytest.fixture(scope="module")
def run(yard: dict) -> dict:
    return _run(yard["splats"], yard["rows"], yard["truth"])


def _objects(result: cs.ConceptSegmentation) -> np.ndarray:
    return ss.top_level(result.lift.lifted.parent)[result.splat_id]


def _best(truth_mask: np.ndarray, pred: np.ndarray) -> tuple[int, float]:
    votes = np.bincount(pred[truth_mask])
    votes[0] = 0
    best = int(votes.argmax())
    theirs = pred == best
    return best, float((truth_mask & theirs).sum() / (truth_mask | theirs).sum())


# --------------------------------------------------------------------------- unit parts


def test_doubles_satisfy_the_protocols(run: dict) -> None:
    assert isinstance(cs.FixedVocabulary([]), cs.Vocabulary)
    assert isinstance(cs.OracleConcepts.__new__(cs.OracleConcepts), cs.ConceptSource)


def test_clean_concepts_dedupes_and_keeps_the_list_short() -> None:
    others = ("other", "another", "a third")
    raw = [cs.Concept(f"Thing {k}", "thing", "furniture", others) for k in range(20)]
    raw += [cs.Concept("thing 1", "thing", "trees"), cs.Concept("  Lawn ", "stuff")]
    raw += [cs.Concept(w, "stuff") for w in ("grass", "gravel", "nonsense", "path", "dirt")]
    out = cs.clean_concepts(raw, max_things=5, max_stuff=3)
    things, stuff = cs.split_concepts(out)
    assert [c.name for c in things] == ["thing 0", "thing 1", "thing 2", "thing 3", "thing 4"]
    assert things[1].category == "furniture"  # the first spelling wins
    assert things[0].queries == ("thing 0", "other", "another")  # at most two other words
    # Cover: the class each word names, once each ("lawn" is grass), unknown words dropped.
    assert [(c.name, c.cover) for c in stuff] == [
        ("Grass", "grass"), ("Gravel", "gravel"), ("Trail", "trail"),
    ]  # fmt: skip
    assert all(c.category == cs.GROUND_CATEGORY and c.prompts for c in stuff)
    assert cs.clean_concepts([cs.Concept("x", "thing", "nope")])[0].category == "other"


def test_cover_words_name_the_shared_classes() -> None:
    assert cs.cover_of("Tall grass").id == "tall-grass"
    assert cs.cover_of("tall-grass").id == "tall-grass"
    assert cs.cover_of("straw").id == "hay"
    assert cs.cover_of("pine needles").id == "forest-floor"
    assert cs.cover_of("carpet") is None
    ids = {c.id for c in sgf.cover_classes()[0]}
    assert set(cs.COVER_WORDS.values()) <= ids


def test_load_concepts_reads_a_runs_concepts_block(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    block = {
        "things": [{"name": "Cable Spool", "category": "equipment", "prompts": ["reel"]}],
        "cover": ["gravel"],
    }
    path.write_text(json.dumps(block), encoding="utf-8")
    things, stuff = cs.split_concepts(cs.load_concepts(path))
    assert things == [cs.Concept("cable spool", "thing", "equipment", ("cable spool", "reel"))]
    assert [c.cover for c in stuff] == ["gravel"]


def test_camera_paths_cut_where_the_camera_jumps() -> None:
    def cam(x: float, yaw: float) -> Camera:
        eye = np.array([x, 0.0, 1.0])
        look = eye + np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return Camera.look_at(eye, look, width=8, height=8)

    cameras = [cam(0, 0), cam(0.5, 0.1), cam(1.0, 0.2), cam(9.0, 0.2), cam(9.2, 2.5)]
    paths = cs.camera_paths(cameras, radius=2.0)
    assert paths == [[0, 1, 2], [3], [4]]
    assert sorted(k for p in paths for k in p) == list(range(len(cameras)))


def test_cells_never_mix_ground_and_the_rest(run: dict, yard: dict) -> None:
    result = run["result"]
    flag = result.ground.flag
    for c in np.unique(result.cell):
        members = flag[result.cell == c]
        assert members.all() or not members.any()
    # The yard's lawn and path are ground; its trees, snags and house mostly not.
    klass = np.asarray(yard["truth"]["class"])[yard["rows"]]
    classes = list(yard["truth"]["classes"])
    lawn = np.isin(klass, [classes.index("grass/low"), classes.index("ground")])
    assert flag[lawn].mean() > 0.9
    tree = klass == classes.index("tree")
    assert flag[tree].mean() < 0.1


def test_track_joins_unite_what_one_track_spans() -> None:
    votes = cs.ConceptVotes(n_cells=4, n_things=2, n_stuff=0)
    # Two views: cells 0-1 and cells 2-3 each in their own mask, both of track 7.
    for cells in ([0, 1], [2, 3]):
        votes.things.append(
            ss._Votes(np.array(cells, np.int32), np.ones(2, np.float32), np.zeros((1, 2), np.int32))
        )
        votes.thing_concept.append(np.array([0]))
        votes.thing_track.append(np.array([7]))
    region = np.array([0, 0, 1, 1])
    joined, joins = cs.track_joins(region, np.array([0, 0]), votes)
    assert joins == 1 and joined.tolist() == [0, 0, 0, 0]
    # Not across concepts.
    joined, joins = cs.track_joins(region, np.array([0, 1]), votes)
    assert joins == 0 and joined.tolist() == [0, 0, 1, 1]


# ------------------------------------------------------------------------ the whole run


def test_named_objects_match_the_true_objects(run: dict, yard: dict) -> None:
    result = run["result"]
    objects = _objects(result)
    obj = run["obj"]
    names = {k + 1: e.get("name") for k, e in enumerate(result.extra)}
    want = {0: "Tree", 1: "Shrub", 2: "Dead tree", 3: "House"}
    for o in np.unique(obj[obj >= 0]):
        mine = obj == o
        if mine.sum() < MIN_OBJECT_SPLATS:
            continue
        best, iou = _best(mine, objects)
        assert iou >= 0.8, (o, iou)
        assert names[best] == want[int(run["thing"][mine][0])]


def test_the_ground_is_its_cover_classes_in_the_shared_schema(run: dict) -> None:
    result = run["result"]
    lift = result.lift
    tops = [k for k in range(len(result.instances)) if lift.top[k] and lift.kind[k] == "ground"]
    assert sorted(str(result.extra[k]["name"]) for k in tops) == ["Grass", "Trail"]
    for k in tops:
        instance = result.instances[k]
        extra = result.extra[k]
        assert instance.parent is None and instance.category == cs.GROUND_CATEGORY
        assert extra["kind"] == "ground" and instance.behaviour == "static"
        assert extra["nameSource"] == "ground-cover"
        assert extra["cover"] in ("grass", "trail")
        assert instance.tags == [
            {"label": extra["name"].lower(), "score": instance.tags[0]["score"]}
        ]
        # Its regions (if more than one) are its children, of its class.
        children = [i for i in result.instances if i.parent == k + 1]
        assert len(children) != 1
        assert instance.splats == (0 if children else instance.splats)
        for child in children:
            assert result.extra[child.id - 1]["cover"] == extra["cover"]
            assert child.category == cs.GROUND_CATEGORY
    # Cover classes on the true lawn and path, by splats.
    cover = np.where(lift.ground, lift.cover, -1)[result.cell]
    stuff = run["stuff"]
    for s in (0, 1):
        mine = (stuff == s) & lift.ground[result.cell]
        assert (cover[mine] == s).mean() > 0.9


def test_ground_inside_a_things_masks_is_the_things(run: dict, yard: dict) -> None:
    # The house stands on a slab of its own at ground level: geometry calls it ground, the
    # house's masks claim it back.
    result = run["result"]
    klass = np.asarray(yard["truth"]["class"])[yard["rows"]]
    house = klass == list(yard["truth"]["classes"]).index("other-static")
    assert result.lift.promoted.sum() > 0
    _, iou = _best(house, _objects(result))
    assert iou >= 0.9


def test_parts_carry_their_objects_category_and_no_name(run: dict) -> None:
    result = run["result"]
    by_id = {i.id: i for i in result.instances}
    parts = [
        i
        for i in result.instances
        if i.parent is not None and result.extra[i.id - 1]["kind"] == "thing"
    ]
    assert parts
    for part in parts:
        top = part
        while top.parent is not None:
            top = by_id[top.parent]
        assert part.category == top.category
        assert "name" not in result.extra[part.id - 1]


def test_what_nobody_named_is_still_an_object(yard: dict) -> None:
    # No "shrub" in the vocabulary: the shrubs come from the class-free pass, unnamed.
    run = _run(yard["splats"], yard["rows"], yard["truth"], named=("tree", "dead tree", "house"))
    result = run["result"]
    objects = _objects(result)
    obj = run["obj"]
    klass = np.asarray(yard["truth"]["class"])[yard["rows"]]
    shrub = klass == list(yard["truth"]["classes"]).index("shrub")
    found = 0
    for o in np.unique(obj[shrub & (obj >= 0)]):
        mine = obj == o
        best, iou = _best(mine, objects)
        assert result.lift.kind[best - 1] == "thing"
        assert "name" not in result.extra[best - 1]
        found += iou >= 0.6
    assert found >= 2
    assert result.lift.stats["leftovers"] >= 2


def test_instances_json_carries_the_concept_fields(run: dict, yard: dict, tmp_path: Path) -> None:
    result = run["result"]
    tiles = {"abc": [1, 10]}
    embedder = ss.FakeEmbedder()
    doc = cs.concept_document(
        result, tiles, embedder=embedder, vocabulary_size=8, vocabulary_model="fixed",
        segmenter="oracle", stand_in=True,
    )  # fmt: skip
    assert doc["format"] == "hexapod.instances" and doc["version"] == 1
    # A stand-in is published under its own name, never as C.
    assert doc["variant"] == cs.VARIANTS["standin"]
    assert doc["variant"]["name"] == "concept-first-standin"
    assert doc["concepts"]["standIn"] is True and doc["concepts"]["cover"] == ["grass", "trail"]
    assert {r["class"] for r in doc["ground"]["cover"]} == {"grass", "trail"}
    assert doc["ground"].keys() >= {"method", "layerM", "cellM", "cover"}
    records = doc["instances"]
    assert [r["id"] for r in records] == list(range(1, len(records) + 1))
    things = {r["name"] for r in records if r.get("nameSource") == "vlm"}
    assert things == {"Tree", "Shrub", "Dead tree", "House"}
    for r in records:
        assert r["kind"] in ("thing", "ground") and r["scaleM"] >= 0
        if r["kind"] == "ground":
            assert r["category"] == cs.GROUND_CATEGORY and r["cover"] in ("grass", "trail")
            assert r["nameSource"] == "ground-cover" and r["behaviour"] == "static"
    ss.write_instances(tmp_path, doc, result.instances)
    again = json.loads((tmp_path / "instances.json").read_text(encoding="utf-8"))
    assert again["instances"][0].keys() >= {"id", "parent", "tags", "category", "kind"}
    sam3 = cs.concept_document(
        result, tiles, embedder=embedder, vocabulary_size=8, vocabulary_model="fixed",
        segmenter="sam3", stand_in=False,
    )  # fmt: skip
    assert sam3["variant"]["name"] == "concept-first"


def test_the_check_sheet_is_written(run: dict, yard: dict, tmp_path: Path) -> None:
    from PIL import Image

    legend = cs.render_check(run["result"], yard["splats"], tmp_path / "check.png")
    with Image.open(tmp_path / "check.png") as image:
        assert image.width == 3 * 512 and image.height > 4 * 384
    assert json.loads((tmp_path / "check.legend.json").read_text()) == legend
    assert len(legend["cover"]) == 2 and legend["objects"]
