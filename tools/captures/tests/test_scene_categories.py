"""Broad scene categories: the committed label mapping and the per-instance assignment."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

import scene_categories as sc
import segment_models as sm

COMMITTED = json.loads(sc.CATEGORIES_FILE.read_text(encoding="utf-8"))


def test_the_committed_file_maps_every_vocabulary_label_to_a_known_category() -> None:
    assert COMMITTED["format"] == sc.FORMAT
    assert COMMITTED["version"] == sc.VERSION
    assert COMMITTED["other"] == sc.OTHER
    ids = [c["id"] for c in COMMITTED["categories"]]
    assert ids == sc.category_ids()
    assert len(set(ids)) == len(ids)
    assert 20 <= len(ids) <= 30
    assert ids[-1] == sc.OTHER
    for c in COMMITTED["categories"]:
        assert re.fullmatch(r"#[0-9a-f]{6}", c["color"]), c
        assert c["name"]
    vocabulary = sm.default_vocabulary()
    assert set(COMMITTED["labels"]) == set(vocabulary)
    assert set(COMMITTED["labels"].values()) <= set(ids)
    # Every category but Other gets labels of its own.
    used = set(COMMITTED["labels"].values())
    assert set(ids) - {sc.OTHER} <= used
    assert sc.main(["--check"]) == 0


@pytest.mark.parametrize(
    ("label", "category"),
    [
        ("pumpkin", "produce"),
        ("gourd", "produce"),
        ("apple", "produce"),
        ("conifer", "trees"),
        ("tree", "trees"),
        ("tree trunk", "trees"),
        ("pine tree", "trees"),
        ("bush", "shrubs"),
        ("deciduous tree", "trees"),
        ("grass", "grass"),
        ("moss", "grass"),
        ("fern", "grass"),
        ("flower arrangement", "flowers"),
        ("log", "wood"),
        ("stump", "wood"),
        ("fallen tree", "wood"),
        ("dirt", "ground"),
        ("forest floor", "ground"),
        ("mud", "ground"),
        ("gravel", "ground"),
        ("rock", "rock"),
        ("stone", "rock"),
        ("stream", "water"),
        ("river", "water"),
        ("snow", "snow"),
        ("fog", "sky"),
        ("sky", "sky"),
        ("cabin", "buildings"),
        ("roof", "buildings"),
        ("shed", "buildings"),
        ("fence", "walls"),
        ("stone wall", "walls"),
        ("pavement", "paths"),
        ("trail", "paths"),
        ("road", "paths"),
        ("fireplug", "fixtures"),
        ("manhole", "fixtures"),
        ("pole", "fixtures"),
        ("car (automobile)", "vehicles"),
        ("bicycle", "vehicles"),
        ("person", "people"),
        ("dog", "animals"),
        ("bird", "animals"),
        ("bench", "furniture"),
        ("picnic table", "furniture"),
        ("chair", "furniture"),
        ("trash can", "containers"),
        ("barrel", "containers"),
    ],
)
def test_common_scene_labels_land_where_a_person_would_look(label: str, category: str) -> None:
    assert COMMITTED["labels"][label] == category


def test_overrides_name_real_labels_and_categories() -> None:
    vocabulary = set(sm.default_vocabulary())
    for label, category in sc.OVERRIDES.items():
        assert label in vocabulary, label
        assert category in sc.category_ids(), category


def test_labels_go_to_the_nearest_category_by_cosine() -> None:
    labels = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    categories = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert sc.assign_labels(labels, categories, ["a", "b"]) == ["a", "b", "b"]


LABELS = {"oak": "trees", "pine": "trees", "lawn": "grass", "bench": "furniture"}


def test_tags_vote_for_a_category_by_score() -> None:
    tags = [
        {"label": "lawn", "score": 0.4},
        {"label": "oak", "score": 0.3},
        {"label": "pine", "score": 0.2},
        {"label": "unknown thing", "score": 0.9},
    ]
    best, share = sc.tag_category(tags, LABELS) or ("", 0.0)
    assert best == "trees"
    assert share == pytest.approx(0.5 / 0.9)
    assert sc.tag_category([{"label": "unknown", "score": 1.0}], LABELS) is None
    assert sc.tag_category([], LABELS) is None


def test_instances_take_their_tags_then_siblings_then_ancestors_then_parts() -> None:
    instances = [
        # A tree with an untagged branch, and a bench tagged inside it.
        {"id": 1, "parent": None, "splats": 10, "tags": [{"label": "oak", "score": 0.5}]},
        {"id": 2, "parent": 1, "splats": 5, "tags": []},
        {"id": 3, "parent": 2, "splats": 5, "tags": [{"label": "bench", "score": 0.4}]},
        # An untagged region whose parts are mostly lawn.
        {"id": 4, "parent": None, "splats": 1, "tags": []},
        {"id": 5, "parent": 4, "splats": 900, "tags": [{"label": "lawn", "score": 0.3}]},
        {"id": 6, "parent": 4, "splats": 100, "tags": [{"label": "pine", "score": 0.3}]},
        # Nothing anywhere.
        {"id": 7, "parent": None, "splats": 3, "tags": []},
        # A coarse "oak" region whose untagged part sits beside a lawn part: lawn.
        {"id": 10, "parent": None, "splats": 50, "tags": [{"label": "oak", "score": 0.3}]},
        {"id": 11, "parent": 10, "splats": 20, "tags": []},
        {"id": 12, "parent": 10, "splats": 80, "tags": [{"label": "lawn", "score": 0.4}]},
    ]
    assert sc.instance_categories(instances, LABELS) == {
        1: "trees",
        2: "trees",
        3: "furniture",
        4: "grass",
        5: "grass",
        6: "trees",
        7: sc.OTHER,
        10: "trees",
        11: "grass",
        12: "grass",
    }
