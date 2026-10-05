"""concept_models.py's pure parts: reading the VLM's answer, naming a detector's phrase,
choosing boxes, painting a cover map. The models themselves need torch and weights and run
on the GPU (infra/modal/segment_concepts.py); nothing here imports them."""

from __future__ import annotations

import numpy as np
import pytest

import concept_models as cm
import concept_scene as cs
import scene_categories


def test_the_prompt_names_every_category_and_the_limits() -> None:
    prompt = cm.vocabulary_prompt(12, max_things=7, max_stuff=3)
    assert "12 views" in prompt and "up to 7" in prompt and "up to 3" in prompt
    for category in scene_categories.category_ids():
        assert category in prompt


def test_an_answer_in_a_code_fence_with_prose_is_read() -> None:
    answer = (
        "Here is the list:\n```json\n"
        '{"things": [{"name": "Cable Spool", "category": "equipment"}, '
        '{"name": "wooden plank", "category": "wood"}, "pumpkin"],\n'
        ' "stuff": [{"name": "gravel", "category": "paths"}, {"name": "dirt", "category": "x"}]}'
        "\n```\nThat is all."
    )
    assert cm.parse_vocabulary(answer) == [
        cs.Concept("cable spool", "thing", "equipment"),
        cs.Concept("wooden plank", "thing", "wood"),
        cs.Concept("pumpkin", "thing", "other"),
        cs.Concept("gravel", "stuff", "paths"),
        cs.Concept("dirt", "stuff", "ground"),
    ]


def test_an_answer_without_json_is_refused() -> None:
    with pytest.raises(ValueError):
        cm.parse_vocabulary("I see a spool and some grass.")
    with pytest.raises(ValueError):
        cm.parse_vocabulary('{"things": [')


def test_a_detector_phrase_names_its_concept() -> None:
    concepts = [
        cs.Concept("cable spool", "thing"),
        cs.Concept("wooden plank", "thing"),
        cs.Concept("plank", "thing"),
        cs.Concept("pumpkin", "thing"),
    ]
    assert cm.match_phrase("cable spool", concepts) == 0
    assert cm.match_phrase("spool", concepts) == 0  # some of the prompt's words
    assert cm.match_phrase("plank", concepts) == 2  # the exact one before the longer
    assert cm.match_phrase("wooden", concepts) == 1
    assert cm.match_phrase("pumpkin spool", concepts) == 0  # most words shared, first
    assert cm.match_phrase("hay", concepts) == -1
    assert cm.match_phrase("", concepts) == -1


def test_boxes_kept_per_concept_without_the_whole_view() -> None:
    boxes = np.array(
        [
            [0, 0, 100, 100],  # the whole 100 x 100 view
            [10, 10, 40, 40],
            [12, 12, 41, 41],  # a repeat of the one above, lower score
            [12, 12, 41, 41],  # the same box, another concept
            [60, 60, 90, 90],
            [5, 5, 9, 9],  # a phrase that named nothing
        ],
        np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.95])
    concepts = np.array([0, 0, 0, 1, 0, -1])
    kept = cm.keep_boxes(boxes, scores, concepts, (100, 100), max_share=0.85, nms_iou=0.6)
    assert kept.tolist() == [1, 3, 4]
    assert cm.box_iou(boxes[1], boxes[2])[0, 0] > 0.6


def test_a_cover_map_paints_finer_masks_over_coarser_on_the_ground_only() -> None:
    region = np.zeros((4, 6), bool)
    region[:, :4] = True
    big = np.ones((4, 6), bool)
    small = np.zeros((4, 6), bool)
    small[1:3, 1:3] = True
    p = np.array([[0.7, 0.3], [0.1, 0.9]])
    out = cm.paint_stuff([small, big], p, region)  # small is class 0, big class 1
    assert out.label[0, 0] == 1 and out.label[1, 1] == 0 and out.label[0, 5] == -1
    assert out.score[1, 1] == pytest.approx(0.7) and out.score[0, 0] == pytest.approx(0.9)
    assert out.score[0, 5] == 0


def test_the_models_satisfy_the_protocols() -> None:
    assert isinstance(cm.QwenVocabulary(), cs.Vocabulary)
    assert isinstance(cm.GroundedSam2Concepts(), cs.ConceptSource)
    assert isinstance(cm.Sam3Concepts(), cs.ConceptSource)
    assert "stand-in" in cm.GroundedSam2Concepts().name
