"""concept_models.py's pure parts: reading the VLM's answer, naming a detector's phrase,
choosing boxes, painting a cover map. The models themselves need torch and weights and run
on the GPU (infra/modal/segment.py, variant `concept-first-standin`); nothing here imports
them."""

from __future__ import annotations

import numpy as np
import pytest

import concept_models as cm
import concept_scene as cs
import scene_categories
import segment_ground_first as sgf


def test_the_prompt_names_every_category_cover_class_and_the_limits() -> None:
    prompt = cm.vocabulary_prompt(12, max_things=7, max_stuff=3)
    assert "12 views" in prompt and "up to 7" in prompt and "up to 3" in prompt
    for category in scene_categories.category_ids():
        assert category in prompt
    for cover in sgf.cover_classes()[0]:
        assert cover.id in prompt


def test_an_answer_in_a_code_fence_with_prose_is_read() -> None:
    answer = (
        "Here is the list:\n```json\n"
        '{"things": [{"name": "Cable Spool", "also": ["wooden reel", "drum"], '
        '"category": "equipment"}, {"name": "wooden plank", "category": "wood"}, "pumpkin"],\n'
        ' "cover": ["gravel", "dirt", "straw", "lava"]}'
        "\n```\nThat is all."
    )
    things, stuff = cs.split_concepts(cm.parse_vocabulary(answer))
    assert things == [
        cs.Concept("cable spool", "thing", "equipment", ("cable spool", "wooden reel", "drum")),
        cs.Concept("wooden plank", "thing", "wood", ("wooden plank",)),
        cs.Concept("pumpkin", "thing", "other", ("pumpkin",)),
    ]
    assert [c.cover for c in stuff] == ["gravel", "dirt", "hay"]  # "lava" names no class


def test_an_answer_without_json_is_refused() -> None:
    with pytest.raises(ValueError):
        cm.parse_vocabulary("I see a spool and some grass.")
    with pytest.raises(ValueError):
        cm.parse_vocabulary('{"things": [')


def test_a_detector_phrase_names_its_concept_by_any_of_its_words() -> None:
    concepts = [
        cs.Concept("cable spool", "thing", prompts=("cable spool", "wooden reel")),
        cs.Concept("wooden plank", "thing"),
        cs.Concept("plank", "thing"),
        cs.Concept("pumpkin", "thing"),
    ]
    assert cm.match_phrase("cable spool", concepts) == 0
    assert cm.match_phrase("reel", concepts) == 0  # another word of it
    assert cm.match_phrase("spool", concepts) == 0  # some of the prompt's words
    assert cm.match_phrase("plank", concepts) == 2  # the exact one before the longer
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
    # One object found under two names keeps the better name.
    kept = cm.keep_boxes(
        boxes, scores, concepts, (100, 100), max_share=0.85, nms_iou=0.6, cross_iou=0.8
    )
    assert kept.tolist() == [1, 4]


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
    # A mask whose best is a contrast prompt (not ground cover) paints nothing.
    rows = cm.classify(np.array([[0.2, 0.1, 0.7], [0.6, 0.3, 0.1]]), classes=2)
    assert rows[0].tolist() == [-1.0, -1.0] and rows[1].tolist() == [0.6, 0.3]
    out = cm.paint_stuff([small, big], rows[::-1], region)  # small: not cover; big: class 0
    assert out.label[1, 1] == 0 and out.score[1, 1] == pytest.approx(0.6)


def test_tiles_cover_the_ground_no_mask_did() -> None:
    region = np.zeros((8, 8), bool)
    region[:, :6] = True
    covered = np.zeros((8, 8), bool)
    covered[:4, :4] = True
    tiles = cm.ground_tiles(region, covered, side=4, least=0.4)
    assert len(tiles) == 3  # the covered tile is left out
    assert all((t & ~region).sum() == 0 and (t & covered).sum() == 0 for t in tiles)


def test_the_models_satisfy_the_protocols() -> None:
    assert isinstance(cm.QwenVocabulary(), cs.Vocabulary)
    assert isinstance(cm.GroundedSam2Concepts(), cs.ConceptSource)
    assert isinstance(cm.Sam3Concepts(), cs.ConceptSource)
    assert "stand-in" in cm.GroundedSam2Concepts().name


def test_sam3_objects_take_the_concept_of_the_prompt_that_found_them() -> None:
    prompts = ["cable spool", "rock"]
    found = {"rock": [3, 5], "cable spool": [1], "a word nobody asked": [9]}
    assert cm.prompt_concepts(found, prompts) == {3: 1, 5: 1, 1: 0}


def test_sam3_cover_is_the_most_probable_class_on_the_ground_only() -> None:
    logits = np.array([[[4.0, -4.0], [0.2, -4.0]], [[-4.0, 4.0], [0.1, -4.0]]])
    region = np.array([[True, True], [True, False]])
    cover = cm.cover_from_logits(logits, region, threshold=0.5)
    # (1, 0): both classes near 0.5, the first above; (1, 1): off the ground.
    assert cover.label.tolist() == [[0, 1], [0, -1]]
    assert cover.score[0, 0] > 0.98 and cover.score[1, 1] == 0
