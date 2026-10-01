"""`segment_models`: everything but the networks, without torch.

The vocabulary and prompts, the mask selection that turns SAM's three candidates per point
into levels, the scoring maths, and the Modal clients against a fake remote that answers
as `infra/modal/world_models.py` does. One smoke test runs the real CPU models and is
skipped where torch and transformers are not installed (they are not in pyproject).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import segment_models as sm
import world_model_client as wmc

HERE = Path(__file__).resolve().parents[1]


# --- vocabulary and prompts -----------------------------------------------------------------


def test_the_vocabulary_is_broad_unique_and_documented() -> None:
    labels = sm.default_vocabulary()
    assert len(labels) >= 1200
    assert len({label.lower() for label in labels}) == len(labels)
    heads = {label.split(" (")[0] for label in labels}  # LVIS: "car (automobile)"
    for word in ("car", "dog", "tree", "bench", "grass", "water", "tent", "person"):
        assert word in heads, word
    assert all(label == label.strip() and "_" not in label for label in labels)
    header = sm.VOCABULARY_FILE.read_text(encoding="utf-8").split("\n", 12)
    assert any("LVIS" in line for line in header) and any("CC BY" in line for line in header)
    assert sm.tag_prompt("Fire Hydrant") == "a photo of a fire hydrant."


def test_every_property_has_several_phrasings_and_a_contrast() -> None:
    wanted = {"movable", "rigid", "elastic", "static", "vegetation", "water", "vehicle"}
    assert set(sm.ATTRIBUTE_PROMPTS) == wanted | {"creature"}
    assert set(sm.ATTRIBUTE_CONTRASTS) == set(sm.ATTRIBUTE_PROMPTS)
    for name, phrases in sm.ATTRIBUTE_PROMPTS.items():
        assert len(phrases) >= 3 and len(set(phrases)) == len(phrases), name
        assert len(sm.ATTRIBUTE_CONTRASTS[name]) >= 2, name


def test_the_module_imports_without_torch() -> None:
    code = "import sys, segment_models; assert 'torch' not in sys.modules, 'torch imported'"
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr


# --- scoring maths, with a fake embedder ----------------------------------------------------


class _WordEmbedder:
    """A text is the bag of the words of `vocab` it contains; an image is a given vector."""

    name = "fake"

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.dim = len(vocab)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        words = [set(t.lower().replace(",", " ").replace(".", " ").split()) for t in texts]
        x = np.array([[w in ws for w in self.vocab] for ws in words], np.float32)
        return sm.l2_normalise(x + 1e-3)

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        return sm.l2_normalise(np.stack(crops).reshape(len(crops), -1))


def test_l2_normalise_gives_unit_rows_and_keeps_zero_rows() -> None:
    x = sm.l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert x.dtype == np.float32
    assert np.allclose(x, [[0.6, 0.8], [0.0, 0.0]])


def test_top_tags_rank_the_matching_label_first() -> None:
    labels = ["dog", "car", "tree"]
    embedder = _WordEmbedder(labels)
    bank = sm.vocabulary_bank(embedder, labels)
    assert bank.shape == (3, 3) and np.allclose(np.linalg.norm(bank, axis=1), 1)
    images = embedder.embed_images([np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.0, 1.0])])
    tags = sm.top_tags(images, bank, labels, k=2)
    assert [t[0]["label"] for t in tags] == ["car", "tree"]
    assert all(t[0]["score"] > t[1]["score"] for t in tags)
    assert all(0 < t[0]["score"] <= 1 for t in tags)


def test_attribute_scores_read_zero_to_one_and_follow_the_prompts() -> None:
    words = sorted(
        {w for group in (*sm.ATTRIBUTE_PROMPTS.values(), *sm.ATTRIBUTE_CONTRASTS.values())
         for p in group for w in p.lower().replace(",", " ").split()}
    )  # fmt: skip
    embedder = _WordEmbedder(words)
    banks = sm.attribute_banks(embedder)
    assert banks.names == list(sm.ATTRIBUTE_PROMPTS)
    # An "image" that is exactly the averaged vegetation phrasing.
    leafy = banks.positive[banks.names.index("vegetation")][None]
    scores = sm.attribute_scores(leafy, banks)
    assert all(v.shape == (1,) and 0 <= v[0] <= 1 for v in scores.values())
    assert scores["vegetation"][0] > 0.9


# --- masks: grid, levels, selection ---------------------------------------------------------


def test_point_grid_is_cell_centres() -> None:
    g = sm.point_grid(4)
    assert g.shape == (16, 2)
    assert np.allclose(np.unique(g[:, 0]), [0.125, 0.375, 0.625, 0.875])


def test_levels_go_from_the_largest_candidate_down() -> None:
    areas = np.array([[10, 500, 80], [300, 300, 5]])
    assert sm.granularity_levels(areas).tolist() == [[2, 0, 1], [0, 1, 2]]


def _box(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    logits = np.full((h, w), -10.0, np.float32)
    logits[y0:y1, x0:x1] = 10.0
    return logits


def test_selection_keeps_one_mask_per_thing_per_level() -> None:
    h = w = 32
    whole = _box(h, w, 4, 28, 4, 28)
    part = _box(h, w, 4, 16, 4, 28)
    sub = _box(h, w, 4, 10, 4, 10)
    other = _box(h, w, 0, 3, 0, 32)
    # Two points on the same thing (same three candidates, any order), one on another.
    logits = np.stack([[sub, whole, part], [part, sub, whole], [other, other, other]])
    iou = np.array([[0.9, 0.95, 0.9], [0.92, 0.9, 0.96], [0.9, 0.9, 0.9]], np.float32)
    kept, scores = sm.select_masks(
        logits,
        iou,
        pred_iou_thresh=0.7,
        stability_thresh=0.8,
        nms_iou=0.7,
        min_area=0.001,
        repeat_iou=0.9,
    )
    levels = [lv for _, _, lv in kept]
    assert levels == sorted(levels)  # coarsest first
    masks = {(lv, int((logits[p, c] > 0).sum())) for p, c, lv in kept}
    # whole (576 px) at level 0, part (288) at 1, sub (36) at 2: once each, the duplicate
    # point merged by NMS; `other` is a whole with no parts, so only its level-0 mask.
    assert masks == {(0, 576), (1, 288), (2, 36), (0, 96)}
    assert all(0 <= s <= 1 for s in scores)


def test_selection_drops_low_confidence_unstable_and_tiny_masks() -> None:
    h = w = 32
    good = _box(h, w, 0, 16, 0, 16)
    tiny = _box(h, w, 0, 1, 0, 1)
    soft = np.full((h, w), 0.5, np.float32)  # > 0 everywhere but far from stable
    logits = np.stack([[good, tiny, soft]])
    iou = np.array([[0.5, 0.99, 0.99]], np.float32)
    kept, _ = sm.select_masks(
        logits,
        iou,
        pred_iou_thresh=0.7,
        stability_thresh=0.85,
        nms_iou=0.7,
        min_area=0.01,
        repeat_iou=0.9,
    )
    assert kept == []


def test_mask_nms_prefers_the_better_score() -> None:
    a = np.zeros((3, 8, 8), bool)
    a[0, :4] = a[1, :5] = a[2, 6:] = True
    assert sm.mask_nms(a, np.array([0.5, 0.9, 0.1]), 0.7).tolist() == [1, 2]


# --- Modal clients --------------------------------------------------------------------------


def test_masks_round_trip_through_the_npz() -> None:
    rng = np.random.default_rng(0)
    masks = [sm.Mask(rng.random((7, 13)) > 0.5, lv, 0.5 + lv / 10) for lv in (0, 1, 2)]
    back = sm.decode_masks(sm.encode_masks(masks, (7, 13)))
    assert [m.level for m in back] == [0, 1, 2]
    assert np.allclose([m.score for m in back], [0.5, 0.6, 0.7])
    assert all(np.array_equal(a.mask, b.mask) and b.mask.dtype == bool for a, b in zip(masks, back))
    assert sm.decode_masks(sm.encode_masks([], (7, 13))) == []


def test_modal_sam2_sends_a_png_and_reads_levelled_masks() -> None:
    rgb = np.zeros((24, 40, 3), np.uint8)
    rgb[:, 20:] = 200

    def remote(cls: str, method: str, request: dict) -> dict:
        assert (cls, method) == ("SegmentMasks", "masks")
        assert request["model"] == sm.SAM2_MODEL and request["points_per_side"] == 32
        (png,) = request["images"]
        image = wmc.decode_png(png)
        assert np.array_equal(image, rgb)
        right = image[..., 0] > 100
        out = [sm.Mask(np.ones(right.shape, bool), 0, 0.9), sm.Mask(right, 1, 0.8)]
        return {"masks": [sm.encode_masks(out, right.shape)], "model": request["model"]}

    source = sm.ModalSam2Masks(remote=remote)
    assert source.name == f"sam2:{sm.SAM2_MODEL}"
    masks = source.masks(rgb)
    assert [m.level for m in masks] == [0, 1]
    assert masks[1].mask[:, 20:].all() and not masks[1].mask[:, :20].any()


def test_modal_sam2_refuses_masks_of_another_size() -> None:
    def remote(cls: str, method: str, request: dict) -> dict:
        return {"masks": [sm.encode_masks([sm.Mask(np.ones((5, 5), bool), 0, 1.0)], (5, 5))]}

    with pytest.raises(ValueError, match="masks of"):
        sm.ModalSam2Masks(remote=remote).masks(np.zeros((8, 8, 3), np.uint8))


def test_modal_siglip_round_trips_and_normalises() -> None:
    calls: list[tuple[str, dict]] = []

    def remote(cls: str, method: str, request: dict) -> dict:
        assert cls == "SegmentEmbed"
        calls.append((method, request))
        n = len(request["images"] if method == "embed_images" else request["texts"])
        x = np.arange(n * 768, dtype=np.float32).reshape(n, 768) + 1
        return {"embeddings": sm.encode_array(x), "model": request["model"]}

    embedder = sm.ModalSiglipEmbedder(remote=remote)
    crops = [np.full((10, 12, 3), k * 50, np.uint8) for k in range(3)]
    x = embedder.embed_images(crops)
    y = embedder.embed_texts(["a dog", "a tree"])
    assert x.shape == (3, 768) and y.shape == (2, 768) and x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1) and np.allclose(np.linalg.norm(y, axis=1), 1)
    assert [m for m, _ in calls] == ["embed_images", "embed_texts"]
    assert np.array_equal(wmc.decode_png(calls[0][1]["images"][2]), crops[2])
    assert calls[1][1]["texts"] == ["a dog", "a tree"]


def test_modal_siglip_refuses_the_wrong_shape() -> None:
    def remote(cls: str, method: str, request: dict) -> dict:
        return {"embeddings": sm.encode_array(np.ones((1, 512), np.float32))}

    with pytest.raises(ValueError, match="embeddings of"):
        sm.ModalSiglipEmbedder(remote=remote).embed_texts(["a dog"])


# --- the real models (CPU), where installed -------------------------------------------------


def test_real_models_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rgb = np.full((240, 320, 3), (120, 170, 220), np.uint8)  # sky
    rgb[150:] = (70, 120, 50)  # grass
    rgb[60:180, 120:200] = (180, 40, 30)  # a red box
    masks = sm.Sam2Masks(points_per_side=8).masks(rgb)
    assert masks and all(m.mask.shape == (240, 320) and m.mask.dtype == bool for m in masks)
    assert {m.level for m in masks} <= {0, 1, 2}
    embedder = sm.SiglipEmbedder()
    x = embedder.embed_images([rgb[60:180, 120:200]])
    y = embedder.embed_texts(["a photo of a red box.", "a photo of a dog."])
    assert x.shape == (1, embedder.dim) and np.allclose(np.linalg.norm(x, axis=1), 1)
    assert (x @ y.T)[0, 0] > (x @ y.T)[0, 1]
