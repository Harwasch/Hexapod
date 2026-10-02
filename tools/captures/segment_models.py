"""The general models segmentation runs on (SCENE_OBJECTS.md §3 steps 2, 4 and 5): class-free
masks at several granularities, an image-text embedding, an open vocabulary to tag with, and
attribute prompts to score properties with. None of them knows our scenes.

* `Sam2Masks` -- a `segment_scene.MaskSource`: SAM 2.1 (`facebook/sam2.1-hiera-tiny`,
  Apache-2.0) prompted with a grid of single points. Each point gives SAM's three candidate
  masks (its answer to "which thing did you mean": subpart, part, whole); sorted by area
  they are `level` 2, 1 and 0. Candidates are kept by SAM's predicted IoU and by stability
  (the mask barely changes when the logit threshold moves), then de-duplicated per level by
  mask IoU, and a finer mask that repeats a coarser one is dropped.
* `SiglipEmbedder` -- a `segment_scene.Embedder`: SigLIP 2 (`google/siglip2-base-patch16-224`,
  Apache-2.0), 768-d, image and text in one space, L2-normalised.
* `default_vocabulary()` -- 1300 labels to tag with: LVIS v1's 1203 categories, COCO-Stuff's
  stuff labels and a few outdoor words (`data/open_vocabulary.txt` says where each is from).
* `ATTRIBUTE_PROMPTS` / `ATTRIBUTE_CONTRASTS` -- a few phrasings per property (movable, rigid,
  elastic, static, vegetation, water, vehicle, creature), averaged; a property's score is a
  two-way softmax of the crop against its prompts and their contrasts, so it reads 0..1.
* `ModalSam2Masks`, `ModalSiglipEmbedder` -- the same two, run on a Modal GPU by
  `infra/modal/world_models.py` (`SegmentMasks`, `SegmentEmbed`), which runs *this* module.

Torch and transformers are imported only when a model is first used, so the module (and its
tests) import without them. They are not in `pyproject.toml`; on a CPU:

    uv run --with torch --with torchvision --with transformers \\
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        python -c "import segment_models as sm; print(sm.Sam2Masks().masks(rgb))"

Measured on 4 idle CPU cores (2026-10-01, transformers 5.18, torch 2.14.1+cpu, 640x480
splat renders): see `Sam2Masks.__doc__` and `SiglipEmbedder.__doc__`. Under a shared,
oversubscribed CPU, torch's threads spin and everything is 5-30x slower.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

import world_model_client as wmc
from segment_scene import Mask

VOCABULARY_FILE = Path(__file__).resolve().parent / "data" / "open_vocabulary.txt"

SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

#: How a label is asked about. SigLIP 2 was trained on lower-case captions.
TAG_TEMPLATE = "a photo of a {}."
#: Softmax temperature for tags and properties: cosine similarities are small and close
#: together, so they are sharpened as CLIP-style zero-shot classification does.
TEMPERATURE = 100.0

#: SCENE_OBJECTS.md §3 step 5. Each property: a few phrasings of "it is", averaged.
ATTRIBUTE_PROMPTS: dict[str, tuple[str, ...]] = {
    "movable": (
        "a photo of an object that can be moved or carried",
        "a photo of a loose object that could be picked up or driven away",
        "a photo of something portable",
    ),
    "rigid": (
        "a photo of a hard, rigid object",
        "a photo of something solid that does not bend",
        "a photo of a stiff object made of metal, stone, wood or plastic",
    ),
    "elastic": (
        "a photo of something flexible that bends and sways",
        "a photo of a soft, elastic object that deforms",
        "a photo of leaves, branches, cloth or grass that move in the wind",
    ),
    "static": (
        "a photo of a fixed structure that never moves",
        "a photo of a building, wall, road or the ground",
        "a photo of something built into the ground",
    ),
    "vegetation": (
        "a photo of vegetation",
        "a photo of a tree, bush, grass or plants",
        "a photo of leaves and foliage",
    ),
    "water": (
        "a photo of water",
        "a photo of a lake, river, pond or puddle",
        "a photo of a water surface",
    ),
    "vehicle": (
        "a photo of a vehicle",
        "a photo of a car, truck, bicycle, boat or trailer",
        "a photo of a machine for transport",
    ),
    "creature": (
        "a photo of a person or an animal",
        "a photo of a living creature",
        "a photo of a human, dog, bird or other animal",
    ),
}
#: What each property's prompts are weighed against: the same kind of phrase, saying "not".
ATTRIBUTE_CONTRASTS: dict[str, tuple[str, ...]] = {
    "movable": (
        "a photo of something fixed in place",
        "a photo of the ground, a building or a large tree",
    ),
    "rigid": (
        "a photo of something soft and flexible",
        "a photo of leaves, cloth or water",
    ),
    "elastic": (
        "a photo of a hard, rigid object",
        "a photo of stone, metal or concrete",
    ),
    "static": (
        "a photo of a small loose object",
        "a photo of a vehicle, a person or an animal",
    ),
    "vegetation": (
        "a photo of a man-made object",
        "a photo of bare ground, rock, sky or a building",
    ),
    "water": (
        "a photo of dry land",
        "a photo of a solid object",
    ),
    "vehicle": (
        "a photo of a building or the ground",
        "a photo of a plant or a household object",
    ),
    "creature": (
        "a photo of an inanimate object",
        "a photo of a plant, the ground or a building",
    ),
}


# --- the vocabulary and the prompts ---------------------------------------------------------


@cache
def _vocabulary() -> tuple[str, ...]:
    lines = VOCABULARY_FILE.read_text(encoding="utf-8").splitlines()
    return tuple(s.strip() for s in lines if s.strip() and not s.startswith("#"))


def default_vocabulary() -> list[str]:
    """The labels tags are scored against (LVIS v1 + COCO-Stuff + a few outdoor words)."""
    return list(_vocabulary())


def tag_prompt(label: str) -> str:
    return TAG_TEMPLATE.format(label.lower())


def text_bank(embedder: Any, phrases: Sequence[Sequence[str]]) -> np.ndarray:
    """(len(phrases), dim): each group of phrasings embedded, averaged, re-normalised."""
    flat = [p for group in phrases for p in group]
    emb = embedder.embed_texts(flat)
    out, k = [], 0
    for group in phrases:
        out.append(emb[k : k + len(group)].mean(axis=0))
        k += len(group)
    return l2_normalise(np.stack(out))


def vocabulary_bank(embedder: Any, labels: Sequence[str] | None = None) -> np.ndarray:
    """(len(labels), dim) text embeddings of the vocabulary (default: `default_vocabulary`)."""
    labels = default_vocabulary() if labels is None else labels
    return text_bank(embedder, [[tag_prompt(label)] for label in labels])


def top_tags(
    image_emb: np.ndarray,
    bank: np.ndarray,
    labels: Sequence[str],
    k: int = 5,
    temperature: float = TEMPERATURE,
) -> list[list[dict[str, float | str]]]:
    """Per image row, the `k` best labels as `{"label", "score"}`, descending; the score is
    the softmax over the whole vocabulary (so it is comparable between instances)."""
    p = _softmax(temperature * (np.atleast_2d(image_emb) @ bank.T))
    best = np.argsort(-p, axis=1, kind="stable")[:, :k]
    return [
        [{"label": labels[j], "score": round(float(row[j]), 4)} for j in idx]
        for row, idx in zip(p, best, strict=True)
    ]


@dataclass
class AttributeBanks:
    names: list[str]
    positive: np.ndarray  # (a, dim)
    negative: np.ndarray  # (a, dim)


def attribute_banks(embedder: Any) -> AttributeBanks:
    names = list(ATTRIBUTE_PROMPTS)
    return AttributeBanks(
        names,
        text_bank(embedder, [ATTRIBUTE_PROMPTS[n] for n in names]),
        text_bank(embedder, [ATTRIBUTE_CONTRASTS[n] for n in names]),
    )


def attribute_scores(
    image_emb: np.ndarray, banks: AttributeBanks, temperature: float = TEMPERATURE
) -> dict[str, np.ndarray]:
    """Per property, (n,) in 0..1: the softmax of "it is" against "it is not"."""
    x = np.atleast_2d(image_emb)
    pos = temperature * (x @ banks.positive.T)
    neg = temperature * (x @ banks.negative.T)
    p = 1.0 / (1.0 + np.exp(neg - pos))
    return {name: p[:, a].astype(np.float32) for a, name in enumerate(banks.names)}


class ZeroShotScoring:
    """`segment_scene.describe` scores tags and properties through these when an embedder
    has them, so the model's own prompts are used (`TAG_TEMPLATE`, several phrasings and
    contrasts per property) rather than bare words. Text banks are built once per run."""

    def score_tags(self, embedding: np.ndarray, labels: Sequence[str]) -> np.ndarray:
        """(n, len(labels)) probabilities: softmax over the labels' prompts."""
        cache = self.__dict__.setdefault("_banks", {})
        key = ("tags", tuple(labels))
        if key not in cache:
            cache[key] = vocabulary_bank(self, list(labels))
        logits = TEMPERATURE * np.atleast_2d(embedding) @ cache[key].T
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def score_properties(self, embedding: np.ndarray) -> dict[str, np.ndarray]:
        """Per property (`ATTRIBUTE_PROMPTS`), (n,) in 0..1."""
        cache = self.__dict__.setdefault("_banks", {})
        if "attributes" not in cache:
            cache["attributes"] = attribute_banks(self)
        return attribute_scores(np.atleast_2d(embedding), cache["attributes"])


# --- small pure helpers ---------------------------------------------------------------------


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero), float32."""
    x = np.asarray(x, np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def point_grid(n_per_side: int) -> np.ndarray:
    """(n², 2) points (x, y) in [0, 1], cell centres -- SAM's own automatic grid."""
    c = (np.arange(n_per_side) + 0.5) / n_per_side
    x, y = np.meshgrid(c, c)
    return np.stack([x.ravel(), y.ravel()], axis=1)


def candidate_levels(areas: np.ndarray, pixels: int, max_area: float) -> tuple[np.ndarray, ...]:
    """Per candidate (points, k): its level among its point's candidates that are not the
    whole image, and whether it is one of them (area at most `max_area` of `pixels`). A
    point whose widest answer is the whole view has its next answer as its whole."""
    small = areas <= max_area * pixels
    return granularity_levels(np.where(small, areas, -1)), small


def granularity_levels(areas: np.ndarray) -> np.ndarray:
    """`areas` (points, k): the k candidate masks of each point prompt. Their level, by area
    within the point: largest 0 (whole), then 1 (part), ... k-1 (subpart). Ties keep order."""
    order = np.argsort(-np.asarray(areas), axis=1, kind="stable")
    levels = np.empty_like(order)
    np.put_along_axis(levels, order, np.arange(order.shape[1])[None, :], axis=1)
    return levels


def stability(logits: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """SAM's stability score: IoU of the mask thresholded at +offset and at -offset."""
    flat = logits.reshape(logits.shape[0], -1)
    inner = (flat > offset).sum(axis=1)
    outer = (flat > -offset).sum(axis=1)
    return inner / np.maximum(outer, 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) IoU of bool masks a (n, h, w) and b (m, h, w)."""
    fa = a.reshape(a.shape[0], -1).astype(np.float32)
    fb = b.reshape(b.shape[0], -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept, best score first: a mask goes if it overlaps a better kept one > iou."""
    order = np.argsort(-scores, kind="stable")
    if order.size == 0:
        return order
    overlap = mask_iou(masks[order], masks[order])
    keep: list[int] = []
    for i in range(order.size):
        if all(overlap[i, j] <= iou for j in keep):
            keep.append(i)
    return order[keep]


def select_masks(
    logits: np.ndarray,
    iou_pred: np.ndarray,
    *,
    pred_iou_thresh: float,
    stability_thresh: float,
    nms_iou: float,
    min_area: float,
    repeat_iou: float,
    max_area: float = 1.0,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """From candidate logits (points, k, h, w) and SAM's predicted IoU (points, k): the kept
    `(point, candidate, level)` and their scores, coarsest level first, best score first.

    Candidates over `max_area` of the image are left out before the levels are counted
    (`candidate_levels`). Per level: thresholds, then NMS by mask IoU. Then a finer mask
    whose IoU with a kept coarser one is above `repeat_iou` is dropped (a whole with no
    parts is one mask)."""
    p, k = iou_pred.shape
    binary = logits > 0
    areas = binary.reshape(p, k, -1).sum(axis=2)
    pixels = logits.shape[2] * logits.shape[3]
    levels, small = candidate_levels(areas, pixels, max_area)
    stab = stability(logits.reshape(p * k, *logits.shape[2:])).reshape(p, k)
    min_px = min_area * pixels
    ok = (iou_pred >= pred_iou_thresh) & (stab >= stability_thresh) & (areas >= min_px) & small
    pts, cand = np.nonzero(ok)
    if pts.size == 0:
        return [], []
    flat = binary[pts, cand]
    return choose_masks(
        pts,
        cand,
        levels[pts, cand],
        iou_pred[pts, cand],
        stab[pts, cand],
        mask_iou(flat, flat),
        nms_iou=nms_iou,
        repeat_iou=repeat_iou,
    )


def choose_masks(
    pts: np.ndarray,
    cand: np.ndarray,
    levels: np.ndarray,
    iou_pred: np.ndarray,
    stab: np.ndarray,
    overlap: np.ndarray,
    *,
    nms_iou: float,
    repeat_iou: float,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """`select_masks`' choice among candidates that passed the thresholds: per level
    (coarsest first) NMS by `overlap` (their mask IoU, (n, n)), then repeats of a kept
    coarser mask dropped. The heavy part, `overlap`, can come from a GPU."""
    kept: list[tuple[int, int, int]] = []
    scores: list[float] = []
    kept_rows: list[int] = []
    for level in range(int(levels.max()) + 1 if levels.size else 0):
        rows = np.flatnonzero(levels == level)
        if rows.size == 0:
            continue
        score = iou_pred[rows] * stab[rows]
        order = np.argsort(-score, kind="stable")
        sub = overlap[np.ix_(rows[order], rows[order])]
        # Greedy NMS: a mask is kept unless a better kept one overlaps it (IoU is symmetric).
        keep: list[int] = []
        suppressed = np.zeros(order.size, bool)
        for i in range(order.size):
            if not suppressed[i]:
                keep.append(i)
                suppressed |= sub[i] > nms_iou
        chosen = rows[order[keep]]
        if kept_rows and chosen.size:
            repeat = overlap[np.ix_(chosen, kept_rows)].max(axis=1) > repeat_iou
            chosen = chosen[~repeat]
        for c in chosen:
            kept.append((int(pts[c]), int(cand[c]), level))
            scores.append(float(iou_pred[c]))
            kept_rows.append(int(c))
    return kept, scores


#: Torch's CPU threads beside a GPU (pre- and post-processing only).
GPU_HOST_THREADS = 4


def _device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- SAM 2.1 --------------------------------------------------------------------------------


@dataclass
class Sam2Masks:
    """SAM 2.1 automatic masks at three granularities (transformers' `Sam2Model`).

    On 4 idle CPU cores, hiera-tiny, a 640x480 image: the image encoder (SAM's fixed 1024
    input) about 1.5 s, then about 0.06 s per point prompt -- the mask decoder's two-way
    transformer runs over all 64x64 image tokens once per point. So 16 points a side about
    17 s, 32 a side (SAM's default, used here and by `ModalSam2Masks`) about 67 s; on a GPU
    it is fast either way, and 32 a side finds the small things a large scan is full of.

    The thresholds are looser than SAM's automatic generator (0.88 / 0.95): splat renders
    are noisy, and at SAM's own values a camp view kept a handful of masks; at 0.6 / 0.8
    the canopies, the roof, the bushes and the ground each came back."""

    model: str = SAM2_MODEL
    points_per_side: int = 32
    points_per_batch: int = 128
    pred_iou_thresh: float = 0.6
    stability_thresh: float = 0.8
    nms_iou: float = 0.7
    #: Smallest mask kept, as a fraction of the image.
    min_area: float = 0.0005
    #: Largest: a candidate over this share of the image is the whole view, not a thing in
    #: it; its point's next answer is its whole (`candidate_levels`).
    max_area: float = 0.8
    #: A finer mask this similar to a kept coarser one is the same thing, dropped.
    repeat_iou: float = 0.9
    device: str | None = None
    _loaded: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"sam2:{self.model}"

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            import torch
            from transformers import Sam2Model, Sam2Processor

            device = _device(self.device)
            processor = Sam2Processor.from_pretrained(self.model)
            model = Sam2Model.from_pretrained(self.model).to(device).eval()
            if device == "cpu":
                torch.set_num_threads(max(1, torch.get_num_threads()))
            else:
                # The GPU does the work; CPU threads would only contend with the renderers.
                torch.set_num_threads(GPU_HOST_THREADS)
            self._loaded = (processor, model, device)
        return self._loaded

    def masks(self, rgb: np.ndarray) -> list[Mask]:
        import torch

        processor, model, device = self._load()
        h, w = rgb.shape[:2]
        inputs = processor(images=np.ascontiguousarray(rgb, np.uint8), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        side = pixel_values.shape[-1]  # SAM's input is a stretched square (1024)
        grid = torch.tensor(point_grid(self.points_per_side) * side, dtype=torch.float32)
        logits, ious = [], []
        with torch.inference_mode():
            embeddings = model.get_image_embeddings(pixel_values)
            for start in range(0, grid.shape[0], self.points_per_batch):
                pts = grid[start : start + self.points_per_batch].to(device)
                out = model(
                    image_embeddings=embeddings,
                    input_points=pts[None, :, None, :],
                    input_labels=torch.ones(1, pts.shape[0], 1, dtype=torch.long, device=device),
                    multimask_output=True,
                )
                logits.append(out.pred_masks[0].float())
                ious.append(out.iou_scores[0].float())
        low = torch.cat(logits)  # (points, 3, 256, 256), on the model's device
        iou_pred = torch.cat(ious)
        p, k = iou_pred.shape
        # The candidates' statistics on the device (the bulk of the work), the choice on
        # the CPU: the same as `select_masks` on the whole array.
        binary = low > 0
        areas = binary.reshape(p, k, -1).sum(dim=2)
        inner = (low > 1.0).reshape(p, k, -1).sum(dim=2)  # `stability`, offset 1
        outer = (low > -1.0).reshape(p, k, -1).sum(dim=2)
        stab = inner.double() / outer.clamp(min=1).double()
        areas_np = areas.cpu().numpy()
        pixels = low.shape[2] * low.shape[3]
        levels, small = candidate_levels(areas_np, pixels, self.max_area)
        stab_np = stab.cpu().numpy()
        iou_np = iou_pred.cpu().numpy()
        ok = (
            (iou_np >= self.pred_iou_thresh)
            & (stab_np >= self.stability_thresh)
            & (areas_np >= self.min_area * pixels)
            & small
        )
        pts, cand = np.nonzero(ok)
        if pts.size == 0:
            return []
        index = torch.as_tensor(pts, device=low.device), torch.as_tensor(cand, device=low.device)
        flat = binary[index].reshape(pts.size, -1).float()
        inter = flat @ flat.T
        size = flat.sum(dim=1)
        overlap = (inter / (size[:, None] + size[None, :] - inter).clamp(min=1.0)).cpu().numpy()
        chosen, scores = choose_masks(
            pts,
            cand,
            levels[pts, cand],
            iou_np[pts, cand],
            stab_np[pts, cand],
            overlap,
            nms_iou=self.nms_iou,
            repeat_iou=self.repeat_iou,
        )
        if not chosen:
            return []
        pick = torch.stack([low[pt, cd] for pt, cd, _ in chosen])[:, None]
        full = torch.nn.functional.interpolate(pick, size=(h, w), mode="bilinear")[:, 0] > 0
        full_np = full.cpu().numpy()
        return [
            Mask(full_np[i], level, float(np.clip(score, 0.0, 1.0)))
            for i, ((_, _, level), score) in enumerate(zip(chosen, scores, strict=True))
            if full_np[i].any()
        ]


# --- SigLIP 2 -------------------------------------------------------------------------------


@dataclass
class SiglipEmbedder(ZeroShotScoring):
    """SigLIP 2 image and text embeddings (transformers' `AutoModel`), L2-normalised.

    On 4 idle CPU cores, base-patch16-224: about 0.13 s an image crop (batches of 32) and
    0.036 s a text (the 1300-label vocabulary in about 47 s, once per run)."""

    model: str = SIGLIP_MODEL
    dim: int = 768
    batch: int = 32
    device: str | None = None
    _loaded: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"siglip:{self.model}"

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            from transformers import AutoModel, AutoProcessor

            device = _device(self.device)
            processor = AutoProcessor.from_pretrained(self.model)
            model = AutoModel.from_pretrained(self.model).to(device).eval()
            self._loaded = (processor, model, device)
        return self._loaded

    def _features(self, out: Any) -> np.ndarray:
        tensor = out if hasattr(out, "shape") else out.pooler_output
        return tensor.float().cpu().numpy()

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        processor, model, device = self._load()
        rows = [np.zeros((0, self.dim), np.float32)]
        for start in range(0, len(crops), self.batch):
            chunk = [np.ascontiguousarray(c, np.uint8) for c in crops[start : start + self.batch]]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            with torch.inference_mode():
                rows.append(self._features(model.get_image_features(**inputs)))
        return l2_normalise(np.concatenate(rows))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        processor, model, device = self._load()
        rows = [np.zeros((0, self.dim), np.float32)]
        for start in range(0, len(texts), self.batch * 4):
            chunk = [t.lower() for t in texts[start : start + self.batch * 4]]
            inputs = processor(
                text=chunk, padding="max_length", max_length=64, truncation=True,
                return_tensors="pt",
            ).to(device)  # fmt: skip
            with torch.inference_mode():
                rows.append(self._features(model.get_text_features(**inputs)))
        return l2_normalise(np.concatenate(rows))


# --- the same, on a Modal GPU ---------------------------------------------------------------


def encode_masks(masks: Sequence[Mask], shape: tuple[int, int]) -> bytes:
    """Masks as a compressed npz: `masks` bit-packed (n, h*w), `levels`, `scores`, `shape`."""
    h, w = shape
    stack = np.zeros((len(masks), h * w), bool)
    for i, m in enumerate(masks):
        stack[i] = np.asarray(m.mask, bool).reshape(-1)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        masks=np.packbits(stack, axis=1),
        levels=np.array([m.level for m in masks], np.int16),
        scores=np.array([m.score for m in masks], np.float32),
        shape=np.array([h, w], np.int32),
    )
    return buffer.getvalue()


def decode_masks(blob: bytes) -> list[Mask]:
    with np.load(io.BytesIO(blob)) as z:
        h, w = (int(v) for v in z["shape"])
        bits = np.unpackbits(z["masks"], axis=1, count=h * w).astype(bool)
        return [
            Mask(bits[i].reshape(h, w), int(level), float(score))
            for i, (level, score) in enumerate(zip(z["levels"], z["scores"], strict=True))
        ]


def encode_array(x: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, x=np.asarray(x))
    return buffer.getvalue()


def decode_array(blob: bytes) -> np.ndarray:
    with np.load(io.BytesIO(blob)) as z:
        return z["x"]


@dataclass
class ModalSam2Masks:
    """`Sam2Masks` on a Modal GPU (`SegmentMasks.masks`): the same code, a denser grid."""

    model: str = SAM2_MODEL
    points_per_side: int = 32
    remote: wmc.Remote = wmc.modal_remote

    @property
    def name(self) -> str:
        return f"sam2:{self.model}"

    def masks(self, rgb: np.ndarray) -> list[Mask]:
        request = {
            "images": [wmc.encode_png(rgb)],
            "model": self.model,
            "points_per_side": self.points_per_side,
        }
        (blob,) = self.remote("SegmentMasks", "masks", request)["masks"]
        out = decode_masks(blob)
        if out and out[0].mask.shape != rgb.shape[:2]:
            raise ValueError(f"masks of {out[0].mask.shape}, image of {rgb.shape[:2]}")
        return out


@dataclass
class ModalSiglipEmbedder(ZeroShotScoring):
    """`SiglipEmbedder` on a Modal GPU (`SegmentEmbed.embed_images` / `.embed_texts`)."""

    model: str = SIGLIP_MODEL
    dim: int = 768
    remote: wmc.Remote = wmc.modal_remote

    @property
    def name(self) -> str:
        return f"siglip:{self.model}"

    def _check(self, response: dict, n: int) -> np.ndarray:
        x = decode_array(response["embeddings"]).astype(np.float32)
        if x.shape != (n, self.dim):
            raise ValueError(f"embeddings of {x.shape}, expected {(n, self.dim)}")
        return l2_normalise(x)

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        request = {"images": [wmc.encode_png(c) for c in crops], "model": self.model}
        return self._check(self.remote("SegmentEmbed", "embed_images", request), len(crops))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        request = {"texts": list(texts), "model": self.model}
        return self._check(self.remote("SegmentEmbed", "embed_texts", request), len(texts))
