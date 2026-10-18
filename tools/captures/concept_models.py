"""The models concept-first segmentation runs on (`concept_scene.py`, bake-off candidate C): a
vision-language model that lists a scene's things and ground cover, and a segmenter that
finds each of them in a view.

* `QwenVocabulary` -- a `concept_scene.Vocabulary`: Qwen3-VL 4B Instruct (Apache-2.0, code
  and weights; `segment_models.QwenNamer`'s model and call, shared with candidate A) shown
  the overview renders, answering in JSON (`vocabulary_prompt`, `parse_vocabulary`): the
  things, a couple of other words for each, and which of `data/ground_cover.json`'s
  classes the ground shows.
* `Sam3Concepts` -- the segmenter the method is built for: SAM 3 (`facebook/sam3`, Meta's
  custom "SAM License"; the weights are gated on Hugging Face). Things by its video model
  along each camera path, every thing's name a prompt, one track per object; stuff by its
  semantic head. It needs transformers 5 and torch >= 2.5, so it runs in
  `infra/modal/segment.py`'s SAM 3 image, which has no gsplat: a run there is seeded with
  an earlier run's gsplat views and class-free masks (`concept_scene.py --seeded`).
  `smoke_sam3` checks the calls on a few frames first.
* `GroundedSam2Concepts` -- a **stand-in** for SAM 3 while its weights are gated: Grounding
  DINO (`IDEA-Research/grounding-dino-base`, Apache-2.0) boxes each thing -- asked one
  thing at a time, since asked several at once it mislabels (on the spool's renders the
  spool came back as "wooden pallet") -- and SAM 2.1 large (`segment_models.Sam2BoxMasks`,
  Apache-2.0) cuts a mask from each box, frame by frame (no tracker: the lift's own voting
  associates views). Cover: SigLIP 2 (Apache-2.0) classifies each class-free mask over the
  ground, and tiles of the ground no mask covered, against the chosen cover classes'
  prompts and `data/ground_cover.json`'s contrast prompts ("mask pooling", research notes
  §1.2.5). Runs on transformers 4.57 and torch 2.4, the segmentation image. A run made with
  it is `concept-first-standin` (`concept_scene.VARIANTS`), never C.

transformers 5's SAM 3 API, as these classes call it (5.19.0's source and the model card):
`Sam3VideoProcessor.init_video_session(video=, inference_device=, processing_device=,
video_storage_device=, dtype=)`, `add_text_prompt(inference_session=, text=[...])`,
`Sam3VideoModel.propagate_in_video_iterator(inference_session=, max_frame_num_to_track=)`
yielding outputs with `frame_idx`, and `postprocess_outputs(session, output)` ->
`{object_ids, scores, boxes, masks, prompt_to_obj_ids}`; `Sam3Model(**Sam3Processor(images=,
text=))` -> `semantic_seg` (batch, 1, h, w) logits.

Torch and transformers are imported only when a model is first used (as `segment_models`),
so the pure helpers here are tested without them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import scene_categories
import segment_ground_first as sgf
import segment_models as sm
from concept_scene import (
    MAX_STUFF,
    MAX_THINGS,
    Concept,
    ConceptMask,
    StuffMap,
    clean_concepts,
)
from segment_scene import Mask

SAM3_MODEL = "facebook/sam3"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"


def _release() -> None:
    """Give the GPU memory of models nothing refers to any more back."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------- the vocabulary


def vocabulary_prompt(views: int, max_things: int = MAX_THINGS, max_stuff: int = MAX_STUFF) -> str:
    """What the VLM is asked: the categories and the cover classes it must choose from."""
    categories = "; ".join(f"{c.id} ({c.name})" for c in scene_categories.CATEGORIES)
    covers = ", ".join(c.id for c in sgf.cover_classes()[0])
    return (
        f"These are {views} views of one outdoor 3D scan, rendered from around it. List what "
        "is in the scene, so that a segmentation model can find each object and each kind of "
        "ground surface by its name.\n"
        f"- things: up to {max_things} kinds of countable objects you can see (for example "
        "'pumpkin', 'cable spool', 'wooden pallet', 'tree', 'car'), most visible first. One "
        "entry per kind, even if there are several of it. Give each a short common English "
        "name (one to three words, singular, lower case), up to two other words an object "
        "detector might know it by ('also'), and the category it belongs to, one of these "
        f"ids: {categories}.\n"
        f"- cover: up to {max_stuff} kinds of ground surface the objects stand on, as ids "
        f"from this list only: {covers}.\n"
        "Do not list the sky, shadows, the background in general, or parts of objects.\n"
        'Answer with JSON only, in this form: {"things": [{"name": "...", "also": ["..."], '
        '"category": "..."}], "cover": ["..."]}'
    )


def parse_vocabulary(text: str) -> list[Concept]:
    """The VLM's answer as concepts (`concept_scene.clean_concepts`): the first JSON object
    in it (code fences and prose around it ignored). Things may be strings or `{"name",
    "also"?, "category"?}`; `cover` (or `stuff`) entries are class ids or words that name
    one. Raises ValueError when there is no such object."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in the answer: {text[:200]!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned[start:])  # an object: it starts at {
    except ValueError as error:
        raise ValueError(f"no JSON object in the answer: {text[:200]!r}") from error
    concepts: list[Concept] = []
    for entry in data.get("things") or []:
        if isinstance(entry, str):
            concepts.append(Concept(entry, "thing"))
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            also = entry.get("also") or []
            also = [a for a in (also if isinstance(also, list) else [also]) if isinstance(a, str)]
            concepts.append(
                Concept(entry["name"], "thing", str(entry.get("category", "")), tuple(also))
            )
    for entry in [*(data.get("cover") or []), *(data.get("stuff") or [])]:
        word = entry.get("name") if isinstance(entry, dict) else entry
        if isinstance(word, str):
            concepts.append(Concept(word, "stuff"))
    return clean_concepts(concepts)


@dataclass
class QwenVocabulary:
    """Qwen3-VL reads the overview renders and lists things and cover (`vocabulary_prompt`),
    through `segment_models.QwenNamer` (the same model and call as candidate A's names).
    Greedy decoding, so the same images give the same list; `answer` keeps the model's own
    words for the record."""

    model: str = sm.QWEN_VL_MODEL
    max_new_tokens: int = 600
    device: str | None = None
    answer: str = field(default="", init=False)
    _namer: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"qwen3-vl:{self.model}"

    def _ask(self, images: list[np.ndarray], prompt: str) -> str:
        if self._namer is None:
            self._namer = sm.QwenNamer(
                model=self.model, device=self.device, max_new_tokens=self.max_new_tokens
            )
        return str(self._namer.ask(images, prompt))

    def concepts(self, images: list[np.ndarray]) -> list[Concept]:
        prompt = vocabulary_prompt(len(images))
        self.answer = self._ask(images, prompt)
        try:
            found = parse_vocabulary(self.answer)
        except ValueError:
            found = []
        if not any(c.kind == "thing" for c in found):
            # Once more, more strictly; a second failure ends the run early and cheaply.
            again = self._ask(images, prompt + "\nOnly the JSON object.")
            self.answer += "\n---\n" + again
            found = parse_vocabulary(again)
        if not found:
            raise ValueError(f"the VLM listed nothing: {self.answer[:300]!r}")
        return found

    def close(self) -> None:
        if self._namer is not None:
            self._namer = None  # the only reference to the model
            _release()


# ------------------------------------------------------------------------ helpers (pure)


def match_phrase(phrase: str, concepts: Sequence[Concept]) -> int:
    """The concept a detector's phrase names (Grounding DINO returns the words of the
    prompt it matched, sometimes only some of them), over each concept's `queries`: an exact
    query; else one whose words hold all of the phrase's, the fewest extra words first; else
    the most words shared. -1 when nothing is shared."""
    words = phrase.lower().replace(".", " ").split()
    if not words:
        return -1
    queries = [(k, q.lower().split()) for k, c in enumerate(concepts) for q in c.queries]
    for k, q in queries:
        if q == words:
            return k
    holding = [(len(q) - len(words), k) for k, q in queries if set(words) <= set(q)]
    if holding:
        return min(holding)[1]
    shared = [(len(set(words) & set(q)), -k) for k, q in queries]
    best, neg = max(shared)
    return -neg if best > 0 else -1


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) IoU of boxes (x0, y0, x1, y1)."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def keep_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    concepts: np.ndarray,
    shape: tuple[int, int],
    *,
    max_share: float,
    nms_iou: float,
    cross_iou: float = 1.0,
) -> np.ndarray:
    """Indices of the detections kept, best score first: none over `max_share` of the frame
    (a box that is the whole view is the ground, not a thing), none of no concept; greedy
    NMS at `nms_iou` within a concept, and at `cross_iou` across concepts (one object found
    under two names keeps the better)."""
    h, w = shape
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    concepts = np.asarray(concepts)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ok = (area <= max_share * h * w) & (area > 0) & (concepts >= 0)
    order = [int(k) for k in np.argsort(-np.asarray(scores), kind="stable") if ok[k]]
    kept: list[int] = []
    for k in order:
        if kept:
            iou = box_iou(boxes[k], boxes[kept])[0]
            same = concepts[kept] == concepts[k]
            if (iou[same] > nms_iou).any() or (iou[~same] > cross_iou).any():
                continue
        kept.append(k)
    return np.asarray(kept, np.int64)


def paint_stuff(
    masks: Sequence[np.ndarray],
    probabilities: np.ndarray,
    region: np.ndarray,
) -> StuffMap:
    """A cover map from classified masks: each mask's most probable class and its score
    painted over the region's pixels, larger masks first, so a finer mask wins where masks
    nest. A mask whose row is negative (not ground cover) is left out."""
    h, w = region.shape
    label = np.full((h, w), -1, np.int16)
    score = np.zeros((h, w), np.float32)
    order = np.argsort([-int(np.count_nonzero(m)) for m in masks], kind="stable")
    for k in order:
        row = np.asarray(probabilities[k], np.float64)
        if row.max() < 0:
            continue
        best = int(np.argmax(row))
        where = np.asarray(masks[k], bool) & region
        label[where] = best
        score[where] = float(row[best])
    return StuffMap(label, score)


def ground_tiles(
    region: np.ndarray, covered: np.ndarray, side: int, least: float
) -> list[np.ndarray]:
    """Masks of the `side`-pixel tiles of `region` that no classified mask covered, each with
    at least `least` of its pixels such ground."""
    h, w = region.shape
    left = region & ~covered
    out = []
    for y in range(0, h, side):
        for x in range(0, w, side):
            if left[y : y + side, x : x + side].sum() >= least * side * side:
                tile = np.zeros((h, w), bool)
                tile[y : y + side, x : x + side] = left[y : y + side, x : x + side]
                out.append(tile)
    return out


def classify(probabilities: np.ndarray, classes: int) -> np.ndarray:
    """Rows over the classes then contrast prompts -> rows over the classes, -1 everywhere
    for a row whose best is a contrast prompt (not ground cover)."""
    p = np.asarray(probabilities, np.float64)
    out = p[:, :classes].copy()
    out[np.argmax(p, axis=1) >= classes] = -1.0
    return out


# --------------------------------------------------------------------- the stand-in: G-SAM 2


@dataclass
class GroundedSam2Concepts:
    """**Stand-in for SAM 3** (module docstring): Grounding DINO boxes (one thing at a time),
    SAM 2.1 large masks, SigLIP 2 mask pooling for the cover. Frame by frame; `track` -1."""

    detector: str = GDINO_MODEL
    sam: str = sm.SAM2_LARGE_MODEL
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    #: A box over this share of the frame is not a thing.
    max_box_share: float = 0.85
    nms_iou: float = 0.6
    #: One object found under two names: the better is kept.
    cross_iou: float = 0.8
    #: Masks under this share of the frame are dropped.
    min_area: float = 0.0005
    #: Class-free masks classified as cover: at least this share on ground pixels, and this
    #: much of the frame; then tiles of what they left, at least `tile_share` ground.
    stuff_ground_share: float = 0.5
    stuff_min_area: float = 0.002
    tile_px: int = 64
    tile_share: float = 0.4
    device: str | None = None
    _loaded: Any = field(default=None, init=False, repr=False)
    _siglip: Any = field(default=None, init=False, repr=False)
    _banks: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"stand-in: grounding-dino ({self.detector}) + {self.sam} + siglip2"

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            device = sm._device(self.device)
            processor = AutoProcessor.from_pretrained(self.detector)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector)
            self._loaded = (processor, model.to(device).eval(), device)
        return self._loaded

    def detect(
        self, rgb: np.ndarray, things: Sequence[Concept]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (n, 4) in pixels, scores and concept indices of every thing found, each
        thing asked on its own (its `queries` as one prompt)."""
        import torch
        from PIL import Image

        processor, detector, device = self._load()
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb)
        boxes, scores, concept = [np.zeros((0, 4))], [np.zeros(0)], [np.zeros(0, np.int64)]
        for k, thing in enumerate(things):
            text = " ".join(f"{q.lower()}." for q in thing.queries)
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = detector(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
            found = result["boxes"].float().cpu().numpy().reshape(-1, 4)
            boxes.append(found)
            scores.append(result["scores"].float().cpu().numpy().reshape(-1))
            concept.append(np.full(len(found), k, np.int64))
        all_boxes, all_scores = np.concatenate(boxes), np.concatenate(scores)
        all_concepts = np.concatenate(concept)
        kept = keep_boxes(
            all_boxes, all_scores, all_concepts, (h, w),
            max_share=self.max_box_share, nms_iou=self.nms_iou, cross_iou=self.cross_iou,
        )  # fmt: skip
        return all_boxes[kept], all_scores[kept], all_concepts[kept]

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        boxer = sm.Sam2BoxMasks(model=self.sam, device=self.device)
        out: list[list[ConceptMask]] = []
        for rgb in frames:
            boxes, scores, concept = self.detect(rgb, things)
            h, w = rgb.shape[:2]
            found = []
            for k, (mask, iou) in enumerate(boxer.box_masks(rgb, boxes)):
                if mask.sum() >= self.min_area * h * w:
                    found.append(ConceptMask(mask, int(concept[k]), float(scores[k] * iou)))
            out.append(found)
        return out

    def _bank(self, stuff: Sequence[Concept]) -> np.ndarray:
        """Text rows: each class's prompts averaged, then each contrast prompt."""
        if self._siglip is None:
            self._siglip = sm.SiglipEmbedder(device=self.device)
        key = tuple(c.cover or c.name for c in stuff)
        if key not in self._banks:
            _, contrast = sgf.cover_classes()
            groups = [list(c.queries) for c in stuff] + [[p] for p in contrast]
            self._banks[key] = sm.text_bank(self._siglip, groups)
        return self._banks[key]

    def stuff(
        self,
        rgb: np.ndarray,
        stuff: Sequence[Concept],
        masks: Sequence[Mask],
        region: np.ndarray,
    ) -> StuffMap:
        h, w = rgb.shape[:2]
        empty = StuffMap(np.full((h, w), -1, np.int16), np.zeros((h, w), np.float32))
        if not stuff or not region.any():
            return empty
        bank = self._bank(stuff)
        chosen: list[np.ndarray] = []
        for m in masks:
            mask = np.asarray(m.mask, bool)
            area = int(mask.sum())
            if area >= self.stuff_min_area * h * w and (mask & region).sum() >= (
                self.stuff_ground_share * area
            ):
                chosen.append(mask)
        covered = np.zeros((h, w), bool)
        for mask in chosen:
            covered |= mask
        chosen += ground_tiles(region, covered, self.tile_px, self.tile_share)
        if not chosen:
            return empty
        crops = []
        for mask in chosen:
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = rgb[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 0  # alone on black, as candidate A's crops
            crops.append(crop)
        assert self._siglip is not None
        logits = 100.0 * np.asarray(self._siglip.embed_images(crops), np.float64) @ bank.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return paint_stuff(chosen, classify(p, len(stuff)), region)


# --------------------------------------------------------------------- SAM 3 (gated)


def thing_prompts(things: Sequence[Concept]) -> dict[str, int]:
    """Every word the vocabulary gives each thing (`Concept.queries`: its name and the VLM's
    other words, as the stand-in's detector is asked), lower-case, -> the thing's index; a
    word two things share is the first's."""
    out: dict[str, int] = {}
    for k, thing in enumerate(things):
        for word in thing.queries:
            out.setdefault(word.strip().lower(), k)
    return out


def prompt_concepts(
    prompt_to_ids: dict[str, list[int]], prompts: Mapping[str, int]
) -> dict[int, int]:
    """Object id -> concept index, from a SAM 3 video output's `prompt_to_obj_ids` (prompt text
    -> object ids) and each prompt's concept. An id under no known prompt is left out."""
    out: dict[int, int] = {}
    for text, ids in prompt_to_ids.items():
        if text in prompts:
            for i in ids:
                out[int(i)] = prompts[text]
    return out


def cover_from_logits(logits: np.ndarray, region: np.ndarray, threshold: float) -> StuffMap:
    """A cover map from SAM 3's semantic logits, one row per cover class (k, h, w): each
    pixel of `region` takes its most probable class when that is at least `threshold`."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
    best = probs.argmax(axis=0)
    score = probs.max(axis=0)
    label = np.where((score >= threshold) & region, best, -1).astype(np.int16)
    return StuffMap(label, np.where(label >= 0, score, 0).astype(np.float32))


@dataclass
class Sam3Concepts:
    """SAM 3 (`facebook/sam3`; transformers >= 5, torch >= 2.5). Things: one video session
    per camera path (`Sam3VideoModel`), every word the vocabulary gives each thing a text
    prompt in it (`thing_prompts`: on the spool's renders SAM 3 found "cable spool" in 1
    view of 64; the VLM's "wooden spool" and "table" are what the stand-in's detector was
    asked too), so the path's frames are encoded once and each object found is a track
    along it (its id; which prompt found it, `prompt_to_obj_ids`, is its concept). One object
    found under two words is two masks of one concept, which the lift's votes join.
    Stuff: the semantic head (`Sam3Model`)
    once per cover class, batched, each pixel of the ground its most probable class over
    `threshold`."""

    model: str = SAM3_MODEL
    threshold: float = 0.5
    min_area: float = 0.0005
    device: str | None = None
    #: bfloat16 on a GPU (as the model card runs the video model), float32 on a CPU.
    half: bool = True
    _video: Any = field(default=None, init=False, repr=False)
    _image: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"sam3:{self.model}"

    def _dtype(self, device: str) -> Any:
        import torch

        return torch.bfloat16 if self.half and device.startswith("cuda") else torch.float32

    def _load_video(self) -> tuple[Any, Any, str]:
        if self._video is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            device = sm._device(self.device)
            model = Sam3VideoModel.from_pretrained(self.model).to(device, dtype=self._dtype(device))
            self._video = (Sam3VideoProcessor.from_pretrained(self.model), model.eval(), device)
        return self._video

    def _load_image(self) -> tuple[Any, Any, str]:
        if self._image is None:
            from transformers import Sam3Model, Sam3Processor

            device = sm._device(self.device)
            model = Sam3Model.from_pretrained(self.model).to(device).eval()
            self._image = (Sam3Processor.from_pretrained(self.model), model, device)
        return self._image

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        import torch

        out: list[list[ConceptMask]] = [[] for _ in frames]
        if not frames or not things:
            return out
        processor, model, device = self._load_video()
        h, w = frames[0].shape[:2]
        prompts = thing_prompts(things)
        session = processor.init_video_session(
            video=np.stack([np.ascontiguousarray(f, np.uint8) for f in frames]),
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype(device),
        )
        session = processor.add_text_prompt(inference_session=session, text=list(prompts))
        with torch.inference_mode():
            for step in model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(frames)
            ):
                done = processor.postprocess_outputs(session, step)
                concept_of = prompt_concepts(done.get("prompt_to_obj_ids") or {}, prompts)
                masks = np.asarray(done["masks"].cpu().numpy(), bool).reshape(-1, h, w)
                scores = done["scores"].float().cpu().numpy().reshape(-1)
                ids = done["object_ids"].cpu().numpy().reshape(-1)
                for m, s, i in zip(masks, scores, ids, strict=True):
                    c = concept_of.get(int(i))
                    if c is None or s < self.threshold or m.sum() < self.min_area * h * w:
                        continue
                    out[int(step.frame_idx)].append(ConceptMask(m, c, float(s), int(i)))
        return out

    def stuff(
        self,
        rgb: np.ndarray,
        stuff: Sequence[Concept],
        masks: Sequence[Mask],
        region: np.ndarray,
    ) -> StuffMap:
        import torch

        h, w = rgb.shape[:2]
        if not stuff or not region.any():
            return StuffMap(np.full((h, w), -1, np.int16), np.zeros((h, w), np.float32))
        processor, model, device = self._load_image()
        texts = [c.name.lower() for c in stuff]
        image = np.ascontiguousarray(rgb, np.uint8)
        inputs = processor(images=[image] * len(texts), text=texts, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.semantic_seg.float()  # (k, 1, h', w')
        up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear")
        return cover_from_logits(up[:, 0].cpu().numpy(), region, self.threshold)


def smoke_sam3(
    frames: list[np.ndarray],
    concepts: Sequence[Concept] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """SAM 3 on a few frames, things and cover, as a run calls it: what it found (a check
    that the weights load and the calls work, before a GPU is spent on a whole scan), with
    `concepts` (a scan's vocabulary; default a spool's) and per thing the frames it was in."""
    source = Sam3Concepts(device=device)
    default = [
        Concept("cable spool", "thing", "other", ("cable spool", "wooden spool", "table")),
        Concept("rock", "thing"),
        Concept("Grass", "stuff", cover="grass"),
        Concept("Dirt", "stuff", cover="dirt"),
    ]
    things = [c for c in (concepts or default) if c.kind == "thing"]
    stuff = [c for c in (concepts or default) if c.kind == "stuff"] or default[2:]
    found = source.things(frames, things)
    region = np.ones(frames[0].shape[:2], bool)
    cover = source.stuff(frames[0], stuff, [], region)
    # What `segment_scene.describe` embeds with, in the same image (transformers 5).
    embedder = sm.SiglipEmbedder(device=device)
    crops = embedder.embed_images([f[: f.shape[0] // 2, : f.shape[1] // 2] for f in frames])
    words = embedder.embed_texts(["a cable spool", "grass"])
    return {
        "embeddings": [list(crops.shape), list(words.shape)],
        "frames": len(frames),
        "framesWith": {
            t.name: sum(1 for f in found if any(m.concept == k for m in f))
            for k, t in enumerate(things)
        },
        "things": [
            [(m.concept, round(m.score, 3), m.track, int(m.mask.sum())) for m in f] for f in found
        ],
        "coverShare": [round(float((cover.label == k).mean()), 3) for k in range(len(stuff))],
    }
