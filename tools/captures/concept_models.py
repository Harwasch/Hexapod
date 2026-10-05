"""The models concept-first segmentation runs on (`concept_scene.py`): a vision-language model
that lists a scene's things and stuff, and a segmenter that finds each of them in a view.

* `QwenVocabulary` -- a `concept_scene.Vocabulary`: Qwen3-VL Instruct (`Qwen/Qwen3-VL-4B-
  Instruct`, Apache-2.0, code and weights) shown the overview renders, answering in JSON
  (`VOCABULARY_PROMPT`, `parse_vocabulary`).
* `Sam3Concepts` -- the segmenter the method is built for: SAM 3 (`facebook/sam3`, Meta's
  custom "SAM License"; the weights are gated on Hugging Face). Things by its video model
  along each camera path, one track per object; stuff by its semantic head. It needs
  transformers 5 (with a torch newer than `infra/modal/segment.py`'s gsplat image has) and
  the gated weights; **written against the documented API and not yet run** -- the account
  the Modal `huggingface` secret belongs to has no access to `facebook/sam3` (2026-10-05).
* `GroundedSam2Concepts` -- a **stand-in** for SAM 3 while its weights are gated: Grounding
  DINO (`IDEA-Research/grounding-dino-base`, Apache-2.0) boxes each thing named, SAM 2.1
  (`facebook/sam2.1-hiera-tiny`, Apache-2.0) cuts a mask from each box, frame by frame (no
  tracker: the lift's own voting associates views); stuff is classified per class-free mask
  over the ground by SigLIP 2 (`segment_models.SiglipEmbedder`, Apache-2.0) against the stuff
  names ("mask pooling", research notes §1.2.5). Runs on transformers 4.57 and torch 2.4, the
  segmentation image. A run made with it is marked `standIn` in `instances.json`.

Torch and transformers are imported only when a model is first used (as `segment_models`),
so the pure helpers here are tested without them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import scene_categories
from concept_scene import (
    MAX_STUFF,
    MAX_THINGS,
    Concept,
    ConceptMask,
    StuffMap,
    clean_concepts,
)
from segment_scene import Mask

QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
SAM3_MODEL = "facebook/sam3"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-tiny"


def _device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _release(*models: Any) -> None:
    """Drop models and give their GPU memory back."""
    import gc

    import torch

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------- the vocabulary


def category_list() -> list[tuple[str, str]]:
    """(id, name) of every broad category the viewer knows."""
    return [(c.id, c.name) for c in scene_categories.CATEGORIES]


def vocabulary_prompt(views: int, max_things: int = MAX_THINGS, max_stuff: int = MAX_STUFF) -> str:
    """What the VLM is asked, with the categories it must choose from."""
    categories = "; ".join(f"{i} ({n})" for i, n in category_list())
    return (
        f"These are {views} views of one outdoor 3D scan, rendered from around it. List what "
        "is in the scene, so that a segmentation model can find each object and each kind of "
        "ground surface by its name.\n"
        f"- things: up to {max_things} kinds of countable objects you can see (for example "
        "'pumpkin', 'cable spool', 'wooden plank', 'tree', 'car'). One entry per kind, even "
        "if there are several of it; most visible first.\n"
        f"- stuff: up to {max_stuff} kinds of ground cover that the objects stand on (for "
        "example 'grass', 'gravel', 'dirt', 'hay', 'asphalt', 'trail').\n"
        "Use short, common English nouns of one to three words, singular, lower case. Do "
        "not list the sky, shadows, the background in general, or parts of objects.\n"
        f"Give each entry the category it belongs to, one of these ids: {categories}.\n"
        'Answer with JSON only, in this form: {"things": [{"name": "...", "category": '
        '"..."}], "stuff": [{"name": "...", "category": "..."}]}'
    )


def parse_vocabulary(text: str) -> list[Concept]:
    """The VLM's answer as concepts (`concept_scene.clean_concepts`): the first JSON object
    in it (code fences and prose around it ignored); entries may be strings or
    `{"name", "category"}`. Raises ValueError when there is no such object."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in the answer: {text[:200]!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned[start:])  # an object: it starts at {
    except ValueError as error:
        raise ValueError(f"no JSON object in the answer: {text[:200]!r}") from error
    concepts: list[Concept] = []
    for key, kind in (("things", "thing"), ("stuff", "stuff")):
        for entry in data.get(key) or []:
            if isinstance(entry, str):
                concepts.append(Concept(entry, kind))
            elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
                concepts.append(Concept(entry["name"], kind, str(entry.get("category", ""))))
    return clean_concepts(concepts)


@dataclass
class QwenVocabulary:
    """Qwen3-VL Instruct reads the overview renders and lists things and stuff
    (`vocabulary_prompt`). Greedy decoding, so the same images give the same list. `answer`
    keeps the model's own words for the record."""

    model: str = QWEN_MODEL
    max_new_tokens: int = 600
    device: str | None = None
    answer: str = field(default="", init=False)
    _loaded: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"qwen3-vl:{self.model}"

    def _load(self) -> tuple[Any, Any, str]:
        if self._loaded is None:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            device = _device(self.device)
            processor = AutoProcessor.from_pretrained(self.model)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model, torch_dtype=torch.bfloat16
            )
            self._loaded = (processor, model.to(device).eval(), device)
        return self._loaded

    def _ask(self, images: list[np.ndarray], prompt: str) -> str:
        import torch
        from PIL import Image

        processor, model, device = self._load()
        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.fromarray(np.ascontiguousarray(im, np.uint8))}
            for im in images
        ]
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1] :]
        return str(processor.batch_decode(new, skip_special_tokens=True)[0])

    def concepts(self, images: list[np.ndarray]) -> list[Concept]:
        prompt = vocabulary_prompt(len(images))
        self.answer = self._ask(images, prompt)
        try:
            found = parse_vocabulary(self.answer)
        except ValueError:
            found = []
        if not any(c.kind == "thing" for c in found):
            # Once more, more strictly; a second failure ends the run early and cheaply.
            self.answer += "\n---\n" + self._ask(images, prompt + "\nOnly the JSON object.")
            found = parse_vocabulary(self.answer.split("\n---\n")[-1])
        if not found:
            raise ValueError(f"the VLM listed nothing: {self.answer[:300]!r}")
        return found

    def close(self) -> None:
        if self._loaded is not None:
            loaded, self._loaded = self._loaded, None
            _release(*loaded)


# ------------------------------------------------------------------------ helpers (pure)


def match_phrase(phrase: str, concepts: Sequence[Concept]) -> int:
    """The concept a detector's phrase names (Grounding DINO returns the words of the
    prompt it matched, sometimes only some of them): the exact query; else the one whose
    words hold all of the phrase's, the fewest extra words first; else the most words
    shared. -1 when nothing is shared."""
    words = phrase.lower().replace(".", " ").split()
    if not words:
        return -1
    queries = [c.query.lower().split() for c in concepts]
    for k, q in enumerate(queries):
        if q == words:
            return k
    holding = [(len(q) - len(words), k) for k, q in enumerate(queries) if set(words) <= set(q)]
    if holding:
        return min(holding)[1]
    shared = [(len(set(words) & set(q)), -k) for k, q in enumerate(queries)]
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
) -> np.ndarray:
    """Indices of the detections kept: not over `max_share` of the frame (a box that is the
    whole view is the ground, not a thing), per concept greedy NMS, best score first."""
    h, w = shape
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ok = (area <= max_share * h * w) & (area > 0) & (np.asarray(concepts) >= 0)
    order = [int(k) for k in np.argsort(-np.asarray(scores), kind="stable") if ok[k]]
    kept: list[int] = []
    for k in order:
        same = [j for j in kept if concepts[j] == concepts[k]]
        if same and box_iou(boxes[k], boxes[same]).max() > nms_iou:
            continue
        kept.append(k)
    return np.asarray(kept, np.int64)


def paint_stuff(
    masks: Sequence[np.ndarray],
    probabilities: np.ndarray,
    region: np.ndarray,
) -> StuffMap:
    """A cover map from classified masks: each mask's class (its most probable) and score
    painted over the region's pixels, larger masks first, so a finer mask wins where
    masks nest."""
    h, w = region.shape
    label = np.full((h, w), -1, np.int16)
    score = np.zeros((h, w), np.float32)
    order = np.argsort([-int(np.count_nonzero(m)) for m in masks], kind="stable")
    for k in order:
        where = np.asarray(masks[k], bool) & region
        label[where] = int(np.argmax(probabilities[k]))
        score[where] = float(np.max(probabilities[k]))
    return StuffMap(label, score)


#: Stuff prompts for SigLIP: phrasings averaged per class (`segment_models.text_bank`).
STUFF_TEMPLATES = (
    "a photo of {}.",
    "a photo of the ground covered with {}.",
    "a close-up photo of {} on the ground.",
)


# --------------------------------------------------------------------- the stand-in: G-SAM 2


@dataclass
class GroundedSam2Concepts:
    """**Stand-in for SAM 3** (module docstring): Grounding DINO boxes, SAM 2.1 masks, SigLIP
    2 mask pooling for stuff. Frame by frame; `track` is -1 (no tracker)."""

    detector: str = GDINO_MODEL
    sam: str = SAM2_MODEL
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    #: A box over this share of the frame is not a thing.
    max_box_share: float = 0.85
    nms_iou: float = 0.6
    #: Masks under this share of the frame are dropped.
    min_area: float = 0.0005
    #: Class-free masks classified as stuff: at least this share on ground pixels, and this
    #: much of the frame.
    stuff_ground_share: float = 0.5
    stuff_min_area: float = 0.002
    device: str | None = None
    _loaded: Any = field(default=None, init=False, repr=False)
    _siglip: Any = field(default=None, init=False, repr=False)
    _banks: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"stand-in: grounding-dino ({self.detector}) + sam2.1 ({self.sam}) + siglip2"

    def _load(self) -> tuple[Any, ...]:
        if self._loaded is None:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )

            device = _device(self.device)
            det_processor = AutoProcessor.from_pretrained(self.detector)
            detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector)
            sam_processor = Sam2Processor.from_pretrained(self.sam)
            sam = Sam2Model.from_pretrained(self.sam)
            self._loaded = (
                det_processor,
                detector.to(device).eval(),
                sam_processor,
                sam.to(device).eval(),
                device,
            )
        return self._loaded

    def detect(
        self, rgb: np.ndarray, things: Sequence[Concept]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (n, 4) in pixels, scores and concept indices of every thing found."""
        import torch
        from PIL import Image

        det_processor, detector, _, _, device = self._load()
        h, w = rgb.shape[:2]
        text = " ".join(f"{c.query.lower()}." for c in things)
        inputs = det_processor(images=Image.fromarray(rgb), text=text, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = detector(**inputs)
        result = det_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        phrases = result.get("text_labels", result.get("labels", []))
        boxes = result["boxes"].float().cpu().numpy().reshape(-1, 4)
        scores = result["scores"].float().cpu().numpy().reshape(-1)
        concept = np.array([match_phrase(str(p), things) for p in phrases], np.int64)
        kept = keep_boxes(
            boxes, scores, concept, (h, w), max_share=self.max_box_share, nms_iou=self.nms_iou
        )
        return boxes[kept], scores[kept], concept[kept]

    def segment_boxes(self, rgb: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SAM 2.1's mask (h, w) bool and predicted IoU for each box, as
        `segment_models.Sam2Masks` prompts it (embeddings once, the stretched 1024 square)."""
        import torch

        _, _, sam_processor, sam, device = self._load()
        h, w = rgb.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), bool), np.zeros(0)
        inputs = sam_processor(images=np.ascontiguousarray(rgb, np.uint8), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        side = pixel_values.shape[-1]
        scale = np.array([side / w, side / h, side / w, side / h], np.float32)
        prompt = torch.tensor(np.asarray(boxes, np.float32) * scale, device=device)
        with torch.inference_mode():
            embeddings = sam.get_image_embeddings(pixel_values)
            out = sam(
                image_embeddings=embeddings,
                input_boxes=prompt[None],
                multimask_output=False,
            )
            low = out.pred_masks[0][:, :1].float()  # (boxes, 1, 256, 256)
            full = torch.nn.functional.interpolate(low, size=(h, w), mode="bilinear")[:, 0] > 0
        iou = out.iou_scores[0][:, 0].float().cpu().numpy()
        return full.cpu().numpy(), iou

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        out: list[list[ConceptMask]] = []
        for rgb in frames:
            boxes, scores, concept = self.detect(rgb, things)
            masks, iou = self.segment_boxes(rgb, boxes)
            h, w = rgb.shape[:2]
            found = [
                ConceptMask(masks[k], int(concept[k]), float(scores[k] * np.clip(iou[k], 0, 1)))
                for k in range(len(boxes))
                if masks[k].sum() >= self.min_area * h * w
            ]
            out.append(found)
        return out

    def _bank(self, stuff: Sequence[Concept]) -> np.ndarray:
        import segment_models as sm

        if self._siglip is None:
            self._siglip = sm.SiglipEmbedder(device=self.device)
        key = tuple(c.query for c in stuff)
        if key not in self._banks:
            phrases = [[t.format(c.query) for t in STUFF_TEMPLATES] for c in stuff]
            self._banks[key] = sm.text_bank(self._siglip, phrases)
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
        if not stuff:
            return empty
        chosen: list[np.ndarray] = []
        crops: list[np.ndarray] = []
        for m in masks:
            mask = np.asarray(m.mask, bool)
            area = int(mask.sum())
            if area < self.stuff_min_area * h * w:
                continue
            if (mask & region).sum() < self.stuff_ground_share * area:
                continue
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = rgb[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 128  # alone on grey (segment_scene's "alone")
            chosen.append(mask)
            crops.append(crop)
        if not crops:
            return empty
        bank = self._bank(stuff)
        assert self._siglip is not None
        emb = self._siglip.embed_images(crops)
        logits = 100.0 * emb @ bank.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return paint_stuff(chosen, p, region)


# --------------------------------------------------------------------- SAM 3 (gated)


@dataclass
class Sam3Concepts:
    """SAM 3 (module docstring): **not yet run** -- the weights are gated and the account has
    no access. Things: one video session per camera path and thing (`Sam3VideoModel`),
    each object a track along it; stuff: the semantic head per stuff name
    (`Sam3Model`), the most probable class over 0.5 per pixel."""

    model: str = SAM3_MODEL
    threshold: float = 0.5
    min_area: float = 0.0005
    device: str | None = None
    _video: Any = field(default=None, init=False, repr=False)
    _image: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"sam3:{self.model}"

    def _load_video(self) -> tuple[Any, Any, str]:
        if self._video is None:
            import torch
            from transformers import Sam3VideoModel, Sam3VideoProcessor  # transformers >= 5

            device = _device(self.device)
            model = Sam3VideoModel.from_pretrained(self.model).to(device, dtype=torch.bfloat16)
            self._video = (Sam3VideoProcessor.from_pretrained(self.model), model.eval(), device)
        return self._video

    def _load_image(self) -> tuple[Any, Any, str]:
        if self._image is None:
            from transformers import Sam3Model, Sam3Processor  # transformers >= 5

            device = _device(self.device)
            model = Sam3Model.from_pretrained(self.model).to(device).eval()
            self._image = (Sam3Processor.from_pretrained(self.model), model, device)
        return self._image

    def things(
        self, frames: list[np.ndarray], things: Sequence[Concept]
    ) -> list[list[ConceptMask]]:
        import torch

        processor, model, device = self._load_video()
        out: list[list[ConceptMask]] = [[] for _ in frames]
        h, w = frames[0].shape[:2]
        for c, concept in enumerate(things):
            session = processor.init_video_session(
                video=[np.ascontiguousarray(f, np.uint8) for f in frames],
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            session = processor.add_text_prompt(inference_session=session, text=concept.query)
            with torch.inference_mode():
                for step in model.propagate_in_video_iterator(
                    inference_session=session, max_frame_num_to_track=len(frames)
                ):
                    done = processor.postprocess_outputs(session, step)
                    masks = np.asarray(done["masks"].cpu().numpy(), bool).reshape(-1, h, w)
                    scores = done["scores"].float().cpu().numpy().reshape(-1)
                    ids = done["object_ids"].cpu().numpy().reshape(-1)
                    for m, s, i in zip(masks, scores, ids, strict=True):
                        if s >= self.threshold and m.sum() >= self.min_area * h * w:
                            # Tracks are per concept: a concept's ids in the high bits.
                            track = c * 100_000 + int(i)
                            out[int(step.frame_idx)].append(ConceptMask(m, c, float(s), track))
        return out

    def stuff(
        self,
        rgb: np.ndarray,
        stuff: Sequence[Concept],
        masks: Sequence[Mask],
        region: np.ndarray,
    ) -> StuffMap:
        import torch

        processor, model, device = self._load_image()
        h, w = rgb.shape[:2]
        probs = np.zeros((len(stuff), h, w), np.float32)
        for s, concept in enumerate(stuff):
            inputs = processor(images=rgb, text=concept.query, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            logits = outputs.semantic_seg.float()  # (1, 1, h', w')
            up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear")
            probs[s] = torch.sigmoid(up)[0, 0].cpu().numpy()
        best = probs.argmax(axis=0)
        score = probs.max(axis=0)
        label = np.where((score >= self.threshold) & region, best, -1).astype(np.int16)
        return StuffMap(label, np.where(label >= 0, score, 0).astype(np.float32))
