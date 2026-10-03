"""The teachers' side of the GPU models in `infra/modal/world_models.py`.

* `FixerFiller` -- a `teacher_fill.Filler`: each view's full render (the unseen side and
  its artifacts included, which is what Fixer is trained on) sent to NVIDIA Fixer, the
  cleaned image back. Only the masked pixels are lifted; the rest is the gate.
* `GenerativeFiller` -- a `teacher_fill.Filler`: each view and its mask sent to a generative
  inpainting model (`inpaint_models`: SDXL inpainting, Qwen-Image's inpainting ControlNet,
  FLUX.1 Fill), with a prompt describing what is around the hole (`context`, set by the
  caller from instances.json) and the same seed for every view; only the masked pixels of
  what comes back are kept.
* `VideoClips` -- a `teacher_motion.ClipSource`: each still sent to Wan 2.2 or Cosmos with
  a prompt and a seed, the clip back as frames at the still's size, `fps` the model's.

Both talk to Modal through one callable, `remote(cls, method, request) -> response`, so
tests (and a dry run) can stand in for the GPU: `modal_remote` is the real one.

    teacher_fill.py fill TILESET OUT --filler world_model_client:FixerFiller
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

APP_NAME = "hexapod-world-models"

Remote = Callable[[str, str, dict], dict]

#: What a living-survey clip is asked for. Gentle, so the plant moves as it does on an
#: ordinary day, which is the motion the sidecar's resonances describe.
PLANT_PROMPT = (
    "A gentle breeze moves the leaves and branches of the trees and plants; they sway "
    "softly and settle. Everything else is still."
)


#: GPU classes by name that `modal_remote` calls in place of the deployed app's: set by a
#: runner already inside a Modal app that defines its own (`infra/modal/fill.py`).
LOCAL_CLASSES: dict[str, Callable[[], object]] = {}


def modal_remote(cls: str, method: str, request: dict) -> dict:
    """Calls `cls.method(request)` on the deployed app (needs MODAL_TOKEN_ID/SECRET), or on
    the class registered in `LOCAL_CLASSES` under that name."""
    if cls in LOCAL_CLASSES:
        instance = LOCAL_CLASSES[cls]()
    else:
        import modal

        app = os.environ.get("HEXAPOD_WORLD_MODELS_APP", APP_NAME)
        instance = modal.Cls.from_name(app, cls)()
    return getattr(instance, method).remote(request)


def encode_png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(blob: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def decode_mp4(blob: bytes) -> list[np.ndarray]:
    """An mp4's frames, RGB uint8."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        capture = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        capture.release()
    finally:
        Path(path).unlink(missing_ok=True)
    if not frames:
        raise ValueError("the clip had no frames")
    return frames


def encode_mp4(frames: Sequence[np.ndarray], fps: float) -> bytes:
    """For tests and dry runs: frames as an mp4 (mjpeg at top quality, which OpenCV writes
    everywhere)."""
    import cv2

    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        path = f.name
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    import cv2

    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


@dataclass
class FixerFiller:
    """NVIDIA Fixer as a `teacher_fill.Filler`."""

    remote: Remote = modal_remote
    name: str = "nvidia-fixer"
    reads_full_render: bool = True

    #: Spread the CPU renderer's point samples over their gaps (normalized convolution, this
    #: sigma in pixels) before Fixer sees the frame: 0 sends the render as it is.
    presmooth_px: float = 0.0
    #: The diffusion step Fixer denoises from: how far it may move from its input. None is
    #: the server's (its README's 250); the camp from outside needed about 50.
    timestep: int | None = None

    def __post_init__(self) -> None:
        if self.name == "nvidia-fixer":
            if self.timestep is not None:
                self.name += f"-t{self.timestep}"
            if self.presmooth_px > 0:
                self.name += f"-presmooth{self.presmooth_px:g}"

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        shown = presmooth(rgb, self.presmooth_px) if self.presmooth_px > 0 else rgb
        request: dict = {"images": [encode_png(shown)]}
        if self.timestep is not None:
            request["timestep"] = int(self.timestep)
        response = self.remote("Fixer", "fix", request)
        (fixed,) = response["images"]
        out = decode_png(fixed)
        return [_resize(out, rgb.shape[1], rgb.shape[0])]


#: What a hole is filled with when the caller says nothing about its surroundings.
GROUND_PROMPT = "the ground seen from above, natural photograph, daylight, sharp detail"
#: Never paint these into a hole: a removed object's labels are added by the caller.
INPAINT_NEGATIVE = (
    "object, bowl, plate, pot, container, ball, toy, fruit, vegetable, hole, crater, shadow, "
    "text, watermark, frame, border, blur, flat colour, smudge, cartoon"
)

#: The Modal class that holds each inpainting model (`infra/modal/world_models.py`).
INPAINT_CLASSES = {
    "sdxl": "InpaintSDXL",
    "qwen": "InpaintQwen",
    "flux": "InpaintFlux",
    "lama": "InpaintSDXL",  # LaMa is held beside SDXL (`inpaint_models.lama_fill`)
}


@dataclass
class GenerativeFiller:
    """A generative inpainting model as a `teacher_fill.Filler` (`inpaint_models`).

    Shown the view as the scan renders it (the hole pre-filled rough, which the model does not
    read: the mask is repainted from scratch) and the mask grown by `grow_px`, so the model
    blends across the hole's edge; what it returns is kept only inside the mask, so the gate
    sees the measured pixels untouched. `context` (`teacher_fill.describe_surroundings`, set
    by the caller per hole or scan) gives the prompt and what not to paint.

    Multi-view consistency: every view is drawn from the same `seed`; with `chain` set,
    `teacher_fill.fill_hole` fills the views in turn and shows each one what the views before
    it filled, lifted and re-rendered, so only what none of them saw is painted afresh (and
    distill reconciles the rest)."""

    model: str = "sdxl"
    remote: Remote = modal_remote
    name: str = ""
    reads_full_render: bool = False
    prompt: str = GROUND_PROMPT
    negative: str = INPAINT_NEGATIVE
    seed: int = 7
    steps: int | None = None
    guidance: float | None = None
    grow_px: int = 6
    #: 1: fill the views in turn, each shown the earlier views' fill (`teacher_fill`).
    chain: int = 0
    #: `lama`: LaMa fills the hole first (`inpaint_models`), the model refines it at
    #: `strength`; `model="lama"` is LaMa alone.
    prefill: str = ""
    strength: float | None = None
    #: The model sees a crop around the hole this many times its size (0: the whole frame):
    #: the hole's surroundings at the model's resolution, not the frame's far edges.
    context_scale: float = 2.5
    #: Crops smaller than this (pixels a side) are grown to it.
    min_crop_px: int = 256
    #: Repaint the void too (`teacher_fill.Filler`: beyond the scan's edge, smeared over by
    #: the rough pre-fill): the model then reads only measured pixels as the hole's context;
    #: what it paints there is discarded. On the pumpkin the smear's flat polygons were what
    #: LaMa and SDXL copied into the hole.
    reads_void: bool = True
    context: dict | None = None
    #: Per call: the model, its seconds, and how well it kept the unmasked pixels by itself.
    received: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model not in INPAINT_CLASSES:
            raise ValueError(f"model {self.model!r}: one of {', '.join(INPAINT_CLASSES)}")
        if self.model == "lama":
            self.prefill, self.strength = "lama", 0.0
        if not self.name:
            self.name = f"inpaint-{self.model}"
            if self.prefill and self.model != "lama":
                self.name += f"-{self.prefill}"
            if self.chain:
                self.name += "-chain"

    @property
    def chain_views(self) -> bool:
        return bool(self.chain)

    def crop(self, mask: np.ndarray) -> tuple[slice, slice]:
        """The window the model is shown: `context_scale` times the mask's bounding box,
        square where the frame allows, inside the frame."""
        h, w = mask.shape
        if self.context_scale <= 0:
            return slice(0, h), slice(0, w)
        ys, xs = np.nonzero(mask)
        side = max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)
        side = max(side * self.context_scale, self.min_crop_px)
        ch, cw = min(h, round(side)), min(w, round(side))
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        y0 = int(np.clip(round(cy - ch / 2), 0, h - ch))
        x0 = int(np.clip(round(cx - cw / 2), 0, w - cw))
        return slice(y0, y0 + ch), slice(x0, x0 + cw)

    def fill(
        self, rgb: np.ndarray, mask: np.ndarray, void: np.ndarray | None = None
    ) -> list[np.ndarray]:
        import cv2

        if not mask.any():
            return [rgb]
        context = self.context or {}
        prompt = str(context.get("prompt") or self.prompt)
        negative = ", ".join(n for n in (context.get("negative"), self.negative) if n)
        grown = mask
        if self.grow_px > 0:
            size = 2 * self.grow_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        window = self.crop(grown)
        shown, asked = rgb[window], grown[window]
        if void is not None and self.reads_void:
            asked = asked | void[window]
        request: dict = {
            "image": encode_png(shown),
            "mask": encode_png(np.repeat(asked[..., None].astype(np.uint8) * 255, 3, axis=2)),
            "prompt": prompt,
            "negative": negative,
            "seed": int(self.seed),
        }
        if self.steps is not None:
            request["steps"] = int(self.steps)
        if self.guidance is not None:
            request["guidance"] = float(self.guidance)
        if self.prefill:
            request["prefill"] = self.prefill
        if self.strength is not None:
            request["strength"] = float(self.strength)
        response = self.remote(INPAINT_CLASSES[self.model], "inpaint", request)
        drawn = _resize(decode_png(response["image"]), shown.shape[1], shown.shape[0])
        diff = (drawn.astype(np.float64) - shown.astype(np.float64))[~asked]
        mse = float(np.mean(diff**2)) if diff.size else 0.0
        self.received.append(
            {
                "model": response.get("model"),
                "seconds": response.get("seconds"),
                "prompt": prompt,
                "crop": [window[1].start, window[0].start, shown.shape[1], shown.shape[0]],
                # How far the model moved what it was not asked to paint (it is discarded).
                "rawOutsidePsnr": round(float(10 * np.log10(255.0**2 / max(mse, 1e-10))), 2),
            }
        )
        out = rgb.copy()
        out[window] = np.where(mask[window][..., None], drawn, shown)
        return [out.astype(np.uint8)]


def presmooth(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution of a point-sampled render: the covered pixels' colours
    averaged over the empty (black) pixels near them, so the gaps between samples fill;
    pixels a few sigmas from any sample stay black."""
    import cv2

    weight = (rgb.max(axis=2) > 0).astype(np.float32)
    w = cv2.GaussianBlur(weight, (0, 0), sigma)
    c = cv2.GaussianBlur(rgb.astype(np.float32) * weight[..., None], (0, 0), sigma)
    out = np.where(w[..., None] > 1e-4, c / np.maximum(w, 1e-4)[..., None], 0.0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


@dataclass
class VideoClips:
    """Wan 2.2 TI2V-5B (`model="Wan"`) or Cosmos-Predict2 Video2World (`model="Cosmos"`) as
    a `teacher_motion.ClipSource`. `fps` is the model's and is checked against each reply.

    `chain` > 1 makes each clip longer than the model's ~5 s: the next clip starts from the
    last frame of the one before and is appended without its first frame (the same picture),
    so the pose is continuous at the joins though the motion's phase is not."""

    model: str = "Wan"
    prompt: str = PLANT_PROMPT
    frames: int | None = None
    remote: Remote = modal_remote
    chain: int = 1
    steps: int | None = None
    fps: float = field(init=False)
    #: What each model call came back as, for the lesson's provenance (and GPU time).
    received: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.model not in ("Wan", "Cosmos"):
            raise ValueError(f"model {self.model!r}: Wan or Cosmos")
        if self.chain < 1:
            raise ValueError("chain: at least 1")
        self.fps = 24.0 if self.model == "Wan" else 16.0

    @property
    def name(self) -> str:
        base = {"Wan": "wan2.2-ti2v-5b", "Cosmos": "cosmos-predict2-2b-video2world"}[self.model]
        return base + (f"-chain{self.chain}" if self.chain > 1 else "")

    def clip(self, still: np.ndarray, seed: int) -> list[np.ndarray]:
        """One clip of `still` (uint8, or floats in 0..1), `chain` model calls long, at the
        still's size."""
        u8 = still if still.dtype == np.uint8 else np.round(np.clip(still, 0, 1) * 255)
        u8 = np.ascontiguousarray(u8, dtype=np.uint8)
        out: list[np.ndarray] = []
        start = u8
        for link in range(self.chain):
            request: dict = {
                "image": encode_png(start),
                "prompt": self.prompt,
                "seed": int(seed) + 7919 * link,
            }
            if self.frames is not None:
                request["frames"] = self.frames
            if self.steps is not None:
                request["steps"] = self.steps
            response = self.remote(self.model, "clip", request)
            if abs(float(response["fps"]) - self.fps) > 1e-6:
                raise ValueError(f"{self.model} sent {response['fps']} fps, not {self.fps}")
            frames = [_resize(f, u8.shape[1], u8.shape[0]) for f in decode_mp4(response["mp4"])]
            self.received.append(
                {
                    "model": response.get("model"),
                    "seed": request["seed"],
                    "link": link,
                    "frames": len(frames),
                    "seconds": response.get("seconds"),
                    "loadSeconds": response.get("loadSeconds"),
                }
            )
            out.extend(frames if link == 0 else frames[1:])
            start = frames[-1]
        return out

    def clips(
        self, stills: Sequence[np.ndarray], cameras: Sequence[object], seeds: Sequence[int]
    ) -> list[list[np.ndarray]]:
        """Clip `k` is of still `k % len(stills)`: every still once per seed."""
        return [
            self.clip(still, int(seed) * 1000 + c)
            for seed in seeds
            for c, still in enumerate(stills)
        ]
