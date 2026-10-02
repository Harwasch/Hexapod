"""The teachers' side of the GPU models in `infra/modal/world_models.py`.

* `FixerFiller` -- a `teacher_fill.Filler`: each view's full render (the unseen side and
  its artifacts included, which is what Fixer is trained on) sent to NVIDIA Fixer, the
  cleaned image back. Only the masked pixels are lifted; the rest is the gate.
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

    def fill(self, rgb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        response = self.remote("Fixer", "fix", {"images": [encode_png(rgb)]})
        (fixed,) = response["images"]
        out = decode_png(fixed)
        return [_resize(out, rgb.shape[1], rgb.shape[0])]


@dataclass
class VideoClips:
    """Wan 2.2 (`model="Wan"`) or Cosmos-Predict2.5 (`model="Cosmos"`) as a
    `teacher_motion.ClipSource`. `fps` is the model's and is checked against each reply."""

    model: str = "Wan"
    prompt: str = PLANT_PROMPT
    frames: int | None = None
    remote: Remote = modal_remote
    fps: float = field(init=False)
    #: What each clip came back as, for the lesson's provenance.
    received: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.model not in ("Wan", "Cosmos"):
            raise ValueError(f"model {self.model!r}: Wan or Cosmos")
        self.fps = 24.0 if self.model == "Wan" else 16.0

    @property
    def name(self) -> str:
        return {"Wan": "wan2.2-ti2v-5b", "Cosmos": "cosmos-predict2.5-2b"}[self.model]

    def clips(
        self, stills: Sequence[np.ndarray], cameras: Sequence[object], seeds: Sequence[int]
    ) -> list[list[np.ndarray]]:
        """Clip `k` is of still `k % len(stills)`: every still once per seed."""
        out = []
        for seed in seeds:
            for c, still in enumerate(stills):
                u8 = still if still.dtype == np.uint8 else np.round(np.clip(still, 0, 1) * 255)
                u8 = u8.astype(np.uint8)
                request: dict = {
                    "image": encode_png(u8),
                    "prompt": self.prompt,
                    "seed": int(seed) * 1000 + c,
                }
                if self.frames is not None:
                    request["frames"] = self.frames
                response = self.remote(self.model, "clip", request)
                if abs(float(response["fps"]) - self.fps) > 1e-6:
                    raise ValueError(f"{self.model} sent {response['fps']} fps, not {self.fps}")
                frames = [_resize(f, u8.shape[1], u8.shape[0]) for f in decode_mp4(response["mp4"])]
                self.received.append(
                    {"model": response.get("model"), "seed": request["seed"], "frames": len(frames)}
                )
                out.append(frames)
        return out
