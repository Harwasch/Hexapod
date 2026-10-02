"""World models on Modal GPUs: the three image/video models the teachers call. **Written
against each model's own documented entry point; only Fixer has run on a GPU** (its
image and class as in `infra/modal/fill.py`, which ran it on 2026-10-02).

    Fixer    nvidia/Fixer (Apache code, NVIDIA Open Model License weights). One image in,
             one image out: a render with 3DGS artifacts -> a clean one. Teacher B's filler.
    Wan      Wan-AI/Wan2.2-TI2V-5B-Diffusers (Apache). Still + prompt -> 121 frames at
             24 fps, 720p class. Teacher A's clip source.
    Distill  gsplat: the lifted fill refined against the filled views, the measured scan
             frozen (tools/captures/distill_fill.py).
    Cosmos   nvidia/Cosmos-Predict2-2B-Video2World (NVIDIA Open Model License, gated), through
             diffusers' Cosmos2VideoToWorldPipeline: still + prompt -> 93 frames at 16 fps,
             1280x704. Teacher A's second clip source. Its guardrail stays on -- the
             licence requires it.
    SegmentMasks / SegmentEmbed
             facebook/sam2.1-hiera-tiny (Apache-2.0) and google/siglip2-base-patch16-224
             (Apache-2.0), run by `tools/captures/segment_models.py` itself (copied into
             the image): class-free masks at three granularities, image/text embeddings.

The request and response of every method are plain dicts of bytes, strings and numbers,
so the client (`tools/captures/world_model_client.py`) needs `modal` and nothing else from
here; `tools/captures/tests/test_world_model_client.py` pins the method names and keys on
both sides by reading this file with `ast`.

What was checked, 2026-10-01, and what was not:

* Fixer: the repository at `FIXER_COMMIT` was read; its inference script's functions are
  what `Fixer.fix` calls, and it expects the base model at `/work/models/base/`, which is
  where the weights volume is mounted. The Hub repo `nvidia/Fixer` holds `base/` and
  `pretrained/` (5.5 GB, not gated). Its own base container is NGC's
  `cosmos-predict2-container:1.2` (needs an NGC key); the image here builds the same
  environment from cosmos-predict2's uv.lock on a public CUDA base instead, and on
  2026-10-02 it loaded every checkpoint key and cleaned Fixer's own examples.
* Wan: the Hub model card's diffusers recipe, with `WanImageToVideoPipeline` for the
  image-conditioned case (the TI2V repository's `expand_timesteps`); 1280x704 is the 720p
  size, the aspect following the input.
* Cosmos: diffusers' own pipeline for the Predict2 Video2World checkpoint, the guardrail
  (`cosmos_guardrail`) built by it. `access()` reports which gated repositories the
  `huggingface` secret's token can read.
* Wan and Cosmos run from CI: `.github/workflows/dream.yml` deploys this app and runs
  `infra/modal/dream.py` (docs/WORLD_MODEL_RUNBOOK.md, section 8).

Secrets: `huggingface` (HF_TOKEN) for Wan, Cosmos and segmentation; Fixer needs none.
Weights are cached in the volume
`hexapod-world-model-weights`, so only the first call downloads.

    modal deploy infra/modal/world_models.py
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-world-models"
app = modal.App(APP_NAME)

WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

FIXER_REPO = "https://github.com/nv-tlabs/Fixer.git"
FIXER_COMMIT = "b39dfcaf4eeec90dc943b057ff368c16252c6c6e"
FIXER_BASE = "nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2"
#: Fixer's resolutions are fixed (its `get_resolution_size`); 1024 -> 1024x576.
FIXER_RESOLUTION = 1024
FIXER_TIMESTEP = 250

WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_FPS = 24.0
WAN_FRAMES = 121
#: The 720p area of TI2V-5B; the clip takes the still's aspect at about this many pixels.
WAN_AREA = 1280 * 704

COSMOS_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_FPS = 16.0
COSMOS_FRAMES = 93
#: Its 720p checkpoint's size: the still is scaled to cover it and centre-cropped.
COSMOS_SIZE = (1280, 704)

#: What the video models download: Wan is open; Cosmos and its guardrail's checkpoint are
#: gated (the licence is accepted per account); the guardrail's prompt model is open.
ACCESS_REPOS = (
    WAN_MODEL,
    COSMOS_MODEL,
    "nvidia/Cosmos-1.0-Guardrail",
    "Qwen/Qwen3Guard-Gen-0.6B",
)

#: Both video models are told the camera does not move: Teacher A tracks pixels, and a
#: moving camera would read as the trunk swaying.
STATIC_CAMERA = (
    "Static locked-off tripod shot, the camera does not move, pan, zoom or shake. "
    "Natural daylight, real footage."
)
NEGATIVE = (
    "camera motion, panning, zooming, handheld shake, cuts, scene change, people appearing, "
    "text, watermark, blur, flicker, low quality"
)

HF_HOME = "/weights/hf"

# --- Fixer ---------------------------------------------------------------------------------

#: cosmos-predict2 at the commit that is its 1.0.9 (what Fixer's Dockerfile pip-installs).
FIXER_COSMOS_REPO = "https://github.com/nvidia-cosmos/cosmos-predict2.git"
FIXER_COSMOS_COMMIT = "661da4774b0ca41d082a0ecbeb47550bcf07e03f"

#: Not FIXER_BASE (NGC, needs a key): the environment that container holds, built from
#: cosmos-predict2's own uv.lock on its Dockerfile's public CUDA base, then Fixer's
#: Dockerfile lines. The same recipe as `infra/modal/fill.py`, where it was run.
fixer_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install uv==0.8.12",
        f"git clone {FIXER_COSMOS_REPO} /cosmos && git -C /cosmos checkout {FIXER_COSMOS_COMMIT}",
        "cd /cosmos && UV_PROJECT_ENVIRONMENT=$(python -c 'import sys; print(sys.prefix)') "
        "uv sync --frozen --inexact --no-install-project --extra cu126",
        'pip install --no-deps "cosmos-predict2==1.0.9"',
        "pip install lpips natsort",
        f"git clone {FIXER_REPO} /work/fixer && git -C /work/fixer checkout {FIXER_COMMIT}",
    )
    .env({"HF_HOME": HF_HOME})
)


@app.cls(
    image=fixer_image,
    gpu="L40S",
    volumes={"/work/models": WEIGHTS},
    timeout=1800,
    scaledown_window=300,
)
class Fixer:
    @modal.enter()
    def load(self) -> None:
        import sys

        import torch

        _ensure_snapshot("nvidia/Fixer", Path("/work/models"), "pretrained/pretrained_fixer.pkl")
        sys.path.insert(0, "/work/fixer/src")
        import inference_pretrained_model as fixer  # Fixer's own script, as a module

        self.fixer = fixer
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.width, self.height = fixer.get_resolution_size(FIXER_RESOLUTION)
        self.model = fixer.load_and_compile_model(
            model_path="/work/models/pretrained/pretrained_fixer.pkl",
            timestep=FIXER_TIMESTEP,
            vae_skip_connection=False,  # as the README runs it (no --vae_skip_connection)
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
            compile=False,
        )
        self.model.set_eval()

    @modal.method()
    def fix(self, request: dict) -> dict:
        """`{"images": [png bytes]}` -> `{"images": [png bytes], "model": ...}`, each output
        the size of its input."""
        import torch
        from PIL import Image

        out = []
        for blob in request["images"]:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            size = image.size
            x = self.fixer.preprocess_image(
                image.resize((self.width, self.height), Image.BILINEAR), self.device, self.dtype
            )
            with torch.no_grad():
                y = self.fixer.model_inference(
                    self.model, 1, self.height, self.width, self.dtype, self.device, x=x
                )
            fixed = self.fixer.postprocess_output(y, size)
            buffer = io.BytesIO()
            fixed.save(buffer, format="PNG")
            out.append(buffer.getvalue())
        return {"images": out, "model": f"nvidia/Fixer@{FIXER_COMMIT[:7]}"}


# --- Wan 2.2 and Cosmos-Predict2 (one diffusers image) -------------------------------------

#: Both video models through diffusers. Cosmos' pipeline builds its own guardrail
#: (`cosmos_guardrail`: a blocklist and Qwen3Guard on the prompt, face blur on the frames;
#: the licence requires it), which needs transformers 5; Wan runs on the same stack.
video_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.40.0",
        "transformers>=5,<6",
        "accelerate>=1.6",
        "ftfy",
        "sentencepiece",
        "protobuf",
        "imageio[ffmpeg]>=2.37",
        "pillow",
        "huggingface_hub",
        "cosmos_guardrail==0.3.2",
    )
    .env({"HF_HOME": HF_HOME})
)


@app.cls(
    image=video_image,
    gpu="H100",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=3600,
    scaledown_window=300,
    max_containers=4,
)
class Wan:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        started = time.time()
        # The TI2V repository's model_index names `WanPipeline` (text to video); its
        # `expand_timesteps` config is what the image-to-video pipeline reads for TI2V.
        vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL, vae=vae, torch_dtype=torch.bfloat16
        ).to("cuda")
        WEIGHTS.commit()
        self.load_seconds = time.time() - started

    @modal.method()
    def clip(self, request: dict) -> dict:
        """`{"image": png, "prompt", "seed", "frames"?, "steps"?}` -> `{"mp4", "fps",
        "model", "seconds", "loadSeconds", "size"}`."""
        import torch
        from PIL import Image

        started = time.time()
        image = Image.open(io.BytesIO(request["image"])).convert("RGB")
        width, height = video_size(image.size, WAN_AREA, 32)
        frames = self.pipe(
            image=image.resize((width, height), Image.LANCZOS),
            prompt=f"{request['prompt']} {STATIC_CAMERA}",
            negative_prompt=NEGATIVE,
            height=height,
            width=width,
            num_frames=int(request.get("frames", WAN_FRAMES)),
            guidance_scale=5.0,
            num_inference_steps=int(request.get("steps", 50)),
            generator=torch.Generator("cuda").manual_seed(int(request["seed"])),
            output_type="np",
        ).frames[0]
        return {
            "mp4": _mp4(_u8(frames), WAN_FPS),
            "fps": WAN_FPS,
            "model": WAN_MODEL,
            "seconds": round(time.time() - started, 1),
            "loadSeconds": round(self.load_seconds, 1),
            "size": [width, height],
        }


@app.cls(
    image=video_image,
    gpu="H100",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=3600,
    scaledown_window=300,
    max_containers=2,
)
class Cosmos:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import Cosmos2VideoToWorldPipeline

        started = time.time()
        # The guardrail (the pipeline's safety_checker) is built by the pipeline: it stays on.
        self.pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
            COSMOS_MODEL, torch_dtype=torch.bfloat16
        ).to("cuda")
        WEIGHTS.commit()
        self.load_seconds = time.time() - started

    @modal.method()
    def clip(self, request: dict) -> dict:
        """`{"image": png, "prompt", "seed", "frames"?, "steps"?}` -> `{"mp4", "fps",
        "model", "seconds", "loadSeconds", "size"}`. Raises if the guardrail blocks it."""
        import torch
        from PIL import Image

        started = time.time()
        image = Image.open(io.BytesIO(request["image"])).convert("RGB")
        width, height = COSMOS_SIZE
        frames = self.pipe(
            image=_cover(image, width, height),
            prompt=f"{request['prompt']} {STATIC_CAMERA}",
            negative_prompt=NEGATIVE,
            height=height,
            width=width,
            num_frames=int(request.get("frames", COSMOS_FRAMES)),
            num_inference_steps=int(request.get("steps", 35)),
            fps=int(COSMOS_FPS),
            generator=torch.Generator("cuda").manual_seed(int(request["seed"])),
            output_type="np",
        ).frames[0]
        frames = _u8(frames)
        if frames.size == 0 or not frames.any():
            raise RuntimeError("Cosmos returned an empty clip (blocked by the guardrail?)")
        return {
            "mp4": _mp4(frames, COSMOS_FPS),
            "fps": COSMOS_FPS,
            "model": COSMOS_MODEL,
            "seconds": round(time.time() - started, 1),
            "loadSeconds": round(self.load_seconds, 1),
            "size": [width, height],
        }


@app.function(image=video_image, secrets=[HF_SECRET], timeout=300)
def access() -> dict:
    """Whether the `huggingface` secret's token can read each repository the video models
    download (a gated one needs its licence accepted on that token's account)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    out: dict = {}
    try:
        out["account"] = api.whoami().get("name")
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out["account"] = f"error: {error}"
    for repo in ACCESS_REPOS:
        try:
            api.auth_check(repo)
            out[repo] = "ok"
        except Exception as error:  # noqa: BLE001 - reported, not raised
            out[repo] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return out


# --- Distill (gsplat) -----------------------------------------------------------------------

#: gsplat's prebuilt CUDA wheel, the one `app.py`'s trainer pins (cp310, torch 2.4 cu124).
GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)
DISTILL_SOURCE = (
    Path(__file__).resolve().parents[2] / "tools" / "captures" / "distill_fill.py"
    if modal.is_local()
    else Path("/root/distill_fill.py")
)

distill_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    # gsplat imports `packaging`, which nothing else here installs.
    .pip_install("numpy==1.26.4", "ninja", "jaxtyping", "rich", "packaging", GSPLAT_WHEEL)
    .add_local_file(DISTILL_SOURCE, "/root/distill_fill.py")
)


@app.cls(image=distill_image, gpu="L40S", timeout=3600, scaledown_window=120)
class Distill:
    @modal.method()
    def run(self, request: dict) -> dict:
        """`{"measured", "init", "views": npz bytes, "iterations"?}` -> `{"inferred": npz,
        "report"}`: `tools/captures/distill_fill.run`, with gsplat."""
        import sys

        sys.path.insert(0, "/root")
        import distill_fill

        return distill_fill.run(request)


# --- Segmentation (SAM 2.1 masks, SigLIP 2 embeddings) -------------------------------------

#: `tools/captures/segment_models.py` runs here as it runs on a CPU: one implementation.
CAPTURES = (
    Path(__file__).resolve().parents[2] / "tools" / "captures"
    if modal.is_local()
    else Path("/root")
)

segment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=4.56",
        "numpy",
        "pillow",
        "huggingface_hub>=0.30",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_file(CAPTURES / "segment_models.py", "/root/segment_models.py")
    .add_local_file(CAPTURES / "world_model_client.py", "/root/world_model_client.py")
)


def _segment_models():
    import sys

    sys.path.insert(0, "/root")
    import segment_models

    return segment_models


@app.cls(
    image=segment_image,
    gpu="L4",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=1800,
    scaledown_window=300,
)
class SegmentMasks:
    @modal.enter()
    def load(self) -> None:
        self.sm = _segment_models()
        self.models: dict = {}

    @modal.method()
    def masks(self, request: dict) -> dict:
        """`{"images": [png], "model"?, "points_per_side"?}` -> `{"masks": [npz], "model"}`:
        per image, `segment_models.encode_masks` of `Sam2Masks(...).masks(rgb)`."""
        import numpy as np
        from PIL import Image

        sm = self.sm
        key = (request.get("model", sm.SAM2_MODEL), int(request.get("points_per_side", 32)))
        if key not in self.models:
            self.models[key] = sm.Sam2Masks(model=key[0], points_per_side=key[1], device="cuda")
            self.models[key]._load()
            WEIGHTS.commit()
        source = self.models[key]
        out = []
        for blob in request["images"]:
            rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)
            out.append(sm.encode_masks(source.masks(rgb), rgb.shape[:2]))
        return {"masks": out, "model": key[0]}


@app.cls(
    image=segment_image,
    gpu="L4",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=1800,
    scaledown_window=300,
)
class SegmentEmbed:
    @modal.enter()
    def load(self) -> None:
        self.sm = _segment_models()
        self.models: dict = {}

    def _embedder(self, request: dict):
        name = request.get("model", self.sm.SIGLIP_MODEL)
        if name not in self.models:
            self.models[name] = self.sm.SiglipEmbedder(model=name, device="cuda")
            self.models[name]._load()
            WEIGHTS.commit()
        return self.models[name]

    @modal.method()
    def embed_images(self, request: dict) -> dict:
        """`{"images": [png], "model"?}` -> `{"embeddings": npz (n, dim) float32, "model"}`."""
        import numpy as np
        from PIL import Image

        crops = [
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
            for b in request["images"]
        ]
        embedder = self._embedder(request)
        x = embedder.embed_images(crops)
        return {"embeddings": self.sm.encode_array(x), "model": embedder.model}

    @modal.method()
    def embed_texts(self, request: dict) -> dict:
        """`{"texts": [str], "model"?}` -> `{"embeddings": npz (n, dim) float32, "model"}`."""
        embedder = self._embedder(request)
        x = embedder.embed_texts(list(request["texts"]))
        return {"embeddings": self.sm.encode_array(x), "model": embedder.model}


# --- shared --------------------------------------------------------------------------------


def video_size(size: tuple[int, int], area: int, multiple: int) -> tuple[int, int]:
    """(width, height) with the aspect of `size`, about `area` pixels, each a multiple."""
    w, h = size
    scale = (area / (w * h)) ** 0.5
    return (
        max(multiple, round(w * scale / multiple) * multiple),
        max(multiple, round(h * scale / multiple) * multiple),
    )


def _u8(frames: object) -> object:
    """A pipeline's `np` frames (floats in 0..1, (T, H, W, 3)) as uint8."""
    import numpy as np

    return np.clip(np.round(np.asarray(frames, np.float32) * 255), 0, 255).astype(np.uint8)


def _cover(image: object, width: int, height: int) -> object:
    """A PIL image scaled to cover `width` x `height`, centre-cropped to it."""
    from PIL import Image

    w, h = image.size  # type: ignore[attr-defined]
    scale = max(width / w, height / h)
    size = (max(width, round(w * scale)), max(height, round(h * scale)))
    image = image.resize(size, Image.LANCZOS)  # type: ignore[attr-defined]
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def _mp4(frames: list, fps: float) -> bytes:
    """Frames (PIL or arrays) as a near-lossless H.264 mp4 (CRF 12): Teacher A tracks pixels."""
    import imageio.v2 as imageio
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio.get_writer(
            f.name, fps=fps, codec="libx264", ffmpeg_params=["-crf", "12"], macro_block_size=1
        )
        for frame in frames:
            writer.append_data(np.asarray(frame))
        writer.close()
        return Path(f.name).read_bytes()


def _ensure_snapshot(repo: str, local: Path, marker: str) -> None:
    """The Hub repo in the weights volume, downloaded once."""
    if (local / marker).exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(local), token=os.environ.get("HF_TOKEN"))
    WEIGHTS.commit()
