"""World models on Modal GPUs: the three image/video models the teachers call. **Written
against each model's own documented entry point; only Fixer has run on a GPU** (its
image and class as in `infra/modal/fill.py`, which ran it on 2026-10-02).

    Fixer    nvidia/Fixer (Apache code, NVIDIA Open Model License weights). One image in,
             one image out: a render with 3DGS artifacts -> a clean one. Teacher B's filler.
    Wan      Wan-AI/Wan2.2-TI2V-5B-Diffusers (Apache). Still + prompt -> 121 frames at
             24 fps, 720p class. Teacher A's clip source.
    Distill  gsplat: the lifted fill refined against the filled views, the measured scan
             frozen (tools/captures/distill_fill.py).
    Cosmos   nvidia/Cosmos-Predict2.5-2B, post-trained (NVIDIA Open Model License, gated).
             Image2World: still + prompt -> 77 frames at 16 fps. Teacher A's second clip
             source. Its guardrails stay on -- the licence requires them.
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
  image-conditioned case; 1280x704 is the 720p size, the aspect following the input.
* Cosmos: the repository at `COSMOS_COMMIT` was read -- `examples/inference.py` with a
  JSON spec, `--inference-type=image2world --model=2B/post-trained`, output saved at 16 fps.
  The Hugging Face token must have accepted the licences of Cosmos-Predict2.5-2B,
  Cosmos-Reason1-7B and Cosmos-Guardrail1 (it had not, on 2026-10-01).
* Wan, Cosmos and Distill have not been built by Modal or run. Their package pins are the
  first guess that `modal deploy` proves; the runbook (docs/WORLD_MODEL_RUNBOOK.md) says
  what to try first.

Secrets: `huggingface` (HF_TOKEN) for Wan, Cosmos and segmentation; Fixer needs none.
Weights are cached in the volume
`hexapod-world-model-weights`, so only the first call downloads.

    modal deploy infra/modal/world_models.py
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
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

COSMOS_REPO = "https://github.com/nvidia-cosmos/cosmos-predict2.5.git"
COSMOS_COMMIT = "a2c298b0a3df3778b973fe65e9e58877b292d8a7"
COSMOS_FPS = 16.0
COSMOS_FRAMES = 77

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


# --- Wan 2.2 -------------------------------------------------------------------------------

wan_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "diffusers==0.35.1",
        "transformers>=4.51,<5",
        "accelerate>=1.6",
        "ftfy",
        "sentencepiece",
        "imageio[ffmpeg]>=2.37",
        "pillow",
        "huggingface_hub>=0.30",
    )
    .env({"HF_HOME": HF_HOME})
)


@app.cls(
    image=wan_image,
    gpu="A100-80GB",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=3600,
    scaledown_window=300,
)
class Wan:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        vae = AutoencoderKLWan.from_pretrained(
            WAN_MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL, vae=vae, torch_dtype=torch.bfloat16
        ).to("cuda")
        WEIGHTS.commit()

    @modal.method()
    def clip(self, request: dict) -> dict:
        """`{"image": png, "prompt", "seed", "frames"?, "steps"?}` -> `{"mp4", "fps", "model"}`."""
        import torch
        from PIL import Image

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
        ).frames[0]
        return {"mp4": _mp4(frames, WAN_FPS), "fps": WAN_FPS, "model": WAN_MODEL}


# --- Cosmos-Predict2.5 ---------------------------------------------------------------------

cosmos_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "git-lfs", "ffmpeg", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.8.12/install.sh | sh",
        f"git clone {COSMOS_REPO} /cosmos && git -C /cosmos checkout {COSMOS_COMMIT}",
        # The repository's own environment: its lockfile, CUDA 12.8 extra.
        "cd /cosmos && /root/.local/bin/uv sync --locked --extra=cu128",
    )
    .env({"HF_HOME": HF_HOME})
)


@app.cls(
    image=cosmos_image,
    gpu="H100",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=3600,
    scaledown_window=300,
)
class Cosmos:
    @modal.method()
    def clip(self, request: dict) -> dict:
        """`{"image": png, "prompt", "seed", "frames"?}` -> `{"mp4", "fps", "model"}`.
        Raises if the guardrail blocks the clip."""
        import json

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "still.png").write_bytes(request["image"])
            spec = {
                "inference_type": "image2world",
                "name": "clip",
                "prompt": f"{request['prompt']} {STATIC_CAMERA}",
                "negative_prompt": NEGATIVE,
                "input_path": "still.png",
                "seed": int(request["seed"]),
                "num_output_frames": int(request.get("frames", COSMOS_FRAMES)),
            }
            (root / "clip.json").write_text(json.dumps(spec))
            subprocess.run(  # noqa: S603 - fixed argv; only file paths we wrote vary
                [
                    "/cosmos/.venv/bin/python",
                    "examples/inference.py",
                    "-i",
                    str(root / "clip.json"),
                    "-o",
                    str(root / "out"),
                    "--inference-type=image2world",
                    "--model=2B/post-trained",
                ],
                cwd="/cosmos",
                check=True,
                env={**os.environ, "HF_HOME": HF_HOME},
            )
            WEIGHTS.commit()
            (video,) = sorted((root / "out").rglob("*.mp4"))
            return {
                "mp4": video.read_bytes(),
                "fps": COSMOS_FPS,
                "model": f"nvidia/Cosmos-Predict2.5-2B@{COSMOS_COMMIT[:7]}",
            }


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


def _segment_models():  # noqa: ANN202 - the module, imported where it was copied
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

    def _embedder(self, request: dict):  # noqa: ANN202 - a segment_models.SiglipEmbedder
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
