"""World models on Modal GPUs: the three image/video models the teachers call. **Written
against each model's own documented entry point; never run on a GPU.**

    Fixer    nvidia/Fixer (Apache code, NVIDIA Open Model License weights). One image in,
             one image out: a render with 3DGS artifacts -> a clean one. Teacher B's filler.
    Wan      Wan-AI/Wan2.2-TI2V-5B-Diffusers (Apache). Still + prompt -> 121 frames at
             24 fps, 720p class. Teacher A's clip source.
    Distill  gsplat: the lifted fill refined against the filled views, the measured scan
             frozen (tools/captures/distill_fill.py).
    Cosmos   nvidia/Cosmos-Predict2.5-2B, post-trained (NVIDIA Open Model License, gated).
             Image2World: still + prompt -> 77 frames at 16 fps. Teacher A's second clip
             source. Its guardrails stay on -- the licence requires them.

The request and response of every method are plain dicts of bytes, strings and numbers,
so the client (`tools/captures/world_model_client.py`) needs `modal` and nothing else from
here; `tools/captures/tests/test_world_model_client.py` pins the method names and keys on
both sides by reading this file with `ast`.

What was checked, 2026-10-01, and what was not:

* Fixer: the repository at `FIXER_COMMIT` was read; its inference script's functions are
  what `Fixer.fix` calls, and it expects the base model at `/work/models/base/`, which is
  where the weights volume is mounted. The Hub repo `nvidia/Fixer` holds `base/` and
  `pretrained/` (5.5 GB). Its base container is NGC's
  `cosmos-predict2-container:1.2`, pulled with the `ngc` secret.
* Wan: the Hub model card's diffusers recipe, with `WanImageToVideoPipeline` for the
  image-conditioned case; 1280x704 is the 720p size, the aspect following the input.
* Cosmos: the repository at `COSMOS_COMMIT` was read -- `examples/inference.py` with a
  JSON spec, `--inference-type=image2world --model=2B/post-trained`, output saved at 16 fps.
  The Hugging Face token must have accepted the licences of Cosmos-Predict2.5-2B,
  Cosmos-Reason1-7B and Cosmos-Guardrail1 (it had not, on 2026-10-01).
* Nothing has been built by Modal or run. Package pins below are the first guess that
  `modal deploy` proves; the runbook (docs/WORLD_MODEL_RUNBOOK.md) says what to try first.

Secrets: `huggingface` (HF_TOKEN), `ngc` (REGISTRY_USERNAME=$oauthtoken,
REGISTRY_PASSWORD=<NGC API key>). Weights are cached in the volume
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
NGC_SECRET = modal.Secret.from_name("ngc")

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

fixer_image = (
    modal.Image.from_registry(FIXER_BASE, secret=NGC_SECRET)
    .run_commands(
        # Fixer's Dockerfile.cosmos, line for line, then its repository at the pinned commit.
        'pip install --no-deps "cosmos-predict2==1.0.9"',
        "pip install lpips vision-aided-loss natsort git+https://github.com/openai/CLIP.git "
        '"torchmetrics[image]" "huggingface_hub>=0.30"',
        f"git clone {FIXER_REPO} /work/fixer && git -C /work/fixer checkout {FIXER_COMMIT}",
    )
    .env({"HF_HOME": HF_HOME})
)


@app.cls(
    image=fixer_image,
    gpu="L40S",
    volumes={"/work/models": WEIGHTS},
    secrets=[HF_SECRET],
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
    .pip_install("numpy==1.26.4", "ninja", "jaxtyping", "rich", GSPLAT_WHEEL)
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
