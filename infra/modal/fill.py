"""Teacher B on Modal: `tools/captures/teacher_fill.py` (drop test, fill from outside) with
NVIDIA Fixer as the filler, on a GPU, for scans already published as tilesets.

docs/WORLD_MODEL_RUNBOOK.md is the order of runs; this file only puts them on Modal. Each
job runs the teacher_fill CLI in a CPU container (the renders are numpy) with the same
`tools/captures` code the CPU runs; `world_model_client.FixerFiller` calls the `Fixer`
class below, which holds the model on an L40S (`world_model_client.LOCAL_CLASSES`, so
nothing has to be deployed first). Nothing is written to any bucket: the strips, reports
and inferred tilesets come back to the caller.

**No NGC container and no secrets.** Fixer's own Dockerfile starts from NGC's
`cosmos-predict2-container:1.2`, which needs an NGC key. That container is
cosmos-predict2's environment, and cosmos-predict2's repository builds the same
environment from its `uv.lock` on a public CUDA base (its Dockerfile); `Fixer` uses that,
then Fixer's Dockerfile lines on top. The weights (`nvidia/Fixer`, not gated) download
into the shared weights volume once.

**Renderer.** `--renderer cpu` renders the views with `splat_render.render` (numpy point
samples) in a CPU container; `--renderer gsplat` runs the same job in a GPU container
(`run_job_gsplat`, an L4) where `splat_render.GsplatRenderer` rasterizes them with gsplat
-- what a viewer draws, and what Fixer is trained to clean. `--parity-test` runs
`tools/captures/tests/test_gsplat_parity.py` (CPU against gsplat on the yard) on that GPU.

`split:<scan>` runs `tools/captures/split_objects.py split` (C4): the scan's chosen objects
(`SPLIT_ARGS`) out into tilesets of their own and the holes they leave filled with the same
filler, renderer and distill; the split tileset comes back as `split.tar.gz`.

Run from the repository root (`.github/workflows/fill.yml` does):

    modal run infra/modal/fill.py --jobs drop:yard,drop:spool
    modal run infra/modal/fill.py --jobs fill:camp --renderer gsplat --distill 1500
    modal run infra/modal/fill.py --jobs split:pumpkin --fillers fixer-t50 --renderer gsplat
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "hexapod-fill"
app = modal.App(APP_NAME)
WEIGHTS = modal.Volume.from_name("hexapod-world-model-weights", create_if_missing=True)
CAPTURES = "/root/captures"
YARD = "/root/yard"

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    LOCAL_CAPTURES = ROOT / "tools" / "captures"
    LOCAL_YARD = ROOT / "data" / "tiles" / "synthetic-yard" / "splat"
else:
    LOCAL_CAPTURES, LOCAL_YARD = Path(CAPTURES), Path(YARD)

PUBLIC = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/runs"
#: Scans by short name: a public tileset URL, or a path in the container (the yard ships).
SCANS: dict[str, str] = {
    "yard": f"{YARD}/tileset.json",
    "spool": f"{PUBLIC}/8e1cc115-cb80-4af2-81fc-dccaf6b65891/package/splat/tileset.json",
    "pumpkin": f"{PUBLIC}/430c1932-5b6a-47b1-bb71-bb7fa2fec86b/package/splat/tileset.json",
    "camp": f"{PUBLIC}/50c25673-0940-4574-9b96-0b21362f83ca/package/splat/tileset.json",
}
#: The dropped cube's half size (m) per scan: the test's region, not a fill setting.
DROP_HALF_SIZE_M = {"yard": 0.4, "spool": 0.15, "pumpkin": 0.15, "camp": 0.4}
#: `split:<scan>`: which objects to take out. The pumpkin's are classed in place (A6), so
#: by id: the red pumpkin (instance 3) and the fragments segmentation left under other ids.
SPLIT_ARGS = {"pumpkin": ["--ids", "3", "--absorb"], "spool": ["--select", "loose", "--absorb"]}
#: Fixer runs at 16:9 (1024x576); views are rendered at that aspect so it is not stretched.
DROP_SIZE = (640, 360)
FILL_SIZE = (1024, 576)

#: The public bucket answers Python's default user agent with 403; curl's is let through.
USER_AGENT = "curl/8.5.0 (hexapod-fill)"

#: Fillers by short name: the CPU stand-in, and Fixer at its README's step (250), at 100
#: and 50, and shown the render presmoothed. Any `teacher_fill.make_filler` spec works too.
FILLERS = {
    "telea": "telea",
    "fixer": "world_model_client:FixerFiller",
    "fixer-t100": "world_model_client:FixerFiller?timestep=100",
    "fixer-t50": "world_model_client:FixerFiller?timestep=50",
    "fixer-smooth": "world_model_client:FixerFiller?presmooth_px=1",
    # Generative inpainting (tools/captures/inpaint_models.py), prompted from instances.json;
    # `-chain`: each view shown the earlier views' fill (teacher_fill._chained_fill).
    "sdxl": "world_model_client:GenerativeFiller?model=sdxl",
    "sdxl-chain": "world_model_client:GenerativeFiller?model=sdxl&chain=1",
    "qwen": "world_model_client:GenerativeFiller?model=qwen",
    "qwen-chain": "world_model_client:GenerativeFiller?model=qwen&chain=1",
    "flux": "world_model_client:GenerativeFiller?model=flux",
    # LaMa alone (texture, never an object), and LaMa's fill refined by SDXL at strength 0.6.
    "lama": "world_model_client:GenerativeFiller?model=lama",
    "lama-chain": "world_model_client:GenerativeFiller?model=lama&chain=1",
    "sdxl-lama": "world_model_client:GenerativeFiller?model=sdxl&prefill=lama&strength=0.6",
    "sdxl-lama-chain": (
        "world_model_client:GenerativeFiller?model=sdxl&prefill=lama&strength=0.6&chain=1"
    ),
}

# --- Fixer -----------------------------------------------------------------------------------

FIXER_REPO = "https://github.com/nv-tlabs/Fixer.git"
FIXER_COMMIT = "b39dfcaf4eeec90dc943b057ff368c16252c6c6e"
#: cosmos-predict2 at the commit that is its 1.0.9 (what Fixer's Dockerfile pip-installs).
COSMOS_REPO = "https://github.com/nvidia-cosmos/cosmos-predict2.git"
COSMOS_COMMIT = "661da4774b0ca41d082a0ecbeb47550bcf07e03f"
FIXER_RESOLUTION = 1024
MODEL_PKL = "/work/models/pretrained/pretrained_fixer.pkl"
FIXER_TIMESTEP = 250

fixer_image = (
    # cosmos-predict2's Dockerfile base; its lockfile is for Python 3.10.
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install uv==0.8.12",
        f"git clone {COSMOS_REPO} /cosmos && git -C /cosmos checkout {COSMOS_COMMIT}",
        # The repository's locked environment (torch 2.6 cu126, transformer-engine, apex,
        # flash-attn as prebuilt wheels), into the image's own Python. `--frozen`, not
        # `--locked`: Modal's PyPI mirror as the default index reads as a stale lock.
        "cd /cosmos && UV_PROJECT_ENVIRONMENT=$(python -c 'import sys; print(sys.prefix)') "
        "uv sync --frozen --inexact --no-install-project --extra cu126",
        # Fixer's Dockerfile.cosmos, then its repository at the pinned commit.
        'pip install --no-deps "cosmos-predict2==1.0.9"',
        "pip install lpips natsort",
        f"git clone {FIXER_REPO} /work/fixer && git -C /work/fixer checkout {FIXER_COMMIT}",
    )
)


@app.cls(
    image=fixer_image,
    gpu="L40S",
    volumes={"/work/models": WEIGHTS},
    memory=49152,
    timeout=3600,
    scaledown_window=120,
)
class Fixer:
    """The same contract as `world_models.Fixer`: `{"images": [png]}` -> `{"images": [png],
    "model"}`, each output the size of its input."""

    @modal.enter()
    def load(self) -> None:
        import torch

        _snapshot("nvidia/Fixer", Path("/work/models"), "pretrained/pretrained_fixer.pkl")
        sys.path.insert(0, "/work/fixer/src")
        import inference_pretrained_model as fixer  # Fixer's own script, as a module

        torch.set_grad_enabled(False)
        self.fixer = fixer
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.width, self.height = fixer.get_resolution_size(FIXER_RESOLUTION)
        self.model = fixer.load_and_compile_model(
            model_path=MODEL_PKL,
            timestep=FIXER_TIMESTEP,
            vae_skip_connection=False,  # as the README runs it
            batch_size=1,
            device=self.device,
            dtype=self.dtype,
            compile=False,
        )
        self.model.set_eval()
        self.calls = 0
        self.seconds = 0.0
        self.load_report = _load_report(self.model, MODEL_PKL)

    @modal.method()
    def fix(self, request: dict) -> dict:
        return self._fix(request)

    def _fix(self, request: dict) -> dict:
        import torch
        from PIL import Image

        started = time.time()
        out = []
        for blob in request["images"]:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            size = image.size
            width, height = self.fixer.get_resolution_size(
                int(request.get("resolution", FIXER_RESOLUTION))
            )
            x = self.fixer.preprocess_image(
                image.resize((width, height), Image.BILINEAR), self.device, self.dtype
            )
            # The diffusion step Fixer denoises from (its README runs 250).
            step = int(request.get("timestep", FIXER_TIMESTEP))
            self.model.timesteps = torch.tensor([step], device="cuda")
            with torch.no_grad():
                y = self.fixer.model_inference(
                    self.model, 1, height, width, self.dtype, self.device, x=x
                )
            buffer = io.BytesIO()
            self.fixer.postprocess_output(y, size).save(buffer, format="PNG")
            out.append(buffer.getvalue())
        self.calls += len(out)
        self.seconds += time.time() - started
        return {"images": out, "model": f"nvidia/Fixer@{FIXER_COMMIT[:7]}"}

    @modal.method()
    def examples(self) -> dict[str, bytes]:
        """The repository's own example renders and what this setup makes of them: the
        check that the model is sound, apart from what our renders look like."""
        out = {}
        for path in sorted(Path("/work/fixer/examples").glob("*.png")):
            blob = path.read_bytes()
            out[f"{path.stem}-in.png"] = blob
            out[f"{path.stem}-out.png"] = self._fix({"images": [blob]})["images"][0]
        out["load.json"] = json.dumps(self.load_report, indent=1).encode()
        return out


def _load_report(model: object, path: str) -> dict:
    """How much of the checkpoint the model took: Fixer loads it with `strict=False`, so a
    key mismatch (another cosmos-predict2) would leave the base weights silently."""
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    report = {}
    for part in ("unet", "vae"):
        own = getattr(model, part).state_dict()
        theirs = checkpoint[f"state_dict_{part}"]
        same = [k for k in theirs if k in own]
        equal = sum(
            1
            for k in same
            if own[k].shape == theirs[k].shape
            and torch.equal(own[k].detach().cpu(), theirs[k].to(own[k].dtype))
        )
        report[part] = {
            "modelKeys": len(own),
            "checkpointKeys": len(theirs),
            "matched": len(same),
            "equalAfterLoad": equal,
            "missing": sorted(set(own) - set(theirs))[:8],
            "unexpected": sorted(set(theirs) - set(own))[:8],
        }
    return report


def _snapshot(repo: str, local: Path, marker: str) -> None:
    """The Hub repo in the weights volume, downloaded once."""
    if (local / marker).exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(local), token=os.environ.get("HF_TOKEN"))
    WEIGHTS.commit()


# --- Generative inpainting, as world_models.InpaintSDXL / InpaintQwen / InpaintFlux --------

#: The workspace's Hugging Face secret: `huggingface` by the runbook, but named otherwise in
#: this workspace (fill.yml finds it by its prefix and passes it here).
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
inpaint_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "diffusers==0.40.0",
        "transformers>=5,<6",
        "accelerate>=1.6",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "huggingface_hub>=1.23,<2",
        "pillow",
    )
    .env({"HF_HOME": "/weights/hf"})
    .add_local_file(LOCAL_CAPTURES / "inpaint_models.py", "/root/inpaint_models.py")
)


def _inpaint_module():  # noqa: ANN202 - inpaint_models, imported where it was copied
    sys.path.insert(0, "/root")
    import inpaint_models

    inpaint_models.find_token()
    return inpaint_models


@app.cls(
    image=inpaint_image,
    gpu="L40S",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    timeout=3600,
    scaledown_window=120,
)
class InpaintSDXL:
    @modal.enter()
    def load(self) -> None:
        self.im = _inpaint_module()
        self.pipe = self.im.load("sdxl")
        self.lama = self.im.load_lama("/weights/lama")
        WEIGHTS.commit()

    @modal.method()
    def inpaint(self, request: dict) -> dict:
        return self.im.inpaint("sdxl", self.pipe, request, lama=self.lama)


@app.cls(
    image=inpaint_image,
    gpu="H100",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=131072,
    timeout=3600,
    scaledown_window=120,
)
class InpaintQwen:
    @modal.enter()
    def load(self) -> None:
        self.im = _inpaint_module()
        self.pipe = self.im.load("qwen")
        WEIGHTS.commit()

    @modal.method()
    def inpaint(self, request: dict) -> dict:
        return self.im.inpaint("qwen", self.pipe, request)


@app.cls(
    image=inpaint_image,
    gpu="H100",
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=98304,
    timeout=3600,
    scaledown_window=120,
)
class InpaintFlux:
    @modal.enter()
    def load(self) -> None:
        self.im = _inpaint_module()
        self.pipe = self.im.load("flux")
        WEIGHTS.commit()

    @modal.method()
    def inpaint(self, request: dict) -> dict:
        return self.im.inpaint("flux", self.pipe, request)


@app.function(image=inpaint_image, secrets=[HF_SECRET], timeout=300)
def inpaint_access() -> dict[str, str]:
    """Whether the workspace's Hugging Face token can read each inpainting model (the gated
    ones need the licence accepted on its account)."""
    im = _inpaint_module()
    token = im.find_token()
    return {"token": "found" if token else "missing", **im.access(token)}


@app.function(
    image=inpaint_image,
    secrets=[HF_SECRET],
    volumes={"/weights": WEIGHTS},
    cpu=4.0,
    memory=16384,
    timeout=3 * 3600,
)
def inpaint_prefetch(keys: list[str]) -> dict[str, float]:
    """Each model's repositories into the weights volume once, before the jobs start, so
    parallel jobs do not each download them (Qwen-Image is about 58 GB). Seconds per key."""
    from huggingface_hub import snapshot_download

    im = _inpaint_module()
    seconds = {}
    for key in keys:
        started = time.time()
        for repo in im.MODELS[key].repos:
            snapshot_download(repo, token=os.environ.get("HF_TOKEN"), max_workers=16)
        if key == "sdxl":
            im.load_lama("/weights/lama", device="cpu")
        WEIGHTS.commit()
        seconds[key] = round(time.time() - started, 1)
    return seconds


# --- Video models that fill a clip's unknown pixels (generative fill bake-off) ---------------

#: tools/captures/video_fill_models.py on diffusers: the stack `world_models.video_image` ran
#: Wan 2.2 on (diffusers 0.40, transformers 5, torch 2.8), with Cosmos' guardrail package.
video_fill_image = (
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
        "numpy",
    )
    .env({"HF_HOME": "/weights/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(LOCAL_CAPTURES / "video_fill_models.py", "/root/video_fill_models.py")
)

#: Generators by key: the Modal class that holds each, its GPU, and what the budget guard
#: plans with: seconds per 49-frame clip and per container start (measured on an L40S, run
#: 37383986439: VACE 116-117 s a clip, Wan2.2 114-117 s, Cosmos 288-303 s, loads 17-37 s;
#: rounded up), and `capS`, the most one clip may take (its class's timeout, about 2.5 times
#: the measured clip).
VIDEO_GPU = "L40S"
GPU_RATES = {"L4": 0.80, "L40S": 1.95, "A10": 1.10, "H100": 3.95}
GENERATORS = {
    "vace": {"cls": "FillVace", "gpu": VIDEO_GPU, "clipS": 150, "loadS": 90, "capS": 300},
    "wan22": {"cls": "FillWan22", "gpu": VIDEO_GPU, "clipS": 150, "loadS": 90, "capS": 300},
    "cosmos": {"cls": "FillCosmos", "gpu": VIDEO_GPU, "clipS": 330, "loadS": 90, "capS": 750},
    # The per-view baseline: LaMa on InpaintSDXL's L40S (SDXL loads with it), ~1 s a keyframe;
    # called by the job one keyframe at a time.
    "lama": {"cls": "InpaintSDXL", "gpu": "L40S", "clipS": 15, "loadS": 120, "capS": 60},
}
#: A generator's container stays this long after its last call (`scaledown_window`).
VIDEO_IDLE_S = 60
#: A container start (the model's load) at most this long.
VIDEO_START_S = 6 * 60
#: The gen/holdout job's own container (gsplat renders, depth, lift, carve, distil), minutes:
#: the estimate per scan, and the timeout of every job.
GEN_JOB_MIN = {"spool": 30, "pumpkin": 35, "camp": 50}
GEN_JOB_CAP_MIN = 75
#: The worst case the guard refuses past: every clip and container start at this many times
#: its estimate, and on top one clip per generator that hangs to its cap.
WORST_FACTOR = 1.5


def _video_module():  # noqa: ANN202 - video_fill_models, imported where it was copied
    sys.path.insert(0, "/root")
    import video_fill_models

    video_fill_models.find_token()
    return video_fill_models


def _video_load(key: str) -> tuple[object, object, float, str]:
    """The model loaded, or why not. A load that raised would end the container and Modal
    would start another for the same call, and another (a 4 h loop once, runbook section 9);
    instead every call to a container that could not load fails at once with the reason."""
    import traceback

    started = time.time()
    try:
        vfm = _video_module()
        pipe = vfm.load(key)
    except Exception:  # noqa: BLE001 - every call reports it
        return None, None, time.time() - started, traceback.format_exc()[-4000:]
    return vfm, pipe, time.time() - started, ""


def _video_fill(owner: object, key: str, request: dict) -> dict:
    """One clip; a failure comes back as `error` with the GPU seconds it took, so the
    run's cost counts it (`actual_gen_cost`)."""
    done = {"loadSeconds": round(owner.load_seconds, 1), "gpu": VIDEO_GPU}  # type: ignore[attr-defined]
    if owner.error:  # type: ignore[attr-defined]
        why = owner.error[-1400:]  # type: ignore[attr-defined]
        return {"error": f"{key} did not load:\n{why}", "seconds": 0.0, **done}
    start = time.time()
    try:
        out = owner.vfm.fill_clip(key, owner.pipe, request)  # type: ignore[attr-defined]
    except Exception as error:  # reported back with its seconds; the other clips go on
        with contextlib.suppress(Exception):
            import torch

            torch.cuda.empty_cache()
        return {"error": repr(error)[:1500], "seconds": round(time.time() - start, 1), **done}
    return {**out, **done}


@app.cls(
    image=video_fill_image,
    gpu=VIDEO_GPU,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=65536,
    timeout=GENERATORS["vace"]["capS"],
    startup_timeout=VIDEO_START_S,
    scaledown_window=VIDEO_IDLE_S,
    max_containers=2,
)
class FillVace:
    """Wan2.1-VACE 1.3B (Apache-2.0): masked video-to-video (`video_fill_models`)."""

    @modal.enter()
    def load(self) -> None:
        self.vfm, self.pipe, self.load_seconds, self.error = _video_load("vace")

    @modal.method()
    def fill_clip(self, request: dict) -> dict:
        return _video_fill(self, "vace", request)


@app.cls(
    image=video_fill_image,
    gpu=VIDEO_GPU,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=65536,
    timeout=GENERATORS["wan22"]["capS"],
    startup_timeout=VIDEO_START_S,
    scaledown_window=VIDEO_IDLE_S,
    max_containers=2,
)
class FillWan22:
    """Wan2.2 TI2V-5B (Apache-2.0): its clean-token conditioning on every known token."""

    @modal.enter()
    def load(self) -> None:
        self.vfm, self.pipe, self.load_seconds, self.error = _video_load("wan22")

    @modal.method()
    def fill_clip(self, request: dict) -> dict:
        return _video_fill(self, "wan22", request)


@app.cls(
    image=video_fill_image,
    gpu=VIDEO_GPU,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=98304,
    timeout=GENERATORS["cosmos"]["capS"],
    startup_timeout=VIDEO_START_S,
    scaledown_window=VIDEO_IDLE_S,
    max_containers=2,
)
class FillCosmos:
    """Cosmos-Predict2 2B Video2World (NVIDIA Open Model License; its guardrail on)."""

    @modal.enter()
    def load(self) -> None:
        self.vfm, self.pipe, self.load_seconds, self.error = _video_load("cosmos")

    @modal.method()
    def fill_clip(self, request: dict) -> dict:
        return _video_fill(self, "cosmos", request)


VIDEO_CLASSES = {"FillVace": FillVace, "FillWan22": FillWan22, "FillCosmos": FillCosmos}


@app.function(image=video_fill_image, secrets=[HF_SECRET], timeout=300)
def video_access(keys: list[str]) -> dict[str, str]:
    """Whether the workspace's Hugging Face token can read each video model's repositories
    (Cosmos and its guardrail are gated)."""
    vfm = _video_module()
    token = vfm.find_token()
    repos = vfm.access(token, tuple(keys))
    readable = [k for k in keys if all(repos.get(r) == "ok" for r in vfm.MODELS[k].repos)]
    return {"token": "found" if token else "missing", "readable": readable, **repos}


@app.function(
    image=video_fill_image,
    secrets=[HF_SECRET],
    volumes={"/weights": WEIGHTS},
    cpu=4.0,
    memory=16384,
    timeout=2 * 3600,
)
def video_prefetch(keys: list[str]) -> dict[str, float]:
    """Each video model's files into the weights volume once, on a CPU, before any GPU
    container waits on a download. Seconds per key."""
    vfm = _video_module()
    token = vfm.find_token()
    seconds = {}
    for key in keys:
        started = time.time()
        vfm.prefetch(key, token)
        WEIGHTS.commit()
        seconds[key] = round(time.time() - started, 1)
    return seconds


# --- Distill (gsplat), as world_models.Distill ----------------------------------------------

GSPLAT_WHEEL = (
    "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
    "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    "#sha256=01e1fd63dc69c9945e70158c818c3bb07fedf4aabcf020e6608d264cf27cc5dd"
)
distill_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    # gsplat imports `packaging`, which nothing else here installs.
    .pip_install("numpy==1.26.4", "ninja", "jaxtyping", "rich", "packaging", GSPLAT_WHEEL)
    .add_local_file(LOCAL_CAPTURES / "distill_fill.py", "/root/distill_fill.py")
)


@app.cls(image=distill_image, gpu="L40S", timeout=3600, scaledown_window=60)
class Distill:
    @modal.method()
    def run(self, request: dict) -> dict:
        sys.path.insert(0, "/root")
        import distill_fill

        return distill_fill.run(request)


# --- The teacher_fill jobs ------------------------------------------------------------------

job_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # tools/captures/pyproject.toml's dependencies.
        "numpy>=1.26",
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11",
        "opencv-python-headless>=4.10",
    )
    .add_local_dir(LOCAL_YARD, YARD)
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "tests/**", "**/*.pyc"],
    )
)

#: The same job with gsplat: the Distill image's torch and wheel (cp310), the captures'
#: dependencies at versions built for numpy 1.26, and the tests (for `--parity-test`).
_gsplat_base = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.4.1+cu124", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy==1.26.4",
        "ninja",
        "jaxtyping",
        "rich",
        "packaging",
        GSPLAT_WHEEL,
        "pillow>=10",
        "laspy[lazrs]>=2.5",
        "pyproj>=3.6",
        "scipy>=1.11,<1.16",
        "opencv-python-headless==4.10.0.84",
        "pytest>=8",
    )
)
gsplat_job_image = (
    _gsplat_base.env({"HEXAPOD_YARD_TILESET": f"{YARD}/tileset.json"})
    .add_local_dir(LOCAL_YARD, YARD)
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.pyc"],
    )
)
#: The generative fill job (`generative_fill.py`): the gsplat job's stack, plus Depth Anything
#: V2 Small through transformers (a version for torch 2.4) and boto3 for the run's photos and
#: poses in the private bucket.
genfill_image = (
    _gsplat_base.pip_install("transformers==4.46.3", "huggingface_hub>=0.26,<1", "boto3")
    .env({"HF_HOME": "/weights/hf"})
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.pyc", "tests/**"],
    )
)
#: The private bucket (a run's photos, poses and placement), as infra/modal/app.py reads it.
STORAGE_SECRET = modal.Secret.from_name("twin-object-storage")

#: Wall seconds spent in each GPU class's calls from this job (cold starts included).
REMOTE_SECONDS: dict[str, float] = {}


class _Timed:
    """A GPU class whose `.method.remote(...)` calls are timed into `REMOTE_SECONDS`."""

    def __init__(self, name: str, cls: object) -> None:
        self._name, self._instance = name, cls()  # type: ignore[operator]

    def __getattr__(self, method: str) -> object:
        bound = getattr(self._instance, method)
        name = self._name

        class _Call:
            @staticmethod
            def remote(*args: object, **kwargs: object) -> object:
                started = time.time()
                try:
                    return bound.remote(*args, **kwargs)
                finally:
                    REMOTE_SECONDS[name] = REMOTE_SECONDS.get(name, 0.0) + time.time() - started

        return _Call


def _get(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as error:
            # Several jobs fetching one scan at once meet the bucket's rate limit (429).
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"GET {url}: HTTP {error.code} {error.reason}") from None
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def _fetch(url: str, out: Path, every: bool = False) -> Path:
    """The tileset, its leaves and its view cones (what teacher_fill reads); `every`: its
    merged parents too (what split_objects rewrites)."""
    import concurrent.futures

    if not url.startswith("https://"):
        source = Path(url)
        shutil.copytree(source.parent, out)
        return out / source.name
    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    (out / "tileset.json").write_bytes(_get(url, 120))
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    uris: list[str] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        uri = tile.get("content", {}).get("uri")
        if uri and (every or not tile.get("children")):
            uris.append(uri)
    if cones := document["root"].get("extras", {}).get("viewCones", {}).get("uri"):
        uris.append(cones)
    # The instances, for `split` (and the embedding beside them).
    if instances := document["root"].get("extras", {}).get("instances", {}).get("uri"):
        uris.append(instances)
        uris.append(str(PurePosixPath(instances).with_name("instances.emb")))

    def get(uri: str) -> None:
        (out / uri).parent.mkdir(parents=True, exist_ok=True)
        (out / uri).write_bytes(_get(f"{base}/{uri}", 600))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(get, uris))
    return out / "tileset.json"


def _remote_classes() -> None:
    sys.path.insert(0, CAPTURES)
    import world_model_client

    world_model_client.LOCAL_CLASSES.update(
        Fixer=lambda: _Timed("Fixer", Fixer),
        Distill=lambda: _Timed("Distill", Distill),
        InpaintSDXL=lambda: _Timed("InpaintSDXL", InpaintSDXL),
        InpaintQwen=lambda: _Timed("InpaintQwen", InpaintQwen),
        InpaintFlux=lambda: _Timed("InpaintFlux", InpaintFlux),
    )


def _teacher_fill(argv: list[str], module: str = "teacher_fill") -> tuple[int, dict | None, str]:
    """`<module>.main(argv)` in this process (`teacher_fill`, or `split_objects`); its JSON
    and its log."""
    import importlib

    main = importlib.import_module(module).main
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 1
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = main(argv)
        except Exception:  # reported back; the other jobs go on
            import traceback

            traceback.print_exc()
    text = stdout.getvalue()
    start = text.find("{")
    try:
        result = json.loads(text[start:]) if start >= 0 else None
    except json.JSONDecodeError:
        result = None
    return code, result, (text[-20000:] + stderr.getvalue()[-20000:])


def _tree(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _tar(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as out:
        out.add(folder, arcname=folder.name)
    return buffer.getvalue()


@app.function(image=job_image, cpu=8.0, memory=65536, timeout=4 * 3600)
def run_job(kind: str, scan: str, filler: str, options: dict) -> dict:
    """One `teacher_fill.py drop|fill` on one scan with one filler (`telea` or `fixer`).
    Returns its report, the strips, and for `fill` the inferred layer (a tar.gz) and the
    measured tileset.json with the layer linked."""
    return _run_job(kind, scan, filler, options)


@app.function(image=gsplat_job_image, gpu="L4", cpu=8.0, memory=65536, timeout=4 * 3600)
def run_job_gsplat(kind: str, scan: str, filler: str, options: dict) -> dict:
    """`run_job` with the views rasterized by gsplat on this container's GPU."""
    return _run_job(kind, scan, filler, {**options, "renderer": "gsplat"})


@app.function(image=gsplat_job_image, gpu="L4", cpu=4.0, memory=32768, timeout=3600)
def parity() -> str:
    """tests/test_gsplat_parity.py on a GPU: its output (the agreement it measured)."""
    import subprocess

    os.chdir(CAPTURES)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "-q", "-rs", "tests/test_gsplat_parity.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f"exit {done.returncode}\n{done.stdout[-20000:]}\n{done.stderr[-5000:]}"


def _run_job(kind: str, scan: str, filler: str, options: dict) -> dict:
    _remote_classes()
    REMOTE_SECONDS.clear()  # a container may run several jobs
    os.chdir(CAPTURES)
    started = time.time()
    spec = FILLERS.get(filler, filler)
    files: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tileset = _fetch(SCANS[scan], root / "scan", every=kind == "split")
        timings = {"fetchS": round(time.time() - started, 1)}
        document = json.loads(tileset.read_text(encoding="utf-8"))
        logs = []
        if "viewCones" not in document["root"].get("extras", {}):
            import view_cones

            t = time.time()
            view_cones.build_view_cones(tileset, tileset.parent, tileset)
            timings["viewConesS"] = round(time.time() - t, 1)
            logs.append("view cones backfilled from the leaves")
        save = root / "save"
        t = time.time()
        module = "teacher_fill"
        if kind == "split":
            width, height = FILL_SIZE
            module = "split_objects"
            argv = ["split", str(tileset.parent), str(root / "split"), "--filler", spec]
            argv += SPLIT_ARGS.get(scan, []) + ["--views", str(options.get("views", 6))]
            if options.get("max_scale_m"):
                argv += ["--max-scale-m", str(options["max_scale_m"])]
            if options.get("distill"):
                argv += ["--distill", str(options["distill"]), "--distill-on", "modal"]
        elif kind == "drop":
            width, height = DROP_SIZE
            argv = [
                "drop",
                str(tileset),
                "--filler",
                spec,
                "--half-size-m",
                str(options.get("half_size_m", DROP_HALF_SIZE_M.get(scan, 0.15))),
                "--views",
                str(options.get("views", 4)),
            ]
        else:
            width, height = FILL_SIZE
            out = root / "inferred"
            argv = ["fill", str(tileset), str(out), "--filler", spec]
            argv += ["--views", str(options.get("views", 8)), "--mode", "ring"]
            if options.get("max_scale_m"):
                argv += ["--max-scale-m", str(options["max_scale_m"])]
            if options.get("distill"):
                argv += ["--distill", str(options["distill"]), "--distill-on", "modal"]
        argv += ["--width", str(width), "--height", str(height), "--save", str(save)]
        argv += ["--renderer", options.get("renderer", "cpu")]
        code, result, log = _teacher_fill(argv, module)
        timings["teacherS"] = round(time.time() - t, 1)
        timings.update({f"{k.lower()}S": round(v, 1) for k, v in REMOTE_SECONDS.items()})
        logs.append(log)
        if save.exists():
            files.update({f"strips/{k}": v for k, v in _tree(save).items()})
        if kind == "split" and code == 0 and (root / "split" / "tileset.json").exists():
            files["split.tar.gz"] = _tar(root / "split")
        if kind == "fill" and code == 0 and (root / "inferred" / "tileset.json").exists():
            import teacher_fill

            files["inferred.tar.gz"] = _tar(root / "inferred")
            teacher_fill.link_inferred(tileset, root / "inferred" / "tileset.json")
            files["measured-tileset.json"] = tileset.read_bytes()
        timings["totalS"] = round(time.time() - started, 1)
        return {
            "kind": kind,
            "scan": scan,
            "filler": filler,
            "ok": code == 0 and result is not None,
            "argv": argv,
            "result": result,
            "timings": timings,
            "files": files,
            "log": "\n".join(logs),
        }


# --- Generative fill (tools/captures/generative_fill.py): gen:<scan>, holdout:<scan> --------

#: What each scan's generative fill fills, from what. `job`: the pipeline run whose photos
#: and COLMAP poses (private bucket) give real cameras; none for an uploaded splat (the camp:
#: its view cones' observers). `args`: generative_fill's region options. `prompt`: what the
#: clips show (a plain caption of the scan; the object a hole is filled under is not named).
GEN_SCANS: dict[str, dict] = {
    "spool": {
        "job": "8e1cc115-cb80-4af2-81fc-dccaf6b65891",
        "args": ["--roi-instance", "57"],
        "prompt": (
            "A weathered round cable-spool table standing on a lawn, filmed on a phone in "
            "daylight; the camera moves slowly and smoothly; nothing in the scene moves."
        ),
    },
    "pumpkin": {
        "job": "430c1932-5b6a-47b1-bb71-bb7fa2fec86b",
        "args": [
            "--roi-instance",
            "74",
            "--hide-instance",
            "74",
            "--roi-grow",
            "0.4",
            "--roi-top",
            "0.45",
        ],
        "prompt": (
            "Dry straw on a hay bale in a garden, filmed on a phone in daylight; the camera "
            "moves slowly and smoothly over it; nothing in the scene moves."
        ),
        "negative": "pumpkin, orange fruit, gourd, ball",
    },
    "camp": {
        "job": None,
        "args": ["--roi-pick", "roof", "--crop-m", "25"],
        "prompt": (
            "Log cabins and tents in a forest clearing seen from above, filmed by a drone in "
            "daylight; the camera moves slowly and smoothly; nothing in the scene moves."
        ),
    },
}
#: `holdout:<scan>`: the share of the cameras that see the region from highest, held out.
HOLDOUT_SHARE = 0.15
#: How long a job waits for one clip (queued behind the others on its generator).
GEN_CALL_TIMEOUT_S = 50 * 60


def _spawn_video(cls: str, method: str, request: dict) -> object:
    if cls in VIDEO_CLASSES:
        return getattr(VIDEO_CLASSES[cls](), method).spawn(request)
    raise ValueError(f"no video class {cls!r}")


def _wait_video(call: object) -> dict:
    """The clip; one not back within `GEN_CALL_TIMEOUT_S` is cancelled (and its container
    stopped), since nobody will collect it."""
    try:
        return call.get(timeout=GEN_CALL_TIMEOUT_S)  # type: ignore[attr-defined]
    except TimeoutError:
        call.cancel(terminate_containers=True)  # type: ignore[attr-defined]
        raise


def _private_client():  # noqa: ANN202 - boto3's client
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def _fetch_private(prefix: str, out: Path) -> int:
    """Every object under `prefix` in the private bucket into `out`; how many."""
    import concurrent.futures

    client = _private_client()
    bucket = os.environ["OBJECT_STORAGE_BUCKET"]
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]

    def get(key: str) -> None:
        # A prefix that is a whole key (a single file) lands under its own name.
        dest = out / (key[len(prefix) :].lstrip("/") or PurePosixPath(key).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(get, keys))
    return len(keys)


@app.function(
    image=genfill_image,
    gpu="L4",
    cpu=8.0,
    memory=65536,
    timeout=GEN_JOB_CAP_MIN * 60,  # the budget: a hung job costs at most this much L4
    volumes={"/weights": WEIGHTS},
    secrets=[STORAGE_SECRET, HF_SECRET],
)
def run_genfill(kind: str, scan: str, options: dict) -> dict:
    """`generative_fill.py run` on one scan with every generator in `options["generators"]`:
    `gen` fills the scan's region; `holdout` first holds out the cameras that see it from
    highest (the ground truth the renders compare with). Returns the report, the renders and
    each generator's inferred layer (`<slug>/inferred.tar.gz`)."""
    _remote_classes()
    import world_model_client

    # The baseline's keyframes, each at most `capS` (the class's own timeout is an hour).
    lama_cap = GENERATORS["lama"]["capS"]
    world_model_client.LOCAL_CLASSES["InpaintSDXL"] = lambda: _Timed(
        "InpaintSDXL",
        InpaintSDXL.with_options(timeout=lama_cap),  # type: ignore[attr-defined]
    )
    REMOTE_SECONDS.clear()
    os.chdir(CAPTURES)
    import generative_fill as gf

    gf.BACKEND = (_spawn_video, _wait_video)
    started = time.time()
    setup = GEN_SCANS[scan]
    files: dict[str, bytes] = {}
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tileset = _fetch(SCANS[scan], root / "scan")
        timings["fetchS"] = round(time.time() - started, 1)
        # (A scan without cameras gets its view cones from its leaves inside generative_fill:
        # the observer points it starts from are not in a published viewcones.bin.)
        argv = ["run", str(tileset), str(root / "out"), "--scan", scan]
        if setup["job"]:
            t = time.time()
            run = f"runs/{setup['job']}"
            counts = {
                "poses": _fetch_private(f"{run}/pose/poses/", root / "poses"),
                "frames": _fetch_private(f"{run}/normalize/frames/", root / "frames"),
                "placement": _fetch_private(f"{run}/place/placement.json", root / "place"),
            }
            timings["privateS"] = round(time.time() - t, 1)
            timings["privateFiles"] = counts  # type: ignore[assignment]
            placement = next((root / "place").rglob("*.json"), None)
            argv += ["--poses", str(root / "poses"), "--frames", str(root / "frames")]
            if placement is not None:
                argv += ["--placement", str(placement)]
        argv += setup["args"] + ["--prompt", setup["prompt"]]
        if setup.get("negative"):
            argv += ["--negative", setup["negative"]]
        if kind == "holdout":
            # The frames from above are held out, and the ROI's look is never
            # treated as known (its shape still is), so the fill has to make it.
            argv += ["--holdout-above", str(options.get("holdout", HOLDOUT_SHARE))]
            argv += ["--unknown-roi"]
        for key in (
            "paths",
            "seeds",
            "rounds",
            "round2_paths",
            "frames_per_clip",
            "distill",
            "max_clips",
            "steps",
        ):
            if options.get(key) is not None:
                argv += [f"--{key.replace('_', '-')}", str(options[key])]
        argv += ["--generators", ",".join(options["generators"]), "--renderer", "gsplat"]
        argv += ["--depth", "depth-anything", "--distill-on", "local"]
        t = time.time()
        # Not through `_teacher_fill`: its progress streams to the run's log as it goes.
        code, log = 1, ""
        try:
            code = gf.main(argv)
        except (Exception, SystemExit):  # noqa: BLE001 - reported back with what it wrote
            import traceback

            log = traceback.format_exc()
            sys.stderr.write(log)
        timings["fillS"] = round(time.time() - t, 1)
        out = root / "out"
        result = (
            json.loads((out / "report.json").read_text(encoding="utf-8"))
            if (out / "report.json").exists()
            else None
        )
        if (out / "renders").exists():
            files.update({f"renders/{k}": v for k, v in _tree(out / "renders").items()})
        if (out / "report.json").exists():
            files["report.json"] = (out / "report.json").read_bytes()
        for layer in sorted(out.glob("*/inferred/tileset.json")):
            files[f"{layer.parent.parent.name}/inferred.tar.gz"] = _tar(layer.parent)
        files["measured-tileset.json"] = tileset.read_bytes()
    # The per-view baseline's calls (InpaintSDXL, an L40S), wall seconds from here.
    timings.update({f"{k.lower()}S": round(v, 1) for k, v in REMOTE_SECONDS.items()})
    timings["totalS"] = round(time.time() - started, 1)
    return {
        "kind": kind,
        "scan": scan,
        "ok": code == 0 and result is not None,
        "argv": argv,
        "result": result,
        "timings": timings,
        "files": files,
        "log": log,
    }


def estimate_gen_cost(jobs: list[tuple[str, str]], generators: list[str], options: dict) -> dict:
    """What a set of gen/holdout jobs should cost, from the planning numbers above, and the
    worst case the guard holds it to. Per generator: its clips, its container starts (per
    job, round 1's parallel seeds up to `max_containers`, and one for round 2), each with a
    load and an idle tail. Per job its L4. The worst case: clips and starts at
    `WORST_FACTOR` times, one clip per generator hung to its cap, every job at its timeout."""
    paths, rounds = int(options.get("paths", 2)), int(options.get("rounds", 2))
    seeds, round2 = int(options.get("seeds", 1)), int(options.get("round2_paths", 1))
    out: dict[str, float] = {}
    worst: dict[str, float] = {}
    clips: dict[str, int] = {}
    for key in generators:
        g = GENERATORS[key]
        first = paths * (seeds if key != "lama" else 1)
        per_job = first + (round2 if rounds >= 2 else 0)
        n = per_job * len(jobs)
        clips[key] = n
        starts = len(jobs) * (min(first, 2) + (1 if rounds >= 2 else 0))
        seconds = n * g["clipS"] + starts * (g["loadS"] + VIDEO_IDLE_S)
        rate = GPU_RATES[g["gpu"]] / 3600
        out[key] = seconds * rate
        worst[key] = (WORST_FACTOR * seconds + max(g["capS"] - g["clipS"], 0)) * rate
    for kind, scan in jobs:
        out[f"{kind}:{scan}"] = GEN_JOB_MIN.get(scan, 45) / 60 * GPU_RATES["L4"]
        worst[f"{kind}:{scan}"] = GEN_JOB_CAP_MIN / 60 * GPU_RATES["L4"]
    return {
        "clips": clips,
        "usd": {k: round(v, 2) for k, v in out.items()},
        "totalUsd": round(sum(out.values()), 2),
        "worstUsd": round(sum(worst.values()), 2),
    }


def actual_gen_cost(results: list[dict]) -> dict:
    """What the jobs cost, from what they report: each generator's call seconds plus, per
    container start (a distinct load time), its load and its idle tail; each job's L4 wall
    time. Modal bills a little more (image pulls, the CPU side), so this is a floor."""
    gpu_s: dict[str, float] = {}
    loads: dict[str, set] = {}
    for r in results:
        report = (r.get("result") or {}) if isinstance(r, dict) else {}
        for name, entry in (report.get("candidates") or {}).items():
            for call in entry.get("calls", []) or []:
                gpu_s[name] = gpu_s.get(name, 0.0) + float(call.get("seconds") or 0.0)
                if call.get("loadSeconds") is not None:
                    loads.setdefault(name, set()).add(float(call["loadSeconds"]))
    usd: dict[str, float] = {}
    for name, s in gpu_s.items():
        starts = loads.get(name, set())
        total = s + sum(starts) + VIDEO_IDLE_S * max(1, len(starts))
        usd[name] = total / 3600 * GPU_RATES[VIDEO_GPU]
    for r in results:
        if isinstance(r, dict) and r.get("timings", {}).get("totalS"):
            usd[f"{r['kind']}:{r['scan']}"] = r["timings"]["totalS"] / 3600 * GPU_RATES["L4"]
            sdxl = r["timings"].get("inpaintsdxlS")
            if sdxl:
                key = f"lama ({r['kind']}:{r['scan']})"
                usd[key] = (float(sdxl) + VIDEO_IDLE_S) / 3600 * GPU_RATES["L40S"]
    return {
        "usd": {k: round(v, 3) for k, v in usd.items()},
        "totalUsd": round(sum(usd.values()), 2),
    }


@app.function(image=job_image, cpu=8.0, memory=65536, timeout=3600)
def probe(scan: str) -> dict:
    """Fixer on one scan's renders, prepared several ways: which input it can work with."""
    _remote_classes()
    os.chdir(CAPTURES)
    import cv2
    import numpy as np
    import teacher_fill as tf
    import view_cones as vc
    from splat_render import load_tileset
    from world_model_client import decode_png, encode_png

    with tempfile.TemporaryDirectory() as work:
        tileset = _fetch(SCANS[scan], Path(work) / "scan")
        if not (tileset.parent / vc.URI).exists():
            vc.build_view_cones(tileset, tileset.parent, tileset)
        splats = load_tileset(tileset)
        grid = vc.cone_grid_from_tileset(tileset)
    w, h = FILL_SIZE
    # The first view fill_scan would make: on a ring outside what the view cones fade.
    texels = vc.lookup(grid.texels, grid.origin, grid.cell, grid.dims, splats.positions)
    targets = splats.positions[texels[:, 2] != vc.OMNI]
    centre = targets.mean(axis=0)
    reach = float(np.percentile(np.linalg.norm(splats.positions - centre, axis=1), 95))
    cams = tf.plan_views(
        grid, targets, count=8, mode="ring", ring_radius_m=1.5 * reach, width=w, height=h
    )[:1]
    files: dict[str, bytes] = {}
    scores: dict[str, dict] = {}
    for k, cam in enumerate(cams):
        cond = tf.condition(splats, cam, grid)
        frame = cond.full
        raw = tf.to_u8(frame.rgb)
        seen = tf.to_u8(cond.seen.rgb)
        hole = (cond.mask | (cond.seen.alpha < tf.SEEN_ALPHA)).astype(np.uint8) * 255
        seen_telea = cv2.inpaint(seen, hole, 5, cv2.INPAINT_TELEA)
        variants = {
            "raw": (raw, FIXER_RESOLUTION, FIXER_TIMESTEP),
            "raw-t100": (raw, FIXER_RESOLUTION, 100),
            "raw-t50": (raw, FIXER_RESOLUTION, 50),
            "raw-t400": (raw, FIXER_RESOLUTION, 400),
            "seen": (seen, FIXER_RESOLUTION, FIXER_TIMESTEP),
            "seen-telea": (seen_telea, FIXER_RESOLUTION, FIXER_TIMESTEP),
            "seen-telea-t100": (seen_telea, FIXER_RESOLUTION, 100),
        }
        covered = frame.alpha >= 0.5
        blur = lambda x: cv2.GaussianBlur(x, (0, 0), 2.0)  # noqa: E731
        for name, (given, resolution, step) in variants.items():
            request = {"images": [encode_png(given)], "resolution": resolution, "timestep": step}
            out = decode_png(Fixer().fix.remote(request)["images"][0])
            scores[f"view{k}-{name}"] = {
                "gate": round(tf.psnr(out, given, covered), 2),
                "gateBlur2": round(tf.psnr(blur(out), blur(given), covered), 2),
                "vsRawBlur2": round(tf.psnr(blur(out), blur(raw), covered), 2),
            }
            files[f"view{k}-{name}.png"] = encode_png(np.concatenate([given, out], axis=1))
    files["scores.json"] = json.dumps(scores, indent=1).encode()
    return {"scan": scan, "files": files}


# --- Round 2: anchor, then propagate (tools/captures/anchor_fill.py): anchor:<scan>, leaveout:<scan>

#: The editor (Qwen-Image-Edit-2511 + its Lightning LoRA) and VACE-14B: tools/captures/
#: anchor_models.py on diffusers 0.40 / transformers 5 / peft, the versions its CPU check ran.
anchor_model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.40.0",
        "transformers==5.18.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "huggingface_hub>=1.23,<2",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "imageio[ffmpeg]>=2.37",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": "/weights/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(LOCAL_CAPTURES / "anchor_models.py", "/root/anchor_models.py")
    .add_local_file(LOCAL_CAPTURES / "video_fill_models.py", "/root/video_fill_models.py")
)
#: What the budget guard plans with (seconds; UNVERIFIED until the first run measures them):
#: an editor call at its fill size with two photos and 8 steps, and with none / 4 steps; a
#: VACE-14B call on one set (~30 frames at 832x480, 25 steps); loads from the weights volume;
#: the idle tails; each class's per-call cap (its timeout).
ANCHOR_GPU = "H100"
EDIT_CALL_S = {"anchor": 12.0, "anchorNoRefs": 6.0, "prop": 6.0, "propNoRefs": 3.5, "update": 5.0}
EDIT_LOAD_S = 240
EDIT_IDLE_S = 120
EDIT_CAP_S = 300
SET_CALL_S = 200
SET_LOAD_S = 300
SET_IDLE_S = 60
SET_CAP_S = 1200
#: The job's own L4 (renders, quality, depth, lift, carving, distil, scores), minutes.
ANCHOR_JOB_MIN = {"spool": 45, "pumpkin": 55}
ANCHOR_JOB_CAP_MIN = 110
#: Views the estimate assumes (the selection's maxima give the worst case).
ANCHOR_VIEWS = {"anchors": 6, "props": 16}


def _anchor_module():  # noqa: ANN202 - anchor_models, imported where it was copied
    sys.path.insert(0, "/root")
    import anchor_models
    import video_fill_models

    video_fill_models.find_token()
    return anchor_models


@app.cls(
    image=anchor_model_image,
    gpu=ANCHOR_GPU,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=131072,
    timeout=EDIT_CAP_S,
    startup_timeout=15 * 60,
    scaledown_window=EDIT_IDLE_S,
    max_containers=2,
)
class EditQwen:
    """Qwen-Image-Edit-2511 (Apache-2.0) with its Lightning LoRAs (Apache-2.0): the anchor and
    propagation fills (`anchor_models.edit`)."""

    @modal.enter()
    def load(self) -> None:
        import traceback

        started = time.time()
        self.error = ""
        try:
            self.am = _anchor_module()
            self.editor = self.am.load_editor()
        except Exception:  # noqa: BLE001 - every call reports it (no reload loop)
            self.error = traceback.format_exc()[-3000:]
        self.load_seconds = round(time.time() - started, 1)

    @modal.method()
    def edit(self, request: dict) -> dict:
        done = {"loadSeconds": self.load_seconds, "gpu": ANCHOR_GPU}
        if self.error:
            return {
                "error": f"the editor did not load:\n{self.error[-1400:]}",
                "seconds": 0.0,
                **done,
            }
        started = time.time()
        try:
            return {**self.am.edit(self.editor, request), **done}
        except Exception as error:  # noqa: BLE001 - reported with its seconds
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            return {"error": repr(error)[:1500], "seconds": round(time.time() - started, 1), **done}


@app.cls(
    image=anchor_model_image,
    gpu=ANCHOR_GPU,
    volumes={"/weights": WEIGHTS},
    secrets=[HF_SECRET],
    memory=131072,
    timeout=SET_CAP_S,
    startup_timeout=15 * 60,
    scaledown_window=SET_IDLE_S,
    max_containers=1,
)
class FillVace14:
    """Wan2.1-VACE-14B (Apache-2.0): every target view as a frame of one clip
    (`anchor_models.fill_set`)."""

    @modal.enter()
    def load(self) -> None:
        import traceback

        started = time.time()
        self.error = ""
        self.distill = False
        try:
            self.am = _anchor_module()
            self.pipe, self.distill = self.am.load_vace14()
        except Exception:  # noqa: BLE001
            self.error = traceback.format_exc()[-3000:]
        self.load_seconds = round(time.time() - started, 1)

    @modal.method()
    def fill_set(self, request: dict) -> dict:
        done = {"loadSeconds": self.load_seconds, "gpu": ANCHOR_GPU, "distillLoaded": self.distill}
        if self.error:
            return {
                "error": f"VACE-14B did not load:\n{self.error[-1400:]}",
                "seconds": 0.0,
                **done,
            }
        started = time.time()
        try:
            return {**self.am.fill_set(self.pipe, request, self.distill), **done}
        except Exception as error:  # noqa: BLE001
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            return {"error": repr(error)[:1500], "seconds": round(time.time() - started, 1), **done}


ANCHOR_CLASSES = {"EditQwen": EditQwen, "FillVace14": FillVace14, "InpaintQwen": InpaintQwen}
#: Each repository the round-2 run reads, and what of it (None: everything).
ANCHOR_REPOS: dict[str, tuple[str, ...] | None] = {
    "Qwen/Qwen-Image-Edit-2511": None,
    "lightx2v/Qwen-Image-Edit-2511-Lightning": (
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    ),
    "Wan-AI/Wan2.1-VACE-14B-diffusers": (
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/*",
        "vae/*",
    ),
    "lightx2v/Wan2.1-Distill-Loras": ("wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors",),
    "depth-anything/prompt-depth-anything-vitl-hf": None,
    "depth-anything/Depth-Anything-V2-Small-hf": None,
}


@app.function(image=anchor_model_image, secrets=[HF_SECRET], timeout=300)
def anchor_access() -> dict[str, str]:
    """Whether the workspace's token can read each repository (none of them is gated)."""
    from huggingface_hub import auth_check

    am_vfm = _anchor_module()  # finds the token into HF_TOKEN
    del am_vfm
    out = {}
    for repo in ANCHOR_REPOS:
        try:
            auth_check(repo, token=os.environ.get("HF_TOKEN"))
            out[repo] = "ok"
        except Exception as error:  # noqa: BLE001 - reported
            out[repo] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return out


@app.function(
    image=anchor_model_image,
    secrets=[HF_SECRET],
    volumes={"/weights": WEIGHTS},
    cpu=4.0,
    memory=16384,
    timeout=3 * 3600,
)
def anchor_prefetch(repos: list[str]) -> dict[str, float]:
    """Each repository into the weights volume once, on a CPU, before a GPU waits on it."""
    from huggingface_hub import snapshot_download

    _anchor_module()
    seconds = {}
    for repo in repos:
        started = time.time()
        patterns = ANCHOR_REPOS.get(repo)
        snapshot_download(
            repo,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=list(patterns) if patterns else None,
            max_workers=16,
        )
        WEIGHTS.commit()
        seconds[repo] = round(time.time() - started, 1)
    return seconds


#: The job's image: the generative fill job's (gsplat on torch 2.4, Python 3.10), plus
#: transformers for Prompt Depth Anything, LPIPS and DreamSim for the scores (evaluation
#: only), torchvision for them; torch must stay 2.4.1 (the gsplat wheel's).
anchorfill_image = (
    _gsplat_base.apt_install("git")
    .run_commands(
        "pip install 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cu124",
        "pip install 'transformers==4.57.1' 'huggingface_hub>=0.34,<1.0' 'boto3' "
        "'lpips==0.1.4' 'dreamsim==0.2.1' 'open-clip-torch==2.32.0' 'timm==1.0.15' "
        "'peft==0.15.2' 'ftfy' 'regex'",
        "python -c \"import torch, numpy; assert torch.__version__.startswith('2.4.1'), "
        "torch.__version__; assert numpy.__version__.startswith('1.26'), numpy.__version__\"",
    )
    .env({"HF_HOME": "/weights/hf", "TORCH_HOME": "/weights/torch"})
    .add_local_dir(
        LOCAL_CAPTURES,
        CAPTURES,
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.pyc", "tests/**"],
    )
)

#: What each scan's round-2 fill reads: the pipeline run with its photos and COLMAP poses
#: (private bucket), a plain caption for the prompts (the scene, not a region), and which
#: cameras its leave-out check holds out (the spool: its highest; the pumpkin: its lowest).
ANCHOR_SCANS: dict[str, dict] = {
    "spool": {
        "job": "8e1cc115-cb80-4af2-81fc-dccaf6b65891",
        "caption": "a weathered round wooden cable-spool table standing on a lawn",
        "leave": "high",
    },
    "pumpkin": {
        "job": "430c1932-5b6a-47b1-bb71-bb7fa2fec86b",
        "caption": "an orange pumpkin and a red pumpkin resting on a bed of dry straw in a garden",
        "leave": "low",
    },
}
#: anchor_fill options a run may set (`--anchor-options k=v,...`), and their checks.
ANCHOR_OPTIONS = {
    "arms": r"(refs|norefs|vace)(\+(refs|norefs|vace))*",
    "seeds": r"[1-8]",
    "prop_seeds": r"[1-4]",
    "anchor_steps": r"[1-9][0-9]?",
    "prop_steps": r"[1-9][0-9]?",
    "update_steps": r"[1-9][0-9]?",
    "update_strengths": r"(0\.[0-9]+)?(\+0\.[0-9]+)*",
    "set_seeds": r"[1-3]",
    "set_steps": r"[1-9][0-9]?",
    "set_distill": r"true|false",
    "distill": r"[0-9]{1,4}",
    "update_distill": r"[0-9]{1,4}",
    "anchors": r"[1-8]\+[1-8]",
    "propagation": r"[0-9]{1,2}\+[0-9]{1,2}",
    "leave_share": r"0\.[0-9]+",
    "lightning": r"true|false",
    "fallback": r"true|false",
    "vae_area": r"[0-9]{5,7}",
}


def parse_anchor_options(text: str) -> dict[str, str]:
    """`k=v,k=v` checked against `ANCHOR_OPTIONS` (lists joined by `+`)."""
    import re

    out = {}
    for pair in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = pair.partition("=")
        if key not in ANCHOR_OPTIONS or not re.fullmatch(ANCHOR_OPTIONS[key], value):
            raise SystemExit(f"anchor option {pair!r}: one of {sorted(ANCHOR_OPTIONS)}, checked")
        out[key] = value
    return out


def anchor_argv(options: dict[str, str]) -> list[str]:
    argv = []
    for key, value in options.items():
        if key == "fallback":
            if value == "true":
                argv.append("--fallback")
            continue
        value = value.replace("+", ",")
        argv += [f"--{key.replace('_', '-')}", value]
    return argv


def _spawn_anchor(cls: str, method: str, request: dict) -> object:
    if cls not in ANCHOR_CLASSES:
        raise ValueError(f"no class {cls!r}")
    return getattr(ANCHOR_CLASSES[cls](), method).spawn(request)


def _wait_anchor(call: object) -> dict:
    """A call back within its class's cap (and the queue before it), else cancelled."""
    try:
        return call.get(timeout=GEN_CALL_TIMEOUT_S)  # type: ignore[attr-defined]
    except TimeoutError:
        call.cancel(terminate_containers=True)  # type: ignore[attr-defined]
        raise


@app.function(
    image=anchorfill_image,
    gpu="L4",
    cpu=8.0,
    memory=65536,
    timeout=ANCHOR_JOB_CAP_MIN * 60,
    volumes={"/weights": WEIGHTS},
    secrets=[STORAGE_SECRET, HF_SECRET],
)
def run_anchorfill(kind: str, scan: str, options: dict) -> dict:
    """`anchor_fill.py run` on one scan: `anchor` on every camera (the layers to publish),
    `leaveout` with the scan's leave-out cameras held out (scored, not published). Returns the
    report, the renders and each arm's layer (`<layer>/inferred.tar.gz`)."""
    os.chdir(CAPTURES)
    sys.path.insert(0, CAPTURES)
    import video_fill_models

    video_fill_models.find_token()
    import anchor_fill as af

    af.BACKEND = (_spawn_anchor, _wait_anchor)
    started = time.time()
    setup = ANCHOR_SCANS[scan]
    files: dict[str, bytes] = {}
    timings: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        tileset = _fetch(SCANS[scan], root / "scan")
        timings["fetchS"] = round(time.time() - started, 1)
        run = f"runs/{setup['job']}"
        t = time.time()
        counts = {
            "poses": _fetch_private(f"{run}/pose/poses/", root / "poses"),
            "frames": _fetch_private(f"{run}/normalize/frames/", root / "frames"),
            "placement": _fetch_private(f"{run}/place/placement.json", root / "place"),
        }
        timings["privateS"] = round(time.time() - t, 1)
        timings["privateFiles"] = counts
        placement = next((root / "place").rglob("*.json"), None)
        argv = ["run", str(tileset), str(root / "out"), "--scan", scan]
        argv += ["--caption", setup["caption"], "--poses", str(root / "poses")]
        argv += ["--frames", str(root / "frames"), "--renderer", "gsplat"]
        if placement is not None:
            argv += ["--placement", str(placement)]
        if kind == "leaveout":
            argv += ["--leave-out", setup["leave"]]
        argv += anchor_argv(options)
        t = time.time()
        code, log = 1, ""
        try:
            code = af.main(argv)
        except (Exception, SystemExit):  # noqa: BLE001 - reported back with what it wrote
            import traceback

            log = traceback.format_exc()
            sys.stderr.write(log)
        timings["fillS"] = round(time.time() - t, 1)
        out = root / "out"
        result = (
            json.loads((out / "report.json").read_text(encoding="utf-8"))
            if (out / "report.json").exists()
            else None
        )
        if (out / "renders").exists():
            files.update({f"renders/{k}": v for k, v in _tree(out / "renders").items()})
        if (out / "report.json").exists():
            files["report.json"] = (out / "report.json").read_bytes()
        for layer in sorted(out.glob("*/inferred/tileset.json")):
            files[f"{layer.parent.parent.name}/inferred.tar.gz"] = _tar(layer.parent)
        files["measured-tileset.json"] = tileset.read_bytes()
    timings["totalS"] = round(time.time() - started, 1)
    return {
        "kind": kind,
        "scan": scan,
        "ok": code == 0 and result is not None,
        "argv": argv,
        "result": result,
        "timings": timings,
        "files": files,
        "log": log,
    }


def anchor_counts(options: dict[str, str], worst: bool = False) -> dict[str, int]:
    """Calls one job makes: editor calls by kind, set calls. `worst`: the selection's
    maxima (8 anchors, 24 propagation views)."""
    arms = options.get("arms", "refs+norefs+vace").split("+")
    anchors = int(options.get("anchors", "4+8").split("+")[1 if worst else 0])
    props = int(options.get("propagation", "12+24").split("+")[1 if worst else 0])
    if not worst:
        anchors = max(anchors, ANCHOR_VIEWS["anchors"])
        props = max(props, ANCHOR_VIEWS["props"])
    seeds, prop_seeds = int(options.get("seeds", 4)), int(options.get("prop_seeds", 2))
    rounds = len([s for s in options.get("update_strengths", "0.4").split("+") if s])
    refs = any(a in arms for a in ("refs", "vace"))
    norefs = "norefs" in arms
    sequential = [a for a in arms if a != "vace"]
    out = {
        "anchor": anchors * seeds * int(refs),
        "anchorNoRefs": anchors * seeds * int(norefs),
        "prop": props * prop_seeds * int("refs" in sequential),
        "propNoRefs": props * prop_seeds * int("norefs" in sequential),
        "update": rounds * (anchors + props) * len(arms),
        "set": int(options.get("set_seeds", 2)) * int("vace" in arms),
    }
    return out


def estimate_anchor_cost(jobs: list[tuple[str, str]], options: dict[str, str]) -> dict:
    """What the round-2 jobs should cost and the worst case the guard holds them to. Per job:
    its editor calls (`EDIT_CALL_S` each), set calls, two editor starts and one VACE start
    (loads and idle tails), its L4. The worst case: the selection's maxima, every call and
    start at `WORST_FACTOR` times, one call per class hung to its cap, every job at its
    timeout."""
    rate_gpu, rate_l4 = GPU_RATES[ANCHOR_GPU] / 3600, GPU_RATES["L4"] / 3600
    usd: dict[str, float] = {}
    worst: dict[str, float] = {}
    for kind, scan in jobs:
        name = f"{kind}:{scan}"
        for label, table, factor in (("", usd, 1.0), ("", worst, WORST_FACTOR)):
            n = anchor_counts(options, worst=table is worst)
            edit_s = sum(n[k] * EDIT_CALL_S[k] for k in EDIT_CALL_S)
            set_s = n["set"] * SET_CALL_S
            starts = 2 * (EDIT_LOAD_S + EDIT_IDLE_S) + (SET_LOAD_S + SET_IDLE_S) * int(n["set"] > 0)
            table[f"{name} editor{label}"] = factor * edit_s * rate_gpu
            table[f"{name} vace{label}"] = factor * set_s * rate_gpu
            table[f"{name} starts{label}"] = factor * starts * rate_gpu
        usd[f"{name} L4"] = ANCHOR_JOB_MIN.get(scan, 60) / 60 * GPU_RATES["L4"]
        worst[f"{name} L4"] = ANCHOR_JOB_CAP_MIN * 60 * rate_l4
    worst["hung editor call"] = EDIT_CAP_S * rate_gpu
    worst["hung VACE call"] = SET_CAP_S * rate_gpu
    return {
        "calls": {f"{k}:{s}": anchor_counts(options) for k, s in jobs},
        "usd": {k: round(v, 3) for k, v in usd.items()},
        "totalUsd": round(sum(usd.values()), 2),
        "worstUsd": round(sum(worst.values()), 2),
    }


def actual_anchor_cost(results: list[dict]) -> dict:
    """What the jobs cost from what they report: per GPU class its calls' seconds plus, per
    container start (a distinct load time), its load and idle tail; per job its L4 wall
    time. A floor: Modal also bills image pulls and the CPU side."""
    seconds = {"EditQwen": 0.0, "FillVace14": 0.0}
    loads: dict[str, set] = {"EditQwen": set(), "FillVace14": set()}
    usd: dict[str, float] = {}
    for r in results:
        report = (r.get("result") or {}) if isinstance(r, dict) else {}
        for call in report.get("calls", []) or []:
            cls = "FillVace14" if str(call.get("key", "")).startswith("set-") else "EditQwen"
            seconds[cls] += float(call.get("seconds") or 0.0)
            if call.get("loadSeconds") is not None:
                loads[cls].add(float(call["loadSeconds"]))
        if isinstance(r, dict) and r.get("timings", {}).get("totalS"):
            usd[f"{r['kind']}:{r['scan']} L4"] = r["timings"]["totalS"] / 3600 * GPU_RATES["L4"]
    idle = {"EditQwen": EDIT_IDLE_S, "FillVace14": SET_IDLE_S}
    for cls, s in seconds.items():
        if s or loads[cls]:
            total = s + sum(loads[cls]) + idle[cls] * max(1, len(loads[cls]))
            usd[cls] = total / 3600 * GPU_RATES[ANCHOR_GPU]
    return {
        "usd": {k: round(v, 3) for k, v in usd.items()},
        "gpuSeconds": {k: round(v, 1) for k, v in seconds.items()},
        "starts": {k: sorted(v) for k, v in loads.items()},
        "totalUsd": round(sum(usd.values()), 2),
    }


def _run_anchor(
    jobs: list[tuple[str, ...]],
    options: dict[str, str],
    out: Path,
    budget_usd: float,
    spent_usd: float,
) -> None:
    """The round-2 jobs: estimate, refuse past the budget, check access, prefetch, run in
    parallel, write each job's files and the cost."""
    for kind, scan in jobs:
        if scan not in ANCHOR_SCANS:
            raise SystemExit(
                f"job {kind}:{scan}: scans are {', '.join(ANCHOR_SCANS)} (camp has no poses)"
            )
    out.mkdir(parents=True, exist_ok=True)
    estimate = estimate_anchor_cost(list(jobs), options)
    estimate["budgetUsd"], estimate["spentUsd"], estimate["options"] = (
        budget_usd,
        spent_usd,
        options,
    )
    (out / "anchor-estimate.json").write_text(json.dumps(estimate, indent=1), encoding="utf-8")
    sys.stdout.write(f"round 2, estimated: {json.dumps(estimate)}\n")
    if budget_usd <= 0:
        raise SystemExit("round 2 runs only with --budget-usd set (the guard)")
    if estimate["worstUsd"] > budget_usd - spent_usd:
        raise SystemExit(
            f"the worst case ${estimate['worstUsd']} (estimated ${estimate['totalUsd']}) is more "
            f"than the ${budget_usd - spent_usd:.2f} left of the budget: not started"
        )
    access = anchor_access.remote()
    (out / "anchor-access.json").write_text(json.dumps(access, indent=1), encoding="utf-8")
    sys.stdout.write(f"model access: {json.dumps(access)}\n")
    refused = [r for r, v in access.items() if v != "ok"]
    if refused:
        raise SystemExit(f"the token cannot read {refused}: not started (no workaround)")
    arms = options.get("arms", "refs+norefs+vace").split("+")
    repos = [r for r in ANCHOR_REPOS if "Wan" not in r or "vace" in arms]
    seconds = anchor_prefetch.remote(repos)
    sys.stdout.write(f"weights fetched: {json.dumps(seconds)}\n")
    results, failed = [], []
    calls = [(kind, scan, options) for kind, scan in jobs]
    for result in run_anchorfill.starmap(calls, return_exceptions=True):
        if isinstance(result, BaseException):
            failed.append(repr(result))
            sys.stdout.write(f"round-2 job raised: {result!r}\n")
            continue
        results.append(result)
        base = f"{result['kind']}-{result['scan']}"
        folder = out / base
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for name, data in result.get("files", {}).items():
            if name.endswith("/inferred.tar.gz"):
                target = out / f"{base}-{name.split('/', 1)[0]}" / "inferred.tar.gz"
            else:
                target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        brief = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "result.json").write_text(json.dumps(brief, indent=1), encoding="utf-8")
        report = result.get("result") or {}
        headline = {
            "heldOut": (report.get("heldOut") or {}).get("mean"),
            "arms": {
                a: (e.get("evidence") or {}).get("gaussians")
                for a, e in (report.get("candidates") or {}).items()
            },
        }
        sys.stdout.write(f"{base}: ok={result['ok']} {json.dumps(headline)[:3000]}\n")
        if not result["ok"]:
            failed.append(base)
    cost = actual_anchor_cost(results)
    cost["estimate"] = estimate["totalUsd"]
    (out / "anchor-cost.json").write_text(json.dumps(cost, indent=1), encoding="utf-8")
    sys.stdout.write(f"round 2, cost from what the jobs report: {json.dumps(cost)}\n")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


@app.local_entrypoint()
def main(
    jobs: str = "drop:yard,drop:spool",
    fillers: str = "telea,fixer",
    views: int = 0,
    distill: int = 0,
    out: str = "fill-out",
    selftest: bool = False,
    probes: str = "",
    renderer: str = "cpu",
    parity_test: bool = False,
    max_scale_m: float = 0.0,
    generators: str = "vace,wan22,cosmos,lama",
    paths: int = 2,
    seeds: int = 1,
    rounds: int = 2,
    round2_paths: int = 1,
    frames_per_clip: int = 49,
    steps: int = 0,
    budget_usd: float = 0.0,
    spent_usd: float = 0.0,
    anchor_options: str = "",
) -> None:
    """Every `kind:scan` in `jobs` with every filler, in parallel containers; each result
    under `out/<kind>-<scan>-<filler>/`, and `out/summary.json`. `selftest`: also Fixer on
    its repository's examples, under `out/selftest/`. `renderer`: `cpu` or `gsplat` (the
    jobs on a GPU). `parity_test`: CPU against gsplat on the yard, in `out/parity.txt`.
    `max_scale_m` (fill jobs, 0 = off): condition without gaussians larger than this.

    `gen:<scan>` and `holdout:<scan>` run the generative fill (`run_genfill`) with every
    generator in `generators` (vace, wan22, cosmos, lama) -- `paths`, `seeds`, `rounds`,
    `round2_paths`, `frames_per_clip` and `steps` (0: each model's own) as
    generative_fill.py takes them. Their estimated cost is written first
    (`out/gen-estimate.json`); with `budget_usd` set, a run whose estimate exceeds what is
    left of it (`budget_usd - spent_usd`) does not start. What they cost, from what they
    report, is `out/gen-cost.json`.

    `anchor:<scan>` and `leaveout:<scan>` run round 2 (`run_anchorfill`, anchor then
    propagate) on the spool or the pumpkin with `anchor_options` (`k=v,...`,
    `ANCHOR_OPTIONS`); its estimate is `out/anchor-estimate.json`, and it runs only with
    `budget_usd` set and its worst case within what is left."""
    if renderer not in ("cpu", "gsplat"):
        raise SystemExit(f"renderer {renderer!r}: cpu or gsplat")
    every = [j.strip() for j in jobs.split(",") if j.strip()]
    gen_jobs = [tuple(j.split(":", 1)) for j in every if j.split(":", 1)[0] in ("gen", "holdout")]
    anchor_jobs = [
        tuple(j.split(":", 1)) for j in every if j.split(":", 1)[0] in ("anchor", "leaveout")
    ]
    own = ("gen", "holdout", "anchor", "leaveout")
    jobs = ",".join(j for j in every if j.split(":", 1)[0] not in own)
    if anchor_jobs:
        # Round 2 (anchor, then propagate), with `anchor_options` as anchor_fill takes them.
        options_v2 = parse_anchor_options(anchor_options)
        _run_anchor(anchor_jobs, options_v2, Path(out), budget_usd, spent_usd)
    if gen_jobs:
        options = {
            "generators": [g.strip() for g in generators.split(",") if g.strip()],
            "paths": paths,
            "seeds": seeds,
            "rounds": rounds,
            "round2_paths": round2_paths,
            "frames_per_clip": frames_per_clip,
            "steps": steps or None,
        }
        _run_gen(gen_jobs, options, Path(out), budget_usd, spent_usd)
    if parity_test:
        Path(out).mkdir(parents=True, exist_ok=True)
        text = parity.remote()
        (Path(out) / "parity.txt").write_text(text, encoding="utf-8")
        sys.stdout.write(f"parity: {text}\n")
    if selftest:
        folder = Path(out) / "selftest"
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in Fixer().examples.remote().items():
            (folder / name).write_bytes(data)
    if probes:
        for result in probe.map([s.strip() for s in probes.split(",") if s.strip()]):
            folder = Path(out) / f"probe-{result['scan']}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, data in result["files"].items():
                (folder / name).write_bytes(data)
            sys.stdout.write(f"probe {result['scan']}: {result['files']['scores.json'].decode()}\n")
    if not jobs:
        return
    calls = []
    for job in (j.strip() for j in jobs.split(",") if j.strip()):
        kind, _, scan = job.partition(":")
        if kind not in ("drop", "fill", "split") or scan not in SCANS:
            raise SystemExit(f"job {job!r}: drop|fill|split:<{'|'.join(SCANS)}>")
        options: dict = {"views": views} if views else {}
        if kind in ("fill", "split") and distill:
            options["distill"] = distill
        if kind in ("fill", "split") and max_scale_m > 0:
            options["max_scale_m"] = max_scale_m
        calls += [(kind, scan, f.strip(), options) for f in fillers.split(",") if f.strip()]
    summary, failed = [], []
    calls, skipped = _prepare_inpainting(calls, Path(out))
    failed += skipped
    runner = run_job_gsplat if renderer == "gsplat" else run_job
    for result in runner.starmap(calls, return_exceptions=True):
        if isinstance(result, BaseException):
            failed.append(repr(result))
            sys.stdout.write(f"job raised: {result!r}\n")
            continue
        folder = Path(out) / f"{result['kind']}-{result['scan']}-{result['filler']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for name, data in result.get("files", {}).items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
        brief = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "report.json").write_text(json.dumps(brief, indent=1), encoding="utf-8")
        summary.append(brief)
        sys.stdout.write(json.dumps({k: brief[k] for k in ("kind", "scan", "filler", "ok")}))
        sys.stdout.write(" " + json.dumps(_headline(brief)) + "\n")
        if not result["ok"]:
            failed.append(folder.name)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def _run_gen(
    jobs: list[tuple[str, ...]], options: dict, out: Path, budget_usd: float, spent_usd: float
) -> None:
    """The generative fill jobs: estimate, refuse past the budget, check access, prefetch,
    run in parallel, write each job's files and the cost."""
    for kind, scan in jobs:
        if scan not in GEN_SCANS:
            raise SystemExit(f"job {kind}:{scan}: scans are {', '.join(GEN_SCANS)}")
    unknown = [g for g in options["generators"] if g not in GENERATORS]
    if unknown:
        raise SystemExit(f"generators {unknown}: known are {', '.join(GENERATORS)}")
    out.mkdir(parents=True, exist_ok=True)
    estimate = estimate_gen_cost(list(jobs), options["generators"], options)
    estimate["budgetUsd"], estimate["spentUsd"] = budget_usd, spent_usd
    (out / "gen-estimate.json").write_text(json.dumps(estimate, indent=1), encoding="utf-8")
    sys.stdout.write(f"generative fill, estimated: {json.dumps(estimate)}\n")
    if budget_usd > 0 and estimate["worstUsd"] > budget_usd - spent_usd:
        raise SystemExit(
            f"the worst case ${estimate['worstUsd']} (estimated ${estimate['totalUsd']}) is more "
            f"than the ${budget_usd - spent_usd:.2f} left of the budget: not started"
        )
    video = [g for g in options["generators"] if g in ("vace", "wan22", "cosmos")]
    if video:
        access = video_access.remote(video)
        (out / "gen-access.json").write_text(json.dumps(access, indent=1), encoding="utf-8")
        sys.stdout.write(f"video models access: {json.dumps(access)}\n")
        # (Decided in the container: this side has `modal` and nothing else.)
        readable = [g for g in video if g in access.get("readable", [])]
        dropped = sorted(set(video) - set(readable))
        if dropped:
            sys.stdout.write(f"not run (the token cannot read them): {dropped}\n")
            options["generators"] = [g for g in options["generators"] if g not in dropped]
        if readable:
            seconds = video_prefetch.remote(readable)
            sys.stdout.write(f"video weights fetched: {json.dumps(seconds)}\n")
    results, failed = [], []
    calls = [(kind, scan, options) for kind, scan in jobs]
    for result in run_genfill.starmap(calls, return_exceptions=True):
        if isinstance(result, BaseException):
            failed.append(repr(result))
            sys.stdout.write(f"gen job raised: {result!r}\n")
            continue
        results.append(result)
        base = f"{result['kind']}-{result['scan']}"
        folder = out / base
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "log.txt").write_text(result["log"], encoding="utf-8")
        for name, data in result.get("files", {}).items():
            if name.endswith("/inferred.tar.gz"):
                # One folder per layer, as publish-fill.yml takes them: <job>/inferred.tar.gz.
                target = out / f"{base}-{name.split('/', 1)[0]}" / "inferred.tar.gz"
            else:
                target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        brief = {k: v for k, v in result.items() if k not in ("files", "log")}
        (folder / "result.json").write_text(json.dumps(brief, indent=1), encoding="utf-8")
        report = result.get("result") or {}
        headline = {
            name: (e.get("evidence") or {k: e.get(k) for k in ("failed", "skipped")})
            for name, e in (report.get("candidates") or {}).items()
        }
        sys.stdout.write(f"{base}: ok={result['ok']} {json.dumps(headline)[:3000]}\n")
        if not result["ok"]:
            failed.append(base)
    cost = actual_gen_cost(results)
    cost["estimate"] = estimate["totalUsd"]
    (out / "gen-cost.json").write_text(json.dumps(cost, indent=1), encoding="utf-8")
    sys.stdout.write(f"generative fill, cost from what the jobs report: {json.dumps(cost)}\n")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def _inpaint_key(filler: str) -> str | None:
    """The inpainting model a filler (short name or spec) runs, or None."""
    spec = FILLERS.get(filler, filler)
    if "GenerativeFiller" not in spec:
        return None
    query = dict(p.partition("=")[::2] for p in spec.partition("?")[2].split("&") if p)
    model = query.get("model", "sdxl")
    return "sdxl" if model == "lama" else model  # LaMa is held by InpaintSDXL


def _prepare_inpainting(calls: list[tuple], out: Path) -> tuple[list[tuple], list[str]]:
    """For the jobs with a generative filler: which models the workspace's Hugging Face token
    can read (`out/inpaint-access.json`); jobs whose model it cannot are dropped (and named
    as failed); the rest have their weights fetched into the volume once, before they run."""
    keys = sorted({k for c in calls if (k := _inpaint_key(c[2]))})
    if not keys:
        return calls, []
    sys.path.insert(0, str(LOCAL_CAPTURES))
    import inpaint_models

    access = inpaint_access.remote()
    out.mkdir(parents=True, exist_ok=True)
    (out / "inpaint-access.json").write_text(json.dumps(access, indent=1), encoding="utf-8")
    sys.stdout.write(f"inpainting access: {json.dumps(access)}\n")
    readable = [
        k for k in keys if all(access.get(r) == "ok" for r in inpaint_models.MODELS[k].repos)
    ]
    kept = [c for c in calls if _inpaint_key(c[2]) in (None, *readable)]
    skipped = [f"{c[0]}-{c[1]}-{c[2]} (no access)" for c in calls if c not in kept]
    if readable:
        seconds = inpaint_prefetch.remote(readable)
        (out / "inpaint-prefetch.json").write_text(json.dumps(seconds), encoding="utf-8")
        sys.stdout.write(f"inpainting weights fetched: {json.dumps(seconds)}\n")
    return kept, skipped


def _headline(brief: dict) -> dict:
    result = brief.get("result") or {}
    if brief["kind"] == "drop":
        return {"heldOut": result.get("heldOut"), "timings": brief["timings"]}
    if brief["kind"] == "split":
        fills = result.get("fills") or {}
        return {
            "objects": result.get("objects"),
            "heldOut": {k: (v or {}).get("heldOut") for k, v in fills.items()},
            "timings": brief["timings"],
        }
    keep = ("gaussians", "views", "meanConfidence", "perView", "distill")
    return {**{k: result.get(k) for k in keep}, "timings": brief["timings"]}
