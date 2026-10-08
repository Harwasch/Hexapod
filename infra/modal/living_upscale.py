"""Living view bake-off, the upscaler: every arm's clip to the render's size with FlashVSR v1.1.

The A-up column of the bake-off (infra/modal/living_bakeoff.py): the model's own pixels, made
1280 x 704 by a video super-resolution model instead of bicubic. A Modal app of its own because
`modal run` builds every image of an app before anything runs, and this one compiles
Block-Sparse-Attention: its build (or a failure of it) must not hold up the arms.

Steps (`--steps`):

    download  CPU: FlashVSR v1.1's weights into `hexapod-living-view-weights`.
    check     CPU: the image builds and its packages resolve, before a GPU is held.
    upscale   A100-80GB, one warm container: every clip the arms kept in the volume
              `hexapod-living-view` (`/data/clips/<arm>/<start>.mp4`), each timed.

    modal run infra/modal/living_upscale.py --steps download,check,upscale --budget-left 3
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-living-upscale"
app = modal.App(APP_NAME)

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path("/root/captures")

#: The bake-off's volumes (living_bakeoff.LV_WEIGHTS, RESULTS).
LV_WEIGHTS = modal.Volume.from_name(
    "hexapod-living-view-weights", create_if_missing=True, version=2
)
RESULTS = modal.Volume.from_name("hexapod-living-view", create_if_missing=True, version=2)
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")
START_SIZE = (1280, 704)

#: Modal list prices, $/s (modal.com/pricing, read 2026-10-08).
GPU_PER_S = {"A100-80GB": 0.000694, "": 0.0}
CPU_CORE_PER_S = 0.0000131
MEMORY_GIB_PER_S = 0.00000222
DOWNLOAD_RESERVATION = {"gpu": "", "cpu": 2.0, "memoryGiB": 8, "timeoutS": 3600}
CHECK_RESERVATION = {"gpu": "", "cpu": 1.0, "memoryGiB": 4, "timeoutS": 600}


def rate_per_s(reservation: dict) -> float:
    """Dollars per second of a container with this reservation."""
    return (
        GPU_PER_S[reservation["gpu"]]
        + reservation["cpu"] * CPU_CORE_PER_S
        + reservation["memoryGiB"] * MEMORY_GIB_PER_S
    )


#: Video super-resolution of every arm's clip (A-up on the page): FlashVSR v1.1 (tiny decoder),
#: its locality-constrained sparse attention on Block-Sparse-Attention, which its authors run
#: on A100s. 4x as its authors recommend for clips smaller than the render, 2x for clips
#: already the render's size (a supersampling pass), then area-resized to the render.
FVSR_REPO = "JunhaoZhuang/FlashVSR-v1.1"
FVSR_CODE = "https://github.com/OpenImagingLab/FlashVSR.git"
FVSR_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
BSA_CODE = "https://github.com/mit-han-lab/Block-Sparse-Attention.git"
BSA_COMMIT = "49d6c39e4dc0303442cda3bb758b3925d4399c49"
SEEDVR_REPO = "ByteDance-Seed/SeedVR2-3B"
UPSCALERS: dict[str, dict] = {
    "flashvsr": {
        "gpu": "A100-80GB",
        "cpu": 8.0,
        "memoryGiB": 64,
        "timeoutS": 2400,
        "needs": (FVSR_REPO,),
        "model": f"{FVSR_REPO} (tiny decoder, sparse ratio 2.0, local range 11)",
        "licence": "Apache-2.0 (weights and code; Block-Sparse-Attention Apache-2.0)",
    },
}

#: FlashVSR's environment (its requirements.txt; torch 2.6 cu124), Block-Sparse-Attention built
#: from source for the A100 (sm_80) only and forward only -- the backward kernels are dropped
#: from the build and their launcher stubbed, since inference never calls them -- then
#: FlashVSR's own `diffsynth` at `FVSR_COMMIT`.
vsr_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("packaging", "ninja", "psutil", "wheel", "setuptools<80")
    .run_commands(
        f"git clone {BSA_CODE} /opt/bsa && git -C /opt/bsa checkout {BSA_COMMIT}"
        " && git -C /opt/bsa submodule update --init csrc/cutlass",
        "sed -i '/flash_bwd_block_hdim/d' /opt/bsa/setup.py",
        "sed -i 's/run_mha_bwd_block_<elem_type, kHeadDim, Is_causal>(params, stream);"
        '/TORCH_CHECK(false, "block-sparse backward not built");/\''
        " /opt/bsa/csrc/block_sparse_attn/flash_api.cpp",
        "cd /opt/bsa && BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE"
        " TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 NVCC_THREADS=2"
        " pip install --no-build-isolation -v . 2>&1 | tail -40",
        # Installed (not imported: the builder has no GPU driver).
        'python -c "import importlib.util as u, sys;'
        " sys.exit(u.find_spec('block_sparse_attn_cuda') is None)\"",
    )
    .pip_install(
        "torchmetrics==1.7.3",
        "torchsde==0.2.6",
        "accelerate==1.8.1",
        "einops==0.8.1",
        "huggingface-hub==0.34.4",
        "matplotlib==3.10.3",
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "peft==0.16.0",
        "pillow==11.0.0",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "transformers==4.46.2",
        "pytorch-lightning==2.5.2",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "protobuf==3.20.3",
        "ftfy==6.3.1",
        "pandas==2.3.0",
        "tqdm",
        "datasets",
    )
    .run_commands(
        f"git clone {FVSR_CODE} /opt/flashvsr && git -C /opt/flashvsr checkout {FVSR_COMMIT}",
        "pip install --no-deps -e /opt/flashvsr",
    )
)


# --- upscaler: FlashVSR v1.1 -------------------------------------------------------------------


@app.function(
    image=vsr_image,
    gpu=UPSCALERS["flashvsr"]["gpu"],
    cpu=UPSCALERS["flashvsr"]["cpu"],
    memory=UPSCALERS["flashvsr"]["memoryGiB"] * 1024,
    timeout=UPSCALERS["flashvsr"]["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    single_use_containers=True,
)
def flashvsr(request: dict) -> dict:
    """Every clip in the results volume (`/data/clips/<arm>/<start>.mp4`, or `request["clips"]`)
    through FlashVSR v1.1 (tiny), as its own v1.1 tiny script runs it, to the render's size.
    The clip is padded (reflected) so the upscaled size is a multiple of 128, its frame count
    padded to FlashVSR's 8n+1 with the last frame repeated, and both trimmed back after."""
    started = time.time()
    weights = Path(f"/lv/{FVSR_REPO}")
    _need([str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    RESULTS.reload()
    names = request.get("clips") or sorted(
        str(p.relative_to("/data/clips").with_suffix(""))
        for p in Path("/data/clips").glob("*/*.mp4")
    )
    code = Path("/opt/flashvsr/examples/WanVSR")
    link = code / "FlashVSR-v1.1"
    if not link.exists():
        link.symlink_to(weights)
    os.chdir(code)
    sys.path.insert(0, str(code))
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffsynth import FlashVSRTinyPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([str(link / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    proj.load_state_dict(torch.load(link / "LQ_proj_in.ckpt", map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in = proj
    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
    )
    pipe.TCDecoder.load_state_dict(torch.load(link / "TCDecoder.ckpt"), strict=False)
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    load = time.time() - started
    width, height = request.get("renderSize", START_SIZE)
    out: dict = {}
    for name in names:
        reader = imageio.get_reader(f"/data/clips/{name}.mp4")
        fps = float(reader.get_meta_data().get("fps", 24.0))
        frames = np.stack([np.asarray(f)[..., :3] for f in reader])
        reader.close()
        n, h, w = frames.shape[:3]
        scale = 4 if w < width else 2
        unit = 128 // scale
        pad_h, pad_w = (-h) % unit, (-w) % unit
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        padded = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        total = ((n + 3 + 7) // 8) * 8 + 1  # FlashVSR keeps total - 4 frames: at least n
        padded = np.concatenate([padded, np.repeat(padded[-1:], total - n, axis=0)])
        lq = torch.from_numpy(padded).to("cuda").permute(3, 0, 1, 2).float() / 127.5 - 1.0
        H4, W4 = (h + pad_h) * scale, (w + pad_w) * scale
        lq = F.interpolate(lq, size=(H4, W4), mode="bicubic", align_corners=False)
        lq = lq.clamp(-1, 1).to(torch.bfloat16)[None]  # (1, C, F, H, W)
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=0,
            LQ_video=lq,
            num_frames=total,
            height=H4,
            width=W4,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=2.0 * 768 * 1280 / (H4 * W4),
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )
        torch.cuda.synchronize()
        seconds = time.time() - t0
        big = ((video.float() + 1) * 127.5).clamp(0, 255)  # (C, T, H, W)
        big = big[:, :n, : h * scale, : w * scale]
        small = F.interpolate(big.permute(1, 0, 2, 3), size=(height, width), mode="area")
        result = small.round().byte().permute(0, 2, 3, 1).cpu().numpy()
        del lq, video, big, small
        out[name] = {
            "mp4": _mp4(result, fps),
            "fps": fps,
            "frames": int(n),
            "scale": scale,
            "modelSize": [int(w), int(h)],
            "srSize": [int(w * scale), int(h * scale)],
            "seconds": round(seconds, 2),
            "framesPerSecond": round(n / seconds, 2),
            "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
        torch.cuda.empty_cache()
    RESULTS.reload()
    for name, clip in out.items():
        target = Path("/data/upscaled/flashvsr") / f"{name}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "upscaler": "flashvsr",
        "model": UPSCALERS["flashvsr"]["model"],
        "code": f"{FVSR_CODE}@{FVSR_COMMIT}",
        "loadSeconds": round(load, 1),
        "clips": out,
        "containerSeconds": round(time.time() - started, 1),
    }


UPSCALER_FUNCTIONS = {"flashvsr": flashvsr}


@app.function(image=vsr_image, cpu=1.0, memory=4096, timeout=600)
def vsr_check() -> dict:
    """CPU only: FlashVSR's image is built (its kernels compiled) and its packages resolve, so
    the GPU container that upscales does not pay for a failed build or import."""
    import importlib.util

    found = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "block_sparse_attn", "block_sparse_attn_cuda", "diffsynth")
    }
    return {"found": found, "code": f"{FVSR_CODE}@{FVSR_COMMIT}", "bsa": BSA_COMMIT}


# --- helpers ---------------------------------------------------------------------------------


def _hf_token() -> str | None:
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
    return os.environ.get("HF_TOKEN")


def _mp4(frames: object, fps: float) -> bytes:
    """Near-lossless H.264 (CRF 14), as the arms keep their clips."""
    import imageio.v2 as imageio
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio.get_writer(
            f.name, fps=fps, codec="libx264", ffmpeg_params=["-crf", "14"], macro_block_size=1
        )
        for frame in frames:  # type: ignore[attr-defined]
            writer.append_data(np.asarray(frame))
        writer.close()
        return Path(f.name).read_bytes()


def _need(paths: list[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=1.23,<2")
    .env({"HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"})
)


@app.function(
    image=download_image,
    secrets=[HF_SECRET],
    volumes={"/lv": LV_WEIGHTS},
    cpu=DOWNLOAD_RESERVATION["cpu"],
    memory=DOWNLOAD_RESERVATION["memoryGiB"] * 1024,
    timeout=DOWNLOAD_RESERVATION["timeoutS"],
)
def download() -> dict:
    """FlashVSR v1.1's weights into /lv/<repo>/ (CPU only), committed."""
    from huggingface_hub import snapshot_download

    started = time.time()
    local = Path("/lv") / FVSR_REPO
    snapshot_download(
        FVSR_REPO,
        local_dir=str(local),
        allow_patterns=["*.ckpt", "*.pth", "*.safetensors", "*.json", "README.md"],
        token=_hf_token(),
        max_workers=8,
    )
    LV_WEIGHTS.commit()
    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    return {"repo": FVSR_REPO, "bytes": size, "seconds": round(time.time() - started, 1)}


# --- the run --------------------------------------------------------------------------------


@app.local_entrypoint()
def main(steps: str = "check", budget_left: float = 0.0, out: str = "lv-out") -> None:
    """The steps; the upscaled clips under `out/upscaled/flashvsr/<arm>/<start>.mp4` and
    `out/upscale-summary.json` (each call's wall time and estimated dollars)."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in steps.split(",") if s]
    summary: dict = {"steps": wanted, "costs": []}

    def dump() -> None:
        (folder / "upscale-summary.json").write_text(json.dumps(summary, indent=1))

    if str(LOCAL_CAPTURES) not in sys.path:
        sys.path.insert(0, str(LOCAL_CAPTURES))
    from modal_calls import SpawnedCalls

    calls = SpawnedCalls()

    def cost(label: str, reservation: dict, seconds: float, extra: dict | None = None) -> None:
        row = {"call": label, "gpu": reservation["gpu"] or "CPU", "seconds": round(seconds, 1)}
        row |= {"dollars": round(seconds * rate_per_s(reservation), 4)} | (extra or {})
        summary["costs"].append(row)
        sys.stdout.write(f"cost {json.dumps(row)}\n")
        dump()

    def step(label: str, function: modal.Function, reservation: dict, *args: object) -> object:
        t0 = time.time()
        call = calls.spawn(label, function, *args)
        try:
            result = calls.get(call, timeout=reservation["timeoutS"] + 900)
        except Exception as error:  # noqa: BLE001 - reported, the run fails at the end
            summary[label] = {"error": repr(error)[:4000]}
            cost(f"{label} (failed)", reservation, time.time() - t0)
            return None
        cost(label, reservation, time.time() - t0, {"containerSeconds": None})
        return result

    def run() -> None:
        if "download" in wanted:
            summary["download"] = step("download flashvsr", download, DOWNLOAD_RESERVATION)
        if "check" in wanted:
            summary["check"] = step("check flashvsr image", vsr_check, CHECK_RESERVATION)
            dump()
            found = (summary["check"] or {}).get("found", {})
            if not found or not all(found.values()):
                raise RuntimeError(f"FlashVSR's image is not usable: {summary['check']}")
        if "upscale" in wanted:
            reservation = UPSCALERS["flashvsr"]
            worst = reservation["timeoutS"] * rate_per_s(reservation)
            if worst > budget_left:
                summary["skipped"] = f"worst ${worst:.2f} > ${budget_left:.2f} left"
                sys.stdout.write(f"upscale skipped: {summary['skipped']}\n")
                return
            result = step("upscale flashvsr", flashvsr, reservation, {})
            if result is None:
                return
            for name, clip in result["clips"].items():
                target = folder / "upscaled" / "flashvsr" / f"{name}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(clip.pop("mp4"))
            summary["upscaled"] = {"flashvsr": result | {"licence": reservation["licence"]}}
            summary["costs"][-1]["containerSeconds"] = result.get("containerSeconds")

    try:
        with calls.guard():
            run()
        problems = [f"{f['call']}: {f['detail']}" for f in calls.failures()]
        if problems:
            raise RuntimeError("; ".join(problems))
    finally:
        summary["calls"] = calls.report()
        summary["dollars"] = round(sum(r["dollars"] for r in summary["costs"]), 4)
        dump()
        sys.stdout.write(f"estimated dollars this run: {summary['dollars']}\n")
