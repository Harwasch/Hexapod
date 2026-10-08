"""Living view bake-off, round 2: Yume 1.5 (Yume-5B-720P) from our render.

Yume's camera is text: each 2-second segment is captioned "First-person perspective." plus a
movement and a rotation phrase from a fixed vocabulary, plus an event text. Idle leaves both
phrases out (as the repository's single-GPU web app does for "None" and "·"); the pan adds
"The camera pans to the right (→)." The event text is the generic one (prompts.md). Two
segments of 32 frames (4 s at 16 fps), every start, one warm H100, driven through the
repository's own `webapp_single_gpu.long_generate` (its single-GPU 5B path).

    download  CPU: the checkpoint into `hexapod-living-view-weights`.
    run       H100: the rollouts, timed; clips to `/data/clips/yume-idle|yume-pan/<start>.mp4`.

Licence: Apache-2.0 (weights and code).

    modal run infra/modal/living_yume.py --steps download,run --budget-left 1.5
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-living-yume"
app = modal.App(APP_NAME)

LV_WEIGHTS = modal.Volume.from_name(
    "hexapod-living-view-weights", create_if_missing=True, version=2
)
RESULTS = modal.Volume.from_name("hexapod-living-view", create_if_missing=True, version=2)
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

YUME_REPO = "stdstu123/Yume-5B-720P"
YUME_CODE = "https://github.com/stdstu12/YUME.git"
YUME_COMMIT = "111c3fab7fb020d1e261a68be6ec78a3fecc8d5b"
YUME_FILES = [
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/*",
    "README.md",
]
#: The generic event text (prompts.md): the coordinator's sentence without its camera
#: sentence, which Yume's camera clause already sets.
YUME_EVENT = (
    "A gentle breeze: leaves, grass and thin branches sway slightly and settle. "
    "Nothing else changes."
)
#: Its 5B script's 4 Euler steps (scripts/inference/sample_5b.sh: --num_euler_timesteps 4);
#: the web app's shift.
YUME_STEPS = 4
YUME_SHIFT = 5.0
YUME_FPS = 16
YUME_SEGMENTS = 2
YUME_FRAME_ZERO = 32
RUN = {"gpu": "H100", "cpu": 8.0, "memoryGiB": 96, "timeoutS": 1200}
DOWNLOAD = {"gpu": "", "cpu": 2.0, "memoryGiB": 8, "timeoutS": 3600}
GPU_PER_S = {"H100": 0.001097, "": 0.0}
FLASH_ATTN_WHEEL = (
    "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)


def rate_per_s(r: dict) -> float:
    return GPU_PER_S[r["gpu"]] + r["cpu"] * 0.0000131 + r["memoryGiB"] * 0.00000222


#: Its requirements.txt's inference packages (diffusers 0.32 for `export_to_video`, flash-attn
#: 2, which its Wan attention asserts), on torch 2.5.1 so the flash-attention wheel fits.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "diffusers==0.32.0",
        "transformers==4.46.3",
        "tokenizers",
        "accelerate==1.0.1",
        "huggingface-hub==0.26.1",
        "peft==0.13.2",
        "safetensors",
        "sentencepiece",
        "easydict==1.13",
        "einops==0.8.0",
        "ftfy==6.3.0",
        "flask",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "av==13.1.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        "tqdm",
        "regex",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .run_commands(
        f"git clone --filter=blob:limit=2m {YUME_CODE} /opt/yume"
        f" && git -C /opt/yume checkout {YUME_COMMIT}"
    )
)


def _hf_token() -> str | None:
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
    return os.environ.get("HF_TOKEN")


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={"/lv": LV_WEIGHTS},
    cpu=DOWNLOAD["cpu"],
    memory=DOWNLOAD["memoryGiB"] * 1024,
    timeout=DOWNLOAD["timeoutS"],
)
def download() -> dict:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    local = Path("/lv") / YUME_REPO
    snapshot_download(
        YUME_REPO, local_dir=str(local), allow_patterns=YUME_FILES, token=_hf_token(), max_workers=8
    )
    LV_WEIGHTS.commit()
    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    return {"repo": YUME_REPO, "bytes": size, "seconds": round(time.time() - t0, 1)}


@app.function(
    image=image,
    gpu=RUN["gpu"],
    cpu=RUN["cpu"],
    memory=RUN["memoryGiB"] * 1024,
    timeout=RUN["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    single_use_containers=True,
)
def run(request: dict) -> dict:
    """Idle and pan from every start through the web app's `long_generate` (I2V, 704x1280,
    two segments). First motion is the first segment decoded."""
    started = time.time()
    code = Path("/opt/yume")
    os.chdir(code)
    sys.path.insert(0, str(code))
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import webapp_single_gpu as web

    web.CKPT_DIR = f"/lv/{YUME_REPO}"
    web.load_wan()
    decode = web.MODELS.vae.decode
    marks: list = []

    def timed_decode(*a, **k):
        out = decode(*a, **k)
        torch.cuda.synchronize()
        marks.append(time.time())
        return out

    web.MODELS.vae.decode = timed_decode
    load = time.time() - started
    RESULTS.reload()
    clips: dict = {}
    for name in request["starts"]:
        for arm, turn in (("yume-idle", "·"), ("yume-pan", "→")):
            marks.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            out_path, final_prompt = web.long_generate(
                web.LongGenArgs(
                    prompt=YUME_EVENT,
                    jpg_path=f"/data/starts/{name}.png",
                    output_dir="/tmp/yume-out",
                    fps=YUME_FPS,
                    sample_steps=YUME_STEPS,
                    sample_num=YUME_SEGMENTS,
                    frame_zero=YUME_FRAME_ZERO,
                    shift=YUME_SHIFT,
                    seed=42,
                    continue_from_last=False,
                    refine_from_image=False,
                    caption_path=None,
                    mode="I2V",
                    resolution="704x1280",
                    memory_optimization=False,
                    vae_memory_optimization=False,
                    camera_movement1="None",
                    camera_movement2=turn,
                )
            )
            torch.cuda.synchronize()
            seconds = time.time() - t0
            data = Path(out_path).read_bytes()
            reader = imageio.get_reader(out_path)
            video = np.stack([np.asarray(f)[..., :3] for f in reader])
            reader.close()
            clips[f"{arm}/{name}"] = {
                "mp4": data,
                "fps": float(YUME_FPS),
                "frames": len(video),
                "size": [int(video.shape[2]), int(video.shape[1])],
                "caption": final_prompt,
                "seconds": round(seconds, 2),
                "firstFrameSeconds": round(marks[0] - t0, 2) if marks else None,
                "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
    RESULTS.reload()
    for key, clip in clips.items():
        path = Path("/data/clips") / f"{key}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "model": f"{YUME_REPO} (bf16, flash-attn 2)",
        "code": f"{YUME_CODE}@{YUME_COMMIT} (webapp_single_gpu.long_generate)",
        "gpu": torch.cuda.get_device_name(),
        "event": YUME_EVENT,
        "settings": {
            "steps": YUME_STEPS,
            "shift": YUME_SHIFT,
            "segments": YUME_SEGMENTS,
            "frameZero": YUME_FRAME_ZERO,
        },
        "loadSeconds": round(load, 1),
        "clips": clips,
        "containerSeconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main(
    steps: str = "download",
    budget_left: float = 0.0,
    starts_names: str = "tree-1,tree-2,camp-1,camp-2",
    out: str = "lv-out",
) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in steps.split(",") if s]
    summary: dict = {"steps": wanted, "costs": []}

    def dump() -> None:
        (folder / "yume-summary.json").write_text(json.dumps(summary, indent=1))

    def cost(label: str, r: dict, seconds: float) -> None:
        row = {"call": label, "gpu": r["gpu"] or "CPU", "seconds": round(seconds, 1)}
        row["dollars"] = round(seconds * rate_per_s(r), 4)
        summary["costs"].append(row)
        sys.stdout.write(f"cost {json.dumps(row)}\n")
        dump()

    if "download" in wanted:
        t0 = time.time()
        summary["download"] = download.remote()
        cost("download yume", DOWNLOAD, time.time() - t0)
    if "run" in wanted:
        worst = RUN["timeoutS"] * rate_per_s(RUN)
        if worst > budget_left:
            summary["skipped"] = f"worst ${worst:.2f} > ${budget_left:.2f}"
            sys.stdout.write(f"run skipped: {summary['skipped']}\n")
            dump()
            return
        t0 = time.time()
        try:
            result = run.remote({"starts": [n for n in starts_names.split(",") if n]})
        except Exception as error:
            summary["run"] = {"error": repr(error)[:4000]}
            cost("run yume (failed)", RUN, time.time() - t0)
            raise
        for key, clip in result["clips"].items():
            path = folder / "clips" / f"{key}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(clip.pop("mp4"))
        summary["run"] = result
        cost("run yume", RUN, time.time() - t0)
    summary["dollars"] = round(sum(r["dollars"] for r in summary["costs"]), 4)
    dump()
