"""Living view bake-off, round 2: Matrix-Game 3.0 (Skywork) from our render.

The distilled real-time path (3 steps, no guidance) as the repository's `generate.py` runs it
on one GPU, from our render with the generic prompt (bakeoff/living-view/prompts.md): idle (all
keys up, mouse still, so every camera pose stays at the first) and a slow pan (a constant small
yaw), two iterations each (57 + 40 = 97 frames), every start, one warm H100.

    download  CPU: the distilled DiT, the T5, the VAEs into `hexapod-living-view-weights`.
    run       H100: the rollouts, timed; clips to `/data/clips/mg3-idle|mg3-pan/<start>.mp4`.

Licences: weights Apache-2.0 (model card); code Apache-2.0 (Matrix-Game-3/LICENSE.txt) in an
MIT repository.

    modal run infra/modal/living_mg3.py --steps download,run --budget-left 1.5
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-living-mg3"
app = modal.App(APP_NAME)

LV_WEIGHTS = modal.Volume.from_name(
    "hexapod-living-view-weights", create_if_missing=True, version=2
)
RESULTS = modal.Volume.from_name("hexapod-living-view", create_if_missing=True, version=2)
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

MG3_REPO = "Skywork/Matrix-Game-3.0"
MG3_CODE = "https://github.com/SkyworkAI/Matrix-Game.git"
MG3_COMMIT = "756776d516233027631008761d4d40ddaea3e23a"
MG3_FILES = [
    "base_distilled_model/*",
    "google/umt5-xxl/*",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "MG-LightVAE_v2.pth",
    "model_index.json",
    "README.md",
]
#: The generic prompt (prompts.md): no scene words, no camera words (the actions own the camera).
MG3_PROMPT = (
    "A realistic view in a light breeze: leaves, grass and thin branches sway slightly and "
    "settle, any water ripples gently, while the ground, buildings and objects stay rigid and "
    "still."
)
MG3_ITERATIONS = 2
MG3_STEPS = 3
MG3_FPS = 16.0  # Wan 2.2's sample rate; 97 frames are 6 s
#: Yaw per frame is 15 degrees x mouse_y (deadzone 0.02): 0.03 is 0.45 degrees a frame.
MG3_PAN_MOUSE = (0.0, 0.03)
RUN = {"gpu": "H100", "cpu": 8.0, "memoryGiB": 96, "timeoutS": 900}
DOWNLOAD = {"gpu": "", "cpu": 2.0, "memoryGiB": 8, "timeoutS": 3600}
GPU_PER_S = {"H100": 0.001097, "": 0.0}


def rate_per_s(r: dict) -> float:
    return GPU_PER_S[r["gpu"]] + r["cpu"] * 0.0000131 + r["memoryGiB"] * 0.00000222


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "diffusers>=0.36.0",
        "transformers==4.57.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio[ffmpeg]",
        "easydict",
        "ftfy",
        "imageio-ffmpeg",
        "numpy==2.2.6",
        "scipy",
        "einops",
        "pandas",
        "trimesh",
        "safetensors==0.7.0",
        "huggingface_hub[hf_xet]",
    )
    .run_commands(
        f"git clone --filter=blob:limit=4m {MG3_CODE} /opt/mg && git -C /opt/mg checkout {MG3_COMMIT}"
    )
)


def _hf_token() -> str | None:
    keys = [k for k in HF_TOKEN_KEYS if os.environ.get(k)]
    keys += sorted(k for k, v in os.environ.items() if v.startswith("hf_"))
    if keys:
        os.environ["HF_TOKEN"] = os.environ[keys[0]]
    return os.environ.get("HF_TOKEN")


def _mp4(frames: object, fps: float) -> bytes:
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
    local = Path("/lv") / MG3_REPO
    snapshot_download(
        MG3_REPO, local_dir=str(local), allow_patterns=MG3_FILES, token=_hf_token(), max_workers=8
    )
    LV_WEIGHTS.commit()
    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    return {"repo": MG3_REPO, "bytes": size, "seconds": round(time.time() - t0, 1)}


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
    """Idle and pan from every start. `get_data` (which draws the repository's random
    benchmark actions) is replaced by ours; `process_video` (which draws the controls over the
    frames) by a capture of the clean frames. First motion is the first decoded chunk."""
    started = time.time()
    code = Path("/opt/mg/Matrix-Game-3")
    os.chdir(code)
    sys.path.insert(0, str(code))
    from types import SimpleNamespace

    import numpy as np
    import pipeline.inference_pipeline as ip
    import torch
    import utils.utils as uu
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

    captured: dict = {}
    ip.process_video = lambda video, *a, **k: captured.__setitem__("video", video)
    action = {"mouse": (0.0, 0.0)}

    def get_data(num_frames, height, width, pil_image, device=None, dtype=None):
        img = torch.from_numpy(np.array(pil_image)).unsqueeze(0).permute(0, 3, 1, 2)
        img = uu.get_video_transform(height, width, lambda x: 2.0 * x - 1.0)(img)
        img = img.transpose(0, 1).unsqueeze(0)
        keyboard = torch.zeros((num_frames, 6))
        mouse = torch.zeros((num_frames, 2))
        mouse[:, 0], mouse[:, 1] = action["mouse"]
        poses = uu.compute_all_poses_from_actions(keyboard, mouse, first_pose=np.zeros(5))
        rotations = np.concatenate([np.zeros((len(poses), 1)), poses[:, 3:5]], axis=1)
        extrinsics = uu.get_extrinsics(rotations.tolist(), poses[:, :3].tolist())
        return (
            img.to(device, dtype),
            extrinsics,
            keyboard.to(device, dtype).unsqueeze(0),
            mouse.to(device, dtype).unsqueeze(0),
        )

    ip.get_data = get_data
    ckpt = f"/lv/{MG3_REPO}"
    args = SimpleNamespace(
        size="704*1280",
        ckpt_dir=ckpt,
        num_iterations=MG3_ITERATIONS,
        output_dir="/tmp/mg3-out",
        save_name="clip",
        use_int8=False,
        verify_quant=False,
        compile_vae=False,
        vae_type="mg_lightvae_v2",
        lightvae_pruning_rate=None,
        use_async_vae=False,
        async_vae_warmup_iters=0,
        fa_version=None,
        interactive=False,
        use_base_model=False,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = WAN_CONFIGS["matrix_game3"]
    pipe = ip.MatrixGame3Pipeline(
        config=cfg,
        checkpoint_dir=ckpt,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
        args=args,
        fa_version=None,
        use_base_model=False,
    )
    decode = pipe.vae.stream_decode
    marks: list = []

    def timed_decode(*a, **k):
        out = decode(*a, **k)
        torch.cuda.synchronize()
        marks.append(time.time())
        return out

    pipe.vae.stream_decode = timed_decode
    load = time.time() - started
    RESULTS.reload()
    clips: dict = {}
    for name in request["starts"]:
        img = Image.open(f"/data/starts/{name}.png").convert("RGB")
        for arm, mouse in (("mg3-idle", (0.0, 0.0)), ("mg3-pan", MG3_PAN_MOUSE)):
            action["mouse"] = mouse
            captured.clear()
            marks.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            pipe.generate(
                MG3_PROMPT,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                shift=cfg.sample_shift,
                num_inference_steps=MG3_STEPS,
                guide_scale=1.0,
                seed=42,
                use_base_model=False,
                args=args,
            )
            torch.cuda.synchronize()
            seconds = time.time() - t0
            video = np.asarray(captured["video"])
            clips[f"{arm}/{name}"] = {
                "mp4": _mp4(video, MG3_FPS),
                "fps": MG3_FPS,
                "frames": len(video),
                "size": [int(video.shape[2]), int(video.shape[1])],
                "mouse": list(mouse),
                "seconds": round(seconds, 2),
                "firstFrameSeconds": round(marks[0] - t0, 2) if marks else None,
                "sustainedFps": round((len(video) - 57) / (marks[-1] - marks[0]), 2)
                if len(marks) > 1
                else None,
                "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
    RESULTS.reload()
    for key, clip in clips.items():
        path = Path("/data/clips") / f"{key}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "model": f"{MG3_REPO} (base_distilled_model, mg_lightvae_v2, bf16, SDPA)",
        "code": f"{MG3_CODE}@{MG3_COMMIT}",
        "gpu": torch.cuda.get_device_name(),
        "prompt": MG3_PROMPT,
        "settings": {"steps": MG3_STEPS, "iterations": MG3_ITERATIONS, "shift": cfg.sample_shift},
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
        (folder / "mg3-summary.json").write_text(json.dumps(summary, indent=1))

    def cost(label: str, r: dict, seconds: float) -> None:
        row = {"call": label, "gpu": r["gpu"] or "CPU", "seconds": round(seconds, 1)}
        row["dollars"] = round(seconds * rate_per_s(r), 4)
        summary["costs"].append(row)
        sys.stdout.write(f"cost {json.dumps(row)}\n")
        dump()

    if "download" in wanted:
        t0 = time.time()
        summary["download"] = download.remote()
        cost("download mg3", DOWNLOAD, time.time() - t0)
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
            cost("run mg3 (failed)", RUN, time.time() - t0)
            raise
        for key, clip in result["clips"].items():
            path = folder / "clips" / f"{key}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(clip.pop("mp4"))
        summary["run"] = result
        cost("run mg3", RUN, time.time() - t0)
    summary["dollars"] = round(sum(r["dollars"] for r in summary["costs"]), 4)
    dump()
