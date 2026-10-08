"""Living view bake-off, round 2: Waypoint-1.5 (Overworld) from our render.

A real-time world model on our render: about 4 s idle (no buttons, no mouse) and about 4 s of a
slow pan (a small constant mouse x velocity), every start, one warm H100. Waypoint takes no
text (`prompt_conditioning: null`): the starting image and the controls are all it gets.

    download  CPU: the checkpoint into the HF cache on `hexapod-living-view-weights`.
    run       H100: the rollouts, timed (first frame, sustained frames per second), saved to the
              results volume as `/data/clips/waypoint-idle|waypoint-pan/<start>.mp4`.

Licences: the weights are Apache-2.0; the Python that runs them (the repository's
`modular_blocks.py`, `transformer/model.py`, `vae/ae_model.py`) is GPL-3.0.

    modal run infra/modal/living_waypoint.py --steps download,run --budget-left 1.5
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "hexapod-living-waypoint"
app = modal.App(APP_NAME)

if modal.is_local():
    LOCAL_CAPTURES = Path(__file__).resolve().parents[2] / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path("/root/captures")

LV_WEIGHTS = modal.Volume.from_name(
    "hexapod-living-view-weights", create_if_missing=True, version=2
)
RESULTS = modal.Volume.from_name("hexapod-living-view", create_if_missing=True, version=2)
HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")

WP_REPO = "Overworld/Waypoint-1.5-1B"
WP_REVISION = "391f92827075edcf4a8b3c8a2ddae010698f8636"
#: Each pipeline call adds one latent frame: 4 video frames at the model's 60 fps.
WP_FPS = 60.0
WP_CALLS = 60  # 240 frames, 4 s
#: The pan: a small constant mouse x velocity (the card gives no scale; this one is small).
WP_PAN_MOUSE = (0.1, 0.0)
WP_WARMUP_CALLS = 8
RUN = {"gpu": "H100", "cpu": 8.0, "memoryGiB": 64, "timeoutS": 900}
DOWNLOAD = {"gpu": "", "cpu": 2.0, "memoryGiB": 8, "timeoutS": 3600}
GPU_PER_S = {"H100": 0.001097, "": 0.0}


def rate_per_s(r: dict) -> float:
    return GPU_PER_S[r["gpu"]] + r["cpu"] * 0.0000131 + r["memoryGiB"] * 0.00000222


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.40.0",
        "transformers>=5,<6",
        "accelerate>=1.6",
        "tensordict",
        "einops",
        "regex",
        "ftfy",
        "sentencepiece",
        "imageio[ffmpeg]>=2.37",
        "pillow",
        "numpy",
        "huggingface_hub[hf_xet]",
    )
    .env({"HF_HOME": "/lv/hf"})
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
    """The checkpoint (no demo media) into the HF cache, committed."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(
        WP_REPO, revision=WP_REVISION, ignore_patterns=["assets/*"], token=_hf_token()
    )
    LV_WEIGHTS.commit()
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return {"repo": WP_REPO, "path": path, "bytes": size, "seconds": round(time.time() - t0, 1)}


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
    """Idle and pan rollouts from every start. The render is centre-cropped to the model's 2:1
    and resized to its 1024x512. Compiled as the card runs it (`apply_inference_patches`,
    `torch.compile` max-autotune) after a warm-up rollout; first frame and sustained frames per
    second are timed warm."""
    # Not offline: diffusers' remote-code loader asks the Hub for the commit (`model_info`)
    # even when every file is in the cache on the volume.
    started = time.time()
    import numpy as np
    import torch
    from diffusers.modular_pipelines import ModularPipeline
    from PIL import Image

    pipe = ModularPipeline.from_pretrained(WP_REPO, revision=WP_REVISION, trust_remote_code=True)
    pipe.load_components(
        names=["transformer", "vae"],
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    compiled = "eager"
    try:
        pipe.transformer.apply_inference_patches()
        pipe.transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)
        compiled = "max-autotune"
    except Exception as error:  # noqa: BLE001 - eager instead, said so
        compiled = f"eager ({type(error).__name__})"
    # No text conditioning in this checkpoint: an empty embedding skips the text encoder.
    empty = torch.zeros((1, 1, 2048), device="cuda", dtype=torch.bfloat16)
    RESULTS.reload()

    def start_image(name: str) -> object:
        img = Image.open(f"/data/starts/{name}.png").convert("RGB")
        w, h = img.size
        crop_h = w // 2  # 2:1
        top = (h - crop_h) // 2 if crop_h <= h else 0
        img = img.crop((0, top, w, top + min(h, crop_h)))
        return img.resize((1024, 512), Image.LANCZOS)

    def rollout(img: object, mouse: tuple[float, float], calls: int) -> dict:
        times, frames = [], []
        state = None
        for k in range(calls):
            torch.cuda.synchronize()
            t0 = time.time()
            if state is None:
                state = pipe(
                    image=img,
                    prompt_embeds=empty,
                    button=set(),
                    mouse=mouse,
                    output_type="np",
                )
                state.values["image"] = None
            else:
                state = pipe(state, button=set(), mouse=mouse, output_type="np")
            out = np.asarray(state.values["images"])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            frames.append(out if out.ndim == 4 else out[None])
        video = np.concatenate(frames)
        steady = sum(times[1:])
        return {
            "frames": video,
            "firstFrameSeconds": round(times[0], 3),
            "sustainedFps": round((len(video) - len(frames[0])) / steady, 1) if steady else None,
            "callSeconds": [round(t, 4) for t in times[:6]],
        }

    names = list(request["starts"])
    warm = start_image(names[0])
    t0 = time.time()
    try:
        rollout(warm, (0.0, 0.0), WP_WARMUP_CALLS)  # compiles (lazily, on the first call)
    except Exception as error:  # noqa: BLE001 - eager instead, said so
        compiled = f"eager (compile failed: {type(error).__name__}: {str(error)[:200]})"
        pipe.transformer._compiled_call_impl = None
        rollout(warm, (0.0, 0.0), WP_WARMUP_CALLS)
    warmup = time.time() - t0
    load = time.time() - started
    clips: dict = {}
    for name in names:
        img = start_image(name)
        for arm, mouse in (("waypoint-idle", (0.0, 0.0)), ("waypoint-pan", WP_PAN_MOUSE)):
            torch.cuda.reset_peak_memory_stats()
            r = rollout(img, mouse, WP_CALLS)
            video = r.pop("frames")
            clips[f"{arm}/{name}"] = {
                "mp4": _mp4(video, WP_FPS),
                "fps": WP_FPS,
                "frames": len(video),
                "size": [int(video.shape[2]), int(video.shape[1])],
                "mouse": list(mouse),
                "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                **r,
            }
    RESULTS.reload()
    for key, clip in clips.items():
        path = Path("/data/clips") / f"{key}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "model": f"{WP_REPO}@{WP_REVISION}",
        "gpu": torch.cuda.get_device_name(),
        "compiled": compiled,
        "warmupSeconds": round(warmup, 1),
        "loadSeconds": round(load, 1),
        "text": "none: prompt_conditioning is null in this checkpoint",
        "input": "our render centre-cropped to 2:1, resized to 1024x512",
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
    """The steps; clips under `out/clips/<arm>/<start>.mp4`, `out/waypoint-summary.json`."""
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in steps.split(",") if s]
    summary: dict = {"steps": wanted, "costs": []}

    def dump() -> None:
        (folder / "waypoint-summary.json").write_text(json.dumps(summary, indent=1))

    def cost(label: str, r: dict, seconds: float) -> None:
        row = {"call": label, "gpu": r["gpu"] or "CPU", "seconds": round(seconds, 1)}
        row["dollars"] = round(seconds * rate_per_s(r), 4)
        summary["costs"].append(row)
        sys.stdout.write(f"cost {json.dumps(row)}\n")
        dump()

    if "download" in wanted:
        t0 = time.time()
        summary["download"] = download.remote()
        cost("download waypoint", DOWNLOAD, time.time() - t0)
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
            cost("run waypoint (failed)", RUN, time.time() - t0)
            raise
        for key, clip in result["clips"].items():
            path = folder / "clips" / f"{key}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(clip.pop("mp4"))
        summary["run"] = result
        cost("run waypoint", RUN, time.time() - t0)
    summary["dollars"] = round(sum(r["dollars"] for r in summary["costs"]), 4)
    dump()
