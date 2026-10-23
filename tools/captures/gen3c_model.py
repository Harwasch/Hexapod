"""GEN3C (NVIDIA, Cosmos-7B conditioned on renders of a 3D cache) as the inferred fill's set
filler: the `gen3c` arm of `anchor_fill`.

GEN3C makes a 121-frame clip at 1280x704. Each frame it is given the 3D cache rendered at
that frame's camera and a mask (1 where the render holds, 0 where it has to make the scene),
and frame 0 is a real image it keeps. Its own scripts build the cache from monocular depth
(MoGe); here the cache is the scan itself: `anchor_fill.build_gen3c_clip` renders the scan
at every frame's camera, held where the kept cameras know it, so the model's whole
generative power goes to the weak and unknown pixels (the paper's own argument, section 3).
Nothing here sees the poses: the renders carry them.

The packing (`pack_request`, `unpack_frames`) runs anywhere with numpy; `load` and `fill`
only in the GEN3C image (`infra/modal/fill.py` `gen3c_image`, the repository
nv-tlabs/GEN3C at `GEN3C_COMMIT` on its path).

Licences: the code Apache-2.0; the weights (nvidia/GEN3C-Cosmos-7B,
nvidia/Cosmos-Tokenize1-CV8x8x8-720p) the NVIDIA Open Model License (commercial use allowed;
"Built on NVIDIA Cosmos" where distributed); google-t5/t5-11b Apache-2.0. The guardrail
models are off, as in every GEN3C inference script NVIDIA ships (`args.disable_guardrail =
True` in their `__main__`); the licence ends a user's rights if a guardrail is bypassed
"without a substantially similar Guardrail appropriate for your use case": the prompt here
is fixed by this code and the inputs are our own scans, which is that guardrail's case to
make before anything made with it ships.
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np

#: The repository and commit the image installs (its `cosmos_predict1` package).
GEN3C_REPO = "https://github.com/nv-tlabs/GEN3C.git"
GEN3C_COMMIT = "db2ffe12ced12ddafcec5e0422ee46ce8520746b"
#: The model's clip: frames and size (width, height).
FRAMES = 121
SIZE = (1280, 704)
#: Its sampler, as NVIDIA's scripts run it (guidance 1, 35 steps).
GUIDANCE = 1.0
STEPS = 35
#: What it reads, under the checkpoint folder, and from each repository only these files
#: (the T5 repository also holds a 45 GB TensorFlow copy).
CHECKPOINT_DIR = "/weights/gen3c"
REPOS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nvidia/GEN3C-Cosmos-7B": ("Gen3C-Cosmos-7B", ("config.json", "model.pt", "README.md")),
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p": (
        "Cosmos-Tokenize1-CV8x8x8-720p",
        ("*.jit", "*.pt", "config.json", "model_config.yaml", "README.md"),
    ),
    "google-t5/t5-11b": (
        "google-t5/t5-11b",
        ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json"),
    ),
}


def pack_request(
    frames: np.ndarray,
    held: np.ndarray,
    prompt: str,
    seed: int,
    steps: int = STEPS,
    guidance: float = GUIDANCE,
) -> dict[str, Any]:
    """The request: `frames` (T, H, W, 3) uint8 (frame 0 the real image), `held` (T, H, W)
    bool (True where the render holds), compressed."""
    if frames.shape[0] != FRAMES or frames.shape[1:3] != (SIZE[1], SIZE[0]):
        raise ValueError(f"frames {frames.shape}: GEN3C takes {FRAMES} x {SIZE[1]} x {SIZE[0]}")
    if held.shape != frames.shape[:3]:
        raise ValueError(f"held {held.shape} against frames {frames.shape}")
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames.astype(np.uint8), held=np.packbits(held, axis=-1))
    return {
        "clip": buf.getvalue(),
        "prompt": prompt,
        "seed": int(seed),
        "steps": int(steps),
        "guidance": float(guidance),
    }


def unpack_request(request: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(request["clip"])) as z:
        frames = z["frames"]
        held = np.unpackbits(z["held"], axis=-1, count=frames.shape[2]).astype(bool)
    return frames, held


def pack_frames(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames.astype(np.uint8))
    return buf.getvalue()


def unpack_frames(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data)) as z:
        return z["frames"]


# --- the GPU side (the GEN3C image) ---------------------------------------------------------------


def load(checkpoint_dir: str = CHECKPOINT_DIR) -> Any:
    """The pipeline, every part on the GPU (T5-11B's encoder in fp32 and the 7B network fit
    an H100's 80 GB together, so nothing is reloaded per call)."""
    import os

    os.chdir("/opt/GEN3C")  # its configs are read by relative path
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline

    return Gen3cPipeline(
        inference_type="video2world",
        checkpoint_dir=checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        enable_prompt_upsampler=False,
        offload_network=False,
        offload_tokenizer=False,
        offload_text_encoder_model=False,
        offload_prompt_upsampler=False,
        offload_guardrail_models=False,
        disable_guardrail=True,
        disable_prompt_encoder=False,
        guidance=GUIDANCE,
        num_steps=STEPS,
        height=SIZE[1],
        width=SIZE[0],
        fps=24,
        num_video_frames=FRAMES,
        seed=0,
    )


def fill(pipe: Any, request: dict[str, Any]) -> dict[str, Any]:
    """One clip: the renders as the single cache buffer (unknown pixels 0, as GEN3C's own
    forward warp leaves them), frame 0 as the kept input frame. Returns the frames made
    (`frames`, packed) and the seconds."""
    import torch

    started = time.time()
    frames, held = unpack_request(request)
    pipe.seed = int(request.get("seed", 0))
    pipe.num_steps = int(request.get("steps", STEPS))
    pipe.guidance = float(request.get("guidance", GUIDANCE))
    x = torch.from_numpy(frames).float().div(127.5).sub(1.0)  # T, H, W, 3 in [-1, 1]
    m = torch.from_numpy(held).float()
    x = x * m[..., None]
    device = "cuda"
    # B, T, N (buffers), C, H, W
    warps = x.permute(0, 3, 1, 2)[None, :, None].to(device)
    masks = m[None, :, None, None].to(device)
    first = x[0].permute(2, 0, 1)[None, :, None].to(device)  # B, C, T=1, H, W
    with torch.no_grad():
        out = pipe.generate(
            prompt=request["prompt"],
            image_path=first,
            rendered_warp_images=warps,
            rendered_warp_masks=masks,
            negative_prompt=None,
        )
    if out is None:
        raise RuntimeError("GEN3C returned nothing (a guardrail would; they are off)")
    video, _ = out
    video = np.asarray(video)
    if video.shape != frames.shape:
        raise RuntimeError(f"GEN3C made {video.shape}, asked {frames.shape}")
    seconds = round(time.time() - started, 1)
    del warps, masks, first
    torch.cuda.empty_cache()
    return {
        "frames": pack_frames(video),
        "seconds": seconds,
        "peakGb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }


def smoke(pipe: Any) -> dict[str, Any]:
    """A synthetic clip (a gradient that pans, a box to make in every frame): that the
    model loads and runs, and how long a clip takes."""
    w, h = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    frames = np.zeros((FRAMES, h, w, 3), np.uint8)
    held = np.ones((FRAMES, h, w), bool)
    for t in range(FRAMES):
        frames[t, ..., 0] = (xx + 4 * t) * 255 // (w + 4 * FRAMES)
        frames[t, ..., 1] = yy * 255 // h
        frames[t, ..., 2] = 128
        if t:
            held[t, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = False
    response = fill(pipe, pack_request(frames, held, "A smooth colour gradient.", 1))
    made = unpack_frames(response["frames"])
    return {
        "seconds": response["seconds"],
        "peakGb": response["peakGb"],
        "shape": list(made.shape),
        "meanAbsHeldDiff": round(float(np.abs(made.astype(float) - frames)[held].mean()), 2),
    }
