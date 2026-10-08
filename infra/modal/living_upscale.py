"""Living view bake-off, the upscalers: every arm's clip to the render's size with FlashVSR
v1.1, and with SeedVR2-3B as a second opinion.

The A-up column of the bake-off (infra/modal/living_bakeoff.py): the model's own pixels, made
1280 x 704 by a video super-resolution model instead of bicubic. A Modal app of its own because
`modal run` builds every image of an app before anything runs: this one's build (it once
compiled Block-Sparse-Attention, 28 minutes) must not hold up the arms. FlashVSR's
locality-constrained sparse attention runs in PyTorch (`block_sparse_attn_func`) on the masks
FlashVSR builds, with the same result as the CUDA kernel. SeedVR2's DiT takes apex's fused
norms; torch's own stand in (`_apex_norms`), so apex is not built.

Steps (`--steps`):

    download         CPU: FlashVSR v1.1's weights into `hexapod-living-view-weights`.
    check            CPU: the image builds and its packages resolve, before a GPU is held.
    upscale          A100-80GB, one warm container: every clip the arms kept in the volume
                     `hexapod-living-view` (`/data/clips/<arm>/<start>.mp4`), each timed.
    seedvr-download  CPU: SeedVR2-3B's weights (DiT, VAE, text embeddings).
    seedvr-check     CPU: its image builds, its code imports, its configs make the models.
    seedvr           H100, one warm container: the same clips through SeedVR2-3B.

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
GPU_PER_S = {"A100-80GB": 0.000694, "H100": 0.001097, "": 0.0}
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


#: Video super-resolution of every arm's clip (A-up on the page): FlashVSR v1.1 (tiny decoder)
#: on an A100, as its authors run it, at the smaller of its scales (2x, 4x) that reaches the
#: render's size (2x for a clip already that size: a supersampling pass), then area-resized to
#: it. Run 10 took 4x for the 848 x 464 clips (0.3 fps against 0.6 at 2x for 1280 x 704: the
#: work goes with the output's pixels); later runs take 2x wherever 2x reaches the render.
#: The timeout is per run: run 10 (12 clips) had 2400 s and used 2257.
FVSR_REPO = "JunhaoZhuang/FlashVSR-v1.1"
FVSR_CODE = "https://github.com/OpenImagingLab/FlashVSR.git"
FVSR_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
UPSCALERS: dict[str, dict] = {
    "flashvsr": {
        "gpu": "A100-80GB",
        "cpu": 8.0,
        "memoryGiB": 64,
        "timeoutS": 1200,
        "needs": (FVSR_REPO,),
        "model": f"{FVSR_REPO} (tiny decoder, sparse ratio 2.0, local range 11)",
        "licence": "Apache-2.0 (weights and code)",
    },
}

#: Block-Sparse-Attention (MIT Han lab, Apache-2.0), FlashVSR's sparse attention kernels, at the
#: commit FlashVSR's README points to. Round 1's build compiled (28 minutes) but its link step
#: called clang++ (the python build's sysconfig names it), which the base lacked: clang and lld
#: are installed now, and it builds (5 minutes, forward kernels only). Its import check needs
#: torch imported first (libc10). Should the build fail, the image builds without it and
#: `flashvsr` runs `block_sparse_attn_func` below (PyTorch, the same masks) and says so.
BSA_CODE = "https://github.com/mit-han-lab/Block-Sparse-Attention.git"
BSA_COMMIT = "49d6c39e4dc0303442cda3bb758b3925d4399c49"
BSA_BUILD = (
    "cd /opt/bsa && BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE"
    " MAX_JOBS=8 NVCC_THREADS=2 pip install --no-build-isolation -v . > /opt/bsa-build.log 2>&1"
    " && cd / && python -c 'import torch, block_sparse_attn' && echo built > /opt/bsa-status"
    " || (echo failed > /opt/bsa-status; tail -60 /opt/bsa-build.log)"
)
#: FlashVSR's environment (its requirements.txt; torch 2.6 cu124) and its own `diffsynth` at
#: `FVSR_COMMIT`, on CUDA's devel image (nvcc for the kernels).
vsr_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "clang", "lld")
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
        # Forward kernels only (FlashVSR never runs backward): half the compile.
        "sed -i '/flash_bwd_block_hdim/d' /opt/bsa/setup.py",
        "sed -i 's/run_mha_bwd_block_<elem_type, kHeadDim, Is_causal>(params, stream);"
        '/TORCH_CHECK(false, "block-sparse backward not built");/\''
        " /opt/bsa/csrc/block_sparse_attn/flash_api.cpp",
        BSA_BUILD,
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
    # diffsynth's downloader imports it (not in FlashVSR's requirements.txt).
    .pip_install("modelscope")
)

#: SeedVR2-3B (A-up's second opinion): one-step diffusion video restoration, as its authors'
#: `projects/inference_seedvr2_3b.py` runs it on one GPU (cfg 1, one step, conditioning noise
#: 0, wavelet colour fix), from the code of their Hugging Face Space at `SVR_SPACE_COMMIT`
#: (their GitHub repository's, plus the Space's demo). The clip goes in at the render's size,
#: bicubic (the model restores at the output size), so nothing is cropped. A second opinion
#: within the budget: clips go in `SVR_ORDER` (one arm across every start before the next)
#: and no clip starts after `request["wallS"]` seconds of the container.
SVR_REPO = "ByteDance-Seed/SeedVR2-3B"
SVR_FILES = ("seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt", "neg_emb.pt")
SVR_SPACE = "https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B"
SVR_SPACE_COMMIT = "1c8f9fbafac52f6fd2f9b42c869ba27262c52c0a"
SVR_SEED = 666
SVR_ORDER = ("ltx", "causal~ctx3", "flf", "causal", "wan")
SVR_WALL_S = 600.0
UPSCALERS["seedvr2"] = {
    "gpu": "H100",
    "cpu": 8.0,
    "memoryGiB": 64,
    "timeoutS": 1200,
    "needs": (SVR_REPO,),
    "model": f"{SVR_REPO} (one step, cfg 1, wavelet colour fix)",
    "licence": "Apache-2.0 (weights and code)",
}
#: Downloaded per repo (SeedVR2's repository also carries apex wheels, not needed).
DOWNLOAD_PATTERNS = {
    FVSR_REPO: ["*.ckpt", "*.pth", "*.safetensors", "*.json", "README.md"],
    SVR_REPO: [*SVR_FILES, "README.md"],
}

#: Flash-attention 2.7.4.post1's prebuilt wheel for torch 2.5 / CUDA 12 / CPython 3.10 (as the
#: Causal Forcing arm's image); SeedVR2's window attention calls `flash_attn_varlen_func`.
FLASH_ATTN_WHEEL = (
    "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
#: SeedVR2's environment: its requirements.txt's modelling packages (diffusers 0.29.1, whose
#: VAE blocks the code imports; the hub before 0.26, which diffusers 0.29 still needs), on
#: torch 2.5.1 so the flash-attention wheel fits; not its training, data or metrics packages.
svr_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "einops==0.7.0",
        "omegaconf==2.3.0",
        "diffusers==0.29.1",
        "huggingface-hub==0.25.2",
        "rotary-embedding-torch==0.5.3",
        "mediapy==1.2.0",
        "numpy==1.26.4",
        "pillow",
        "safetensors",
        "tqdm",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
    )
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone {SVR_SPACE} /opt/seedvr"
        f" && git -C /opt/seedvr checkout {SVR_SPACE_COMMIT}"
    )
    .pip_install(FLASH_ATTN_WHEEL)
)

#: Key blocks gathered per pass of `block_sparse_attn_func`, in bytes (each of K and V).
GATHER_BYTES = 1 << 28


def block_sparse_attn_func(
    q: object,
    k: object,
    v: object,
    cu_seqlens_q: object,
    cu_seqlens_k: object,
    head_mask_type: object,
    streaming_info: object,
    base_blockmask: object,
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    deterministic: bool = False,
    softmax_scale: float | None = None,
    is_causal: bool = False,
    exact_streaming: bool = False,
    return_attn_probs: bool = False,
) -> object:
    """Block-Sparse-Attention's `block_sparse_attn_func` as FlashVSR calls it (one sequence,
    no dropout, not causal), in PyTorch: q, k, v `(L, heads, dim)`; `base_blockmask` `(1, heads,
    query blocks, key blocks)`, blocks of 128 tokens. Each query block attends to exactly the
    key blocks its row selects -- they are gathered, padded to the longest row, and given to
    `scaled_dot_product_attention` with the padding masked -- so the result is the kernel's
    (to rounding), and the work is the selected blocks', not the dense product's."""
    import torch
    import torch.nn.functional as F

    blk = 128
    lq, heads, dim = q.shape  # type: ignore[attr-defined]
    lk = k.shape[0]  # type: ignore[attr-defined]
    mask = base_blockmask[0].to(torch.bool)  # type: ignore[index]
    nq, nk = int(mask.shape[1]), int(mask.shape[2])
    device = q.device  # type: ignore[attr-defined]

    def blocks(x: object, n: int, length: int) -> object:
        x = F.pad(x, (0, 0, 0, 0, 0, n * blk - length))
        return x.view(n, blk, heads, dim).permute(2, 0, 1, 3)  # (heads, n, blk, dim)

    qb, kb, vb = blocks(q, nq, lq), blocks(k, nk, lk), blocks(v, nk, lk)
    key_ok = (torch.arange(nk * blk, device=device) < lk).view(nk, blk)
    counts = mask.sum(-1)  # (heads, nq)
    width = max(1, int(counts.max()))
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)[..., :width]
    chosen = torch.arange(width, device=device) < counts[..., None]  # (heads, nq, width)
    hidx = torch.arange(heads, device=device)[:, None, None]
    step = max(1, GATHER_BYTES // (heads * width * blk * dim * qb.element_size()))
    out = torch.empty_like(qb)
    for s in range(0, nq, step):
        e = min(nq, s + step)
        idx = order[:, s:e]  # (heads, c, width)
        c = e - s
        ks = kb[hidx, idx].reshape(heads, c, width * blk, dim)
        vs = vb[hidx, idx].reshape(heads, c, width * blk, dim)
        ok = (chosen[:, s:e, :, None] & key_ok[idx]).reshape(heads, c, 1, width * blk)
        out[:, s:e] = F.scaled_dot_product_attention(
            qb[:, s:e], ks, vs, attn_mask=ok, scale=softmax_scale
        )
    out = torch.nan_to_num(out)  # a query block that selected nothing reads nothing
    return out.permute(1, 2, 0, 3).reshape(nq * blk, heads, dim)[:lq]


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
    """Every clip in the results volume (`/data/clips/<arm>/<start>.mp4`; only `request["arms"]`
    if given, or exactly `request["clips"]`) through FlashVSR v1.1 (tiny), as its own v1.1 tiny
    script runs it, to the render's size.
    The clip is padded (reflected) so the upscaled size is a multiple of 128, its frame count
    padded to FlashVSR's 8n+1 with the last frame repeated, and both trimmed back after."""
    started = time.time()
    weights = Path(f"/lv/{FVSR_REPO}")
    _need([str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    RESULTS.reload()
    names = _clip_names(request)
    code = Path("/opt/flashvsr/examples/WanVSR")
    link = code / "FlashVSR-v1.1"
    if not link.exists():
        link.symlink_to(weights)
    os.chdir(code)
    sys.path.insert(0, str(code))
    import types

    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn.functional as F

    # FlashVSR's DiT imports `block_sparse_attn_func` from Block-Sparse-Attention: the CUDA
    # kernels when the image built them, else the PyTorch one here.
    attention = "pytorch"
    try:
        import block_sparse_attn  # noqa: F401

        attention = "kernel"
    except Exception:  # noqa: BLE001 - the image's build failed: the PyTorch path
        kernel = types.ModuleType("block_sparse_attn")
        kernel.block_sparse_attn_func = block_sparse_attn_func  # type: ignore[attr-defined]
        sys.modules["block_sparse_attn"] = kernel
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
    target = request.get("outSize")  # kept at this size (letterboxed), not resized back
    label = request.get("label", "flashvsr")
    out: dict = {}
    for name in names:
        reader = imageio.get_reader(f"/data/clips/{name}.mp4")
        fps = float(reader.get_meta_data().get("fps", 24.0))
        frames = np.stack([np.asarray(f)[..., :3] for f in reader])
        reader.close()
        n, h, w = frames.shape[:3]
        if target:
            fw, fh = _fit(w, h, *target)
            scale = next((k for k in (2, 3, 4) if w * k >= fw and h * k >= fh), 4)
        else:
            scale = 2 if 2 * w >= width and 2 * h >= height else 4
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
        if target:
            fit = F.interpolate(big.permute(1, 0, 2, 3), size=(fh, fw), mode="area")
            result = _letterbox(fit.round().byte().permute(0, 2, 3, 1).cpu().numpy(), *target)
            mp4, crf = _deliver(result, fps, *target)
        else:
            small = F.interpolate(big.permute(1, 0, 2, 3), size=(height, width), mode="area")
            result = small.round().byte().permute(0, 2, 3, 1).cpu().numpy()
            mp4, crf = _mp4(result, fps), 14
        del lq, video, big
        out[name] = {
            "mp4": mp4,
            "crf": crf,
            "fps": fps,
            "frames": int(n),
            "scale": scale,
            "modelSize": [int(w), int(h)],
            "srSize": [int(w * scale), int(h * scale)],
            "outSize": list(target) if target else [int(width), int(height)],
            "seconds": round(seconds, 2),
            "framesPerSecond": round(n / seconds, 2),
            "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "attention": attention,
        }
        torch.cuda.empty_cache()
    RESULTS.reload()
    for name, clip in out.items():
        path = Path("/data/upscaled") / label / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "upscaler": "flashvsr",
        "model": UPSCALERS["flashvsr"]["model"],
        "code": f"{FVSR_CODE}@{FVSR_COMMIT}",
        "attention": attention,
        "bsaBuild": _bsa_status(),
        "label": label,
        "loadSeconds": round(load, 1),
        "clips": out,
        "containerSeconds": round(time.time() - started, 1),
    }


# --- upscaler: SeedVR2-3B ----------------------------------------------------------------------


def _apex_norms() -> None:
    """`apex.normalization` as SeedVR2's DiT imports it (`FusedLayerNorm`, `FusedRMSNorm`), from
    torch: the same parameters (so the checkpoint loads strictly) and the same maths, the
    statistics in float32 and the result in the input's dtype, as apex's kernels do."""
    import types

    import torch
    from torch import nn

    class FusedRMSNorm(nn.Module):
        def __init__(
            self, normalized_shape: int, elementwise_affine: bool = True, eps: float = 1e-6
        ) -> None:
            super().__init__()
            self.eps = eps
            shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
            self.weight = nn.Parameter(torch.ones(shape)) if elementwise_affine else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
            if self.weight is not None:
                y = y * self.weight.float()
            return y.to(x.dtype)

    def FusedLayerNorm(  # apex's name
        normalized_shape: int, elementwise_affine: bool = True, eps: float = 1e-5
    ) -> nn.Module:
        return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    apex = types.ModuleType("apex")
    norms = types.ModuleType("apex.normalization")
    norms.FusedRMSNorm = FusedRMSNorm  # type: ignore[attr-defined]
    norms.FusedLayerNorm = FusedLayerNorm  # type: ignore[attr-defined]
    apex.normalization = norms  # type: ignore[attr-defined]
    sys.modules["apex"], sys.modules["apex.normalization"] = apex, norms


def _seedvr_env() -> Path:
    """The Space's code importable from /opt/seedvr, as one rank of one, with apex's norms."""
    code = Path("/opt/seedvr")
    os.chdir(code)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    for key, value in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "12355"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
    ):
        os.environ.setdefault(key, value)
    _apex_norms()
    return code


@app.function(
    image=svr_image,
    gpu=UPSCALERS["seedvr2"]["gpu"],
    cpu=UPSCALERS["seedvr2"]["cpu"],
    memory=UPSCALERS["seedvr2"]["memoryGiB"] * 1024,
    timeout=UPSCALERS["seedvr2"]["timeoutS"],
    volumes={"/lv": LV_WEIGHTS, "/data": RESULTS},
    single_use_containers=True,
)
def seedvr2(request: dict) -> dict:
    """The clips (as `flashvsr` picks them) through SeedVR2-3B, as
    `projects/inference_seedvr2_3b.py` runs it: the clip bicubic to the render's size (a
    multiple of 16), padded to 4n+1 frames with its last, VAE-encoded, one DiT step from noise
    with the clean latent as the condition, decoded, the wavelet colour fix against the input,
    trimmed back. Each clip is timed from its frames on the GPU to the colour-fixed result."""
    started = time.time()
    weights = Path(f"/lv/{SVR_REPO}")
    _need([str(weights / f) for f in SVR_FILES])
    RESULTS.reload()

    def rank(name: str) -> tuple[int, str]:
        arm, _, start = name.partition("/")
        row = arm + ("~" + start.partition("~")[2] if "~" in start else "")
        return (SVR_ORDER.index(row) if row in SVR_ORDER else len(SVR_ORDER), start)

    names = sorted(_clip_names(request), key=rank)
    wall = float(request.get("wallS", SVR_WALL_S))
    skipped: list[str] = []
    code = _seedvr_env()
    link = code / "ckpts"  # main.yaml reads the VAE from ./ckpts/ema_vae.pth
    if not link.exists():
        link.symlink_to(weights)
    import datetime

    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn.functional as F
    from common.config import load_config
    from common.distributed import init_torch
    from common.seed import set_seed
    from einops import rearrange
    from omegaconf import OmegaConf
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    runner = VideoDiffusionInfer(load_config("./configs_3b/main.yaml"))
    OmegaConf.set_readonly(runner.config, False)
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    runner.configure_dit_model(device="cuda", checkpoint=str(weights / "seedvr2_ema_3b.pth"))
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()
    positive = torch.load(weights / "pos_emb.pt").to("cuda")
    negative = torch.load(weights / "neg_emb.pt").to("cuda")
    load = time.time() - started
    width, height = request.get("renderSize", START_SIZE)
    target = request.get("outSize")
    label = request.get("label", "seedvr2")
    out: dict = {}
    for name in names:
        if time.time() - started > wall:
            skipped.append(name)
            continue
        reader = imageio.get_reader(f"/data/clips/{name}.mp4")
        fps = float(reader.get_meta_data().get("fps", 24.0))
        frames = np.stack([np.asarray(f)[..., :3] for f in reader])
        reader.close()
        n, h, w = frames.shape[:3]
        if target:  # restored at the output size (letterboxed), multiples of 16
            width, height = (v // 16 * 16 for v in _fit(w, h, *target))
        set_seed(SVR_SEED, same_across_ranks=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            video = torch.from_numpy(frames).to("cuda").permute(0, 3, 1, 2).float() / 255.0
            video = F.interpolate(video, size=(height, width), mode="bicubic", align_corners=False)
            video = video.clamp(0, 1) * 2 - 1  # (T, C, H, W), -1..1
            cond = rearrange(video, "t c h w -> c t h w")
            pad = (-(n - 1)) % 4
            if pad:
                cond = torch.cat([cond, cond[:, -1:].repeat(1, pad, 1, 1)], dim=1)
            latent = runner.vae_encode([cond])[0]
            noise = torch.randn_like(latent)
            condition = runner.get_condition(noise, task="sr", latent_blur=latent)
            with torch.autocast("cuda", torch.bfloat16, enabled=True):
                sample = runner.inference(
                    noises=[noise],
                    conditions=[condition],
                    texts_pos=[positive],
                    texts_neg=[negative],
                    dit_offload=False,
                )[0]
            sample = rearrange(sample, "c t h w -> t c h w")[:n]
            parts = []
            for k in range(0, n, 8):  # the colour fix in slices: 4K frames are large
                fixed = wavelet_reconstruction(sample[k : k + 8].float(), video[k : k + 8])
                parts.append(((fixed.clamp(-1, 1) + 1) * 127.5).round().byte().cpu())
            torch.cuda.synchronize()
            seconds = time.time() - t0
            result = torch.cat(parts).permute(0, 2, 3, 1).numpy()
        del video, cond, latent, noise, condition, sample
        if target:
            result = _letterbox(result, *target)
            mp4, crf = _deliver(result, fps, *target)
        else:
            mp4, crf = _mp4(result, fps), 14
        out[name] = {
            "mp4": mp4,
            "crf": crf,
            "outSize": list(target) if target else [int(width), int(height)],
            "fps": fps,
            "frames": int(n),
            "scale": round(width / w, 3),
            "modelSize": [int(w), int(h)],
            "srSize": [int(width), int(height)],
            "seconds": round(seconds, 2),
            "framesPerSecond": round(n / seconds, 2),
            "peakMemoryGB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
        torch.cuda.empty_cache()
    RESULTS.reload()
    for name, clip in out.items():
        path = Path("/data/upscaled") / label / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(clip["mp4"])
    RESULTS.commit()
    return {
        "upscaler": "seedvr2",
        "label": label,
        "model": UPSCALERS["seedvr2"]["model"],
        "code": f"{SVR_SPACE}@{SVR_SPACE_COMMIT}",
        "loadSeconds": round(load, 1),
        "wallS": wall,
        "skipped": skipped,
        "clips": out,
        "containerSeconds": round(time.time() - started, 1),
    }


@app.function(image=svr_image, cpu=2.0, memory=8192, timeout=900)
def svr_check() -> dict:
    """CPU only: SeedVR2's image builds, the flash-attention wheel is installed (its CUDA module
    may not load without a GPU), the inference code imports with torch's norms for apex's, and
    its configs make the VAE and (on the meta device) the DiT, so the H100 does not pay for a
    failed build, import or config."""
    import importlib.metadata
    import traceback
    import types

    found: dict = {}
    _seedvr_env()
    try:
        found["flashAttnWheel"] = importlib.metadata.version("flash_attn").startswith("2.7.4")
    except importlib.metadata.PackageNotFoundError:
        found["flashAttnWheel"] = False
    try:
        import flash_attn  # noqa: F401

        found["flashAttnImport"] = "ok"
    except Exception:  # noqa: BLE001 - a CPU container may lack the CUDA driver
        found["flashAttnImport"] = traceback.format_exc()[-600:]
        stub = types.ModuleType("flash_attn")
        stub.flash_attn_varlen_func = None  # type: ignore[attr-defined]
        sys.modules["flash_attn"] = stub
    for name, statement in (
        ("infer", "from projects.video_diffusion_sr.infer import VideoDiffusionInfer"),
        ("colorfix", "from projects.video_diffusion_sr.color_fix import wavelet_reconstruction"),
        ("distributed", "from common.distributed import init_torch"),
        (
            "models",
            (
                "import torch\n"
                "from common.config import create_object, load_config\n"
                "config = load_config('./configs_3b/main.yaml')\n"
                "with torch.device('meta'):\n"
                "    create_object(config.dit.model)\n"
                "    create_object(config.vae.model)\n"
            ),
        ),
    ):
        try:
            exec(statement, {})  # noqa: S102 - fixed statements
            found[name] = True
        except Exception:  # noqa: BLE001 - reported
            found[name] = False
            found[f"{name}Error"] = traceback.format_exc()[-1500:]
    return {"found": {k: v for k, v in found.items() if isinstance(v, bool)}, "detail": found}


UPSCALER_FUNCTIONS = {"flashvsr": flashvsr, "seedvr2": seedvr2}


@app.function(image=vsr_image, cpu=1.0, memory=4096, timeout=600)
def vsr_check() -> dict:
    """CPU only: FlashVSR's image builds and its code imports (with the PyTorch sparse
    attention standing in), so the GPU container that upscales does not pay for a failed
    build or import."""
    import traceback
    import types

    found: dict = {}
    code = Path("/opt/flashvsr/examples/WanVSR")
    os.chdir(code)
    sys.path.insert(0, str(code))
    kernel = types.ModuleType("block_sparse_attn")
    kernel.block_sparse_attn_func = block_sparse_attn_func  # type: ignore[attr-defined]
    sys.modules["block_sparse_attn"] = kernel
    for name, statement in (
        ("diffsynth", "from diffsynth import FlashVSRTinyPipeline, ModelManager"),
        ("tcdecoder", "from utils.TCDecoder import build_tcdecoder"),
        ("lqproj", "from utils.utils import Causal_LQ4x_Proj"),
    ):
        try:
            exec(statement, {})  # noqa: S102 - fixed import statements
            found[name] = True
        except Exception:  # noqa: BLE001 - reported
            found[name] = False
            found[f"{name}Error"] = traceback.format_exc()[-1500:]
    found["prompt"] = (code / "prompt_tensor" / "posi_prompt.pth").exists()
    found["bsaBuild"] = _bsa_status()  # "built" or "failed": reported, not a gate
    return {"found": {k: v for k, v in found.items() if isinstance(v, bool)}, "detail": found}


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


def _fit(w: int, h: int, width: int, height: int) -> tuple[int, int]:
    """The size a w x h clip takes inside width x height, its aspect kept (even sides)."""
    k = min(width / w, height / h)
    return min(width, round(w * k / 2) * 2), min(height, round(h * k / 2) * 2)


def _letterbox(frames: object, width: int, height: int) -> object:
    """Frames (T, h, w, 3) centred on black at width x height (unchanged when they fill it)."""
    import numpy as np

    t, h, w = frames.shape[:3]  # type: ignore[attr-defined]
    if (w, h) == (width, height):
        return frames
    out = np.zeros((t, height, width, 3), np.uint8)
    top, left = (height - h) // 2, (width - w) // 2
    out[:, top : top + h, left : left + w] = frames
    return out


#: Delivery limits for the page's clips: 2560x1408 at most 8 MB, 3840x2112 at most 14 MB.
def _deliver(frames: object, fps: float, width: int, height: int) -> tuple[bytes, int]:
    """H.264 High, yuv420p, +faststart, from CRF 18 up until the clip fits its limit."""
    import imageio.v2 as imageio
    import numpy as np

    limit = (8 if width * height <= 2560 * 1408 else 14) * 1024 * 1024
    data, crf = b"", 18
    for crf in (18, 20, 22, 24, 26, 28, 30):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            writer = imageio.get_writer(
                f.name,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=[
                    "-crf", str(crf), "-preset", "slow", "-profile:v", "high",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                ],
            )  # fmt: skip
            for frame in frames:  # type: ignore[attr-defined]
                writer.append_data(np.asarray(frame))
            writer.close()
            data = Path(f.name).read_bytes()
        if len(data) <= limit:
            break
    return data, crf


def _bsa_status() -> str:
    path = Path("/opt/bsa-status")
    return path.read_text().strip() if path.exists() else "not in image"


def _clip_names(request: dict) -> list[str]:
    """`request["clips"]`, or every `<arm>/<start>` in /data/clips (of `request["arms"]` only,
    when given)."""
    return request.get("clips") or sorted(
        str(p.relative_to("/data/clips").with_suffix(""))
        for p in Path("/data/clips").glob("*/*.mp4")
        if not request.get("arms") or p.parent.name in request["arms"]
    )


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
def download(repo: str = FVSR_REPO) -> dict:
    """An upscaler's weights (FlashVSR v1.1's, SeedVR2-3B's) into /lv/<repo>/ (CPU only),
    committed."""
    from huggingface_hub import snapshot_download

    started = time.time()
    local = Path("/lv") / repo
    snapshot_download(
        repo,
        local_dir=str(local),
        allow_patterns=DOWNLOAD_PATTERNS[repo],
        token=_hf_token(),
        max_workers=8,
    )
    LV_WEIGHTS.commit()
    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    return {"repo": repo, "bytes": size, "seconds": round(time.time() - started, 1)}


# --- the run --------------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    steps: str = "check",
    budget_left: float = 0.0,
    out: str = "lv-out",
    arms: str = "",
    sv_arms: str = "",
    sv_wall_s: float = SVR_WALL_S,
    vsr_size: str = "",
    sv_size: str = "",
) -> None:
    """The steps (upscale: the clips of `arms`, comma-separated, or of every arm; seedvr: of
    `sv_arms`, within `sv_wall_s` seconds of its container); the
    upscaled clips under `out/upscaled/<upscaler>/<arm>/<start>.mp4` and
    `out/upscale-summary.json` (each call's wall time and estimated dollars). Each upscaler runs
    only if its worst case (timeout x rate) fits in what `budget_left` still holds."""
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

    def sized(size: str, upscaler: str) -> dict:
        if not size:
            return {}
        width, height = (int(v) for v in size.lower().split("x"))
        return {"outSize": [width, height], "label": f"{upscaler}-{width}x{height}"}

    requests = {
        "flashvsr": ({"arms": [a for a in arms.split(",") if a]} if arms else {})
        | sized(vsr_size, "flashvsr"),
        "seedvr2": ({"arms": [a for a in sv_arms.split(",") if a]} if sv_arms else {})
        | {"wallS": sv_wall_s}
        | sized(sv_size, "seedvr2"),
    }

    def checked(label: str, function: modal.Function) -> None:
        summary[label] = step(label, function, CHECK_RESERVATION)
        dump()
        found = (summary[label] or {}).get("found", {})
        if not found or not all(found.values()):
            raise RuntimeError(f"{label}: the image is not usable: {summary[label]}")

    def upscale(upscaler: str) -> None:
        reservation = UPSCALERS[upscaler]
        worst = reservation["timeoutS"] * rate_per_s(reservation)
        left = budget_left - sum(r["dollars"] for r in summary["costs"])
        if worst > left:
            summary.setdefault("skipped", {})[upscaler] = f"worst ${worst:.2f} > ${left:.2f} left"
            sys.stdout.write(f"{upscaler} skipped: worst ${worst:.2f} > ${left:.2f} left\n")
            return
        function = UPSCALER_FUNCTIONS[upscaler]
        result = step(f"upscale {upscaler}", function, reservation, requests[upscaler])
        if result is None:
            return
        label = result.get("label", upscaler)
        for name, clip in result["clips"].items():
            target = folder / "upscaled" / label / f"{name}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(clip.pop("mp4"))
        summary.setdefault("upscaled", {})[label] = result | {"licence": reservation["licence"]}
        summary["costs"][-1]["containerSeconds"] = result.get("containerSeconds")
        dump()

    def run() -> None:
        if "download" in wanted:
            summary["download"] = step("download flashvsr", download, DOWNLOAD_RESERVATION)
        if "check" in wanted:
            checked("check flashvsr image", vsr_check)
        if "upscale" in wanted:
            upscale("flashvsr")
        if "seedvr-download" in wanted:
            summary["seedvr-download"] = step(
                "download seedvr2", download, DOWNLOAD_RESERVATION, SVR_REPO
            )
        if "seedvr-check" in wanted:
            checked("check seedvr2 image", svr_check)
        if "seedvr" in wanted:
            upscale("seedvr2")

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
