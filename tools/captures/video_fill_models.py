"""Video models that fill the unknown pixels of a rendered clip: the server side of
`generative_fill.RemoteClipFiller`, run on a Modal GPU by `infra/modal/fill.py` (`FillVace`,
`FillWan22`, `FillCosmos`), which run *this* module.

A clip is a camera path rendered from the measured splat: frame 0 at a real camera (its
photo, where the scan has one), each later frame a step towards what the scan never saw.
Every pixel is either **known** (the scan saw it from about there) or **to generate** (the
mask). A model is given the known pixels and asked only for the rest, so its frames agree
with the measurement where there is one and the video prior keeps the generated part
consistent from frame to frame -- the shape of what was missed comes out of those views,
not out of a rule.

| key      | weights                                            | licence                                   | how the known pixels are given                         |
| -------- | -------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| `vace`   | Wan-AI/Wan2.1-VACE-1.3B-diffusers                  | Apache-2.0                                | natively: VACE's masked video-to-video (mask white = generate) |
| `wan22`  | Wan-AI/Wan2.2-TI2V-5B-Diffusers                    | Apache-2.0                                | TI2V's own clean-token conditioning (`expand_timesteps`), generalised from the first frame to every known latent token |
| `cosmos` | nvidia/Cosmos-Predict2-2B-Video2World              | NVIDIA Open Model License; guardrail on   | frame 0 as its Video2World condition; the known latents of later frames replaced at each step's noise level (RePaint) |

The NVIDIA Open Model License ends if its guardrail is bypassed: diffusers' Cosmos pipeline
builds it (`cosmos_guardrail`: a blocklist and Qwen3Guard on the prompt, a video content
filter and face blur on the frames, from nvidia/Cosmos-1.0-Guardrail) and refuses to run
without it; nothing here touches it. The per-step replacement acts on the latents through
the pipeline's own `callback_on_step_end`.

A request is `{"clip": npz (frames (n, h, w, 3) uint8, masks (n, h, w) bool: True =
generate), "prompt", "negative"?, "seed", "steps"?, "guidance"?}`, `h, w` already the model's
size (`MODELS[key].size`) and `n` a multiple of 4 plus 1. The answer is `{"clip": npz
(frames), "model", "seconds", "size"}`: every frame as the model drew it -- the caller keeps
only the masked pixels and gates the rest against what it sent.

Torch and diffusers are imported only on a GPU, so the module (and its tests) import
without them: `latent_known` is numpy.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

VACE_MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
WAN22_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
COSMOS_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_GUARDRAIL = ("nvidia/Cosmos-1.0-Guardrail", "Qwen/Qwen3Guard-Gen-0.6B")


@dataclass(frozen=True)
class Spec:
    """One model: its repositories (and what of them to fetch), the frame size it is run
    at (w, h), its latent compression (time, space) and token patch, frames per second,
    and the default steps and guidance."""

    repos: tuple[str, ...]
    licence: str
    size: tuple[int, int]
    temporal: int
    spatial: int
    patch: int
    fps: float
    steps: int
    guidance: float
    #: `snapshot_download` patterns per repository (None: everything).
    patterns: tuple[str, ...] | None = None


MODELS: dict[str, Spec] = {
    # 480p is what the 1.3B model was trained at (its card); 16 fps.
    "vace": Spec(
        (VACE_MODEL,),
        "Apache-2.0",
        (832, 480),
        4,
        8,
        2,
        16.0,
        30,
        5.0,
        (
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ),
    ),
    # TI2V-5B's 720p size; its VAE compresses 16x16 in space.
    "wan22": Spec(
        (WAN22_MODEL,),
        "Apache-2.0",
        (1280, 704),
        4,
        16,
        2,
        24.0,
        30,
        5.0,
        (
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ),
    ),
    # The 720p 16 fps checkpoint the diffusers folders hold (not the .pt files beside them).
    "cosmos": Spec(
        (COSMOS_MODEL, *COSMOS_GUARDRAIL),
        "NVIDIA Open Model License (guardrail required and on)",
        (1280, 704),
        4,
        8,
        2,
        16.0,
        30,
        7.0,
        (
            "model_index.json",
            "config.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ),
    ),
}

#: What no generated frame should hold.
NEGATIVE = (
    "people, person, animal, text, watermark, logo, frame, border, blur, motion blur, "
    "flicker, distortion, cartoon, painting, low quality, camera shake"
)


def model_name(key: str) -> str:
    return MODELS[key].repos[0]


# --- the masks, in latent space (numpy) ------------------------------------------------------


def latent_frames(n: int, temporal: int = 4) -> list[slice]:
    """The pixel frames each latent frame of a causal video VAE encodes: frame 0 alone,
    then `temporal` at a time."""
    if n < 1 or (n - 1) % temporal:
        raise ValueError(f"{n} frames: must be {temporal}k + 1")
    return [slice(0, 1)] + [
        slice(1 + temporal * k, 1 + temporal * (k + 1)) for k in range((n - 1) // temporal)
    ]


#: A latent cell is given as known when at most this share of its pixels (over its frames)
#: is to be generated: the scan's scattered unknown specks (a gaussian or two too few
#: views) would otherwise turn most cells of a frame into generated ones, and a model that
#: is given almost nothing draws a scene of its own (run 37383986439: Wan2.2's known pixels
#: at 13-19 dB, Cosmos' at 16-20 dB). The specks are shown with the scan's render there.
CELL_GENERATE_SHARE = 0.1
#: A cell at least this much to be generated is part of a hole proper, and only those grow.
CELL_HOLE_SHARE = 0.5


def latent_known(
    masks: np.ndarray, temporal: int, spatial: int, patch: int = 1, grow: int = 1
) -> np.ndarray:
    """Which latent cells (t, h / spatial, w / spatial) are given as known: a cell is known
    when at most `CELL_GENERATE_SHARE` of its block (its frames, `spatial` x `spatial`
    pixels) is to be generated, and it is not within `grow` cells of a hole (a cell at least
    `CELL_HOLE_SHARE` generated: a latent cell's decoder reaches its neighbours); with
    `patch` > 1, only whole patches of `patch` x `patch` cells (a transformer token)."""
    n, h, w = masks.shape
    if h % (spatial * patch) or w % (spatial * patch):
        raise ValueError(f"{w}x{h} is not a multiple of {spatial * patch}")
    share = np.zeros((len(latent_frames(n, temporal)), h // spatial, w // spatial), np.float32)
    for k, frames in enumerate(latent_frames(n, temporal)):
        block = masks[frames].mean(axis=0, dtype=np.float32)
        share[k] = block.reshape(h // spatial, spatial, w // spatial, spatial).mean(axis=(1, 3))
    gen = share > CELL_GENERATE_SHARE
    if grow > 0:
        hole = share >= CELL_HOLE_SHARE
        for dy in range(-grow, grow + 1):
            for dx in range(-grow, grow + 1):
                gen |= np.roll(np.roll(hole, dy, axis=1), dx, axis=2) & _edge_ok(gen.shape, dy, dx)
    if patch > 1:
        t, lh, lw = gen.shape
        tok = gen.reshape(t, lh // patch, patch, lw // patch, patch).any(axis=(2, 4))
        gen = np.repeat(np.repeat(tok, patch, axis=1), patch, axis=2)
    return ~gen


def _edge_ok(shape: tuple[int, ...], dy: int, dx: int) -> np.ndarray:
    """False where `np.roll` wrapped round the frame edge."""
    ok = np.ones(shape, bool)
    if dy > 0:
        ok[:, :dy] = False
    elif dy < 0:
        ok[:, dy:] = False
    if dx > 0:
        ok[:, :, :dx] = False
    elif dx < 0:
        ok[:, :, dx:] = False
    return ok


def pack_clip(
    frames: np.ndarray, masks: np.ndarray | None = None, void: np.ndarray | None = None
) -> bytes:
    """A clip as one npz: frames, the pixels to generate, and of those the void (nothing
    measured there: the others show the scan's own render, a hint VACE may use)."""
    buffer = io.BytesIO()
    arrays = {"frames": np.asarray(frames, np.uint8)}
    if masks is not None:
        arrays["masks"] = np.asarray(masks, bool)
    if void is not None:
        arrays["void"] = np.asarray(void, bool)
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpack_clip(blob: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(io.BytesIO(blob)) as z:
        return z["frames"], (z["masks"] if "masks" in z.files else None)


def unpack_void(blob: bytes) -> np.ndarray | None:
    with np.load(io.BytesIO(blob)) as z:
        return z["void"] if "void" in z.files else None


def greyed(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """The frames with every pixel to generate set to mid grey: what the models are shown
    there (VACE's own convention for an inpainting hole)."""
    out = np.array(frames, np.uint8, copy=True)
    out[masks] = 127
    return out


# --- the GPU side ------------------------------------------------------------------------------


def find_token(environ: dict[str, str] | None = None) -> str | None:
    """The Hugging Face token (`inpaint_models.find_token`'s rule)."""
    import os

    env = os.environ if environ is None else environ
    if env.get("HF_TOKEN"):
        return env["HF_TOKEN"]
    for name in sorted(env):
        if env[name].startswith("hf_") and ("HF" in name.upper() or "HUGGING" in name.upper()):
            env["HF_TOKEN"] = env[name]
            return env[name]
    return None


def access(token: str | None, keys: tuple[str, ...] = tuple(MODELS)) -> dict[str, str]:
    """Per repository the models need: `ok`, or why the token cannot read it."""
    from huggingface_hub import auth_check

    out = {}
    for key in keys:
        for repo in MODELS[key].repos:
            try:
                auth_check(repo, token=token)
                out[repo] = "ok"
            except Exception as error:  # noqa: BLE001 - reported, not raised
                out[repo] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return out


def prefetch(key: str, token: str | None) -> None:
    """The model's repositories into the Hub cache (`HF_HOME`), only what it loads."""
    from huggingface_hub import snapshot_download

    spec = MODELS[key]
    for k, repo in enumerate(spec.repos):
        patterns = list(spec.patterns) if (k == 0 and spec.patterns) else None
        snapshot_download(repo, token=token, allow_patterns=patterns, max_workers=16)


def load(key: str, device: str = "cuda") -> Any:
    import torch

    if key == "vace":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

        vae = AutoencoderKLWan.from_pretrained(
            VACE_MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanVACEPipeline.from_pretrained(VACE_MODEL, vae=vae, torch_dtype=torch.bfloat16)
        # The diffusers example's shift for 480p.
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        return pipe.to(device)
    if key == "wan22":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        vae = AutoencoderKLWan.from_pretrained(
            WAN22_MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN22_MODEL, vae=vae, torch_dtype=torch.bfloat16
        )
        if not pipe.config.expand_timesteps:
            raise RuntimeError(
                "Wan2.2 TI2V-5B is expected to condition per token (expand_timesteps)"
            )
        return pipe.to(device)
    if key == "cosmos":
        from diffusers import Cosmos2VideoToWorldPipeline

        # The guardrail (`safety_checker`) is built by the pipeline; the pipeline refuses to
        # run without it, and runs it on every prompt and every clip.
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(COSMOS_MODEL, torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        # diffusers 0.40 asks every component for `.device` to find where to compute;
        # cosmos_guardrail 0.3.2's checker answers from a face filter that has none (every
        # call of run 37381466964 failed there). Where to compute is said here instead.
        here = torch.device(device)
        type(pipe)._execution_device = property(lambda self: here)  # type: ignore[assignment]
        return pipe
    raise ValueError(f"model {key!r}: one of {', '.join(MODELS)}")


def fill_clip(key: str, pipe: Any, request: dict) -> dict:
    """One clip's masked pixels drawn by the model (the module docstring)."""
    started = time.time()
    frames, masks = unpack_clip(request["clip"])
    if masks is None:
        raise ValueError("the clip carries no masks")
    spec = MODELS[key]
    _, h, w = masks.shape
    if (w, h) != spec.size:
        raise ValueError(f"{key} runs at {spec.size}, the clip is {w}x{h}")
    prompt = str(request["prompt"])
    negative = str(request.get("negative") or NEGATIVE)
    seed = int(request.get("seed", 0))
    steps = int(request.get("steps") or spec.steps)
    guidance = float(request.get("guidance") or spec.guidance)
    # Every model is shown the scan's render where it has a surface (a hint, and what a
    # mostly known latent cell is encoded from) and mid grey where it has nothing.
    void = unpack_void(request["clip"])
    shown = greyed(frames, masks if void is None else void)
    if key == "vace":
        out = _vace(pipe, shown, masks, prompt, negative, seed, steps, guidance)
    elif key == "wan22":
        out = _wan22(pipe, shown, masks, prompt, negative, seed, steps, guidance)
    elif key == "cosmos":
        out = _cosmos(pipe, shown, masks, prompt, negative, seed, steps, guidance)
    else:
        raise ValueError(f"model {key!r}")
    if out.shape != frames.shape:
        raise RuntimeError(f"{key} returned {out.shape}, not {frames.shape}")
    return {
        "clip": pack_clip(out),
        "model": model_name(key),
        "seconds": round(time.time() - started, 1),
        "size": [w, h],
        "steps": steps,
    }


def _u8(video: Any) -> np.ndarray:
    """A pipeline's `np` frames (floats in 0..1, (n, h, w, 3)) as uint8."""
    return np.clip(np.round(np.asarray(video, np.float32) * 255), 0, 255).astype(np.uint8)


def _vace(pipe: Any, shown, masks, prompt, negative, seed, steps, guidance) -> np.ndarray:
    """VACE's masked video-to-video: white = generate. Where the scan has a surface it did not
    see from here, the frame shows its render (VACE's reactive input, a hint it may repaint);
    where it has nothing, mid grey (VACE's own inpainting convention)."""
    import torch
    from PIL import Image

    n, h, w = masks.shape
    video = [Image.fromarray(f) for f in shown]
    mask = [Image.fromarray(np.where(m, 255, 0).astype(np.uint8)) for m in masks]
    result = pipe(
        video=video,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative,
        height=h,
        width=w,
        num_frames=n,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator("cuda").manual_seed(seed),
        output_type="np",
    ).frames[0]
    return _u8(result)


def _video_tensor(frames: np.ndarray, device: str, dtype: Any) -> Any:
    """(1, 3, n, h, w) in -1..1."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device=device, dtype=torch.float32)
    x = x.permute(3, 0, 1, 2).unsqueeze(0) / 127.5 - 1.0
    return x.to(dtype)


def _wan22(pipe: Any, shown, masks, prompt, negative, seed, steps, guidance) -> np.ndarray:
    """TI2V-5B conditions its first frame by giving that frame's latent tokens clean, at
    timestep 0, while the rest denoise (`WanImageToVideoPipeline`, `expand_timesteps`). The
    same mechanism takes any set of tokens: every token `latent_known` gives is clean from the
    encoded clip (`shown`), the others are generated."""
    import torch

    device = "cuda"
    vae = pipe.vae
    transformer = pipe.transformer
    spec = MODELS["wan22"]
    known = latent_known(masks, spec.temporal, spec.spatial, spec.patch)
    with torch.no_grad():
        video = _video_tensor(shown, device, vae.dtype)
        z = vae.encode(video).latent_dist.mode().float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        condition = (z - mean) * inv_std
        if condition.shape[2:] != known.shape:
            raise RuntimeError(f"latents {tuple(condition.shape)} against mask {known.shape}")
        generate = torch.from_numpy((~known).astype(np.float32)).to(device)[None, None]
        g = torch.Generator(device).manual_seed(seed)
        latents = torch.randn(condition.shape, generator=g, device=device, dtype=torch.float32)
        do_cfg = guidance > 1.0
        pipe.text_encoder.to(device)  # (back from the CPU, where the last clip left it)
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        dtype = transformer.dtype
        prompt_embeds = prompt_embeds.to(dtype)
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(dtype)
        # The text encoder (umt5-xxl, ~11 GB) waits on the CPU while the clip is denoised and
        # decoded: the 720p decode needs the room (run 37381466964 ran out of memory there).
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        pipe.scheduler.set_timesteps(steps, device=device)
        token_gen = generate[0][0][:, :: spec.patch, :: spec.patch]
        for t in pipe.scheduler.timesteps:
            model_input = ((1 - generate) * condition + generate * latents).to(dtype)
            timestep = (token_gen * t).flatten().unsqueeze(0)
            noise = transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_cfg:
                uncond = transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_embeds,
                    return_dict=False,
                )[0]
                noise = uncond + guidance * (noise - uncond)
            latents = pipe.scheduler.step(noise, t, latents, return_dict=False)[0]
        latents = (1 - generate) * condition + generate * latents
        del noise, model_input, condition, video, z
        torch.cuda.empty_cache()
        decoded = vae.decode((latents / inv_std + mean).to(vae.dtype), return_dict=False)[0]
        video_np = ((decoded.float().clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 3, 0).cpu().numpy()
    del decoded
    torch.cuda.empty_cache()
    return _u8(video_np)


def _cosmos(pipe: Any, shown, masks, prompt, negative, seed, steps, guidance) -> np.ndarray:
    """Video2World on frame 0 (its own conditioning), and after every step the latents of the
    known cells replaced by the clip's own at that step's noise level: `x0 + sigma * eps` in
    the pipeline's EDM parameterisation (`latents = x0 + sigma * noise`)."""
    import torch
    from PIL import Image

    device = "cuda"
    vae = pipe.vae
    spec = MODELS["cosmos"]
    n, h, w = masks.shape
    known = latent_known(masks, spec.temporal, spec.spatial, 1)
    sigma_data = float(pipe.scheduler.config.sigma_data)
    with torch.no_grad():
        video = _video_tensor(shown, device, vae.dtype)
        z = vae.encode(video).latent_dist.mode().float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        x0 = (z - mean) / std * sigma_data
    if x0.shape[2:] != known.shape:
        raise RuntimeError(f"latents {tuple(x0.shape)} against mask {known.shape}")
    keep = torch.from_numpy(known.astype(np.float32)).to(device)[None, None]
    g = torch.Generator(device).manual_seed(seed + 1)
    eps = torch.randn(x0.shape, generator=g, device=device, dtype=torch.float32)

    def replace(pipeline: Any, step: int, timestep: Any, kwargs: dict) -> dict:
        latents = kwargs["latents"]
        sigma = pipeline.scheduler.sigmas[min(step + 1, len(pipeline.scheduler.sigmas) - 1)]
        sigma = sigma.to(device=latents.device, dtype=torch.float32)
        given = x0 + sigma * eps
        mixed = keep * given + (1 - keep) * latents.float()
        return {"latents": mixed.to(latents.dtype)}

    result = pipe(
        image=Image.fromarray(np.asarray(shown[0], np.uint8)),
        prompt=prompt,
        negative_prompt=negative,
        height=h,
        width=w,
        num_frames=n,
        num_inference_steps=steps,
        guidance_scale=guidance,
        fps=int(spec.fps),
        generator=torch.Generator(device).manual_seed(seed),
        output_type="np",
        callback_on_step_end=replace,
        callback_on_step_end_tensor_inputs=["latents"],
    ).frames[0]
    return _u8(result)
