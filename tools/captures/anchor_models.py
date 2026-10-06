"""The models of the inferred-fill bake-off's round 2 ("anchor, then propagate"): the server
side of `anchor_fill`'s editor and set filler, run on Modal H100s by `infra/modal/fill.py`
(`EditQwen`, `FillVace14`), and the depth and perceptual models its job runs itself.

| role                     | weights                                                      | licence    |
| ------------------------ | ------------------------------------------------------------ | ---------- |
| anchor and residual fill | Qwen/Qwen-Image-Edit-2511                                    | Apache-2.0 |
| ...its 4/8-step LoRA     | lightx2v/Qwen-Image-Edit-2511-Lightning                      | Apache-2.0 |
| joint fill of all views  | Wan-AI/Wan2.1-VACE-14B-diffusers                             | Apache-2.0 |
| ...its optional 4-step   | lightx2v/Wan2.1-Distill-Loras (T2V-14B, rank 64)             | Apache-2.0 |
| depth, anchored          | depth-anything/prompt-depth-anything-vitl-hf                 | Apache-2.0 |
| depth, fallback          | depth-anything/Depth-Anything-V2-Small-hf                    | Apache-2.0 |
| exact-mask fallback      | Qwen/Qwen-Image + InstantX/Qwen-Image-ControlNet-Inpainting  | Apache-2.0 |
| scores only (not shipped)| LPIPS (BSD-2, AlexNet), DreamSim (MIT; DINO ViT-B/16)        | --         |

**The editor** (`edit`). Qwen-Image-Edit-2511 regenerates a whole image from its input
images and a prompt. Picture 1 is the masked render at the target pose (the pixels to make
painted a key colour); pictures 2-3, when given, are real photos. Each request carries a
per-pixel **strength**: 0 where the scan knows the pixel, partial (0.3-0.6) where it saw it
badly, 1 where it must be made. At the end of every denoising step the output tokens whose
strength is at or below the next noise level are put back to the render's own latents noised
to that level (`hold`): a known token ends exactly as rendered, a weak one is denoised freely
only from its strength down (SDEdit per token), an unknown one is generated from pure noise.
`anchor_fill` then registers the result to the render on the known pixels and composites it
back, so known pixels never change.

**The set filler** (`fill_set`). Wan2.1-VACE-14B's masked video-to-video over frames that
are not a camera path but the set of target views (the ObjFiller-3D recipe): real photos and
anchor fills as frames with nothing masked, then the views still to fill, in a smooth tour.

Torch, diffusers and transformers are imported only where a model runs, so the module (and
its tests) import without them.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"
LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_FILES = {
    4: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    8: "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}
VACE14_MODEL = "Wan-AI/Wan2.1-VACE-14B-diffusers"
WAN_DISTILL_REPO = "lightx2v/Wan2.1-Distill-Loras"
WAN_DISTILL_FILE = "wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors"
PROMPT_DEPTH_MODEL = "depth-anything/prompt-depth-anything-vitl-hf"
MONO_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
LICENCES = {
    EDIT_MODEL: "Apache-2.0",
    LIGHTNING_REPO: "Apache-2.0",
    VACE14_MODEL: "Apache-2.0",
    WAN_DISTILL_REPO: "Apache-2.0",
    PROMPT_DEPTH_MODEL: "Apache-2.0",
    MONO_DEPTH_MODEL: "Apache-2.0",
}
#: The editor's scheduler with the Lightning LoRA (its distillation's fixed shift of 3).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
#: The key colour the pixels to make are painted in picture 1.
KEY = (255, 0, 255)
#: A token is one 2x2 patch of 8x-downsampled latents: 16x16 pixels.
TOKEN_PX = 16
#: VACE-14B at 480p (its card's lower size), 16 fps; frames 4k + 1, at most 81.
VACE_SIZE = (832, 480)
VACE_MAX_FRAMES = 81


# --- images on the wire ---------------------------------------------------------------------------


def encode_png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(blob: bytes) -> np.ndarray:
    from PIL import Image

    image = Image.open(io.BytesIO(blob))
    return np.asarray(image.convert("L" if image.mode in ("L", "I", "I;16") else "RGB"))


def encode_strength(strength: np.ndarray) -> bytes:
    """A strength map (0..1) as an 8-bit PNG."""
    return encode_png(np.clip(np.round(np.asarray(strength) * 255), 0, 255).astype(np.uint8))


def decode_strength(blob: bytes) -> np.ndarray:
    return decode_png(blob).astype(np.float32) / 255.0


# --- the per-token hold (numpy) ----------------------------------------------------------------------


def fit_size(width: int, height: int, multiple: int = TOKEN_PX) -> tuple[int, int]:
    return max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)


def token_strength(
    strength: np.ndarray, width: int, height: int, token_px: int = TOKEN_PX
) -> np.ndarray:
    """Per output token (row-major over the `height/token x width/token` grid the pipeline
    packs latents in; a token is `2 * vae_scale_factor` pixels, 16 for Qwen-Image) the
    strength it is held to: the largest of its pixels', so a token with any pixel to make is
    free from that pixel's strength."""
    import cv2

    s = cv2.resize(
        np.asarray(strength, np.float32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    th, tw = height // token_px, width // token_px
    blocks = s[: th * token_px, : tw * token_px].reshape(th, token_px, tw, token_px)
    return blocks.max(axis=(1, 3)).reshape(-1)


def held(tokens: np.ndarray, sigma: float) -> np.ndarray:
    """The tokens held to the render at noise level `sigma`: those whose strength is at or
    below it (a strength of 1 or more is never held)."""
    return (tokens <= sigma) & (tokens < 1.0)


# --- the editor (GPU) --------------------------------------------------------------------------------


@dataclass
class Editor:
    """Qwen-Image-Edit-2511 loaded, with its Lightning LoRA as adapters (unfused, so a
    request may run without it) and the two schedulers."""

    pipe: Any
    default_scheduler: Any
    lightning_scheduler: Any
    lightning: dict[int, str]
    model: str = EDIT_MODEL


def load_editor(device: str = "cuda", lightning: tuple[int, ...] = (4, 8)) -> Editor:
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(EDIT_MODEL, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    default = pipe.scheduler
    fast = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    names = {}
    for steps in lightning:
        name = f"lightning{steps}"
        pipe.load_lora_weights(
            LIGHTNING_REPO, weight_name=LIGHTNING_FILES[steps], adapter_name=name
        )
        names[steps] = name
    pipe = pipe.to(device)
    return Editor(pipe, default, fast, names)


@contextlib.contextmanager
def _adapters(editor: Editor, steps: int, lightning: bool) -> Iterator[int]:
    """The pipeline set for a request: the Lightning adapter of `steps` (else the nearest
    one there is) and its scheduler, or neither. Yields the steps to run."""
    pipe = editor.pipe
    if lightning and editor.lightning:
        have = min(editor.lightning, key=lambda k: abs(k - steps))
        pipe.set_adapters([editor.lightning[have]], [1.0])
        pipe.scheduler = editor.lightning_scheduler
        try:
            yield have
        finally:
            pipe.scheduler = editor.default_scheduler
    else:
        if editor.lightning:
            pipe.disable_lora()
        try:
            yield steps
        finally:
            if editor.lightning:
                pipe.enable_lora()


def render_latents(pipe: Any, render: np.ndarray, width: int, height: int) -> Any:
    """The render encoded as the pipeline's packed output latents, (1, tokens, channels)."""
    import torch
    from PIL import Image

    device = pipe._execution_device
    image = pipe.image_processor.preprocess(Image.fromarray(render), height, width).unsqueeze(2)
    with torch.no_grad():
        z = pipe._encode_vae_image(image.to(device=device, dtype=pipe.vae.dtype), generator=None)
    c, h, w = z.shape[1], z.shape[3], z.shape[4]
    return pipe._pack_latents(z, 1, c, h, w)


def hold_callback(x0: Any, eps: Any, tokens: np.ndarray) -> Any:
    """The step-end callback that puts the held tokens back to `x0` noised to the next
    level, `(1 - sigma) x0 + sigma eps` (flow matching)."""
    import torch

    strength = torch.as_tensor(tokens, device=x0.device, dtype=torch.float32)

    def callback(pipe: Any, i: int, t: Any, kwargs: dict) -> dict:
        latents = kwargs["latents"]
        sigmas = pipe.scheduler.sigmas
        sigma = float(sigmas[min(i + 1, len(sigmas) - 1)])
        keep = (strength <= sigma) & (strength < 1.0)
        if not bool(keep.any()):
            return {"latents": latents}
        noised = (1.0 - sigma) * x0.float() + sigma * eps.float()
        mixed = torch.where(keep[None, :, None], noised, latents.float())
        return {"latents": mixed.to(latents.dtype)}

    return callback


@contextlib.contextmanager
def _vae_area(area: int | None) -> Iterator[None]:
    """The pipeline encodes every input image at about `area` pixels (its module constant,
    read at each call; 1024 x 1024 by default): fewer tokens per reference photo."""
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as module

    old = module.VAE_IMAGE_SIZE
    if area:
        module.VAE_IMAGE_SIZE = int(area)
    try:
        yield
    finally:
        module.VAE_IMAGE_SIZE = old


def edit(editor: Editor, request: dict) -> dict:
    """One request (`anchor_fill.EditRequest.wire`): `image` (picture 1, the condition),
    `render` (the scan's own render, what known tokens are held to), `strength` (8-bit
    map), `references` (pictures 2-3), `prompt`, `negative`?, `seed`, `steps`, `lightning`,
    `hold`, `size` [w, h], `cfg` (true CFG without Lightning), `vae_area` (pixels each
    input image is encoded at). Returns `image` (PNG at `size`), `seconds`, and what ran."""
    import torch
    from PIL import Image

    started = time.time()
    pipe = editor.pipe
    token_px = 2 * int(pipe.vae_scale_factor)
    width, height = fit_size(*request["size"], token_px)
    condition = decode_png(request["image"])
    images = [Image.fromarray(condition)] + [
        Image.fromarray(decode_png(r)) for r in request.get("references", [])
    ]
    seed = int(request.get("seed", 0))
    device = pipe._execution_device
    kwargs: dict[str, Any] = {}
    with _adapters(
        editor, int(request.get("steps", 8)), bool(request.get("lightning", True))
    ) as steps:
        cfg = 1.0 if request.get("lightning", True) else float(request.get("cfg", 4.0))
        if cfg > 1.0:
            kwargs.update(negative_prompt=request.get("negative") or " ", true_cfg_scale=cfg)
        else:
            kwargs.update(true_cfg_scale=1.0)
        generator = torch.Generator(device=device).manual_seed(seed)
        if request.get("hold", True) and "strength" in request and "render" in request:
            render = decode_png(request["render"])
            x0 = render_latents(pipe, render, width, height)
            g = torch.Generator(device=device).manual_seed(seed + 7919)
            eps = torch.randn(x0.shape, generator=g, device=device, dtype=torch.float32).to(
                x0.dtype
            )
            tokens = token_strength(decode_strength(request["strength"]), width, height, token_px)
            if tokens.size != x0.shape[1]:
                raise RuntimeError(f"{tokens.size} token strengths for {x0.shape[1]} tokens")
            kwargs.update(
                latents=eps,
                callback_on_step_end=hold_callback(x0, eps, tokens),
                callback_on_step_end_tensor_inputs=["latents"],
            )
        with _vae_area(request.get("vae_area")):
            out = pipe(
                image=images,
                prompt=str(request["prompt"]),
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                output_type="np",
                **kwargs,
            ).images[0]
    rgb = np.clip(np.round(np.asarray(out, np.float32) * 255), 0, 255).astype(np.uint8)
    w0, h0 = request["size"]
    if rgb.shape[:2] != (h0, w0):
        rgb = np.asarray(Image.fromarray(rgb).resize((w0, h0), Image.LANCZOS))
    return {
        "image": encode_png(rgb),
        "seconds": round(time.time() - started, 2),
        "model": editor.model,
        "steps": steps,
        "lightning": bool(request.get("lightning", True)),
        "size": [w0, h0],
    }


# --- the set filler (GPU) ------------------------------------------------------------------------------


def load_vace14(device: str = "cuda", distill: bool = True) -> tuple[Any, bool]:
    """VACE-14B (bf16 transformer, fp32 VAE) and whether the 4-step distill LoRA loaded (an
    adapter, off unless a request asks)."""
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline

    vae = AutoencoderKLWan.from_pretrained(VACE14_MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(VACE14_MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    loaded = False
    if distill:
        try:
            pipe.load_lora_weights(
                WAN_DISTILL_REPO, weight_name=WAN_DISTILL_FILE, adapter_name="distill"
            )
            pipe.disable_lora()
            loaded = True
        except Exception as error:  # noqa: BLE001 - reported; full steps still run
            print(f"the distill LoRA did not load: {error!r}")
    return pipe.to(device), loaded


def fill_set(
    pipe: Any, request: dict, distill_loaded: bool = False, size: tuple[int, int] = VACE_SIZE
) -> dict:
    """The masked frames of `request["clip"]` (`video_fill_models.pack_clip`: frames,
    masks with True = generate, void) filled by VACE. `steps`, `guidance`, `shift`, and
    `distill` (the 4-step LoRA, guidance 1) when it loaded."""
    import torch
    from diffusers import UniPCMultistepScheduler
    from PIL import Image

    import video_fill_models as vfm

    started = time.time()
    frames, masks = vfm.unpack_clip(request["clip"])
    if masks is None:
        raise ValueError("the clip carries no masks")
    n, h, w = masks.shape
    if (w, h) != tuple(size) or n > VACE_MAX_FRAMES or (n - 1) % 4:
        raise ValueError(f"{n} frames of {w}x{h}: VACE-14B takes 4k+1 <= 81 of {size}")
    void = vfm.unpack_void(request["clip"])
    shown = vfm.greyed(frames, masks if void is None else void)
    use_distill = bool(request.get("distill")) and distill_loaded
    steps = int(request.get("steps") or (6 if use_distill else 25))
    guidance = 1.0 if use_distill else float(request.get("guidance") or 5.0)
    shift = float(request.get("shift") or (5.0 if use_distill else 3.0))
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=shift)
    if distill_loaded:
        if use_distill:
            pipe.enable_lora()
        else:
            pipe.disable_lora()
    result = pipe(
        video=[Image.fromarray(f) for f in shown],
        mask=[Image.fromarray(np.where(m, 255, 0).astype(np.uint8)) for m in masks],
        prompt=str(request["prompt"]),
        negative_prompt=str(request.get("negative") or vfm.NEGATIVE),
        height=h,
        width=w,
        num_frames=n,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(pipe._execution_device).manual_seed(int(request.get("seed", 0))),
        output_type="np",
    ).frames[0]
    out = vfm._u8(result)
    if out.shape != frames.shape:
        raise RuntimeError(f"VACE returned {out.shape}, not {frames.shape}")
    return {
        "clip": vfm.pack_clip(out),
        "model": VACE14_MODEL,
        "seconds": round(time.time() - started, 1),
        "steps": steps,
        "guidance": guidance,
        "distill": use_distill,
    }


# --- depth (the job's own GPU) --------------------------------------------------------------------------


@dataclass
class PromptDepth:
    """Prompt Depth Anything: metric depth of an image, given a low-resolution metric depth
    to agree with (here the scan's own rendered depth, its holes filled smoothly). Scale is
    normalised around `CANONICAL_M` for the model and restored after."""

    model: str = PROMPT_DEPTH_MODEL
    device: str = "cuda"
    name: str = "prompt-depth-anything-vitl"
    _parts: Any = None
    #: The prompt's median is moved to this many metres for the model (its training range).
    CANONICAL_M: float = 2.0
    #: The prompt is given at most this many pixels on its long side (LiDAR-like).
    PROMPT_PX: int = 256

    def depth(self, image: np.ndarray, prompt: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from transformers import AutoImageProcessor, PromptDepthAnythingForDepthEstimation

        if self._parts is None:
            processor = AutoImageProcessor.from_pretrained(self.model)
            net = PromptDepthAnythingForDepthEstimation.from_pretrained(self.model)
            self._parts = (processor, net.to(self.device).eval())
        processor, net = self._parts
        h, w = image.shape[:2]
        p = np.asarray(prompt, np.float64)
        ok = np.isfinite(p) & (p > 0)
        if not ok.any():
            raise ValueError("an empty depth prompt")
        scale = self.CANONICAL_M / float(np.median(p[ok]))
        p = np.where(ok, p, np.nan) * scale
        filled = fill_holes(p)
        long = max(h, w)
        f = min(1.0, self.PROMPT_PX / long)
        small = cv2.resize(
            filled.astype(np.float32), (max(1, round(w * f)), max(1, round(h * f))), cv2.INTER_AREA
        )
        # The processor reads the prompt in millimetres (its default prompt scale).
        inputs = processor(images=image, prompt_depth=small * 1000.0, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = net(**inputs)
        pred = processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        d = pred["predicted_depth"].float().cpu().numpy()
        return d / scale


@dataclass
class MonoDepth:
    """Depth Anything V2 Small: relative inverse depth (the fallback; anchored by fitting)."""

    model: str = MONO_DEPTH_MODEL
    device: str = "cuda"
    name: str = "depth-anything-v2-small"
    _parts: Any = None
    disparity_out: bool = True

    def depth(self, image: np.ndarray, prompt: np.ndarray | None = None) -> np.ndarray:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._parts is None:
            processor = AutoImageProcessor.from_pretrained(self.model)
            net = AutoModelForDepthEstimation.from_pretrained(self.model).to(self.device).eval()
            self._parts = (processor, net)
        processor, net = self._parts
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = net(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred[:, None], size=image.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        return pred.float().cpu().numpy()


def fill_holes(depth: np.ndarray, levels: int = 8) -> np.ndarray:
    """`depth` with its NaN holes filled by normalised convolution at growing scales (a
    smooth continuation of what is around them; the far background where nothing is)."""
    import cv2

    d = np.asarray(depth, np.float64)
    known = np.isfinite(d)
    if known.all():
        return d
    if not known.any():
        return np.ones_like(d)
    out = np.where(known, d, 0.0)
    filled = known.copy()
    value = out.copy()
    sigma = 1.0
    for _ in range(levels):
        w = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
        s = cv2.GaussianBlur(np.where(known, d, 0.0).astype(np.float32), (0, 0), sigma)
        ok = (w > 1e-3) & ~filled
        value[ok] = s[ok] / w[ok]
        filled |= ok
        if filled.all():
            break
        sigma *= 2.0
    value[~filled] = float(np.nanmax(np.where(known, d, np.nan)))
    return value


# --- perceptual scores (the job's GPU; evaluation only) ---------------------------------------------------


@dataclass
class Perceptual:
    """LPIPS (AlexNet) as a spatial map, and DreamSim when it loads; both on images in a
    mask (DreamSim on the mask's bounding box, outside the mask grey in both)."""

    device: str = "cuda"
    _lpips: Any = None
    _dreamsim: Any = None
    dreamsim_error: str = ""

    def lpips_map(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import lpips
        import torch

        if self._lpips is None:
            self._lpips = (
                lpips.LPIPS(net="alex", spatial=True, verbose=False).to(self.device).eval()
            )
        t = lambda x: (
            torch.from_numpy(np.asarray(x, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        ).to(self.device)
        with torch.no_grad():
            m = self._lpips(t(a), t(b))
        return m[0, 0].float().cpu().numpy()

    def lpips(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float(self.lpips_map(a, b)[mask].mean())

    def dreamsim(self, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
        if not mask.any() or self.dreamsim_error:
            return None
        try:
            import torch
            from dreamsim import dreamsim
            from PIL import Image

            if self._dreamsim is None:
                import os

                model, preprocess = dreamsim(
                    pretrained=True,
                    device=self.device,
                    dreamsim_type="dino_vitb16",
                    cache_dir=os.environ.get("DREAMSIM_CACHE", "/tmp/dreamsim"),
                )
                self._dreamsim = (model, preprocess)
            model, preprocess = self._dreamsim
            ys, xs = np.nonzero(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = lambda im: np.where(mask[y0:y1, x0:x1, None], im[y0:y1, x0:x1], 127).astype(
                np.uint8
            )
            ta = preprocess(Image.fromarray(crop(a))).to(self.device)
            tb = preprocess(Image.fromarray(crop(b))).to(self.device)
            with torch.no_grad():
                return float(model(ta, tb).item())
        except Exception as error:  # noqa: BLE001 - recorded; the other scores stand
            self.dreamsim_error = repr(error)[:500]
            return None
