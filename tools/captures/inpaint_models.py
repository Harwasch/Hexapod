"""Generative image inpainting for Teacher B's holes: the server side of
`world_model_client.GenerativeFiller`, run on a Modal GPU by `infra/modal/fill.py` and
`infra/modal/world_models.py` (`InpaintSDXL`, `InpaintQwen`, `InpaintFlux`), which run *this*
module. NVIDIA Fixer cleans what a render shows; it does not invent what is missing (on the
pumpkin's hole it kept Telea's flat pre-fill). These paint the masked pixels from a prompt
and the pixels around them.

| key    | weights                                                    | licence                        |
| ------ | ---------------------------------------------------------- | ------------------------------ |
| `sdxl` | diffusers/stable-diffusion-xl-1.0-inpainting-0.1           | CreativeML OpenRAIL++-M        |
| `qwen` | Qwen/Qwen-Image + InstantX/Qwen-Image-ControlNet-Inpainting | Apache-2.0 (both)             |
| `flux` | black-forest-labs/FLUX.1-Fill-dev (gated)                  | FLUX.1 [dev] Non-Commercial    |

`flux` is here for comparison only: its licence allows non-commercial use of the model (its
outputs are the user's), so it cannot run in a product; and the account must have accepted
it on the Hub. OpenRAIL++-M is permissive with use restrictions (its Attachment A) that pass
on to anyone redistributing the weights; Apache-2.0 has none.

A request is `{"image": png, "mask": png (white = paint), "prompt", "negative"?, "seed",
"steps"?, "guidance"?, "strength"?}`; the answer `{"image": png at the input's size,
"model", "seconds"}`. The whole frame comes back as the model drew it: the caller keeps
only the masked pixels (`GenerativeFiller` composites), so the model may re-encode the rest.

Torch and diffusers are imported only on a GPU, so the module (and its tests) import
without them.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

SDXL_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
QWEN_MODEL = "Qwen/Qwen-Image"
QWEN_CONTROLNET = "InstantX/Qwen-Image-ControlNet-Inpainting"
FLUX_FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"


@dataclass(frozen=True)
class Spec:
    """One model's defaults: denoising steps, guidance, and the multiple a side must be."""

    repos: tuple[str, ...]
    steps: int
    guidance: float
    multiple: int


MODELS: dict[str, Spec] = {
    # The model card's recipe: 20-30 steps, guidance 8, strength 0.99 (1.0 can drift colour).
    "sdxl": Spec((SDXL_MODEL,), 30, 8.0, 8),
    # The ControlNet card's: 30 steps, true CFG 4.
    "qwen": Spec((QWEN_MODEL, QWEN_CONTROLNET), 30, 4.0, 16),
    # The model card's: 50 steps, guidance 30.
    "flux": Spec((FLUX_FILL_MODEL,), 50, 30.0, 16),
}


def model_name(key: str) -> str:
    return " + ".join(MODELS[key].repos)


def fit_size(width: int, height: int, multiple: int, area: int = 1024 * 1024) -> tuple[int, int]:
    """(w, h) with the aspect of `width x height`, at most `area` pixels (never upscaled
    past the input), each side a multiple of `multiple`."""
    scale = min(1.0, (area / (width * height)) ** 0.5)
    return (
        max(multiple, int(width * scale) // multiple * multiple),
        max(multiple, int(height * scale) // multiple * multiple),
    )


def load(key: str, device: str = "cuda") -> Any:
    """The pipeline for `key`, on `device` (weights through HF_HOME, the token from HF_TOKEN)."""
    import torch

    if key == "sdxl":
        from diffusers import AutoPipelineForInpainting

        pipe = AutoPipelineForInpainting.from_pretrained(
            SDXL_MODEL, torch_dtype=torch.float16, variant="fp16"
        )
    elif key == "qwen":
        from diffusers import QwenImageControlNetInpaintPipeline, QwenImageControlNetModel

        controlnet = QwenImageControlNetModel.from_pretrained(
            QWEN_CONTROLNET, torch_dtype=torch.bfloat16
        )
        pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
            QWEN_MODEL, controlnet=controlnet, torch_dtype=torch.bfloat16
        )
    elif key == "flux":
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(FLUX_FILL_MODEL, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"inpainting model {key!r}: one of {', '.join(MODELS)}")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def inpaint(key: str, pipe: Any, request: dict) -> dict:
    """One request (module docstring) through `pipe`, a `load(key)`."""
    import torch
    from PIL import Image

    started = time.time()
    spec = MODELS[key]
    image = Image.open(io.BytesIO(request["image"])).convert("RGB")
    mask = Image.open(io.BytesIO(request["mask"])).convert("L")
    size = image.size
    width, height = fit_size(*size, spec.multiple)
    image_in = image.resize((width, height), Image.LANCZOS)
    # Nearest, then binarised: a soft edge would leave part-painted pixels in the hole.
    mask_in = mask.resize((width, height), Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
    steps = int(request.get("steps", spec.steps))
    guidance = float(request.get("guidance", spec.guidance))
    generator = torch.Generator("cuda").manual_seed(int(request.get("seed", 0)))
    prompt = str(request["prompt"])
    negative = str(request.get("negative", ""))
    if key == "sdxl":
        out = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            image=image_in,
            mask_image=mask_in,
            width=width,
            height=height,
            strength=float(request.get("strength", 0.99)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
    elif key == "qwen":
        out = pipe(
            prompt=prompt,
            negative_prompt=negative or " ",
            control_image=image_in,
            control_mask=mask_in,
            controlnet_conditioning_scale=float(request.get("control", 1.0)),
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=guidance,
            generator=generator,
        ).images[0]
    else:
        out = pipe(
            prompt=prompt,
            image=image_in,
            mask_image=mask_in,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]
    buffer = io.BytesIO()
    out.convert("RGB").resize(size, Image.LANCZOS).save(buffer, format="PNG")
    return {
        "image": buffer.getvalue(),
        "model": model_name(key),
        "seconds": round(time.time() - started, 2),
        "size": [width, height],
    }


def access(token: str | None) -> dict[str, str]:
    """Per repository every model needs: `ok`, or why the token cannot read it (gated and
    not accepted, missing) -- checked without downloading."""
    from huggingface_hub import auth_check

    out = {}
    for spec in MODELS.values():
        for repo in spec.repos:
            try:
                auth_check(repo, token=token)
                out[repo] = "ok"
            except Exception as error:  # noqa: BLE001 - reported, not raised
                out[repo] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
    return out
