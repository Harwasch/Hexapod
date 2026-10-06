"""The 3D object-completion models of the inferred-fill bake-off's object round: the server side
of `object_fill`'s generators, run on Modal GPUs by `infra/modal/fill_objects.py`.

Each model is given one real photo of an object, its mask (the segmentation's, rendered from
the scan) and, where it takes one, the scan's own points in that camera (a pointmap), and
returns the whole object -- the sides no camera saw included -- as gaussians or a coloured
point set in the model's frame, with whatever pose the model gives for it. `object_fill`
registers it to the scan and keeps only what the scan never saw.

| method | weights                                      | licence (code / weights)                     | gate                |
| ------ | -------------------------------------------- | -------------------------------------------- | ------------------- |
| A      | facebook/sam-3d-objects                      | SAM License / SAM License (commercial use    | gated (Meta)        |
|        |                                              | allowed with its conditions)                 |                     |
|        | ...its depth model, unused with a pointmap   | MoGe: MIT                                    | --                  |
| B      | TencentARC/Pixal3D                           | MIT / MIT                                    | --                  |
|        | ...its image encoder                         | DINOv3 License (facebook/dinov3-vitl16-...)  | gated (Meta)        |
|        | ...its background remover (never loaded)     | briaai/RMBG-2.0: CC BY-NC 4.0                | gated; NOT USABLE   |
| B'     | microsoft/TRELLIS.2-4B (Pixal3D's backbone)  | MIT / MIT, same DINOv3 and RMBG-2.0          | as B                |
| C      | Stable-X/trellis-vggt-v0-2 (ReconViaGen 0.5) | MIT label, but fine-tuned from VGGT-1B       | research only       |
|        |                                              | (CC BY-NC 4.0); plus TRELLIS.2 as B'         |                     |
| S      | microsoft/TRELLIS-image-large (stand-in)     | MIT / MIT; DINOv2 (torch.hub) Apache-2.0     | --                  |

`REPOS` lists every repository each method reads, and `access` checks which of them the
workspace's token can read, without downloading anything.

Torch and the models' own packages are imported only where a model runs, so the module (and
its tests) import without them.
"""

from __future__ import annotations

from typing import Any

#: Each method's repositories: (repo, what it is, licence as its card states it, use).
#: `use`: "load" (the run downloads it), "never" (listed for the licence note only).
REPOS: dict[str, list[tuple[str, str, str, str]]] = {
    "sam3d": [
        ("facebook/sam-3d-objects", "SAM 3D Objects checkpoints", "SAM License", "load"),
        ("Ruicheng/moge-vitl", "MoGe (SAM 3D's depth model; built, unused)", "MIT", "load"),
    ],
    "pixal3d": [
        ("TencentARC/Pixal3D", "Pixal3D checkpoints", "MIT", "load"),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "image encoder", "DINOv3 License", "load"),
        ("briaai/RMBG-2.0", "background remover (masks are given)", "CC BY-NC 4.0", "never"),
    ],
    "trellis2": [
        ("microsoft/TRELLIS.2-4B", "TRELLIS.2 checkpoints", "MIT", "load"),
        ("microsoft/TRELLIS-image-large", "its sparse-structure decoder", "MIT", "load"),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "image encoder", "DINOv3 License", "load"),
        ("briaai/RMBG-2.0", "background remover (masks are given)", "CC BY-NC 4.0", "never"),
    ],
    "reconviagen": [
        ("Stable-X/trellis-vggt-v0-2", "ReconViaGen's VGGT sparse structure", "MIT", "never"),
        ("facebook/VGGT-1B", "the VGGT it is fine-tuned from", "CC BY-NC 4.0", "never"),
        ("facebook/VGGT-1B-Commercial", "VGGT, commercial terms", "VGGT License", "never"),
    ],
    "trellis": [
        ("microsoft/TRELLIS-image-large", "TRELLIS (v1) image-to-3D", "MIT", "load"),
    ],
}
#: The model pages an owner accepts for a gated method, by method.
GATES = {
    "sam3d": ["https://huggingface.co/facebook/sam-3d-objects"],
    "pixal3d": ["https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"],
    "trellis2": ["https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"],
}


def find_token(environ: dict[str, str] | None = None) -> str | None:
    """The Hugging Face token (`inpaint_models.find_token`'s rule): `HF_TOKEN`, else the first
    variable a Modal secret put in the environment that looks like one, copied to `HF_TOKEN`."""
    import os

    env = os.environ if environ is None else environ
    if env.get("HF_TOKEN"):
        return env["HF_TOKEN"]
    for name in sorted(env):
        if env[name].startswith("hf_") and ("HF" in name.upper() or "HUGGING" in name.upper()):
            env["HF_TOKEN"] = env[name]
            return env[name]
    return None


def access(token: str | None, methods: list[str] | None = None) -> dict[str, Any]:
    """Who the token is, and per method and repository: `ok` or why it cannot be read (gated
    and not accepted, missing), with the licence and gate the Hub reports. Nothing is
    downloaded. A method is `readable` when every repository it loads is."""
    from huggingface_hub import HfApi, auth_check

    api = HfApi(token=token)
    out: dict[str, Any] = {"methods": {}, "repos": {}}
    try:
        out["user"] = api.whoami().get("name")
    except Exception as error:  # noqa: BLE001 - reported, the check goes on
        out["user"] = None
        out["whoamiError"] = f"{type(error).__name__}: {error}"[:300]
    for method in methods or list(REPOS):
        readable = True
        for repo, _what, licence, use in REPOS[method]:
            if repo not in out["repos"]:
                entry: dict[str, Any] = {"cardLicence": licence}
                try:
                    auth_check(repo, token=token)
                    entry["access"] = "ok"
                except Exception as error:  # noqa: BLE001 - this is the answer
                    entry["access"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
                try:
                    info = api.model_info(repo)
                    entry["gated"] = getattr(info, "gated", None)
                    card = getattr(info, "card_data", None) or {}
                    entry["hubLicence"] = (
                        card.get("license")
                        if isinstance(card, dict)
                        else getattr(card, "license", None)
                    )
                except Exception as error:  # noqa: BLE001
                    entry["infoError"] = f"{type(error).__name__}: {str(error)[:200]}"
                out["repos"][repo] = entry
            if use == "load" and out["repos"][repo]["access"] != "ok":
                readable = False
        out["methods"][method] = {
            "readable": readable,
            "gates": GATES.get(method, []) if not readable else [],
        }
    return out


# --- arrays over the wire --------------------------------------------------------------------------

#: A model's gaussians: positions (n, 3), unit quaternions (w, x, y, z), scales, colours in
#: [0, 1], opacities in [0, 1].
GAUSSIAN_FIELDS = ("positions", "rotations", "scales", "colours", "opacities")


def pack_array(a: Any, dtype: str = "float32") -> dict[str, Any]:
    """An array as plain bytes (no numpy pickle across numpy versions)."""
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return {"dtype": dtype, "shape": list(arr.shape), "data": arr.tobytes()}


def unpack_array(d: dict[str, Any]) -> Any:
    import numpy as np

    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"]).copy()


def pack_gaussians(
    positions: Any, rotations: Any, scales: Any, colours: Any, opacities: Any
) -> dict[str, Any]:
    return {
        "positions": pack_array(positions),
        "rotations": pack_array(rotations),
        "scales": pack_array(scales),
        "colours": pack_array(colours, "float16"),
        "opacities": pack_array(opacities, "float16"),
    }


# --- TRELLIS (v1), the stand-in: microsoft/TRELLIS-image-large ---------------------------------------

TRELLIS_MODEL = "microsoft/TRELLIS-image-large"
#: The sampler settings of TRELLIS's own example (12 steps each; guidance 7.5 and 3).
TRELLIS_SS = {"steps": 12, "cfg_strength": 7.5}
TRELLIS_SLAT = {"steps": 12, "cfg_strength": 3.0}
_SH_C0 = 0.28209479177387814


def load_trellis() -> Any:
    """The image-to-3D pipeline on the GPU (its DINOv2 encoder through torch.hub)."""
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipe = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL)
    pipe.cuda()
    return pipe


def generate_trellis(pipe: Any, request: dict[str, Any]) -> dict[str, Any]:
    """One object from the request's frames (RGBA crops, the mask as alpha): one frame with
    `run`, several with `run_multi_image` (stochastic: each step conditioned on one of
    them). Gaussians only, in TRELLIS's object frame (z up, within [-0.5, 0.5])."""
    import io
    import time

    import numpy as np
    import torch
    from PIL import Image

    started = time.time()
    images = [Image.open(io.BytesIO(f["png"])).convert("RGBA") for f in request["frames"]]
    seed = int(request.get("seed", 1))
    ss = {**TRELLIS_SS, **(request.get("ss") or {})}
    slat = {**TRELLIS_SLAT, **(request.get("slat") or {})}
    with torch.no_grad():
        if len(images) == 1:
            out = pipe.run(
                images[0],
                seed=seed,
                formats=["gaussian"],
                sparse_structure_sampler_params=ss,
                slat_sampler_params=slat,
            )
        else:
            out = pipe.run_multi_image(
                images,
                seed=seed,
                formats=["gaussian"],
                sparse_structure_sampler_params=ss,
                slat_sampler_params=slat,
                mode="stochastic",
            )
    g = out["gaussian"][0]
    xyz = g.get_xyz.detach().float().cpu().numpy()
    rot = g.get_rotation.detach().float().cpu().numpy()
    scale = g.get_scaling.detach().float().cpu().numpy()
    opacity = g.get_opacity.detach().float().cpu().numpy().reshape(-1)
    dc = g._features_dc.detach().float().cpu().numpy().reshape(len(xyz), -1)[:, :3]
    colour = np.clip(0.5 + _SH_C0 * dc, 0.0, 1.0)
    torch.cuda.empty_cache()
    return {
        "key": request["key"],
        "frame": "z-up",
        "model": TRELLIS_MODEL,
        "views": [f["view"] for f in request["frames"]],
        "count": int(len(xyz)),
        "seconds": round(time.time() - started, 1),
        "gaussians": pack_gaussians(xyz, rot, scale, colour, opacity),
    }
