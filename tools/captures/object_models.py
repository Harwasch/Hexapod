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
                        card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
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
