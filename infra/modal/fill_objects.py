"""The inferred fill's object round on Modal: the pumpkins' unseen undersides completed by 3D
object-completion models (`tools/captures/object_fill.py`, `object_models.py`), registered to
the real scan, the measured splats untouched.

Run from the repository root (`.github/workflows/fill-objects.yml` does, on a push to
`wm-fill-objects` whose message carries an `[objects|...]` token):

    modal run infra/modal/fill_objects.py --check            # which models the token can read
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

APP_NAME = "hexapod-fill-objects"
app = modal.App(APP_NAME)
CAPTURES = "/root/captures"

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    LOCAL_CAPTURES = ROOT / "tools" / "captures"
else:
    LOCAL_CAPTURES = Path(CAPTURES)

HF_SECRET = modal.Secret.from_name(os.environ.get("HEXAPOD_HF_SECRET", "huggingface"))

#: The access check: the Hub client alone, on a CPU.
access_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.34,<1.0")
    .add_local_file(LOCAL_CAPTURES / "object_models.py", "/root/object_models.py")
)


@app.function(image=access_image, secrets=[HF_SECRET], cpu=1.0, memory=1024, timeout=300)
def objects_access(methods: list[str]) -> dict:
    """Whose token the secret holds and which repositories each method reads it can read
    (`object_models.access`); nothing is downloaded."""
    sys.path.insert(0, "/root")
    import object_models

    keys = sorted(k for k in os.environ if "HF" in k.upper() or "HUGGING" in k.upper())
    out = object_models.access(object_models.find_token(), methods)
    out["secretKeys"] = keys
    return out


@app.local_entrypoint()
def main(
    check: bool = False,
    jobs: str = "",
    methods: str = "",
    budget_usd: float = 0.0,
    spent_usd: float = 0.0,
    options: str = "",
    out: str = "objects-out",
) -> None:
    """`check`: which repositories each method reads the token can read (`access.json`)."""
    sys.path.insert(0, str(LOCAL_CAPTURES))
    import object_models

    wanted = [m.strip() for m in methods.split("+") if m.strip()] or list(object_models.REPOS)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    if check:
        result = objects_access.remote(wanted)
        (folder / "access.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        sys.stdout.write(f"access: {json.dumps(result, indent=1)}\n")
