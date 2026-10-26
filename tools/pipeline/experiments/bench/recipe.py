"""The production train params, explicit schedule overlays, and one guarded seed edit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from recipe import Recipe

PIPELINE = Path(__file__).resolve().parents[2]
SEED_CALL = re.compile(r"(?m)^(?P<indent>[ \t]*)set_random_seed\(42 \+ local_rank\)[ \t]*$")


def patch_seed(trainer: Path, seed: int) -> Path:
    if seed == 42:
        return trainer
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    source = trainer.read_text()
    if len(SEED_CALL.findall(source)) != 1:
        raise ValueError("expected exactly one set_random_seed(42 + local_rank) line")
    patched = trainer.with_name(f"{trainer.stem}_bench_seed{seed}.py")
    patched.write_text(
        SEED_CALL.sub(lambda m: f"{m['indent']}set_random_seed({seed} + local_rank)", source)
    )
    return patched


def train_params(schedule: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be {train: {parameter: value}}")
    document = yaml.safe_load((PIPELINE / "recipes/photo-reconstruct.yaml").read_text())
    params = dict(next(stage for stage in document["stages"] if stage["id"] == "train")["params"])
    if schedule == "screen":
        params.update(schedule_scale=0.1, train_max_side=800)
    elif schedule != "full":
        raise ValueError("schedule must be screen or full")
    if set(overrides) - {"train"} or not isinstance(overrides.get("train", {}), dict):
        raise ValueError("overrides must be {train: {parameter: value}}")
    params.update(overrides.get("train", {}))
    # Raw argv could redirect the dataset, change the split or load an external ckpt.
    # New supported knobs should be added as stage params, with an explicit audit.
    forbidden = {"trainer", "python", "extra_args", "holdout_script", "init_from"}
    if forbidden & set(overrides.get("train", {})):
        raise ValueError("bench refuses trainer/argv/init overrides that bypass frozen inputs")
    if params.get("data_factor", 1) != 1:
        raise ValueError("bench uses train_max_side with data_factor=1 for identical preprocessing")
    return params


def make_recipe(params: dict[str, Any], tier: str) -> Recipe:
    return Recipe.from_dict(
        {
            "name": "frozen-bench",
            "version": 1,
            "inputs": ["frames", "poses"],
            "stages": [
                {
                    "id": "train",
                    "impl": "gsplat",
                    "params": params,
                    "gpu": {"tier": tier, "preemptible": False},
                }
            ],
        }
    )
