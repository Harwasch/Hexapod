"""Unrun ARKit PR1 benchmark. Existing bundle on hexapod-bench; no production writes.

modal run infra/modal/arkit_bench.py --bundle-path /bench/inputs/99c02b1d.hexcap \
    --tag arkit-pr1-001 --revision COMMIT --cap 12
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
SHARED = HERE / "app.py" if modal.is_local() else Path("/root/bench_training_app.py")
spec = importlib.util.spec_from_file_location("arkit_bench_training", SHARED)
assert spec is not None and spec.loader is not None
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
sys.path.insert(0, str(shared.LOCAL_PIPELINE) if modal.is_local() else shared.PIPELINE_DIR)

app = modal.App("hexapod-bench-arkit")
volume = modal.Volume.from_name("hexapod-bench", create_if_missing=False)
cpu_image = shared.cpu_image.add_local_file(SHARED, "/root/bench_training_app.py")
gpu_image = shared.image.add_local_file(SHARED, "/root/bench_training_app.py")
CPU_TIMEOUT = 3600
GPU_TIMEOUT = 7200


@app.function(
    image=cpu_image,
    cpu=(4, 4),
    memory=8192,
    timeout=CPU_TIMEOUT,
    volumes={"/bench": volume},
    max_containers=1,
    retries=0,
)
def prepare(bundle_path: str, tag: str, revision: str) -> None:
    import stages  # noqa: F401
    from executor import execute
    from experiments.bench.scene import seal
    from recipe import Recipe, load_recipe
    from runners import RunnerSet
    from workdir import Workdir

    volume.reload()
    # Not resolve(): Modal mounts the volume at /bench through a symlink.
    bundle = Path(os.path.normpath(bundle_path))
    if not bundle.is_relative_to("/bench") or not bundle.is_file():
        raise ValueError("bundle must already exist on hexapod-bench")
    source = load_recipe("photo-reconstruct")
    for arm in ("v17", "pr1"):
        destination = Path("/bench/arkit") / tag / arm
        if destination.exists():
            raise ValueError("benchmark tag already exists; choose a new tag")
        work = Workdir.create(destination / "work")
        work.input_path("upload").mkdir()
        shutil.copyfile(bundle, work.input_path("upload") / bundle.name)
        stages_config = []
        for stage in source.stages[:2]:
            # As the API's capture_bundles.PARAMS: route both stages to the bundle branch.
            params = {**stage.params, "capture_bundle": True}
            if stage.id == "pose":
                params.update(arkit_refine=arm == "pr1", arkit_shared_lens=arm == "pr1")
            stages_config.append({"id": stage.id, "impl": stage.impl, "params": params})
        recipe = Recipe.from_dict(
            {
                "name": "arkit-bench",
                "version": 17 if arm == "v17" else 18,
                "inputs": ["upload"],
                "stages": stages_config,
            }
        )
        execute(recipe, work, RunnerSet.local())
        if arm == "pr1":
            report = json.loads((work.out_dir("pose") / "poses" / "poses.json").read_text())
            if report["refinement"]["status"] != "accepted":
                volume.commit()
                raise ValueError(
                    "PR1 fell back; inspect the CPU pose report before buying GPU arms"
                )
        scene = destination / "scene"
        shutil.copytree(work.out_dir("normalize") / "frames", scene / "frames")
        shutil.copytree(work.out_dir("pose") / "poses", scene / "poses")
        # Identical scene salt gives both arms the identical deterministic held-out names.
        seal(scene, "arkit-99c02b1d", bundle_path, "root", revision)
    volume.commit()


@app.function(
    image=gpu_image,
    gpu="L40S",
    cpu=(4, 4),
    memory=16384,
    timeout=GPU_TIMEOUT,
    volumes={"/bench": volume},
    max_containers=1,
    retries=0,
    single_use_containers=True,
)
def train(tag: str, arm: str, revision: str) -> dict[str, Any]:
    from experiments.bench.recipe import train_params
    from experiments.bench.worker import claim, run_training

    volume.reload()
    destination = Path("/bench/arkit") / tag / arm
    run = destination / "run"
    # Refuse an automatic replay instead of buying a second training attempt.
    claim(run, attempt_started=time.time())
    volume.commit()
    params = train_params("full", {"train": {"test_every": 8, "block_parallel": 1}})
    plan = {
        "runId": f"{tag}-{arm}",
        "scene": "arkit-99c02b1d",
        "benchCommit": revision,
        "variant": {"name": arm, "overrides": {}},
        "seed": 42,
        "schedule": "full",
        "tier": "l40s",
        "trainParams": params,
        "keepSplat": True,
        "renders": 16,
    }
    result = run_training(destination / "scene", run, plan)
    volume.commit()
    return result


@app.local_entrypoint()
def main(bundle_path: str, tag: str, revision: str, cap: float = 12.0) -> None:
    import re

    from experiments.bench.ledger import hourly_rate

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tag):
        raise ValueError("tag must be a fresh simple directory name")
    # Timeout plus startup/replay allowances. Forecast only; use Modal's billing too.
    maximum = (CPU_TIMEOUT + 240) / 3600 * hourly_rate("cpu4")
    maximum += 2 * (GPU_TIMEOUT + 240) / 3600 * hourly_rate("l40s")
    if maximum > cap:
        raise ValueError(f"timeout reservation ${maximum:.2f} exceeds cap ${cap:.2f}")
    prepare.remote(bundle_path, tag, revision)
    for arm in ("v17", "pr1"):
        result = train.remote(tag, arm, revision)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        if result.get("status") != "complete":
            raise RuntimeError(f"{arm} failed; stop before another paid arm")
