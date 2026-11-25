"""Unrun ARKit pose/depth benchmark. Existing bundle on hexapod-bench; no production writes.

modal run infra/modal/arkit_bench.py --bundle-path /bench/inputs/99c02b1d.hexcap \
    --tag arkit-pr2-001 --revision COMMIT --arms pr1+init --cap 6
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
sys.path.insert(
    0, str(shared.LOCAL_PIPELINE) if modal.is_local() else shared.PIPELINE_DIR
)

app = modal.App("hexapod-bench-arkit")
volume = modal.Volume.from_name("hexapod-bench", create_if_missing=False)
cpu_image = shared.cpu_image.add_local_file(SHARED, "/root/bench_training_app.py")
gpu_image = shared.image.add_local_file(SHARED, "/root/bench_training_app.py")
CPU_TIMEOUT = 3600
# The v17 arm trained 60k steps in 5575 s wall on L40S; keep about 2x headroom.
GPU_TIMEOUT = 10800
ARMS = ("v17", "pr1", "pr1+init", "pr1+init+depthloss")


def select_arms(value: str) -> tuple[str, ...]:
    selected = tuple(value.split(","))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(a not in ARMS for a in selected)
    ):
        raise ValueError(f"arms must be comma-separated unique choices from {ARMS}")
    return selected


def arm_params(arm: str) -> dict[str, Any]:
    select_arms(arm)
    return {
        "test_every": 8,
        "block_parallel": 1,
        "arkit_depth_init": arm in {"pr1+init", "pr1+init+depthloss"},
        "arkit_depth_loss": arm == "pr1+init+depthloss",
    }


@app.function(
    image=cpu_image,
    cpu=(4, 4),
    memory=8192,
    timeout=CPU_TIMEOUT,
    volumes={"/bench": volume},
    max_containers=1,
    retries=0,
)
def prepare(bundle_path: str, tag: str, revision: str, arms: tuple[str, ...]) -> None:
    import stages  # noqa: F401
    from executor import execute
    from experiments.bench.scene import seal, verify
    from recipe import Recipe, load_recipe
    from runners import RunnerSet
    from workdir import Workdir

    volume.reload()
    # Not resolve(): Modal mounts the volume at /bench through a symlink.
    bundle = Path(os.path.normpath(bundle_path))
    if not bundle.is_relative_to("/bench") or not bundle.is_file():
        raise ValueError("bundle must already exist on hexapod-bench")
    source = load_recipe("photo-reconstruct")
    for arm in arms:
        destination = Path("/bench/arkit") / tag / arm
        if destination.exists():
            raise ValueError(
                "this arm already exists; choose an unrun arm or fresh tag"
            )
        base = "v17" if arm == "v17" else "pr1"
        prepared = Path("/bench/arkit") / tag / "_prepared" / base
        if (prepared / "scene").exists():
            manifest, _ = verify(prepared / "scene")
            if (
                manifest["benchCommit"] != revision
                or manifest["sourcePrefix"] != bundle_path
            ):
                raise ValueError("tag belongs to another revision or bundle")
            shutil.copytree(prepared / "scene", destination / "scene")
            continue
        work = Workdir.create(prepared / "work")
        work.input_path("upload").mkdir()
        shutil.copyfile(bundle, work.input_path("upload") / bundle.name)
        stages_config = []
        for stage in source.stages[:2]:
            # As the API's capture_bundles.PARAMS: route both stages to the bundle branch.
            params = {**stage.params, "capture_bundle": True}
            if stage.id == "pose":
                params.update(
                    arkit_refine=base == "pr1", arkit_shared_lens=base == "pr1"
                )
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
        if base == "pr1":
            report = json.loads(
                (work.out_dir("pose") / "poses" / "poses.json").read_text()
            )
            if report["refinement"]["status"] != "accepted":
                volume.commit()
                raise ValueError(
                    "PR1 fell back; inspect the CPU pose report before buying GPU arms"
                )
        scene = prepared / "scene"
        shutil.copytree(work.out_dir("normalize") / "frames", scene / "frames")
        shutil.copytree(work.out_dir("pose") / "poses", scene / "poses")
        # Identical scene salt gives both arms the identical deterministic held-out names.
        seal(scene, "arkit-99c02b1d", bundle_path, "root", revision)
        shutil.copytree(scene, destination / "scene")
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
    params = train_params("full", {"train": arm_params(arm)})
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
        "timeoutSeconds": GPU_TIMEOUT,
    }
    result = run_training(destination / "scene", run, plan)
    volume.commit()
    return result


@app.function(
    image=gpu_image,
    gpu="L40S",
    cpu=(4, 4),
    memory=16384,
    timeout=3600,
    volumes={"/bench": volume},
    max_containers=1,
    retries=0,
    single_use_containers=True,
)
def evaluate_existing(tag: str, arm: str) -> None:
    """Re-run run_training's external test and renders on an already trained arm.

    The first v17 arm trained fully, then lost its evaluation to a missing plan key.
    This repeats that exact evaluate.py call on its saved PLYs; it never retrains.
    """
    import subprocess
    import tempfile

    import training
    from experiments.bench.recipe import train_params

    volume.reload()
    root = Path("/bench/arkit") / tag / arm
    scene, run = root / "scene", root / "run"
    if (run / "test.json").exists() or not (run / "trained.ply").is_file():
        raise ValueError("arm already evaluated or not trained")
    params = train_params("full", {"train": arm_params(arm)})
    with tempfile.TemporaryDirectory() as temporary:
        data = training.build_dataset(
            scene / "frames",
            scene / "poses",
            Path(temporary) / "evaluation",
            max_side=params.get("train_max_side"),
        )
        command = [
            training.trainer_python(None),
            str(Path(shared.PIPELINE_DIR) / "experiments" / "bench" / "evaluate.py"),
            "--data", str(data),
            "--ply", str(run / "trained.ply"),
            "--trainer", str(training.trainer_script(params.get("trainer"))),
            "--split", str(scene / "split.json"),
            "--scene", str(scene),
            "--out", str(run / "test.json"),
            "--degree", str(params.get("ship_sh_degree", 0)),
            "--renders", "16",
            "--render-dir", str(run / "renders"),
        ]  # fmt: skip
        if (run / "full_sh.ply").is_file():
            command += ["--raw-ply", str(run / "full_sh.ply")]
        if params.get("antialiased"):
            command.append("--antialiased")
        subprocess.run(command, check=True, timeout=3300)  # noqa: S603
    volume.commit()


@app.local_entrypoint()
def main(
    bundle_path: str,
    tag: str,
    revision: str,
    cap: float = 20.0,
    arms: str = ",".join(ARMS),
) -> None:
    import re

    from experiments.bench.ledger import hourly_rate

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tag):
        raise ValueError("tag must be a fresh simple directory name")
    selected = select_arms(arms)
    # Timeout plus startup/replay allowances. Forecast only; use Modal's billing too.
    maximum = (CPU_TIMEOUT + 240) / 3600 * hourly_rate("cpu4")
    maximum += len(selected) * (GPU_TIMEOUT + 240) / 3600 * hourly_rate("l40s")
    if maximum > cap:
        raise ValueError(f"timeout reservation ${maximum:.2f} exceeds cap ${cap:.2f}")
    sys.stdout.write(
        f"Forecast reservation for {len(selected)} arm(s): ${maximum:.2f}\n"
    )
    prepare.remote(bundle_path, tag, revision, selected)
    for arm in selected:
        result = train.remote(tag, arm, revision)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        if result.get("status") != "complete":
            raise RuntimeError(f"{arm} failed; stop before another paid arm")
