"""Container/local execution. This module has no Modal or object-storage dependency."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import rehearsal
from .recipe import make_recipe, patch_seed
from .results import parse_result
from .scene import prepare_pool, read_json, verify, write_json

ARTIFACTS = (
    "train_metrics.json",
    "holdout.json",
    "converge.json",
    "trainer_timing.json",
    "test.json",
)


def run_training(
    scene: Path, destination: Path, plan: dict[str, Any], *, billed_since: float | None = None
) -> dict[str, Any]:
    entered = time.time() if billed_since is None else billed_since
    import stages  # noqa: F401 - register implementations, never execute a pose stage
    import training
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    destination.mkdir(parents=True, exist_ok=False)
    params = dict(plan["trainParams"])
    split_digest = manifest_digest = None
    train_seconds = None
    gpu_name = None
    error = None
    with tempfile.TemporaryDirectory(prefix="bench-work-") as temporary:
        work = Workdir.create(Path(temporary) / plan["runId"])
        try:
            # Freeze may be explicitly replaced by another client while training runs.
            # Pin a container-local snapshot and validate it after the copy as well.
            manifest, _ = verify(scene, plan.get("manifestDigest"))
            snapshot = Path(temporary) / "frozen"
            shutil.copytree(scene, snapshot)
            manifest, split = verify(snapshot, manifest["digest"])
            scene = snapshot
            split_digest, manifest_digest = split["digest"], manifest["digest"]
            prepare_pool(scene, work.inputs_dir, split)
            if plan.get("rehearsal"):
                params.update(rehearsal.stand_in_params())
                gpu_name = "CPU rehearsal (no GPU)"
            else:
                original = training.trainer_script(None)
                trainer = patch_seed(original, plan["seed"])
                if trainer != original:
                    params["trainer"] = str(trainer)
                smi = shutil.which("nvidia-smi")
                if smi:
                    gpu_name = subprocess.check_output(  # noqa: S603 - fixed argv, no shell
                        [smi, "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10
                    ).strip()
            recipe = make_recipe(params, plan["tier"])
            stage_start = time.monotonic()
            try:
                execute(recipe, work, RunnerSet(cpu=LocalRunner(), gpu=LocalRunner()))
            finally:
                train_seconds = time.monotonic() - stage_start
            shipped = work.out_dir("train") / "trained.ply"
            shutil.copyfile(shipped, destination / "trained.ply")
            if plan.get("rehearsal"):
                test = rehearsal.evaluation(split["test"])
                write_json(destination / "test.json", test)
            else:
                # Full model/images live outside the runner and are only opened now.
                data = training.build_dataset(
                    scene / "frames",
                    scene / "poses",
                    Path(temporary) / "evaluation",
                    max_side=params.get("train_max_side"),
                )
                command = [
                    training.trainer_python(None),
                    str(Path(__file__).with_name("evaluate.py")),
                    "--data",
                    str(data),
                    "--ply",
                    str(shipped),
                    "--trainer",
                    str(training.trainer_script(params.get("trainer"))),
                    "--split",
                    str(scene / "split.json"),
                    "--scene",
                    str(scene),
                    "--out",
                    str(destination / "test.json"),
                    "--degree",
                    str(params.get("ship_sh_degree", 0)),
                ]
                raw = training.latest_ply(work.work_dir("train") / "gsplat")
                if raw is not None:
                    command += ["--raw-ply", str(raw)]
                if params.get("antialiased"):
                    command.append("--antialiased")
                remaining = plan["timeoutSeconds"] - (time.time() - entered) - 15
                if remaining <= 0:
                    raise TimeoutError("no evaluation time left in function budget")
                subprocess.run(command, check=True, timeout=remaining)  # noqa: S603
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            sources = [
                work.out_dir("train"),
                work.out_dir("train") / "holdout",
                work.work_dir("train") / "gsplat",
            ]
            for name in ARTIFACTS[:-1]:
                source = next((s / name for s in sources if (s / name).is_file()), None)
                if source is not None:
                    shutil.copyfile(source, destination / name)
        result = parse_result(
            plan,
            destination,
            params=params,
            split_digest=split_digest,
            manifest_digest=manifest_digest,
            wall=time.time() - entered,
            train_seconds=train_seconds,
            gpu_name=gpu_name,
            error=error,
        )
        # Return these small reports as well, so the local copy needs no volume download.
        result["artifacts"] = {
            name: read_json(destination / name)
            for name in ARTIFACTS
            if (destination / name).is_file()
        }
        write_json(destination / "result.json", result)
        if not plan.get("keepSplat"):
            (destination / "trained.ply").unlink(missing_ok=True)
        return result
