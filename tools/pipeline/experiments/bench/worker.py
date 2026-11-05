"""Container/local execution. This module has no Modal or object-storage dependency."""

from __future__ import annotations

import base64
import contextlib
import filecmp
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import rehearsal
from .gpu import GpuSampler
from .ledger import STARTUP_SECONDS
from .live_sync import watching
from .recipe import make_recipe, patch_photometry, patch_seed
from .results import METRICS, parse_result
from .scene import prepare_pool, read_json, verify, write_json

ARTIFACTS = (
    "train_metrics.json",
    "prepass.json",
    "holdout.json",
    "converge.json",
    "trainer_timing.json",
    "train_memory.json",
    "post_processing.json",
    "gpu_samples.json",
    "test-trained.json",
    "quality.json",
    "test.json",
)


def publish_cache(source: Path, target: Path) -> None:
    """Publish immutable cache entries atomically when several variants share a volume."""
    target.mkdir(parents=True, exist_ok=True)
    # Arrays precede entry markers/manifests, including when a copy is interrupted.
    for path in sorted(source.iterdir(), key=lambda p: (p.suffix == ".json", p.name)):
        if not path.is_file() or path.name.endswith(".tmp.npz"):
            continue
        # Only committed predictions and the small metadata files are cache entries.
        if path.suffix not in (".npy", ".npz", ".json"):
            continue
        with tempfile.NamedTemporaryFile(dir=target, prefix=".publish-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(path, temporary)
            temporary.replace(target / path.name)
        finally:
            temporary.unlink(missing_ok=True)


def topdown_preview(
    scene: Path, ply: Path, destination: Path, params: dict[str, Any], remaining: float
) -> dict[str, Any]:
    """Best effort, at most a minute within the existing function reservation."""
    deadline = time.monotonic() + min(60, remaining)
    try:
        import numpy as np

        import sfm
        import training
        from holdout_error import read_ply

        from .topdown_maths import frame_camera

        if remaining <= 0:
            raise TimeoutError("no top-down time left in function budget")
        model = sfm.read_model(scene / "poses")
        up = sfm.camera_up(model)
        columns = read_ply(ply)
        camera = frame_camera(
            np.stack([columns[k] for k in ("x", "y", "z")], axis=1),
            [image.centre for image in model.images],
            None if up is None else up.up,
        )
        if up is not None:
            camera["upConsistency"] = up.consistency
        camera_path = destination / "renders/topdown-camera.json"
        write_json(camera_path, camera)
        output = destination / "renders/topdown.png"
        command = [
            training.trainer_python(None),
            str(Path(__file__).with_name("topdown.py")),
            "--ply",
            str(ply),
            "--camera",
            str(camera_path),
            "--out",
            str(output),
            "--degree",
            str(params.get("ship_sh_degree", 0)),
        ]
        if params.get("antialiased"):
            command.append("--antialiased")
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("no top-down time left after camera framing")
        subprocess.run(command, check=True, timeout=left)  # noqa: S603
        png = output.read_bytes()
        if len(png) > 1_000_000 or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("top-down output is not a small PNG")
        return {
            "status": "complete",
            "path": "renders/topdown.png",
            "camera": camera,
            "pngBase64": base64.b64encode(png).decode("ascii"),
        }
    except Exception as error:
        return {"status": "failed", "warning": f"{type(error).__name__}: {error}"}


class RestartedError(RuntimeError):
    """This run already started once: Modal is re-running a preempted input."""


def claim(destination: Path, *, attempt_started: float) -> None:
    """Mark the run started, or refuse: the caller commits the volume before training.

    A preempted container's input is restarted by Modal on another container. Without
    this marker the second attempt would train again and bill again, and the ledger
    would only ever see the last attempt's cost.
    """
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "started.json"
    try:
        with marker.open("x") as handle:
            handle.write(json.dumps({"attemptStartedAt": attempt_started}) + "\n")
    except FileExistsError as error:
        raise RestartedError(f"run {destination.name} already started ({marker})") from error


def restart_result(plan: dict[str, Any], error: RestartedError) -> dict[str, Any]:
    """What a refused restart returns. Unknown cost: the ledger keeps the reservation."""
    return {
        "schemaVersion": 1,
        "runId": plan["runId"],
        "scene": plan["scene"],
        "tag": plan.get("tag"),
        "variant": plan.get("variant", {"name": "unknown", "overrides": {}}),
        "seed": plan.get("seed"),
        "schedule": plan["schedule"],
        "splitDigest": plan.get("splitDigest"),
        "manifestDigest": plan.get("manifestDigest"),
        "benchCommit": plan.get("benchCommit"),
        "gpu": {"tier": plan.get("tier"), "name": None},
        # A refused restart has no new measurements. Keep the report shape so it is
        # visible in tagged results and cannot crash the batch's Markdown report.
        "test": dict.fromkeys(METRICS),
        "splats": None,
        "trainSeconds": None,
        "wallSeconds": None,
        "rehearsal": bool(plan.get("rehearsal")),
        "status": "preempted",
        "error": f"Preempted; restart refused, no second attempt: {error}",
        "costUsd": None,
    }


def billed_from(
    *, started_at: float | None, booted_at: float, first_call: bool, now: float
) -> tuple[float, str]:
    """When this call's bill began, and how that was known.

    A cold call is billed from the container's start. When the sandbox's uptime cannot
    say when that was, the import time misses the image mount and Python's start, so
    the startup limit is billed on top of it rather than nothing.
    """
    if not first_call:
        return now, "warm-call"
    if started_at is not None:
        return started_at, "container-uptime"
    return booted_at - STARTUP_SECONDS, "boot-minus-startup-limit"


def run_training(
    scene: Path,
    destination: Path,
    plan: dict[str, Any],
    *,
    billed_since: float | None = None,
    live_commit: Callable[[], object] | None = None,
) -> dict[str, Any]:
    entered = time.time() if billed_since is None else billed_since
    import stages  # noqa: F401 - register implementations, never execute a pose stage
    import training
    from executor import execute
    from runners import LocalRunner, RunnerSet
    from workdir import Workdir

    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "result.json").exists():
        raise ValueError(f"run {destination.name} already has a result")
    params = dict(plan["trainParams"])
    prior_cache = destination.parent.parent / "prior_cache" / str(plan["scene"])
    split_digest = manifest_digest = None
    train_seconds = None
    gpu_name = None
    error = None
    topdown: dict[str, Any] = (
        {"status": "skipped", "warning": "evaluation did not finish"}
        if plan.get("topdown")
        else {"status": "off"}
    )
    sampler = GpuSampler(None if plan.get("rehearsal") else shutil.which("nvidia-smi"))
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
                if params.get("post_processing") == "ppisp":
                    # The synthetic schedule is only 30/300 steps. This exercises
                    # argv validation; it is not a measured controller schedule.
                    params["ppisp_controller_activation_num_steps"] = 1
                gpu_name = "CPU rehearsal (no GPU)"
            else:
                original = training.trainer_script(None)
                trainer = patch_seed(original, plan["seed"])
                if params.get("post_processing") == "ppisp":
                    trainer = patch_photometry(trainer)
                if trainer != original:
                    params["trainer"] = str(trainer)
                smi = shutil.which("nvidia-smi")
                if smi:
                    with contextlib.suppress(OSError, subprocess.SubprocessError):
                        gpu_name = subprocess.check_output(  # noqa: S603 - fixed argv, no shell
                            [smi, "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=2
                        ).strip()
            if prior_cache.exists():
                shutil.copytree(
                    prior_cache, work.checkpoint_dir("train") / "prior_cache", dirs_exist_ok=True
                )
            recipe = make_recipe(params, plan["tier"])
            stage_start = time.monotonic()
            sampler.start()
            try:
                with watching(plan, work, scene, destination, live_commit):
                    execute(recipe, work, RunnerSet(cpu=LocalRunner(), gpu=LocalRunner()))
            finally:
                train_seconds = time.monotonic() - stage_start
                sampler.stop()
                cached = work.checkpoint_dir("train") / "prior_cache"
                if cached.exists():
                    publish_cache(cached, prior_cache)
            trained = work.out_dir("train") / "trained.ply"
            shipped = trained
            shutil.copyfile(trained, destination / "trained.ply")
            if params.get("_bench_quality") is not None:
                shipped = work.out_dir("quality") / "gated.ply"
                shutil.copyfile(
                    work.out_dir("quality") / "quality.json", destination / "quality.json"
                )
                gate = read_json(destination / "quality.json")["bar"]
                shutil.copyfile(
                    shipped, destination / ("faded.ply" if gate == "fade" else "gated.ply")
                )
            raw = training.latest_ply(work.work_dir("train") / "gsplat")
            if (
                plan.get("keepSplat")
                and raw is not None
                and not filecmp.cmp(raw, shipped, shallow=False)
            ):
                shutil.copyfile(raw, destination / "full_sh.ply")
            if plan.get("rehearsal"):
                test = rehearsal.evaluation(split["test"])
                write_json(destination / "test.json", test)
                write_json(destination / "test-trained.json", test)
                if plan.get("topdown"):
                    topdown = {"status": "skipped", "warning": "rehearsal has no GPU renders"}
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
                if raw is not None:
                    command += ["--raw-ply", str(raw)]
                if params.get("antialiased"):
                    command.append("--antialiased")
                if plan.get("renders", 0):
                    command += [
                        "--renders",
                        str(plan["renders"]),
                        "--render-dir",
                        str(destination / "renders"),
                    ]
                remaining = plan["timeoutSeconds"] - (time.time() - entered) - 15
                if remaining <= 0:
                    raise TimeoutError("no evaluation time left in function budget")
                subprocess.run(command, check=True, timeout=remaining)  # noqa: S603
                if shipped != trained:
                    # Same untouched cameras, pixels and renderer; only gate output
                    # changes. Keep the training export's score beside the headline.
                    before = command.copy()
                    before[before.index("--ply") + 1] = str(trained)
                    before[before.index("--out") + 1] = str(destination / "test-trained.json")
                    if "--render-dir" in before:
                        before[before.index("--render-dir") + 1] = str(
                            destination / "renders-trained"
                        )
                    remaining = plan["timeoutSeconds"] - (time.time() - entered) - 15
                    if remaining <= 0:
                        raise TimeoutError("no pre-gate evaluation time left in function budget")
                    subprocess.run(before, check=True, timeout=remaining)  # noqa: S603
                else:
                    shutil.copyfile(destination / "test.json", destination / "test-trained.json")
                if plan.get("topdown"):
                    topdown = topdown_preview(
                        scene,
                        shipped,
                        destination,
                        params,
                        plan["timeoutSeconds"] - (time.time() - entered) - 15,
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            sampler.stop()
            # Keep the full merged stdout/stderr even when training or gating fails.
            # This lives on the bench volume; only the small error tail is returned.
            if work.log_path("train").is_file():
                shutil.copyfile(work.log_path("train"), destination / "trainer.log")
            write_json(destination / "gpu_samples.json", {"samples": sampler.samples})
            sources = [
                work.out_dir("train"),
                work.work_dir("train"),
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
        result["topdown"] = topdown
        write_json(destination / "result.json", result)
        if not plan.get("keepSplat"):
            for name in ("trained.ply", "faded.ply", "gated.ply"):
                (destination / name).unlink(missing_ok=True)
        return result
