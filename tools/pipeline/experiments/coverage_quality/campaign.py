"""Resume the bounded $5 campaign on hexapod-bench; no network until a paid action/download.

Run locally as the bench is documented: from tools/pipeline,
uv run --with modal==1.6.2 --with matplotlib==3.10.9 python -m
experiments.coverage_quality.campaign ... (Matplotlib is needed for local pool plots).
Use ONE --local-out for this campaign. Claims are single-use locally and on volume;
unknown outcomes keep their reservation. No automatic client retry or budget recycling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.bench.ledger import hourly_rate, locked
from experiments.bench.scene import read_json, verify, write_json
from experiments.coverage_quality.selection import choose_test_views

CAMPAIGN = "coverage-quality-20261125-v2"
ROOT = Path("/bench/research") / CAMPAIGN
BEACH_RUN = "4769ee65e33c44f2aacd87a45dfa3811"
REHEARSAL_RUN = f"{CAMPAIGN}-rehearse-c935-r2"
SPENT_USD = 0.5844
C935_SCALE_SOURCE = "ARKit metric poses (job c9357a92 georeference scaleSource=arkit, scale 1.0)"
BEACH_SCALE_SOURCE = (
    "camera-height estimate from production job 4dd02171 georeference (unverified, not ARKit)"
)
BEACH_SCALE = 2.232937
CAP_USD = 5.0
# Same startup + teardown + refused-redelivery allowance as bench ledger.py. Fixed
# CPU/RAM limits prevent exceeding the rate through an oversized reservation.
OVERHEAD_SECONDS = 240
ACTIONS = {
    "reeval-c935-rehearsal": ("l4", 360),
    "main-c935": ("l40s", 5700),
    "main-beach": ("l4", 900),
}
DEPENDENCIES = {
    "reeval-c935-rehearsal": ("freeze", "rehearse-beach"),
    "main-c935": ("reeval-c935-rehearsal", "rehearse-beach"),
    "main-beach": ("main-c935", "rehearse-beach"),
}
DOWNLOADS = ("patches.csv", "fit.json", "provenance.json", "fit_error.png", "status.json")
# Pinned v17 recipe at 06aa56975; do not silently inherit later production defaults.
V17_TRAIN = {
    "init_from": "sfm",
    "iterations": 30000,
    "schedule_full_at": 60,
    "schedule_floor": 0.25,
    "schedule_growth": "sqrt",
    "schedule_max_scale": 2.0,
    "schedule_min_large_steps": 60000,
    "schedule_large_frames": 2036,
    "test_every": 0,
    "checkpoint_every": 5000,
    "streaming_export": True,
    "strategy": "mcmc",
    "opacity_reg": 0.002,
    "cap_max": 5626152,
    "gaussian_density": 0.1,
    "budget_mode": "auto",
    "budget_headroom": 2.5,
    "max_gpu_tier": "l40s",
    "budget_floor": 200000,
    "converge": False,
    "holdout_error": False,
    "ship_sh_degree": 3,
    "train_max_side": None,
    "blocks": 1,
    "pose_opt": False,
    "app_opt": False,
    "bilateral_grid": False,
    "antialiased": False,
    "packed": True,
    "batch_size": 1,
}


def action_name(action: str) -> str:
    """Preserve old claim paths for metadata downloads; remaining actions are single-use."""
    return f"{action}-r2" if action.startswith("rehearse") else action


def dependency_ready(action: str, result: dict[str, Any]) -> bool:
    return result.get("status") == "complete" or (
        action == "main-c935" and result.get("status") == "partial"
    )


def budget(spent_usd: float = SPENT_USD) -> dict[str, Any]:
    # Conservative baseline: prior freeze + failed rehearsals are already billed.
    # Always reserve ALL three remaining calls, even after one finishes. This avoids
    # double-allocating apparent savings or trusting a stale billing snapshot.
    if not math.isfinite(spent_usd) or spent_usd < SPENT_USD:
        raise ValueError("spent-usd must be finite and at least the recorded $0.5844")
    actions: dict[str, dict[str, Any]] = {
        name: {
            "tier": tier,
            "timeoutSeconds": seconds,
            "worstCaseUsd": (seconds + OVERHEAD_SECONDS) * hourly_rate(tier) / 3600,
        }
        for name, (tier, seconds) in ACTIONS.items()
    }
    remaining = sum(row["worstCaseUsd"] for row in actions.values())
    total = spent_usd + remaining
    if total > CAP_USD:
        raise ValueError("campaign exceeds $5 before any Modal import")
    return {
        "campaign": CAMPAIGN,
        "capUsd": CAP_USD,
        "spentUsd": spent_usd,
        "remainingWorstCaseUsd": remaining,
        "totalWorstCaseUsd": total,
        "overheadSecondsPerCommand": OVERHEAD_SECONDS,
        "commands": actions,
        "policy": "single attempt; no refunds/retries for failed or unknown outcomes",
    }


def training_plan(action: str, commit: str) -> dict[str, Any]:
    if action not in {"rehearse-c935", "main-c935"}:
        raise ValueError("only c935 actions train")
    params = dict(V17_TRAIN)
    if action.startswith("rehearse"):
        params.update(
            iterations=6000,
            schedule_scale=0.05,
            cap_max=200000,
            train_max_side=800,
            checkpoint_every=0,
            schedule_min_large_steps=0,
        )
    plan = {
        "runId": f"{CAMPAIGN}-{action_name(action)}",
        "scene": "c935-coverage",
        "benchCommit": commit,
        "schedule": "full",
        "tier": "l40s",
        "seed": 42,
        "variant": {"name": action, "overrides": {"train": params}},
        "trainParams": params,
        "keepSplat": True,
        "renders": 0,
        "rehearsal": False,
        "timeoutSeconds": ACTIONS.get(action, ("l40s", 480))[1] - 30,
        "calibrationEvaluationOnly": True,
    }

    validate_training_plan(plan, 633)
    return plan


def validate_training_plan(plan: dict[str, Any], registered: int) -> int:
    """Use the production trainer's CPU-only checks and argv builder on exact params."""
    import harmonics
    import training
    from experiments.bench.recipe import make_recipe

    params = plan["trainParams"]
    make_recipe(params, plan["tier"])
    harmonics.check_degree(params.get("ship_sh_degree"), name="ship_sh_degree")
    scale = training.frame_schedule(registered, params)
    batch = training.check_batch_size(params.get("batch_size", 1))
    steps = training.scaled_steps(int(params["iterations"]), scale)
    training.gsplat_argv(
        "python",
        Path("trainer.py"),
        Path("dataset"),
        Path("result"),
        strategy=params["strategy"],
        max_steps=params["iterations"],
        steps_scaler=scale,
        cap_max=params["cap_max"],
        opacity_reg=params["opacity_reg"],
        batch_size=batch,
        packed=params["packed"],
        antialiased=params["antialiased"],
    )
    return steps


def evaluation_argv(action: str, plan: dict[str, Any]) -> list[str]:
    label = "beach" if action == "main-beach" else "c935"
    scene = "c935-coverage" if label == "c935" else "beach-full"
    run = f"{CAMPAIGN}-{action_name(action)}" if label == "c935" else BEACH_RUN
    if action == "reeval-c935-rehearsal":
        run = REHEARSAL_RUN
    argv = [
        os.environ.get("GSPLAT_PYTHON", "/opt/trainer/bin/python"),
        "-m",
        "experiments.coverage_quality.bench_eval",
        "--run",
        f"/bench/runs/{run}",
        "--scene",
        f"/bench/scenes/{scene}",
        "--label",
        label,
        "--out",
        str(ROOT / action_name(action)),
        "--max-patches",
        "64",
    ]
    argv += ["--time-budget", str(plan.get("evaluationSeconds", 360))]
    if action.startswith(("rehearse", "reeval")):
        argv += ["--max-test-frames", "4"]
    if label == "c935":
        argv += [
            "--metres-per-unit",
            "1",
            "--scale-source",
            C935_SCALE_SOURCE,
        ]
    else:
        argv += [
            "--metres-per-unit",
            str(plan.get("metresPerUnit") or BEACH_SCALE),
            "--scale-source",
            str(plan.get("scaleSource") or BEACH_SCALE_SOURCE),
        ]
    return argv


def execute(action: str, plan: dict[str, Any], volume: Any) -> dict[str, Any]:
    """Bounded repair/main call; reuse completed freeze without any R2 secret."""
    from experiments.bench.worker import claim, run_training

    budget(float(plan["spentUsd"]))
    began = time.monotonic()
    volume.reload()
    destination = ROOT / action_name(action)
    for dependency in DEPENDENCIES[action]:
        if not dependency_ready(
            dependency, read_json(ROOT / action_name(dependency) / "status.json")
        ):
            raise ValueError(f"server prerequisite not complete: {dependency}")
    claim(destination, attempt_started=time.time())
    volume.commit()  # A redelivered input can never repeat train/freeze/evaluation.
    result: dict[str, Any] = {"action": action, "status": "failed"}
    try:
        if action == "main-c935":
            scene = Path("/bench/scenes/c935-coverage")
            manifest, split = verify(scene)
            run_plan = training_plan(action, plan["commit"])
            validate_training_plan(run_plan, len(split["pool"]))
            run_plan.update(
                manifestDigest=manifest["digest"],
                splitDigest=split["digest"],
                evaluationTestNames=choose_test_views(split["test"], 4),
            )
            run = Path("/bench/runs") / run_plan["runId"]
            claim(run, attempt_started=time.time())
            volume.commit()
            trained = run_training(scene, run, run_plan)
            if trained["status"] != "complete":
                raise RuntimeError(f"bench train failed: {trained.get('error')}")
            volume.commit()  # Retain private model if calibration later times out.
        remaining = ACTIONS[action][1] - (time.monotonic() - began) - 20
        if remaining <= 0:
            raise TimeoutError("no calibration time remains")
        plan["evaluationSeconds"] = max(30, int(remaining) - 15)
        env = {
            **os.environ,
            "PYTHONPATH": "/opt/pipeline",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        subprocess.run(  # noqa: S603 - fixed repository-built argv, no shell
            evaluation_argv(action, plan), env=env, check=True, timeout=remaining
        )
        provenance = read_json(destination / "provenance.json")
        result["status"] = provenance.get("status", "complete")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic() - began
    result["worstCaseUsd"] = budget()["commands"][action]["worstCaseUsd"]
    write_json(destination / "status.json", result)
    volume.commit()
    return result


def build_app() -> tuple[Any, Any, dict[str, Any]]:
    import modal

    from modal_adapter import GPU_NAMES

    # SDK 1.6.2 public metadata lookup: no function invocation, no image build. Runtime
    # mounts overlay only experiment code, preserving the existing trainer/CUDA image.
    info = modal.Function.from_name("twin-pipeline", "run_stage_l40s").info()
    image_id = info.image_info.image_id
    if not image_id:
        raise ValueError("no prebuilt training image ID; campaign refuses an unpriced rebuild")
    repo = Path(__file__).resolve().parents[4]
    image = modal.Image.from_id(image_id).add_local_dir(
        repo / "tools/pipeline/experiments",
        "/opt/pipeline/experiments",
        copy=False,
        ignore=["**/__pycache__/**"],
    )
    app = modal.App(CAMPAIGN)
    volume = modal.Volume.from_name("hexapod-bench", create_if_missing=False)
    functions = {}
    for action, (tier, timeout) in ACTIONS.items():

        def invoke(plan: dict[str, Any], action: str = action) -> dict[str, Any]:
            return execute(action, plan, volume)

        gpu = {} if tier == "cpu4" else {"gpu": GPU_NAMES[tier]}
        functions[action] = app.function(
            image=image,
            **gpu,
            env={
                "PYTHONPATH": "/opt/pipeline",
                "COVERAGE_IMAGE_ID": image_id,
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            },
            cpu=(4, 4) if tier == "cpu4" else (2, 2),
            memory=(8192, 8192) if tier == "cpu4" else (16384, 16384),
            timeout=timeout,
            startup_timeout=60,
            retries=0,
            max_containers=1,
            single_use_containers=True,
            serialized=True,
            volumes={"/bench": volume},
            name=action,
            secrets=[],
        )(invoke)
    return app, volume, functions


def reserve_local(output: Path, action: str, spent_usd: float = SPENT_USD) -> None:
    output.mkdir(parents=True, exist_ok=True)
    book = output / "campaign.jsonl"
    budget(spent_usd)  # Includes prior spend and all three remaining reservations.
    with locked(book) as stream:
        entries = [json.loads(line) for line in book.read_text().splitlines()]
        started = {row["action"] for row in entries}
        if action_name(action) in started:
            raise ValueError(
                "action already attempted; use download, never retry unknown paid work"
            )
        for dependency in DEPENDENCIES[action]:
            status = output / action_name(dependency) / "status.json"
            if not status.is_file() or not dependency_ready(dependency, read_json(status)):
                raise ValueError(f"required rehearsal/step not complete: {dependency}")
        stream.write(
            json.dumps(
                {
                    "action": action_name(action),
                    "spentBaselineUsd": spent_usd,
                    "reservedUsd": budget(spent_usd)["commands"][action]["worstCaseUsd"],
                }
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def download(volume: Any, action: str, output: Path) -> None:
    destination = output / action_name(action)
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in DOWNLOADS:
        relative = str((ROOT / action_name(action) / name).relative_to("/bench"))
        try:
            blocks = volume.read_file(relative)
            content = bytearray()
            for block in blocks:
                total += len(block)
                if total > 32 * 1024 * 1024:
                    raise ValueError("derived download exceeds 32 MiB; no private files downloaded")
                content.extend(block)
        except FileNotFoundError:
            continue
        (destination / name).write_bytes(content)


def pool(output: Path) -> None:
    from .fitting import plot_fit, pooled_fit, read_rows

    rows = [
        r
        for action in ("main-c935", "main-beach")
        for r in read_rows(output / action / "patches.csv")
    ]
    directory = output / "pooled"
    directory.mkdir(parents=True, exist_ok=True)
    results = {target: pooled_fit(rows, target) for target in ("error", "l1", "lpips")}
    write_json(directory / "fit.json", results)
    for scene in ("c935", "beach"):
        target = directory / scene
        target.mkdir(exist_ok=True)
        plot_fit(
            [r for r in rows if r["scene"] == scene],
            {"error": results["error"]["pooled"]},
            target,
        )
    write_json(
        directory / "provenance.json",
        {
            "campaign": CAMPAIGN,
            "inputs": ["main-c935/patches.csv", "main-beach/patches.csv"],
            "privateDataDownloaded": False,
            "inputProvenance": {
                action: read_json(output / action / "provenance.json")
                for action in ("main-c935", "main-beach")
                if (output / action / "provenance.json").is_file()
            },
            "scaleCaution": "Beach patch sizes and mm/px use an unverified camera-height estimate",
            "sceneIndicator": "intercept only",
            "transfer": "all parameters retuned without held scene; no test intercept fitting",
        },
    )


def upload_pooled(output: Path) -> None:
    """Upload ONLY the four locally fitted derived reports; no paid Modal function."""
    import modal

    volume = modal.Volume.from_name("hexapod-bench", create_if_missing=False)
    files = ("fit.json", "provenance.json", "c935/fit_error.png", "beach/fit_error.png")
    with volume.batch_upload(force=True) as batch:
        for relative in files:
            path = output / "pooled" / relative
            if path.is_file():
                batch.put_file(path, str((ROOT / "pooled" / relative).relative_to("/bench")))


def parser() -> argparse.ArgumentParser:
    plan = budget()
    costs = "; ".join(f"{n}: <=${r['worstCaseUsd']:.6f}" for n, r in plan["commands"].items())
    cli = argparse.ArgumentParser(
        description=__doc__, epilog=f"{costs}; TOTAL <=${plan['totalWorstCaseUsd']:.6f}"
    )
    cli.add_argument(
        "--spent-usd",
        type=float,
        default=SPENT_USD,
        help="pre-continuation billed spend, >=0.5844; reserves all three remaining calls",
    )
    cli.add_argument("command", choices=("plan", *ACTIONS, "download", "pool"))
    cli.add_argument("--local-out", type=Path, required=True)
    cli.add_argument(
        "--action",
        choices=(*ACTIONS, "freeze", "rehearse-c935", "rehearse-beach"),
        help="download only: zero paid functions",
    )
    cli.add_argument(
        "--metres-per-unit",
        type=float,
        help="model metres/unit conversion; Beach default is scale-uncertain 2.232937",
    )
    cli.add_argument(
        "--scale-source",
        help="conversion source, including uncertainty; never implies personal verification",
    )
    return cli


def main() -> None:
    args = parser().parse_args()
    report = budget(args.spent_usd)
    report["trainingPlans"] = {
        action: training_plan(action, "local-validation") for action in ("main-c935",)
    }
    for plan in report["trainingPlans"].values():
        plan["validatedSteps"] = validate_training_plan(plan, 633)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    sys.stdout.flush()
    output = args.local_out.expanduser().resolve()
    if args.command == "plan":
        return
    if args.command == "pool":
        pool(output)
        upload_pooled(output)
        return
    if args.command == "download":
        if not args.action:
            raise ValueError("download requires --action")
        import modal

        download(modal.Volume.from_name("hexapod-bench"), args.action, output)
        return
    if (args.metres_per_unit is None) != (args.scale_source is None):
        raise ValueError("provide metric conversion and source together, or neither")
    app, volume, functions = build_app()
    from experiments.bench.__main__ import git_commit

    payload = {
        "commit": git_commit(),
        "spentUsd": args.spent_usd,
        "metresPerUnit": args.metres_per_unit,
        "scaleSource": args.scale_source,
    }
    with app.run():
        marker = str((ROOT / action_name(args.command) / "started.json").relative_to("/bench"))
        try:
            # Read metadata only, BEFORE creating a paid function call.
            b"".join(volume.read_file(marker))
        except FileNotFoundError:
            pass
        else:
            raise ValueError("server action already claimed; download derived outputs instead")
        reserve_local(output, args.command, args.spent_usd)
        result = functions[args.command].remote(payload)
        download(volume, args.command, output)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if result["status"] != "complete" and not (
        args.command.startswith("main-") and result["status"] == "partial"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
