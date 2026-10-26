"""Ephemeral hexapod-bench app. Invoke via python -m experiments.bench; never deploy.

The shared training image retains its cached layers. The extra file is a runtime mount
so this module can import the image definition again inside a Modal container.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
SHARED_PATH = HERE / "app.py" if modal.is_local() else Path("/root/bench_training_app.py")
spec = importlib.util.spec_from_file_location("bench_training_app", SHARED_PATH)
assert spec is not None and spec.loader is not None
shared = importlib.util.module_from_spec(spec)
sys.modules["bench_training_app"] = shared
spec.loader.exec_module(shared)
sys.path.insert(0, str(shared.LOCAL_PIPELINE))

from experiments.bench.ledger import STARTUP_SECONDS, TIMEOUTS, hourly_rate  # noqa: E402
from experiments.bench.scene import write_json  # noqa: E402
from modal_adapter import GPU_NAMES  # noqa: E402

app = modal.App("hexapod-bench")
volume = modal.Volume.from_name("hexapod-bench", create_if_missing=True)
image = shared.image.add_local_file(SHARED_PATH, "/root/bench_training_app.py")
_calls = 0


def _billed_since() -> tuple[float, str]:
    from experiments.bench.worker import billed_from

    global _calls
    _calls += 1
    return billed_from(
        started_at=shared.STARTED_AT,
        booted_at=shared.BOOTED_AT,
        first_call=_calls == 1,
        now=time.time(),
    )


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=TIMEOUTS["freeze"],
    retries=0,
    volumes={"/bench": volume},
    secrets=[modal.Secret.from_name("twin-object-storage")],
    max_containers=1,
    startup_timeout=STARTUP_SECONDS,
    single_use_containers=True,
)
def freeze(plan: dict[str, Any]) -> dict[str, Any]:
    from experiments.bench.freeze import ReadOnlyR2, freeze_scene

    started, billed = _billed_since()
    volume.reload()
    transfer = shared._transfer()
    # Only the list/get capability crosses into freeze_scene; never S3Transfer itself.
    store = ReadOnlyR2(transfer._client, os.environ["OBJECT_STORAGE_BUCKET"])
    try:
        manifest = freeze_scene(
            store,
            Path("/bench/scenes"),
            plan["scene"],
            plan["sourceRun"],
            plan["benchCommit"],
            generation=plan.get("generation"),
            refreeze=plan.get("refreeze", False),
        )
        result = {
            "runId": plan["runId"],
            "scene": plan["scene"],
            "status": "complete",
            "manifest": manifest,
            "error": None,
        }
    except Exception as error:
        result = {
            "runId": plan["runId"],
            "scene": plan["scene"],
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    result["wallSeconds"] = time.time() - started
    result["billedFrom"] = billed
    result["costUsd"] = result["wallSeconds"] * hourly_rate("cpu4") / 3600
    write_json(Path("/bench/runs") / plan["runId"] / "result.json", result)
    volume.commit()
    return result


def _train(plan: dict[str, Any]) -> dict[str, Any]:
    from experiments.bench.worker import RestartedError, claim, restart_result, run_training

    started, billed = _billed_since()
    volume.reload()
    run = Path("/bench/runs") / plan["runId"]
    try:
        claim(run, attempt_started=started)
    except RestartedError as error:
        # Modal re-ran a preempted input. Train nothing; the ledger keeps the run's
        # worst-case reservation, which covers the lost attempt and this refusal.
        refused = restart_result(plan, error)
        write_json(run / f"preempted-{int(time.time())}.json", refused)
        volume.commit()
        return refused
    # Committed before training, so a restart on another container sees it.
    volume.commit()
    result = run_training(Path("/bench/scenes") / plan["scene"], run, plan, billed_since=started)
    volume.commit()
    # Commit latency is part of the container bill, and is returned to the local ledger.
    result["wallSeconds"] = time.time() - started
    result["billedFrom"] = billed
    result["costUsd"] = result["wallSeconds"] * hourly_rate(plan["tier"]) / 3600
    write_json(run / "result.json", result)
    volume.commit()
    return result


FUNCTIONS = {
    (tier, schedule): app.function(
        image=image,
        gpu=GPU_NAMES[tier],
        cpu=(shared.GPU_CPU_CORES, shared.GPU_CPU_CORES),
        memory=(shared.GPU_MEMORY_MIB, shared.GPU_MEMORY_MIB),
        timeout=TIMEOUTS[schedule],
        retries=0,
        volumes={"/bench": volume},
        name=f"train_{tier}_{schedule}",
        max_containers=4,
        startup_timeout=STARTUP_SECONDS,
        single_use_containers=True,
        # Deliberately NO object-storage secret on any GPU function.
    )(_train)
    for tier in GPU_NAMES
    for schedule in ("screen", "full")
    if tier in {"l4", "a10", "l40s", "a100"}
}
