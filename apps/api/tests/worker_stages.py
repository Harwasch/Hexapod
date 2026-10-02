"""Stage implementations that exist only for the worker's tests.

They are registered the way any implementation is — a decorator — and the worker imports
this module by name (`WORKER_IMPL_MODULES=tests.worker_stages`), which is the same seam a
deployment with its own stages would use. Nothing here is imported by `app`.

Importing `app.worker.pipeline_bridge` first is what puts `tools/pipeline` on the path;
every bare import below comes from there.
"""

from __future__ import annotations

import json
import os
import signal
import time

from app.worker.pipeline_bridge import ensure_importable

ensure_importable()

from artifacts import ArtifactDecl  # noqa: E402
from contracts import StageContext, StageOutcome  # noqa: E402
from registry import stage_impl  # noqa: E402
from stages import CANONICAL_PLY, SPLAT_TILES  # noqa: E402

FIRST = ArtifactDecl("first.json", content_type="application/json")
SECOND = ArtifactDecl("second.json", content_type="application/json")
THIRD = ArtifactDecl("third.json", content_type="application/json")
SLOW = ArtifactDecl("slow.json", content_type="application/json")
RESUMED = ArtifactDecl("resumed.json", content_type="application/json")
DOOMED = ArtifactDecl("doomed.json", content_type="application/json")
TRAINED = ArtifactDecl("trained.json", content_type="application/json")


def _write(ctx: StageContext, name: str, **fields: object) -> None:
    ctx.output(name).write_text(
        json.dumps({"stage": ctx.stage_id, "attempt": ctx.attempt, **fields}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _quick(name: str, decl: ArtifactDecl) -> None:
    @stage_impl(name, produces=(decl,), summary=f"test stage {name}")
    def _impl(ctx: StageContext) -> StageOutcome:
        # A run counter in the workdir, so a test can prove a skipped stage really was
        # not re-run rather than inferring it from timings.
        # Long enough that a watcher polling the table sees the run part-finished; this
        # is what the "step rows appear as stages run" test turns on.
        time.sleep(float(ctx.param("delay", 0.0)))
        counter = ctx.work_dir / "runs.txt"
        runs = len(counter.read_text().splitlines()) if counter.exists() else 0
        counter.write_text("x\n" * (runs + 1), encoding="utf-8")
        ctx.log(f"{ctx.stage_id}: run {runs + 1}")
        _write(ctx, decl.name, runs=runs + 1, pid=os.getpid())
        return StageOutcome(metrics={"runs": runs + 1})


for _name, _decl in (("t_first", FIRST), ("t_second", SECOND), ("t_third", THIRD)):
    _quick(_name, _decl)


@stage_impl("t_slow", produces=(SLOW,), summary="sleeps, so a test can interrupt it mid-stage")
def t_slow(ctx: StageContext) -> StageOutcome:
    seconds = float(ctx.param("seconds", 1.0))
    ctx.log(f"sleeping {seconds}s")
    time.sleep(seconds)
    _write(ctx, SLOW.name, seconds=seconds)
    return StageOutcome(metrics={"seconds": seconds})


@stage_impl("t_fails", produces=(DOOMED,), summary="always raises")
def t_fails(ctx: StageContext) -> StageOutcome:
    ctx.log(f"attempt {ctx.attempt} is going to fail")
    raise RuntimeError("this stage is broken on purpose")


@stage_impl("t_resumes", produces=(RESUMED,), summary="fails once, then resumes from checkpoint")
def t_resumes(ctx: StageContext) -> StageOutcome:
    """The preemption contract, exercised for real.

    The first attempt leaves a marker in `checkpoint/` and dies. A6 keeps `checkpoint/`
    across attempts and wipes `out/`, so the second attempt finds the marker and finishes
    — which is what `job_steps.attempt` and `checkpoint_key` exist to record.
    """
    marker = ctx.checkpoint_dir / "tried"
    if not marker.exists():
        marker.write_text("1", encoding="utf-8")
        raise RuntimeError("killed before it could finish")
    _write(ctx, RESUMED.name, resumedFromCheckpoint=True)
    return StageOutcome(metrics={"resumed": True})


@stage_impl("t_normalize", produces=(CANONICAL_PLY,), summary="a stand-in for A8's ingest_splat")
def t_normalize(ctx: StageContext) -> StageOutcome:
    ctx.output(CANONICAL_PLY.name).write_bytes(b"ply\nformat binary_little_endian 1.0\n")
    return StageOutcome(metrics={"gaussians": 3})


@stage_impl(
    "t_package",
    consumes=("canonical.ply", "georef.json"),
    produces=(SPLAT_TILES,),
    summary="a stand-in for A8's splat_tiles, writing the members that artifact declares",
)
def t_package(ctx: StageContext) -> StageOutcome:
    out = ctx.output(SPLAT_TILES.name)
    (out / "tileset.json").write_text(json.dumps({"asset": {"version": "1.1"}}), encoding="utf-8")
    (out / "splat.glb").write_bytes(b"glTF-ish bytes")
    (out / "collision.bin").write_bytes(b"\x1f\x8b collision-ish bytes")
    (out / "viewcones.bin").write_bytes(b"\x1f\x8b view-cone-ish bytes")
    return StageOutcome(metrics={"tiles": 1})


@stage_impl(
    "t_gpu_train",
    produces=(TRAINED,),
    summary="a GPU stage that checkpoints, is preempted once, and finishes on the next try",
)
def t_gpu_train(ctx: StageContext) -> StageOutcome:
    """The cloud path's stand-in for B2's trainer.

    It counts, writing `checkpoint/progress.json` after every iteration, and at `die_at`
    it signals itself away — which is what a provider reclaiming a container is. The
    adapter's checkpoint syncer has already copied `checkpoint/` out by then, so the next
    attempt resumes from that iteration rather than from zero.
    """
    path = ctx.checkpoint_dir / "progress.json"
    done = int(json.loads(path.read_text(encoding="utf-8"))["iteration"]) if path.is_file() else 0
    target = int(ctx.param("iterations", 6))
    die_at = int(ctx.param("die_at", 3))
    died = ctx.checkpoint_dir / "died-once"
    ctx.log(f"training from {done} to {target} in pid {os.getpid()}")
    while done < target:
        done += 1
        path.write_text(json.dumps({"iteration": done}), encoding="utf-8")
        if done == die_at and not died.exists():
            died.write_text("1", encoding="utf-8")
            ctx.log("the provider is taking the machine back")
            time.sleep(float(ctx.param("sync_grace_s", 1.0)))
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5.0)  # SIGTERM ends the process well before this
    _write(ctx, TRAINED.name, iterations=done, resumed=died.exists())
    return StageOutcome(metrics={"iterations": done}, summary=f"trained to {done}")


# --- for tests/test_worker_stops.py ----------------------------------------------------

STOPPED = ArtifactDecl("stopped.json", content_type="application/json")


@stage_impl("t_gpu_slow", produces=(TRAINED,), summary="a remote stage that takes its time")
def t_gpu_slow(ctx: StageContext) -> StageOutcome:
    """Runs in the fake Modal container (`tests/fake_modal`): `iterations` steps of `step_s`
    each, checkpointing as it goes, and says which process did the work -- so a test can
    tell one call re-attached to from two calls."""
    target = int(ctx.param("iterations", 10))
    for done in range(1, target + 1):
        time.sleep(float(ctx.param("step_s", 0.5)))
        (ctx.checkpoint_dir / "progress.json").write_text(json.dumps({"done": done}))
        ctx.log(f"step {done} of {target}")
    _write(ctx, TRAINED.name, iterations=target, pid=os.getpid())
    return StageOutcome(metrics={"iterations": target})


@stage_impl("t_notes_stop", produces=(STOPPED,), summary="says how it was stopped, then stops")
def t_notes_stop(ctx: StageContext) -> StageOutcome:
    """Sleeps, and if it is stopped writes down what it was stopped *with*: the recipe
    process turns the supervisor's two signals into two exceptions."""
    try:
        time.sleep(float(ctx.param("seconds", 30.0)))
    except BaseException as error:
        (ctx.work_dir / "stopped-by.txt").write_text(type(error).__name__, encoding="utf-8")
        raise
    _write(ctx, STOPPED.name)
    return StageOutcome(metrics={})


@stage_impl("t_oom", produces=(TRAINED,), summary="runs out of GPU memory above a cap")
def t_oom(ctx: StageContext) -> StageOutcome:
    """Logs the training stage's budget line and, above `fits` gaussians, the traceback a
    CUDA out-of-memory leaves -- the two things the worker reads to retry it lower."""
    cap = int(ctx.param("cap_max", 1_000_000))
    ctx.log(f"gsplat: cap_max auto -> {cap}; 3.10M footprints^2 of surface x 0.3 = {cap}")
    if cap > int(ctx.param("fits", 0)):
        ctx.log("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        raise RuntimeError("the trainer exited with status 1")
    _write(ctx, TRAINED.name, cap=cap)
    return StageOutcome(metrics={"capMax": cap})


@stage_impl("t_timeout", produces=(TRAINED,), summary="runs out of time")
def t_timeout(ctx: StageContext) -> StageOutcome:
    from errors import RemoteTimeoutError

    raise RemoteTimeoutError(ctx.recipe, ctx.stage_id, ctx.impl, "modal", "FunctionTimeoutError")


@stage_impl("t_contract", produces=(TRAINED,), summary="asks for an input it never declared")
def t_contract(ctx: StageContext) -> StageOutcome:
    ctx.input("poses")
    raise AssertionError("unreachable: the input is undeclared")


@stage_impl("t_dies", produces=(TRAINED,), summary="dies without a word on its stdout")
def t_dies(ctx: StageContext) -> StageOutcome:
    """What a segfault or an import error in a native library looks like from outside: a
    few words on stderr and the process gone, no event reported."""
    import sys

    sys.stderr.write("fatal: the native trainer could not map its weights\n")
    sys.stderr.flush()
    os._exit(9)
