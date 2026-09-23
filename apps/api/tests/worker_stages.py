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
