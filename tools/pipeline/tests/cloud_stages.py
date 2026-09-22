"""Stage implementations for the cloud tests, importable by name from a child process.

`SubprocessAdapter` launches `run_stage.py` with `implModules`, and that process imports
this module to find the implementations -- exactly the seam a deployment uses for its own
stages. Nothing here is imported by the pipeline itself.
"""

from __future__ import annotations

import json
import os
import signal
import time

from artifacts import ArtifactDecl
from contracts import StageContext, StageOutcome
from registry import stage_impl

COUNTED = ArtifactDecl("counted.json", content_type="application/json")
SURVIVED = ArtifactDecl("survived.json", content_type="application/json")
BROKEN = ArtifactDecl("broken.json", content_type="application/json")


def _progress(ctx: StageContext) -> int:
    path = ctx.checkpoint_dir / "progress.json"
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["iteration"])


def _advance(ctx: StageContext, iteration: int) -> None:
    (ctx.checkpoint_dir / "progress.json").write_text(
        json.dumps({"iteration": iteration}), encoding="utf-8"
    )


@stage_impl("t_cloud_counts", produces=(COUNTED,), summary="counts, checkpointing as it goes")
def t_cloud_counts(ctx: StageContext) -> StageOutcome:
    done = _progress(ctx)
    target = int(ctx.param("iterations", 4))
    ctx.log(f"counting from {done} to {target} in pid {os.getpid()}")
    while done < target:
        done += 1
        _advance(ctx, done)
    ctx.output(COUNTED.name).write_text(
        json.dumps({"iterations": done, "pid": os.getpid()}), encoding="utf-8"
    )
    return StageOutcome(metrics={"iterations": done}, summary=f"counted to {done}")


@stage_impl(
    "t_cloud_dies",
    produces=(SURVIVED,),
    summary="checkpoints, is killed by a signal, and finishes on the next attempt",
)
def t_cloud_dies(ctx: StageContext) -> StageOutcome:
    """The real version of being preempted: a process that is signalled away mid-stage.

    It counts to `die_at`, leaves that in `checkpoint/`, waits long enough for the
    adapter's checkpoint syncer to have copied it out, and then SIGTERMs itself. A
    container reclaimed by a provider goes the same way, and the next attempt picks up
    from whatever the last sync captured.
    """
    done = _progress(ctx)
    target = int(ctx.param("iterations", 6))
    die_at = int(ctx.param("die_at", 3))
    marker = ctx.checkpoint_dir / "died-once"
    while done < target:
        done += 1
        _advance(ctx, done)
        if done == die_at and not marker.exists():
            marker.write_text("1", encoding="utf-8")
            ctx.log(f"reached {done}; the machine is being taken back")
            # Long enough that the syncer beside this process has copied `checkpoint/`
            # out several times over. It is a margin, not a race: the syncer runs every
            # 50 ms in the test that uses this.
            time.sleep(float(ctx.param("sync_grace_s", 1.0)))
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5.0)  # unreachable in practice; SIGTERM ends the process
    ctx.output(SURVIVED.name).write_text(
        json.dumps({"iterations": done, "resumed": marker.exists()}), encoding="utf-8"
    )
    return StageOutcome(metrics={"iterations": done})


@stage_impl("t_cloud_breaks", produces=(BROKEN,), summary="raises, on the provider")
def t_cloud_breaks(ctx: StageContext) -> StageOutcome:
    raise RuntimeError("the trainer ran out of memory")
