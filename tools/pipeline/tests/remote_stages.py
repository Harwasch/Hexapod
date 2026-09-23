"""Stage implementations for the remote tests, in a module `run_stage` can import.

Separate from `test_remote.py` for the reason `cloud_stages.py` is separate from
`test_cloud.py`, and it is not a style choice: `run_stage` imports every name in
`implModules`, so stages defined in a test file get registered once by pytest (as
`tests.test_remote`) and again by `run_stage` (as `test_remote`), and the second
registration raises `DuplicateImplError`. A module imported by one bare name is imported
once -- which is also exactly how a deployment ships its own stages.
"""

from __future__ import annotations

import json
import threading

from artifacts import ArtifactDecl
from contracts import StageContext, StageOutcome
from registry import stage_impl

SLOW = ArtifactDecl("slow.json", content_type="application/json")
READS = ArtifactDecl("read.json", content_type="application/json")

#: Held closed by the test that needs to observe an *interval* checkpoint sync, and left
#: open otherwise. An event rather than a sleep, so the test is a rendezvous and not a
#: race on a loaded machine.
GATE = threading.Event()
GATE.set()


@stage_impl("t_remote_slow", produces=(SLOW,), summary="checkpoints, then waits to be synced")
def t_remote_slow(ctx: StageContext) -> StageOutcome:
    """Writes a checkpoint and blocks until the test lets it finish.

    A stage that returned immediately would only ever see the final sync, so the thing
    worth testing -- that something beside the stage copies `checkpoint/` out while it is
    still running -- would never happen.
    """
    (ctx.checkpoint_dir / "progress.json").write_text(
        json.dumps({"iteration": 7}), encoding="utf-8"
    )
    GATE.wait(timeout=30.0)
    ctx.output(SLOW.name).write_text(json.dumps({"done": True}), encoding="utf-8")
    return StageOutcome(metrics={"iterations": 7}, summary="slow")


@stage_impl("t_remote_reads", consumes=("seed",), produces=(READS,), summary="echoes its input")
def t_remote_reads(ctx: StageContext) -> StageOutcome:
    """Reads one input and writes no checkpoint, which is what makes it useful twice."""
    text = ctx.input("seed").read_text(encoding="utf-8")
    ctx.output(READS.name).write_text(json.dumps({"seed": text}), encoding="utf-8")
    return StageOutcome(metrics={"chars": len(text)}, summary="echoed")


TIMED = ArtifactDecl("timed.json", content_type="application/json")


@stage_impl("t_remote_times_out", produces=(TIMED,), summary="its own code times out")
def t_remote_times_out(ctx: StageContext) -> StageOutcome:
    raise TimeoutError("read timed out")
