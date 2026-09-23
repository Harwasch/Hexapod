"""The container's half, driven here.

`remote.execute` is what a Modal function calls, and the reason it takes a `Transfer`
rather than reaching for a bucket is so that this file can be the thing that proves it.
Everything below runs in this process against `LocalTransfer`: the fetch, the run, the
interval checkpoint sync, the upload, and what comes back.

The one thing it cannot prove is that Modal calls it correctly — that is
`infra/modal/app.py`, which nothing here can execute. The split exists so the untestable
part is as small as an image, a GPU and one call.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from remote_stages import GATE, READS, SLOW

import remote
from adapters import LocalTransfer
from artifacts import ArtifactDecl
from cloud import StageRequest, Transfer
from cloud_stages import BROKEN, COUNTED


def request_for(
    impl: str,
    *,
    produces: tuple[ArtifactDecl, ...],
    inputs: dict[str, str] | None = None,
    every_s: float = 60.0,
) -> StageRequest:
    return StageRequest(
        recipe="t",
        run_id="run-1",
        stage_id="train",
        impl=impl,
        attempt=1,
        tier="l4",
        preemptible=True,
        params={},
        inputs=inputs or {},
        produces=produces,
        checkpoint_key="runs/run-1/train/checkpoint",
        outputs_key="runs/run-1/train/transfer/out",
        checkpoint_every_s=every_s,
    )


def test_a_stage_runs_and_its_outputs_are_uploaded(tmp_path: Path) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")
    outcome = remote.execute(
        request_for("t_cloud_counts", produces=(COUNTED,)),
        transfer,
        tmp_path / "sandbox",
        impl_modules=("cloud_stages",),
    )
    assert outcome.metrics["iterations"] == 4
    assert outcome.summary == "counted to 4"
    assert outcome.uploaded_bytes > 0
    # Uploaded under the outputs key, by artifact name, which is what `CloudRunner` pulls
    # back and checks against what the stage declared it produces.
    assert (tmp_path / "bucket" / "runs/run-1/train/transfer/out" / COUNTED.name).is_file()


def test_inputs_are_fetched_by_artifact_name(tmp_path: Path) -> None:
    """The container fetches its own bytes; the key is the pipeline's, the name is the
    stage's, and `run_stage` resolves `ctx.input("seed")` to `inputs/seed`."""
    transfer = LocalTransfer(tmp_path / "bucket")
    source = tmp_path / "seed.txt"
    source.write_text("from the bucket", encoding="utf-8")
    transfer.put("runs/run-1/transfer/inputs/seed", source)

    outcome = remote.execute(
        request_for(
            "t_remote_reads",
            produces=(READS,),
            inputs={"seed": "runs/run-1/transfer/inputs/seed"},
        ),
        transfer,
        tmp_path / "sandbox",
        impl_modules=("remote_stages",),
    )
    assert outcome.fetched_bytes == len("from the bucket")
    assert outcome.metrics["chars"] == len("from the bucket")


def test_a_checkpoint_left_by_an_earlier_attempt_comes_back(tmp_path: Path) -> None:
    """Resume, which is the whole reason the checkpoint key is in the request.

    `t_cloud_counts` counts from whatever `checkpoint/progress.json` says, so seeding the
    bucket with iteration 3 and asking for 4 must do one iteration's work, not four.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "progress.json").write_text(json.dumps({"iteration": 3}), encoding="utf-8")
    transfer.put("runs/run-1/train/checkpoint", earlier)

    sandbox = tmp_path / "sandbox"
    remote.execute(
        request_for("t_cloud_counts", produces=(COUNTED,)),
        transfer,
        sandbox,
        impl_modules=("cloud_stages",),
    )
    assert json.loads((sandbox / "checkpoint" / "progress.json").read_text())["iteration"] == 4


def test_the_checkpoint_is_synced_while_the_stage_is_still_running(tmp_path: Path) -> None:
    """The point of a syncer rather than one upload at the end.

    A container that is taken back does not get to run its upload, so a checkpoint that
    only appears when the stage finishes is worth nothing to the attempt that is killed.
    This holds the stage open and asserts the checkpoint reached the bucket before it
    returned — which is only possible if something beside the stage put it there.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    synced = tmp_path / "bucket" / "runs/run-1/train/checkpoint" / "progress.json"
    GATE.clear()

    outcome: list[remote.RemoteOutcome] = []
    runner = threading.Thread(
        target=lambda: outcome.append(
            remote.execute(
                request_for("t_remote_slow", produces=(SLOW,), every_s=0.05),
                transfer,
                tmp_path / "sandbox",
                impl_modules=("remote_stages",),
            )
        ),
        daemon=True,
    )
    runner.start()
    try:
        for _ in range(200):  # up to 10s, and normally the first or second look
            if synced.is_file():
                break
            threading.Event().wait(0.05)
        assert synced.is_file(), "the checkpoint never reached the bucket while running"
        assert json.loads(synced.read_text())["iteration"] == 7
    finally:
        GATE.set()
        runner.join(timeout=15.0)
    assert not runner.is_alive()
    assert outcome and outcome[0].summary == "slow"


def test_a_failed_stage_raises_and_uploads_nothing(tmp_path: Path) -> None:
    """The exception is the contract: `poll` classifies what `get` re-raises.

    And `out/` stays where it is. A half-written artifact from a failed attempt would be
    wiped by the next attempt's `prepare_stage` anyway, after being paid for twice.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    with pytest.raises(RuntimeError, match="ran out of memory"):
        remote.execute(
            request_for("t_cloud_breaks", produces=(BROKEN,)),
            transfer,
            tmp_path / "sandbox",
            impl_modules=("cloud_stages",),
        )
    assert not (tmp_path / "bucket" / "runs/run-1/train/transfer/out").exists()


def test_an_empty_checkpoint_never_overwrites_a_good_one(tmp_path: Path) -> None:
    """The attempt that most needs its checkpoint is the one that would lose it.

    `t_remote_reads` writes no checkpoint at all, so the final sync has an empty
    directory in hand. Putting that would replace the earlier attempt's progress with
    nothing.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "keep.json").write_text(json.dumps({"iteration": 9}), encoding="utf-8")
    transfer.put("runs/run-1/train/checkpoint", earlier)
    source = tmp_path / "seed.txt"
    source.write_text("x", encoding="utf-8")
    transfer.put("runs/run-1/transfer/inputs/seed", source)

    # The stage reads its input and writes nothing into checkpoint/, but the fetch put
    # the earlier attempt's file there -- so this also pins that a resumed checkpoint is
    # re-uploaded intact rather than being treated as empty.
    remote.execute(
        request_for(
            "t_remote_reads", produces=(READS,), inputs={"seed": "runs/run-1/transfer/inputs/seed"}
        ),
        transfer,
        tmp_path / "sandbox",
        impl_modules=("remote_stages",),
    )
    kept = tmp_path / "bucket" / "runs/run-1/train/checkpoint" / "keep.json"
    assert json.loads(kept.read_text())["iteration"] == 9


def test_a_sync_that_throws_does_not_take_the_stage_down(tmp_path: Path) -> None:
    """A checkpoint is insurance. Losing the insurance must not lose the run."""

    class BrokenPut(LocalTransfer):
        def put(self, key: str, source: Path) -> int:
            if key.endswith("checkpoint"):
                raise OSError("the bucket said no")
            return super().put(key, source)

    transfer: Transfer = BrokenPut(tmp_path / "bucket")
    outcome = remote.execute(
        request_for("t_cloud_counts", produces=(COUNTED,), every_s=0.05),
        transfer,
        tmp_path / "sandbox",
        impl_modules=("cloud_stages",),
    )
    assert outcome.metrics["iterations"] == 4
    assert outcome.uploaded_bytes > 0


def test_the_outcome_survives_a_round_trip_as_json(tmp_path: Path) -> None:
    """It crosses a process boundary as a dict, so it has to be one."""
    transfer = LocalTransfer(tmp_path / "bucket")
    outcome = remote.execute(
        request_for("t_cloud_counts", produces=(COUNTED,)),
        transfer,
        tmp_path / "sandbox",
        impl_modules=("cloud_stages",),
    )
    document = json.loads(json.dumps(outcome.to_dict()))
    assert document["metrics"]["iterations"] == 4
    assert document["summary"] == "counted to 4"
    assert document["uploadedBytes"] > 0
