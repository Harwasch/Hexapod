"""The worker's half of the cloud seam, end to end.

`tools/pipeline/tests/test_cloud_preemption.py` proves the pipeline resumes rather than
restarts. This file proves the worker around it does the rest: a GPU stage really is
dispatched, a preemption really is recorded as one rather than as a failure, and the run
comes out of it with a provider, a tier and a cost on `jobs` instead of three null
columns.

Nothing is mocked. The supervisor starts a real child process, which builds a real
`CloudRunner` and dispatches the stage to `SubprocessAdapter` -- a second real process,
which is killed by a real SIGTERM halfway through and resumed on the next attempt.

The end-to-end tests move bytes through a shared directory rather than the bucket, for a
reason worth stating: moto's S3 exists only inside *this* process, and the worker's child
is a different one, so a bucket the child could reach would have to be a real MinIO. The
`ObjectStoreTransfer` itself is covered against moto in the first test here; it is the
same `Transfer` protocol on both sides of that line, which is the point of it being one.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models import Job
from app.models.enums import RunStatus
from app.storage import S3Storage
from app.worker.claim import claim_next
from app.worker.cloud import ObjectStoreTransfer, build_runners
from app.worker.config import WorkerConfig
from app.worker.pipeline_bridge import PipelineError
from app.worker.runner import JobSupervisor
from tests.test_worker import config, make_capture, queue_job, steps_by_stage

#: Chosen so the arithmetic is obvious in an assertion: a dollar a second.
DOLLAR_A_SECOND = "subprocess:a100=3600.0"


def cloud_config(tmp_path: Path, **overrides: object) -> WorkerConfig:
    defaults: dict[str, object] = {
        "runner": "cloud",
        "cloud_providers": ("subprocess",),
        "cloud_poll_s": 0.05,
        "checkpoint_every_s": 0.05,
        "max_attempts": 3,
        # The bytes go through a directory rather than the bucket. moto lives in this
        # process and the child is a different one, so a bucket the child could reach
        # would have to be a real MinIO; `ObjectStoreTransfer` is covered against moto
        # directly above, and this is the same `Transfer` protocol either way.
        "cloud_transfer_dir": tmp_path / "shared",
    }
    defaults.update(overrides)
    return config(tmp_path, "worker-a", **defaults)


# --- the transfer -------------------------------------------------------------------


def test_the_transfer_round_trips_through_the_bucket(
    storage: S3Storage,
    tmp_path: Path,
) -> None:
    """A directory's members are objects under the key and a file is an object at it --
    the same encoding `LocalTransfer` uses, so a stage cannot behave differently here."""
    transfer = ObjectStoreTransfer(storage)
    source = tmp_path / "checkpoint"
    (source / "shard").mkdir(parents=True)
    (source / "shard" / "0.bin").write_bytes(b"abcd")
    (source / "progress.json").write_text('{"iteration": 4}', encoding="utf-8")
    single = tmp_path / "canonical.ply"
    single.write_bytes(b"ply-bytes")

    assert transfer.put("runs/j/train/checkpoint", source) == 20
    assert transfer.put("runs/j/transfer/inputs/upload", single) == 9
    assert transfer.exists("runs/j/train/checkpoint")
    assert transfer.exists("runs/j/transfer/inputs/upload")
    assert not transfer.exists("runs/j/train/nothing")

    back = tmp_path / "back"
    assert transfer.get("runs/j/train/checkpoint", back) == 20
    assert json.loads((back / "progress.json").read_text()) == {"iteration": 4}
    assert (back / "shard" / "0.bin").read_bytes() == b"abcd"
    file_back = tmp_path / "restored" / "canonical.ply"
    assert transfer.get("runs/j/transfer/inputs/upload", file_back) == 9
    assert file_back.read_bytes() == b"ply-bytes"

    # A key with nothing under it is a first attempt with no checkpoint, not an error.
    assert transfer.get("runs/j/train/nothing", tmp_path / "empty") == 0
    transfer.delete("runs/j/train/checkpoint")
    assert not transfer.exists("runs/j/train/checkpoint")


# --- placement ----------------------------------------------------------------------


def test_a_placement_whose_fallback_is_itself_interruptible_is_refused(
    storage: S3Storage,
    tmp_path: Path,
) -> None:
    """The last provider is the one a preempted stage falls back *to*. If it can be
    taken away as well there is nothing to fall back to, and the cheap tier is retried
    until the budget is gone -- which is how it stops being the cheap tier."""
    with pytest.raises(PipelineError, match="must not itself be interruptible"):
        build_runners(storage, providers=("subprocess", "fake"), sandbox=tmp_path / "sandbox")


def test_an_unknown_provider_is_refused_by_name(
    storage: S3Storage,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown cloud provider 'runpod-community'"):
        build_runners(storage, providers=("runpod-community",), sandbox=tmp_path / "sandbox")


def test_no_providers_configured_leaves_gpu_stages_with_nowhere_to_go(
    storage: S3Storage,
    tmp_path: Path,
) -> None:
    """The honest default: `cloud` with no provider list is `local`, and a `gpu:` stage
    fails at planning naming the tier it wanted rather than running somewhere unasked."""
    runners = build_runners(storage, providers=(), sandbox=tmp_path / "sandbox")
    assert runners.gpu is None


# --- the whole path -----------------------------------------------------------------


def run_job(
    sessions: sessionmaker[Session],
    storage: S3Storage,
    job_id: uuid.UUID,
    worker_config: WorkerConfig,
) -> str:
    return JobSupervisor(sessions, storage, worker_config).run(job_id)


def test_a_gpu_stage_is_dispatched_preempted_and_resumed_and_the_job_says_what_it_cost(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One job, two attempts at its GPU stage, and three columns that stop being null.

    The stage counts to six, is signalled away at three, and the second attempt starts
    at three -- which is only true because `CloudRunner` brought the checkpoint home from
    the bucket and pushed it back out. `one` is a CPU stage and runs in the worker's own
    process throughout: `gpu:` is still the only routing signal there is.
    """
    monkeypatch.setenv("PIPELINE_GPU_RATES", DOLLAR_A_SECOND)
    capture = make_capture(db)
    job = queue_job(db, capture, "t-gpu")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, cloud_config(tmp_path)) == "complete"

    steps = steps_by_stage(db, job.id)
    # The CPU stage ran once, here. The GPU stage took two attempts, on the provider.
    assert steps["one"].attempt == 1
    assert steps["train"].attempt == 2
    assert steps["train"].status is RunStatus.COMPLETE
    assert steps["train"].metrics["provider"] == "subprocess"
    assert steps["train"].metrics["tier"] == "a100"
    assert steps["train"].metrics["preemptions"] == 1
    assert steps["train"].checkpoint_key == f"runs/{job.id}/train/checkpoint"

    workdir = tmp_path / "runs" / str(job.id)
    trained = json.loads((workdir / "stages" / "train" / "out" / "trained.json").read_text())
    # Six iterations, reached on attempt 2, having resumed rather than started over.
    assert trained["iterations"] == 6
    assert trained["resumed"] is True
    assert trained["attempt"] == 2
    assert (
        json.loads((workdir / "stages" / "train" / "checkpoint" / "progress.json").read_text())[
            "iteration"
        ]
        == 6
    )

    db.expire_all()
    finished = db.get(Job, job.id)
    assert finished is not None
    assert finished.status is RunStatus.COMPLETE
    assert finished.provider == "subprocess"
    assert finished.tier == "a100"
    # At a dollar a second, the cost is the billed seconds -- and it includes the
    # attempt that was taken away, which is the whole point of recording it per attempt.
    assert finished.cost_usd is not None
    stage_seconds = float(steps["train"].metrics["stageBilledS"])
    assert float(finished.cost_usd) == pytest.approx(stage_seconds, abs=0.001)
    assert float(steps["train"].metrics["billedS"]) < stage_seconds


def test_a_preempted_attempt_is_marked_preempted_rather_than_merely_failed(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
) -> None:
    """`preempted_at` records the provider taking the machine back, and nothing else.

    A stage that simply fails twice leaves it null (see `test_worker.py`); this one is
    interrupted, so it is set -- and the step names the checkpoint the next attempt
    resumed from, which is the pair A2 added the columns for.
    """
    capture = make_capture(db)
    job = queue_job(db, capture, "t-gpu")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()

    assert run_job(sessions, storage, job.id, cloud_config(tmp_path)) == "complete"

    steps = steps_by_stage(db, job.id)
    assert steps["train"].preempted_at is not None
    assert steps["one"].preempted_at is None
    # With no rate for this host anywhere, the run still says where it ran and how long
    # for; it just has no price to multiply by. That is the honest answer, not a zero.
    db.expire_all()
    finished = db.get(Job, job.id)
    assert finished is not None
    assert finished.provider == "subprocess"
    assert finished.cost_usd is None
