"""The cloud seam: the transfer, the placement, the cost, and a second adapter.

`test_cloud_preemption.py` holds the one test this step is really for. This file holds
the rest of the contract -- that a failure is not a preemption, that a stage which keeps
losing its cheap box moves to the reliable one, that what is not watched is cancelled
rather than left billing, and that a run's cost adds up across stages and attempts.

The `SubprocessAdapter` tests at the end are the ones that prove the protocol is real:
the same five methods, a registered implementation running in a genuine process, and a
genuine signal standing in for a provider reclaiming a machine.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from adapters import FakeAdapter, LocalTransfer, RemoteExecution, SubprocessAdapter
from artifacts import ArtifactDecl
from cloud import AttemptLedger, CloudRunner, Placement, RemoteHandle, StageKeys, run_cost
from cloud_stages import BROKEN, COUNTED, SURVIVED
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from errors import NoRunnerError, PreemptedError, RemoteStageError
from executor import execute
from providers import PROVIDERS, Rate, provider, rates_from_env, with_rates
from recipe import Recipe
from registry import stage_impl
from runners import RunnerSet
from workdir import Workdir

OUTPUT = ArtifactDecl("out.json", content_type="application/json")
PIPELINE_DIR = Path(__file__).resolve().parents[1]


@stage_impl("t_gpu", produces=(OUTPUT,), summary="a GPU stage that runs elsewhere")
def t_gpu(ctx: StageContext) -> StageOutcome:
    raise AssertionError("t_gpu runs on the provider, never here")


def counts(run: RemoteExecution) -> Iterator[None]:
    """A remote that does four units of work and writes whatever the stage declares."""
    for _ in range(4):
        yield
    for decl in run.request.produces:
        run.output(decl.name).write_text(
            json.dumps({"tier": run.request.tier, "attempt": run.request.attempt}),
            encoding="utf-8",
        )


def gpu_recipe(stage_id: str = "train", tier: str = "a100") -> Recipe:
    return make_recipe(
        [{"id": stage_id, "impl": "t_gpu", "gpu": {"tier": tier, "preemptible": True}}], inputs=[]
    )


def runner(placement: Placement, transfer: LocalTransfer) -> CloudRunner:
    return CloudRunner(
        placement,
        transfer,
        poll_interval_s=0.0,
        checkpoint_every_s=1.0,
        sleep=lambda _seconds: None,
    )


# --- the transfer -------------------------------------------------------------------


def test_the_transfer_round_trips_a_file_and_a_directory(tmp_path: Path) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "a.bin").write_bytes(b"12345")
    (source / "b.txt").write_text("hello", encoding="utf-8")
    single = tmp_path / "one.ply"
    single.write_bytes(b"ply")

    assert transfer.put("runs/r/dir", source) == 10
    assert transfer.put("runs/r/file", single) == 3
    assert transfer.exists("runs/r/dir") and transfer.exists("runs/r/file")

    back = tmp_path / "back"
    assert transfer.get("runs/r/dir", back) == 10
    assert (back / "nested" / "a.bin").read_bytes() == b"12345"
    assert (back / "b.txt").read_text() == "hello"
    file_back = tmp_path / "again" / "one.ply"
    assert transfer.get("runs/r/file", file_back) == 3
    assert file_back.read_bytes() == b"ply"

    # A key that holds nothing is a first attempt with no checkpoint, not an error.
    assert transfer.get("runs/r/nothing", tmp_path / "empty") == 0
    transfer.delete("runs/r/dir")
    assert not transfer.exists("runs/r/dir")


def test_the_keys_all_hang_off_the_checkpoint_key_the_workdir_already_defined() -> None:
    workdir = Workdir(Path("/tmp/runs/job-7"))
    context = StageContext(
        recipe="r",
        run_id="job-7",
        stage_id="train",
        impl="t_gpu",
        params={},
        attempt=1,
        gpu_tier="a100",
        out_dir=workdir.out_dir("train"),
        work_dir=workdir.work_dir("train"),
        checkpoint_dir=workdir.checkpoint_dir("train"),
        log_path=workdir.log_path("train"),
        checkpoint_key="runs/job-7/train/checkpoint",
        attempts_path=workdir.attempts_path("train"),
        _inputs={},
        _produces={},
    )

    keys = StageKeys.of(context)

    assert keys.root == "runs/job-7"
    assert keys.stage == "runs/job-7/train"
    assert keys.checkpoint == "runs/job-7/train/checkpoint"
    assert keys.outputs == "runs/job-7/train/transfer/out"
    assert keys.input("stages/pose/out/poses") == "runs/job-7/transfer/stages/pose/out/poses"


def test_an_input_is_sent_once_however_many_stages_read_it(tmp_path: Path) -> None:
    """Two stages consuming the same artifact do not upload it twice."""
    transfer = LocalTransfer(tmp_path / "bucket")
    sent: list[str] = []
    original = transfer.put

    class Counting(LocalTransfer):
        def put(self, key: str, source: Path) -> int:
            sent.append(key)
            return original(key, source)

    counting = Counting(transfer.root)
    adapter = FakeAdapter(counting, tmp_path / "sandbox", script=counts)
    workdir = seeded_workdir(tmp_path / "run")
    recipe = make_recipe(
        [
            {"id": "one", "impl": "t_gpu_a", "gpu": {"tier": "a100"}},
            {"id": "two", "impl": "t_gpu_b", "gpu": {"tier": "a100"}},
        ],
        inputs=["upload"],
    )

    execute(recipe, workdir, RunnerSet.cloud(runner(Placement((adapter,)), counting)))

    uploads = [key for key in sent if key.endswith("inputs/upload")]
    assert len(uploads) == 1


@stage_impl("t_gpu_a", consumes=("upload",), produces=(ArtifactDecl("a.json"),))
def t_gpu_a(ctx: StageContext) -> StageOutcome:
    raise AssertionError("runs on the provider")


@stage_impl("t_gpu_b", consumes=("upload",), produces=(ArtifactDecl("b.json"),))
def t_gpu_b(ctx: StageContext) -> StageOutcome:
    raise AssertionError("runs on the provider")


# --- states -------------------------------------------------------------------------


def test_a_stage_that_fails_on_the_provider_is_a_failure_not_a_preemption(
    tmp_path: Path,
) -> None:
    """The distinction the whole protocol turns on: one of these resumes, one does not."""
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = FakeAdapter(
        transfer, tmp_path / "sandbox", script=counts, fail_with="CUDA out of memory"
    )
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(RemoteStageError, match="CUDA out of memory"):
        execute(gpu_recipe(), workdir, RunnerSet.cloud(runner(Placement((adapter,)), transfer)))

    assert AttemptLedger.read(workdir.attempts_path("train")).entries[0].state == "failed"
    assert AttemptLedger.read(workdir.attempts_path("train")).preemptions == 0


def test_a_run_nobody_is_watching_any_more_is_cancelled_rather_than_left_billing(
    tmp_path: Path,
) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")
    cancelled: list[str] = []

    class Interrupts(FakeAdapter):
        def poll(self, handle: RemoteHandle):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt("the job was cancelled")

        def cancel(self, handle: RemoteHandle) -> None:
            cancelled.append(handle.id)

    adapter = Interrupts(transfer, tmp_path / "sandbox", script=counts)
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(KeyboardInterrupt):
        execute(gpu_recipe(), workdir, RunnerSet.cloud(runner(Placement((adapter,)), transfer)))

    assert len(cancelled) == 1


def test_a_stage_that_never_starts_is_cancelled_and_reported_rather_than_waited_on(
    tmp_path: Path,
) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")

    class NeverStarts(FakeAdapter):
        def poll(self, handle: RemoteHandle):  # type: ignore[no-untyped-def]
            from cloud import Poll

            return Poll(state="pending", billed_s=0.0)

    adapter = NeverStarts(transfer, tmp_path / "sandbox", script=counts)
    ticks = iter([0.0, 1.0, 2.0, 100.0, 200.0, 300.0])
    waiting = CloudRunner(
        Placement((adapter,)),
        transfer,
        poll_interval_s=0.0,
        max_wait_s=10.0,
        clock=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(RemoteStageError, match="still pending"):
        execute(gpu_recipe(), workdir, RunnerSet.cloud(waiting))


# --- placement ----------------------------------------------------------------------


def test_a_stage_that_keeps_losing_the_cheap_box_moves_to_the_reliable_one(
    tmp_path: Path,
) -> None:
    """Two preemptions, then Modal. Without this the cheap tier is billed three times
    for the same work and the job still dead-letters."""
    transfer = LocalTransfer(tmp_path / "bucket")
    cheap = FakeAdapter(
        transfer,
        tmp_path / "cheap",
        name="vast",
        interruptible=True,
        script=counts,
        preempt_after_polls=2,
        rates={"a100": Rate(0.52, "test")},
    )
    reliable = FakeAdapter(
        transfer,
        tmp_path / "reliable",
        name="modal",
        interruptible=False,
        script=counts,
        rates={"a100": Rate(2.50, "test")},
    )
    placement = Placement.of(cheap, reliable, preemptions_before_fallback=2)
    cloud = runner(placement, transfer)
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    for number in (1, 2):
        with pytest.raises(PreemptedError):
            execute(gpu_recipe(), workdir, RunnerSet.cloud(cloud), attempts={"train": number})
    execute(gpu_recipe(), workdir, RunnerSet.cloud(cloud), attempts={"train": 3})

    assert [request.attempt for request in cheap.submitted] == [1, 2]
    assert [request.attempt for request in reliable.submitted] == [3]
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [entry.provider for entry in ledger.entries] == ["vast", "vast", "modal"]
    # The two lost attempts are in the bill. They were paid for.
    assert ledger.preemptions == 2
    # Two lost seconds on the cheap host and five on the reliable one -- the lost ones
    # are in the bill, because they were paid for.
    assert [entry.billed_s for entry in ledger.entries] == [1.0, 1.0, 5.0]
    assert ledger.usd == pytest.approx((2 * 0.52 * 1.0 + 2.50 * 5.0) / 3600.0, abs=5e-6)


def test_a_placement_refuses_to_fall_back_onto_another_interruptible_provider() -> None:
    class Fake:
        name = "vast"
        interruptible = True

    with pytest.raises(NoRunnerError, match="must not itself be interruptible"):
        Placement.of(Fake(), Fake())  # type: ignore[arg-type]

    with pytest.raises(NoRunnerError, match="at least one ProviderAdapter"):
        Placement.of()


def test_placement_only_moves_on_and_never_back() -> None:
    class Cheap:
        name = "cheap"
        interruptible = True

    class Steady:
        name = "steady"
        interruptible = False

    placement = Placement.of(Cheap(), Steady(), preemptions_before_fallback=2)  # type: ignore[arg-type]
    chosen = [placement.adapter_for(n).name for n in range(6)]
    assert chosen == ["cheap", "cheap", "steady", "steady", "steady", "steady"]


# --- cost ---------------------------------------------------------------------------


def test_a_tier_nobody_has_priced_records_the_seconds_and_no_cost(tmp_path: Path) -> None:
    """The honest answer for an unsurveyed tier. An invented rate would be worse than
    this, because a plausible number in a cost column is a number that gets believed."""
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = FakeAdapter(transfer, tmp_path / "sandbox", script=counts, rates={})
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    execute(
        gpu_recipe(tier="l4"), workdir, RunnerSet.cloud(runner(Placement((adapter,)), transfer))
    )

    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    # Five simulated seconds: four units of work plus the poll that found it finished.
    assert ledger.billed_s == 5.0
    assert ledger.usd is None
    assert ledger.unpriced_s == 5.0
    total = run_cost(workdir)
    assert total.usd is None and total.billed_s == 5.0 and not total.complete
    assert total.provider == "fake" and total.tier == "l4"


def test_a_runs_cost_is_every_stage_and_every_attempt_of_it(tmp_path: Path) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = FakeAdapter(
        transfer, tmp_path / "sandbox", script=counts, rates={"a100": Rate(3.60, "test")}
    )
    workdir = seeded_workdir(tmp_path / "run")
    recipe = make_recipe(
        [
            {"id": "one", "impl": "t_gpu_a", "gpu": {"tier": "a100"}},
            {"id": "two", "impl": "t_gpu_b", "gpu": {"tier": "a100"}},
        ],
        inputs=["upload"],
    )

    execute(recipe, workdir, RunnerSet.cloud(runner(Placement((adapter,)), transfer)))

    total = run_cost(workdir)
    # Two stages, five simulated seconds each, at a round $3.60/h = half a cent each.
    assert total.attempts == 2
    assert total.billed_s == 10.0
    assert total.usd == pytest.approx(2 * 3.60 * 5.0 / 3600.0, abs=2e-6)
    assert total.complete


def test_a_truncated_ledger_costs_history_and_not_the_next_attempt(tmp_path: Path) -> None:
    path = tmp_path / "attempts.json"
    path.write_text('{"attempts": [{"attem', encoding="utf-8")
    assert AttemptLedger.read(path).entries == ()


# --- the price table ----------------------------------------------------------------


def test_the_table_carries_only_what_was_surveyed() -> None:
    for entry in PROVIDERS:
        assert set(entry.rates) == {"a100"}, entry.name
        assert entry.rates["a100"].source == "A0 provider survey"
    modal = provider("modal")
    assert modal is not None
    assert modal.rate("l4") is None
    assert modal.usd_per_hour_a100 == 2.50
    assert provider("nobody") is None


def test_a_deployment_supplies_its_own_rates_without_editing_the_table() -> None:
    parsed = rates_from_env({"PIPELINE_GPU_RATES": "modal:l4=0.80, vast:a100=0.44, rubbish"})
    assert parsed["modal"]["l4"].usd_per_hour == 0.80
    assert parsed["vast"]["a100"].usd_per_hour == 0.44
    assert "rubbish" not in parsed

    merged = with_rates(parsed)
    modal = next(entry for entry in merged if entry.name == "modal")
    assert modal.rate("l4") is not None
    assert modal.rate("l4").usd_per_hour == 0.80  # type: ignore[union-attr]
    # The surveyed figure is still there, and still says where it came from.
    assert modal.rate("a100").source == "A0 provider survey"  # type: ignore[union-attr]
    # The originals are untouched: `with_rates` returns a copy.
    assert provider("modal").rate("l4") is None  # type: ignore[union-attr]


def test_a_malformed_rate_is_skipped_rather_than_raised_on() -> None:
    assert rates_from_env({"PIPELINE_GPU_RATES": "modal:l4=free,x:y=-1,=,:"}) == {}
    assert rates_from_env({}) == {}


# --- the second adapter: a real process ---------------------------------------------


def subprocess_cloud(tmp_path: Path, **overrides: object) -> tuple[CloudRunner, SubprocessAdapter]:
    transfer = LocalTransfer(tmp_path / "bucket")
    settings: dict[str, object] = {
        "impl_modules": ("tests.cloud_stages",),
        "rates": {"a100": Rate(1.00, "test")},
    }
    settings.update(overrides)
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox", **settings)  # type: ignore[arg-type]
    return (
        CloudRunner(Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05),
        adapter,
    )


def test_a_real_stage_runs_in_a_real_process_through_the_same_five_methods(
    tmp_path: Path,
) -> None:
    """No fake anywhere: a registered implementation, a subprocess, and the protocol."""
    cloud, _ = subprocess_cloud(tmp_path)
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe(
        [
            {
                "id": "count",
                "impl": "t_cloud_counts",
                "params": {"iterations": 5},
                "gpu": {"tier": "a100"},
            }
        ],
        inputs=[],
    )

    execute(recipe, workdir, RunnerSet.cloud(cloud))

    produced = json.loads((workdir.out_dir("count") / COUNTED.name).read_text())
    assert produced["iterations"] == 5
    assert produced["pid"] != os.getpid()
    step = json.loads(workdir.step_path("count").read_text())
    # The stage's own metrics came back through `poll`, alongside the runner's.
    assert step["metrics"]["iterations"] == 5
    assert step["metrics"]["provider"] == "subprocess"
    assert step["summary"] == "counted to 5"
    # The log the provider produced is the stage's log.
    assert "counting from 0 to 5" in workdir.log_path("count").read_text()


def test_a_process_killed_by_a_signal_is_preempted_and_the_next_attempt_continues(
    tmp_path: Path,
) -> None:
    """The realism gate for the preemption story: a genuine SIGTERM mid-stage.

    The stage counts to three, waits for the syncer beside it to have copied
    `checkpoint/` out, and is killed. `poll` sees a negative return code -- the same
    thing a reclaimed container is -- and reports `preempted`, not `failed`. The second
    attempt finds three iterations already done.
    """
    cloud, _ = subprocess_cloud(tmp_path)
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe(
        [
            {
                "id": "train",
                "impl": "t_cloud_dies",
                "params": {"iterations": 6, "die_at": 3, "sync_grace_s": 1.0},
                "gpu": {"tier": "a100", "preemptible": True},
            }
        ],
        inputs=[],
    )

    with pytest.raises(PreemptedError):
        execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 1})

    progress = workdir.checkpoint_dir("train") / "progress.json"
    assert json.loads(progress.read_text())["iteration"] == 3

    execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 2})

    survived = json.loads((workdir.out_dir("train") / SURVIVED.name).read_text())
    assert survived == {"iterations": 6, "resumed": True}
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [entry.state for entry in ledger.entries] == ["preempted", "succeeded"]
    assert ledger.usd is not None and ledger.usd > 0


def test_a_stage_that_raises_on_the_box_comes_back_as_a_failure_with_its_own_words(
    tmp_path: Path,
) -> None:
    cloud, _ = subprocess_cloud(tmp_path)
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    recipe = make_recipe(
        [{"id": "boom", "impl": "t_cloud_breaks", "gpu": {"tier": "a100"}}], inputs=[]
    )

    with pytest.raises(RemoteStageError, match="ran out of memory"):
        execute(recipe, workdir, RunnerSet.cloud(cloud))

    assert not (workdir.out_dir("boom") / BROKEN.name).exists()
