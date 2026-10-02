"""What happens to a remote call when the process watching it stops, and what it may cost.

A remote call is somebody else's machine. It does not stop because the worker polling it
did, so each way the worker can stop gets its own answer (`cloud` module docstring):

* a **cancel** (the job was cancelled, the lease was lost) cancels the call and enters
  what it billed in the ledger;
* a **detach** (a deploy) leaves it running, and the next runner **re-attaches** to it
  from the `CallBook` instead of submitting it again -- and does not pay for it twice;
* a **crash** leaves the book as it was, and the next attempt re-attaches the same way.

The stop arrives as an exception raised in the runner's own loop: the injected `sleep`
raises it, which is where a signal handler's exception lands in the real thing (the
worker's `app/worker/child.py`; `apps/api/tests/test_worker_stops.py` drives those).

Also here: each attempt's keys are its own, the run's dollar cap, and the three ways a
call runs out of time, none of which the worker retries.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from test_fanout import pieces

from adapters import FakeAdapter, LocalTransfer, RemoteExecution
from artifacts import ArtifactDecl
from cloud import (
    CALL_BOOK,
    Attempt,
    AttemptLedger,
    CallBook,
    CallRecord,
    CloudRunner,
    Placement,
    Poll,
    Reattachable,
    RemoteHandle,
    StageRequest,
)
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from errors import (
    CancelRequested,
    CostCapError,
    DetachRequested,
    RemoteStageError,
    RemoteTimeoutError,
)
from executor import execute
from providers import Rate
from recipe import Recipe
from registry import stage_impl
from runners import LocalRunner, RunnerSet
from workdir import Workdir

OUTPUT = ArtifactDecl("out.json", content_type="application/json")
#: A dollar a second, so a cost in an assertion is a number of polls.
DOLLAR_A_SECOND = Rate(3600.0, "test: a dollar a second")


@stage_impl("t_far", produces=(OUTPUT,), summary="a GPU stage that runs elsewhere")
def t_far(ctx: StageContext) -> StageOutcome:
    raise AssertionError("t_far runs on the provider, never here")


def recipe(stage_id: str = "train") -> Recipe:
    return make_recipe(
        [{"id": stage_id, "impl": "t_far", "gpu": {"tier": "a100", "preemptible": True}}],
        inputs=[],
    )


def works(units: int) -> Callable[[RemoteExecution], Iterator[None]]:
    """A remote that does `units` of work, checkpointing each, and writes its output."""

    def script(run: RemoteExecution) -> Iterator[None]:
        for done in range(units):
            (run.checkpoint_dir / "progress.json").write_text(json.dumps({"done": done + 1}))
            run.log(f"unit {done + 1} of {units}")
            yield
        run.output(OUTPUT.name).write_text(
            json.dumps({"attempt": run.request.attempt, "outputsKey": run.request.outputs_key})
        )

    return script


class Provider(FakeAdapter):
    """A `FakeAdapter` whose calls live with the provider, as Modal's do: a runner in a new
    process -- here, a new `CloudRunner` -- can pick one up by its id."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reattached: list[str] = []
        self.cancelled: list[str] = []

    def reattach(
        self, handle: RemoteHandle, request: StageRequest | None, submitted_at: float
    ) -> None:
        if handle.id not in self._runs:
            raise LookupError(f"no call {handle.id}")
        self.reattached.append(handle.id)

    def cancel(self, handle: RemoteHandle) -> None:
        self.cancelled.append(handle.id)
        super().cancel(handle)


def provider(tmp_path: Path, units: int = 4, **kwargs: Any) -> Provider:
    transfer = LocalTransfer(tmp_path / "bucket")
    return Provider(
        transfer,
        tmp_path / "sandbox",
        name="modal",
        interruptible=False,
        rates={"a100": DOLLAR_A_SECOND, "l4": DOLLAR_A_SECOND},
        script=works(units),
        **kwargs,
    )


def runner(
    adapter: Any, tmp_path: Path, *, sleep: Callable[[float], None] | None = None, **kw: Any
) -> CloudRunner:
    return CloudRunner(
        Placement((adapter,)),
        LocalTransfer(tmp_path / "bucket"),
        poll_interval_s=0.0,
        checkpoint_every_s=0.0,
        sleep=sleep or (lambda _seconds: None),
        **kw,
    )


def stops(after: int, error: BaseException) -> Callable[[float], None]:
    """A `sleep` that raises `error` on its `after`-th call: a signal, landing mid-watch."""
    calls = [0]

    def sleep(_seconds: float) -> None:
        calls[0] += 1
        if calls[0] >= after:
            raise error

    return sleep


def run(workdir: Workdir, cloud: CloudRunner, attempt: int = 1, stage: str = "train") -> Any:
    return execute(
        recipe(stage),
        workdir,
        RunnerSet(cpu=LocalRunner(), gpu=cloud),
        attempts={stage: attempt},
    )


def book_of(workdir: Workdir, stage: str = "train") -> CallBook:
    return CallBook.read(workdir.stage_dir(stage) / CALL_BOOK)


def ledger_of(workdir: Workdir, stage: str = "train") -> AttemptLedger:
    return AttemptLedger.read(workdir.attempts_path(stage))


# --- a detach leaves the call running, and the next runner picks it up --------------


def test_a_detached_call_is_left_running_and_the_next_runner_re_attaches_to_it(
    tmp_path: Path,
) -> None:
    """The deploy case. The stage is half done on the provider when the worker goes away;
    the next worker polls the same call to the end instead of staging and paying again."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    assert isinstance(adapter, Reattachable)

    with pytest.raises(DetachRequested):
        run(workdir, runner(adapter, tmp_path, sleep=stops(2, DetachRequested())))

    book = book_of(workdir)
    assert book.detached
    (call,) = book.calls.values()
    assert adapter.cancelled == [], "a detach must not cancel"
    assert call.attempt == 1 and call.request is not None
    assert not ledger_of(workdir).entries, "nothing is paid for until it ends"

    result = run(workdir, runner(adapter, tmp_path))

    assert len(adapter.submitted) == 1, "the call was not submitted a second time"
    assert adapter.reattached == [call.handle.id]
    (entry,) = ledger_of(workdir).entries
    assert entry.call == call.handle.id and entry.state == "succeeded"
    assert entry.billed_s == pytest.approx(5.0), "one call's seconds, polled by two runners"
    assert not (workdir.stage_dir("train") / CALL_BOOK).exists(), "nothing is out there now"
    produced = json.loads((workdir.out_dir("train") / "out.json").read_text())
    assert produced["attempt"] == 1
    step = result.steps[0]
    assert step.attempt == 1
    assert step.metrics["billedS"] == pytest.approx(5.0)
    assert "re-attached to call" in workdir.log_path("train").read_text()


def test_a_call_that_cannot_be_re_attached_is_cancelled_on_a_detach(tmp_path: Path) -> None:
    """A provider without `reattach` -- a subprocess of the process going away -- has
    nothing for the next worker to pick up, so a detach is a cancel for it."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    transfer = LocalTransfer(tmp_path / "bucket")
    plain = FakeAdapter(
        transfer,
        tmp_path / "sandbox",
        interruptible=False,
        script=works(4),
        rates={"a100": DOLLAR_A_SECOND},
    )
    assert not isinstance(plain, Reattachable)

    with pytest.raises(DetachRequested):
        run(workdir, runner(plain, tmp_path, sleep=stops(2, DetachRequested())))

    assert not (workdir.stage_dir("train") / CALL_BOOK).exists()
    (entry,) = ledger_of(workdir).entries
    assert entry.state == "failed" and entry.detail.startswith("cancelled:")


# --- a cancel stops paying, and says what it had cost --------------------------------


@pytest.mark.parametrize("stop", [CancelRequested(), KeyboardInterrupt()])
def test_a_cancel_cancels_the_call_and_records_what_it_billed(
    tmp_path: Path, stop: BaseException
) -> None:
    """The cancel case, and anything else that stops the watch: the call is cancelled,
    its cost so far is in the ledger -- a cancelled run still paid for its GPU -- and it
    is struck from the book."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)

    with pytest.raises(type(stop)):
        run(workdir, runner(adapter, tmp_path, sleep=stops(2, stop)))

    (handle_id,) = adapter.cancelled
    (entry,) = ledger_of(workdir).entries
    assert entry.call == handle_id
    assert entry.state == "failed" and entry.detail.startswith("cancelled:")
    assert entry.billed_s == pytest.approx(2.0)
    assert entry.usd == pytest.approx(2.0)
    assert ledger_of(workdir).preemptions == 0, "a cancel is not a preemption"
    assert not (workdir.stage_dir("train") / CALL_BOOK).exists()


# --- a crash leaves the book, and the next attempt adopts the call -------------------


def test_after_a_crash_the_next_attempt_adopts_the_call_on_its_own_keys(tmp_path: Path) -> None:
    """A SIGKILLed worker runs nothing on the way out: the book is as it was, not marked
    detached, and the worker counts the next run as attempt 2. That attempt still picks
    up attempt 1's call rather than starting a parallel one, and reads its outputs and
    checkpoint from the keys *that call* writes to."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    with pytest.raises(DetachRequested):
        run(workdir, runner(adapter, tmp_path, sleep=stops(2, DetachRequested())))
    book = book_of(workdir)
    book.detached = False  # what a crash leaves: the book, unmarked
    book.save()

    result = run(workdir, runner(adapter, tmp_path), attempt=2)

    assert len(adapter.submitted) == 1
    produced = json.loads((workdir.out_dir("train") / "out.json").read_text())
    assert produced == {"attempt": 1, "outputsKey": "runs/run/train/transfer/out"}
    (entry,) = ledger_of(workdir).entries
    assert entry.attempt == 2, "entered under the attempt that saw it end"
    # The checkpoint came home from attempt 1's key: four units of work, done once.
    progress = json.loads((workdir.checkpoint_dir("train") / "progress.json").read_text())
    assert progress == {"done": 4}
    assert result.steps[0].attempt == 2


def test_a_call_already_in_the_ledger_is_not_paid_for_twice(tmp_path: Path) -> None:
    """Killed after it wrote the ledger and before it struck the call from the book: the
    next runner re-attaches to a call that is already paid for, and the ledger keeps the
    one entry it has for it."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    with pytest.raises(DetachRequested):
        run(workdir, runner(adapter, tmp_path, sleep=stops(2, DetachRequested())))
    (call,) = book_of(workdir).calls.values()
    already = Attempt(
        attempt=1,
        provider="modal",
        tier="a100",
        state="succeeded",
        billed_s=7.0,
        usd=7.0,
        call=call.handle.id,
    )
    AttemptLedger((already,)).write(workdir.attempts_path("train"))

    result = run(workdir, runner(adapter, tmp_path))

    assert ledger_of(workdir).entries == (already,)
    assert result.steps[0].metrics["costUsd"] == pytest.approx(7.0)


def test_a_call_of_this_same_attempt_that_cannot_be_picked_up_fails_the_attempt(
    tmp_path: Path,
) -> None:
    """A new call of the same attempt would share its keys with one that may still be
    running, so the attempt fails, and the next one starts clean on keys of its own."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    request = _request("runs/run/train/checkpoint", "runs/run/train/transfer/out", attempt=1)
    ghost = CallRecord(RemoteHandle("gone", "elsewhere", "a100"), 1.0, 1, request)
    CallBook(workdir.stage_dir("train") / CALL_BOOK, calls={"": ghost}).save()

    with pytest.raises(RemoteStageError, match="could not be picked up"):
        run(workdir, runner(adapter, tmp_path))
    assert adapter.submitted == []
    assert not (workdir.stage_dir("train") / CALL_BOOK).exists()

    CallBook(workdir.stage_dir("train") / CALL_BOOK, calls={"": ghost}).save()
    run(workdir, runner(adapter, tmp_path), attempt=2)
    (fresh,) = adapter.submitted
    assert fresh.checkpoint_key == "runs/run/train/checkpoint-a2"
    assert fresh.outputs_key == "runs/run/train/transfer/out-a2"


# --- calls nothing is going to resume are cancelled before anything runs ------------


def test_reap_cancels_every_call_but_the_ones_the_run_resumes(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    handles = {
        stage: adapter.submit(_request(f"runs/run/{stage}/checkpoint", "x", attempt=1))
        for stage in ("pose", "train", "late")
    }
    for stage, handle in handles.items():
        CallBook(
            workdir.stage_dir(stage) / CALL_BOOK,
            calls={"": CallRecord(handle, 1.0, 1, None)},
        ).save()
    cloud = runner(adapter, tmp_path)

    cancelled = cloud.reap(workdir, keep="train")

    assert sorted(cancelled) == sorted([handles["pose"].id, handles["late"].id])
    assert book_of(workdir, "train").calls, "the stage about to run keeps its call"
    assert not (workdir.stage_dir("pose") / CALL_BOOK).exists()
    assert "left by an earlier process" in workdir.log_path("pose").read_text()

    # A book known only from the database -- its workdir lost -- is never resumed.
    orphaned = book_of(workdir, "train")
    orphaned.orphaned = True
    orphaned.save()
    assert cloud.reap(workdir, keep="train") == [handles["train"].id]


# --- each attempt writes to keys of its own ------------------------------------------


def test_an_orphaned_call_cannot_write_over_the_next_attempt(tmp_path: Path) -> None:
    """A call nobody stopped -- its record lost with the disk -- finishes late and uploads
    its `out/`, just before the next attempt's outputs are fetched. On a shared key that
    upload would be what the next attempt took home; on keys per attempt it is not."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path)
    orphan = adapter.submit(
        _request("runs/run/train/checkpoint", "runs/run/train/transfer/out", attempt=1)
    )

    class Racing(LocalTransfer):
        def get(self, key: str, target: Path) -> int:
            if "/transfer/out" in key:
                while adapter.poll(orphan).state == "running":
                    pass  # the orphan finishes and uploads, at the worst moment
            return super().get(key, target)

    cloud = CloudRunner(
        Placement((adapter,)),
        Racing(tmp_path / "bucket"),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
    )
    run(workdir, cloud, attempt=2)

    produced = json.loads((workdir.out_dir("train") / "out.json").read_text())
    assert produced["attempt"] == 2
    assert (tmp_path / "bucket" / "runs/run/train/transfer/out" / "out.json").is_file()


# --- the dollar cap -----------------------------------------------------------------


def test_no_call_is_submitted_once_the_run_has_spent_its_cap(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    workdir.stage_dir("pose").mkdir(parents=True)
    spent = Attempt(
        attempt=1, provider="modal", tier="cpu4", state="succeeded", billed_s=1.0, usd=12.0
    )
    AttemptLedger((spent,)).write(workdir.attempts_path("pose"))
    adapter = provider(tmp_path)

    with pytest.raises(CostCapError, match=r"\$12.00 against a cap of \$10.00"):
        run(workdir, runner(adapter, tmp_path, cost_cap_usd=10.0))
    assert adapter.submitted == []


def test_a_call_whose_running_cost_would_pass_the_cap_is_cancelled(tmp_path: Path) -> None:
    """A dollar a second against a $2.50 cap: the third second is the one that would go
    over, so the call is cancelled there -- and what it cost is still entered."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    adapter = provider(tmp_path, units=20)

    with pytest.raises(CostCapError, match="was cancelled"):
        run(workdir, runner(adapter, tmp_path, cost_cap_usd=2.5))

    assert len(adapter.cancelled) == 1
    (entry,) = ledger_of(workdir).entries
    assert entry.usd == pytest.approx(3.0) and entry.detail.startswith("cancelled:")
    assert not (workdir.stage_dir("train") / CALL_BOOK).exists()


def test_the_pieces_of_a_fan_out_are_priced_together_against_the_cap(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = Provider(
        transfer,
        tmp_path / "sandbox",
        name="modal",
        interruptible=False,
        rates={"l4": DOLLAR_A_SECOND},
        script=pieces(["a", "b"], 2, {"a": 9, "b": 9}),
    )
    fanned = make_recipe(
        [{"id": "train", "impl": "t_fanned", "gpu": {"tier": "l4", "preemptible": True}}],
        inputs=[],
    )

    with pytest.raises(CostCapError, match="part"):
        execute(
            fanned,
            workdir,
            RunnerSet(cpu=LocalRunner(), gpu=runner(adapter, tmp_path, cost_cap_usd=6.0)),
        )

    assert len(adapter.cancelled) == 2, "both pieces stop, not just one"
    parts = [e for e in ledger_of(workdir).entries if e.part in ("a", "b")]
    assert {e.part for e in parts} == {"a", "b"}
    assert all(e.detail.startswith("cancelled:") for e in parts)


# --- a detach in the middle of a fan-out --------------------------------------------


def test_a_fan_out_is_picked_up_where_the_last_process_left_it(tmp_path: Path) -> None:
    """Three pieces, two at a time. `a` has finished when the worker goes; `b` and `c` are
    running. The next runner asks the head nothing, runs `a` not at all, re-attaches to
    `b` and `c`, and joins."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = Provider(
        transfer,
        tmp_path / "sandbox",
        name="modal",
        interruptible=False,
        rates={"l4": DOLLAR_A_SECOND},
        script=pieces(["a", "b", "c"], 2, {"a": 1, "b": 6, "c": 6}),
    )
    fanned = make_recipe(
        [{"id": "train", "impl": "t_fanned", "gpu": {"tier": "l4", "preemptible": True}}],
        inputs=[],
    )
    with pytest.raises(DetachRequested):
        execute(
            fanned,
            workdir,
            RunnerSet(
                cpu=LocalRunner(), gpu=runner(adapter, tmp_path, sleep=stops(4, DetachRequested()))
            ),
        )
    book = book_of(workdir)
    assert book.detached and book.fan_out is not None
    assert book.parts_done == ["a"] and sorted(book.calls) == ["b", "c"]
    submitted = len(adapter.submitted)

    execute(fanned, workdir, RunnerSet(cpu=LocalRunner(), gpu=runner(adapter, tmp_path)))

    roles = [r.params.get("fanout", {}).get("role") for r in adapter.submitted[submitted:]]
    assert roles == ["join"], "nothing but the join was submitted again"
    assert sorted(adapter.reattached) == sorted(c.handle.id for c in book.calls.values())
    merged = json.loads((workdir.out_dir("train") / "merged.json").read_text())
    assert merged == {"merged": ["a", "b", "c"]}
    assert not (workdir.stage_dir("train") / CALL_BOOK).exists()


# --- running out of time ------------------------------------------------------------


class Ends:
    """An adapter whose one call ends as told, with its last log line arriving only after
    the verdict -- as a traceback can reach a provider's log after the call's result."""

    name = "modal"
    interruptible = False

    def __init__(self, poll: Poll, last_line: str = "") -> None:
        self._poll = poll
        self._last = last_line
        self._polled = False

    def rate(self, tier: str) -> Rate | None:
        return None

    def submit(self, request: StageRequest) -> RemoteHandle:
        return RemoteHandle("only", self.name, request.tier)

    def poll(self, handle: RemoteHandle) -> Poll:
        self._polled = True
        return self._poll

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        lines = [self._last] if self._polled and self._last else []
        return lines[since:]

    def cancel(self, handle: RemoteHandle) -> None:
        pass


def test_a_call_that_outran_its_function_s_limit_is_a_timeout(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    ended = Ends(Poll(state="failed", detail="FunctionTimeoutError: 6h", timed_out=True))

    with pytest.raises(RemoteTimeoutError, match="FunctionTimeoutError"):
        run(workdir, runner(ended, tmp_path))


def test_a_failure_s_last_log_lines_are_read_after_its_verdict(tmp_path: Path) -> None:
    """What the worker reads to tell an out-of-memory from any other failure."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    ended = Ends(
        Poll(state="failed", detail="CalledProcessError: exit 1"),
        last_line="torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    )

    with pytest.raises(RemoteStageError) as raised:
        run(workdir, runner(ended, tmp_path))
    assert not isinstance(raised.value, RemoteTimeoutError)
    assert "CUDA out of memory" in workdir.log_path("train").read_text()


def test_max_wait_is_a_timeout(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    clock = [0.0]

    def tick(_seconds: float) -> None:
        clock[0] += 100.0

    adapter = provider(tmp_path, units=50)
    cloud = runner(adapter, tmp_path, sleep=tick, clock=lambda: clock[0], max_wait_s=500.0)

    with pytest.raises(RemoteTimeoutError, match="still running after 500s"):
        run(workdir, cloud)


def hangs(run: RemoteExecution) -> Iterator[None]:
    """A trainer that reports a tenth of its steps in ten seconds, then never again."""
    run.log("loss=0.1| :  10%|#  | 10/100 [00:10<01:30,  1.00it/s]")
    while True:
        yield


def keeps_going(run: RemoteExecution) -> Iterator[None]:
    """A trainer that slows down as it goes -- its projection grows -- and finishes."""
    for step in range(1, 41):
        elapsed = step * step  # each step slower than the last
        remaining = max(1, (40 - step) * step * 2)
        run.log(
            f"loss=0.1| : {step}/40 [{elapsed // 60:02d}:{elapsed % 60:02d}<"
            f"{remaining // 60:02d}:{remaining % 60:02d},  1.00it/s]"
        )
        yield
    run.output(OUTPUT.name).write_text("{}")


def test_a_trainer_that_stops_reporting_is_cancelled_as_overdue(tmp_path: Path) -> None:
    """Its progress projected 100 s; at twice that plus the slack, it is overdue -- hours
    before the provider's own six-hour limit would have stopped it."""
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    clock = [0.0]

    def tick(_seconds: float) -> None:
        clock[0] += 10.0

    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = Provider(
        transfer, tmp_path / "sandbox", name="modal", interruptible=False, script=hangs
    )
    cloud = runner(
        adapter,
        tmp_path,
        sleep=tick,
        clock=lambda: clock[0],
        deadline_factor=2.0,
        deadline_slack_s=60.0,
    )

    with pytest.raises(RemoteTimeoutError, match="overdue"):
        run(workdir, cloud)
    assert clock[0] <= 2.0 * 100.0 + 60.0 + 20.0


def test_a_trainer_that_is_slow_but_reporting_is_left_alone(tmp_path: Path) -> None:
    workdir = seeded_workdir(tmp_path / "run", upload=False)
    clock = [0.0]

    def tick(_seconds: float) -> None:
        clock[0] += 60.0

    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = Provider(
        transfer, tmp_path / "sandbox", name="modal", interruptible=False, script=keeps_going
    )
    cloud = runner(
        adapter,
        tmp_path,
        sleep=tick,
        clock=lambda: clock[0],
        deadline_factor=2.0,
        deadline_slack_s=0.0,
    )

    run(workdir, cloud)


def _request(checkpoint: str, outputs: str, *, attempt: int) -> StageRequest:
    return StageRequest(
        recipe="test",
        run_id="run",
        stage_id="train",
        impl="t_far",
        attempt=attempt,
        tier="a100",
        preemptible=True,
        params={},
        inputs={},
        produces=(OUTPUT,),
        checkpoint_key=checkpoint,
        outputs_key=outputs,
    )
