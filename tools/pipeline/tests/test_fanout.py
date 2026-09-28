"""Fanning one stage out over several remote calls (`cloud.CloudRunner`, `contracts.FanOut`).

**Nothing here trains anything.** The stage is a script run by `FakeAdapter`, deterministic
and clockless, that plays the three roles a block run plays -- a head that plans pieces,
pieces that each leave a result in their own checkpoint, and a join that needs every
result -- so what is tested is the runner's side: how many pieces are in flight at once,
that a lost piece is retried alone while the others carry on, that a finished piece is
never run again, that every call is priced and summed, and that the join comes last and
sees everything. `test_blocks.py` runs the real train stage through the same seam.

Why the runner does this at all: the spool at `blocks: 2` trained its blocks one after
another on one L4 -- 2,530 s then 2,207 s, 2.3 h and $1.73 -- and Modal bills per
GPU-second, so the same blocks on two GPUs at once cost the same and end ~37 min sooner.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from adapters import FakeAdapter, LocalTransfer, RemoteExecution
from artifacts import ArtifactDecl
from cloud import (
    TERMINAL,
    AttemptLedger,
    CloudRunner,
    Placement,
    Poll,
    RemoteHandle,
    StageRequest,
    _FanOutLog,
    run_cost,
)
from conftest import make_recipe
from contracts import (
    FANOUT_METRIC,
    FANOUT_PARAM,
    FanOut,
    FanOutPart,
    StageContext,
    StageOutcome,
    fanout_role,
)
from errors import PreemptedError, RemoteStageError
from executor import execute
from providers import Rate
from recipe import Recipe
from registry import stage_impl
from runners import RunnerSet
from workdir import Workdir

OUTPUT = ArtifactDecl("merged.json", content_type="application/json")
RATE = Rate(0.80, "test: an L4 hour")


@stage_impl("t_fanned", produces=(OUTPUT,), summary="a stage that splits into pieces")
def t_fanned(ctx: StageContext) -> StageOutcome:
    raise AssertionError("t_fanned runs on the provider, never here")


def recipe() -> Recipe:
    return make_recipe(
        [{"id": "train", "impl": "t_fanned", "gpu": {"tier": "l4", "preemptible": True}}],
        inputs=[],
    )


def pieces(parts: Sequence[str], parallel: int, ticks: dict[str, int] | None = None) -> Any:
    """The scripted stage: a head that asks for `parts` minus the ones already in its
    checkpoint, pieces that work `ticks[id]` units and leave `results/<id>.json`, and a
    join that writes the output only if every piece's result reached it."""

    def script(run: RemoteExecution) -> Iterator[None]:
        role, part = fanout_role(run.request.params)
        checkpoint = run.checkpoint_dir
        if role == "head":
            (checkpoint / "shared").mkdir(exist_ok=True)
            (checkpoint / "shared" / "plan.json").write_text(json.dumps(list(parts)))
            todo = [p for p in parts if not (checkpoint / "results" / f"{p}.json").exists()]
            run.metrics[FANOUT_METRIC] = FanOut(
                parts=tuple(FanOutPart(p, (f"results/{p}.json",)) for p in todo),
                parallel=parallel,
                share=("shared",),
            ).to_json()
            yield
            return
        if role == "part":
            assert part is not None
            # A piece starts from the shared members and nothing else of the stage's.
            assert (checkpoint / "shared" / "plan.json").is_file()
            assert not (checkpoint / "results").exists()
            for _ in range((ticks or {}).get(part, 2)):
                run.log(f"working on {part}")
                yield
            (checkpoint / "results").mkdir(exist_ok=True)
            (checkpoint / "results" / f"{part}.json").write_text(json.dumps({"part": part}))
            return
        assert role == "join", f"a call with no role: {run.request.params}"
        yield
        found = sorted(p.stem for p in (checkpoint / "results").glob("*.json"))
        if found == sorted(parts):
            run.output(OUTPUT.name).write_text(json.dumps({"merged": found}))

    return script


class Watched:
    """A `FakeAdapter`, counting the calls in flight at once, and failing (or preempting)
    the first `times` calls of chosen pieces."""

    def __init__(
        self,
        inner: FakeAdapter,
        *,
        fail: Collection[str] = (),
        preempt: Collection[str] = (),
        times: int = 1,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.interruptible = inner.interruptible
        self.fail = set(fail)
        self.preempt = set(preempt)
        self.times = times
        self.live: set[str] = set()
        self.peak = 0
        self.calls: list[tuple[str | None, str | None]] = []
        self.doomed: dict[str, str] = {}
        self.lost: dict[str, int] = {}

    def rate(self, tier: str) -> Rate | None:
        return self.inner.rate(tier)

    def submit(self, request: StageRequest) -> RemoteHandle:
        handle = self.inner.submit(request)
        role, part = fanout_role(request.params)
        self.calls.append((role, part))
        self.live.add(handle.id)
        self.peak = max(self.peak, len(self.live))
        if part is not None and self.lost.get(part, 0) < self.times:
            if part in self.fail:
                self.doomed[handle.id] = "failed"
            elif part in self.preempt:
                self.doomed[handle.id] = "preempted"
            if handle.id in self.doomed:
                self.lost[part] = self.lost.get(part, 0) + 1
        return handle

    def poll(self, handle: RemoteHandle) -> Poll:
        if handle.id in self.doomed:
            self.live.discard(handle.id)
            state = self.doomed[handle.id]
            return Poll(state="failed" if state == "failed" else "preempted", billed_s=3.0)
        poll = self.inner.poll(handle)
        if poll.state in TERMINAL:
            self.live.discard(handle.id)
        return poll

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        return self.inner.logs(handle, since=since)

    def cancel(self, handle: RemoteHandle) -> None:
        self.live.discard(handle.id)
        self.inner.cancel(handle)


def setup(
    tmp_path: Path,
    parts: Sequence[str],
    parallel: int,
    *,
    ticks: dict[str, int] | None = None,
    max_parallel: int = 8,
    part_attempts: int = 3,
    **doom: Any,
) -> tuple[Workdir, Watched, CloudRunner]:
    transfer = LocalTransfer(tmp_path / "bucket")
    fake = FakeAdapter(
        transfer,
        tmp_path / "sandbox",
        name="modal",
        interruptible=False,
        rates={"l4": RATE},
        script=pieces(parts, parallel, ticks),
    )
    adapter = Watched(fake, **doom)
    runner = CloudRunner(
        Placement((adapter,)),
        transfer,
        poll_interval_s=0.0,
        checkpoint_every_s=0.0,
        sleep=lambda _seconds: None,
        max_parallel=max_parallel,
        part_attempts=part_attempts,
    )
    return Workdir.create(tmp_path / "run"), adapter, runner


def run(workdir: Workdir, runner: CloudRunner, attempt: int = 1) -> None:
    execute(recipe(), workdir, RunnerSet.cloud(runner), attempts={"train": attempt})


def ledger(workdir: Workdir) -> AttemptLedger:
    return AttemptLedger.read(workdir.attempts_path("train"))


def test_the_parts_run_at_once_bounded_by_parallel_and_the_join_comes_last(
    tmp_path: Path,
) -> None:
    parts = ["p0", "p1", "p2", "p3", "p4"]
    workdir, adapter, runner = setup(tmp_path, parts, parallel=2, ticks={"p0": 6})

    run(workdir, runner)

    assert adapter.peak == 2
    roles = [role for role, _ in adapter.calls]
    assert roles == ["head", "part", "part", "part", "part", "part", "join"]
    # In the order the head gave, every piece exactly once, the join after them all.
    assert [part for role, part in adapter.calls if role == "part"] == parts
    merged = json.loads((workdir.out_dir("train") / "merged.json").read_text())
    assert merged == {"merged": parts}
    # Each result was brought home into the stage's own checkpoint; nothing else was.
    home = workdir.checkpoint_dir("train")
    assert sorted(p.name for p in (home / "results").iterdir()) == [f"{p}.json" for p in parts]
    step = json.loads(workdir.step_path("train").read_text())["metrics"]
    assert step["fanOutParts"] == 5 and step["fanOutPeak"] == 2 and step["fanOutCalls"] == 5
    assert FANOUT_METRIC not in step
    # The pieces' keys are cleaned up once the join has used what they left.
    keys = tmp_path / "bucket" / "runs" / workdir.root.name / "train" / "parts"
    assert not keys.exists() or not any(keys.iterdir())
    log = workdir.log_path("train").read_text()
    assert "[p0] working on p0" in log


def test_every_call_is_priced_and_the_attempt_is_their_sum(tmp_path: Path) -> None:
    """Concurrent calls each bill their own GPU-seconds; the step's `billedS` and
    `costUsd` are the sum over the head, the pieces and the join, as is `run_cost`."""
    workdir, _adapter, runner = setup(tmp_path, ["p0", "p1", "p2"], parallel=4)

    run(workdir, runner)

    entries = ledger(workdir).entries
    assert [entry.part for entry in entries] == ["", "p0", "p1", "p2", "join"]
    assert all(entry.state == "succeeded" and entry.attempt == 1 for entry in entries)
    billed = sum(entry.billed_s for entry in entries)
    usd = sum(entry.usd or 0.0 for entry in entries)
    assert billed > 0 and usd == pytest.approx(billed / 3600 * 0.80, abs=1e-5)
    step = json.loads(workdir.step_path("train").read_text())["metrics"]
    assert step["billedS"] == pytest.approx(billed)
    assert step["costUsd"] == pytest.approx(usd, abs=1e-5)
    assert step["fanOutBilledS"] == pytest.approx(sum(e.billed_s for e in entries[1:4]))
    assert run_cost(workdir).billed_s == pytest.approx(billed)
    assert run_cost(workdir).usd == pytest.approx(usd, abs=1e-5)


def test_a_failed_part_is_retried_alone_while_the_others_carry_on(tmp_path: Path) -> None:
    workdir, adapter, runner = setup(tmp_path, ["p0", "p1", "p2"], parallel=3, fail=["p1"])

    run(workdir, runner)

    parts = [part for role, part in adapter.calls if role == "part"]
    assert sorted(str(part) for part in parts) == ["p0", "p1", "p1", "p2"]
    states = [(entry.part, entry.state) for entry in ledger(workdir).entries]
    assert ("p1", "failed") in states and ("p1", "succeeded") in states
    assert [s for p, s in states if p in ("p0", "p2")] == ["succeeded", "succeeded"]
    assert "part p1 failed on call 1 of 3; resubmitting it alone" in (
        workdir.log_path("train").read_text()
    )
    assert (workdir.out_dir("train") / "merged.json").is_file()


def test_a_part_that_never_succeeds_ends_the_attempt_after_the_others_and_is_all_that_reruns(
    tmp_path: Path,
) -> None:
    """The attempt gives up on p1 only after p0 and p2 have finished and come home, so
    the worker's next attempt -- whose head sees their results -- runs p1 and nothing else."""
    workdir, adapter, runner = setup(
        tmp_path, ["p0", "p1", "p2"], parallel=3, part_attempts=2, fail=["p1"], times=2
    )

    with pytest.raises(RemoteStageError, match="part p1 failed"):
        run(workdir, runner)

    home = workdir.checkpoint_dir("train") / "results"
    assert sorted(p.name for p in home.iterdir()) == ["p0.json", "p2.json"]
    assert [part for role, part in adapter.calls if role == "part"].count("p1") == 2
    assert not any(role == "join" for role, _ in adapter.calls)

    adapter.calls.clear()
    run(workdir, runner, attempt=2)

    assert adapter.calls == [("head", None), ("part", "p1"), ("join", None)]
    assert json.loads((workdir.out_dir("train") / "merged.json").read_text()) == {
        "merged": ["p0", "p1", "p2"]
    }
    attempts = {entry.attempt for entry in ledger(workdir).entries}
    assert attempts == {1, 2}


def test_a_part_lost_to_preemption_every_time_ends_the_attempt_as_a_preemption(
    tmp_path: Path,
) -> None:
    workdir, _adapter, runner = setup(
        tmp_path, ["p0", "p1"], parallel=2, part_attempts=2, preempt=["p0"], times=2
    )

    with pytest.raises(PreemptedError):
        run(workdir, runner)

    assert ledger(workdir).preemptions == 2
    assert (workdir.checkpoint_dir("train") / "results" / "p1.json").is_file()


def test_the_runner_caps_what_a_stage_asks_for(tmp_path: Path) -> None:
    """A deployment's ceiling (`max_parallel`, under the provider's GPU concurrency limit)
    wins over the stage's `parallel`."""
    workdir, adapter, runner = setup(tmp_path, ["p0", "p1", "p2"], parallel=8, max_parallel=1)

    run(workdir, runner)

    assert adapter.peak == 1
    step = json.loads(workdir.step_path("train").read_text())["metrics"]
    assert step["fanOutParallel"] == 1


def test_a_part_that_finishes_without_its_result_is_run_again(tmp_path: Path) -> None:
    """Success with nothing to collect is not success: the join would merge a hole."""
    workdir, adapter, runner = setup(tmp_path, ["p0", "p1"], parallel=2)
    script = pieces(["p0", "p1"], 2)
    forgot: set[str] = set()

    def forgetful(run: RemoteExecution) -> Iterator[None]:
        role, part = fanout_role(run.request.params)
        yield from script(run)
        if role == "part" and part == "p0" and part not in forgot:
            forgot.add(part)
            (run.checkpoint_dir / "results" / "p0.json").unlink()

    adapter.inner._script = forgetful

    run(workdir, runner)

    states = [(entry.part, entry.state) for entry in ledger(workdir).entries]
    assert states.count(("p0", "failed")) == 1 and states.count(("p0", "succeeded")) == 1
    detail = next(e.detail for e in ledger(workdir).entries if e.state == "failed")
    assert "results/p0.json did not come back" in detail


def test_a_head_that_does_not_fan_out_is_the_whole_stage(tmp_path: Path) -> None:
    """A stage that ignores the role -- every stage but a block run -- is one call, as
    before, and its ledger is the one it always wrote."""
    transfer = LocalTransfer(tmp_path / "bucket")

    def whole(run: RemoteExecution) -> Iterator[None]:
        assert run.request.params[FANOUT_PARAM] == {"role": "head"}
        yield
        run.output(OUTPUT.name).write_text("{}")

    fake = FakeAdapter(transfer, tmp_path / "sandbox", rates={"l4": RATE}, script=whole)
    runner = CloudRunner(
        Placement((fake,)), transfer, poll_interval_s=0.0, sleep=lambda _seconds: None
    )
    workdir = Workdir.create(tmp_path / "run")

    run(workdir, runner)

    assert len(fake.submitted) == 1
    document = json.loads(workdir.attempts_path("train").read_text())
    assert "part" not in document["attempts"][0]


def test_an_unreadable_fan_out_is_no_fan_out() -> None:
    assert FanOut.parse(None) is None
    assert FanOut.parse("not json") is None
    assert FanOut.parse(json.dumps({"parts": []})) is None
    assert FanOut.parse(json.dumps({"parts": [{"id": "a"}, {"id": "a"}]})) is None
    spec = FanOut(parts=(FanOutPart("b0", ("x",)),), parallel=3, share=("s",))
    assert FanOut.parse(spec.to_json()) == spec
    assert fanout_role({}) == (None, None)
    assert fanout_role({FANOUT_PARAM: {"role": "part", "part": "b1"}}) == ("part", "b1")
    assert fanout_role({FANOUT_PARAM: {"role": "sideways"}}) == (None, None)


class Lines:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)


def splat_line(step: int, total: int) -> str:
    payload = {"v": 1, "step": step, "total": total, "count": 10, "of": 10, "bytes": 99}
    key = f"runs/r/train/parts/b0/checkpoint/live/splat_{step:06d}.spz"
    return "live-splat: " + json.dumps({**payload, "key": key, "up": None})


def test_the_bar_follows_the_slowest_part_and_the_viewer_the_furthest() -> None:
    """The worker reads the newest progress and live lines off the stage log: the bar is
    the slowest piece's (the stage ends when it does), the splat the most-trained one's."""
    sink = Lines()
    tail = _FanOutLog(cast("StageContext", sink))

    tail.line("b0", "loss=0.1| : 50%|#| 500/1000 [00:05<00:05, 99.00it/s]")
    tail.line("b1", "loss=0.1| : 20%|#| 200/1000 [00:02<00:08, 99.00it/s]")
    tail.line("b0", "loss=0.1| : 60%|#| 600/1000 [00:06<00:04, 99.00it/s]")
    tail.line("b1", splat_line(250, 1000))
    tail.line("b0", splat_line(500, 1000))
    tail.line("b1", splat_line(500, 1000))
    tail.line("b1", splat_line(400, 1000))
    tail.line("b0", "an ordinary line")

    shown = sink.lines
    assert [line for line in shown if "it/s" in line] == [
        "[b0] loss=0.1| : 50%|#| 500/1000 [00:05<00:05, 99.00it/s]",
        "[b1] loss=0.1| : 20%|#| 200/1000 [00:02<00:08, 99.00it/s]",
    ]
    live = [line for line in shown if "live-splat: " in line]
    assert [json.loads(line.split("live-splat: ")[1])["step"] for line in live] == [250, 500, 500]
    assert "[b1] live: a snapshot at step 400 of 1000 is not shown" in shown[-2]
    assert shown[-1] == "[b0] an ordinary line"
