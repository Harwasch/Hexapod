"""The process that actually runs a recipe. It has no database.

Why a separate process at all, when the executor is a library call:

* **A stage can run for hours** (B2 trains on a GPU), so the heartbeat cannot be "between
  stages". Putting the recipe in a child leaves the supervisor free to beat the lease,
  drain progress and watch for a cancellation on a two-second tick the whole time the
  stage is running — the heartbeat is not interleaved with the work, it is a different
  process from it.
* **Cancellation has to arrive mid-stage.** A thread cannot be interrupted; a process can
  be signalled. `POST /jobs/{id}/cancel` takes effect within one supervisor tick because
  the supervisor signals this process and, if that is ignored, SIGKILLs it.
* **The signal says why.** SIGUSR2 is a cancel (the job was cancelled, or the lease was
  lost) and SIGUSR1 a detach (the worker is shutting down for a deploy). `Interrupts`
  turns each into the pipeline's `CancelRequested` or `DetachRequested`, raised in the
  main thread wherever it is, and `CloudRunner` answers them differently: a cancel
  cancels the remote call and records what it cost, a detach leaves it running and
  writes down where it is, for the next worker to re-attach to. Until the 2026-10 audit
  this process had no handlers and the supervisor sent SIGTERM, whose default action is
  to die on the spot: a GPU on Modal kept training for nobody for up to six hours.

  SIGTERM and SIGINT themselves are left to the supervisor. A shutdown that signals the
  whole process group -- systemd's default, a terminal's Ctrl-C -- reaches this process
  as well as the worker, and read as a cancel it would stop the GPU call on every
  deploy, the one case where it must be left running. The worker decides, and says which
  with one of the two signals above; this process ignores the other two (with a Python
  handler, not SIG_IGN, which the tools it runs would inherit).
* **A stage that dies does not take the worker with it.** A segfault in a native trainer
  is a failed step, not a lost queue.

It reads one JSON spec file, writes one JSON event per line to stdout, and exits 0 or 1.
Everything it produces lands in the workdir, where the supervisor picks it up.

Until B1b this process had no credentials either. With `runner: cloud` it has the
bucket's, because a stage running on somebody else's machine has to fetch its inputs from
somewhere and pushing every byte through the process whose job is to hold a lease is the
wrong shape (`app/worker/cloud.py` says more). It still has no database session, which is
the part that mattered: nothing it does can write a row.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import Any

from app.storage.factory import get_storage
from app.worker import events
from app.worker.cloud import build_runners
from app.worker.events import Event
from app.worker.pipeline_bridge import (
    CancelRequested,
    CloudRunner,
    DetachRequested,
    PlannedStage,
    Recipe,
    RunnerSet,
    StepResult,
    StopRequested,
    Workdir,
    execute,
    load_recipe,
    plan_recipe,
)

#: What the supervisor sends for each of the two ways a recipe process is stopped. Not
#: SIGTERM: see the module docstring for why that one is left to the supervisor.
CANCEL_SIGNAL = signal.SIGUSR2
DETACH_SIGNAL = signal.SIGUSR1

#: The exit status of a process that stopped because it was asked to.
EXIT_STOPPED = 3


@dataclass(frozen=True)
class ChildSpec:
    """Everything the child is told. Written by the supervisor into the workdir."""

    recipe: str
    workdir: str
    runner: str = "stub"
    recipe_dir: str | None = None
    impl_modules: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    attempts: dict[str, int] | None = None
    #: Per-stage parameter overrides for this run — the capture's coordinate, its sensor,
    #: and whatever `jobs.params` asked for. See `app.worker.params`.
    params: dict[str, dict[str, Any]] | None = None
    #: Only read when `runner == "cloud"`. See `app.worker.cloud.build_runners`.
    cloud_providers: tuple[str, ...] = ()
    preemptions_before_fallback: int = 2
    modal_app: str = ""
    cloud_poll_s: float = 5.0
    checkpoint_every_s: float = 60.0
    #: Scratch for an adapter that runs a stage on this machine; never the workdir.
    sandbox: str | None = None
    #: A directory both this worker and the machine running the stage can see. Unset,
    #: the bytes go through the bucket.
    transfer_dir: str | None = None
    #: The job's dollar ceiling and the overdue-trainer guard (`CloudRunner`); a cap of 0
    #: or None is no cap.
    cost_cap_usd: float | None = None
    deadline_factor: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "workdir": self.workdir,
            "runner": self.runner,
            "recipeDir": self.recipe_dir,
            "implModules": list(self.impl_modules),
            "skip": list(self.skip),
            "attempts": dict(self.attempts or {}),
            "params": dict(self.params or {}),
            "cloudProviders": list(self.cloud_providers),
            "preemptionsBeforeFallback": self.preemptions_before_fallback,
            "modalApp": self.modal_app,
            "cloudPollS": self.cloud_poll_s,
            "checkpointEveryS": self.checkpoint_every_s,
            "sandbox": self.sandbox,
            "transferDir": self.transfer_dir,
            "costCapUsd": self.cost_cap_usd,
            "deadlineFactor": self.deadline_factor,
        }

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n", "utf-8")
        return path

    @staticmethod
    def read(path: Path) -> ChildSpec:
        document = json.loads(path.read_text(encoding="utf-8"))
        return ChildSpec(
            recipe=str(document["recipe"]),
            workdir=str(document["workdir"]),
            runner=str(document.get("runner", "stub")),
            recipe_dir=document.get("recipeDir"),
            impl_modules=tuple(str(name) for name in document.get("implModules", ())),
            skip=tuple(str(name) for name in document.get("skip", ())),
            attempts={str(k): int(v) for k, v in dict(document.get("attempts", {})).items()},
            params={str(k): dict(v) for k, v in dict(document.get("params", {})).items()},
            cloud_providers=tuple(str(n) for n in document.get("cloudProviders", ())),
            preemptions_before_fallback=int(document.get("preemptionsBeforeFallback", 2)),
            modal_app=str(document.get("modalApp", "")),
            cloud_poll_s=float(document.get("cloudPollS", 5.0)),
            checkpoint_every_s=float(document.get("checkpointEveryS", 60.0)),
            sandbox=document.get("sandbox"),
            transfer_dir=document.get("transferDir"),
            cost_cap_usd=(
                None if document.get("costCapUsd") is None else float(document["costCapUsd"])
            ),
            deadline_factor=float(document.get("deadlineFactor", 2.0)),
        )


class Interrupts:
    """The supervisor's two signals, as the two ways a recipe process is asked to stop.

    Each handler raises its exception in the main thread, wherever the run is: in a
    poll loop, an upload, a stage's own `time.sleep`. The one place it must not land is
    between a remote call being created and its id being written into the `CallBook` --
    a call nobody could then find -- so `held()` defers a stop that arrives inside it
    until the block is done, then raises it. A cancel that arrives while a detach is
    held replaces it: stopping the call is the stronger request.
    """

    def __init__(self) -> None:
        self._held = 0
        self._pending: StopRequested | None = None

    def install(self) -> None:
        signal.signal(CANCEL_SIGNAL, self._on_cancel)
        signal.signal(DETACH_SIGNAL, self._on_detach)
        signal.signal(signal.SIGTERM, self._left_to_the_supervisor)
        signal.signal(signal.SIGINT, self._left_to_the_supervisor)

    @staticmethod
    def _left_to_the_supervisor(_signum: int, _frame: FrameType | None) -> None:
        """A shutdown signalled to the whole group. The worker got it too, and will say
        whether this is a cancel or a detach; if it is gone, `watch_for_orphaning` ends
        this process within a second, and SIGKILL is the backstop either way."""

    def _on_cancel(self, _signum: int, _frame: FrameType | None) -> None:
        self._stop(CancelRequested())

    def _on_detach(self, _signum: int, _frame: FrameType | None) -> None:
        self._stop(DetachRequested())

    def _stop(self, request: StopRequested) -> None:
        if self._held:
            if self._pending is None or isinstance(request, CancelRequested):
                self._pending = request
            return
        raise request

    @contextmanager
    def held(self) -> Iterator[None]:
        self._held += 1
        try:
            yield
        finally:
            self._held -= 1
            if not self._held and self._pending is not None:
                pending, self._pending = self._pending, None
                raise pending


#: This process's. Module-level because a signal handler is process-wide anyway.
INTERRUPTS = Interrupts()


def build_runner_set(spec: ChildSpec) -> RunnerSet:
    """Which runners this run uses. The one place the three options are spelled out.

    `cloud` is `local` plus a GPU runner, so a recipe's CPU stages still run here and
    only the stages that declare `gpu:` are dispatched — the routing signal has not
    changed, only where the GPU stage ends up.
    """
    if spec.runner == "stub":
        return RunnerSet.stubbed()
    if spec.runner == "local":
        return RunnerSet.local()
    if spec.runner == "cloud":
        return build_runners(
            get_storage(),
            providers=spec.cloud_providers,
            sandbox=Path(spec.sandbox or spec.workdir),
            impl_modules=spec.impl_modules,
            preemptions_before_fallback=spec.preemptions_before_fallback,
            modal_app=spec.modal_app,
            poll_interval_s=spec.cloud_poll_s,
            checkpoint_every_s=spec.checkpoint_every_s,
            transfer_dir=Path(spec.transfer_dir) if spec.transfer_dir else None,
            shield=INTERRUPTS.held,
            cost_cap_usd=spec.cost_cap_usd,
            deadline_factor=spec.deadline_factor,
        )
    raise ValueError(f"unknown runner {spec.runner!r}: expected stub, local or cloud")


def reap_calls(runners: RunnerSet, recipe: Recipe, workdir: Workdir, skip: Sequence[str]) -> None:
    """Cancel the remote calls an earlier process left that this run will not pick up.

    Only the first stage this run executes can re-attach to a call: the ones before it
    are skipped as done, and the ones after it will run again on inputs this run makes,
    so a call of theirs is working on stale ones. Done before anything runs, so such a
    call stops billing now rather than whenever the run reaches its stage.
    """
    if not isinstance(runners.gpu, CloudRunner):
        return
    first = next((s.id for s in plan_recipe(recipe).stages if s.id not in set(skip)), None)
    runners.gpu.reap(workdir, keep=first)


def load_impl_modules(names: Sequence[str]) -> None:
    """Import the modules whose `@stage_impl`s a recipe needs.

    Both processes do this: the child so it can run the stages, and the supervisor so it
    can resolve the recipe well enough to know which stage comes next and how many times
    it has been tried. Importing a stage module only registers declarations.
    """
    for module in names:
        import_module(module)


def resolve_recipe(name: str, recipe_dir: str | None) -> Recipe:
    """A recipe by name: the deployment's own directory first, then the shipped ones.

    `jobs.recipe` is a name rather than a path, so a run recorded today can be repeated
    tomorrow by a worker whose checkout is somewhere else.
    """
    if recipe_dir:
        local = Path(recipe_dir) / f"{name}.yaml"
        if local.is_file():
            return load_recipe(local)
    return load_recipe(name)


class _Reporter:
    """A StageObserver that prints. This is the whole of the child's output protocol."""

    def __init__(self, workdir: Workdir | None = None) -> None:
        self.failed_stage = ""
        self._workdir = workdir

    def emit(self, event: Event) -> None:
        sys.stdout.write(event.to_json() + "\n")
        sys.stdout.flush()

    def stage_started(self, stage: PlannedStage, attempt: int) -> None:
        log = self._workdir.log_path(stage.id) if self._workdir is not None else None
        self.emit(
            Event(
                kind=events.STAGE_STARTED,
                stage_id=stage.id,
                ordinal=stage.index,
                impl=stage.impl.name,
                attempt=attempt,
                # Before the stage writes a line: where this attempt's part of it begins.
                log_from=log.stat().st_size if log is not None and log.is_file() else 0,
            )
        )

    def stage_finished(self, stage: PlannedStage, result: StepResult) -> None:
        self.emit(
            Event(
                kind=events.STAGE_FINISHED,
                stage_id=stage.id,
                ordinal=stage.index,
                impl=stage.impl.name,
                attempt=result.attempt,
                step=dict(result.to_dict()),
            )
        )

    def stage_skipped(self, stage: PlannedStage, result: StepResult) -> None:
        self.emit(
            Event(
                kind=events.STAGE_SKIPPED,
                stage_id=stage.id,
                ordinal=stage.index,
                impl=stage.impl.name,
                attempt=result.attempt,
                step=dict(result.to_dict()),
            )
        )

    def stage_failed(self, stage: PlannedStage, attempt: int, error: BaseException) -> None:
        self.failed_stage = stage.id
        self.emit(
            Event(
                kind=events.STAGE_FAILED,
                stage_id=stage.id,
                ordinal=stage.index,
                impl=stage.impl.name,
                attempt=attempt,
                error=_message(error),
                error_type=type(error).__name__,
            )
        )


def _message(error: BaseException) -> str:
    text = str(error) or type(error).__name__
    return text if len(text) <= 2000 else text[:1997] + "..."


#: How often the orphan guard looks at its parent.
ORPHAN_CHECK_S = 1.0


def watch_for_orphaning(interval_s: float = ORPHAN_CHECK_S) -> None:
    """Stop if the supervisor dies, instead of running on as an orphan.

    A worker that is SIGKILLed cannot clean up after itself, and a recipe process left
    behind would keep writing into a workdir that another worker is about to reclaim and
    resume — two processes in one `out/`. So the child watches its own parent: when
    `getppid()` changes it has been reparented, and it leaves immediately.

    `os._exit` rather than an exception, because the point is to stop *now*, mid-stage.
    The half-written `out/` it leaves is exactly what A6's "clear `out/` at the start of
    every attempt" exists for.
    """
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(interval_s)
        os._exit(2)

    threading.Thread(target=watch, daemon=True).start()


def run(spec: ChildSpec) -> int:
    watch_for_orphaning()
    INTERRUPTS.install()
    load_impl_modules(spec.impl_modules)
    workdir = Workdir(Path(spec.workdir))
    reporter = _Reporter(workdir)
    try:
        recipe = resolve_recipe(spec.recipe, spec.recipe_dir).with_params(spec.params or {})
        runners = build_runner_set(spec)
        reap_calls(runners, recipe, workdir, spec.skip)
        execute(
            recipe,
            workdir,
            runners,
            observer=reporter,
            skip=set(spec.skip),
            attempts=spec.attempts or {},
        )
    except StopRequested as stop:
        # Asked to, by the supervisor: whatever had to happen to a remote call on the way
        # out (cancelled, or left running and written down) has happened by now. The
        # supervisor is no longer reading, so this is for a person reading the stream.
        reporter.emit(
            Event(
                kind=events.RUN_FAILED,
                stage_id=reporter.failed_stage,
                error=f"the recipe process was stopped: {stop.reason}",
                error_type=type(stop).__name__,
            )
        )
        return EXIT_STOPPED
    except Exception as error:
        reporter.emit(
            Event(
                kind=events.RUN_FAILED,
                stage_id=reporter.failed_stage,
                error=_message(error),
                error_type=type(error).__name__,
            )
        )
        return 1
    reporter.emit(Event(kind=events.RUN_FINISHED))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one recipe and report on stdout.")
    parser.add_argument("spec", type=Path, help="path to the JSON spec the supervisor wrote")
    args = parser.parse_args(argv)
    return run(ChildSpec.read(args.spec))


if __name__ == "__main__":
    raise SystemExit(main())
