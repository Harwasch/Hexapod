"""The process that actually runs a recipe. It has no database.

Why a separate process at all, when the executor is a library call:

* **A stage can run for hours** (B2 trains on a GPU), so the heartbeat cannot be "between
  stages". Putting the recipe in a child leaves the supervisor free to beat the lease,
  drain progress and watch for a cancellation on a two-second tick the whole time the
  stage is running — the heartbeat is not interleaved with the work, it is a different
  process from it.
* **Cancellation has to arrive mid-stage.** A thread cannot be interrupted; a process can
  be signalled. `POST /jobs/{id}/cancel` takes effect within one supervisor tick because
  the supervisor sends SIGTERM and, if that is ignored, SIGKILL.
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
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from app.storage.factory import get_storage
from app.worker import events
from app.worker.cloud import build_runners
from app.worker.events import Event
from app.worker.pipeline_bridge import (
    PlannedStage,
    Recipe,
    RunnerSet,
    StepResult,
    Workdir,
    execute,
    load_recipe,
)


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
        )


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
        )
    raise ValueError(f"unknown runner {spec.runner!r}: expected stub, local or cloud")


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

    def __init__(self) -> None:
        self.failed_stage = ""

    def emit(self, event: Event) -> None:
        sys.stdout.write(event.to_json() + "\n")
        sys.stdout.flush()

    def stage_started(self, stage: PlannedStage, attempt: int) -> None:
        self.emit(
            Event(
                kind=events.STAGE_STARTED,
                stage_id=stage.id,
                ordinal=stage.index,
                impl=stage.impl.name,
                attempt=attempt,
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
    load_impl_modules(spec.impl_modules)
    reporter = _Reporter()
    try:
        recipe = resolve_recipe(spec.recipe, spec.recipe_dir).with_params(spec.params or {})
        execute(
            recipe,
            Workdir(Path(spec.workdir)),
            build_runner_set(spec),
            observer=reporter,
            skip=set(spec.skip),
            attempts=spec.attempts or {},
        )
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
