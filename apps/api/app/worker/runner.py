"""Supervising one claimed job from `in-progress` to a terminal status.

The shape, and why:

* the recipe runs in a **child process** (`app.worker.child`), so the thing that beats the
  lease is never the thing doing the work. A stage that trains for two hours does not
  stop the heartbeat, because the heartbeat is a different process;
* the supervisor's loop *is* the heartbeat. Every `poll_s` it renews `lease_expires_at`,
  re-reads `jobs.status` to see whether somebody cancelled, and drains whatever the child
  has reported since the last tick. One loop, three jobs, no threads except the one that
  reads the child's stdout;
* every write commits. The panel is polling `GET /jobs`, and a step row that only lands
  when the run is over is not a live stage list. It also keeps the worker from holding a
  transaction open across a stage, which is what A0 measured pinning the vacuum horizon.

What happens when a worker dies mid-stage: nothing here runs, so `lease_expires_at`
simply passes. The job is then claimable by anyone (see `claim.claimable`), and the next
worker resumes it — the stages that already completed are **skipped**, because their
`step.json` and outputs are still in the workdir, and the stage that was interrupted is
re-run from scratch with `attempt` incremented (A6 wipes `out/` at the start of every
attempt and keeps `checkpoint/`). If the workdir is gone — another machine, a cleaned
disk — nothing is skipped and the job restarts from the beginning, which is the honest
answer rather than a resume that silently has no inputs.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, Job, JobStep
from app.models.enums import CaptureStatus, RunStatus, UploadStatus
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker import claim, events, outputs, params, registration, steps
from app.worker.child import ChildSpec, load_impl_modules, resolve_recipe
from app.worker.config import WorkerConfig
from app.worker.events import Event
from app.worker.pipeline_bridge import (
    PIPELINE_DIR,
    ArtifactRef,
    PipelineError,
    Plan,
    Workdir,
    plan_recipe,
)

log = logging.getLogger("app.worker")

#: Pushed onto the event queue when the child's stdout reaches EOF.
_EOF = object()

Terminal = Literal["complete", "error", "cancelled", "lost"]


@dataclass
class _RunState:
    """What one child process reported, accumulated as it reported it."""

    outcome: str = ""
    failed_stage: str = ""
    error: str = ""
    error_type: str = ""
    #: Stage id -> the artifacts it produced, from the StepResult it sent.
    produced: dict[str, tuple[ArtifactRef, ...]] = field(default_factory=dict)

    def stage_producing(self, artifact: str) -> str | None:
        for stage_id, refs in self.produced.items():
            if any(ref.name == artifact for ref in refs):
                return stage_id
        return None

    def ref(self, artifact: str) -> ArtifactRef | None:
        for refs in self.produced.values():
            for candidate in refs:
                if candidate.name == artifact:
                    return candidate
        return None


class JobSupervisor:
    """Runs one job to a terminal status. One instance per job; not reused."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        config: WorkerConfig,
    ) -> None:
        self._sessions = session_factory
        self._storage = storage
        self._config = config
        self._id = config.worker_id

    # --- the outer loop: attempts ------------------------------------------------

    def run(self, job_id: uuid.UUID, *, stop: threading.Event | None = None) -> Terminal:
        db = self._sessions()
        try:
            return self._run(db, job_id, stop)
        except Exception as error:
            # One job must not take the worker down. A bucket that has gone away, a
            # workdir on a full disk, a bug here -- the job says what happened and the
            # loop goes on to the next one. A person can retry it from the panel, which
            # is the same affordance a dead-lettered job gets.
            log.exception("worker %s: job %s failed in the supervisor", self._id, job_id)
            return self._report_supervisor_failure(job_id, error)
        finally:
            db.close()

    def _report_supervisor_failure(self, job_id: uuid.UUID, error: Exception) -> Terminal:
        """Record the failure on a session of its own: the one that raised may be unusable."""
        db = self._sessions()
        try:
            job = db.get(Job, job_id)
            if job is None:
                return "lost"
            return self._dead_letter(db, job, f"the worker failed while running this job: {error}")
        except Exception:
            log.exception("worker %s: could not record the failure of job %s", self._id, job_id)
            return "error"
        finally:
            db.close()

    def _run(self, db: Session, job_id: uuid.UUID, stop: threading.Event | None) -> Terminal:
        job = db.get(Job, job_id)
        if job is None:
            return "lost"
        try:
            load_impl_modules(self._config.impl_modules)
            recipe = resolve_recipe(job.recipe, self._recipe_dir())
            plan = plan_recipe(recipe)
            stage_params = self._stage_params(db, job, plan)
        except (PipelineError, ValueError) as error:
            # A recipe that will not resolve will not resolve on the next attempt either,
            # and neither will a `params` of the wrong shape, so this dead-letters
            # immediately rather than burning the attempt budget.
            return self._dead_letter(db, job, f"recipe {job.recipe!r} did not resolve: {error}")
        workdir_root = self._config.workdir_for(job_id)
        try:
            self._seed(db, job, recipe.inputs, workdir_root)
        except StorageUnavailableError as error:
            return self._dead_letter(db, job, f"could not fetch the capture's files: {error}")

        while True:
            completed = self._completed_stages(db, job_id, workdir_root)
            attempts = self._attempts(db, job_id, plan, completed)
            exhausted = self._exhausted(plan, completed, attempts)
            if exhausted is not None:
                stage_id, attempt = exhausted
                last = self._last_error(db, job)
                return self._dead_letter(
                    db,
                    job,
                    f"stage {stage_id!r} has been attempted {attempt - 1} times without "
                    f"completing and will not be retried again{last}",
                )
            state = self._supervise(db, job, workdir_root, completed, attempts, stage_params, stop)
            if state.outcome == "stopped":
                # This worker is shutting down. Let go of the lease so the next one can
                # take the job now rather than waiting it out; the stages that finished
                # are in the workdir and will be skipped.
                claim.release(db, job.id, worker_id=self._config.worker_id)
                return "lost"
            if state.outcome == "cancelled":
                return self._finish_cancelled(db, job)
            if state.outcome == "lost":
                return "lost"
            if state.outcome == events.RUN_FINISHED:
                return self._finish_complete(db, job, state)
            if not state.failed_stage:
                # Nothing to retry from: the run never reached a stage.
                return self._dead_letter(db, job, state.error or "the run failed before any stage")
            job.error = state.error
            db.commit()
            time.sleep(self._config.retry_backoff_s)
            if (
                claim.heartbeat(
                    db,
                    job.id,
                    worker_id=self._config.worker_id,
                    lease_s=self._config.lease_s,
                )
                is not claim.Heartbeat.HELD
            ):
                return "lost"

    # --- one child process --------------------------------------------------------

    def _supervise(
        self,
        db: Session,
        job: Job,
        workdir_root: Path,
        completed: set[str],
        attempts: dict[str, int],
        stage_params: dict[str, dict[str, Any]],
        stop: threading.Event | None = None,
    ) -> _RunState:
        spec_path = ChildSpec(
            recipe=job.recipe,
            workdir=str(workdir_root),
            runner=self._config.runner,
            recipe_dir=self._recipe_dir(),
            impl_modules=self._config.impl_modules,
            skip=tuple(sorted(completed)),
            attempts=attempts,
            params=stage_params,
        ).write(workdir_root / "child.json")
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell, our own module
            [sys.executable, "-u", "-m", "app.worker.child", str(spec_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._child_env(),
            cwd=str(_API_ROOT),
            text=True,
        )
        inbox: queue.Queue[object] = queue.Queue()
        reader = threading.Thread(target=_pump, args=(process.stdout, inbox), daemon=True)
        reader.start()

        state = _RunState()
        current: JobStep | None = None
        next_beat = time.monotonic()
        while True:
            now = time.monotonic()
            if stop is not None and stop.is_set():
                _stop(process, self._config.terminate_grace_s)
                state.outcome = "stopped"
                return state
            if now >= next_beat:
                beat = claim.heartbeat(
                    db, job.id, worker_id=self._config.worker_id, lease_s=self._config.lease_s
                )
                if beat is not claim.Heartbeat.HELD:
                    # Cancelled, or reclaimed while this worker was not looking. Either
                    # way the child must stop now, not at the end of its stage.
                    _stop(process, self._config.terminate_grace_s)
                    state.outcome = "cancelled" if beat is claim.Heartbeat.CANCELLED else "lost"
                    return state
                next_beat = now + self._config.poll_s
            try:
                item = inbox.get(timeout=max(0.01, next_beat - time.monotonic()))
            except queue.Empty:
                continue
            if item is _EOF:
                break
            if isinstance(item, Event):
                current = self._apply(db, job, item, workdir_root, state, current)
        process.wait()
        if not state.outcome:
            # No RUN_FINISHED and no RUN_FAILED: the child was killed from outside.
            state.outcome = events.RUN_FAILED
            state.error = state.error or (
                f"the process running the recipe exited with code {process.returncode} "
                f"without reporting a result"
            )
            if current is not None:
                state.failed_stage = current.stage_id
                steps.fail_step(
                    db, current, log_key=self._upload_log(job.id, workdir_root, current.stage_id)
                )
        return state

    def _apply(
        self,
        db: Session,
        job: Job,
        event: Event,
        workdir_root: Path,
        state: _RunState,
        current: JobStep | None,
    ) -> JobStep | None:
        if event.kind == events.STAGE_STARTED:
            return steps.start_step(
                db,
                job.id,
                stage_id=event.stage_id,
                ordinal=event.ordinal,
                impl=event.impl,
                attempt=event.attempt,
            )
        if event.kind in (events.STAGE_FINISHED, events.STAGE_SKIPPED):
            refs = tuple(ArtifactRef.from_dict(entry) for entry in event.step.get("artifacts", ()))
            state.produced[event.stage_id] = refs
            if event.kind == events.STAGE_SKIPPED:
                return None
            step = current or steps.start_step(
                db,
                job.id,
                stage_id=event.stage_id,
                ordinal=event.ordinal,
                impl=event.impl,
                attempt=event.attempt,
            )
            uploaded = [
                result
                for ref in refs
                if (result := outputs.upload_artifact(self._storage, workdir_root, job.id, ref))
            ]
            steps.finish_step(
                db,
                step,
                metrics=_metrics(event.step),
                log_key=self._upload_log(job.id, workdir_root, event.stage_id),
                checkpoint_key=_optional_str(event.step.get("checkpointKey")),
                artifacts=uploaded,
            )
            return None
        if event.kind == events.STAGE_FAILED:
            state.failed_stage = event.stage_id
            state.error = event.error
            state.error_type = event.error_type
            if current is not None:
                steps.fail_step(
                    db, current, log_key=self._upload_log(job.id, workdir_root, event.stage_id)
                )
            return None
        if event.kind in (events.RUN_FINISHED, events.RUN_FAILED):
            state.outcome = event.kind
            if event.kind == events.RUN_FAILED:
                state.failed_stage = state.failed_stage or event.stage_id
                state.error = state.error or event.error
                state.error_type = state.error_type or event.error_type
            return current
        return current

    # --- terminal states ----------------------------------------------------------

    def _finish_complete(self, db: Session, job: Job, state: _RunState) -> Terminal:
        if not self._still_ours(db, job):
            return "lost"
        capture = db.get(Capture, job.capture_id)
        ref = state.ref("registration.json")
        if capture is not None and ref is not None:
            document = registration.Registration.read(self._config.workdir_for(job.id) / ref.path)
            registration.register(
                db,
                self._storage,
                capture=capture,
                job_id=job.id,
                registration=document,
                tiles_stage_id=state.stage_producing("splat"),
                thumbnail_stage_id=state.stage_producing("thumbnail.jpg"),
            )
        elif capture is not None:
            # A recipe with no `register` stage still finished; the capture is processed
            # even though there is nothing to put on the globe.
            capture.status = CaptureStatus.COMPLETE
        job.status = RunStatus.COMPLETE
        job.error = None
        self._close(job)
        db.commit()
        return "complete"

    def _finish_cancelled(self, db: Session, job: Job) -> Terminal:
        """`POST /jobs/{id}/cancel` already set the status; this closes out the run.

        The workdir is left exactly as the killed stage left it: its `out/` may be half
        written, which is precisely why A6 clears `out/` at the start of every attempt.
        Nothing half-written is uploaded, and no `artifacts` row is created for it.
        """
        steps.stop_active_steps(db, job.id, RunStatus.CANCELLED)
        db.refresh(job)
        if job.finished_at is None:
            job.finished_at = datetime.now(tz=UTC)
        job.lease_expires_at = None
        db.commit()
        return "cancelled"

    def _dead_letter(self, db: Session, job: Job, why: str) -> Terminal:
        """Stop trying, and say why in the place the panel already renders."""
        if not self._still_ours(db, job):
            return "lost"
        steps.stop_active_steps(db, job.id, RunStatus.ERROR)
        db.refresh(job)
        job.status = RunStatus.ERROR
        job.error = why
        self._close(job)
        capture = db.get(Capture, job.capture_id)
        if capture is not None:
            capture.status = CaptureStatus.ERROR
        db.commit()
        return "error"

    def _close(self, job: Job) -> None:
        now = datetime.now(tz=UTC)
        job.finished_at = now
        if job.claimed_at is not None:
            job.duration_s = (now - job.claimed_at).total_seconds()
        # The lease is over; `claimed_by` stays, because who ran it is a fact worth keeping.
        job.lease_expires_at = None

    def _still_ours(self, db: Session, job: Job) -> bool:
        db.refresh(job)
        return job.status is RunStatus.IN_PROGRESS and job.claimed_by == self._config.worker_id

    # --- helpers ------------------------------------------------------------------

    def _stage_params(self, db: Session, job: Job, plan: Plan) -> dict[str, dict[str, Any]]:
        """What this capture adds to the recipe's parameters: where it is, what took it."""
        capture = db.get(Capture, job.capture_id)
        if capture is None:
            return {}
        resolved = params.stage_params(plan, capture, job)
        # Refused here rather than in the child, so a `params` naming a stage the recipe
        # does not have dead-letters with the recipe error instead of a failed run.
        plan.recipe.with_params(resolved)
        return resolved

    def _recipe_dir(self) -> str | None:
        return str(self._config.recipe_dir) if self._config.recipe_dir else None

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{_API_ROOT}:{existing}" if existing else str(_API_ROOT)
        env["PIPELINE_DIR"] = str(PIPELINE_DIR)
        return env

    def _upload_log(self, job_id: uuid.UUID, workdir_root: Path, stage_id: str) -> str | None:
        return outputs.upload_log(self._storage, workdir_root, job_id, stage_id)

    def _seed(self, db: Session, job: Job, inputs: tuple[str, ...], workdir_root: Path) -> None:
        """Put the capture's uploaded bytes where the recipe says its inputs live.

        Skipped when the directory is already populated, so a reclaimed or retried job
        does not download a 12 GB video again. (It *is* a whole-object read into memory;
        B1, which moves workdirs between machines, is where streaming belongs.)
        """
        work = Workdir.create(workdir_root)
        capture = db.get(Capture, job.capture_id)
        for name in inputs:
            target = work.input_path(name)
            if target.is_dir() and any(target.iterdir()):
                continue
            if name != "upload" or capture is None:
                continue
            target.mkdir(parents=True, exist_ok=True)
            for source in capture.files:
                if source.status is not UploadStatus.COMPLETE:
                    continue
                (target / Path(source.filename).name).write_bytes(
                    self._storage.get_object(source.storage_key)
                )

    def _completed_stages(self, db: Session, job_id: uuid.UUID, workdir_root: Path) -> set[str]:
        """Stages that may be skipped: complete in the database **and** still on disk."""
        rows = steps.steps_of(db, job_id)
        workdir = Workdir(workdir_root)
        return {
            stage_id
            for stage_id, row in rows.items()
            if row.status is RunStatus.COMPLETE and workdir.step_path(stage_id).is_file()
        }

    def _attempts(
        self, db: Session, job_id: uuid.UUID, plan: Plan, completed: set[str]
    ) -> dict[str, int]:
        rows = steps.steps_of(db, job_id)
        return {
            stage.id: (rows[stage.id].attempt if stage.id in rows else 0) + 1
            for stage in plan.stages
            if stage.id not in completed
        }

    def _exhausted(
        self, plan: Plan, completed: set[str], attempts: dict[str, int]
    ) -> tuple[str, int] | None:
        for stage in plan.stages:
            if stage.id in completed:
                continue
            attempt = attempts.get(stage.id, 1)
            return (stage.id, attempt) if attempt > self._config.max_attempts else None
        return None

    def _last_error(self, db: Session, job: Job) -> str:
        db.refresh(job)
        return f": {job.error}" if job.error else ""


_API_ROOT = Path(__file__).resolve().parents[2]


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _metrics(step: dict[str, Any]) -> dict[str, Any]:
    """The stage's own metrics, plus the three executor facts A2 gave no column to.

    `durationS`, `runner` and `summary` are the executor's, not the stage's; a stage that
    emits a metric by one of those names loses it, which is a trade worth making for not
    adding three columns to `job_steps` before A10 has asked for them.
    """
    metrics = dict(step.get("metrics") or {})
    metrics["durationS"] = round(float(step.get("durationS") or 0.0), 3)
    metrics["runner"] = step.get("runner")
    metrics["summary"] = step.get("summary")
    return metrics


def _pump(stream: IO[str] | None, inbox: queue.Queue[object]) -> None:
    """Read the child's stdout into the queue, and say when it ends.

    A thread rather than a select loop because there is exactly one pipe, and because the
    supervisor's own loop has to stay on its heartbeat schedule regardless of whether the
    child is saying anything.
    """
    if stream is not None:
        for line in stream:
            event = Event.from_json(line)
            if event is not None:
                inbox.put(event)
        stream.close()
    inbox.put(_EOF)


def _stop(process: subprocess.Popen[str], grace_s: float) -> None:
    """SIGTERM, then SIGKILL. A stage that ignores the first does not get to keep running."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
