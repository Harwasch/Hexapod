"""Supervising one claimed job from `in-progress` to a terminal status.

The shape, and why:

* the recipe runs in a **child process** (`app.worker.child`), so the thing that beats the
  lease is never the thing doing the work. A stage that trains for two hours does not
  stop the heartbeat, because the heartbeat is a different process;
* the supervisor's loop *is* the heartbeat. Every `poll_s` it renews `lease_expires_at`,
  re-reads `jobs.status` to see whether somebody cancelled, and drains whatever the child
  has reported since the last tick. One loop, three jobs, no threads except the one that
  reads the child's stdout -- and `claim.LeaseKeeper`, which renews the lease as well, so
  that it does not lapse while this loop is away downloading the capture or uploading a
  finished stage (the 2026-09-27 two-slot dead-letters; see the worker README);
* every write commits. The panel is polling `GET /jobs`, and a step row that only lands
  when the run is over is not a live stage list. It also keeps the worker from holding a
  transaction open across a stage, which is what A0 measured pinning the vacuum horizon.

What happens when a worker dies mid-stage: nothing here runs, so `lease_expires_at`
simply passes. The job is then claimable by anyone (see `claim.claimable`), and the next
worker resumes it — the stages that already completed are **skipped**, because their
`step.json` and outputs are still in the workdir, and the stage that was interrupted is
re-run with `attempt` incremented (A6 wipes `out/` at the start of every attempt and
keeps `checkpoint/`, so a stage that checkpoints continues rather than restarting). If
the workdir is gone — another machine, a cleaned disk — nothing is skipped and the job
restarts from the beginning, which is the honest answer rather than a resume that
silently has no inputs.

B1b added one distinction to that loop and one number to the job:

* **a preemption is not a failure.** `PreemptedError` from the cloud runner marks
  `job_steps.preempted_at` and records the checkpoint, and it buys the stage extra
  attempts (`max_preemptions`) rather than spending the budget meant for stages that
  are actually broken. Which provider the next attempt goes to is the pipeline's
  decision, taken from the same attempt ledger — see `cloud.Placement`.
* **`jobs.cost_usd`, `jobs.provider` and `jobs.tier` are filled in** at every terminal
  status, from the per-stage ledgers in the workdir. Every attempt is in that total,
  including the ones that were preempted: the time a cheap host billed before it took
  the machine back was still bought.

And the audit after it, four more:

* **a stop says why** (`child.CANCEL_SIGNAL`, `child.DETACH_SIGNAL`). A cancel or a lost
  lease signals the recipe process with SIGUSR2, and it cancels its remote call on the way
  out; this worker shutting down sends SIGUSR1, and the call is left running and written down
  (`cloud.CallBook`), marks the step `detached`, and lets the next worker re-attach to
  it at the *same* attempt -- a deploy spends nothing. The book is copied onto the
  step's row (`metrics.remoteCalls`) on every heartbeat, so a worker on another machine,
  whose workdir does not have it, can still cancel the calls by id.
* **a failure is read before it is retried** (`app.worker.retry`): CUDA running out of
  memory gets one retry at a lower gaussian cap, a timeout or a broken recipe none.
* **a job has a dollar ceiling** (`cost_cap_usd`), held by the cloud runner call by
  call and here between attempts.
* **the recipe process's stderr is kept** (`CHILD_STDERR` in the workdir) and its tail
  goes into the job's error, so a process that died without a word says why.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import IO, Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, Job, JobStep
from app.models.enums import CaptureStatus, RunStatus, UploadStatus
from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker import alerts, claim, events, outputs, params, reaper, registration, retry, steps
from app.worker.child import (
    CANCEL_SIGNAL,
    DETACH_SIGNAL,
    ChildSpec,
    load_impl_modules,
    resolve_recipe,
)
from app.worker.config import WorkerConfig
from app.worker.events import Event
from app.worker.pipeline_bridge import (
    CALL_BOOK,
    PIPELINE_DIR,
    ArtifactRef,
    AttemptLedger,
    CallBook,
    CallRecord,
    PipelineError,
    Plan,
    RunCost,
    StepResult,
    Workdir,
    checksum_of,
    latest_live,
    latest_progress,
    plan_recipe,
    run_cost,
    tail_of,
)
from app.worker.publish import Publisher

log = logging.getLogger("app.worker")

#: Publishes of one finished run before its tiles are withheld: each further one is
#: because an attach moved the site while the run was publishing (registration.LiveMoved).
REPUBLISH_ATTEMPTS = 3

#: `type(PreemptedError).__name__`, as it arrives over the child's line protocol. The
#: one place the cloud runner's "the machine was taken back" is translated into a
#: supervisor decision.
PREEMPTED = "PreemptedError"

#: The recipe process's stderr, in the run's workdir. Appended to by every process of the
#: run, each one's part starting with a line that says when; its tail goes into the job's
#: error when a run fails. It used to go to /dev/null, so a process that died of an
#: uncaught exception or an import error left nothing behind but its exit code.
CHILD_STDERR = "recipe-process.stderr.log"
#: How much of it the job's error carries.
STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 2000

#: The step metrics that say a stage was stopped for a deploy, and which remote calls it
#: has in flight (`app.worker.steps`, where they are defined beside the rows).
DETACHED = steps.DETACHED
REMOTE_CALLS = steps.REMOTE_CALLS

#: How much of a running stage's log the heartbeat reads: the progress line and the live
#: viewer's lines are all near the end, and a live-cameras line is up to ~25 kB.
LIVE_TAIL_BYTES = 262_144

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
    #: True when the last stage ended because a provider took the machine back, which
    #: is ordinary operation rather than something to spend the retry budget on.
    preempted: bool = False
    #: Stage id -> the artifacts it produced, from the StepResult it sent.
    produced: dict[str, tuple[ArtifactRef, ...]] = field(default_factory=dict)
    #: Stage id -> how long its log was when this process started it: where the attempt's
    #: own lines begin, which is what a failure is classified from (`app.worker.retry`).
    log_from: dict[str, int] = field(default_factory=dict)

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
        publish_storage: ObjectStorage | None = None,
    ) -> None:
        self._sessions = session_factory
        self._storage = storage
        self._config = config
        self._id = config.worker_id
        # Optional, and defaulting to `storage`, so a caller that has one bucket keeps
        # the behaviour it had: `Publisher` treats same-bucket as nothing to publish.
        self._publish = Publisher(private=storage, public=publish_storage or storage)
        #: Why the job was dead-lettered, for the dead-man's switch's `/fail` ping.
        self._failure = ""
        #: How a ping is sent; replaced in tests.
        self.sender: alerts.Sender = alerts.send
        #: The recipe process this supervisor last started, so nothing tidies a workdir
        #: while it is still running in it (`_tidy`).
        self._process: subprocess.Popen[str] | None = None

    # --- the outer loop: attempts ------------------------------------------------

    def run(self, job_id: uuid.UUID, *, stop: threading.Event | None = None) -> Terminal:
        # The lease is renewed from its own thread for the whole supervision, not only
        # from the heartbeat in `_supervise`: this thread also downloads the capture and
        # uploads every finished stage, each for longer than a lease (`claim.LeaseKeeper`).
        # The dead-man's switch (`alerts.RunWatch`) pings from a thread of its own for the
        # same reason, and says `/start` now: a job reaches here the moment it is claimed.
        with (
            claim.LeaseKeeper(
                self._sessions,
                job_id,
                worker_id=self._config.worker_id,
                lease_s=self._config.lease_s,
                interval_s=min(self._config.poll_s, self._config.lease_s / 3),
            ),
            alerts.RunWatch(
                self._config.heartbeat_url,
                job_id,
                queue_url=self._config.queue_check_url,
                every_s=self._config.heartbeat_every_s,
                sender=self.sender,
            ) as watch,
        ):
            db = self._sessions()
            try:
                outcome = self._run(db, job_id, stop)
            except Exception as error:
                # One job must not take the worker down. A bucket that has gone away, a
                # workdir on a full disk, a bug here -- the job says what happened and the
                # loop goes on to the next one. A person can retry it from the panel,
                # which is the same affordance a dead-lettered job gets.
                log.exception("worker %s: job %s failed in the supervisor", self._id, job_id)
                outcome = self._report_supervisor_failure(job_id, error)
            finally:
                db.close()
            watch.finish(outcome, self._failure)
            return outcome

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
        try:
            # Before `_recover_calls`: a stage finished here drops the copy of its calls
            # its row still has from before they came home, which would otherwise be
            # written back as an orphaned book and cancelled -- a call long over.
            self._settle_finished(db, job, plan, workdir_root, stop)
        except outputs.UploadStopped:
            claim.release(db, job.id, worker_id=self._config.worker_id)
            return "lost"
        self._recover_calls(db, job_id, workdir_root)

        while True:
            completed = self._completed_stages(db, job_id, workdir_root)
            attempts = self._attempts(db, job_id, plan, completed)
            exhausted = self._exhausted(plan, completed, attempts, workdir_root)
            if exhausted is not None:
                stage_id, attempt, preemptions = exhausted
                last = self._last_error(db, job)
                lost = (
                    f" ({preemptions} of them ended with the provider taking the machine "
                    f"back, which is why it was allowed more than "
                    f"{self._config.max_attempts})"
                    if preemptions
                    else ""
                )
                return self._dead_letter(
                    db,
                    job,
                    f"stage {stage_id!r} has been attempted {attempt - 1} times without "
                    f"completing and will not be retried again{lost}{last}",
                )
            over = self._over_cap(workdir_root)
            if over is not None:
                return self._dead_letter(db, job, over)
            run_params, dropped = params.without_stale_roi(stage_params, plan, completed)
            if dropped:
                log.warning(
                    "worker %s: job %s recomputes its poses, so the region of interest on "
                    "%s is in a frame that no longer exists; training uncropped",
                    self._id,
                    job.id,
                    ", ".join(dropped),
                )
            state = self._supervise(db, job, workdir_root, completed, attempts, run_params, stop)
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
                return self._finish_complete(db, job, state, stop)
            if not state.failed_stage:
                # Nothing to retry from: the run never reached a stage.
                return self._dead_letter(db, job, state.error or "the run failed before any stage")
            job.error = state.error
            db.commit()
            if not state.preempted:
                decision = self._read_failure(state, attempts, workdir_root)
                if not decision.retry:
                    return self._dead_letter(db, job, f"{decision.why}: {state.error}")
                if decision.params:
                    stage_params = self._override(db, job, plan, state.failed_stage, decision)
            if state.preempted:
                # Worth saying out loud in the log: this is the cheap tier doing what the
                # cheap tier does, not the stage being broken. The next attempt resumes
                # from the checkpoint and may go to a different provider.
                log.info(
                    "worker %s: job %s stage %s was preempted; resuming it",
                    self._id,
                    job.id,
                    state.failed_stage,
                )
            # A wait on the stop flag rather than a sleep: a deploy that lands between two
            # attempts lets go of the job now, as it would have mid-stage. The failed
            # attempt is already on its row as failed, so the next worker counts the next
            # attempt from it -- nothing here is a detach.
            if stop is not None and stop.wait(self._config.retry_backoff_s):
                claim.release(db, job.id, worker_id=self._config.worker_id)
                return "lost"
            if stop is None:
                time.sleep(self._config.retry_backoff_s)
            beat = claim.heartbeat(
                db, job.id, worker_id=self._config.worker_id, lease_s=self._config.lease_s
            )
            if beat is claim.Heartbeat.CANCELLED:
                # Cancelled while it waited to try again. Not "lost": nobody else has the
                # job, and a cancel is closed out here or nowhere -- the recorded calls
                # cancelled, `cost_usd` written, the lease cleared, the workdir tidied and
                # the dead-man's switch told. Reading every not-HELD beat as lost (until
                # the 2026-10 review) skipped all of that and left the check to alert.
                return self._finish_cancelled(db, job)
            if beat is not claim.Heartbeat.HELD:
                self._log_lost(db, job.id)
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
            cloud_providers=self._config.cloud_providers,
            preemptions_before_fallback=self._config.preemptions_before_fallback,
            modal_app=self._config.modal_app,
            cloud_poll_s=self._config.cloud_poll_s,
            checkpoint_every_s=self._config.checkpoint_every_s,
            sandbox=str(self._config.sandbox_for(job.id)),
            transfer_dir=(
                str(self._config.cloud_transfer_dir) if self._config.cloud_transfer_dir else None
            ),
            cost_cap_usd=self._config.cost_cap_usd or None,
            deadline_factor=self._config.deadline_factor,
        ).write(workdir_root / "child.json")
        stderr_path = workdir_root / CHILD_STDERR
        with stderr_path.open("ab") as stderr:
            stderr.write(
                f"--- recipe process started {datetime.now(tz=UTC).isoformat()}\n".encode()
            )
            stderr.flush()
            stderr_from = stderr.tell()
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell, our own module
                [sys.executable, "-u", "-m", "app.worker.child", str(spec_path)],
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=self._child_env(),
                cwd=str(_API_ROOT),
                text=True,
            )
        state = self._watch_child(db, job, process, workdir_root, stop)
        if state.outcome == events.RUN_FAILED and process.returncode not in (None, 0):
            said = _tail_lines(stderr_path, stderr_from)
            if said:
                state.error = (
                    f"{state.error}\n--- the recipe process's stderr, last lines ---\n{said}"
                )
        return state

    def _watch_child(
        self,
        db: Session,
        job: Job,
        process: subprocess.Popen[str],
        workdir_root: Path,
        stop: threading.Event | None,
    ) -> _RunState:
        """The heartbeat loop over one recipe process, until it ends or is stopped.

        Nothing that goes wrong in here leaves the recipe process behind. The loop writes
        to the database every tick (the heartbeat, the progress line, the calls) and
        uploads every finished stage, and any of those can raise -- a connection Neon
        dropped, a bucket that answered 500. Until the 2026-10 review that exception went
        straight to `_report_supervisor_failure` with the recipe process still running:
        the job was dead-lettered, `_tidy` deleted `inputs/` and `work/` under a live
        process, its GPU call billed on, and the slot claimed another job beside the
        orphan. Now the process is stopped first, and with the signal that matches what
        happens to the job next (`_signal_after_failure`).
        """
        self._process = process
        try:
            state, current, ended = self._watch_loop(db, job, process, workdir_root, stop)
        except BaseException:
            _stop(process, self._config.terminate_grace_s, self._signal_after_failure(job.id))
            raise
        if not ended:
            return state
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
                    db,
                    current,
                    log_key=self._upload_log(job.id, workdir_root, current.stage_id),
                    checkpoint_key=self._checkpoint_key(
                        job.id, workdir_root, current.stage_id, current.attempt
                    ),
                )
        return state

    def _watch_loop(
        self,
        db: Session,
        job: Job,
        process: subprocess.Popen[str],
        workdir_root: Path,
        stop: threading.Event | None,
    ) -> tuple[_RunState, JobStep | None, bool]:
        """`_watch_child`'s loop: the state so far, the step running, and whether the
        recipe process's output ended (False: it was stopped, and `state` says why)."""
        inbox: queue.Queue[object] = queue.Queue()
        reader = threading.Thread(target=_pump, args=(process.stdout, inbox), daemon=True)
        reader.start()

        state = _RunState()
        current: JobStep | None = None
        next_beat = time.monotonic()
        while True:
            now = time.monotonic()
            if stop is not None and stop.is_set():
                # This worker is going away (a deploy), not the job: the recipe process
                # leaves its remote call running and written down, and the step is marked
                # so the next worker resumes it without spending an attempt.
                _stop(process, self._config.terminate_grace_s, DETACH_SIGNAL)
                if current is not None:
                    self._report_calls(db, current, workdir_root)
                    self._mark_detached(db, current)
                state.outcome = "stopped"
                return state, current, False
            if now >= next_beat:
                beat = claim.heartbeat(
                    db, job.id, worker_id=self._config.worker_id, lease_s=self._config.lease_s
                )
                if beat is not claim.Heartbeat.HELD:
                    # Cancelled, or reclaimed while this worker was not looking. Either
                    # way the child must stop now, not at the end of its stage. A cancel
                    # takes its remote call with it. A job reclaimed by another worker
                    # does not: that worker's recipe process re-attaches to the call --
                    # from the `calls.json` the two share, when they share a volume (two
                    # slots, `WORKER_CONCURRENCY` >= 2) -- and a cancel from here stopped
                    # the very call it had adopted and struck it from the book (until the
                    # 2026-10 review). So the call is detached for it; one on another
                    # volume is cancelled by that worker, from the row's copy of it.
                    signum = CANCEL_SIGNAL
                    if beat is claim.Heartbeat.LOST and claim.resumed_elsewhere(db, job.id):
                        signum = DETACH_SIGNAL
                    _stop(process, self._config.terminate_grace_s, signum)
                    state.outcome = "cancelled" if beat is claim.Heartbeat.CANCELLED else "lost"
                    if beat is claim.Heartbeat.LOST:
                        self._log_lost(db, job.id)
                    elif current is not None:
                        # The call is cancelled and struck from the book by now; the row's
                        # copy goes with it, or the reaper (`app.worker.reaper`) would
                        # find a call of a cancelled job and cancel it a second time.
                        self._report_calls(db, current, workdir_root)
                    return state, current, False
                if current is not None:
                    self._report_progress(db, current, workdir_root)
                    self._report_calls(db, current, workdir_root)
                next_beat = now + self._config.poll_s
            try:
                item = inbox.get(timeout=max(0.01, next_beat - time.monotonic()))
            except queue.Empty:
                continue
            if item is _EOF:
                return state, current, True
            if isinstance(item, Event):
                try:
                    current = self._apply(db, job, item, workdir_root, state, current, stop)
                except outputs.UploadStopped:
                    # A stop landed between two objects of a finished stage's upload. The
                    # stop branch at the top of the loop takes it from here: the recipe
                    # process is detached and the step left unfinished, for the next
                    # worker to finish from its `step.json` (`_settle_finished`).
                    continue

    def _signal_after_failure(self, job_id: uuid.UUID) -> int:
        """How to stop a recipe process this supervisor is abandoning on an exception.

        `run` turns the exception into a dead-letter when the job is still this worker's,
        and a dead-lettered job's call is cancelled: CANCEL, which also enters what the
        call billed in the ledger. When it is not -- reclaimed by another worker, finished
        elsewhere, or the database unreachable, in which case the dead-letter will not be
        written either and the lapsed lease hands the job to whoever resumes it -- the
        call is left running and written down for that worker to pick up: DETACH. Either
        way the dead-letter path still cancels anything left in the books
        (`_cancel_recorded_calls`), so a detach here can never outlive a dead-letter.
        Asked on a session of its own: the supervisor's may be the thing that failed.
        """
        session = self._sessions()
        try:
            job = session.get(Job, job_id)
            ours = (
                job is not None
                and job.status is RunStatus.IN_PROGRESS
                and job.claimed_by == self._config.worker_id
            )
        except Exception:
            ours = False
        finally:
            session.close()
        return CANCEL_SIGNAL if ours else DETACH_SIGNAL

    def _apply(
        self,
        db: Session,
        job: Job,
        event: Event,
        workdir_root: Path,
        state: _RunState,
        current: JobStep | None,
        stop: threading.Event | None = None,
    ) -> JobStep | None:
        if event.kind == events.STAGE_STARTED:
            state.log_from[event.stage_id] = event.log_from
            return steps.start_step(
                db,
                job.id,
                stage_id=event.stage_id,
                ordinal=event.ordinal,
                impl=event.impl,
                attempt=event.attempt,
                started_at=(
                    datetime.fromtimestamp(event.started_at, tz=UTC)
                    if event.started_at is not None
                    else None
                ),
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
            self._record_finished(db, job, step, event.step, workdir_root, stop)
            return None
        if event.kind == events.STAGE_FAILED:
            state.failed_stage = event.stage_id
            state.error = event.error
            state.error_type = event.error_type
            state.preempted = event.error_type == PREEMPTED
            if current is not None:
                # The failed attempt's calls are struck from the book by now (cancelled,
                # or ended); a copy left on the row would be cancelled again by id.
                self._report_calls(db, current, workdir_root)
                steps.fail_step(
                    db,
                    current,
                    log_key=self._upload_log(job.id, workdir_root, event.stage_id),
                    preempted=state.preempted,
                    checkpoint_key=self._checkpoint_key(
                        job.id, workdir_root, event.stage_id, event.attempt
                    ),
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

    def _record_finished(
        self,
        db: Session,
        job: Job,
        step: JobStep,
        result: dict[str, Any],
        workdir_root: Path,
        stop: threading.Event | None,
    ) -> None:
        """Upload a finished stage's artifacts and log, and finish its row from `result`,
        its `StepResult` as a dict. Idempotent -- the keys are the stage's own and a
        retried upload writes the same ones -- which is what lets `_settle_finished` call
        it again for a stage whose worker stopped in the middle of it. Raises
        `outputs.UploadStopped` between two objects once `stop` is set."""
        refs = tuple(ArtifactRef.from_dict(entry) for entry in result.get("artifacts", ()))
        stopping = stop.is_set if stop is not None else None
        # Uploads are minutes for a large stage (a 514-tile package): end the
        # transaction first so the session is not left idle inside one.
        _end_transaction(db)
        uploaded = [
            artifact
            for ref in refs
            if (
                artifact := outputs.upload_artifact(
                    self._storage, workdir_root, job.id, ref, stopping=stopping
                )
            )
        ]
        # The final live-cameras line lands just before the stage ends; read it now,
        # before a heartbeat could, so the finished step keeps it for the viewer.
        self._report_progress(db, step, workdir_root)
        steps.finish_step(
            db,
            step,
            metrics=_metrics(result),
            log_key=self._upload_log(job.id, workdir_root, step.stage_id),
            checkpoint_key=_optional_str(result.get("checkpointKey")),
            artifacts=uploaded,
        )

    def _settle_finished(
        self,
        db: Session,
        job: Job,
        plan: Plan,
        workdir_root: Path,
        stop: threading.Event | None,
    ) -> None:
        """Finish the rows of stages that finished while their worker was not looking.

        The recipe process does not wait for the supervisor: it writes a stage's
        `step.json`, reports it, and goes on to the next stage while the supervisor
        uploads the one that finished. A worker stopped in the middle of that upload -- a
        deploy's SIGTERM waited for it until fly.toml's 30 s `kill_timeout` SIGKILLed the
        worker -- left the stage finished on disk and `in-progress` on its row, and the
        next worker ran it again: for `train`, hours of GPU billed twice. So before
        anything runs, each stage in order that is not complete on its row but has this
        attempt's proof of finishing (`_proof_of_finish`) has its upload redone and its
        row finished, and is skipped like any completed stage. The first stage without
        that proof is where the run resumes, and nothing after it is looked at.
        """
        rows = steps.steps_of(db, job.id)
        workdir = Workdir(workdir_root)
        for stage in plan.stages:
            row = rows.get(stage.id)
            if row is not None and row.status is RunStatus.COMPLETE:
                if workdir.step_path(stage.id).is_file():
                    continue
                return
            result = self._proof_of_finish(row, workdir, stage.id)
            if row is None or result is None:
                return
            log.info(
                "worker %s: job %s stage %s finished (attempt %d) while its worker was "
                "recording it; finishing its upload instead of running it again",
                self._id,
                job.id,
                stage.id,
                row.attempt,
            )
            self._record_finished(db, job, row, result, workdir_root, stop)

    @staticmethod
    def _proof_of_finish(
        row: JobStep | None, workdir: Workdir, stage_id: str
    ) -> dict[str, Any] | None:
        """The stage's `step.json`, if it proves the attempt on `row` finished; else None.

        All of it has to hold, because a `step.json` outlives the attempt that wrote it --
        a Refine re-runs `train` beside the preview's, and a killed attempt leaves the
        last successful one's:

        * the row says that attempt is still running (`in-progress`);
        * the file names that attempt and that stage;
        * it was written after the row's attempt started -- not an earlier run's, which
          is what a re-run stage finds beside it;
        * every artifact it lists is in `out/` with the checksum it gives, so `out/` was
          neither wiped by a later start nor half rewritten.

        The row's `started_at` is the recipe process's own stamp of the start
        (`events.Event.started_at`), taken before `out/` is cleared and so before this
        attempt's `step.json` can exist, on the clock that file's mtime is on. A row the
        supervisor had to start itself, with its own clock, may fail the third and be run
        again, which is what happened to every stage before this: nothing that fails it
        is mistaken for finished.
        """
        if row is None or row.status is not RunStatus.IN_PROGRESS or row.started_at is None:
            return None
        path = workdir.step_path(stage_id)
        try:
            written = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            document = json.loads(path.read_text(encoding="utf-8"))
            result = StepResult.from_dict(document)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if result.stage_id != stage_id or result.attempt != row.attempt:
            return None
        if written < row.started_at:
            return None
        for ref in result.artifacts:
            produced = workdir.root / ref.path
            if not produced.exists() or checksum_of(produced) != ref.checksum:
                return None
        return dict(document)

    def _tiles_dir(self, job: Job, state: _RunState) -> Path | None:
        """The run's packaged tileset in its workdir, if it is still there: what a carried
        kind bound to positions is checked against (`carry.plan_carry`). A workdir another
        machine has is not here, and the check reads the bucket instead."""
        ref = state.ref("splat")
        if ref is None:
            return None
        found = self._config.workdir_for(job.id) / ref.path
        return found if found.is_dir() else None

    # --- terminal states ----------------------------------------------------------

    def _finish_complete(
        self, db: Session, job: Job, state: _RunState, stop: threading.Event | None = None
    ) -> Terminal:
        ref = state.ref("registration.json")
        document = (
            registration.Registration.read(self._config.workdir_for(job.id) / ref.path)
            if ref is not None
            else None
        )
        published = None
        for attempt in range(1, REPUBLISH_ATTEMPTS + 1):
            if document is not None:
                # Publishing is minutes for a large capture, and a deploy's SIGTERM that
                # waited for it was SIGKILLed at the 30 s `kill_timeout`. Every stage is
                # finished on its row by now, so a worker that lets go here costs the next
                # one a publish -- idempotent copies -- and no stage.
                if self._stopping(db, job, stop):
                    return "lost"
                # The tileset the site shows now: the sidecars beside it that still hold
                # for the new tiles are carried into the new generation (worker/carry.py).
                live = registration.live_tileset_url(db, db.get(Capture, job.capture_id))
                # Copy to the public bucket first, with no transaction open: for a large
                # capture it takes minutes, and a session idle in a transaction that long
                # is killed by the database (see registration.publish_outputs).
                _end_transaction(db)
                published = registration.publish_outputs(
                    self._storage,
                    publish=self._publish,
                    job_id=job.id,
                    registration=document,
                    tiles_stage_id=state.stage_producing("splat"),
                    thumbnail_stage_id=state.stage_producing("thumbnail.jpg"),
                    coverage_stage_id=state.stage_producing("coverage_enu.ply"),
                    carry_from=live,
                    tiles_dir=self._tiles_dir(job, state),
                )
                if self._stopping(db, job, stop):
                    return "lost"
            if not self._still_ours(db, job):
                return "lost"
            capture = db.get(Capture, job.capture_id)
            if capture is not None and document is not None:
                try:
                    registration.register(
                        db,
                        self._storage,
                        publish=self._publish,
                        capture=capture,
                        job_id=job.id,
                        registration=document,
                        tiles_stage_id=state.stage_producing("splat"),
                        thumbnail_stage_id=state.stage_producing("thumbnail.jpg"),
                        coverage_stage_id=state.stage_producing("coverage_enu.ply"),
                        published=published,
                    )
                except registration.LiveMoved as moved:
                    # An attach cut a generation while this run was publishing, and
                    # repointing would drop what it attached. Nothing was written; publish
                    # again on top of it.
                    db.rollback()
                    log.warning(
                        "worker: job %s: the site moved to %s while publishing (attempt %d); "
                        "publishing again on top of it",
                        job.id,
                        moved.url,
                        attempt,
                    )
                    continue
            elif capture is not None:
                # A recipe with no `register` stage still finished; the capture is processed
                # even though there is nothing to put on the globe.
                capture.status = CaptureStatus.COMPLETE
            break
        else:
            # The site kept moving under every publish. Its new tiles are withheld, as for a
            # publish that failed: the site stays as the attaches left it, and the run --
            # which succeeded -- still completes.
            log.error(
                "worker: job %s: the site moved under %d publishes; its tiles are withheld",
                job.id,
                REPUBLISH_ATTEMPTS,
            )
            if not self._still_ours(db, job):
                return "lost"
            capture = db.get(Capture, job.capture_id)
            if capture is not None and document is not None:
                registration.register(
                    db,
                    self._storage,
                    publish=self._publish,
                    capture=capture,
                    job_id=job.id,
                    registration=document,
                    tiles_stage_id=state.stage_producing("splat"),
                    published=registration.Published(
                        tileset=None, thumbnail=None, coverage=None, withheld=True
                    ),
                )
        job.status = RunStatus.COMPLETE
        job.error = None
        self._close(job)
        db.commit()
        if self._config.tidy_finished_runs:
            self._tidy(self._config.workdir_for(job.id))
        return "complete"

    def _stopping(self, db: Session, job: Job, stop: threading.Event | None) -> bool:
        """True, having let go of the job, when this worker is shutting down."""
        if stop is None or not stop.is_set():
            return False
        claim.release(db, job.id, worker_id=self._config.worker_id)
        return True

    def _tidy(self, workdir_root: Path) -> None:
        """After a run that ended, drop what the workdir contract says is disposable.

        Every stage's `work/` is scratch by A6's own definition ("safe to delete at any
        time"), and `inputs/` is a copy of what is still in the bucket -- `_seed` fetches
        it again if a retry ever needs it. Both are the bulk of a Lane 2 run: the uploaded
        video, every candidate frame ffmpeg extracted before selection, COLMAP's database.
        On a 20 GB worker volume, keeping them would fill it in three or four captures.
        `out/`, `step.json` and `checkpoint/` stay, which is all retry-from-stage reads.
        A failed or cancelled run is tidied the same way: a retry fetches its inputs
        again exactly as a retry of a finished one does. A failure to tidy is logged and
        nothing else: the run is over either way.

        Never under a recipe process that is still running in the workdir: deleting its
        `inputs/` and `work/` would fail the stage in ways that say nothing about why.
        `_watch_child` stops the process before any way out of it, so this is the last
        line, not the first; a process found alive here is stopped as a cancel first,
        since every caller is closing the run.
        """
        if self._process is not None and self._process.poll() is None:
            log.error(
                "worker %s: the recipe process %d was still running when its run ended; "
                "stopping it before tidying %s",
                self._id,
                self._process.pid,
                workdir_root,
            )
            _stop(self._process, self._config.terminate_grace_s, CANCEL_SIGNAL)
        workdir = Workdir(workdir_root)
        doomed = [workdir.inputs_dir]
        if workdir.stages_dir.is_dir():
            doomed += [stage / "work" for stage in sorted(workdir.stages_dir.iterdir())]
        for path in doomed:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
            except OSError:
                log.warning("worker: could not tidy %s after a finished run", path, exc_info=True)

    def _finish_cancelled(self, db: Session, job: Job) -> Terminal:
        """`POST /jobs/{id}/cancel` already set the status; this closes out the run.

        The workdir is left exactly as the killed stage left it: its `out/` may be half
        written, which is precisely why A6 clears `out/` at the start of every attempt.
        Nothing half-written is uploaded, and no `artifacts` row is created for it.
        """
        workdir_root = self._config.workdir_for(job.id)
        # The recipe process cancelled its call when told to; this catches one it could not
        # (killed before it got to it), so a cancelled job leaves no GPU running.
        _end_transaction(db)
        self._cancel_recorded_calls(job.id, workdir_root)
        steps.stop_active_steps(db, job.id, RunStatus.CANCELLED)
        db.refresh(job)
        if job.finished_at is None:
            job.finished_at = datetime.now(tz=UTC)
        job.lease_expires_at = None
        # A cancelled run still ran, and a GPU still billed for the part of it that did.
        self._record_cost(job)
        db.commit()
        if self._config.tidy_finished_runs:
            self._tidy(workdir_root)
        return "cancelled"

    def _dead_letter(self, db: Session, job: Job, why: str) -> Terminal:
        """Stop trying, and say why in the place the panel already renders.

        Any remote call still written down for the run is cancelled first -- a worker
        that crashed on the last attempt it was allowed leaves one running, and nothing
        would ever pick it up again. Then the workdir is tidied like a finished run's:
        a failed run kept its inputs and every stage's scratch until the volume filled,
        though a retry fetches the inputs again and scratch is scratch.
        """
        if not self._still_ours(db, job):
            return "lost"
        workdir_root = self._config.workdir_for(job.id)
        _end_transaction(db)
        self._cancel_recorded_calls(job.id, workdir_root)
        steps.stop_active_steps(db, job.id, RunStatus.ERROR)
        db.refresh(job)
        job.status = RunStatus.ERROR
        job.error = why
        self._failure = why
        self._close(job)
        capture = db.get(Capture, job.capture_id)
        if capture is not None:
            capture.status = CaptureStatus.ERROR
        db.commit()
        if self._config.tidy_finished_runs:
            self._tidy(workdir_root)
        return "error"

    def _close(self, job: Job) -> None:
        now = datetime.now(tz=UTC)
        job.finished_at = now
        if job.claimed_at is not None:
            job.duration_s = (now - job.claimed_at).total_seconds()
        # The lease is over; `claimed_by` stays, because who ran it is a fact worth keeping.
        job.lease_expires_at = None
        self._record_cost(job)

    def _record_cost(self, job: Job) -> None:
        """What this run was billed, from the per-stage ledgers the runner wrote.

        Every attempt of every stage is in the total, preempted ones included: a cost
        that counted only the attempt that happened to succeed would say a stage which
        was killed twice on a cheap host cost a third of what it did, and the cheap host
        would keep looking cheap.

        `cost_usd` stays null when no tier the run used has a rate anybody has measured
        — `tools/pipeline/providers.py` ships only surveyed figures and an operator's own
        rates come from `PIPELINE_GPU_RATES`. `provider` and `tier` are still recorded in
        that case, because where the work ran is a fact either way, and the seconds are
        in the step's metrics.
        """
        total: RunCost = run_cost(Workdir(self._config.workdir_for(job.id)))
        if total.provider is None:
            return  # Nothing was dispatched: a stub or local run has no bill.
        job.provider = total.provider
        job.tier = total.tier
        if total.usd is not None:
            job.cost_usd = Decimal(str(round(total.usd, 4)))
        log.info(
            "job %s: %.1f billed second(s) on %s (%s) over %d attempt(s), %d preemption(s); "
            "cost %s",
            job.id,
            total.billed_s,
            total.provider,
            total.tier,
            total.attempts,
            total.preemptions,
            "unpriced" if total.usd is None else f"${total.usd:.4f}",
        )

    def _log_lost(self, db: Session, job_id: uuid.UUID) -> None:
        """The line that says a job was taken from under this slot, and by whom."""
        log.warning(
            "worker %s: lost job %s mid-run: %s", self._id, job_id, claim.holder(db, job_id)
        )

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

    @staticmethod
    def _report_progress(db: Session, step: JobStep, workdir_root: Path) -> None:
        """Copy the newest progress line in a running stage's log onto its row.

        On the heartbeat, so it costs one small read a tick. A stage that prints no
        progress line (most of them) leaves the row alone.
        """
        # Enough of the log to hold the newest live-cameras line (~25 kB) with room over.
        text = tail_of(Workdir(workdir_root).log_path(step.stage_id), LIVE_TAIL_BYTES)
        found = latest_progress(text)
        if found is not None:
            steps.report_progress(db, step, found.to_dict())
        # What the live viewer draws: the newest cameras and intermediate splat the stage
        # has logged (`tools/pipeline/live.py`), under `metrics.live`.
        live = latest_live(text)
        if live:
            steps.report_live(db, step, live)

    def _upload_log(self, job_id: uuid.UUID, workdir_root: Path, stage_id: str) -> str | None:
        return outputs.upload_log(self._storage, workdir_root, job_id, stage_id)

    @staticmethod
    def _checkpoint_key(
        job_id: uuid.UUID, workdir_root: Path, stage_id: str, attempt: int = 1
    ) -> str | None:
        """The key of a stage's checkpoint, or None when there is nothing in it.

        Recorded on an attempt that did *not* finish, which is the case with no
        StepResult to read it out of — and the case where it matters most, because it is
        what the next attempt resumes from. The attempt's own key: each attempt syncs to
        one of its own (`outputs.checkpoint_key`).
        """
        directory = Workdir(workdir_root).checkpoint_dir(stage_id)
        if not directory.is_dir() or not any(directory.iterdir()):
            return None
        return outputs.checkpoint_key(job_id, stage_id, attempt)

    def _seed(self, db: Session, job: Job, inputs: tuple[str, ...], workdir_root: Path) -> None:
        """Put the capture's uploaded bytes where the recipe says its inputs live.

        A file already in the workdir is not fetched again, so a reclaimed or retried job
        does not download a 12 GB video again. Streamed to disk, not read into memory:
        the worker machine has 2 GB and an iPhone video is routinely larger than that,
        and a whole-object read of one killed the worker before the first stage ran.
        """
        work = Workdir.create(workdir_root)
        capture = db.get(Capture, job.capture_id)
        # What to fetch, read now, as plain values: the downloads below take minutes for
        # a large video, and the session's transaction -- opened by these reads and the
        # ones before them -- must not sit idle across them. Neon terminates a connection
        # idle in a transaction, and the next statement on it fails (`_end_transaction`).
        wanted = [
            (source.storage_key, Path(source.filename).name)
            for source in (capture.files if capture is not None else ())
            if source.status is UploadStatus.COMPLETE
        ]
        _end_transaction(db)
        for name in inputs:
            if name != "upload" or capture is None:
                continue
            target = work.input_path(name)
            target.mkdir(parents=True, exist_ok=True)
            for key, filename in wanted:
                # File by file, not "the directory has something in it": a worker that
                # stopped between two files, or mid-file (`download_file` writes beside
                # the target and renames), left a directory that is not empty and not
                # complete, and a resume that skipped it ran the recipe on half a capture.
                path = target / filename
                if path.is_file():
                    continue
                self._storage.download_file(key, path)

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
        """The attempt each stage still to run is about to make.

        One more than the row's -- except for a stage this worker's predecessor was
        stopped in the middle of for a deploy (`DETACHED`): that attempt did not fail, its
        remote call may well still be running and is about to be re-attached to, so it
        carries on as the same attempt. A crash is not a detach and still counts: a stage
        that keeps killing its worker must run out of attempts.
        """
        rows = steps.steps_of(db, job_id)
        attempts: dict[str, int] = {}
        for stage in plan.stages:
            if stage.id in completed:
                continue
            row = rows.get(stage.id)
            if row is None:
                attempts[stage.id] = 1
            elif row.status is RunStatus.IN_PROGRESS and (row.metrics or {}).get(DETACHED):
                attempts[stage.id] = row.attempt
            else:
                attempts[stage.id] = row.attempt + 1
        return attempts

    def _exhausted(
        self, plan: Plan, completed: set[str], attempts: dict[str, int], workdir_root: Path
    ) -> tuple[str, int, int] | None:
        """The attempt budget, extended rather than replaced.

        `max_attempts` is the budget for a stage that is failing. An attempt the provider
        ended -- a preemption -- is not that, so it does not spend it; the stage gets
        `max_preemptions` of those on top. The cap is still hard, because a placement
        that keeps losing the machine after the fallback is a problem with the placement
        and retrying it forever would pay for the same hours again and again.
        """
        for stage in plan.stages:
            if stage.id in completed:
                continue
            attempt = attempts.get(stage.id, 1)
            preemptions = self._preemptions(workdir_root, stage.id)
            allowance = self._config.max_attempts + min(preemptions, self._config.max_preemptions)
            return (stage.id, attempt, preemptions) if attempt > allowance else None
        return None

    @staticmethod
    def _preemptions(workdir_root: Path, stage_id: str) -> int:
        """How often this stage has had its machine taken back, from the pipeline's own
        ledger. The same file `cloud.Placement` reads to decide where to send it next,
        so the budget here and the provider choice there cannot disagree."""
        return AttemptLedger.read(Workdir(workdir_root).attempts_path(stage_id)).preemptions

    def _last_error(self, db: Session, job: Job) -> str:
        db.refresh(job)
        return f": {job.error}" if job.error else ""

    # --- remote calls that outlive a recipe process ---------------------------------

    @staticmethod
    def _book(workdir_root: Path, stage_id: str) -> CallBook:
        return CallBook.read(Workdir(workdir_root).stage_dir(stage_id) / CALL_BOOK)

    @staticmethod
    def _report_calls(db: Session, step: JobStep, workdir_root: Path) -> None:
        """Copy the stage's `CallBook` onto its row, as `metrics.remoteCalls`.

        The workdir is on this machine's volume; the row is where a worker anywhere can
        read it. A job reclaimed on another machine -- a different volume, or this one
        replaced -- has no book, and this copy is how its calls are still found and
        cancelled by id (`_recover_calls`). Ids, providers and when, not the requests:
        enough to cancel, which is all a call without its workdir is good for.
        """
        book = JobSupervisor._book(workdir_root, step.stage_id)
        calls = {
            slot: {key: value for key, value in record.to_dict().items() if key != "request"}
            for slot, record in sorted(book.calls.items())
        }
        metrics = dict(step.metrics or {})
        if metrics.get(REMOTE_CALLS, {}) == calls:
            return
        if calls:
            metrics[REMOTE_CALLS] = calls
        else:
            metrics.pop(REMOTE_CALLS, None)
        step.metrics = metrics
        db.commit()

    @staticmethod
    def _mark_detached(db: Session, step: JobStep) -> None:
        """Say on the row that this attempt was interrupted by a deploy (`_attempts`)."""
        step.metrics = {**(step.metrics or {}), DETACHED: True}
        db.commit()

    def _recover_calls(self, db: Session, job_id: uuid.UUID, workdir_root: Path) -> None:
        """Put back, as `orphaned` books, the calls the database knows of and this workdir
        does not: the job was last run on another volume, or this one lost its files.

        The recipe process cancels an orphaned book before it runs anything
        (`CloudRunner.reap`) rather than re-attaching to it -- the inputs those calls were
        given came from a workdir that is gone, and the run starts over -- so a GPU left
        running by a worker that died on another machine stops now, not in six hours.
        """
        for stage_id, step in steps.steps_of(db, job_id).items():
            if step.status is RunStatus.COMPLETE:
                continue
            known = (step.metrics or {}).get(REMOTE_CALLS)
            path = Workdir(workdir_root).stage_dir(stage_id) / CALL_BOOK
            if not known or not isinstance(known, dict) or path.exists():
                continue
            calls = {
                str(slot): CallRecord.from_dict(entry)
                for slot, entry in known.items()
                if isinstance(entry, dict) and entry.get("id")
            }
            if calls:
                log.warning(
                    "worker %s: job %s stage %s had remote call(s) %s in flight on a workdir "
                    "this worker does not have; cancelling them",
                    self._id,
                    job_id,
                    stage_id,
                    ", ".join(record.handle.id for record in calls.values()),
                )
                CallBook(path, calls=calls, orphaned=True).save()
        _end_transaction(db)

    def _cancel_recorded_calls(self, job_id: uuid.UUID, workdir_root: Path) -> None:
        """Cancel every remote call still written down in the run's workdir, from here.

        For a run that is over while a call may not be: dead-lettered after a crash on its
        last attempt, or cancelled after its recipe process was killed before it could
        cancel. Built from the same configuration the recipe process uses, so the same
        adapters; nothing to do (and nothing built) for a run with no books. Each call
        cancelled is struck from the step rows' copies too. Never raises: the job is
        being closed, and a provider being unreachable must not stop that -- it is
        logged, the row keeps its copy of the call, and the reaper (`app.worker.reaper`)
        tries it again on its next pass.
        """
        workdir = Workdir(workdir_root)
        if self._config.runner != "cloud" or not workdir.stages_dir.is_dir():
            return
        if not any(workdir.stages_dir.glob(f"*/{CALL_BOOK}")):
            return
        try:
            runner = reaper.calls_runner(
                self._storage, self._config, self._config.sandbox_for(workdir_root.name)
            )
            cancelled = runner.reap(workdir, keep=None) if runner is not None else []
            if cancelled:
                log.info("worker %s: cancelled remote call(s) %s", self._id, cancelled)
                reaper.strike_calls(self._sessions, job_id, dict.fromkeys(cancelled, "cancelled"))
        except Exception:
            log.exception("worker %s: could not cancel the remote calls in %s", self._id, workdir)

    # --- what a failure is, and what it may cost ------------------------------------

    def _over_cap(self, workdir_root: Path) -> str | None:
        """Why not to start another attempt, when the run has spent its cap; else None.

        Between attempts, from the ledgers. Not while a call is written down as still out
        there: that one is billing as well, and the recipe process is what can both price
        it and cancel it (`CloudRunner`), so it is left to hold the cap.
        """
        cap = self._config.cost_cap_usd
        workdir = Workdir(workdir_root)
        if not cap or cap <= 0:
            return None
        if workdir.stages_dir.is_dir() and any(workdir.stages_dir.glob(f"*/{CALL_BOOK}")):
            return None
        spent = run_cost(workdir).usd
        if spent is None or spent < cap:
            return None
        return (
            f"the run has been billed ${spent:.2f}, at or over its ${cap:.2f} cap "
            f"(WORKER_JOB_COST_CAP_USD); not starting another attempt"
        )

    def _read_failure(
        self, state: _RunState, attempts: dict[str, int], workdir_root: Path
    ) -> retry.Decision:
        """Classify a stage's failure from its error and its attempt's log, decide on the
        next attempt (`app.worker.retry`), and write the failure into its history."""
        stage_id = state.failed_stage
        log_path = Workdir(workdir_root).log_path(stage_id)
        logged = _read_from(log_path, state.log_from.get(stage_id, 0))
        failure = retry.classify(state.error_type, state.error, logged)
        attempt = attempts.get(stage_id, 1)
        history = retry.History.of(Workdir(workdir_root).stage_dir(stage_id))
        if attempt <= 1:
            # The stage is starting over -- a person's Retry resets the attempts -- and
            # what an earlier run of it failed of is that run's.
            history.entries.clear()
        decision = retry.decide(stage_id, failure, history)
        history.add(attempt, failure)
        if failure.kind != "other":
            log.info(
                "worker %s: stage %s failed (%s): %s",
                self._id,
                stage_id,
                failure.kind,
                "retrying" + (f" with {decision.params}" if decision.params else "")
                if decision.retry
                else "not retrying",
            )
        return decision

    def _override(
        self, db: Session, job: Job, plan: Plan, stage_id: str, decision: retry.Decision
    ) -> dict[str, dict[str, Any]]:
        """Write a retry's parameters into `jobs.params[stage]` -- the existing per-run
        override, so the change is on the job for anyone to see and outlives this worker
        -- and return the run's parameters with it applied."""
        current = dict(job.params or {})
        current[stage_id] = {**dict(current.get(stage_id) or {}), **decision.params}
        job.params = current
        job.error = (
            f"{job.error or 'the stage failed'}\n--- retrying stage {stage_id!r} with "
            f"{json.dumps(decision.params, sort_keys=True)} ---"
        )
        db.commit()
        return self._stage_params(db, job, plan)


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


def _stop(process: subprocess.Popen[str], grace_s: float, signum: int = CANCEL_SIGNAL) -> None:
    """`signum` -- which says why (`child.CANCEL_SIGNAL`, `child.DETACH_SIGNAL`) -- then
    SIGKILL. A stage that ignores the first does not get to keep running."""
    if process.poll() is not None:
        return
    process.send_signal(signum)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


#: The most of a failed attempt's log a failure is classified from: the end of it, where
#: a traceback is, with room for a training stage's chatter before it.
FAILURE_LOG_BYTES = 262_144


def _read_from(path: Path, offset: int, limit: int = FAILURE_LOG_BYTES) -> str:
    """What was written to `path` after `offset`, at most its last `limit` bytes."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(offset, end - limit, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail_lines(path: Path, offset: int) -> str:
    """The last lines this process wrote to its stderr, for a job's error: bounded in
    lines and characters, because the error is a column a person reads in a panel."""
    lines = [line for line in _read_from(path, offset).splitlines() if line.strip()]
    tail = "\n".join(lines[-STDERR_TAIL_LINES:])
    return tail if len(tail) <= STDERR_TAIL_CHARS else "..." + tail[-STDERR_TAIL_CHARS:]


def _end_transaction(db: Session) -> None:
    """Commit whatever the session holds so it is not idle in a transaction.

    Called before long object-store I/O. Neon (like any Postgres with
    `idle_in_transaction_session_timeout`) terminates a connection left idle inside a
    transaction, and the next statement on it then fails. Committing here writes nothing
    the step would not have written anyway: every change made so far is already meant
    to be durable.
    """
    if db.in_transaction():
        db.commit()
