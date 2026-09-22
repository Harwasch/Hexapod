# app/worker — the thing that turns a queued job into a running pipeline

```bash
cd apps/api
uv run python -m app.worker          # poll forever (what C1 deploys beside the API)
uv run python -m app.worker --once   # claim and run one job, then exit
```

The API inserts a `not-started` job and reads status back. This package is everything in
between: claim it, run its recipe, write each stage's transition as it happens, put the
logs and artifacts in object storage, register the finished capture, and give up in a way
that says why.

## Where it lives, and why here

`tools/pipeline` must not depend on `apps/api` — no models, no session, no HTTP. That rule
is about one direction. This package is the other one: it already owns the ORM models and
the session factory, so the claim loop needs no duplicate schema and a step transition is
a `commit()` rather than a round trip through the API it would be part of. The pipeline
stays a library, imported through `pipeline_bridge.py`, which is the only module that
touches `sys.path` and the only place those generic module names (`plan`, `registry`,
`contracts`) are ever spelled.

Nothing in the API imports this package, so `apps/api` still starts with the pipeline
absent from disk.

## The claim is a lease, not a held lock

A0 measured the alternative and A2's schema comments carry the numbers: `SELECT … FOR
UPDATE SKIP LOCKED` held open for a job's duration releases in 0.02 s on SIGKILL, and
holds the row for about **2 h 51 min** when the worker is merely frozen — SIGSTOP, a
wedged host, a partition — because that is when the default TCP keepalives give up. The
open transaction pins the vacuum horizon the whole time. A committed claim plus a lease
reclaims in 3.06 s however the worker died.

So `claim.py` does exactly one statement: `SKIP LOCKED` selects the oldest claimable row
inside an `UPDATE` that sets `claimed_by`, `claimed_at` and `lease_expires_at`, and it
commits immediately. "Claimable" is `claim.claimable()`, which is also what
`tests/test_capture_models.py` asserts on — one predicate, not two that can disagree.

## The heartbeat is a different process from the work

A stage can run for hours (B2 trains on a GPU), so the heartbeat cannot be "between
stages", and a thread cannot be interrupted when the job is cancelled. So the recipe runs
in a **child process** — `app.worker.child`, one JSON spec in, one JSON event per line out
— and the supervisor stays free:

```
JobSupervisor._supervise, every poll_s (2 s by default):
  ├─ heartbeat: push lease_expires_at forward, and read jobs.status back
  ├─ drain whatever the child has reported, and commit it
  └─ has the child exited?
```

Three consequences worth stating:

- **cancellation lands mid-stage.** `POST /jobs/{id}/cancel` sets the status; the next
  heartbeat sees it, SIGTERMs the child and SIGKILLs it after `terminate_grace_s`. Worst
  case is one poll interval plus the grace — seconds, not the length of the stage;
- **a stage that segfaults is a failed step, not a lost worker**;
- **the child has no database and no credentials.** Everything that touches the bucket or
  the session happens in the supervisor, from the events the child sends.

The child also watches its own parent: a worker that is SIGKILLed cannot clean up, and an
orphaned recipe process writing into a workdir another worker is about to resume would be
two processes in one `out/`. When `getppid()` changes, the child exits immediately.

## What happens when a worker dies mid-stage

Nothing runs to tidy up, so the row stays `in-progress` and `lease_expires_at` simply
passes. After that the job is claimable by anyone, and the worker that takes it:

- **skips** the stages that are `complete` in the database *and* still have their
  `step.json` in the workdir — their outputs are reused, not recomputed;
- **re-runs** the interrupted stage with `attempt` incremented. A6 clears `out/` at the
  start of every attempt and keeps `checkpoint/`, so a half-written output can never be
  mistaken for a produced artifact and a stage that checkpoints resumes rather than
  restarts;
- **restarts from the beginning** if the workdir is gone — a different machine, a cleaned
  disk. That is the honest answer; a resume whose inputs are missing is not.

A worker asked to stop politely (SIGTERM, SIGINT) does not wait for the stage to finish:
it stops the recipe process on its next tick and **clears** the lease instead of leaving
it to lapse, so the job is claimable at once.

## Giving up: the dead-letter path

`job_steps.attempt` is one budget covering both ways a stage fails to finish — it raised,
or its worker died. When the next attempt would exceed `worker_max_attempts` (3), the job
goes to `error` with a message naming the stage, the count and the last error, and nothing
claims it again. Without that cap a capture that kills its worker would be picked up
forever by whoever is next.

A recipe that will not resolve at all dead-letters on the first look, because it will not
resolve on the second either.

**Retrying is then a person's decision**, through `POST /jobs/{id}/retry`: it resumes at
`fromStage` (or at the stage that failed), keeps every completed step and its artifacts,
and resets the attempt budget of the stages being re-run — otherwise "Retry" on a
dead-lettered job would fail instantly and say nothing new.

## Who registers the finished capture

The `register` stage writes `registration.json` — slug, title, recipe, georeference, the
artifacts it produced — and **the worker performs the registration**. The alternative was
a registration endpoint the pipeline calls; that would have put an HTTP client and a token
into a project whose whole point is that it has neither, needed a second authorisation
path for a caller already inside the trust boundary, and left a run that succeeded but
registered nothing when the POST failed.

## What lands where

| Thing | Where it goes |
| --- | --- |
| step transitions | `job_steps`, committed **as each stage starts and finishes** |
| stage logs | object storage, `runs/<job id>/<stage id>/log.txt` → `job_steps.log_key` |
| artifacts | object storage, `runs/<job id>/<stage id>/<name>`, one `artifacts` row each |
| the workdir | `WORKER_WORKDIR/<job id>`, kept after the run — retry-from-stage reads it |
| the site | `sites` + the capture's `site_id`, from `registration.json` |

A deployment with no bucket still runs: the log and artifact uploads are skipped and say
so by leaving `log_key` null, rather than failing the job.

## Configuration

`WORKER_*` in the environment, read through `app.config.Settings` into `WorkerConfig`:
`WORKDIR`, `RUNNER` (`stub` until A8 makes Lane 1's stages real), `RECIPE_DIR`,
`IMPL_MODULES`, `LEASE_S`, `POLL_S`, `IDLE_S`, `MAX_ATTEMPTS`, `RETRY_BACKOFF_S`.

## Verify

```bash
cd apps/api && uv run ruff check . && uv run mypy . && uv run pytest -q tests/test_worker.py
```
