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

## Several jobs at once

`WORKER_CONCURRENCY=N` (default 1) runs N **slots** in one worker process (`loop.py`). A
slot is what the whole worker used to be -- claim one job, supervise it to the end in its
own recipe process, claim the next -- and nothing about a job is shared between slots:

- each slot claims with its own id, `host:pid/<n>` (one slot keeps the plain `host:pid`),
  so `claimed_by`, the heartbeat and `_still_ours` behave exactly as between two separate
  workers: a lease one slot let lapse and another reclaimed is *lost* to the first;
- each job's lease is renewed by its own slot's heartbeat, and a cancel is seen by that
  heartbeat and stops that job's recipe process only;
- SIGTERM sets one stop flag: every slot stops its own recipe process and clears its own
  lease, so every job is claimable at once;
- `--max-jobs` counts across slots, and a slot reserves its place before claiming.

`tests/test_worker_concurrency.py` runs two 30-second jobs in two slots against Postgres:
both run at once under distinct owners, both leases advance past their length, a cancel
stops one while the other keeps running, and a stop hands the survivor back.

**What N the 2 GB machine can take** is a memory question -- the slots' threads only
wait, and the GPU and CPU-heavy stages (`pose`, `train`, `quality`) run on Modal. Measured
2026-09-27 (`/proc` RSS and `getrusage` peaks, on this repository's code):

| process | RSS |
| --- | --- |
| the worker (supervisor) after its imports and a DB session | ~105 MB, plus a few MB a slot |
| a job's recipe process, idle while Modal runs a stage (pipeline, numpy, PIL, modal) | ~80 MB |
| `normalize` of a 25 s 4K HEVC clip (ffmpeg decoding, 100 frames kept) | peak ~380 MB (ffmpeg) + the 80 |
| `place`/`package`/`thumbnail` of a 500k-gaussian SH3 splat (Lane 1 recipe, same code) | peak ~395 MB |
| the same at 1M gaussians (the phone's "Best" cap) | peak ~750 MB |

So a job spends most of its life at ~80 MB and peaks at 0.4-0.75 GB for a few seconds at
either end. **N = 2 is safe on the 2 GB machine** for phone captures up to 1M gaussians:
two worst-case peaks landing together are ~1.6 GB with the supervisor. N = 3 fits only
while peaks do not coincide, which nothing guarantees, so it wants the 4 GB machine
(`fly scale memory 4096 --process-group worker`, or the `[[vm]]` block in `fly.toml`); so
does any N with
Lane 1 uploads of several million gaussians, which scale at ~0.75 GB per million. The
20 GB volume holds about two captures' workdirs in flight, so N > 2 wants it extended
too. If the machine does run out, the kernel kills the largest process -- a recipe
process, whose stage then fails and is retried under the attempt budget -- not the
supervisor.

## Configuration

`WORKER_*` in the environment, read through `app.config.Settings` into `WorkerConfig`:
`WORKDIR`, `RUNNER` (`stub` until A8 makes Lane 1's stages real), `RECIPE_DIR`,
`IMPL_MODULES`, `LEASE_S`, `POLL_S`, `IDLE_S`, `MAX_ATTEMPTS`, `RETRY_BACKOFF_S`,
`CONCURRENCY` (slots, 1-8, default 1).

## Verify

```bash
cd apps/api && uv run ruff check . && uv run mypy . && uv run pytest -q tests/test_worker.py
```
