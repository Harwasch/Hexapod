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
- each job's lease is renewed by a `claim.LeaseKeeper` thread of its own, for as long as
  its supervisor holds it, and a cancel is seen by that slot's heartbeat and stops that
  job's recipe process only;
- SIGTERM sets one stop flag: every slot stops its own recipe process and clears its own
  lease, so every job is claimable at once;
- `--max-jobs` counts across slots, and a slot reserves its place before claiming.

`tests/test_worker_concurrency.py` runs two 30-second jobs in two slots against Postgres:
both run at once under distinct owners, both leases advance past their length, a cancel
stops one while the other keeps running, and a stop hands the survivor back.

**Why the lease has a thread of its own (the 2026-09-27 dead-letters).** With N = 2 on
Fly, jobs were taken by the other slot every ~40-60 s and dead-lettered with "stage
'normalize' has been attempted 3 times" though nothing had failed. The heartbeat in
`_supervise` was the only thing renewing the lease, and the supervising thread also
downloads the capture before the first stage (`_seed`) and uploads each finished stage's
artifacts and log before recording it (`_apply`; normalize's ~100 frames to R2) -- each
longer than the 30 s lease. The first heartbeat after the download was worse than late: it
ran in the transaction the supervisor's earlier reads had opened, and `now()` is the start
of the transaction, so it wrote a lease that had already run out and reported HELD. With
one slot nothing else was polling and none of it showed. With two, the idle slot (polling
every `idle_s`) claimed the job the moment its lease lapsed, re-ran the stage as the next
attempt -- A6 wiping the `out/` the first slot was still uploading -- and the two traded it
until the attempt budget was spent. Now `claim.LeaseKeeper` renews the lease every
`min(poll_s, lease_s / 3)` from its own thread and session whatever the supervisor is
doing, and every lease is `statement_timestamp() + lease_s`. The tests with slow uploads
and a slow download in `test_worker_concurrency.py` reproduce the failure without either.
A reclaim leaves two lines in the worker's log: "claimed job <id>" a second time for the
same id, and "lost job <id> mid-run: ... claimed_by=<slot>" from the slot that lost it.

**What N the 2 GB machine can take** is a memory question -- the slots' threads only
wait, and the GPU and CPU-heavy stages (`pose`, `train`, `quality`) run on Modal. Measured
2026-09-27 (`/proc` RSS and `getrusage` peaks, on this repository's code):

| process | RSS |
| --- | --- |
| the worker (supervisor) after its imports and a DB session | ~105 MB, plus a few MB a slot |
| a job's recipe process, idle while Modal runs a stage (pipeline, numpy, PIL, modal) | ~80 MB |
| `normalize` of a 25 s 4K HEVC clip (ffmpeg decoding, 100 frames kept) | peak ~380 MB (ffmpeg) + the 80 |
| `place`/`package`/`thumbnail` of a 500k-gaussian SH3 splat (Lane 1 recipe, same code) | peak ~395 MB |
| the same at 1M gaussians (the phone's old "Best" cap) | peak ~750 MB |

Those last two rows were the stages loading the whole splat. Since 2026-09-27 `normalize`
(Lane 1's ingest), `place`, `thumbnail` and `ground_samples` read it a chunk at a time
(`tools/pipeline/splat_io.py`, `splat_stream.py`; README "Stages after training, a chunk
at a time"), in memory that does not grow with it. Measured the same way
(`tools/pipeline/tests/memory_probe.py`, one process per measurement):

| process | whole splat | a chunk at a time |
| --- | --- | --- |
| Lane 1 `normalize` + `thumbnail` + `ground_samples`, 1M-gaussian SH3 upload | 711 MB | 139 MB |
| the same, 2M | 1,386 MB | 144 MB |
| Lane 2 `quality` + `place` (+ `thumbnail` + `ground_samples` chunked), 1M | 328 MB | 184 MB |
| the same, 8M | fails at 1.5 GB | 193 MB |

`package` (`tools/captures/splat_tiles.py`) is made out-of-core the same way by the
large-scene plan's tiler; until that lands it is the one stage here whose peak still
grows with the splat (5M synthetic gaussians: 1.6 GB, per its own docstring).

So a job spends most of its life at ~80 MB and peaks at ~0.45 GB (ffmpeg, at the start)
and, apart from `package`, ~0.2 GB at the end, whatever the capture's size. **N = 2 is
safe on the 2 GB machine** for a capture of any size once `package` is chunked too; N = 3
was not measured. The 20 GB volume holds about two captures' workdirs in flight, so N > 2
wants it extended as well. If the machine does run out, the kernel kills the largest
process -- a recipe process, whose stage then fails and is retried under the attempt
budget -- not the supervisor.

Lane 2's gaussian count is sized to the capture (`cap_max: auto`,
`tools/pipeline/gaussian_budget.py`) and bounded by the L4's memory (~8.7M at 1600 px).
The recipe's `budget_max: 2000000`, set from the old rows of the first table (a 2M splat
peaked near 1.5 GB here), is gone; a deployment whose `package` still loads the whole
splat should put a `budget_max` back in a run's params for its worker.

## Configuration

`WORKER_*` in the environment, read through `app.config.Settings` into `WorkerConfig`:
`WORKDIR`, `RUNNER` (`stub` until A8 makes Lane 1's stages real), `RECIPE_DIR`,
`IMPL_MODULES`, `LEASE_S`, `POLL_S`, `IDLE_S`, `MAX_ATTEMPTS`, `RETRY_BACKOFF_S`,
`CONCURRENCY` (slots, 1-8, default 1).

## Verify

```bash
cd apps/api && uv run ruff check . && uv run mypy . && uv run pytest -q tests/test_worker.py
```
