# app/worker — the thing that turns a queued job into a running pipeline

```bash
cd apps/api
uv run python -m app.worker          # poll until idle for WORKER_IDLE_EXIT_S (0: forever)
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

Four consequences worth stating:

- **cancellation lands mid-stage.** `POST /jobs/{id}/cancel` sets the status; the next
  heartbeat sees it, signals the child to cancel (SIGUSR2) and SIGKILLs it after
  `terminate_grace_s` (15 s).
  Worst case is one poll interval plus the grace — seconds, not the length of the stage —
  and a GPU call the stage had out is cancelled with it (below). A cancel that lands
  while the worker waits `retry_backoff_s` between two attempts is closed out the same way;
- **a stage that segfaults is a failed step, not a lost worker**;
- **the supervisor failing does not leave the child behind.** Every tick writes to the
  database and every finished stage is uploaded, and any of it can raise. The child is
  stopped before the exception goes anywhere -- a cancel when the job is about to be
  dead-lettered, a detach when another worker will resume it -- and nothing tidies a
  workdir while its child is alive. It used to be dead-lettered and tidied under a live
  recipe process, while the slot claimed another job beside it;
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
- **finishes** a stage the recipe process finished while the worker was uploading it
  (`_settle_finished`): its row is still `in-progress`, but its `step.json` proves the
  attempt finished — the same stage and attempt, written after that attempt started (the
  recipe process stamps the start, `started_at`), every artifact's checksum matching
  `out/` — so the upload is redone and the row finished. It used to run again: for
  `train`, hours of GPU billed twice;
- **re-runs** the interrupted stage with `attempt` incremented. A6 clears `out/` at the
  start of every attempt and keeps `checkpoint/`, so a half-written output can never be
  mistaken for a produced artifact and a stage that checkpoints resumes rather than
  restarts;
- **restarts from the beginning** if the workdir is gone — a different machine, a cleaned
  disk. That is the honest answer; a resume whose inputs are missing is not.

A worker asked to stop politely (SIGTERM, SIGINT) does not wait for the stage to finish:
it stops the recipe process on its next tick -- or between two objects of an upload, or
before or after the publish, rather than at fly.toml's 30 s `kill_timeout` -- and
**clears** the lease instead of leaving it to lapse, so the job is claimable at once. That stop is a *detach*, not a failure: the
step is marked `detached` and the next worker runs it as the **same attempt**, so a deploy
spends nothing of the budget.

## A GPU call outlives the process that made it

A stage on Modal is somebody else's machine, and it does not stop because the recipe
process polling it did. Until the 2026-10 audit the call id lived only in that process's
memory and the child had no signal handler, so a cancelled job, a lost lease and a deploy
all left the GPU training for nobody for up to six hours — and the next attempt spawned
a second call writing to the same keys. Now the supervisor's signal says why it is
stopping the child, and `child.Interrupts` turns it into an exception the pipeline's
`CloudRunner` acts on (`tools/pipeline/cloud.py` has the mechanism):

| why the recipe process stops | signal | what happens to the call |
| --- | --- | --- |
| the job was cancelled, or the lease was lost and the job is over | SIGUSR2 | cancelled (Modal: containers terminated); what it billed goes into the ledger |
| the lease was lost to a worker that now holds the job (two slots, one volume) | SIGUSR1 | left running for that worker, which re-attaches to it (or, on another volume, cancels it from the row) |
| this worker is shutting down (a deploy) | SIGUSR1 | left running, recorded as `detached`; the next worker **re-attaches** (`FunctionCall.from_id`) instead of submitting again, at the same attempt |
| the process or the worker is killed outright | — | nothing runs; the record is still there and the retry (a new attempt) adopts the call |
| the workdir is gone (another machine) | — | the call's id is on the step's row (`metrics.remoteCalls`, copied every heartbeat); the next worker writes it back as `orphaned` and the call is cancelled before anything runs |

The recipe process ignores SIGTERM and SIGINT and waits for one of those two: a shutdown
that signals the whole process group (systemd's default, a terminal's Ctrl-C) would
otherwise reach it as a cancel and stop the GPU call on every deploy.

The record is `stages/<id>/calls.json` (`CallBook`): written the moment a call is
submitted — with the stop signal held back until it is — and struck off once the call has
ended, come home and been entered in `attempts.json`, which is idempotent per call id so
a re-attached call is never paid for twice. A call recorded for a stage the run is not
about to resume is cancelled before the run starts, and any call still recorded when a
job is dead-lettered or cancelled is cancelled from the supervisor. Every key a call
writes is per attempt (`checkpoint`, `checkpoint-a2`, …), so a call that escaped all of
that cannot write over the next attempt's checkpoint or outputs. A provider whose calls
cannot be re-attached to (`subprocess`, a child of the process going away) is cancelled
on a detach as well.

**A job cancelled while no worker holds it** is never claimed again, and until the
2026-10 review nothing else read its calls: a worker that crashed (thirty seconds of
lease and a restart), one whose restarts ran out, and one stopped by `fly machine stop`
(it detaches the call for a successor, exits 0 and stays stopped) all left the GPU
training for nobody for up to six hours. Now the worker **reaps** (`reaper.py`) when it
starts and every `reap_every_s` (300 s) after: every call still recorded for a job that
is over -- in this volume's books, and on the step rows (`metrics.remoteCalls`) -- is
cancelled by id through its provider's adapter (`CloudRunner.cancel_recorded`), and
struck off as `metrics.reapedCalls`, so a second pass finds nothing. A job a worker may
still be closing (its lease run out for less than a lease) is left to it, a cancel that
fails is tried again on the next pass, and a record older than a day is struck off as
expired. The API **wakes the worker on a cancel** of a job a worker had claimed
(`worker_wake.schedule_reap`, after the commit and off the request path, as for an
enqueue but without the queue check's `/start`), so the start-up pass is what answers it.
A worker that cancels its own job's call copies the struck-off book onto the row at once,
so the reaper never cancels anything twice.

## Giving up: the dead-letter path

`job_steps.attempt` is one budget covering both ways a stage fails to finish — it raised,
or its worker died. When the next attempt would exceed `worker_max_attempts` (3), the job
goes to `error` with a message naming the stage, the count and the last error, and nothing
claims it again. Without that cap a capture that kills its worker would be picked up
forever by whoever is next.

A recipe that will not resolve at all dead-letters on the first look, because it will not
resolve on the second either.

**Not every failure gets the whole budget** (`retry.py`). Under a six-hour Modal limit,
three attempts at a stage that timed out were eighteen GPU-hours. The worker reads each
failure first — its error, and the end of the failed attempt's own lines of the stage log
(the recipe process reports where they begin; an out-of-memory the attempt logged and got
past, such as a fan-out piece resubmitted alone, is not its verdict) — and the classes are
kept narrow, because a retry spent on a failure put in the wrong one is the cheaper
mistake:

| failure | what the worker does |
| --- | --- |
| CUDA out of memory (`CUDA out of memory`, `OutOfMemoryError` in the error or the attempt's last 50 log lines) | **one** retry, with `cap_max` at 0.7x the cap the attempt's `gsplat: cap_max …` line names, written into `jobs.params[stage]`; a second, or one with no cap to lower, is not retried |
| out of time (`RemoteTimeoutError`: Modal's `FunctionTimeoutError`, `max_wait_s`, the overdue-trainer deadline) | not retried |
| a broken recipe or stage contract (`errors.BAD_INPUT_ERRORS`) | not retried |
| the run reached its dollar cap (`CostCapError`) | not retried |
| anything else | `worker_max_attempts`, as before |

Each stage's verdicts are in `stages/<id>/failures.json`, which is how "one" retry stays
one across a worker restart.

**A job has a dollar ceiling**, `WORKER_JOB_COST_CAP_USD` (20; 0 turns it off), over the
attempt ledgers' `costUsd`: `CloudRunner` submits no call once the run is at it and
cancels a running call — or a fan-out's pieces together — whose running cost would take
the run over it, and the worker starts no attempt past it. Priced calls only: a tier with
no rate is not stopped by a figure nobody has.

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
| remote calls in flight | `stages/<id>/calls.json`, and a copy on the step: `metrics.remoteCalls` |
| what each failed attempt failed of | `stages/<id>/failures.json` |
| the recipe process's stderr | `recipe-process.stderr.log` in the workdir; its tail in `jobs.error` on a failure |
| the site | `sites` + the capture's `site_id`, from `registration.json` |
| what a browser fetches | the public bucket, `runs/<job id>/p<generation>/<stage id>/<name>`, one generation per publish |

A deployment with no bucket still runs: the log and artifact uploads are skipped and say
so by leaving `log_key` null, rather than failing the job.

A directory artifact's members go up eight at a time (`app/storage/parallel.py`; normalize's frames and
package's tiles are hundreds of small objects whose cost is round trips), and every
artifact object is written with the `Cache-Control` a browser should get for it
(`outputs.cache_control_for`, the same rule as the tile proxy in `functions/r2/[[path]].js`).
A stage that ran on a provider is **copied** into its artifact keys inside the bucket from
what the provider left under `runs/<job>/<stage>/transfer/out` (`out-a<N>` for a call
attempt N submitted), rather than uploaded again from the workdir — but only when its
`step.json` says `runner: cloud`, names the key its outputs came home from
(`metrics.outputsKey`, the call's own: a call re-attached to wrote under the attempt that
submitted it), and the objects there match the workdir's files name for name and size for
size; anything else is uploaded as before. The key is never guessed from the attempt
number: after a Retry resets the attempts, `out-a2` can hold an earlier run's outputs.
The download into the workdir stays: `place`, `package` and the rest read it there.

Publishing (`publish.py`) copies a tileset to the public bucket eight at a time and its
`tileset.json` last, after every tile, so a public root always means its tiles are there; a
copy that fails leaves no root and registers no site. Every publish copies into a
generation of its own, `runs/<job>/p<generation>/...` (`app/services/published.py`; the
generation is a hash of what is published), because a Refine re-runs the same job and
rewrites its own keys: the site moves to the new generation only once it is complete, the
live one is never written again -- so `immutable` holds there and only there -- and a
republish that fails leaves the site, its thumbnail and its overlay as they were. A run's
own keys are uploaded with the short lifetime for the same reason.

Every run that ends — finished, failed or cancelled — drops its `inputs/` (a copy of what
is in the bucket, which a retry fetches again) and every stage's `work/` (scratch);
`out/`, `step.json` and `checkpoint/` stay. It used to be finished runs only, and a failed
run kept a 12 GB video on a 20 GB volume until somebody deleted it.

## Watching the runs that are going on

`WORKER_HEARTBEAT_URL` is a dead-man's switch for **active runs only**
(`alerts.py`, healthchecks.io-style): `<url>/start` when a job is claimed and again every
minute while it runs (from a thread of its own, beside the lease keeper), `<url>` when it
finishes or is cancelled, `<url>/fail` with the reason when it is dead-lettered — and
nothing while the worker is idle, or when it lets a job go for a deploy (the next
worker's `/start`, or the check's grace, says what happened). The keep-alive is a
`/start` because each one restarts the check's grace for the run, so the check is set
with a **long period (30 days) and a grace of a few minutes (5)**: a worker killed mid-run
goes quiet and alerts a grace later, and an idle one never does. It used to be a success
ping, which fed only the period — short, it alerted on every idle afternoon; long, it
missed a run that had died. `QUEUE_CHECK_URL` is pinged when a job is claimed; the API
starts it when one is queued and no job is running (one queued behind a two-hour run
would alert after the grace), so a job nobody claims alerts too. A ping is a thread with a
5 s timeout whose every error is logged and dropped: it never holds up the worker and
never fails a job.

## Room on the volume

Before it claims, the worker looks at the free space on `WORKER_WORKDIR`'s volume
(`disk.py`). Below `WORKER_MIN_FREE_GB` (5) it tidies every run that has ended, then
evicts whole workdirs of runs that ended more than `WORKER_EVICT_AFTER_DAYS` (7) ago,
oldest first, until there is room; a retry of one of those starts over from the upload.
If there is still no room it does not claim, and logs `NOT CLAIMING` at error level once
a minute: the job stays queued rather than failing on a full disk after its download.
A run that is not over is never touched. The one job it still claims is a run a deploy
**detached** whose workdir is on this volume (`DiskGuard.resumable_here`): resuming it
downloads nothing, and its GPU call is running, waiting to be re-attached to.

The supervisor's session also commits before it downloads the capture (`_seed`), as it
already did before uploads: a session idle in a transaction for the length of a 12 GB
download is one Neon terminates, and the next statement on it fails.

## Several jobs at once

`WORKER_CONCURRENCY=N` (default 1) runs N **slots** in one worker process (`loop.py`). A
slot is what the whole worker used to be -- claim one job, supervise it to the end in its
own recipe process, claim the next -- and nothing about a job is shared between slots:

- each slot claims with its own id, `host:pid/<n>` (one slot keeps the plain `host:pid`),
  so `claimed_by`, the heartbeat and `_still_ours` behave exactly as between two separate
  workers: a lease one slot let lapse and another reclaimed is *lost* to the first, which
  stops its recipe process with a **detach**, not a cancel -- the GPU call is the other
  slot's now, adopted from the `calls.json` the two share;
- each job's lease is renewed by a `claim.LeaseKeeper` thread of its own, for as long as
  its supervisor holds it, and a cancel is seen by that slot's heartbeat and stops that
  job's recipe process only;
- SIGTERM sets one stop flag: every slot stops its own recipe process and clears its own
  lease, so every job is claimable at once;
- `--max-jobs` counts across slots, and a slot reserves its place before claiming.

`tests/test_worker_concurrency.py` runs two 30-second jobs in two slots against Postgres:
both run at once under distinct owners, both leases advance past their length, a cancel
stops one while the other keeps running, and a stop hands the survivor back.

**CPU-only slots.** `WORKER_CPU_ONLY_SLOTS=N` adds N slots *beside* the general ones that
claim only a recipe with no `gpu:` stage — `claim_next(recipes=...)`, a filter inside the
same `SKIP LOCKED` select. Which recipes qualify is read from the recipes when the worker
starts (`loop.cpu_only_recipes`, through the same lookup a job's recipe goes through, so a
deployment's own `WORKER_RECIPE_DIR` copy is the one judged); today that is `splat-ingest`.
Production runs one general slot and one of these: a one-minute ingest no longer waits
behind a two-hour training run, and two training runs — two GPUs billed, two videos on a
20 GB volume — never run at once, which is why plain `WORKER_CONCURRENCY=2` was not the
answer. Slot ids run on across both kinds (`host:pid/0` general, `host:pid/1` CPU-only).
The test file runs a stand-in training run in the general slot and an ingest in the
CPU-only one, and checks the CPU-only slot never takes the second training run.

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

`package` (`tools/captures/splat_tiles.py`) has since been made out-of-core the same way:
it holds 12-16 bytes a gaussian for the whole scan, and its docstring gives 8M gaussians
(a 2 GB PLY) packaging at a 380 MB peak (`tests/test_splat_tiles_memory.py`).

So a job spends most of its life at ~80 MB and peaks at ~0.45 GB (ffmpeg, at the start)
and ~0.2 GB at the end (~0.4 GB while `package` runs on an 8M splat), whatever the
capture's size. **N = 2 is safe on the 2 GB machine** -- which is what production's one
general slot plus one CPU-only slot is: the worst overlap, Lane 2's ffmpeg beside a Lane 1
`package`, is ~0.45 + ~0.4 GB with the ~0.1 GB supervisor. N = 3 was not measured. The 20 GB volume holds about two captures' workdirs in flight, so N > 2
wants it extended as well. If the machine does run out, the kernel kills the largest
process -- a recipe process, whose stage then fails and is retried under the attempt
budget -- not the supervisor.

Lane 2's gaussian count is sized to the capture (`cap_max: auto`,
`tools/pipeline/gaussian_budget.py`) and bounded by the L4's memory (~8.7M at 1600 px).
The recipe's `budget_max: 2000000`, set from the old rows of the first table (a 2M splat
peaked near 1.5 GB here), is gone; a deployment whose `package` still loads the whole
splat should put a `budget_max` back in a run's params for its worker.

## Idle: backing off, exiting, and being woken

An empty queue is polled every `WORKER_IDLE_S` (2 s) for `WORKER_IDLE_BACKOFF_AFTER_S`
(60 s), then doubling a period at a time up to `WORKER_IDLE_MAX_S` (30 s)
(`loop.poll_delay`). Once no slot has run or claimed anything for `WORKER_IDLE_EXIT_S`
(900 s by default; 0 polls forever, and `.env.example` sets 0 for a checkout) the worker
exits with status 0 and its machine stops. Neon, with nobody polling, scales to zero.

The exit is only between jobs. A slot is busy from the start of its claim to the end of its
job (`loop._Idle`), the idle clock restarts when a job ends, and the decision to exit takes
the same lock every claim takes and asks the queue once more under it
(`claim.anything_claimable`) -- so a job committed while the worker was deciding keeps it
up, and no slot starts a claim it would then abandon. SIGTERM is unchanged: a stop wakes a
slot out of even the longest backoff at once.

What starts it again is the API: every route that queues a job asks Fly's Machines API to
start the stopped `worker` machines once the job is committed, and looks again 20 s later
if it found one still up -- the worker may have been exiting at that moment
(`app/services/worker_wake.py`, docs/DEPLOYMENT.md § Waking the worker).
`tests/test_worker_idle.py` runs the backoff, the exit, the race and the exit status.

## Configuration

`WORKER_*` in the environment, read through `app.config.Settings` into `WorkerConfig`:
`WORKDIR`, `RUNNER` (`stub` until A8 makes Lane 1's stages real), `RECIPE_DIR`,
`IMPL_MODULES`, `LEASE_S`, `POLL_S`, `IDLE_S`, `IDLE_BACKOFF_AFTER_S`, `IDLE_MAX_S`,
`IDLE_EXIT_S`, `MAX_ATTEMPTS`, `RETRY_BACKOFF_S`, `CONCURRENCY` (general slots, 1-8,
default 1), `CPU_ONLY_SLOTS` (0-4, default 0), `JOB_COST_CAP_USD` (20; 0 off),
`DEADLINE_FACTOR` (2; 0 off), `HEARTBEAT_URL` (unset: no pings), `MIN_FREE_GB` (5; 0
off), `EVICT_AFTER_DAYS` (7). And `QUEUE_CHECK_URL`, unprefixed because the API reads it
too.

## Verify

```bash
cd apps/api && uv run ruff check . && uv run mypy . && uv run pytest -q tests/test_worker.py \
  tests/test_worker_stops.py
```
