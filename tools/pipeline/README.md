# tools/pipeline — the capture pipeline's spine

Turns an uploaded capture into the artifact set the console needs, by running an ordered
list of stages. This project is the executor, the stage registry, the artifact and workdir
contract, and the runner seam. It is **not** the worker — that is `apps/api/app/worker/`,
which imports this project as a library.

**Lane 1 is real.** `splat-ingest` runs end to end on a CPU: a `.ply` or `.spz` in,
`canonical.ply`, a `splat/` tileset, a thumbnail, ground samples, a manifest and a
registration out. Since 2026-09-23 it **converts the file's up axis** rather than assuming
z -- see [The up axis](#the-up-axis) -- which is what stopped uploads landing on their side.

**Lane 2 is real up to the GPU, and the GPU half is built and checked but has never run.**
The state of each piece, in the three-state vocabulary of `docs/HANDOFF.md`:

| Piece                                         | State        | Evidence                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| frames from an iPhone-shaped HEVC `.mov`      | verified     | portrait (display matrix -90), HEVC, `mdta` location: 100/100 frames upright, location read (`tests/test_normalize.py` and a real-frame check)                                                                                                                                                               |
| poses (COLMAP 3.9.1, CPU)                     | verified     | rendered orbit 40/40 (CI, on a GitHub runner too); real photographs, 50/50 exhaustive; timing in [Where pose runs](#where-pose-runs). Matching is not deterministic, so a low registration is retried: other mapper seeds, then one fresh matching pass, keeping the best (`mapperAttempts` in `poses.json`) |
| levelling by camera-up                        | verified     | real reconstruction: dominant plane 0.39 deg from up after levelling; rendered orbit within 5 deg (CI)                                                                                                                                                                                                       |
| the EXIF similarity, applied (`place`)        | verified     | rendered orbit with synthetic GPS: sparse points 0.011 m from the scene placed, 0.41 m unplaced (CI)                                                                                                                                                                                                         |
| a video with no location                      | verified     | falls back to the capture's `lat`/`lon`, recorded `manual`; with neither it refuses by name (CI)                                                                                                                                                                                                             |
| `train` argv and output layout (gsplat 1.5.3) | verified     | parsed by v1.5.3's own `simple_trainer.py` CLI in a CPU replica of the image's venv (CI job `trainer`), then run for real on an L4 (below)                                                                                                                                                                   |
| the Modal training image                      | verified     | built by Modal on 2026-09-23 (about 8 minutes, once); CUDA 12.4.1, torch 2.4.1, gsplat 1.5.3                                                                                                                                                                                                                 |
| the GPU path end to end                       | verified     | `modal.yml` smoke, 2026-09-23: 500 steps on an L4, 68 s billed, $0.015, PSNR/SSIM read from gsplat's stats file, `trained.ply` placed and packaged to a tileset                                                                                                                                              |
| `ModalAdapter` against a live workspace       | verified     | the same smoke: submit, poll to success, log tail, outputs back through R2. Its first two real runs found two bugs, both fixed and tested (below)                                                                                                                                                            |
| a full-length training run on a real capture  | **unproven** | the smoke trained 500 steps; 30,000 steps on a real video has not run. Its cost is unmeasured -- scale from the smoke's billed seconds                                                                                                                                                                       |

- `normalize` / `ffmpeg_frames` extracts and selects frames and scrapes the container's
  metadata. It runs here and in CI, on a generated clip.
- `pose` / `colmap` runs real structure from motion and is scored against known poses —
  40 frames rendered from the committed synthetic tree, 40/40 registered, 0.059° median
  rotation error, 0.092% of scene extent in translation. CI installs `colmap` so this
  runs there too, and the test file fails rather than skipping if that install goes away.
- `train` / `gsplat` **dispatches** a training run: the dataset, the argv, the metrics,
  the PLY. **No training run has been executed in this repository** — `gsplat` needs CUDA
  and there is no GPU here — so its tests drive a stand-in trainer and say so in their
  names. Since 2026-09-23 the argv and file names are checked against gsplat v1.5.3's own
  `simple_trainer.py`, which corrected four transcription errors, each of which would have
  cost a GPU run: `--ckpt` means _evaluate_, not resume (so a preempted attempt now
  restarts, and says so); there is no PLY without `--save_ply`; world normalisation is on
  by default and would have exported the splat in a rotated, rescaled frame; and the
  stats and PLY names are zero-based (`val_step29999.json`, `point_cloud_29999.ply`).

- `georeference` / `exif_gps` reads each frame's own EXIF GPS, defines an east/north/up
  frame about the median fix, and has `colmap model_aligner` solve for the similarity
  that takes the reconstruction into it — which is where a **metric scale** comes from
  when a capture has no ARKit. Scored against GPS written onto the same rendered orbit at
  the cameras' true positions: 40/40 fixes aligned, 0.0088 m median residual, scale
  within 1.3e-5 of `umeyama` on the same points.

`glomap`, `arkit`, `opensplat` and `robust` are still stubs with honest contracts: they
declare exactly what they read and write, and raise with the step that lands them rather
than pretending. Each one's stub says in a comment why it is still one — `arkit` is the
one B4 looked at and left, because there is no ARKit capture here and the on-disk format
is the capture app's rather than Apple's.

It has no dependency on `apps/api`: no models, no database connection, no HTTP. The
dependency runs one way only. A stage that wants something registered writes a file saying
so (`registration.json`) and the worker, which has the credentials and the session, does
it.

```bash
uv run python run_recipe.py splat-ingest --workdir /tmp/run-1 --runner local \
  --seed upload=./capture.ply
uv run python run_recipe.py photo-reconstruct --workdir /tmp/run-2 --plan-only
```

The recipe's placement parameters are the deployment's defaults; a run overrides them per
capture (see [Parameters per run](#parameters-per-run)), which is how the worker hands a
stage the coordinate the operator dropped the capture at.

## A recipe is an ordered list of stages, not a DAG

```yaml
name: photo-reconstruct
version: 1
inputs: [upload]
stages:
  - { id: normalize, impl: ffmpeg_frames, params: { fps: 4, keep: 400 } }
  - { id: pose, impl: colmap } # | glomap | arkit
  - { id: mask, impl: none } # | robust | sls
  - { id: train, impl: gsplat, gpu: { tier: l4, preemptible: true } }
```

- `inputs` are the artifacts the caller seeds into `inputs/` before the run. Everything
  else must be produced by an earlier stage.
- `impl` names a registered Python callable. **Swapping `impl` is the entire extensibility
  story.**
- `params` are passed to the implementation untouched, and recorded in `recipe.json`.
- `gpu:` is the only routing signal there is. A stage declares it or it does not, and that
  single fact chooses the runner. There is no second code path.

A DAG with `needs` edges was considered and rejected: the real graph is almost linear, and
the one genuine fan-out (mesh, point cloud and splat from the same poses) is three
consecutive stages.

**A stage does not declare its artifacts in the recipe — the implementation does**, next to
the code that reads and writes them. A recipe that could restate the contract is a recipe
that can lie about it.

## The artifact contract

An artifact's **name is its path**. `splat` produced by the stage `package` lives at
`stages/package/out/splat`; `georef.json` produced by `georeference` lives at
`stages/georeference/out/georef.json`. A later stage never computes that path: it asks for
the artifact by name and the executor hands it the resolved location.

```python
@stage_impl(
    "gsplat",
    consumes=("frames", "poses"),
    optional_consumes=("masks",),  # present only if some earlier stage made them
    produces=(CANONICAL_PLY, TRAIN_METRICS),
)
def gsplat(ctx: StageContext) -> StageOutcome:
    frames = ctx.input("frames")  # refuses anything not in `consumes`
    out = ctx.output("canonical.ply")  # refuses anything not in `produces`
    train(frames, out)
    return StageOutcome(metrics={"psnr": 28.4})
```

`optional_consumes` is how `mask: none` and `mask: robust` are both legal without the
trainer or the executor knowing which one ran.

Both lanes converge on `canonical.ply`, always east/north/up about the placed origin —
Lane 1 normalises an already-reconstructed splat into it, Lane 2's `place` stage turns the
trainer's `trained.ply` (COLMAP's frame) into it — so one `package` implementation serves
both.

### What fails, and when

Everything that can be caught before a stage runs is caught before **anything** runs —
no directory is created, no implementation is invoked:

| Refusal                                                                             | Raised by                 |
| ----------------------------------------------------------------------------------- | ------------------------- |
| `impl` is not registered (message names the recipe, the stage and every known impl) | `UnknownImplError`        |
| a stage is in `skip` but has no previous `step.json` in the workdir                 | `ResumeError`             |
| a stage consumes an artifact no earlier stage produces                              | `UnresolvedArtifactError` |
| two stages produce the same artifact name                                           | `DuplicateArtifactError`  |
| a declared recipe input was never seeded into the workdir                           | `MissingInputError`       |
| a `gpu:` stage and no GPU runner configured                                         | `NoRunnerError`           |

And at the end of each stage, in `BaseRunner` — so every runner enforces them identically:

| Refusal                                                                                | Raised by                 |
| -------------------------------------------------------------------------------------- | ------------------------- |
| declared a `produces` it did not write (or a `required_members` file it did not write) | `MissingArtifactError`    |
| wrote something into `out/` it never declared                                          | `UndeclaredArtifactError` |
| asked for an input or output it never declared                                         | `StageContractError`      |
| the implementation raised (chained, with the stage named)                              | `StageFailedError`        |

## The workdir

```
<workdir>/
  recipe.json              the recipe exactly as executed, plus where each artifact comes from
  artifacts.json           every artifact produced: path, kind, size, sha256
  inputs/<name>            what the caller seeded (for both shipped recipes: upload/)
  stages/<stage id>/
    out/<artifact name>    everything the stage declared it produces, and nothing else
    work/                  scratch; never an artifact, safe to delete at any time
    checkpoint/            survives a killed attempt — the resume seam
    log.txt                the stage's log
    step.json              the StepResult for the stage's last attempt
```

`workdir.py` is the only module that builds any of these paths, so B1 can relocate a
workdir onto a GPU container by changing `root` and nothing else.

`artifacts.json` deliberately carries no timings: two runs of the same recipe over the same
inputs produce a byte-identical manifest. Wall time lives in `step.json`.

### StepResult

One per stage run — the unit A7 writes to `job_step` and A10 renders:

`stageId`, `impl`, `runner`, `attempt`, `durationS`, `artifacts` (name, path, kind,
contentType, bytes, sha256 checksum), `metrics`, `logPath`, `checkpointKey`, `gpuTier`,
`summary`.

### Watching a run, and resuming one

`execute()` takes three optional arguments, all added by A7 and none of which let the API
into this project:

```python
execute(recipe, workdir, runners, observer=worker, skip={"pose"}, attempts={"train": 2})
```

- **`observer`** is told `stage_started` / `stage_finished` / `stage_skipped` /
  `stage_failed` as they happen. A7 writes a `job_step` row from each one, which is what
  makes the panel's stage list live rather than a batch that appears at the end. An
  observer that raises stops the run where it stands — that is how a cancelled job is
  noticed between stages.
- **`skip`** names stages whose previous `step.json` in this workdir stands. They are not
  re-run and their results are read back, so a resumed run's `artifacts.json` says exactly
  what a fresh run's would. Skipping a stage with no previous result raises `ResumeError`
  during planning, before anything runs — a cleaned-up workdir gets a fresh run, never a
  half one.
- **`attempts`** maps a stage id to the attempt number to record, which is A2's
  `job_steps.attempt`.

### Resuming a killed stage

A GPU stage runs on a preemptible tier, where being killed is ordinary operation rather
than an error:

- `out/` is **cleared** at the start of every attempt, so a half-written output from a
  killed attempt can never be mistaken for a produced artifact;
- `checkpoint/` is **kept** across attempts. It is the one directory the executor does not
  clear. A stage sees `ctx.has_checkpoint` and `ctx.checkpoint_dir`;
- `ctx.checkpoint_key` is the object-storage key `CloudRunner` syncs that directory to
  (`runs/<run id>/<stage id>/checkpoint`), and it lands in the StepResult when the stage
  left anything behind, so `job_step.checkpoint_key` has something to record.

The sync is on an **interval** while the stage runs, not at the end: a checkpoint that
only appears when the stage finishes is worth nothing to an attempt that never does.
What survives a preemption is whatever the last sync captured, so `checkpoint_every_s`
is the knob that decides how much work is thrown away.

`tests/test_cloud_preemption.py` is the proof, and it asserts on _work not repeated_: a
ten-iteration stage is cut off at the same point on both attempts, so it only ever
reaches ten by the second attempt continuing the first. Remove the checkpoint restore
and the same test never finishes — which is a test in the file, not a claim.

## Runners

```
Runner.run(stage, workdir) -> StepResult
├── LocalRunner   calls the registered implementation in this process; a stage that needs
│                 an external tool shells out through StageContext.run()
├── StubRunner    CI: fabricates each declared artifact deterministically
└── CloudRunner   cloud.py: submits the stage to a provider, tails its log, brings its
                  outputs and its checkpoint back
```

`RunnerSet(cpu=..., gpu=...)` routes on one fact: whether the stage declares `gpu:`.
`RunnerSet.stubbed()` puts StubRunner in both slots; `RunnerSet.local()` leaves `gpu` empty,
so a GPU stage fails with a message that names the tier it wanted; `RunnerSet.cloud(...)`
runs CPU stages here and sends `gpu:` stages away.

### Running a stage somewhere else

`CloudRunner` needs two things, both protocols defined in `cloud.py` and both injected,
because this project may not grow a `boto3` and may not import `apps/api`:

- **`ProviderAdapter`** — `submit`, `poll`, `logs`, `cancel`, `rate`. `poll` returns
  `preempted` as a state of its own, separate from `failed`: one resumes and the other
  does not, and that is the distinction the whole path exists for.
- **`Transfer`** — `put`, `get`, `exists`, `delete` over opaque keys. The worker supplies
  an S3 implementation over its `ObjectStorage`; `LocalTransfer` here does the same over
  a directory. The pipeline says what to move and under which key and knows nothing about
  buckets — the same split `app/worker/registration.py` set out.

Three adapters ship, and they are not equally real:

| adapter             | where it runs        | verified                                 |
| ------------------- | -------------------- | ---------------------------------------- |
| `FakeAdapter`       | in process, no clock | yes — it drives every cloud test         |
| `SubprocessAdapter` | a local process      | yes — including preemption, by SIGTERM   |
| `ModalAdapter`      | Modal                | **no. Not one line of it has ever run.** |

`ModalAdapter` says so itself, at length, in its own module docstring. Its Modal calls
were first written from memory rather than from documentation, because Modal was
unreachable from the environment that wrote it. They have since been **read against
`modal==1.5.5`** — the installed SDK's source and docstrings, plus Modal's docs — which
found five defects, one of them fatal: `poll` caught the builtin `TimeoutError`, while a
zero-timeout poll raises `modal.exception.TimeoutError`, which does not inherit from it,
so every healthy stage was dead-lettered on its first poll. `tests/test_modal_adapter.py`
now pins the classification against fakes that mirror the real exception hierarchy.

That check moved the adapter from _guessed_ to _read_; the first real runs moved it to
_verified_, and found two more things no reading could have:

- **"Not finished yet" is the builtin `TimeoutError`.** `FunctionCall.get(timeout=0)` in
  `modal==1.5.5` raises a bare builtin `TimeoutError()` (`poll_function`), not
  `modal.exception.TimeoutError`. The adapter called that a failure, so the first smoke
  was dead-lettered on its first poll. The builtin now means "keep polling", and
  `remote.execute` converts a stage's own `TimeoutError` so the two cannot be confused.
- **A container imports `app.py` itself**, as `/root/app.py` with no repository around
  it; computing the repo root there crash-looped every container on `IndexError: 2`. CI
  now imports the file the way the container entrypoint does.

And one guard that came out of the second: Modal answers "no output yet" identically for
a training stage and for a container that crash-loops before the function starts. The
adapter reports `pending` until `run_stage`'s own first line appears in the call's log,
and `CloudRunner` cancels a stage still pending after `max_pending_s` (30 minutes) rather
than holding the worker for `max_wait_s` (a day).

### The remote half

A provider's container fetches its own bytes, which is the one thing `SubprocessAdapter`
cannot show you — a local process is filled from the outside. `remote.py` is that sequence
performed from the inside: fetch the inputs and any checkpoint a previous attempt left, run
the stage while a syncer copies `checkpoint/` out on an interval, upload `out/` on success,
and hand back what the stage reported.

It takes a `Transfer` and a directory and imports no provider SDK, which is the whole point
— `tests/test_remote.py` drives every one of those steps here, against `LocalTransfer`. The
Modal-shaped part is as small as it can be: `infra/modal/app.py` is an image, a GPU, a
secret and one call, deploying a `run_stage_<tier>` per tier because Modal fixes a
function's GPU at decoration time.

```bash
uv run --project tools/pipeline --with modal modal deploy infra/modal/app.py
```

CI builds that App on every run, which is a real check of everything the decorators take —
it caught two path bugs the day it was written. It is not a check of the `gpu=` string: an
App with a nonsense GPU name builds perfectly well and is refused only server-side. And a
**training** stage will not run on it until `TRAINING_PACKAGES` is filled with a CUDA,
torch and gsplat triple somebody has actually built; it is empty rather than guessed.

`Placement` decides which adapter an attempt goes to, from the preemptions recorded in
`stages/<id>/attempts.json`. After `preemptions_before_fallback` of them the stage moves
to the next provider, and the last one may not itself be interruptible — otherwise a
two-hour stage on a host that preempts hourly is retried until the budget is gone, having
paid for the same two hours three times.

### What a run cost

`attempts.json` holds one entry per attempt — provider, tier, state, billed seconds, and
the price if there is one — including the attempts that were preempted, because those
were paid for too. `run_cost(workdir)` totals it, and the worker writes that onto
`jobs.cost_usd`, `jobs.provider` and `jobs.tier`.

`providers.py` is the price table, and it carries **only the four A100 rates A0 actually
measured**. A tier with no rate records its billed seconds and no cost; it does not get
an invented number, because a plausible price in a cost column is a price that will be
believed. A deployment supplies its own through `PIPELINE_GPU_RATES`.

Everything that is not "run the implementation" — clearing the previous attempt, keeping the
checkpoint, checking the declared `produces` exist, hashing them, writing the StepResult —
happens once, in `BaseRunner`. A runner overrides `_invoke` and nothing else, so a stub
cannot drift from the real thing by forgetting a rule.

**StubRunner is not a testing nicety.** It is what lets the whole pipeline, Lane 2 included,
run end to end in CI on a machine with no GPU and no network. It knows nothing about splats:
it writes exactly the artifacts the implementation declares, in the shape it declares them,
with contents derived by hashing (recipe, stage, impl, params, artifact, and the checksums
of that stage's inputs). So its output is byte-identical across runs and across machines, a
changed parameter changes the bytes, and an implementation that gains an output gains a stub
for it with no edit to the runner.

## Adding an implementation

Two edits, neither of them in the executor:

```python
@stage_impl("glomap", consumes=("frames",), produces=(POSES,))
def glomap(ctx: StageContext) -> StageOutcome:
    ctx.run(["glomap", "mapper", ...])
```

...and `impl: glomap` in the recipe. `tests/test_registry.py` is the evidence: it registers
a brand-new stage and artifact in a test file and runs it through both runners.

## The `tools/captures` import

`tools/captures/splat_tiles.convert()` already packs SPZ into `KHR_gaussian_splatting`
3D Tiles byte-stably, and this sprint automates around that code rather than replacing it.
The two directories are separate uv projects and the sibling declares `package = false` — it
is a set of scripts, not a distribution — so it cannot be a path dependency and there is
nothing to install.

`captures_bridge.py` is therefore a `sys.path` insertion, in one module, computed from that
file's own location — at import rather than lazily, because `SplatFormatError` is caught by
name and an exception class cannot be imported inside a function. It re-exports `read_ply`,
`unpack_spz`, `pack_spz` and `convert`, which is the whole of what Lane 1 needs from the
sibling project. Two things keep it honest rather than hidden:

- `mypy_path = ["../captures"]`, so mypy resolves `splat_tiles` to the real file and
  type-checks every call into it (with `follow_imports = "silent"`, since that project does
  not run mypy and its errors are not this project's to fix);
- `tests/test_captures_bridge.py` runs the real `convert()` on a generated PLY and asserts
  the files it writes are exactly the `required_members` the `splat` artifact declares — the
  same declaration StubRunner fabricates from.

**What A8 changed in `tools/captures` and what it did not.** `splat_tiles.read_ply` was
rewritten and `unpack_spz` added; `convert`, `pack_spz` and `build_glb` were not touched at
all. That distinction is the fixture byte-identity gate: CI runs
`git diff --exit-code -- data/tiles/synthetic-tree`, so what that file _writes_ is frozen
and what it _reads_ is not. And no SPZ round trip may sit inside that gate — A0 #4 measured
`unpack → pack` differing by one byte in 228,016 at a rotation rounding boundary, so
`tests/test_spz_ingest.py` generates its `.spz` from the committed PLY rather than
committing one.

`apps/api` carries `numpy` and `pillow` for the same reason it carries the pipeline on its
path: the worker runs these stages in that environment. Nothing under `app/api` or
`app/services` imports either.

## Recipes shipped

- **`splat-ingest`** — Lane 1, no GPU: `normalize → georeference → package → thumbnail →
ground_samples → manifest → register`. Every stage is real.
- **`photo-reconstruct`** — Lane 2: `normalize → pose → mask → train → compensate →
georeference → place → package → thumbnail → ground_samples → manifest → register`. Only `train`
  declares `gpu:`; `compensate` gains one when its impl becomes `imc` (B3), since asking
  for an L4 to run `none` would be billing a GPU to do nothing. Both lanes end in the same
  artifact set, so the console cannot tell which one made a site except by reading its
  manifest.

`thumbnail`, `ground_samples` and `manifest` were added in A8 as **a recipe edit and three
decorators**: `stages.py` gained three `@stage_impl`s and three `ArtifactDecl`s, and the
recipes gained three entries. `executor.py`, `runners.py`, `plan.py` and `workdir.py` are
untouched by them, and `StubRunner` fabricates the new artifacts with no edit of its own.
`tests/test_lane1.py` asserts that rather than leaving it as a claim.

## Where pose runs

`pose` runs on the **worker's CPU**, not the GPU box. The GPU is billed by the second and
only `train` needs one. COLMAP's CPU path is the one every finding in `sfm.py` was measured
on. And shipping frames to Modal for SfM and back would add a round trip the stage does
not otherwise need. What it costs, measured on 4 cores of this development container
(COLMAP 3.9.1, the Ubuntu 24.04 package; real iPhone-portrait frames, 1080×1920, orbiting
one object), with the machine partly contended, so these numbers are upper bounds:

| Matcher                        | Frames | Features / max side | Extract | Match | Map   | Total     | Registered |
| ------------------------------ | ------ | ------------------- | ------- | ----- | ----- | --------- | ---------- |
| exhaustive                     | 50     | 8192 / 2400         | 80 s    | 927 s | 54 s  | 1061 s    | 50/50      |
| **exhaustive**                 | **50** | **4096 / 1600**     | 127 s   | 502 s | 24 s  | **653 s** | **50/50**  |
| sequential                     | 100    | 8192 / 2400         | 292 s   | 635 s | 146 s | 1073 s    | 60/100     |
| sequential                     | 100    | 4096 / 1600         | 169 s   | 382 s | 95 s  | 646 s     | 46/100     |
| sequential + loop (vocab tree) | 100    | 4096 / 1600         | 182 s   | 535 s | 128 s | 845 s     | 76/100     |
| sequential + loop, overlap 5   | 100    | 4096 / 1600         | 154 s   | 319 s | 144 s | 616 s     | 51/100     |

What this decided, in `recipes/photo-reconstruct.yaml`:

- **`exhaustive`, still.** It is the only matcher that registered every frame. Sequential
  matching is linear rather than quadratic, but it lost a quarter to a half of the orbit
  even with vocabulary-tree loop closure (Flickr100K 32K words, sha256 `d37d8f19…`). So
  `sfm.matcher_argv` can express loop closure, but no recipe asks for it.
- **`keep: 100`, down from 400.** Exhaustive matching is quadratic. Scaling the 50-frame
  match by pairs gives about 2 000 s of matching for 100 frames on 4 cores, and about
  9 h for 400. 100 frames of a one-minute orbit is one every 0.6 s. Frames are chosen by
  **`sharpness-windowed`**, the sharpest of each of 100 equal stretches, so that the cut
  cannot lose a whole blurred side the way global top-K could. (Since recipe v9 that is
  a photo set's rule only: a video's frames are chosen by camera motion, `select:
viewpoint` -- the sharpest of each window of ~10% of the view or ~1 deg of viewpoint,
  as many as the capture covers, up to `keep_video` -- see `keyframes.py`. Frame size is
  `max_side: auto`, 1600 unless the capture measurably holds more; `resolution.py`.)
- **4096 features at 1600 px** instead of the stage's 8192 at 2400: the same 50/50 in 62%
  of the time. Only the SfM sees the downscale; `train` reads the full frames.

That puts a real capture's pose at roughly **35–45 minutes on 4 dedicated cores**. This
is an extrapolation, not a measurement of 100 frames exhaustive; the 100-frame exhaustive
run at 8192 features was stopped rather than waited out. Thinning has a floor as well as
a ceiling: every third of the same 100 frames (30) registered only **4**. Feature
extraction peaked at 1.7 GB resident (matching 81 MB, mapping 52 MB), which is why
`docs/DEPLOYMENT.md § GPU training — Modal` sizes the Fly worker up before Lane 2. The
lever after that is COLMAP's GPU SIFT and matching, which would put `pose` on the Modal
box too. The Ubuntu package is built without CUDA (`colmap help`: "without CUDA"), so that
needs a COLMAP build in the training image. It is not done.

A fixture-sized check of the same stage runs in CI: 40 rendered frames, exhaustive,
40/40 registered.

### `mapper: global`, and the minutes that are not COLMAP

`pose` now runs on Modal's `cpu4`. Its first real 87-frame video took 301 s, of which
COLMAP was 192 s (extract 3.7, match 121, map 67). Two changes aim at the rest:

- **`mapper: global`** (opt-in; `global_sfm.py`) maps with GLOMAP as COLMAP 4 ships it,
  through the prebuilt `pycolmap==4.2.0` wheel in the CPU image, on a copy of the 3.9.1
  database; a result under `min_registered_fraction`, or a mapper that cannot run, falls
  back to the incremental mapper on the same matches. On the 40-frame orbit: 40/40,
  0.104° / 0.172% against incremental's 0.122° / 0.207%. Whether it saves most of the
  67 s of mapping on a real video is **unmeasured**; turn it on per run with
  `{"pose": {"mapper": "global"}}` and compare `mapS` with `globalMapS`.
- **Transfers.** A frames artifact is ~100 objects, moved one request at a time by both
  the worker and the container. Both now move eight at once, and every remote stage
  records where its wall time went: `stageInS` and `outputsBackS` (worker side),
  `remoteFetchS`, `remoteStageS` and `remoteUploadS` (container side), beside `billedS`.
  What is left of `billedS` after those is container start and queueing. The next real
  run is what attributes the 110 s; nothing here measured it.

### `colmap: "4.2"`: the pose stage on COLMAP 4.2

`pose` runs apt's COLMAP 3.9.1. `{"pose": {"colmap": "4.2"}}` runs extraction, matching
and incremental mapping on COLMAP 4.2 instead, through the `pycolmap==4.2.0` wheel the CPU
image already carries for `mapper: global` (`colmap4.py`): no second COLMAP build, and
the same matching plan, exhaustive fallback, mapper seeds and `poses.json` -- whose
`version` says which COLMAP ran, beside a new `meanReprojectionErrorPx` for both. 4.2
reads a faiss vocabulary tree, not 3.9.1's FLANN one, so the image carries the faiss build
of the same 32K-word tree at `$COLMAP4_VOCAB_TREE`, and pairs a video's frames the way
3.9.1 does (4.2's own `quadratic_overlap` drops the linear window; `colmap4._matching`).
The poses artifact stays 3.9.1's three files, which 3.9.1's `model_aligner` reads.

On the rendered orbits (4 cores, `colmap4.py` has the table) 4.2 matched 1.35-5x and
mapped 2-2.3x faster, with pose error within run-to-run noise of 3.9.1's: 87 frames at 1600x1200, sequential with
loop closure and the recipe's settings, 475 s -> 294 s (match 255 -> 189 s, map
193 -> 83 s), 87/87 both. The spool's 179 frames spent 266 s matching and 323 s mapping
on 3.9.1. Measured there on Modal's cpu4 (two previews each, 2026-09-28): 4.2 matched in
43-49 s and mapped in 187-225 s, 179/179 registered, with the same preview (24.06-24.07 dB
/ LPIPS 0.169 against 24.07-24.15 / 0.166-0.168). 4.2 is the recipe's default since;
`colmap: "3.9"` runs the old CLI.

## Refine from the preview

The phone's Refine re-runs `train` in the preview's workdir with `init_from: preview`
(`init_seed.py`). Every `train` run leaves a seed -- the centres, colours and opacities of
its visible gaussians -- in its `checkpoint/`, which a re-run keeps and `CloudRunner`
carries to the GPU box. The Refine appends that seed, cropped to the support mask, to
COLMAP's points (with empty tracks, so the depth loss keeps its real observations), and
trains `init_schedule_scale` (1.0: measured better than 0.5 on the spool, 25.85 vs 25.11 dB) of the schedule, because gsplat's `sfm` init turns
exactly those points into its starting gaussians. A seed trained against other poses is
refused by fingerprint and the run trains from COLMAP's points on its own schedule.

**Expected**: the measured 30k-step Refine was 1,278 s of L4 training (21 min, $0.31 all
in); half the steps from a dense start should be roughly 650-750 s (the early steps cost
more, starting near the cap rather than growing to it), about $0.14-0.17 less. **Check
on the first run**: `train_metrics.json`'s `init` block (seed points, budget) and
`requestedIterations` (15,000); held-out PSNR/SSIM/LPIPS against the 30k run's
(`train_metrics.json` of the earlier Refine); `trainSeconds`; the quality stage's
`keepPct`; and floaters in the viewer. If quality falls short, raise
`init_schedule_scale` (the phone may send it) before abandoning the seed; `init_from:
sfm` restores the old behaviour.

## How many gaussians, and for how long

`train`'s gaussian cap is `cap_max: auto` (`gaussian_budget.py`): the supported surface
counted in its own finest-view pixels -- per sparse point, depth / focal for the camera
that saw it largest, at the size training reads the frames; voxels of 32 of those
footprints, one face each -- times `gaussian_density` 0.1. A Refine counts its support
mask in full and the rest at a tenth. Clamped to the preview's 200k and to the L4's
memory at that frame size (gsplat's own 1M/2M/3M MCMC measurements: ~8.7M at 1600 px,
~5.4M at 2400; training rasterizes `--packed`, whose saving the model does not yet count).
The 2M `budget_max` the recipe used to set is gone: it was what `place`/`package` could
load whole on the 2 GB worker, and the stages after training now read the splat a chunk
at a time (see "Stages after training, a chunk at a time"); `budget_max` remains an
override. Measured offline, 2026-09-27: 3DGS's Truck model at its 979 px, **1.52M** (15.2M
footprints^2; the calibration point); four local phone/photo models at 1600 px, 0.42M-0.81M.
An integer `cap_max` is an override (the preview's 200k); a phone tier multiplies the
budget (`density_scale`: Quick 0.5, Best 2).

`converge: true` (`convergence.py`, `converge_trainer.py`) runs the trainer through a
wrapper that, after MCMC's densification ends (25k of 30k, scaled), evaluates the held-out
split every 500 steps and stops once the best PSNR of the last 2,000 steps is under
0.05 dB above the best before them -- by adding the next step to the trainer's own save,
export and evaluation lists, so the PLY and stats land where they always do. A budget over
1M may run a longer maximum, `sqrt(budget / 1M)` up to 2x. `train_metrics.json` has
`budget` (every input, and which clamp applied) and `convergence` (the held-out curve,
`stepsRun` of `stepsMax`, whether and why it stopped).

**Not yet run on a GPU. Check on the first run**: that `convergence.hook.hooked` is true
(the wrapper found gsplat's `cli` and the trainer's `Runner`); `peakMemoryGb` against the
memory model at the budget it chose; `trainSeconds` for a 1.5-2M budget on the L4
(~35-50 min at 30k, extrapolated from gsplat's A100 table and the spool's 21 min at
500k); where the curve flattens relative to `refineStopIter` -- the rule can only save
the last sixth of a schedule, so a curve still rising at the end says the maximum, not
the rule, is what binds; and `[benchmark:recipe]` against `[benchmark:recipe-500k]`.

## Blocks, one GPU each

A budget more than one GPU trains (`blocks: auto`), or `blocks: <n>`, trains the scene as
blocks and merges them into one `trained.ply` (`blocks.py` has the recipe and its sources).
Measured on the L4 with the spool forced to 2 blocks: quality matched the whole run
(26.42 dB / LPIPS 0.1056 merged, 26.39 / 0.1088 whole), but one call trained the blocks in
turn -- a 7.5k-step coarse pass, then 2,530 s and 2,207 s of blocks -- for 2.3 h and
$1.73 against ~1 h and $0.67 whole. Modal bills per GPU-second, so the blocks now train
**at once, one GPU each**, which costs the same seconds and ends with the longest block.

`CloudRunner` does the fanning out (`contracts.FanOut`), because it already owns what a
piece of work on a GPU needs -- retries, preemption, fallback, the checkpoint and the
attempt ledger. One stage attempt is three kinds of call through the same adapter:

| call     | runs                                                        | writes                        |
| -------- | ----------------------------------------------------------- | ----------------------------- |
| head     | the prior (or coarse pass) and the camera test, **once**    | `checkpoint/blocks/plan.json` |
| part × N | one block each, `block_parallel` (4) at once, longest first | `blocks/block_NNN/` + `.json` |
| join     | merge, merged evaluation, held-out error                    | the stage's outputs           |

A part runs on its own checkpoint key (the prior and the plan copied onto it), so N parts
syncing at once never overwrite one another; only the paths it declares come home. A part
that fails or is preempted is resubmitted alone (`part_attempts`, 3 calls) while the others
carry on; one that never succeeds ends the attempt only after the rest have finished, and
the next attempt's head lists only the blocks with no record -- a finished block is never
trained twice, and with a single block left the head trains it itself. Every call is an
entry in `attempts.json` (`part`: the block, or `join`), so `billedS`, `costUsd` and
`run_cost` are sums over concurrent calls; `fanOutWallS`, `fanOutBilledByPart` and
`fanOutPeak` are on the step. In the stage log each part's lines carry `[bN]`; the progress
bar follows the slowest block (the stage ends when it does) and the live viewer the most
advanced block's snapshot.

No GPU waits on another: the head returns before the parts start and the join starts after
they end. That is why the fan-out is not a GPU container spawning children, nor a Modal CPU
function orchestrating them (`cpu4`): the first bills a GPU to wait, and the second would
need its own retries, fallback and pricing, and its children's seconds would never reach
the ledger. `block_parallel` is capped by `CloudRunner(max_parallel=8)`: keep that under
the Modal workspace's GPU concurrency limit divided by the runs the worker trains at once,
since a part queued beyond the limit sits `pending` and is cancelled after `max_pending_s`.
`block_parallel: 1` (or a runner that does not fan out, such as `LocalRunner`) trains the
blocks in turn in one call, as before, bounded by the 5 h in-attempt budget.

**Expected, not yet measured.** The spool at 2 blocks: 2.3 h less the shorter block
(2,207 s), about **1.7 h**, for the same ~$1.73 plus two container starts and dataset
copies (a few cents) -- and with a Preview's prior, no coarse pass either. A large scene of
4 blocks of ~45 min each: head + 45 min + join instead of head + 3 h + join (and no 5 h
yield), for the same GPU-seconds.

**Each block's schedule follows its own frames** (`block_schedule`). `frames`, the
default, is the single run's rule -- `training.schedule_scale`, linear in frames up to
`schedule_full_at` (60), never below `schedule_floor` -- applied to the frames the block
is given, then the block's own gaussian factor and convergence stop, as for a whole run.
A block given every frame keeps the run's schedule exactly: the spool's two blocks (all
156 training cameras each) are unchanged. With the recipe's 60 and the 50-frame minimum a
block must have, `frames` shortens only a block of 50-59 frames; `share` (the run's
schedule times the block's share of the frames, so each frame is visited about as often
as in the whole run) is what would shrink a large scene's blocks, and is opt-in until a
GPU run has measured it. `full` is every block the whole schedule.

**Validate on the GPU**: the spool at `blocks: 2` (and `block_parallel: 1` as the
control) -- `fanOutWallS`, the two parts' `billedS` against the serial 2,530 s / 2,207 s,
PSNR/LPIPS against 26.42 / 0.1056; then a large capture at `blocks: 4` with
`block_schedule: share` against `frames`.

**Where a part's billed seconds go.** Measured on the L4 (spool, `blocks: 2`, bilateral
grid, job 62d796d4): parts billed 4,191 s and 4,323 s against 2,967 s and 3,048 s of
`blockSeconds` (the trainer's process), ~1,250 s each beyond it. Everything a part does
outside the trainer was timed on this repository's CPU on a real 100-frame capture with a
1M-gaussian prior and a 1M-gaussian SH-3 block: the dataset (now hard-linked) under 0.5 s,
both budgets 0.4 s, the prior 0.2 s, the seed 5 s, the ring, the read-back, the crop and
the parts file 2.5 s -- about 10 s. So the minutes are before the function body (the GPU
queue, the container's start), after it returns (the runner noticing), or in transfers,
and nothing recorded which. Now every call says, additively, to its billed figure:

| metric         | what                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fanOutPhases` | per part: `start` (submit to function body: queue + cold start; `cold:1` if the container was new), `import`, `fetch`, the stage's `stageDataset`, `budget`, `prepare`, `loadPrior`, `dataset`, `seed`, `train`, `post`, `stageOther`, `finalSync`, `output`, `rest` (the result reaching the runner, clock skew); not summed: `bgSync` (the syncer beside the stage), `collect` (the runner fetching the result) |
| `headPhases`   | the head call, the same way (`prepare` is the prior and the camera test)                                                                                                                                                                                                                                                                                                                                          |
| `remotePhases` | the last call: the join (`merge`, `eval`, `holdout`) or the one call                                                                                                                                                                                                                                                                                                                                              |
| `blockPhases`  | each block record's `phases`, as the join saw them                                                                                                                                                                                                                                                                                                                                                                |
| `blockTrainer` | inside each block's trainer (`trainer_timing.json`): `setup`, `eval`, `ply`, `steps`, `evals`                                                                                                                                                                                                                                                                                                                     |

The container reports `remoteStartedAt`/`remoteEnteredAt`/`containerBootedAt` (wall
clock), `containerCall`, `remoteImportS`, `remoteFinalSyncS`/`remoteOutputS` (with their
bytes) and the syncer's `remoteSyncs`/`remoteSyncS`/`remoteSyncBytes`; the runner submits
with its own wall clock and puts the two together (`cloud.call_phases`).

**One slow part, and one that waited (job 33bc1bff, spool, `blocks: 2`, L4).** b1 waited
1,697 s before its function body ran, then trained 28,786 steps at ~18 it/s; b0 started in
12 s and trained 29,926 steps at ~3 it/s -- 10,089 s of trainer steps, billed 10,285 s --
on similar blocks (626k / 605k gaussians, the same 156 cameras, no ring). Both report
`cold:1` (`containerCall` 1: each was its container's first call), so b0 did _not_ run in
the head's warm container, and nothing of the head's could have shared it. What was on b0's
side: its final checkpoint sync, which touches no GPU, took 65 s against b1's 4.4 s. A GPU
function that asks Modal for no CPU is reserved 0.125 of a core and bursts into what its
host's other tenants leave, and the trainer is CPU-fed (four DataLoader workers decoding a
JPEG a step, a Python loop launching each step's kernels), with thread pools sized to the
_host's_ `os.cpu_count()`. So, ranked: a CPU-starved container (most likely); a slow or
throttled GPU; different work (least: same code, cameras and schedule). Changed:

- **Every GPU function reserves 2 cores and 8 GiB** (`infra/modal/app.py`, `GPU_CPU_CORES`,
  `GPU_MEMORY_MIB`; Modal bills max(reserved, used), so at most +$0.16/h over an L4's
  $0.80), and the image caps OpenMP/MKL/OpenBLAS/OpenCV pools at 4 threads.
- **The machine is measured**: `remoteHost` (per call; `headHost`, and `fanOutHosts` per
  part) from `host.HostWatch` -- `cores` (CPU seconds over wall: what the call actually
  got), `cpus`, `cpuQuota`, `throttledS`, `psiCpu`, `load`; `gpuUtil`/`gpuUtilMin`,
  `smMHz`/`smMHzMin`, `tempC`, `powerW`, `throttle` (nvidia-smi's reasons, OR-ed), and
  `remoteGpu`. The next run says which it was: starved is low `gpuUtil` with low `cores`
  and high `throttledS`/`psiCpu`; a slow GPU is high `gpuUtil` at a low `smMHz` or a
  `throttle` bit; different work is neither.
- **A call leaves nothing running** (a warm container's next call would share it):
  `remote.execute` kills every process started under it, however it ends
  (`remoteReaped`), names any thread still alive (`remoteLeftoverThreads`), joins the
  checkpoint syncer for as long as a sync in flight takes, and the app removes the
  sandbox. `progress.stream` kills its tool when the reading stops. Not
  `single_use_containers`: reuse only happens within Modal's scale-down window, it saves
  a cold start, and the slow part was a fresh container anyway.

**Billed from the container, not the submit.** A finished Modal call is now billed from
its container's start (a cold call: `containerStartedAt`, the sandbox's uptime subtracted
from the clock; else `containerBootedAt`) or its entry (a warm call) to `remoteFinishedAt`
(`cloud.container_billing`), clamped to the runner's wall time; the wait before it is
`queueS` (per call in `attempts.json`, summed on the step), not a cost. `billing`
(`billingBasis` on the step) says `container` or `wall-proxy` -- the old figure, kept for
a call that reported no clocks (failed, preempted, an older image). In `call_phases`,
`start` is then the cold start alone and `queue` sits beside the sum. Not counted by
either: a container's idle scale-down window after its last call, which Modal bills and
no call owns; and the reserved CPU and memory, which `providers.py` does not price.

**A part may start on an L40S** (`modal_adapter.GPU_FALLBACKS`: `l4 -> (l4, l40s)`,
deployed as `run_stage_l4_fallback` with `gpu=["L4", "L40S"]`, Modal's ranked list: the
first type free wins). Parts only -- a single run, a head and a join keep their L4. The
L40S is 2.44x the L4's price and trained ~2.3x faster on our benchmark, so a block costs
~17% more by that benchmark (+6% on training seconds at list prices), and ends in under
half the time instead of waiting half an hour. The A10 ($1.10, 1.38x) is left out until
gsplat has been timed on one. The container reports its GPU, and the call is priced and
ledgered at that tier (`fanOutTiers`); a deployment without the fallback function runs the
part on its own tier; `ModalAdapter(part_fallback=False)` turns it off.

What was cut, none of it a training step (the merged `trained.ply` stays byte-identical
to the serial one's, `test_blocks.py`):

- **Evaluation renders.** gsplat's `eval()` writes every val frame's ground truth and
  render side by side as a full-size PNG, which `converge_trainer.py` then deleted unread:
  ~1.1 s of zlib per 4 Mpx canvas measured here, ~1.7 s at 2,400 px, x 22 frames x the 15
  evaluations of a converging 30k run -- **~7-10 min per block**, and per single run with
  `converge` (inside `blockSeconds`, which is why it did not show as overhead). The
  wrappers now give the trainer an `imageio` that skips `renders/`; the join's merged
  evaluation runs through the block wrapper for the same reason (~40 s).
- **LPIPS on intermediate evaluations**, which the rule never reads (PSNR only): skipped,
  and dropped from their stats; the last evaluation, the one reported, keeps it.
- **Re-uploading what was just downloaded.** `S3Transfer.get` now remembers what it
  fetched, so the syncer's first sync no longer sends back a part's prior, or the join's
  every finished block (hundreds of MB to GBs at the L4's budget ceiling).
- **Bringing home more than the result.** A finished part's declared members are fetched
  one by one, not its whole key (its prior and live snapshots came back too).

### Batched steps (`batch_size`)

`batch_size: B` (1-8, default 1) trains B images a step. gsplat v1.5.3 scales every
learning rate by `sqrt(B)` and Adam's eps by `1/sqrt(B)` but does not shorten the
schedule, so the stage divides `--steps_scaler` by B: the same images trained on, in B
times fewer steps. The refine window, SH interval and evaluation steps are scaled by the
trainer with it, and `converge_trainer.py` scales its window by `cfg.steps_scaler`, so
the convergence stop follows the shorter run unchanged. Refused with `depth_loss` (a
batch of per-frame SfM point lists does not collate); frames must share one size; the
`absgrad` assert in `rasterization` is multi-GPU only. The memory model counts B frames
of raster memory, so the budget's ceiling -- and with it `blocks: auto` -- accounts for
it. Whether B > 1 is faster per image on an L4 is exactly what is not yet measured:
`batch_size: 2` and `4` on the spool against 1, comparing `trainSeconds` and PSNR/LPIPS.

## Optimised parents (`optimise_lod`)

`package` merges each parent tile from its subtree by Hierarchical 3DGS's moment matching;
Phase 1 measured that ahead of thinned parents on PSNR, SSIM and holes, not on LPIPS at
the switch distance -- merged parents are blurry. `optimise_lod` (recipe 10) optimises
them against the photos, as H3DGS Sec. 5.1 does its interior nodes: leaves frozen, a random
training frame and a log-uniform tau in [3, 64] px each step, Cesium's own REPLACE cut at
tau rendered at full resolution, the trainer's 0.8 L1 + 0.2 D-SSIM, train_post.py's
learning rates, opacity kept at most 0.99 for SPZ. It is a GPU stage between `place` and
`package` because the tree is a function of `canonical.ply` -- after `quality`'s crop and
`place`'s east/north/up -- and it builds that tree with the packer's own code
(`splat_tiles.hierarchy`); the parents come back keyed by tile and cell and fingerprinted
by the PLY's sha256, and `package` refuses them for any other tree. `lod_parents.py` is
the stage, `lod_optimise.py` the torch half, `lod_maths.py` the tested numpy half.

The parents are kept only if the held-out frames' loss at the cut fell; a failure, a
rejection or `enabled: false` leaves the merged parents, as before. Iterations are planned
so each parent is optimised about as often as H3DGS's 15,000 optimise each of its nodes
(15,000 x ln 2 / ln 20 = 3,471 choices), from the measured rate at which the training
frames' cuts choose it: 3,471 when there is one parent tile (a scan of about 1M gaussians
at 100k a tile), at most 15,000.

**Not yet run on a GPU. Expected** on the L4: a step renders the cut (up to every leaf)
and back-propagates like a training step at the same count, so ~10-15 steps a second at
~1M gaussians (the measured Refine ran 23 steps a second at 500k); 3.5k steps is ~5 min,
plus ~2 min of tree building, target renders and evaluation and the container's start --
about $0.10 at $0.80 an hour; the 15,000-step ceiling ~$0.40; `budget_s` (1 h) bounds it.
**Check on the first run** (`stages/optimise_lod/out/lod_parents/summary.json`):
`itPerSecond` and `loopSeconds`; `timesChosen` (every parent tile trained); `before` /
`after` held-out loss, PSNR and LPIPS per tau; `switchDistance` -- each parent alone from
where its error projects to 16 px against its own leaves, merged and optimised, the
measure Phase 1 made; `accepted`. `experiments/lod_compare.py` repeats Phase 1's exact
protocol (root at the switch distance, Cesium's cut from four distances) on the GPU, on
two tilesets packed from the run's `canonical.ply` with and without `--parents`.

## Stages after training, a chunk at a time

`quality`, `place`, `thumbnail`, `ground_samples` and Lane 1's `normalize` never hold the
splat. They read it in fixed row ranges (`splat_io.py`: `SplatReader` parses the PLY
header as the packager does and reads a range with one read, `PlyWriter` appends
`canonical.ply` byte-identical to `gaussians.write_ply`, `ColumnStore` keeps
per-gaussian results on disk between passes), and get every whole-splat statistic in
passes over the ranges (`outofcore.py`: radix selection that reproduces `np.median` and
`np.percentile` bit for bit, and group-by counts in hash partitions). `splat_stream.py`
is `gaussians.orient`/`transform`/`render_thumbnail`/`ground_samples` on such a stream;
`quality.support_pass` regroups the opaque occluders into the same 2^18-row blocks the
whole-splat stage used and tests each camera against spatial cells of a chunk before it
projects any of it.

The outputs are the whole-splat stages' own: `tests/test_chunked_equivalence.py` runs
each beside the path it replaced (for `quality`, a frozen copy of the old stage,
`tests/quality_in_memory.py`) on a few hundred thousand gaussians cut into prime-sized
chunks and requires the same bytes -- except a ground sample's longitude and latitude,
whose cell mean is a float64 sum here and a float32 pairwise one in `np.mean` (under a
micrometre). Peak memory (`VmHWM`), measured by `tests/memory_probe.py` in a process of its own
(synthetic orbit, 16 cameras, held-out arrays):

| gaussians | whole-splat quality + place | chunked quality + place + thumbnail + ground |
| --------- | --------------------------- | -------------------------------------------- |
| 250k      | 114 MB                      | 162 MB                                       |
| 1M        | 328 MB                      | 184 MB                                       |
| 4M        | 1,067 MB                    | 182 MB                                       |
| 8M        | fails under a 1.5 GB limit  | 193 MB, 81 s                                 |

Lane 1 on a phone's SH3 upload (gsplat's 59-float rows), `normalize` + `thumbnail` +
`ground_samples`: 711 MB whole against 139 MB chunked at 1M gaussians, 1,386 MB against
144 MB at 2M.

`tests/test_bounded_memory.py` holds the scaling (150k against 600k gaussians, with the
fixed-size buffers shrunk so both are past them: no growth, where the whole-splat path
grows 125 MB) and, with `PIPELINE_BENCH=1`, the 8M run under a 1.5 GB address-space limit.
Each stage takes `chunk_gaussians` (2^18 rows by default); nothing it computes depends on
it.

## Shipping SH (view-dependent colour)

gsplat trains spherical harmonics to degree 3 -- 45 `f_rest_*` of a gaussian's 59 floats
-- and until `ship_sh_degree` every capture shipped degree 0: one colour from every side.
The packer and both web renderers already carry and draw SH when a PLY has it; the
pipeline was dropping it in the middle. **`ship_sh_degree`** (0-3, **0 by default**, so
nothing changes until it is chosen) is how many bands ship:

| lane | where it is set                  | what carries it                                                                                  |
| ---- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| 2    | `train: {ship_sh_degree: N}`     | `trained.ply` -> `quality`'s `gated.ply` -> `place`'s `canonical.ply` -> `package`'s tiles       |
| 1    | `normalize: {ship_sh_degree: N}` | the upload's own bands (a gsplat-style PLY, a Scaniverse `.spz`) -> `canonical.ply` -> the tiles |

It is not gsplat's own `--sh_degree` (the degree it trains at, left at its default 3):
training at 3 and shipping the first N bands is the least-squares best degree-N colour,
since the bands are orthonormal. The bands are truncated channel-major -- degree 1 of a
degree-3 file is `f_rest_{0-2, 15-17, 30-32}`, renumbered `f_rest_0-8` -- and written after
`f_dc_*`, where the trainers put them (`gaussians.ply_properties`; degree 0 is the
fourteen, byte for byte).

**Turning them.** A gaussian's SH colour is a function of direction in the frame it was
fitted in, so every stage that turns the splat turns the bands: `place` (COLMAP's frame
into east/north/up: the EXIF similarity or camera-up levelling, and a heading) and Lane
1's `normalize` (up axis and heading), both through `gaussians.transform`, which now
applies `harmonics.rotate` -- each band by the rotation's real Wigner D-matrix in the
trainers' basis, fitted exactly against Inria's `eval_sh` rather than taken from a
recurrence whose conventions differ in every source. `tests/test_harmonics.py` holds it to
the definition for degrees 1, 2 and 3: turn the splat by R and look from R d, and every
gaussian shows the colour it showed from d (to 2e-6), through `place` and `normalize` as
well as on arrays; degree 1 is also held to its closed form `P R P^T`, and each band's
matrices to being an orthogonal representation (D(R1 R2) = D(R1) D(R2)). The recentring
is a translation and turns nothing; a mirror is refused. Nothing else in the pipeline
turns a splat: `real_tree.py`'s similarity (the Minnetonka rig step) reads only the
fourteen, so its tiles stay degree 0 whatever the run shipped.

**Measured as shipped.** `holdout_error.py --sh-degree N` renders the held-out frames at
the degree `trained.ply` ships, so the per-gaussian error `quality` gates with is the
error of what is published; `meanPsnrFullSh` (every band trained) sits beside `meanPsnr`
in `holdout.json`, `train_metrics.json` and `quality.json`, so what the cut costs is
visible. A block run merges the same truncation into `trained.ply` and measures it the
same way. `train_metrics.json`'s settings, `place`'s metrics, `source_meta.json` and the
manifest's `splat.shDegree` (read back off the tiles, as CesiumJS counts their
attributes) say what shipped.

**What it costs, and choosing.** On disk, from the packer (`splat_tiles.convert`): real
SH-3 scans (nianticlabs/spz's samples) 1.17-1.20x the tileset bytes at degree 1 and
1.6-1.7x at degree 3; synthetic 300k-gaussian splats with trained-looking coefficients
(Laplace, b = 0.03) 1.15x and 1.51x (16.1, 18.5 and 24.4 bytes a gaussian), with larger
coefficients (b = 0.1) 1.27x and 2.15x; packing 1.6, 2.1 and 3.2 s. `canonical.ply` grows
from 56 to 92 (degree 1) or 236 (degree 3) bytes a gaussian, read a chunk at a time like
the rest. What it costs the viewers -- frame rate, GPU memory, load time -- is what has
not been measured, and what `experiments/sh_compare.py` exists for:

```bash
# 1. A run of the capture that ships degree 3: jobs.params
#      {"train": {"ship_sh_degree": 3}}
#    Raw PLYs with f_rest_* are not kept by a run: gsplat's export is in the training
#    container's work/, and a block run's parts leave checkpoint/ once merged. So a run at
#    degree 3 is the source; its canonical.ply is already placed, its SH turned.
# 2. One site, three ways, as three sites side by side (1.5 scan-widths apart, east):
cd tools/pipeline
uv run python experiments/sh_compare.py <run>/stages/place/out/canonical.ply ../../data/tiles \
  --georef <run>/stages/georeference/out/georef.json --site spool
#    or from the run's trained.ply (COLMAP's frame), placed by the real place stage:
#    ... <run>/stages/train/out/trained.ply out/ --georef georef.json --place [--poses poses/]
# 3. Seed and look: data/tiles/spool-sh{0,1,3}/ each hold splat/ and a site.json.
cd ../../apps/api && uv run python -m app.seed && cd ../.. && pnpm dev
```

`sh_compare.json` beside them has, per degree, the tiles, bytes, bytes a gaussian, the ratio
to degree 0 and the seconds to pack. In the console, with the renderer under comparison
chosen (settings; PlayCanvas by default, Cesium, Spark): fly to each, read frame rate and
GPU memory in the developer panel (`D`) with the others out of view, time the first tile
and the whole tileset in the browser's network panel, and look at shine and colour as the
view goes round. Then set `ship_sh_degree` in `recipes/photo-reconstruct.yaml` (and
`splat-ingest.yaml`) to the degree chosen. The site folders are local only
(`data/tiles/*-sh[0-3]/` is gitignored).

## Lane 1

```
upload/capture.ply ──▶ canonical.ply ──▶ splat/ ──▶ thumbnail.jpg
      or .spz            source_meta       ▲        ground_samples.json
                              │            │            manifest.json
                         georef.json ──────┘        registration.json
```

### What arrives, and what is refused

`normalize` (`ingest_splat`) reads a **3DGS PLY** (Scaniverse, Polycam, Postshot, Luma,
OpenSplat, gsplat) or an **`.spz`** — the format Scaniverse exports natively and the one
`splat_tiles.pack_spz` already writes, so ingesting it is a 38-line inverse and nothing
else. A PLY that carries `red/green/blue/alpha` instead of `f_dc_*`/`opacity` is converted;
`f_rest_*` is read and dropped unless the run ships SH (`ship_sh_degree`, 0 by default --
see [Shipping SH](#shipping-sh-view-dependent-colour)): at degree 3 it would quadruple
`canonical.ply`, and whether the colour is worth that is still to be chosen.

Everything else **refuses with a message that names the file and the problem** rather than
producing a splat-shaped nothing: ASCII PLY, a vertex element with list properties, a
truncated file, an `.spz` with the wrong magic, a compressed PlayCanvas/SuperSplat export
(named, with the `ply_compressed` impl that will read it), and an upload with no splat in
it at all. `tests/test_ply_variants.py` holds one case per shape.

Two of those were A0 #3, and they had to be fixed **together** in
`tools/captures/splat_tiles.py`'s reader: a type map that knew five of PLY's sixteen scalar
type names (so a compressed export was an unhandled `KeyError: 'uint'`), and a header
parser that kept collecting `property` lines past the second `element` (so a mesh PLY's
trailing `element face` joined the vertex dtype, every gaussian was read at the wrong
stride, and the read came back **silently** with `|x| max = 1.7e38`). Fixing only the type
map would have turned the loud failure into the silent one.

### The up axis

Everything downstream reads `canonical.ply` as east/north/up with z up -- the tileset's
node matrix, the thumbnail, the ground samples -- and until 2026-09-23 nothing converted
an upload into that frame. So a Scaniverse `.spz` (y up) and a 3DGS/COLMAP `.ply` (y down)
both landed tipped 90 degrees, and `splat_ground` measured "ground" along the capture's
depth, which the viewer's clamp then dutifully rested on the terrain.

`ingest_splat` now turns the file into east/north/up (`gaussians.orient`): positions, each
gaussian's quaternion (`q' = q_R * q`), and -- when the run ships them -- the SH bands
above DC, by the rotation's Wigner D-matrix ([Shipping SH](#shipping-sh-view-dependent-colour));
the DC colour is exactly invariant. Which axis is up:

| Source                                    | Default | Evidence                                                                                                                                                                   |
| ----------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.spz`                                    | `y`     | the SPZ README ("RUB coordinate system following the OpenGL and three.js convention"); two real Scaniverse share-page scans render upright y-up and upside down y-down     |
| `.spz` counter-examples                   | --      | six of Spark's sample `.spz` files are y **down** (Spark's quick start rotates `butterfly.spz` 180 degrees): written without declaring a frame. They need `upAxis: "-y"`   |
| `.ply`                                    | `-y`    | the SPZ README ("PLY ... typically uses RDF"), Niantic's own `saveSplatToPly` converting to right/down/forward; Inria's `train` renders upright y-down                     |
| Inria 3DGS / gsplat / nerfstudio (COLMAP) | `-y`    | the COLMAP world frame is the first camera's, y down, tilted by however it was held (nerfstudio's parser says so); nerfstudio's own exporter writes its z-up world instead |
| Polycam, Luma, KIRI, Postshot `.ply`      | `-y`    | **not measured** -- no public sample downloadable without an account; this is the PLY convention, and the override is the remedy                                           |

A capture's `metadata.upAxis` (`z`, `-z`, `y`, `-y`, `x`, `-x`) overrides the default and
`metadata.headingDeg` turns it about the vertical; the API refuses anything else at
creation, and the worker hands both to whichever stage runs `ingest_splat`. There is no
`auto`: a plane normal has a sign nothing in a splat resolves, and an object has no ground
under it. The origin moves to the footprint's centre and the lower quartile of the per-cell
ground heights, which cannot change the viewer's clamp (a constant vertical shift moves
every sample by the same amount) and puts the placed coordinate in the middle of the
capture. `source_meta.json`'s `frame` records all of it.

Before and after on real downloads -- the thumbnails are in the session's scratchpad, and
`tests/test_up_axis.py` holds the same facts on synthetic trees of known orientation,
needles and all:

| Sample                               | Before (z assumed)                                 | After (format default)                            |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------------------- |
| Scaniverse `oebjag65cbkuvm42` (.spz) | lying down: the elevation view shows it from above | a hedge standing on a path                        |
| Scaniverse `jb4dj3iobbwt6px2` (.spz) | lying down: seen from above                        | a bin and bushes upright on the ground            |
| Inria `train` (.ply)                 | the locomotive on its side                         | upright                                           |
| Spark `cat.spz` (.spz, y-down file)  | lying down: seen from above                        | upside down, as its file is: needs `upAxis: "-y"` |

`.spz` versions 2 and 3 are read; version 4 (ZSTD streams) is refused by name.

### `ground_samples.json`

The capture's own ground height per grid cell — the half of the height-offset subtraction
that used to be done by hand. **Since B4 it is consumed**: `catalog` carries the whole
document into `registration.json`, the worker adds `origin.height` to each cell's `z` and
puts the result on the registered asset as `renderConfig.groundSamples`, and the viewer
samples the ground at exactly those longitudes and latitudes and takes the median
difference (`apps/web/src/cesium/placement.ts`). The slope cancels, because both sides of
every subtraction are at the same point — which is what the bounding-box clamp it replaces
could not do, and why a sloping capture used to need a person to type a correction into
`heightOffsetM`.

```json
{
  "frame": "enu",
  "origin": { "lat": 28.0389, "lon": -82.6966, "height": 22.5 },
  "cellM": 2.0,
  "percentile": 5.0,
  "method": "p5 of the gaussian up-coordinate in each 2 m cell",
  "medianZ": 3.941,
  "samples": [{ "lon": -82.6966, "lat": 28.0389, "z": 1.211, "n": 5156 }]
}
```

`{lon, lat, z, n}` is the shape `tools/captures/ground_samples.py` already prints, so there
is one shape rather than two. `z` is the capture's own up axis in metres relative to the
placed origin, so a sample's ellipsoid height is `origin.height + z` — and that addition is
done once, in `app/worker/registration.py`, so the browser compares two heights in one
datum. The cells are ranked by how many gaussians they hold and tie-broken by position —
deterministic, unlike the sibling script, which picks cells with a seeded RNG.

`method` is spelled out because a low percentile of a _tree_ is canopy, not ground: this is
the capture's own low surface per cell, and it is the ground only where the capture has one.

### `georef.json`

Where the capture's local frame sits, and how that was decided. Every producer writes the
same five keys — `lat`, `lon`, `height`, `georefMethod`, `scaleSource`, `uncertaintyM` —
because `splat_tiles`, `capture_manifest`, `catalog` and the worker all read them, and
`app/models/enums.py` fixes the two vocabularies.

`manual_placement` (Lane 1) writes a coordinate somebody chose, `scaleSource: unresolved`
and ten metres of uncertainty. `exif_gps` (Lane 2) writes one of two shapes:

- **aligned** — three or more frames carried an EXIF fix and there is a pose model, so the
  document also carries an `alignment` block: the similarity `colmap model_aligner`
  solved for, its per-image residuals, and the count within the RANSAC threshold. This is
  the only thing in the project that makes `scaleSource: exif-gps` true.
- **located** — fixes but no poses, or an iPhone video whose only coordinate is the one
  `ffmpeg_frames` scraped off the container. The capture is placed and nothing else is
  claimed: `scaleSource: unresolved`, `alignment: null`, and ten metres — the same as a
  hand placement, because a coordinate with no orientation is not a better answer.

Three things it will not say:

- **the height is not an ellipsoid height.** `GPSAltitude` is metres above mean sea level,
  tens of metres from the WGS84 ellipsoid the globe draws. The number goes in with
  `fixes.heightDatum` beside it saying so, and the viewer's clamp is what actually rests
  the model on the ground.
- **the residual is not the accuracy.** Every fix in one capture shares the receiver's
  bias, and a bias common to all of them moves the whole reconstruction without changing a
  single residual. `uncertaintyM` is floored at five metres for that reason, and the
  residual is reported separately as what it is.
- **the similarity is applied, and not by this stage.** `alignment.applied` is `true`
  since the `place` stage exists: `train` writes `trained.ply` in COLMAP's own frame
  (gsplat's world normalisation is off for exactly this), `georef.json` carries the
  transform as `frame`, and `place` turns the splat by it into `canonical.ply`. Checked
  against the scene rather than against itself: on the rendered orbit, COLMAP's sparse
  points placed this way sit a median 0.011 m from the tree and ground they were rendered
  from, and 0.41 m unplaced.

Without GPS -- the ordinary iPhone video -- `frame` is a **levelling by camera-up**: the
mean of every registered frame's up vector, which is gravity for footage filmed the way
people hold phones (nerfstudio's default `orientation_method="up"` and gsplat's
`similarity_from_cameras` use the same estimate). On a real 50-frame reconstruction of
hand-held photographs the dominant plane came out 0.39 degrees from vertical after
levelling, with 71% of points above it and 3% below. Heading is not knowable from images
(the capture's `headingDeg` turns it), scale stays unresolved (`scale`, metres per model
unit, defaults to 1), and the splat is recentred on its own footprint as Lane 1 is. A
video with no location falls back to the capture's own `lat`/`lon` -- the console sends
where its camera was looking -- recorded as `manual`; with none of the three the stage
refuses rather than placing the capture at (0, 0).

### `manifest.json`

Sensor, date, resolution, georeference method, scale source, uncertainty, licence and the
tools that produced everything else — the row of the plan's artifact table that is easiest
to skip and the one that lets the inspector stay honest.

Three things it says plainly:

- **`georeference.uncertaintyM` is 10 m, not 0**, for a capture placed by hand, and
  `scaleSource` is `unresolved` until somebody says where metric scale came from. A
  hand-placed capture with zero uncertainty is the inspector claiming a survey.
- **`resolution.gsdM` is `null`.** A splat has no pixels behind it. The honest analogue is
  `medianGaussianM`, the median gaussian radius, and it is reported instead of inventing a
  ground sample distance for a phone scan.
- **`splat.gaussiansPackaged` and `bboxLocalM` are read back out of the GLB that was
  actually written**, not out of the packer's return value, so the manifest and the tileset
  cannot disagree.

It carries **no wall clock and no run id**. `capturedAt` is the capture's date; when the run
happened is already in `step.json` and in the job row, and putting it here is the one thing
that would stop two runs over the same bytes producing byte-identical outputs. `tools` names
the pipeline version, numpy, Pillow, and the **sha256 of `splat_tiles.py`** — that project is
`package = false` and has no version number, and a checksum is a more useful thing to compare
two runs on than a version string nobody bumps.

The whole manifest rides along in `registration.json`, so the worker writes it into the
site's metadata without fetching a second file, and `bboxLocalM` is what gives the site a
boundary the size of the capture instead of A7's 60 m placeholder square.

## Parameters per run

A recipe is a deployment-level document. It can say a capture is placed by hand; it cannot
say _where this one_ was placed, what took it, or what the site should be called.

```python
placed = {"georeference": {"lat": 51.5007, "lon": -0.1246}}
recipe = load_recipe("splat-ingest").with_params(placed)
```

Merged over the recipe's own parameters, keyed by **stage id** (two stages may run the same
impl), and an override naming a stage the recipe does not have is **refused** — a coordinate
that silently went nowhere would put the site in the Gulf of Guinea and say nothing.

The worker resolves these per run in `app/worker/params.py`: the capture's coordinate goes
to whichever stage places captures by hand, its sensor and date to whichever stage produces
`source_meta.json`, its slug and name to `catalog`, and `jobs.params` — recorded since A2
and ignored until A8 — wins over all of them.

## Verify

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -q
```
