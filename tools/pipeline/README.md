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

| Piece                                         | State        | Evidence                                                                                                                                       |
| --------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| frames from an iPhone-shaped HEVC `.mov`      | verified     | portrait (display matrix -90), HEVC, `mdta` location: 100/100 frames upright, location read (`tests/test_normalize.py` and a real-frame check) |
| poses (COLMAP 3.9.1, CPU)                     | verified     | rendered orbit 40/40 (CI); real photographs, 50/50 exhaustive; timing in [Where pose runs](#where-pose-runs)                                   |
| levelling by camera-up                        | verified     | real reconstruction: dominant plane 0.39 deg from up after levelling; rendered orbit within 5 deg (CI)                                         |
| the EXIF similarity, applied (`place`)        | verified     | rendered orbit with synthetic GPS: sparse points 0.011 m from the scene placed, 0.41 m unplaced (CI)                                           |
| a video with no location                      | verified     | falls back to the capture's `lat`/`lon`, recorded `manual`; with neither it refuses by name (CI)                                               |
| `train` argv and output layout (gsplat 1.5.3) | verified     | parsed by v1.5.3's own `simple_trainer.py` CLI in a CPU replica of the image's venv (CI job `trainer`); file names read from the source        |
| the Modal training image                      | **unproven** | every artifact it names exists and was pinned; the App builds locally; no image has been built by Modal                                        |
| a training run                                | **unproven** | none has happened. `.github/workflows/modal.yml` is the proof, at about $0.07-0.20                                                             |
| `ModalAdapter` against a live workspace       | **unproven** | read against `modal==1.5.5`; the same workflow's smoke is its first real call                                                                  |

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

That check moves the adapter from _guessed_ to _read_. It does not move it to _verified_,
and the table above is unchanged on purpose: running it needs an account and a token this
repository does not have, so `ModalAdapter` stays **unproven**.

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
  cannot lose a whole blurred side the way global top-K could.
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
`f_rest_*` is read and dropped, because `convert` never reads it and carrying it would
quadruple `canonical.ply` for nothing.

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
gaussian's quaternion (`q' = q_R * q`), and nothing else -- the colour is exactly
invariant, because `canonical.ply` keeps only the view-independent SH DC term. Which axis
is up:

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
