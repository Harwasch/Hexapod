# tools/pipeline — the capture pipeline's spine

Turns an uploaded capture into the artifact set the console needs, by running an ordered
list of stages. This project is the executor, the stage registry, the artifact and workdir
contract, and the runner seam. It is **not** the worker — that is `apps/api/app/worker/`,
which imports this project as a library.

**Lane 1 is real.** `splat-ingest` runs end to end on a CPU: a `.ply` or `.spz` in,
`canonical.ply`, a `splat/` tileset, a thumbnail, ground samples, a manifest and a
registration out. Lane 2's GPU stages are still stubs with honest contracts — they declare
exactly what they read and write, and raise with the step that lands them rather than
pretending.

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

Both lanes converge on `canonical.ply` — Lane 1 normalises an already-reconstructed splat
into it, Lane 2's trainer writes it — so one `package` implementation serves both.

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

B1 runs on preemptible GPUs, where being killed is ordinary operation rather than an error,
so the contract accounts for it now even though nothing checkpoints yet:

- `out/` is **cleared** at the start of every attempt, so a half-written output from a
  killed attempt can never be mistaken for a produced artifact;
- `checkpoint/` is **kept** across attempts. It is the one directory the executor does not
  clear. A stage sees `ctx.has_checkpoint` and `ctx.checkpoint_dir`;
- `ctx.checkpoint_key` is the object-storage key B1 syncs that directory to
  (`runs/<run id>/<stage id>/checkpoint`), and it lands in the StepResult when the stage
  left anything behind, so `job_step.checkpoint_key` has something to record.

## Runners

```
Runner.run(stage, workdir) -> StepResult
├── LocalRunner   calls the registered implementation in this process; a stage that needs
│                 an external tool shells out through StageContext.run()
├── StubRunner    CI: fabricates each declared artifact deterministically
└── CloudRunner   B1. RunnerSet.gpu is the hole it slots into.
```

`RunnerSet(cpu=..., gpu=...)` routes on one fact: whether the stage declares `gpu:`.
`RunnerSet.stubbed()` puts StubRunner in both slots; `RunnerSet.local()` leaves `gpu` empty,
so a GPU stage fails with a message that names the tier it wanted.

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
georeference → package → thumbnail → ground_samples → manifest → register`. Only `train`
  declares `gpu:`; `compensate` gains one when its impl becomes `imc` (B3), since asking
  for an L4 to run `none` would be billing a GPU to do nothing. Both lanes end in the same
  artifact set, so the console cannot tell which one made a site except by reading its
  manifest.

`thumbnail`, `ground_samples` and `manifest` were added in A8 as **a recipe edit and three
decorators**: `stages.py` gained three `@stage_impl`s and three `ArtifactDecl`s, and the
recipes gained three entries. `executor.py`, `runners.py`, `plan.py` and `workdir.py` are
untouched by them, and `StubRunner` fabricates the new artifacts with no edit of its own.
`tests/test_lane1.py` asserts that rather than leaving it as a claim.

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

### `ground_samples.json`

The capture's own ground height per grid cell — half of the height-offset subtraction a
person does by hand today. Nothing consumes it yet: B4 is where the console samples terrain
at the same longitude and latitude and takes the median difference.

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

`{lon, lat, z, n}` is the shape `tools/captures/ground_samples.py` already prints, so B4
reads one shape rather than two. `z` is the capture's own up axis in metres relative to the
placed origin, so a sample's ellipsoid height is `origin.height + z`. The cells are ranked
by how many gaussians they hold and tie-broken by position — deterministic, unlike the
sibling script, which picks cells with a seeded RNG.

`method` is spelled out because a low percentile of a _tree_ is canopy, not ground: this is
the capture's own low surface per cell, and it is the ground only where the capture has one.

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
