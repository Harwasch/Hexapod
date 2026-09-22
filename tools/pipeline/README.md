# tools/pipeline — the capture pipeline's spine

Turns an uploaded capture into the artifact set the console needs, by running an ordered
list of stages. This project is the executor, the stage registry, the artifact and workdir
contract, and the runner seam. It is **not** the worker — that is `apps/api/app/worker/`,
which imports this project as a library — and at this step most stages are stubs with
honest contracts: they declare exactly what they read and write, and raise rather than
pretend.

It has no dependency on `apps/api`: no models, no database connection, no HTTP. The
dependency runs one way only. A stage that wants something registered writes a file saying
so (`registration.json`) and the worker, which has the credentials and the session, does
it.

```bash
uv run python run_recipe.py splat-ingest --workdir /tmp/run-1 --seed upload=./capture.ply
uv run python run_recipe.py photo-reconstruct --workdir /tmp/run-2 --plan-only
```

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
file's own location. Two things keep it honest rather than hidden:

- `mypy_path = ["../captures"]`, so mypy resolves `splat_tiles` to the real file and
  type-checks every call into it (with `follow_imports = "silent"`, since that project does
  not run mypy and its errors are not this project's to fix);
- `tests/test_captures_bridge.py` runs the real `convert()` on a generated PLY and asserts
  the files it writes are exactly the `required_members` the `splat` artifact declares — the
  same declaration StubRunner fabricates from. A8 swaps the stub body for
  `splat_tiles_convert(...)` and nothing else changes.

## Recipes shipped

- **`splat-ingest`** — Lane 1, no GPU: `normalize → georeference → package → register`.
  A8 makes each stage real and inserts `thumbnail`, `ground_samples` and `manifest` before
  `register`, which is a recipe edit and three decorators.
- **`photo-reconstruct`** — Lane 2: `normalize → pose → mask → train → compensate →
georeference → package → register`. Only `train` declares `gpu:`; `compensate` gains one
  when its impl becomes `imc` (B3), since asking for an L4 to run `none` would be billing a
  GPU to do nothing.

## Verify

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -q
```
