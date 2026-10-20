# Handoff — continuing this prototype from a less restricted environment

You are picking up a capture pipeline that is built but has never been deployed, largely
because the environment it was written in could not reach a single cloud provider. If you
are reading this from a machine with ordinary network access, **most of what is unverified
here becomes verifiable in an afternoon**, and that is the highest-value thing you can do
first.

Written 2026-09-22 at commit `8fabef7` on `claude/funny-carson-937ydv`, and revised the
same day from an environment that **does** have ordinary network access. Three of the
questions below are now answered; §3 says which, and what answering them cost and found.
What is still missing is not network — it is **accounts**.

## 1. Read these first, in this order

1. `docs/ARCHITECTURE.md` — what the system is.
2. `tools/pipeline/README.md` — the recipe/stage/runner model, which is the core abstraction.
3. `docs/DEPLOYMENT.md` — the topology, the handover, and its own `§ What is unverified`.
4. `docs/DECISIONS/0007-fly-for-the-api-cloudflare-for-the-edge.md` — why Fly and not
   Cloudflare for compute. It contains a correction worth reading as a model for how
   claims in this repo are expected to be handled.
5. `git log --oneline origin/main..HEAD` — 94 commits, each one step, each message
   explaining the decision rather than the diff.

## 2. Where the project stands

Fourteen of nineteen planned steps are done. 905 tests, CI green across six jobs. Nothing
is merged to `main` and nothing is deployed.

**Works end to end, no terminal step:** drop a Gaussian splat file into the console or send
it from a phone via QR code → uploads direct to object storage over presigned multipart →
a worker claims it on a lease → seven stages → SPZ 3D-Tiles, measured boundary, thumbnail,
ground samples, manifest → a site on the globe, placed by its own measured ground. A second
console at `/admin.html` shows every capture, run and artifact and reconciles storage
against the database both ways.

**Real but incomplete:** raw video produces frames and camera poses (actual COLMAP, 40/40
registered, 0.059° median rotation error) and EXIF GPS georeferences to 8.8 mm on the test
orbit. Then it stops, because training has never run.

**The five remaining steps:**

| Step            | What it delivers                                               | Blocked by                 |
| --------------- | -------------------------------------------------------------- | -------------------------- |
| C3              | `docs/PIPELINE.md` and the remaining ADRs                      | nothing                    |
| B5              | 4D: measured motion played back, with a mandatory null control | nothing (C2 unblocked it)  |
| B3              | Crisp splats from captures that moved                          | a GPU — i.e. a Modal token |
| the deploy      | The first real URL                                             | accounts                   |
| B3's validation | `none` vs `robust` vs `imc` on one windy capture               | a GPU _and_ a real capture |
| the remote half | written: `tools/pipeline/remote.py` + `infra/modal/app.py`     | done, except a real run    |

## 2a. Capture → splat, as of 2026-09-23

A phone or desktop video should become a correctly placed splat on the globe. Two things
changed:

- **Lane 1 no longer lands uploads on their side.** `normalize` converts the file's up axis
  into east/north/up by rotating the positions and each gaussian's quaternion. The colour
  needs no rotation: only the SH DC term is kept, and it is rotation-invariant. It then
  recentres the origin on the splat's footprint and base. Each format's default up axis is
  cited in `tools/pipeline/README.md § The up axis`. A capture's `upAxis` / `headingDeg`
  metadata overrides it. There is deliberately no `auto` mode: a plane normal has a sign
  nothing in a splat resolves.
- **Lane 2 is complete up to the GPU.** The GPU half is built and checked, and has never
  run:
  - frames from an iPhone-shaped HEVC `.mov` come out upright;
  - COLMAP poses run on the worker's CPU;
  - `train` goes to Modal;
  - the georeference falls back to the capture's own coordinate when the video has no
    location, and refuses rather than placing it at (0, 0);
  - a new `place` stage applies the EXIF similarity, or levels by camera-up.

  The state of each piece is in the table at the top of `tools/pipeline/README.md`.

**What to run, in order:**

1. **Push the branch.** `.github/workflows/modal.yml` runs on a push to
   `living-models` that touches `infra/modal/**`, so this push triggers it. It
   creates the Modal secret `twin-object-storage` from the R2 private-bucket pair, deploys
   `twin-pipeline` (the first image build takes 15–30 min), and trains 500 steps on an L4,
   for about $0.07–0.20. Its job summary carries the gaussian count, PSNR and billed
   seconds. Once the file is on `main`, the Actions tab dispatches it.
2. **Deploy the worker image** (`deploy.yml` → `api`, or `fly deploy --remote-only`). It
   carries COLMAP 3.9.1, `WORKER_RUNNER=cloud` and `WORKER_CLOUD_PROVIDERS=modal`, and it
   keeps uid 1000 so the existing volume stays writable.
3. **Resize the worker** before a real Lane 2 capture: see `docs/DEPLOYMENT.md § GPU
training — Modal`.
4. **Upload a short orbit video and process it as `photo-reconstruct`.** This is the
   first end-to-end Lane 2 run. Nothing has done it yet.

## 3. What the network unblocked, and what it did not

Re-probed 2026-09-22 from an environment with ordinary egress. The result is worth stating
precisely, because "we have the internet now" turned out to be a smaller change than it
sounds: **everything that needed a document is done; everything that needs an account is
not.**

| Was blocked                                                                       | Now                                                        | What that settled                                                                                                                             |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `modal.com`, `api.modal.com`                                                      | reachable                                                  | `ModalAdapter` read against `modal==1.5.5`. **Five defects, one fatal** — see below. Since run for real on an L4: **verified**.               |
| `developers.cloudflare.com`                                                       | reachable                                                  | The R2 public-bucket claim confirmed verbatim. The bucket split is built (§5a).                                                               |
| `runpod.io/pricing`, `vast.ai/pricing`                                            | reachable                                                  | The GPU rates are surveyed (§5b). Vast has no list price — its page says so.                                                                  |
| `console.neon.tech/api/v2`, `api.fly.io`, `registry.fly.io`, `api.cloudflare.com` | reachable (401 unauthenticated, which is the right answer) | Nothing. `provision.yml` still needs credentials to run, and has still never run.                                                             |
| `ghcr.io`                                                                         | reachable, **but there is no Docker daemon here**          | `docker build` still cannot run locally. CI's `image` job builds and runs the real image nightly and on changes to it, so this stays covered. |

Note for whoever re-probes: `api.neon.tech` does not resolve, but `console.neon.tech` does
— and the latter is the host `provision.yml` actually calls, so that table row was always
pointing at the wrong name.

**What reading the Modal SDK found**, since "we looked and it was fine" would have been
the least useful possible outcome:

- `poll()` caught the builtin `TimeoutError`. A zero-timeout poll raises
  `modal.exception.TimeoutError`, which inherits from `modal.Error` and **not** from the
  builtin — so every healthy stage was dead-lettered on its first poll. Fatal.
- None of the three guessed preemption markers exist. Preemption arrives as
  `InternalFailure`.
- `FunctionTimeoutError` and `OutputExpiredError` both subclass Modal's `TimeoutError`, so
  the obvious `isinstance` fix would have read a timed-out stage as running forever.
- `gpu="A10G"` is AWS's instance name; Modal's tier is `A10`. Nothing local catches this —
  a Modal `App` with a nonsense GPU name builds fine and is rejected only server-side.
- `cancel()` defaults to `terminate_containers=False`, which cancels the input and leaves
  the container billing.

**The GPU path is proven.** On 2026-09-23 `.github/workflows/modal.yml` built the training
image on Modal, then trained 500 steps on an L4 (68 s billed, $0.015), read PSNR from
gsplat's own stats, and placed and packaged the result. The `train` stage, the adapter
and the image moved to verified; the first real runs found two bugs no test here could
(see the pipeline README, "The remote half"). What remains unproven is a full-length
run on a real capture, and with it the real cost of one.

The remote half now exists. `tools/pipeline/remote.py` is what a container does — fetch,
run, sync the checkpoint on an interval, upload — and it takes a `Transfer`, so
`tests/test_remote.py` drives all of it here. `infra/modal/app.py` is the Modal wrapper:
an image, a GPU, a secret and one call, deploying a `run_stage_<tier>` per tier. CI builds
that App on every run, which caught two deploy-time path bugs and cannot catch a wrong
`gpu=` string. Since 2026-09-23 the image is filled: CUDA 12.4, torch 2.4.1+cu124, and the gsplat 1.5.3
wheel pinned by sha256. Every pin was resolved rather than recalled. It deploys as
`twin-pipeline` through `.github/workflows/modal.yml`, which then runs a small real
training run. See §2a.

## 4. Setting up, now that the network allows it

Cloudflare, Neon and Fly all ship MCP servers with `claude-code` as a supported target.
Doing the first provisioning run interactively over OAuth is **better than the CI path in
this repository**, because failures are visible and fixable in the moment and nothing has
ever run:

```bash
# Cloudflare — inside Claude Code
/plugin marketplace add cloudflare/skills
/plugin install cloudflare@cloudflare

# Neon — interactive, writes the agent config for you
npx neon@latest init

# Fly — note this CREATES A BILLABLE MACHINE, not just config
fly mcp launch
```

Neon's old `/sse` endpoint is deprecated and stops working on or after 2026-10-01; use
`https://mcp.neon.tech/mcp`. The command above is the current one — this file first said
`neon mcp`, which has been replaced.

**There is a cheaper route that needs no interactive session at all.** An agent with the
GitHub tools can dispatch a workflow and read its logs, which is how the branch's CI has
been driven. So tokens can live as **repository secrets** and never enter an agent's
environment: put them there, and the first provisioning run and the first GPU run are both
a `workflow_dispatch` away. The names the workflows already read are in
`docs/DEPLOYMENT.md`; Modal needs two more, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`,
from `modal token new`.

Either way: `deploy.yml` with GitHub secrets is the ongoing path, because unattended
deploys need tokens regardless, and `provision.yml` is the rebuild-from-scratch path
rather than the primary one.

## 5. Two decisions, both now settled

**(a) The bucket split — settled: B, and built.** The facts below were re-checked
against Cloudflare's own documentation on 2026-09-22 and all hold. The application now
takes `OBJECT_STORAGE_PUBLIC_BUCKET`, publishes only the tileset and the thumbnail into
it, and refuses to start in production without it. What follows is kept as the reasoning,
not as an open question.

Enabling R2's public URL makes the **whole bucket** world-readable, and this API uses one
bucket: raw uploads land under `captures/` and pipeline outputs under `runs/`. Deploying as
designed publishes every scan anyone uploads. Keys carry UUIDs so they are not enumerable
through the bucket, but they are not secret, and a custom domain has the identical
property. Cloudflare also documents `r2.dev` as a rate-limited debug hostname rather than a
CDN, which makes it wrong for tiles regardless.

- **A — accept it.** Defensible for a solo prototype whose captures are trees and public
  landscapes. Still use a custom domain rather than `r2.dev`.
- **B — split the buckets.** Private for `captures/`, public for published tiles. The
  correct architecture. Needs an application change: a second storage setting and a publish
  path, since `object_storage_bucket` is currently one value used for both. The Cloudflare
  MCP server makes the infrastructure half of this cheap.

**(b) Cost display — settled by surveying, not by estimating.** The table carried only
the four A100 rates A0 could measure, so the L4 the shipped `train` stage requests had no
price. Rather than add recalled numbers tagged unverified, the published price lists were
read: Modal's L4 at $0.7992 an hour and A10 at $1.1016, RunPod's L4 at $0.49 Secure and
$0.44 Community. `Rate.source` now distinguishes a measured figure from a list price read
on a named date.

Three things came out of it that outlast the numbers:

- **A0's survey checks out.** RunPod publishes exactly $1.59 and $1.19 an A100 hour;
  Modal's 80 GB A100 lists at $2.4984 against A0's $2.50. A survey nobody can re-run now
  has an independent source behind it.
- **Modal's `a100` is the 80 GB part**, and `gpu="A100"` alone selects the 40 GB one,
  which is priced 19% lower. `GPU_NAMES` maps the tier to `A100-80GB` explicitly, so the
  price and the hardware cannot disagree.
- **Vast stays unpriced on purpose.** Its page says prices are set by the market and not
  by Vast, so there is no list price to read. That is the reason A0's note about paying
  20-40% over sticker is in `providers.py`, and it is now the table's worked example of a
  tier that should have no number.

## 6. How this project expects to be worked on

These are not style preferences; they are why the defect list below exists.

- **One step per commit**, with the step name, and a message that explains the decision
  rather than restating the diff.
- **Run the gate yourself before believing a report.** The full gate is in
  `docs/DEPLOYMENT.md` and the sprint plan; the short version is `pnpm format:check &&
pnpm lint && pnpm typecheck && pnpm test`, `pnpm e2e` if `apps/web` changed, plus
  `uv run ruff check . && uv run mypy . && uv run pytest` in each of `apps/api`,
  `tools/pipeline` and `tools/captures`.
- **Dispatch CI per step.** Local green has been wrong three times: a `boto3` signing
  divergence, a Python-version mismatch between two machines, and a test asserting that
  floating-point arithmetic is exact (it held at exactly 0.0 on one machine and 1.2e-6 on
  another).
- **Never calibrate a tolerance on one machine's arithmetic.** Set bounds where a real
  regression shows, and record what you measured.
- **When something cannot be verified, leave it stubbed and say why.** `glomap`, `arkit`
  and `opensplat` are stubs with reasons in the code for exactly this reason. A plausible,
  unexecuted implementation is the failure mode this project exists to avoid — its whole
  premise is a world model that states what it does not know.

**Trip-wires that will bite you:**

- `data/tiles/synthetic-tree` is byte-identity gated (`git diff --exit-code`). Regenerate
  and confirm if you touch `synthetic_tree.py`.
- Splat tile transforms are translation-only and the root never moves; position checksums
  are over site-frame positions (`splat_tiles.tile_offsets`, `site_positions`). A new reader
  of tile positions composes the offsets: `data/tiles/synthetic-far` is the fixture to test
  it on (ADR 0011).
- LOD and SH fixtures are regenerated under `OPENBLAS_CORETYPE=Sandybridge`, the yard under
  Haswell, the yard's sidecars under Sandybridge; commands in
  docs/DECISIONS/0012-licence-clean-room.md. Every one also runs with numpy's AVX-512 loops off
  (`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`, set by tools/captures'
  conftest): their float64 `exp`/`log` differ in the last bit from the AVX2 loops the fixtures
  were written with, and CI's AVX-512 runners wrote another yard.
- `scripts/licence_scan.py` fails on a stale allowlist entry. A change that edits an
  allowlisted line (`gaussians.py`, `splat_io.py`, `harmonics.py`, `experiments/benchmark.py`,
  the seed's `WITHDRAWN_SITES`, `worlds/core/catalog.ts`, the research JSON) updates
  `scripts/licence_allowlist.json` in the same commit.
- `packages/contracts/openapi.json` is committed; any schema change needs
  `export_openapi` + `pnpm contracts:generate`.
- `pnpm format:check` runs **before** lint in CI and covers YAML and Markdown.
- An e2e test asserts an **exact** tool-rail button count.
- The repo's `.env` leaks into `apps/api` settings tests. This has caused three separate
  false results; override explicitly rather than trusting ambient config.
- Two `apps/api` pytest sessions at once deadlock on the shared test database.
- Superseded revisions are rows: `SiteRead.assets` hides them, `GET /assets` does not. A re-run
  never rewrites an asset's URL; it adds a revision (ADR 0013).
- A run's tiles must be placed about its site frame's origin exactly, or registration refuses it
  (`FrameMismatch`) and the job dead-letters. The frame is decided when the job starts.
- In a site frame, `normalize` and `georeference` must be told the same located point
  (`worker/params.located_point`) or `manual_placement` refuses the run; and anything turned from
  frame metres into degrees uses the exact ENU inverse, never the flat 111,320 m/° (metres out at
  2 km).
- A run's `site_context` (its site's reference epoch, check targets and the capture's clicks,
  `worker/site_context.py`) is seeded once and frozen: a resumed or retried run keeps the
  reference it started with, and a target edited meanwhile reaches only the next run. The
  reference is the best-verdict, oldest published epoch whose run kept frames, poses,
  `placement.json` and `georef.json`; a re-run of an epoch only takes one that joined before it.
- Every quality gate ships **off** (D4). One turned on rejects the capture on its first
  attempt (retry kind `quality-gate`, never retried) and keeps it as a `rejected` epoch, with
  the gate in `alignment`; a later run that passes publishes the same epoch.
- `uv run` in `tools/pipeline` without `--no-sync` removes pycolmap (installed by hand, as CI
  does); `apps/api/tests/test_registration_live.py` runs only with `COLMAP_GLOBAL_PYTHON` set to
  an interpreter that has pycolmap 4.2, and skips in CI.
- `georef_method` and `scale_source` keep their Postgres enum order equal to the Python members'
  (`test_0010_*`, `test_0033_*`): a new value goes in where the enum declares it, `BEFORE` the
  "nothing" member, and its downgrade rebuilds the type.
- The confidence grid is opt-in and must stay so: `splat_tiles.convert(confidence=None)` writes
  every fixture's bytes; the grid's descriptor is appended to the root extras **after**
  `collision`, and `confidenceMin` goes right after `gaussians` (ADR 0016). Nothing confidence
  goes in `quality.json` (`test_chunked_equivalence` compares its keys).
- The confidence grid is built from `quality`'s `support/` (every trained gaussian), never from
  `gated.ply` or `canonical.ply` — under `bar: strict` those hold keep only.
- `data/tiles/synthetic-yard-epochs` is regenerated by `tools/captures/synthetic_cell_grids.py`
  and a test compares every byte; it is the fixture the API and web cell-grid decoders are held to.
- `data/tiles/synthetic-yard-epochs/fill-targets.json` is the API's own fill-targets answer over
  those grids (ids replaced by placeholders): apps/api `test_fill_targets_api.py` rewrites it with
  `HEXAPOD_WRITE_FILL_TARGETS=1`, and tools/captures' fill gate (`rescan_gate.py`) is held to it.
- An inferred fill from outside is gated (ADR 0016 § Fill): `teacher_fill.py fill` needs
  `--still-weak`, fill.yml needs `fill_targets_url` for every `fill:` job, and publish-fill.yml
  refuses a layer without `evidence.rescanGate`. Never add an override; never put the
  publish-fill commit token in a commit message (it starts a GPU run).
- A new contract route that documents its own error responses names `Problem` as their model
  (`responses={409: {"model": Problem, ...}}`): one bodiless error response turns an untyped
  `vi.spyOn(api, "POST")` into `undefined` data and breaks every web test that mocks POST.
- An epoch's confidence lives in `epoch_maps` per **asset**, and an attach moves the row's URL
  with the generation it cuts (`epoch_maps.repoint`); a `confidence` sidecar is not attachable.
  `GET /sites/{id}/epochs` loads the maps eagerly (`test_query_counts.py`).
- A new `photo-reconstruct` stage bumps the recipe version, `SHIPPED_RECIPE_VERSIONS`,
  `apps/api/tests/test_quality_api.py` `STAGES`, `experiments/run_variants.py` `STAGE_ORDER` and
  the upload page's `STAGE_TEXT` together.
- Postgres may need `service postgresql start` before `apps/api` tests — and a fresh
  container may have no database at all. What worked here: `apt-get update && apt-get
install -y postgresql-16-postgis-3`, then create a `twin` role with password `twin` and
  the `twin`/`twin_test` databases, then `CREATE EXTENSION postgis` in each. The compose
  file and CI both use `postgis/postgis:16-3.5`, which is simpler where a Docker daemon
  exists; this environment has the `docker` CLI and no daemon.

## 7. The three-state honesty model

The pipeline's seventeen stage implementations are tracked as **verified** (runs, and is
exercised on a machine that is not the development one), **unproven** (real code that has
never executed against the real thing), and **stub** (raises, and names the step it lands
in). Thirteen, zero and four respectively since `gsplat` ran on an L4 in the
`modal.yml` smoke; `ModalAdapter` and the Modal training image, outside the registry,
moved with it. A full-length run on a real capture is still unproven.

Keep that distinction. A green test suite hides it, and it is the single most useful thing
this sprint established about its own work.
