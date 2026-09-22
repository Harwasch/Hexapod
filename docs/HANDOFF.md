# Handoff — continuing this prototype from a less restricted environment

You are picking up a capture pipeline that is built but has never been deployed, largely
because the environment it was written in could not reach a single cloud provider. If you
are reading this from a machine with ordinary network access, **most of what is unverified
here becomes verifiable in an afternoon**, and that is the highest-value thing you can do
first.

Written 2026-09-22, at commit `8fabef7` on `claude/funny-carson-937ydv`.

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
| B3              | Crisp splats from captures that moved                          | a GPU                      |
| the deploy      | The first real URL                                             | accounts                   |
| B3's validation | `none` vs `robust` vs `imc` on one windy capture               | a GPU _and_ a real capture |

## 3. What was impossible here — verify these first

Every one of these was measured, not assumed. If your environment differs, **re-running
them is the fastest way to turn this project's largest unknowns into facts.**

| Blocked from the authoring environment                            | What it left unverified                                                                                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `api.cloudflare.com`, `api.fly.io`, `api.neon.tech`               | All of `provision.yml`. Never run, not once, not with dummy credentials.                                                                   |
| `developers.cloudflare.com`, `*.mcp.cloudflare.com`, `github.com` | The Cloudflare agent-setup plugin could not be installed.                                                                                  |
| `modal.com`, `api.modal.com`, `rest.runpod.io`, `console.vast.ai` | `train` has never trained. `ModalAdapter` has never executed a line, and its API calls were written _from memory_, not documentation.      |
| `ghcr.io` blob CDN (403)                                          | `docker build` never ran locally — though CI's `image` job does build and run the real image on every run, so this one is already covered. |

**Start with the GPU.** One Modal account and token turns three unproven things into
verified ones at once: the `train` stage, the Modal adapter, and the ability to start B3.
Expect the adapter to be wrong in places — its docstrings carry inline `UNVERIFIED:`
markers naming exactly what to check, and the remote half (the deployed Modal function
that fetches inputs, runs `run_stage` and syncs `checkpoint/`) **is not in this repository
at all** and has to be written.

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
neon mcp

# Fly — note this CREATES A BILLABLE MACHINE, not just config
fly mcp launch
```

Neon's old `/sse` endpoint is deprecated and stops working on or after 2026-10-01; use
`https://mcp.neon.tech/mcp`.

Then: first run interactively via MCP, ongoing deploys via `deploy.yml` with GitHub
secrets (unattended deploys need tokens regardless), and `provision.yml` as the
rebuild-from-scratch path rather than the primary one.

## 5. Two decisions left open, deliberately

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

**(b) Cost display.** The provider price table carries only the four A100 rates that were
actually surveyed, each tagged with its source. The shipped `train` stage requests an L4,
which has no surveyed rate, so a real run records billed seconds and leaves `cost_usd`
null rather than printing a plausible number someone would believe. If you would rather
see an estimate, add recalled list prices tagged `source: recalled, unverified` — the
tagging is the point.

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
- `packages/contracts/openapi.json` is committed; any schema change needs
  `export_openapi` + `pnpm contracts:generate`.
- `pnpm format:check` runs **before** lint in CI and covers YAML and Markdown.
- An e2e test asserts an **exact** tool-rail button count.
- The repo's `.env` leaks into `apps/api` settings tests. This has caused three separate
  false results; override explicitly rather than trusting ambient config.
- Two `apps/api` pytest sessions at once deadlock on the shared test database.
- Postgres may need `service postgresql start` before `apps/api` tests.

## 7. The three-state honesty model

The pipeline's sixteen stage implementations are tracked as **verified** (runs, and is
exercised on a machine that is not the development one), **unproven** (real code that has
never executed against the real thing), and **stub** (raises, and names the step it lands
in). Eleven, one and four respectively; `gsplat` is the unproven one, and `ModalAdapter`
is a twelfth outside the stage registry.

Keep that distinction. A green test suite hides it, and it is the single most useful thing
this sprint established about its own work.
