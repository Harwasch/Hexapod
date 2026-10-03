# Deployment

**Production exists, deployed by the operator** (as of 2026-09-23): Pages, the Fly app
`twin-api` with its `worker` process and a 20 GB volume, Neon, and the two R2 buckets. The
Modal pair is a repository secret that `provision.yml` forwards to Fly. That is the
operator's report; nothing in this repository has observed it, and this document was
first written before any of it existed. What it describes is what the repository
_codifies_ — every row of the table below has a file behind it — plus, at the end, a
[handover](#handover): create accounts, mint tokens, paste secrets, dispatch. Everything
after a token exists is `.github/workflows/provision.yml`. **The GPU half of Lane 2 has
never run anywhere**; [GPU training — Modal](#gpu-training--modal) says what proves it and
what that costs. Read [what is unverified](#what-is-unverified) before trusting any
provider-specific claim in here.

| Layer                           | Where                        | Codified in                                            |
| ------------------------------- | ---------------------------- | ------------------------------------------------------ |
| Web bundle                      | Cloudflare Pages             | `wrangler.toml`, `infra/pages/_headers`                |
| API (`app` process)             | Fly.io                       | `fly.toml`                                             |
| Worker (`worker` process)       | Fly.io, same image           | `fly.toml` (`[processes]`, `[[mounts]]`)               |
| Captures, tiles, `catalog.json` | Cloudflare R2                | `infra/cors/production.json`, `infra/cors/apply-r2.sh` |
| Lane 2 training (`train`)       | Modal, one GPU per stage     | `infra/modal/app.py`, `.github/workflows/modal.yml`    |
| Database                        | Neon (Postgres 16 + PostGIS) | `.github/workflows/provision.yml`                      |
| Provisioning                    | GitHub Actions, on demand    | `.github/workflows/provision.yml`                      |
| Deploys                         | GitHub Actions, on demand    | `.github/workflows/deploy.yml`                         |

Why Fly for the compute and Cloudflare for the edge, when consolidating on one provider was
on the table: [ADR 0007](DECISIONS/0007-fly-for-the-api-cloudflare-for-the-edge.md). The
short version is the worker: it is a poll loop that supervises hours-long runs and renews a
lease every two seconds, and a sleep-on-idle runtime has nowhere to put that. (It does stop
when there is nothing to do — on its own terms, never mid-run; see
[waking the worker](#waking-the-worker-and-letting-neon-sleep).)

The frontend never serves large 3D assets. The browser streams them from Cesium ion or from
R2, and uploads go browser → R2 directly over presigned URLs. The API carries metadata.

## One image, two commands

`infra/api.Dockerfile` builds a single image. Its `CMD` runs `alembic upgrade head` and then
uvicorn; `python -m app.worker` is the same image run as the worker. On Fly the two
`[processes]` entries replace `CMD`, and migrations move to `[deploy] release_command` so
that they run once per deploy rather than once per machine boot.

```bash
docker build -f infra/api.Dockerfile -t twin-api .
docker run -p 8000:8000 --env-file .env twin-api              # the API
docker run --env-file .env twin-api python -m app.worker      # the worker
docker run --env-file .env twin-api python -m app.seed        # seed the catalogue, once
```

CI's `image` job builds this image on every run and asserts against the running container
that the API is healthy, that `/api/v1/recipes` lists both recipes, that the pipeline's
stages import in the image's venv, that `python -m app.worker --once` exits cleanly, that
every command `fly.toml` names exists on `PATH` in the image, and that a production
configuration both starts when it is complete and refuses to start when it is not.

## Web — Cloudflare Pages

```bash
pnpm install --frozen-lockfile
pnpm --filter @twin/web build            # -> apps/web/dist
cp infra/pages/_headers apps/web/dist/   # Pages reads this from the deployed root
pnpm dlx wrangler pages deploy           # project and directory come from wrangler.toml
```

`.github/workflows/deploy.yml` runs exactly those four commands.

It is a **Direct Upload** Pages project, not a Git-integrated one: the build happens in the
workflow, where the `VITE_*` values live as repository secrets and variables. There is
therefore nothing to connect and no build command to configure, which is why
`provision.yml` can create the project with one API call.

The bundle is **three HTML entry points**, not a single-page app — `index.html` (the globe),
`upload.html` (what a phone opens after scanning the handoff QR: no CesiumJS), and
`admin.html` (the data console). So there is deliberately no `_redirects` file and no
SPA fallback: Pages serves each document, and a catch-all rewrite to `index.html` would turn
every typo into a silently-served globe instead of a 404. The handoff URL the API mints is
`<PUBLIC_WEB_BASE>/upload.html#<token>` — a literal path, which must keep working.

Only `index.html` may load CesiumJS. `node apps/web/scripts/check-bundle.mjs [dist]` reads a
build the way a browser would (entry scripts, modulepreloads, stylesheets, then every static
import) and exits non-zero if `admin.html`, `upload.html` or `view.html` reaches a Cesium
chunk; it also prints what each page loads, raw and gzipped. e2e cannot see this — the dev
server does no chunking — and it is exactly what broke once: a shared `tslib` bundled inside
`cesium-*.js` made the console modulepreload the whole engine (`vite.config.ts` says how).

Build-time environment, all public once the bundle ships:

| Variable                       | Production value                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | your token, scopes `assets:read` + `geocode`, Allowed URLs = your Pages origin      |
| `VITE_API_BASE_URL`            | `https://twin-api.fly.dev` (absolute; the dev proxy is dev-only)                    |
| `VITE_ENABLE_PHOTOREALISTIC`   | on by default; `false` unless your token has Google access and you accept the terms |
| `VITE_ENABLE_DEV_TOOLS`        | leave unset/false                                                                   |
| `VITE_DEFAULT_*_ASSET_ID`      | optional                                                                            |
| `VITE_OFFLINE_CATALOG_URL`     | `https://tiles.example.com/catalog.json` — the offline catalog, see below           |

`infra/pages/_headers` sets caching and a few conservative security headers. Two notes worth
keeping: there is no `Content-Security-Policy`, because CesiumJS spawns workers and
instantiates WebAssembly and a policy written without a browser to test it against breaks
the globe on first load; and `/cesium/*` is `immutable` only because `vite.config.ts` copies
Cesium's static directories (which are not content-hashed) under a path named for the
installed version, `/cesium/<version>/`, so upgrading the dependency changes the URLs rather
than the contents behind them. A patch to the `cesium` package's `Build/` would have to change
that name too; the engine patch does not touch it. `/fonts/*` (self-hosted, licences beside
them) is immutable on the same terms: a new cut of a font gets a new file name.

### The `/r2/` tile proxy, and the one host it serves

`functions/r2/[[path]].js` is a Pages Function that serves the public bucket's `r2.dev`
URL from the web app's own origin, over HTTP/2-3 and with cache lifetimes (r2.dev gives
HTTP/1.1 and neither). `HEAD /r2/` answers `X-Tile-Proxy: <the pinned host>`, and the web
app (`apps/web/src/lib/tileProxy.ts`) sends a URL through it only when the URL is on that
host. Any other `r2.dev` bucket — a tileset added through Add data, a demo or seed bucket —
loads from its own URL, as it would with no proxy. (The probe used to answer `1`, and every
`pub-*.r2.dev` URL was rewritten to a function that 404s all but one host.)

Whatever it answers is served from the origin where the console keeps the write token in
`localStorage`, so it is locked to exactly what it is for:

- **One host.** `TILE_PROXY_HOST`, in `wrangler.toml`'s `[vars]`, names the public bucket's
  `pub-<32 hex>.r2.dev` hostname. It is committed empty; `deploy.yml` writes it in before
  `wrangler pages deploy`, from the `public_bucket_url` input `provision.yml` passes, else
  the `R2_PUBLIC_URL` variable, else the host of the offline catalog URL. Any other host
  is a 404 that is never fetched. (It used to fetch _any_ `pub-*.r2.dev` host — every public
  R2 bucket in the world.)
- **Our Content-Type, by extension.** `.json`, `.glb`, `.b3dm`, `.pnts`, `.bin`, `.emb`,
  `.f32`, `.u8`, `.ply`, `.spz`, `.webp`, `.jpg`/`.jpeg`, `.png` — what the bucket actually
  holds. Anything else, including `.html` and `.svg`, is a 404. The upstream's own
  Content-Type is never passed through.
- **`X-Content-Type-Options: nosniff` and `Content-Security-Policy: sandbox; default-src
'none'`** on every proxied response, so a mislabelled object opened as a page can run
  nothing.

**Unset is off, not open.** With no `TILE_PROXY_HOST` — a custom domain on the public
bucket, a dispatch that could not work the host out, a `wrangler pages deploy` run by hand —
the probe answers 404 without the header, every proxied path is a 404, and the web app
loads tiles from the bucket's own URL. A custom domain needs no proxy: Cloudflare already
serves it over HTTP/2 with caching, and the web app only ever routes `r2.dev` hosts here.
Tiles published under a _previous_ r2.dev host (an older public bucket) are not served
through the proxy: the web app loads them from that host directly, over HTTP/1.1 and
uncached, until they are republished into the current bucket.

## API and worker — Fly.io

```bash
fly apps create twin-api
fly volumes create twin_worker_data --size 50 --region iad --app twin-api
fly deploy --remote-only        # from the repository root: the build context is the repo
```

Those are the commands; `provision.yml` runs them for you, and the volume before the deploy
is not an ordering preference — the `worker` process group will not start without its mount.

The build context has to be the repository root — `infra/api.Dockerfile` copies
`tools/pipeline` and `tools/captures` as well as `apps/api`, and the worker cannot start
without them.

`fly.toml` declares:

- **two process groups from one image.** `app` runs uvicorn behind Fly's proxy and TLS;
  `worker` runs `python -m app.worker` with no service at all.
- **`auto_stop_machines = "suspend"`, `min_machines_running = 0`** on the `app` group
  only. A single-user API idles for hours. Suspend rather than stop: Fly snapshots the
  machine's memory (it allows this up to 2 GB; the API's is 1 GB), so the next request
  resumes a running uvicorn instead of booting Python and importing the app. The first
  request after a deploy is still a cold boot — a new image has no snapshot — and a socket
  held across a suspend (the database pool, an R2 or Anthropic keep-alive) may be found
  reset once on resume; `pool_pre_ping` (`app/db.py`) checks a pooled connection before
  using it. The worker must never be stopped by a proxy that sees no requests, and it is
  not in `http_service.processes` for that reason.
- **the worker stops itself when idle.** `WORKER_IDLE_EXIT_S=900`: with nothing running
  or queued for 15 minutes it exits 0 and its machine stays stopped; queueing a job starts
  it again. See [waking the worker](#waking-the-worker-and-letting-neon-sleep).
- **two worker slots**: `WORKER_CONCURRENCY=1` general slot and `WORKER_CPU_ONLY_SLOTS=1`,
  which claims only recipes with no `gpu:` stage, so a one-minute `splat-ingest` is not
  queued behind a two-hour training run and two training runs never share the machine.
- **a health check** on `GET /api/v1/health`.
- **`kill_signal = "SIGTERM"`, `kill_timeout = "30s"`.** The worker treats SIGTERM as a stop
  flag: on its next tick it stops the recipe process, clears the lease on the job it holds
  and exits, so the next worker can take that job immediately instead of waiting the lease
  out. 30 s is fourteen ticks of headroom. A deploy does not stop a GPU call in flight: the
  recipe process is told it is a shutdown (SIGUSR1), leaves the Modal call running and
  writes down its id, and the next worker re-attaches to it at the same attempt
  (`apps/api/app/worker/README.md`, "A GPU call outlives the process that made it"). A
  cancelled job's call is cancelled within seconds. (Both keys are at the _top_ of
  `fly.toml`, before any table header. TOML gives a bare key to whichever table precedes
  it, so written next to the health check — where they read most naturally — they
  silently become fields of that check and Fly never sees them. CI asserts they are
  top-level.)
- **`[[restart]]` for the worker, `on-failure`.** A worker that crashes is restarted (ten
  tries); one that exits 0 on purpose is left stopped.
- **a 20 GB volume** at `/data`, on the `worker` group only, with
  `WORKER_WORKDIR=/data/worker`. See [the workdir](#the-workers-workdir). It asked for
  50 GB until the first real provisioning run met Fly's "To create more than 20GB in
  volumes please add a payment method"; `fly volumes extend` raises it later without a
  redeploy.

### Configuration

`fly.toml`'s `[env]` carries only what is true of any account: `APP_ENV=production`,
`ALLOW_PRIVATE_URLS=false`, `API_PORT`, `OBJECT_STORAGE_REGION=auto`, `WORKER_WORKDIR`.
Everything account-specific — including things that are not strictly secret, like your own
domain names — goes through `fly secrets set`, so nothing about your deployment is
committed.

| Secret                                                    | Notes                                                                        |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `DATABASE_URL`                                            | `postgresql+psycopg://…` with PostGIS enabled                                |
| `API_WRITE_TOKEN`                                         | **required**: the shared token every write must carry                        |
| `API_HANDOFF_SECRET`                                      | signing key for phone-handoff tokens; set it explicitly once >1 process      |
| `API_CORS_ORIGINS`                                        | your Pages origin, exact scheme + host, comma-separated                      |
| `OBJECT_STORAGE_ENDPOINT_URL`                             | `https://<account-id>.r2.cloudflarestorage.com` (the S3 API domain)          |
| `OBJECT_STORAGE_BUCKET`                                   | the **private** bucket: uploads under `captures/`, run outputs under `runs/` |
| `OBJECT_STORAGE_PUBLIC_BUCKET`                            | **required in production**: the only bucket the world can read               |
| `OBJECT_STORAGE_ACCESS_KEY` / `OBJECT_STORAGE_SECRET_KEY` | the R2 API token's pair                                                      |
| `OBJECT_STORAGE_PUBLIC_URL`                               | the **public bucket's** base — its custom domain, not the S3 API domain      |
| `TILES_BASE_URL`                                          | optional: a CDN prefix in front of the published tiles                       |
| `PUBLIC_API_BASE`                                         | the API's own public origin, used when seeding absolute tileset URLs         |
| `PUBLIC_WEB_BASE`                                         | the web origin, for the handoff URL a QR code encodes                        |
| `CESIUM_ION_SERVER_TOKEN`                                 | optional; `assets:read` for job monitoring. Never a `VITE_` variable         |
| `ANTHROPIC_API_KEY`                                       | optional; without it the plan drafter is rule-based and says so              |
| `FLY_API_TOKEN`                                           | a token that can start this app's machines; see waking the worker, below     |
| `QUEUE_CHECK_URL`                                         | optional; a healthchecks.io-style check for jobs queued and never claimed    |
| `SENTRY_DSN`                                              | optional; error reporting, see [Observability](#observability)               |

### Production refuses to start when it cannot do its job

Three startup guards, all in `app/main.py`, all deliberate:

1. **No `API_WRITE_TOKEN`.** An unset token means writes are open, which is how a fresh
   checkout runs. `APP_ENV=production` plus an empty token raises at startup rather than
   serving an internet-facing API whose writes are open.
2. **Nowhere to serve capture tiles from.** Production with neither `TILES_BASE_URL` nor
   object storage configured used to log an error and carry on, seeding every capture
   against the `/api/v1/tiles` static mount — which production disables and the image has no
   `data/tiles` to serve in any case. The result was a catalogue of URLs that 404 in
   somebody's browser hours later, with one log line as the only trace. It now raises:

   > `APP_ENV=production` needs somewhere to serve capture tiles from, and has neither: set
   > `OBJECT_STORAGE_ENDPOINT_URL`, `OBJECT_STORAGE_BUCKET`, `OBJECT_STORAGE_ACCESS_KEY` and
   > `OBJECT_STORAGE_SECRET_KEY` (uploads need them regardless), or set `TILES_BASE_URL` to
   > the public prefix the tiles are published under. With neither, every capture would be
   > seeded against the `/api/v1/tiles` static mount, which production disables and this
   > image does not carry — so every tileset URL would 404.

   A bucket alone is enough; `TILES_BASE_URL` is for putting a CDN in front of it, not a
   second requirement.

3. **One bucket in both roles.** Production with storage configured and no
   `OBJECT_STORAGE_PUBLIC_BUCKET` raises. This is the guard with the widest blast radius
   and the quietest failure: a deployment that gets it wrong works perfectly. See
   **Two buckets, one key scheme** below for why.

On Fly a failed guard means the machine exits, the health check never passes and
`fly deploy` rolls back with the old version still serving. The failure is at deploy time,
in your terminal, which is the whole point. Note that `python -m app.seed` and
`python -m app.worker` do **not** build the app and so do not run these guards — but they
share the configuration, so an API that starts is an API whose configuration the seeder will
also find.

### The worker's workdir

`WORKER_WORKDIR` defaults to `var/worker`, which resolves to `/app/var/worker` _inside the
container_: on the image's own writable layer, thrown away on every restart, and competing
with the image's layers for the root disk while a 12 GB capture is in flight.

That matters because the workdir **outlives the run on purpose**: "retry from this stage"
reads the outputs of the stages that already succeeded out of it, and A6's `checkpoint/`
contract is worth nothing once the directory is gone. `fly.toml` therefore mounts a 20 GB
volume at `/data` on the worker group and sets `WORKER_WORKDIR=/data/worker` — roughly one
and a half 12 GB captures in flight, which is a starting size rather than a judgement about
how much room the worker needs. Grow it with `fly volumes extend`.

**If your host cannot give the worker a volume**, the deployment still works and nothing is
corrupted — a worker that dies mid-run leaves a lease that lapses, and A7's reclaim hands
the job to the next worker, which starts the recipe over. What you lose is the work: every
completed stage is recomputed, and a Lane 2 run that was two hours into `train` pays those
two hours again. You also need a root disk big enough for the largest capture plus the
image. Size the machine accordingly and treat restarts as expensive.

### Waking the worker, and letting Neon sleep

A worker that polls the queue every two seconds forever keeps two things awake that need
not be: its own machine, and Neon's compute, which scales to zero only after five minutes
with no queries. So the worker now stops when it has nothing to do, and the API starts it
when it queues something.

**The worker side** (`app/worker/loop.py`):

| Setting                       | `fly.toml` | Default | What it does                                                          |
| ----------------------------- | ---------- | ------- | --------------------------------------------------------------------- |
| `WORKER_IDLE_S`               | —          | 2       | how often an empty queue is polled at first                           |
| `WORKER_IDLE_BACKOFF_AFTER_S` | —          | 60      | then doubling once a period: 4, 8, 16 s a minute at a time…           |
| `WORKER_IDLE_MAX_S`           | —          | 30      | …up to this                                                           |
| `WORKER_IDLE_EXIT_S`          | 900        | 900     | exit 0 after this long with nothing running or claimed; 0 never exits |

The exit is only ever between jobs — a slot holding a job counts as busy, and the clock
restarts when a job ends — so nothing about leases or SIGTERM changes. An exit status of 0
leaves the machine **stopped**: `fly.toml`'s `[[restart]]` policy for the group is
`on-failure`, so Fly restarts a worker that crashed and leaves one that finished alone.
The volume and the workdir stay as they are. `.env.example` sets `WORKER_IDLE_EXIT_S=0`,
because a checkout has nothing that would start the worker again.

**The API side** (`app/services/worker_wake.py`). Every route that queues a job —
`POST /captures/{id}/process`, the phone's process and refine, `POST /jobs/{id}/retry` —
adds a background task that runs after the commit and after the response has been sent.
It lists the app's machines through Fly's Machines API
(`https://api.machines.dev/v1/apps/$FLY_APP_NAME/machines`; Fly sets `FLY_APP_NAME` on
every machine), and starts each one whose `config.metadata.fly_process_group` is `worker`
and whose state is `stopped` or `suspended`. Every request has a 5 s timeout, a failure is
a warning in the log and never an error to the caller, and with no `FLY_API_TOKEN` the
whole thing is a no-op — which is what development gets.

A **cancel** of a job a worker had claimed (`POST /jobs/{id}/cancel`, the phone's stop)
wakes the worker the same way, without the queue check's `/start`: a worker that was not
there to see the cancel — crashed, out of restarts, or stopped with `fly machine stop` —
may have left a GPU call running, and the worker cancels the calls of every job that is
over when it starts (`app/worker/reaper.py`), and every five minutes after.

**The race** is a job committed just as the worker decides to exit. The worker asks the
queue once more, under the lock every claim takes, immediately before exiting; a job that
lands after that question finds the machine still `started`, which a start request does not
change. So when the wake call finds a worker machine already up, it looks again 20 s later
and starts any that have stopped by then. A worker that took the job cannot be among them:
it would not exit for another 15 minutes.

**`FLY_API_TOKEN`** is a Fly **secret on the app** (`fly secrets set`), read by the `app`
machines — not the GitHub secret of the same name, which is org-scoped so that
provisioning can create the app. Mint the narrowest token that can start machines:

```bash
fly tokens create deploy -a twin-api     # app-scoped; can start (and deploy) this app's machines
fly secrets set -a twin-api FLY_API_TOKEN='FlyV1 fm2_…'
```

The value is used as Fly prints it (`FlyV1 …`); an older personal token without that prefix
is sent as a bearer token. A deploy token can do more than start machines — it can deploy
the app — and Fly's macaroon tokens can be attenuated further; a narrower token that still
reaches `/machines/{id}/start` has not been tried here. `provision.yml` does not set this
secret yet. Without it, a worker that has exited stays stopped until a person runs
`fly machine start` — so either set it or set `WORKER_IDLE_EXIT_S = "0"` in `fly.toml`.

**`QUEUE_CHECK_URL`** (optional) is a [healthchecks.io](https://healthchecks.io)-style check
URL. Queueing a job onto an idle worker pings `<url>/start`; the worker pings `<url>`
whenever it claims a job. Set the check's **period long (30 days)** — it hears nothing
while nobody queues anything — and its **grace longer than a cold start (10 minutes)**, and
a job that is queued and never claimed — an expired token, a machine that will not boot —
becomes an alert instead of a phone that says "queued" forever. No `/start` is sent for a
job queued while a worker is running another (a job in progress under a live lease): it
waits for that run, two hours of training and more, and a `/start` for it alerted after
the grace every time. The run in progress has `WORKER_HEARTBEAT_URL`, and the worker pings
this check when it claims the queued job afterwards.

**Neon** then sees no queries from an idle deployment and suspends its compute five
minutes after the worker's last poll. The first query after that waits for the compute to
start (Neon quotes a few hundred milliseconds); a connection the pool held across the
suspend is checked before use (`pool_pre_ping`) and replaced.

## GPU training — Modal

Lane 2 (`photo-reconstruct`: a phone or desktop video in, a placed splat out) splits
across two machines, and the split is one fact: **only a stage that declares `gpu:` leaves
the worker.** That is `train`. Frames, poses, georeference, placement and packaging all run
on the Fly worker, exactly as Lane 1 does.

| Where          | What runs                                                                        | Why there                                                                                    |
| -------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Fly `worker`   | `normalize` (ffmpeg), `pose` (COLMAP 3.9.1, CPU), `georeference`, `place`, tiles | CPU work; the worker already holds the upload, and COLMAP's CPU path is what was measured    |
| Modal, one GPU | `train` (gsplat 1.5.3 `simple_trainer.py`)                                       | needs CUDA; billed per second only while it runs, so an idle deployment costs nothing for it |

`fly.toml` sets `WORKER_RUNNER=cloud` and `WORKER_CLOUD_PROVIDERS=modal`. The worker then
calls `run_stage_<tier>` in the Modal app `twin-pipeline` (`WORKER_MODAL_APP`'s default),
with `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` from its Fly secrets. The GPU container moves
bytes through the **private** bucket under `runs/<run id>/`, using the Modal secret
`twin-object-storage`. Until the app is deployed, a Lane 2 run should fail at `train` on
the function lookup (not observed). Lane 1 is unaffected either way.

### Deploying it, and proving it

`.github/workflows/modal.yml`, in one job:

1. It checks that the secrets are present: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
   `CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and the variable
   `R2_BUCKET`. It reads nothing `provision.yml` does not already read.
2. It creates or updates the Modal secret `twin-object-storage` from that R2 pair, via a
   JSON file, so the values never reach a command line.
3. It runs `modal deploy infra/modal/app.py`. The image's last build step runs
   `infra/modal/check_trainer.py`, so an image whose trainer refuses the pipeline's argv
   fails the deploy instead of the first real capture.
4. It runs a smoke (`infra/modal/smoke.py`). Forty 640×480 renders of the committed
   synthetic tree go through COLMAP on the runner, then 500 gsplat steps on a Modal L4
   through `CloudRunner` → `ModalAdapter` → R2, then georeference, placement and
   packaging. It asserts that gaussians came back, that PSNR was read from gsplat's own
   stats file, that every step ran, and that a tileset was written. It deletes its
   `runs/<run id>/` keys afterwards.

It runs on a push to `living-models` that touches `infra/modal/**`, the
pipeline's modules or recipes, `tools/captures/*.py` or the workflow itself. The image
carries a copy of the pipeline, and the `train` stage's own code runs inside it, so any of
those changes is a change to what the GPU box runs. Each such push costs a smoke. Once it is on
`main` it can be dispatched from the Actions tab (smoke on or off, steps, tier).

**Cost.**

- **Each smoke: about $0.07–0.20.** An L4 is $0.000222/s (Modal's list price, read
  2026-09-22). Budget 5–15 minutes of GPU per smoke. That is an estimate: the cold start
  pulls an image estimated at roughly 10 GB, and it should dominate the time, not the
  training.
- **The first deploy** also builds the image on Modal's builders: 15–30 minutes, once.
  That covers torch, the gsplat wheel, and fused-ssim compiled with nvcc. Later deploys
  reuse every unchanged layer.
- **A real capture** at the recipe's 30 000 steps: unmeasured, because no real capture
  has been trained anywhere yet. The smoke's attempt ledger (billed seconds for 500
  steps, in its job summary) is the first number to scale from; every hour of L4 is
  $0.80.
- **Idle costs nothing.** No container stays warm.

`rehearse` it locally first if you have changed it. That runs the same flow with a
subprocess in place of Modal, a directory in place of R2, and the test suite's stand-in
trainer:

```bash
cd tools/pipeline && uv run python ../../infra/modal/smoke.py --rehearse --work /tmp/smoke
```

### The worker image carries COLMAP

`infra/api.Dockerfile` is built on `ubuntu:24.04` for one package: `colmap` 3.9.1, the
version every finding in `tools/pipeline/sfm.py` was measured on. Debian bookworm has 3.8.
The COLMAP closure is 176 packages, about 370 MB installed. It lands on the `app` machines
too, because Fly runs one image for both process groups. `QT_QPA_PLATFORM=offscreen` is set
because COLMAP links Qt. ffmpeg is not a system package: the pipeline uses the binary in
the `imageio-ffmpeg` wheel. CI's `image` job asserts the COLMAP version and the variable
against the built image.

**`pose` does not run on the worker.** It runs on Modal's CPU box, tier `cpu4` (4
physical cores, 8 GB, no GPU), from its own Ubuntu 24.04 image with COLMAP 3.9.1. The
worker stays `shared-cpu-2x` with 2 GB, which is right for Lane 1 and for supervising
Lane 2, and wrong for doing `pose` in place:

- **Memory.** COLMAP's feature extraction peaked at **1.7 GB** resident, measured on
  1080×1920 iPhone frames at the recipe's 1600 px and 4 threads. With the worker's own
  Python processes alongside, 2 GB would be an out-of-memory kill waiting to happen.
- **CPU.** `pose` on 100 frames is roughly **40 minutes of 4 busy cores** (extrapolated;
  `tools/pipeline/README.md § Where pose runs`), and Fly's shared CPUs are throttled
  under sustained load.

Priced 2026-09-23, `iad`: a Fly worker big enough (`performance-4x`, 8 GB) is
**$124.00 a month**, always on; `shared-cpu-4x` with 8 GB is $23.66 but throttled. Modal's
`cpu4` is **$0.2526 an hour, billed per second only while a capture is posed** -- about
$0.17 for the 40 minutes above. The cost of that choice is one more round trip: the
frames go up to R2 and the poses come back, which for 100 JPEG frames is tens of MB.

**Why COLMAP is still in the image, then.** Checked 2026-10-02 with dropping its ~370 MB
in mind: `georeference` (`exif_gps`) declares no `gpu:`, so it runs on the worker under
every runner, and it calls `colmap model_aligner`. With `WORKER_RUNNER=local` — a
development box, a deployment without Modal — `pose` runs on the worker as well. And CI's
`image` job asserts `sfm.colmap_version() == "3.9.1"` in the built image. Taking it out
would mean moving `georeference` to Modal too, or reimplementing the alignment.

### Disk: the 20 GB volume

A Lane 2 run in flight holds several things on the worker's volume at once:

- the upload (a phone video is 0.1–2 GB a minute);
- every candidate frame ffmpeg extracted before selection (`fps: 4` over the whole clip,
  as JPEG);
- COLMAP's database;
- the trained PLY coming back from Modal (hundreds of MB);
- the packaged tiles.

When a run ends — finished, failed or cancelled — the worker deletes `inputs/` and every
stage's `work/` (`WorkerConfig.tidy_finished_runs`, on by default). That leaves `out/`,
`step.json` and `checkpoint/`, which is everything "retry from this stage" reads: a few
hundred MB per Lane 2 run. **20 GB is therefore enough for one capture of up to about 8 GB
in flight, with room for dozens of finished runs.** Past that, run `fly volumes extend`,
which needs no redeploy.

The worker also checks before it claims. Below `WORKER_MIN_FREE_GB` (5) free it evicts
the workdirs of runs that ended more than `WORKER_EVICT_AFTER_DAYS` (7) ago, oldest
first; if that is not enough it does **not** claim, and logs `NOT CLAIMING` at error
level once a minute — the job stays queued (and alerts, with `QUEUE_CHECK_URL`) instead of
failing on a full disk after its download. A run whose workdir was evicted still retries;
it starts over from the upload.

That is also why the worker has one general slot and not two: a second Lane 2 run in
flight is a second upload, frame set and database on the same volume (and a second GPU
billed). The second slot is CPU-only (`WORKER_CPU_ONLY_SLOTS`) and takes only
`splat-ingest`, whose run holds the uploaded splat, its normalized copy, the packer's
sorted working file (about the splat's size) and the tiles — a few times the upload while
`package` runs, not measured on this volume. A very large ingest beside a very large video
is the case that wants `fly volumes extend`.

### Uploads: what the API accepts

Nothing in the upload path limits size or type, and a video needs nothing Lane 1 did not:

- **Size.** Parts are 8 MiB until a file would need more than S3's 10 000. Past 78 GiB the
  part size grows (`choose_part_size`), so the only ceiling is the provider's. R2 and S3
  both allow 5 TiB for one object.
- **Part URLs** are presigned 32 at a time (256 MiB of upload per window), so a 12 GB video
  never needs one response of 1 500 URLs.
- **Content type** is whatever the client declares, stored on the object;
  `application/octet-stream` otherwise. Nothing routes on it: Lane 2 opens whatever it
  is given with ffmpeg, and Lane 1 reads a splat by its extension and header.
- **The worker streams the upload to disk** (`ObjectStorage.download_file`). It never
  holds the file in memory, which matters on a 2 GB machine.
- **Which lane runs is the client's choice.** `POST /captures/{id}/process` takes a
  `recipe`. A video must be sent as `photo-reconstruct`. Sent as `splat-ingest`, it fails
  at `normalize` naming the formats Lane 1 reads (`.ply`, `.spz`), not silently.

## Database — Neon

Any PostgreSQL 16 with `CREATE EXTENSION postgis`. The first migration creates the extension
if the role is allowed to; on a managed service that does not allow it, enable PostGIS in the
console first. On Neon the owner role may create it, so `provision.yml` runs
`CREATE EXTENSION IF NOT EXISTS postgis` over `psql` and a refusal is a named failure there
rather than an opaque one inside `release_command` later.

The connection string must use the `postgresql+psycopg://` scheme (SQLAlchemy's driver
selector), which is **not** the scheme the provider hands you — rewrite it.
`provision.yml` does that rewrite, and appends `sslmode=require` if Neon's string has no
`sslmode` of its own.

Migrations run from `fly.toml`'s `release_command`, so a deploy whose migration fails does
not replace the running version. Back up with the provider's tooling; the schema is small
and the heavy data lives in ion and R2.

Neon scales its compute to zero after five minutes with no queries, which an idle
deployment now gives it: the worker exits once it has been idle for `WORKER_IDLE_EXIT_S`
and the API machine suspends. See
[waking the worker](#waking-the-worker-and-letting-neon-sleep).

## Object storage and CORS — R2

Large assets never transit the API: the browser PUTs to presigned S3 multipart URLs and
CesiumJS reads tiles straight from the bucket. Both are cross-origin, so the bucket's CORS
configuration is part of the deployment, not an afterthought.

### The Modal token is a runtime credential, and lives on Fly

Worth stating plainly because the intuition points the wrong way: **the worker is what
calls Modal.** `app/worker/cloud.py` builds `ModalAdapter` and `spawn`s from the Fly
machine, at run time, so `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are ordinary runtime
secrets alongside `DATABASE_URL` — not a build-time or CI credential. Modal's SDK reads
both straight from the environment, which is why there is no setting for them in
`app/config.py`.

Three credentials are involved and they point in different directions:

| Held by                      | What                                    | For                                                                        |
| ---------------------------- | --------------------------------------- | -------------------------------------------------------------------------- |
| **Fly secrets**              | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | the worker spawning a stage on Modal                                       |
| **Modal's own secret store** | `twin-object-storage` (the R2 pair)     | the GPU container fetching inputs and uploading outputs                    |
| GitHub Actions secrets       | the same Modal pair, optionally         | only so there is one place to type it — `provision.yml` forwards it to Fly |

`provision.yml` sets both halves or neither and fails if given one: half a pair is a
worker that starts and cannot authenticate.

**The client is an optional extra.** `modal` is not a dependency of `apps/api`;
`infra/api.Dockerfile` installs it with `--extra modal` because the worker needs it, and
a build that drops the flag produces an image that cannot dispatch. That is not a silent
failure: `Worker.from_settings` refuses to start when `WORKER_CLOUD_PROVIDERS` names a
provider whose client is missing, rather than claiming a capture and failing on it.

### Two buckets, one key scheme

**Making an R2 bucket readable makes the whole bucket readable.** Cloudflare's public
bucket feature "allows users to expose the contents of their R2 buckets directly to the
Internet"; there is no per-prefix scoping, and a custom domain has the identical property.
Run with one bucket and a public URL on it and every raw capture anyone uploads is
world-readable, along with every run's frames, logs and checkpoints. Keys carry UUIDs so
they are not enumerable _through the bucket_, but they are not secret: they are in the
database, in the Outputs view, and in the URLs the console renders. Nothing about the
deployment looks wrong — the globe works.

So production runs two buckets and one key scheme. A published object keeps the key it
already had, with the publish's generation added after the job id —
`runs/<job>/p<generation>/<stage>/…` (below) — and the bucket differs.

| Bucket                         | Holds                                                                                                                    | Public |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------ |
| `OBJECT_STORAGE_BUCKET`        | `captures/` (raw uploads) and `runs/` (frames, poses, logs, checkpoints, artifacts)                                      | no     |
| `OBJECT_STORAGE_PUBLIC_BUCKET` | what a run publishes (the packaged tileset, the thumbnail) and what `app.seed.publish` writes (`sites/`, `catalog.json`) | yes    |

Two paths put things in the public bucket, and they differ for a reason.
`app/seed/publish.py` writes `sites/` and `catalog.json` **straight there**, because those
exist only to be fetched by a browser and never hold anything else — and so does
`POST /sites/{id}/thumbnail`, at `sites/<site id>/thumbnail.<ext>`. (It used to write to the
private bucket while saving a URL on the public host, so every uploaded thumbnail 404'd once
there were two buckets.)
`app/worker/publish.py` **copies** a finished run's tileset and thumbnail across with a
server-side `CopyObject`, because a run produces those into the private bucket alongside
things that must stay there. A copy, not a move: the private bucket keeps the originals,
which are what the artifacts table and reconciliation read.

The copies go eight at a time, and `tileset.json` goes **last**, once every tile's copy has
returned, so a public `tileset.json` always means the files it names are there. A copy
that fails stops the publish before the root is copied, and the run registers no site
(as before). The first real 514-tile capture took about eight minutes to publish one copy
and one read-back at a time; against a fake store with a fixed 20 ms a request, the same
514 objects went from 21.4 s (1,029 requests) to 1.4 s (515). The artifact uploads before
it run eight at a time as well.

**Every publish writes a generation of its own** (`app/services/published.py`). A run's own
keys are not written once: a phone's Refine re-runs the _same_ job from `train`, and
`package` and `thumbnail` write the same tile names again with new geometry ("Retry from
this stage" does the same). Those keys used to be copied across unchanged while every
non-JSON key under `runs/` was served `immutable` for a year, so after a Refine browsers
and the edge kept the preview's tiles under the new, short-cached `tileset.json` —
geometry from two reconstructions in one scan — and a republish that failed half way had
already overwritten part of the live site. Now the copies go to
`runs/<job>/p<generation>/…`, the site's asset (and thumbnail, and coverage overlay) are
repointed there only once the whole tileset is in, and the live generation is never
written again: a republish that fails changes nothing a viewer sees, and leaves the site
as it was. The generation is a hash of the published objects' keys, sizes and ETags, so
publishing the same bytes again lands on the same keys (a retried `register` keeps the
browser's cache); without ETags it is random. What was attached beside the live tiles —
objects, a fill, a backfilled grid, the streamed LOD, a plant rig — is carried into the new
generation only where it still holds for the new splats, and what is not is flagged on the
asset ("Sidecars: one publisher", below).

`Cache-Control` follows from that, by one rule shared with the tile proxy
(`functions/r2/[[path]].js`, `outputs.cache_control_for`): a year and `immutable` for a
non-JSON key inside a generation, five minutes with a week of `stale-while-revalidate`
for everything else — JSON (backfill workflows used to rewrite `tileset.json` in place), `sites/`, the
run's own keys in the private bucket (uploaded with the short lifetime, because a Refine
rewrites them), and copies published before generations existed. Each copy is written
with the lifetime of the key it lands on, so a browser reading the public bucket's own
URL is told the same as one going through the proxy.

**Old generations stay.** A Refined or re-registered run leaves its previous generation
in the public bucket, unreferenced; nothing deletes it yet. Removing a generation no
site's asset or thumbnail URL names (`runs/<job>/p<generation>/`) is a safe cleanup, and
a follow-up — like the gap below, it is about the public bucket, which nothing walks.

One consequence worth stating plainly: **reconciliation does not see the public bucket.**
`app/services/reconcile.py` walks `captures/` and `runs/` in the private bucket, so a
published copy whose original is deleted stays served until something removes it too.
That is a known gap, not an oversight.

`r2.dev` is not the answer for the public bucket either. Cloudflare documents it as
rate-limited and "intended for non-production traffic", so connect a custom domain and
put it in `OBJECT_STORAGE_PUBLIC_URL`.

**CORS: one document per bucket.** A bucket has exactly **one** CORS configuration and
setting it replaces what was there, which is why a bucket in two roles needs its two rules
combined. After the split each bucket has one role and takes one document:

| File                         | Role                                                                        |
| ---------------------------- | --------------------------------------------------------------------------- |
| `infra/cors/upload.json`     | the **private** bucket: browser PUTs to presigned URLs                      |
| `infra/cors/tiles.json`      | the **public** bucket: CesiumJS reading `tileset.json` and `.glb`           |
| `infra/cors/production.json` | both rules in one document, for a deployment that still has a single bucket |
| `infra/cors/dev-minio.xml`   | both rules in one XML document, for the single dev bucket                   |

`ExposeHeaders: ["ETag"]` in the upload rule is load-bearing. Without it the browser reads
`etag === null` from each part's response and a multipart upload can never be completed.
`apps/api/tests/test_cors_rules.py` fails if it is dropped, if the XML and JSON documents
drift apart, or if `production.json` stops matching the two rules it combines.

Apply it — once per bucket, naming the document after a `--`:

```bash
AWS_ACCESS_KEY_ID=…  AWS_SECRET_ACCESS_KEY=… \
  infra/cors/apply-r2.sh <account-id> twin-assets https://twin.example.com -- upload.json
AWS_ACCESS_KEY_ID=…  AWS_SECRET_ACCESS_KEY=… \
  infra/cors/apply-r2.sh <account-id> twin-public https://twin.example.com -- tiles.json
```

With one bucket, omit the `--` and it applies `production.json`, which is both rules
combined. `provision.yml` runs the two-bucket form.

The script substitutes your real origin into the committed placeholder (so the origin never
has to be committed), applies the document with `aws s3api put-bucket-cors` against
`https://<account-id>.r2.cloudflarestorage.com` with `--region auto`, and then prints what
the bucket reports back. **Read that output.** It is the first time any of this meets a real
R2 API.

### Sidecars: one publisher

A scan's directory holds more than its tiles. Other steps add files beside them and declare
them on the root tile's `extras`, where the web finds them: the segmentation's
`instances.json` (+ `.emb`), a backfilled `collision.bin`, an inferred fill under
`inferred/<name>/`, PlayCanvas's streamed level of detail under `sog/`, a plant rig
(`rig.json`, `motion.json`, `plants.json`, named by the asset's `renderConfig.rigUrl`). The
GitHub workflows that make them (`publish-instances.yml`, `publish-fill.yml`,
`collision-backfill.yml`, `streamed-lod-backfill.yml`, `living-plants.yml`) used to write
them **in place**: read the live `tileset.json`, inject a key, write it back. That breaks
three ways — two runs at once drop each other's key; inside a generation, served
`immutable` for a year, a rewritten file is stale in every cache that holds it; and a worker
republish cuts a generation from the run's own outputs, which have none of them, so
objects, collision, fill and streamed LOD vanished from the live site.

So there is **one publisher: the API.** A workflow stages its files and asks the API to
attach them; the API cuts a new generation and repoints the asset. Nothing writes a
published directory in place any more.

**`POST /api/v1/assets/{asset_id}/sidecars`**, with `Authorization: Bearer
$API_WRITE_TOKEN`:

| Field           | Required | Meaning                                                                                                                                                                                                 |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stagingPrefix` | yes      | `staging/assets/<asset id>/<token>/` in the **private** bucket (`OBJECT_STORAGE_BUCKET`); the token is letters, digits, `.`, `_`, `-` (a run id and attempt). Every object under it is attached.        |
| `basedOn`       | yes      | the tileset URL the files were computed against. Accepted when the asset's current tiles are those tiles — the same URL, or a later generation another attach cut from them; **409** after a republish. |
| `files`         | no       | exactly the paths that must be staged; a partial upload is then a 422 instead of an attach.                                                                                                             |
| `extras`        | no       | root `extras` keys to set, each replaced whole; `null` removes one. A list of `{uri, …}` entries (`inferredLayers`) is merged **by uri**, so send only your own entry. Every `uri` named must exist.    |
| `rigUrl`        | no       | sets `renderConfig.rigUrl` in the same transaction as the new URL (`null` clears it) — the separate PATCH `living-plants.yml` made raced with everything else.                                          |

The staged layout is the layout beside `tileset.json`: `staging/…/run-7/instances.json`
lands at `<new generation>/instances.json`, `staging/…/run-7/inferred/fixer/0.glb` at
`<new generation>/inferred/fixer/0.glb`. A staged file is refused (422, before anything is
copied) unless every path segment is plain (letters, digits, `.`, `_`, `-`, not starting
with `.`; at most six segments), its extension is one the tile proxy serves (`json`, `glb`,
`bin`, `emb`, `f32`, `u8`, `webp`), it is at most 1 GiB, and the attach holds at most 5,000
files and 8 GiB. It may not be `tileset.json` (the API writes that) or one of the scan's tiles,
and nothing under `objects/` or `fills/` (a split rewrites the scan's tiles; it is a new
tileset, not files beside one). `extras.gaussians` belongs to the tileset and cannot be set.

What the API does, holding the asset's row lock (`app/services/attach.py`):

1. lists the asset's current directory in the public bucket — a generation, or a **legacy
   prefix** published before generations (`runs/<job>/package/splat/`: the spool, pumpkin
   and camp scans, infra/modal/segment.py `SCANS`) — and refuses an asset that is not a
   run's tileset there (an ion asset, a seeded `sites/` scan: 409);
2. checks `basedOn` holds the same tiles (the tileset without its root extras, and every
   tile's size and ETag);
3. writes a **new** generation, `runs/<job>/p<generation>/…`: every current object — tiles
   and every sidecar already there — copied server side, eight at a time; the staged files
   beside them (a staged file under `inferred/<name>/` or `sog/` replaces that whole
   directory, so a smaller rebuild leaves no stale chunk); and **last**, once every copy has
   returned, `tileset.json` with the merged root extras. Every object gets
   `Cache-Control: public, max-age=31536000, immutable`, `tileset.json` included: nothing
   ever writes a generation twice;
4. moves the asset's URL to the new `tileset.json`, clears the asset's flags for the kinds it
   attached, commits, and deletes the staged files.

It answers with `url`, `previousUrl`, `generation`, `copied`, `staged`, `attached` and
`carried` (sidecar kinds), the root `extras` keys now declared, and the asset. A failure
before the commit leaves the asset exactly as it was: the half-written generation is keys
nothing points at, and the staged files stay for a retry.

**Serialised per asset.** The attach takes `SELECT … FOR UPDATE` on the asset, so a second
attach waits (up to 60 s, then 409 — retry) and builds on the first's generation; neither
loses the other's files. The lock is held across the copies — seconds for a thousand tiles
— with `idle_in_transaction_session_timeout` set to ten minutes for that transaction alone,
so a process that freezes holding it is cut off by the database rather than by TCP hours
later. The worker's `register` takes the same lock before it repoints: if an attach moved
the asset while the run was publishing, it publishes again on top of the attach's
generation (up to three times, then the run's tiles are withheld and the site is left as the
attaches made it).

**A republish carries what still holds** (`app/worker/carry.py`). Before copying, the worker
reads the live generation and decides each sidecar kind by what it depends on (the table is
docs/SCENE_OBJECTS.md, section 8): a kind bound to the splats' positions (objects, skins,
the rig) is carried when every new tile's position checksum — computed by the worker from
the run's own tiles, with the function the binding was written with — is one its binding
lists, so a re-pack with another spherical-harmonics degree keeps it; a kind bound to the
tiles' bytes (collision, view cones, `sog/`) is carried only when the new tiles are the very
same tiles; a kind keyed by instance ids (materials, telemetry) goes with `instances`; an inferred fill,
placed in the scan's frame with no splat indices, is always carried; a kind the run makes
itself (the packer's `collision.bin`, `viewcones.bin`) is replaced by the run's. Every
dropped kind becomes a flag on the asset — `sidecarFlags` in every asset response, e.g.
`{"kind": "instances", "action": "Objects need re-segmenting", "reason": …, "jobId": …}` —
and a warning in the worker's log; attaching that kind again clears it. With one bucket
nothing can be carried (a run's tileset is its own keys), so everything is dropped and
flagged.

**What a run provides is only what this attempt wrote.** A Refine re-runs the same job, and
its `package` uploads into the same `runs/<job>/<stage>/splat/` in the private bucket as the
first attempt did. The worker now removes whatever is under that prefix and not in the
upload (`outputs._prune`): a `collision.bin` the first packer wrote and the second did not
would otherwise have been published beside the new tiles and read by the carry plan as the
new run's own grid, replacing the live one.

**The workflows' side: one script.** The five workflows that publish beside the tiles —
`publish-instances.yml`, `publish-fill.yml`, `collision-backfill.yml`,
`streamed-lod-backfill.yml`, `living-plants.yml` — all go through
`tools/captures/attach_sidecars.py`, and none of them writes the public bucket or a
`tileset.json` any more. The build job resolves the asset's **current** tileset
(`attach_sidecars.py resolve`: `GET /api/v1/assets/{id}`), checks its files against it, and
writes the request beside them as `attach.json` (`attach_sidecars.py manifest`), kept in the
review artifact. The publish job runs `attach_sidecars.py attach <dir>`: it uploads the files
with the R2 pair to `staging/assets/<asset id>/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT/` in the
private bucket (clearing anything a failed earlier try left under that prefix), POSTs the
request, retries a 409 that says another attach holds the asset, and ends with exit status 3
on a 409 that says the tiles changed under it — run the workflow again on the asset's current
tiles. By hand, the same request is:

```bash
PREFIX="staging/assets/$ASSET_ID/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT/"
curl --fail-with-body -sS -X POST "$TWIN_API_URL/api/v1/assets/$ASSET_ID/sidecars" \
  -H "Authorization: Bearer $API_WRITE_TOKEN" -H "Content-Type: application/json" \
  -d "$(jq -n --arg prefix "$PREFIX" --arg base "$BASED_ON" --slurpfile extras extras.json \
        '{stagingPrefix: $prefix, basedOn: $base, extras: $extras[0]}')"
```

What each workflow stages and sends is in docs/SCENE_OBJECTS.md, section 8. What their
publish jobs read:

| Name                    | Kind     | Read by                                       | For                                                        |
| ----------------------- | -------- | --------------------------------------------- | ---------------------------------------------------------- |
| `API_WRITE_TOKEN`       | secret   | all five                                      | the attach is a write                                      |
| `CLOUDFLARE_ACCOUNT_ID` | secret   | all five                                      | the R2 endpoint, `https://<id>.r2.cloudflarestorage.com`   |
| `R2_ACCESS_KEY_ID`      | secret   | all five                                      | staging in the private bucket                              |
| `R2_SECRET_ACCESS_KEY`  | secret   | all five                                      | its other half                                             |
| `R2_BUCKET`             | variable | all five (default `twin-assets`)              | the **private** bucket the API reads `staging/` from       |
| `TWIN_API_URL`          | variable | all five (default `https://twin-api.fly.dev`) | the API the asset is read from and the attach is sent to   |
| `SCAN_ASSET_IDS`        | variable | publish-instances, publish-fill               | which asset each scan of infra/modal/segment.py `SCANS` is |

None of them needs the public bucket's name or URL any more. They still use the repository's
one R2 pair, which can write the public bucket too; a pair scoped to the private bucket alone
would do for all five, but it would have to live under other secret names, because
`R2_ACCESS_KEY_ID` is also what provisioning gives the API (below).

**`SCAN_ASSET_IDS`: what the owner fills in.** publish-instances and publish-fill name a scan
the way segment.yml and fill.yml do (`spool`, `pumpkin`, `camp`: infra/modal/segment.py
`SCANS`, legacy public URLs), but an attach is to an **asset**, and asset ids are random
UUIDs the production database made, not knowable from the repository (the seed data has no
run scans). Set the repository variable to a JSON object naming the three:

```bash
curl -s "$TWIN_API_URL/api/v1/assets" | jq '[.[] | select(.source.url? // "" |
  test("/runs/(8e1cc115-cb80-4af2-81fc-dccaf6b65891|430c1932-5b6a-47b1-bb71-bb7fa2fec86b|50c25673-0940-4574-9b96-0b21362f83ca)/"))
  | {id, name, url: .source.url}]'
# then, in Settings → Secrets and variables → Actions → Variables:
#   SCAN_ASSET_IDS = {"spool": "<id>", "pumpkin": "<id>", "camp": "<id>"}
```

Until it is set, `attach_sidecars.py resolve --scan` finds a scan's asset by its run — the
one gaussian-splat asset whose tileset is under `runs/<job>/` of the scan's legacy URL — and
says so in the log; no match, or more than one, is a refusal that names the variable. The
other three workflows take the asset from the capture (collision-backfill, living-plants:
`fetch_capture.py`) or as their input (streamed-lod-backfill: `asset=<uuid>`).

**Split objects are not attached.** `split_objects.py` rewrites the scan's own tiles, so a
split is a new tileset, not files beside one; fill.yml's `split:<scan>` jobs only keep it in
their artifact for review, nothing publishes it, and the attach refuses `objects/`, `fills/`,
`extras.objects` and `extras.split`. Publishing a split needs a replace-tiles publish
(docs/SCENE_OBJECTS.md, section 8), left until a split is wanted on the live site.

**Two operator steps.** A lifecycle rule that expires `staging/` in the private bucket after
a week, for attaches that failed and were never retried:

```bash
aws s3api put-bucket-lifecycle-configuration --bucket twin-assets \
  --endpoint-url https://<account-id>.r2.cloudflarestorage.com --region auto \
  --lifecycle-configuration '{"Rules":[{"ID":"staging","Status":"Enabled","Filter":{"Prefix":"staging/"},"Expiration":{"Days":7}}]}'
```

And migration 0009 (`assets.sidecar_flags`), which `release_command` applies on deploy.

`Cache-Control`, once more: an attach writes its whole generation immutable, JSON included,
and since every workflow goes through it nothing writes a generation twice — so the tile
proxy now serves every non-JSON key inside a generation for a year, sidecars too
(`collision.bin`, `instances.emb`, `sog/`, `inferred/…`; it used to keep those short, when
backfills rewrote them in place). A legacy prefix (`runs/<job>/package/splat/`, written in
place for years) is outside any generation and stays short. The worker's own publish still
writes a generation's JSON with the short lifetime, and the proxy still serves JSON short
whatever the object says; that is only slower than it needs to be, never wrong, and the two
can move to immutable together.

### Narrowing the credentials: operator steps

Two credentials are broader than they need to be today. Neither is a code change — the
runtime configuration is deliberately left as it is — so these are steps for whoever holds
the accounts, in the order to take them.

**One R2 key pair does three jobs.** `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` become the
API's and the worker's `OBJECT_STORAGE_*` pair on Fly, _and_ `modal.yml` copies the same
pair into the Modal secret `twin-object-storage` for the GPU container. They need different
things:

| Who                   | Needs                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| API and worker (Fly)  | Object Read & Write on the **private** bucket, and on the **public** one (publishing is a `CopyObject`) |
| GPU container (Modal) | Object Read & Write on the **private** bucket only: inputs in, `runs/<id>/` out                         |

So a leak from Modal's side today is write access to the bucket the world reads. To split it:

1. R2 → Manage API tokens → Create API token: **Object Read & Write**, "Specify bucket(s)" →
   the **private** bucket only (`R2_BUCKET`, `twin-assets` by default). Note the pair.
2. Replace the Modal secret with it — from a file, so the values are not on a command line:
   write `OBJECT_STORAGE_ENDPOINT_URL`, `OBJECT_STORAGE_ACCESS_KEY`,
   `OBJECT_STORAGE_SECRET_KEY`, `OBJECT_STORAGE_BUCKET` and `OBJECT_STORAGE_REGION=auto` as a
   JSON object and run `modal secret create twin-object-storage --force --from-json <file>`.
   Delete the file. **Re-running `modal.yml` puts the broad pair back** (it builds the
   secret from `R2_ACCESS_KEY_ID`), so repeat this step after any run of it.
3. Make sure the repository's pair — the one Fly gets — is scoped to **exactly the two
   buckets** (`R2_BUCKET` and `R2_PUBLIC_BUCKET`) and nothing else on the account. If it
   was minted for "all buckets", mint a two-bucket token, put it in `R2_ACCESS_KEY_ID` /
   `R2_SECRET_ACCESS_KEY`, and re-run **Provision** (it re-sets the Fly secrets).
4. Revoke the old token in R2 → Manage API tokens once uploads, a Lane 2 run and a publish
   have all worked on the new ones.

**The phone key's hash is in `fly.toml`.** A salted PBKDF2 hash, 200,000 rounds, of a key of
about 59 bits — not the key, and slow to attack — but it is in a public repository, where an
offline guesser has all the time it likes and the API's rate limit does not apply. To take
it out of the repository:

1. Generate a new key (the old hash has been public, so rotate rather than move it) and its
   hash, in `apps/api`:
   `uv run python -c "import secrets, sys; from app.services.phone_key import hash_key; k = '-'.join(secrets.token_hex(2) for _ in range(4)); print(k); print(hash_key(k, salt=secrets.token_bytes(16)), file=sys.stderr)"`
   — the key on stdout is for the phone; the hash on stderr is for Fly.
2. `fly secrets set -a twin-api API_PHONE_KEY_HASH='<the hash>'` — single quotes, because
   the hash contains `$`.
3. In the same change, delete the `API_PHONE_KEY_HASH` line from `fly.toml`'s `[env]`, so
   the old value cannot come back with a later deploy, and deploy.
4. On each phone: **Forget key**, then type the new one.

### MinIO, for development

`pnpm infra:up` applies `dev-minio.xml` automatically via the `minio-init` one-shot. By
hand, or against a MinIO you run elsewhere:

```bash
mc alias set local http://localhost:9000 twin twin-secret
mc cors set local/twin-assets infra/cors/dev-minio.xml   # mc takes XML, not JSON
```

If your `mc` predates the `cors` subcommand (added in 2024), use the S3 API instead — it is
the same call the AWS CLI makes:

```bash
python - <<'PY'
import boto3, json
boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="twin",
    aws_secret_access_key="twin-secret",
    region_name="us-east-1",
).put_bucket_cors(
    Bucket="twin-assets",
    CORSConfiguration={"CORSRules": json.load(open("infra/cors/upload.json"))["CORSRules"]},
)
PY
```

### Two R2 specifics, both of which apply to presigning as much as to CORS

- **Presigning happens on the S3 API domain; public reads come from the public domain.**
  That is why `OBJECT_STORAGE_ENDPOINT_URL` and `OBJECT_STORAGE_PUBLIC_URL` are separate
  settings. Get this wrong and uploads work while every tile 404s, or the reverse — which
  is the last check `provision.yml` runs, and the only one that catches it.
- **R2 wants `region="auto"`.** `fly.toml` sets `OBJECT_STORAGE_REGION=auto` for you.

### The public URL, and what enabling it costs

`provision.yml` enables R2's **free managed public URL** — the `*.r2.dev` development URL —
rather than connecting a custom domain, and this is a deliberate trade with two sides.

What it buys: no DNS record, and one fewer name to keep in step. The Pages origin,
`API_CORS_ORIGINS`, the bucket's CORS rule and the ion token's allow-list all have to
agree already; a custom tiles domain would be a fifth name with no automation behind it.
The `r2.dev` URL is handed back by the API that enables it, so it can be _computed_ rather
than agreed.

What it costs, and neither of these is small:

- **It makes the whole bucket world-readable, including `captures/`.** That prefix is where
  browser uploads land. Keys contain UUIDs, so they are not enumerable through the bucket
  itself, but they are not secret either: anyone who has a key has the bytes. The same is
  true of a custom domain on a public bucket — public is public — so the fix is not a
  domain but a second, private bucket for uploads, or a Worker in front that signs reads.
  That is not built. If your captures are sensitive, do not deploy this as it stands.
- **`r2.dev` is rate-limited by Cloudflare and is not intended for production traffic.**
  It is fine for one person looking at their own captures; it is not a CDN.

**The custom-domain path**, when you want it: connect the domain under R2 → your bucket →
Settings → Public access, then set the `R2_PUBLIC_URL` repository variable to it (e.g.
`https://tiles.example.com`). `provision.yml` then stops touching the managed URL
altogether and uses yours everywhere — Fly secret, seeded tileset URLs, offline catalog.
Re-run it and re-run **Deploy** so the rebuilt bundle points at the new catalog.

> **Still unverified, and deliberately flagged as such.** It has been reported that R2
> rejects a narrow `AllowedHeaders: ["content-type"]` rule and requires `["*"]`. C1 was
> meant to check that against the real thing and **could not**: `developers.cloudflare.com`
> is unreachable from this environment and there is no R2 account to try it against. The
> claim rests on an assertion, not on evidence. The committed documents use `["*"]`, which
> is accepted either way, so nothing depends on the answer — but do not repeat the claim as
> fact. What _is_ established is that the narrow form is shape-valid to botocore, so nothing
> in CI or in the test suite would catch R2 rejecting it: it would first fail at deploy.
> `apply-r2.sh` prints the bucket's own `get-bucket-cors` afterwards so that whoever runs it
> first can correct this note.

### Presigned URLs are SigV4, explicitly

`S3Storage` passes `signature_version="s3v4"` explicitly. This is not redundant: with the
default, botocore's `_default_s3_presign_to_sigv2` makes `generate_presigned_url` emit
**SigV2** for every region except `auto`, while `client.meta.config.signature_version` still
reports `s3v4`. MinIO in dev (`us-east-1`) would have presigned SigV2 while R2 in production
(`auto`) presigned SigV4 — a dev/prod split no configuration inspection can see. The tests
assert on the emitted URL string for that reason, and CI runs them against a real MinIO.

### Publishing tiles, and what the console does when the API is down

**The image contains no `data/tiles`.** The API's `/api/v1/tiles` static mount is
development-only and production disables it outright. Publish the tiles once, from a
checkout that has them:

```bash
cd apps/api && uv run python -m app.seed.publish
```

That uploads `data/tiles/<slug>/**` to `sites/<slug>/**` in the bucket and writes
`catalog.json` beside it. Re-run it with `--catalog-only` after registering a pipeline run,
so the offline catalog keeps up.

Two objects in the bucket matter to the browser rather than to the API:

- **`sites/<slug>/**`** — a capture's 3D Tiles, fetched directly by CesiumJS. Needs the read
  rule (`GET`/`HEAD`, `Accept-Ranges` exposed) and public read access, or a signed-URL
  worker in front.
- **`catalog.json`** — every site, in the shape `GET /api/v1/sites/{id}` returns. The web app
  fetches it only after a catalog request has actually failed; it is what
  `VITE_OFFLINE_CATALOG_URL` points at.

That second one is what "offline" means now. Capture tiles used to be a static mount on the
API, so an unreachable API meant unreachable tiles. The bytes now sit behind a host that does
not go down with the API — but _which_ captures exist is a database fact, and the database is
the thing that is unreachable. Publishing `catalog.json` is what makes that fact readable
without the API. Every write form stays disabled in that state: an offline capture is
readable, never editable.

## Provisioning and deploying from GitHub Actions

Two workflows, both **`workflow_dispatch` only** and neither of them on `push`. They need
credentials that do not exist in this repository, and a `push` trigger would paint every
commit on main red for want of a secret and teach everyone to ignore the badge.

| Workflow                          | What it does                                                          |
| --------------------------------- | --------------------------------------------------------------------- |
| `.github/workflows/provision.yml` | creates the infrastructure, then calls `deploy.yml`, then verifies it |
| `.github/workflows/deploy.yml`    | builds and deploys the API, the worker and the web bundle             |
| `.github/workflows/modal.yml`     | deploys the GPU app and proves it with a small training run           |

`modal.yml` is the exception to "dispatch only", narrowly: it also runs on a push to
`living-models` that touches `infra/modal/**` or the pipeline code the GPU
container carries, because `workflow_dispatch` only works for a workflow file that is on
the default branch and this one is not yet. See [GPU training — Modal](#gpu-training--modal).

`provision.yml` exists because of an asymmetry that was measured rather than assumed:
**every provider API is unreachable from the development environment this was written in —
`api.cloudflare.com`, `api.fly.io`, `api.neon.tech`, `registry.fly.io` all fail to connect —
and all of them are reachable from a GitHub Actions runner.** So anything a token can do,
CI can do, and what is left for a person is the part that genuinely needs one: creating
accounts, minting tokens, pasting them into repository secrets. Everything downstream of a
token is a dispatch.

It runs four jobs, in order:

1. **infra** — the Neon project and PostGIS; the R2 bucket, its public URL and its CORS
   document; the Pages project; the Fly app, the worker's volume and every Fly secret. One
   job rather than four, because the values that pass between those steps are secret and a
   job output is neither masked nor scoped. Nothing secret crosses a job boundary in that
   file; the jobs below re-derive the connection string from `NEON_API_KEY`.
2. **deploy** — `deploy.yml`, _called_ (`workflow_call`), not copied. There is one
   description of how this project is deployed and it stays in one file.
3. **seed** — `python -m app.seed` and `python -m app.seed.publish`, on the runner rather
   than over `fly ssh console`: publishing copies `data/tiles/<slug>/**` into the bucket and
   the deployed image deliberately carries no `data/tiles`. The checkout does.
4. **verify** — health, then the recipe catalogue, then a real upload through the presigned
   path. See [what it checks, and why in that order](#what-verification-actually-checks).

Two inputs: `dry_run` (say what exists and what would be created, create nothing) and
`deploy` (chain into the deploy and the checks; on by default).

Names are not repeated: the Fly app, its region, the worker's volume and its size are read
out of `fly.toml`, and the Pages project out of `wrangler.toml`, so provisioning cannot
drift from what `fly deploy` and `wrangler pages deploy` will look for.

`deploy.yml` on its own is unchanged — dispatch it from the Actions tab choosing
`everything`, `api` or `web` — except that it now also accepts a `workflow_call` with two
optional URL inputs, which is how provisioning hands it values a dispatch would have to be
told. Neither workflow uses a third-party action: flyctl comes from Fly's own installer,
wrangler from npm.

CI parses both workflows on every run and fails if either one names a secret or a variable
that this document does not, or if either one grows a `push` trigger.

### Secrets

Repository → Settings → Secrets and variables → Actions → **Secrets**.

| Name                           | Required             | What to put in it                                                          |
| ------------------------------ | -------------------- | -------------------------------------------------------------------------- |
| `NEON_API_KEY`                 | yes                  | a Neon API key (see the scopes below)                                      |
| `CLOUDFLARE_API_TOKEN`         | yes                  | a custom Cloudflare token, two permissions                                 |
| `CLOUDFLARE_ACCOUNT_ID`        | yes                  | the 32-hex id in your Cloudflare dashboard URL                             |
| `R2_ACCESS_KEY_ID`             | yes                  | the R2 **S3** access key id — a different credential from the token above  |
| `R2_SECRET_ACCESS_KEY`         | yes                  | its secret                                                                 |
| `FLY_API_TOKEN`                | yes                  | an **org-scoped** Fly token                                                |
| `API_WRITE_TOKEN`              | yes                  | `openssl rand -hex 32`, generated by you and kept                          |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | strongly recommended | the browser ion token; without it the globe has no ion terrain or imagery  |
| `CESIUM_ION_SERVER_TOKEN`      | no                   | `assets:read`, for server-side asset monitoring. Never a `VITE_` variable  |
| `ANTHROPIC_API_KEY`            | no                   | without it the plan drafter is rule-based and says so                      |
| `MODAL_TOKEN_ID`               | GPU only             | `modal token new`. Forwarded to Fly, because the worker is what dispatches |
| `MODAL_TOKEN_SECRET`           | GPU only             | its other half; provisioning refuses one without the other                 |

`API_WRITE_TOKEN` is the one secret this could have generated and deliberately does not.
Every write the console and the worker make carries it, a Fly secret cannot be read back,
and a value printed into a log to tell you what it is would not be a secret any more. So
you make it and you keep it.

`API_HANDOFF_SECRET` is the opposite case and is the **only** value provisioning generates:
nobody needs to read it, it only has to be the same on every process. It is generated once,
when `fly secrets list` shows the app does not have it, and left alone afterwards —
regenerating it every run would invalidate every outstanding phone-handoff link.

### Variables

Repository → Settings → Secrets and variables → Actions → **Variables**. Every one of these
is optional; the default is in the right-hand column.

| Name                         | Default                    | What it is for                                                                                |
| ---------------------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| `VITE_API_BASE_URL`          | _(passed in)_              | the API origin the bundle is built against                                                    |
| `VITE_OFFLINE_CATALOG_URL`   | _(passed in)_              | `catalog.json`'s public URL                                                                   |
| `VITE_ENABLE_PHOTOREALISTIC` | on                         | `false` unless your ion token has Google access and you accept the terms                      |
| `R2_BUCKET`                  | `twin-assets`              | the bucket's name                                                                             |
| `R2_PUBLIC_BUCKET`           | `<R2_BUCKET>-public`       | the **only** bucket made public; it must differ from `R2_BUCKET`                              |
| `R2_PUBLIC_URL`              | the `r2.dev` URL           | the public bucket's URL; set it to a custom domain to take the managed one out of the picture |
| `WEB_BASE_URL`               | the `pages.dev` URL        | set it to a custom domain on the Pages project                                                |
| `NEON_PROJECT_NAME`          | `hexapod-twin`             | which Neon project to find or create                                                          |
| `NEON_REGION_ID`             | `aws-us-east-1`            | Neon's name for the region `fly.toml`'s `primary_region` is in                                |
| `FLY_ORG`                    | `personal`                 | the Fly organization to create the app in                                                     |
| `TWIN_API_URL`               | `https://twin-api.fly.dev` | the API the five sidecar workflows read an asset from and attach through                      |
| `SCAN_ASSET_IDS`             | _(found by run)_           | `{"spool": "<asset id>", …}`: which asset each segmented scan is ("Sidecars: one publisher")  |

The first two are marked _(passed in)_ because `provision.yml` computes them and hands them
to `deploy.yml` directly — a provisioned first deploy needs no variables set at all. A
**standalone** `deploy.yml` dispatch has nothing to hand it those values, so set them once
provisioning has told you what they are; the provision run's job summary prints both, ready
to copy.

### What verification actually checks

In this order, because it is the order that catches things:

1. **`/api/v1/health`** — the app boots and its database answers. A `false` database here is
   a wrong `DATABASE_URL` or a PostGIS that never got enabled.
2. **`/api/v1/recipes`** — the deployed image carries `tools/pipeline`. A 404 is a lost
   `COPY`, and CI's `image` job catches the same thing against a locally built image.
3. **a real CORS preflight** against a real presigned URL — `OPTIONS` with the Pages origin,
   asserting both `Access-Control-Allow-Origin` and, separately,
   `Access-Control-Expose-Headers: ETag`.
4. **a real upload** — register a file, `PUT` the bytes to the presigned URL, read the
   `ETag` off the response, send it to `/complete`.
5. **a public read** of the object just uploaded, at `OBJECT_STORAGE_PUBLIC_URL`.

Three, four and five are last and they are the load-bearing ones: they are the only checks
that exercise presigning, the bucket's CORS document and the endpoint/public-URL pair
_together_. Nothing earlier catches a missing `ExposeHeaders: ["ETag"]`, and neither does the
`PUT` on its own — curl is not bound by CORS, so the upload succeeds on a bucket whose rules
are wrong. What fails without that header is a **browser's** ability to read the `ETag` it
must send back to `/complete`, which is why step 3 asks for the preflight explicitly instead
of inferring it from step 4 passing. Step 5 is the one that catches an endpoint and a public
URL belonging to different buckets or different accounts.

Verification leaves one capture named `Provisioning check <run id>` and one small text
object in the bucket behind. Deleting them is a console job; leaving them is harmless.

## Security checklist

- No secrets in git; `.env` is ignored, `.env.example` is the template, and everything
  account-specific is a Fly secret or a repository secret.
- The browser ion token has the narrowest scopes and an origin allow-list.
- The API is not a fetch proxy: the only outbound call is the documented ion asset metadata
  read, gated by a server token.
- CORS is explicit, on the API (`API_CORS_ORIGINS`) and on the bucket
  (`infra/cors/production.json`); only `GET/POST/PATCH/DELETE`.
- `pnpm audit --audit-level high` and `pip-audit` run in CI.
- Dataset URLs are validated (scheme, credentials, private hosts in production).
- Reads are open so the world stays viewable; every mutating endpoint requires
  `Authorization: Bearer $API_WRITE_TOKEN`, and the worker uses the same token. So does one
  read: storage reconciliation (`POST /storage/reconciliation`), which walks the private
  bucket and lists what is in it.
- The routes whose cost is the attack are rate-limited per client, in process
  (`app/services/ratelimit.py`): wrong phone keys (each is 200,000 PBKDF2 rounds; a right
  key is never limited), reconciliation, and step-log reads. A refusal is a `429` with
  `Retry-After`. The client is the address in `API_CLIENT_IP_HEADER` (default
  `Fly-Client-IP`, which Fly's proxy sets on every request, and which is believed only when
  `FLY_APP_NAME` says the process is on Fly — anywhere else any client could send it, so
  the socket address is used); behind another proxy set the header that proxy sets, or
  empty to use the socket address. An IPv6 client is its /64, since every address in a
  subscriber's /64 is theirs to rotate through. None of it is reachable by the phone page's
  10-second polling.
- The `/r2/` tile proxy fetches from one pinned bucket host and labels what it serves
  itself; see [the tile proxy](#the-r2-tile-proxy-and-the-one-host-it-serves).
- One queued-or-running job per capture is a partial unique index (migration `0008`), not
  only a check in code, so a double-click on Process is a `409`, not a second run.

## Observability

`apps/web/src/lib/log.ts` and `lib/timing.ts` expose sinks for a vendor (Sentry,
OpenTelemetry). The API adds a `Server-Timing` header, and `fly logs` is the shipper until
there is a reason for another.

**API logs are JSON lines in production** (`app/observability.py`): `time`, `level`,
`logger`, `message`, any structured fields, and `exception` with the traceback. Until this
was configured the API set up no logging at all, so every `twin.api` INFO line (the
production tiles notice, among others) was dropped and warnings came out with no time or
logger name. uvicorn's access and error lines go through the same handler, so one format is
interleaved, not two.

| Variable                    | Default                           | What it does                                                         |
| --------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| `LOG_LEVEL`                 | `INFO`                            | level of the API's own `twin.*` loggers; libraries stay at `WARNING` |
| `LOG_FORMAT`                | `json` in production, else `text` | one JSON object per line, or a line a person reads                   |
| `SENTRY_DSN`                | unset                             | error reporting; unset, `sentry-sdk` is never even imported          |
| `SENTRY_TRACES_SAMPLE_RATE` | `0`                               | the share of requests traced; `0` sends errors only                  |

All four are optional; set `SENTRY_DSN` as a Fly secret (`fly secrets set SENTRY_DSN=…`) —
it is account-specific — and the others the same way if you want something other than the
default. Nothing in `fly.toml` needs to change.

**No line carries a credential.** Every line is redacted before it is written: `Bearer`
values, phone-handoff tokens (`h1.…`), the query string of any presigned URL (its signature
is a working credential for an hour), PBKDF2 hashes, and the literal value of every secret
the process was configured with — `FLY_API_TOKEN`, the password in `DATABASE_URL`, and the
path of `WORKER_HEARTBEAT_URL` and `QUEUE_CHECK_URL` (whoever has a check's URL can ping it,
or keep it quiet) among them. Sentry gets the same treatment, breadcrumbs' `data` included
(its httpx integration records every outgoing request's URL there), and is configured with
no request bodies and no stack-frame locals — a frame holding `Settings` holds every secret
the API has. The wake-up code logs as `twin.worker_wake`, so its INFO lines (which worker
machines a queued job started) are written; as `app.worker_wake` they were dropped.

**A 422 is the caller's fault and a 500 is ours.** Only deliberate validation failures
(`InvalidInputError` in `app/services/errors.py`, URL checks, request validation) are
`422`s. Any other `ValueError` — a failed parse of something the API produced itself, a
pydantic model rejecting a stored row — used to be a 422 too, carrying the bug's own message
as if the caller had made it, and logged nowhere. It is now a `500` whose body says nothing
of the internals, and an `ERROR` line (and so a Sentry event) with the traceback.

The worker's runs have a dead-man's switch, off until it is given URLs (Fly secrets, both
optional; a free healthchecks.io check each is the shape they are written for):

| Secret                 | Pinged                                                                                                                                                  | Alerts when                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `WORKER_HEARTBEAT_URL` | `/start` when a job is claimed and every minute while it runs; success at the end; `/fail` with the reason when it is dead-lettered. Nothing while idle | a run fails, or goes quiet: the worker died, hung or was stopped and nobody resumed the job |
| `QUEUE_CHECK_URL`      | `/start` by the API when a job is queued and none is running; success by the worker when it claims one                                                  | a job is queued and never claimed: no worker, or one that cannot claim (a full disk)        |

**Set the two checks up like this** (healthchecks.io, Simple schedule), or they alert for the
wrong things. healthchecks.io keeps two clocks: the _period_, from the last success, and the
_grace_, which also runs from each `/start` — a run must end within the grace of its last
`/start`, and every `/start` starts it again.

| Check                  | Period  | Grace      | Why                                                                                                                                                   |
| ---------------------- | ------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WORKER_HEARTBEAT_URL` | 30 days | 5 minutes  | the keep-alive is a `/start` a minute, so a run that goes quiet alerts five minutes later; nothing is sent while idle, which must not alert for weeks |
| `QUEUE_CHECK_URL`      | 30 days | 10 minutes | a cold start: the machine boots and the worker claims; the long period because an idle deployment queues nothing                                      |

The keep-alive used to be a success ping, which only fed the period: a short period alerted
every time the worker sat idle, and one long enough to stay quiet through that missed a run
that went quiet for hours. A grace shorter than a deploy's handover of a running job (the next worker's claim
sends the next `/start`) alerts on every deploy mid-run.

A ping never blocks the worker and never fails a job (a thread each, a 5 s timeout,
errors logged and dropped). `fly.toml`'s `[[restart]]` restarts a worker that dies
(`on-failure`, ten tries); one that keeps dying is what the two checks report.

**What a job may cost.** `WORKER_JOB_COST_CAP_USD` (default 20, 0 turns it off) is a
ceiling on one job's `costUsd`, across stages and attempts: no GPU call starts past it, a
running one is cancelled when its cost would go over it, and the job is dead-lettered
saying so. `costUsd` on Modal includes the 2 cores and 8 GiB each GPU function reserves
(+$0.158/h on an L4); it does not include a container's idle minute after its last call.

## Handover

**Four steps.** Create accounts, mint tokens, paste secrets, dispatch. Everything else that
used to be on this list — creating the bucket, the Pages project, the Fly app, the volume,
the database; enabling PostGIS; rewriting the connection string; applying CORS; setting
eleven Fly secrets; seeding; publishing the tiles; checking it — is
`.github/workflows/provision.yml` now.

### 1. Create the accounts

| Account        | Needed for                    | Why no workflow can do it                                                                               |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Neon**       | Postgres 16 + PostGIS         | sign-up, email confirmation and plan acceptance are acts by a person; there is no API before one exists |
| **Cloudflare** | R2 and Pages                  | same, plus a payment method on the account before R2 will serve                                         |
| **Fly.io**     | the API and the worker        | same, and Fly requires a card on file before it will run a machine                                      |
| **Cesium ion** | terrain and imagery, optional | same. Skip it and the globe still loads, without ion terrain                                            |

Every free tier here is enough to stand this up. The reason none of it is automatable is not
technical: creating an account is agreeing to terms and attaching a means of payment, and a
CI runner cannot consent on your behalf.

### 2. Mint the tokens — with these scopes and no more

A token that lives in a CI secret is a token that is as powerful as its worst day, so each
of these is the narrowest thing that still works.

**Neon** — Account settings → API keys → Create API key.
_Scope:_ Neon's API keys are **account-scoped**; the console offers no per-project key. This
is the one credential on the list that cannot be narrowed, and it is worth knowing: it can
create and delete any project on that account. If that matters to you, make a Neon account
that holds only this project. → `NEON_API_KEY`

**Cloudflare** — My Profile → API Tokens → Create Token → Create Custom Token. Exactly two
permissions, both **Account**-level:

- `Workers R2 Storage` → **Edit** (create the bucket, enable its public URL)
- `Cloudflare Pages` → **Edit** (create the project, deploy to it)

Account Resources: **Include → your one account**. No Zone permissions at all — not DNS, not
Zone Settings, not Cache Purge — because using the managed `r2.dev` URL is precisely what
removes the need for them. → `CLOUDFLARE_API_TOKEN`

Note the account id while you are there: it is the 32-hex string in the dashboard URL.
→ `CLOUDFLARE_ACCOUNT_ID`

**Cloudflare R2, separately** — R2 → Manage API tokens → Create API token.
_Scope:_ **Object Read & Write**, and under "Specify bucket(s)" choose **only** the two
buckets this deployment uses (`twin-assets` and `twin-assets-public` unless you set the
`R2_BUCKET` / `R2_PUBLIC_BUCKET` variables): publishing copies from one into the other with
this pair. See [Narrowing the credentials](#narrowing-the-credentials-operator-steps) for
giving Modal a pair that reaches only the private one. This is an S3-compatible key pair, a different kind of credential from the token above, and there is no
API that mints one — which is why it is on this list. The API needs it in any case:
`OBJECT_STORAGE_ACCESS_KEY` and `OBJECT_STORAGE_SECRET_KEY` are what presign every upload.
→ `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`

If the bucket does not exist yet, create it in the R2 dashboard first so the token can be
scoped to it — or scope the token to the account, run **Provision** once, and then narrow
it. Provisioning creates the bucket either way.

**Fly.io** — `fly tokens create org <your-org>` (or Account → Access Tokens in the
dashboard).
_Scope:_ **org-scoped**, and this is a real widening over what `deploy.yml` alone needs. An
app-scoped deploy token (`fly tokens create deploy -a twin-api`) cannot create the app it is
scoped to, so provisioning cannot use one. Once the app exists you may narrow
`FLY_API_TOKEN` to the deploy token — `deploy.yml` is happy with it — at the cost of having
to widen it again to re-run **Provision**. → `FLY_API_TOKEN`

**Cesium ion** — Access Tokens → Create token, scopes `assets:read` and `geocode`, and set
its **Allowed URLs** to the Pages origin once you know it. It ships inside the bundle and is
readable by anyone who loads the page; the allow-list is the only thing that stops someone
else spending it. → `VITE_CESIUM_ION_ACCESS_TOKEN`

**And one you generate yourself:** `openssl rand -hex 32` → `API_WRITE_TOKEN`. Keep it. The
data console asks for it, the worker uses it, and a Fly secret cannot be read back.

### 3. Paste them into the repository

Settings → Secrets and variables → Actions. Seven secrets are required; the other five are
optional, and two of those five are the Modal pair, which is needed only if a GPU stage is
ever dispatched. [The table above](#secrets) says which is which, and the
[variables](#variables) are all optional — every one of them has a working default, so a
first run needs none of them. If you are going to use a custom domain for the tiles or for the web app,
set `R2_PUBLIC_URL` and `WEB_BASE_URL` now rather than after — changing them later means a
re-run and a rebuild.

### 4. Dispatch

Actions → **Provision** → Run workflow.

- Leave `dry_run` off unless you want a look first. With it on, nothing is created: the run
  reports what already exists and what it would make, and because every lookup is an
  authenticated read it is also the cheapest way to find out that one of your tokens is
  wrong.
- Leave `deploy` on. The run then provisions, deploys, seeds, publishes the tiles, and
  checks the result.

Read the job summary: it prints the API origin, the web origin, the bucket's public URL and
the two repository variables to set so that a **later, standalone** `deploy.yml` dispatch
builds the same bundle. From then on, ordinary deploys are Actions → **Deploy**.

Then look at, in this order:

1. The **verify** job's log. It is the only place the presigned upload and the CORS
   preflight are exercised, and it names exactly what is wrong when one of them is.
2. The web origin: the globe loads and the catalogue lists `synthetic-tree`.
3. `/admin.html`, with `API_WRITE_TOKEN` — create a capture and upload a small file from a
   real browser. Verification proves the API and the bucket agree; this proves a browser
   agrees with both of them.
4. `fly logs --app twin-api` — the worker claiming a `splat-ingest` job you queue.

### What a re-run does

Nothing, if nothing has changed. Every resource is looked up before it is created and every
create that loses a race to an identical one is treated as success:

| Resource         | How it is made idempotent                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neon project     | found by name; created only if absent. The branch, database and role are read back from the API either way, so both paths produce the same string |
| PostGIS          | `CREATE EXTENSION IF NOT EXISTS`                                                                                                                  |
| R2 bucket        | listed by name; on a create, an "already exists" error is accepted as success                                                                     |
| R2 public URL    | read first, enabled only when it is off; skipped entirely when `R2_PUBLIC_URL` is set                                                             |
| R2 CORS          | `put-bucket-cors` replaces the document wholesale — the API's own semantics                                                                       |
| Pages project    | listed by name; on a create, an "already exists" error is followed by a fetch of the existing project                                             |
| Fly app          | `fly apps list` first                                                                                                                             |
| **Fly volume**   | **counted**, not merely looked for: zero creates one, one creates none, and more than one stops the run rather than adding to the pile            |
| Fly secrets      | set to the same values; `API_HANDOFF_SECRET` is generated only when the app does not already have it                                              |
| Seed and publish | both are upserts over slugs and keys                                                                                                              |

The volume is the one where a duplicate is not an error but a bill — a second 20 GB volume
is charged monthly and nothing would ever mount it — which is why it is the only resource
the workflow refuses to proceed past when it finds the count wrong. It will not clean that
up for you: destroy the spare with `fly volumes destroy <id>` and re-run.

The one thing a re-run does _not_ leave alone: it deploys again, and verification adds
another `Provisioning check <run id>` capture.

### If provisioning half-succeeds

Re-run it. That is the whole procedure, and it is the reason for all of the above.

The jobs are ordered so that a failure stops everything downstream — a Neon that will not
answer means no Fly secrets get set, not a Fly app configured against a database that does
not exist — so the state a failed run leaves behind is a prefix of the state a successful
one would have, and re-running fills in the rest. A few specifics worth knowing:

- **It failed in `infra`.** Nothing downstream ran. Fix the cause and re-run; everything
  already created is found rather than remade.
- **It failed in `deploy`.** The infrastructure is complete. `fly.toml`'s `release_command`
  runs the migrations before any machine is replaced, so a failed migration leaves the old
  version serving and nothing half-applied. Re-run **Provision**, or just **Deploy** —
  but if you dispatch **Deploy** on its own, set `VITE_API_BASE_URL` and
  `VITE_OFFLINE_CATALOG_URL` first or the bundle is built pointing at nothing.
- **It failed in `seed` or `verify`.** Everything is deployed; what is missing is the
  catalogue or your confidence in it. Re-running **Provision** is safe and cheap.
- **`fly volumes` reports two volumes.** Provisioning stops there by design. Destroy one.
- **The `r2.dev` URL could not be enabled.** Enable it by hand under R2 → your bucket →
  Settings → Public access, or connect a custom domain and set `R2_PUBLIC_URL`. Then re-run.

### What is still a human's job, and why

Honestly, and not quietly left on the list:

- **Creating the four accounts.** Consent, identity and payment. No API precedes them.
- **Minting the five tokens.** Every provider requires interactive authentication to issue a
  credential, on purpose. A workflow that could mint its own tokens would be a workflow that
  could escalate itself.
- **Choosing and connecting a custom domain**, if you want one. The DNS side is a change on
  a zone this deployment has no permission over, and asking for that permission would widen
  `CLOUDFLARE_API_TOKEN` from two account permissions to include a zone. Using the managed
  `r2.dev` URL and the `pages.dev` origin is what keeps the token narrow.
- **Setting the ion token's Allowed URLs** to the Pages origin, once you know it. The ion
  API can do this; this deployment deliberately holds no ion credential that is allowed to.
- **Setting the two repository variables** the summary prints, if you want standalone
  `deploy.yml` dispatches to work. Writing a repository variable from a workflow needs a
  personal access token with administrative scope on the repository, which is a much larger
  credential than the two lines of copying it would save.
- **Reading the verify job.** Nothing else in this project has ever spoken to a provider.

## What is unverified

Undersold on purpose. Most of this list was written before production existed. The
operator has since deployed it, but nothing here has observed that deployment. So each item
below is still true of this repository's evidence, whatever it now says about the world:

- **The GPU path has never run.** Everything below is unproven until
  `.github/workflows/modal.yml` passes once:
  - the Modal image has never been built by Modal;
  - `ModalAdapter` has never spoken to a live workspace;
  - no gsplat training run has happened.

  What _is_ verified:
  - the App builds locally and in CI;
  - gsplat v1.5.3's own `simple_trainer.py` CLI accepts the pipeline's argv, in a CPU
    replica of the image's venv (CI's `trainer` job);
  - every pinned artifact was resolved, and the gsplat wheel is pinned by sha256;
  - the whole smoke flow passes locally with `--rehearse`.

  Two things only the first build and run can prove: that the CUDA image compiles
  fused-ssim, and that `modal_adapter.GPU_NAMES` names tiers Modal accepts. An App with a
  wrong GPU name builds fine locally and is refused only server-side.

- **The worker image's COLMAP has not run on Fly.** CI builds the image and asserts the
  version. Pose on a real capture on a Fly machine is an extrapolation from the timings in
  `tools/pipeline/README.md`, measured on a 4-core development container.
- **Handover commands were written from documentation.** Every command in the handover was
  written from the tools' documented interfaces, and none has been run from this
  repository.
- **Waking the worker has not met Fly's API.** `app/services/worker_wake.py` is written
  against the Machines API's documented shapes (`GET /v1/apps/{app}/machines`,
  `config.metadata.fly_process_group`, `POST …/machines/{id}/start`, a `FlyV1 …` token in
  `Authorization`) and tested against a stand-in for it. That a deploy token may start
  machines, that a stopped worker comes back with its volume, and Fly's suspend of the
  1 GB `app` machine have not been observed from here. `QUEUE_CHECK_URL`'s worker half
  (the ping on claim) is a separate change.
- **`provision.yml` has never run, and could not have been tested from where it was
  written.** `api.cloudflare.com`, `api.fly.io`, `api.neon.tech` and `registry.fly.io` are
  all unreachable from that environment — which is the workflow's whole premise, and also
  why not one of its requests has ever been sent. Its API shapes come from the providers'
  documented interfaces, and those documentation sites were unreachable too. In rough order
  of how sure they are:

  | Call                                                                  | Confidence                                                                                                                                                                                                                                                                                                                 |
  | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `GET/POST https://console.neon.tech/api/v2/projects`                  | reasonable — a stable, widely-used endpoint                                                                                                                                                                                                                                                                                |
  | Neon's branches / databases / `connection_uri` chain                  | moderate. `connection_uri` takes `branch_id`, `database_name` and `role_name`; whether all three are required is not certain, and the branch is picked by a `default == true` field with a first-branch fallback                                                                                                           |
  | `psql … CREATE EXTENSION postgis` on Neon                             | moderate. Asserted, not evidenced, that Neon's owner role may install PostGIS without the console                                                                                                                                                                                                                          |
  | `GET/POST /accounts/{id}/r2/buckets`                                  | reasonable. The list response is read as `.result.buckets` with a bare-array fallback                                                                                                                                                                                                                                      |
  | **`PUT /accounts/{id}/r2/buckets/{name}/domains/managed`**            | **least sure of anything here.** The path, the `{"enabled": true}` body and reading the hostname back out of `.result.domain` are all reconstructed from memory of Cloudflare's R2 API, not read. If one call in this file is wrong, expect it to be this one — and it fails with an instruction to enable the URL by hand |
  | `GET/POST /accounts/{id}/pages/projects`                              | reasonable, including `.result.subdomain`                                                                                                                                                                                                                                                                                  |
  | `flyctl apps create` / `volumes create --yes` / `secrets set --stage` | moderate. `--stage` and `--yes` are documented flags but were not run, and flyctl's `--json` field casing has changed across versions, which is why the jq accepts either                                                                                                                                                  |
  | `aws s3api put-bucket-cors` against R2                                | unchanged from C1: still the first thing that will meet the real R2 API                                                                                                                                                                                                                                                    |

- **The bucket split is code and documentation, not a deployed fact.** The exposure it
  removes is verified — Cloudflare's public-bucket page was read on 2026-09-22 and says a
  public bucket exposes "the contents of their R2 buckets", with no prefix scoping, and
  documents `r2.dev` as rate-limited and "intended for non-production traffic". What is
  unverified is the other side: no R2 account has two buckets, no `CopyObject` has crossed
  between them, and `provision.yml`'s two-bucket path has run exactly as often as its
  one-bucket path did, which is never. `apps/api/tests/test_publish_bucket.py` proves the
  split against moto, which is layout and not authorisation.
- **The four-step handover is a claim, not a measurement.** Nobody has followed it. The step
  count is honest about what the workflow attempts, not about what it achieves.
- **Cloudflare Containers.** The claims in ADR 0007 about GA, sleep-on-idle and the Durable
  Object entrypoint are second-hand; `developers.cloudflare.com` is unreachable from here.
- **`fly.toml` has never been parsed by flyctl.** CI parses it with `tomllib` and checks the
  commands it names exist in the image, and that is all. Key names, `auto_stop_machines`
  accepting a string, `initial_size` on a mount, per-process `[[vm]]` blocks: all from Fly's
  documented schema, none exercised.
- **`wrangler.toml` has never been read by wrangler.** If it is rejected, deleting it and
  passing the directory positionally (`wrangler pages deploy apps/web/dist --project-name
twin-web`) is the equivalent.
- **`infra/pages/_headers` has never been served by Pages.** The syntax is Pages'; the
  effect has not been observed.
- **The R2 `AllowedHeaders` question is still open**, as flagged above.
- **Neither workflow has ever run.** Not once, not with dummy credentials. What CI does
  check, every run, is that both files parse as YAML, that neither has grown a `push`
  trigger, and that every `secrets.*` and `vars.*` name they read appears in this document.
- **The r2.dev URL makes the whole bucket public**, `captures/` included. That is stated
  above rather than discovered later, but it is a property of this design that nobody has
  seen in operation either.
- What _is_ verified, on every CI run: the image builds, the API in it is healthy against a
  real Postgres, the recipe catalogue lists both recipes, the pipeline imports in the image's
  venv, the worker starts and exits cleanly, a production configuration starts, a
  misconfigured one refuses to, and every command `fly.toml` names exists in the image.
