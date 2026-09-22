# Deployment

**Nothing here has been deployed.** No account exists, no `fly deploy` has run, no bucket
has been created, and no `wrangler pages deploy` has been made. What this document
describes is what the repository now _codifies_ — every row of the table below has a file
behind it — plus, at the end, a [handover](#handover) which is now four steps long: create
accounts, mint tokens, paste secrets, dispatch. Everything after a token exists is
`.github/workflows/provision.yml`. Read the
[what is unverified](#what-is-unverified) section before trusting any provider-specific
claim in here; none of it has run.

| Layer                           | Where                        | Codified in                                            |
| ------------------------------- | ---------------------------- | ------------------------------------------------------ |
| Web bundle                      | Cloudflare Pages             | `wrangler.toml`, `infra/pages/_headers`                |
| API (`app` process)             | Fly.io                       | `fly.toml`                                             |
| Worker (`worker` process)       | Fly.io, same image           | `fly.toml` (`[processes]`, `[[mounts]]`)               |
| Captures, tiles, `catalog.json` | Cloudflare R2                | `infra/cors/production.json`, `infra/cors/apply-r2.sh` |
| Database                        | Neon (Postgres 16 + PostGIS) | `.github/workflows/provision.yml`                      |
| Provisioning                    | GitHub Actions, on demand    | `.github/workflows/provision.yml`                      |
| Deploys                         | GitHub Actions, on demand    | `.github/workflows/deploy.yml`                         |

Why Fly for the compute and Cloudflare for the edge, when consolidating on one provider was
on the table: [ADR 0007](DECISIONS/0007-fly-for-the-api-cloudflare-for-the-edge.md). The
short version is the worker: it is a poll loop that supervises hours-long runs and renews a
lease every two seconds, and a sleep-on-idle runtime has nowhere to put that.

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
the globe on first load; and `/cesium/*` is _not_ `immutable`, because `vite.config.ts`
copies Cesium's static directories to fixed, unhashed paths, so upgrading the dependency
changes the contents of URLs that keep their names.

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
- **`auto_stop_machines = "stop"`, `min_machines_running = 0`** on the `app` group only.
  A single-user API idles for hours; the worker must never be stopped by a proxy that sees
  no requests, and it is not in `http_service.processes` for that reason.
- **a health check** on `GET /api/v1/health`.
- **`kill_signal = "SIGTERM"`, `kill_timeout = "30s"`.** The worker treats SIGTERM as a stop
  flag: on its next tick it stops the recipe process, clears the lease on the job it holds
  and exits, so the next worker can take that job immediately instead of waiting the lease
  out. 30 s is fourteen ticks of headroom. (Both keys are at the _top_ of `fly.toml`,
  before any table header. TOML gives a bare key to whichever table precedes it, so written
  next to the health check — where they read most naturally — they silently become fields of
  that check and Fly never sees them. CI asserts they are top-level.)
- **a 50 GB volume** at `/data`, on the `worker` group only, with
  `WORKER_WORKDIR=/data/worker`. See [the workdir](#the-workers-workdir).

### Configuration

`fly.toml`'s `[env]` carries only what is true of any account: `APP_ENV=production`,
`ALLOW_PRIVATE_URLS=false`, `API_PORT`, `OBJECT_STORAGE_REGION=auto`, `WORKER_WORKDIR`.
Everything account-specific — including things that are not strictly secret, like your own
domain names — goes through `fly secrets set`, so nothing about your deployment is
committed.

| Secret                                                    | Notes                                                                   |
| --------------------------------------------------------- | ----------------------------------------------------------------------- |
| `DATABASE_URL`                                            | `postgresql+psycopg://…` with PostGIS enabled                           |
| `API_WRITE_TOKEN`                                         | **required**: the shared token every write must carry                   |
| `API_HANDOFF_SECRET`                                      | signing key for phone-handoff tokens; set it explicitly once >1 process |
| `API_CORS_ORIGINS`                                        | your Pages origin, exact scheme + host, comma-separated                 |
| `OBJECT_STORAGE_ENDPOINT_URL`                             | `https://<account-id>.r2.cloudflarestorage.com` (the S3 API domain)     |
| `OBJECT_STORAGE_BUCKET`                                   | one bucket; uploads under `captures/`, published tiles under `sites/`   |
| `OBJECT_STORAGE_ACCESS_KEY` / `OBJECT_STORAGE_SECRET_KEY` | the R2 API token's pair                                                 |
| `OBJECT_STORAGE_PUBLIC_URL`                               | the bucket's **public** base — its custom domain, not the S3 API domain |
| `TILES_BASE_URL`                                          | optional: a CDN prefix in front of the published tiles                  |
| `PUBLIC_API_BASE`                                         | the API's own public origin, used when seeding absolute tileset URLs    |
| `PUBLIC_WEB_BASE`                                         | the web origin, for the handoff URL a QR code encodes                   |
| `CESIUM_ION_SERVER_TOKEN`                                 | optional; `assets:read` for job monitoring. Never a `VITE_` variable    |
| `ANTHROPIC_API_KEY`                                       | optional; without it the plan drafter is rule-based and says so         |

### Production refuses to start when it cannot do its job

Two startup guards, both in `app/main.py`, both deliberate:

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
contract is worth nothing once the directory is gone. `fly.toml` therefore mounts a 50 GB
volume at `/data` on the worker group and sets `WORKER_WORKDIR=/data/worker`.

**If your host cannot give the worker a volume**, the deployment still works and nothing is
corrupted — a worker that dies mid-run leaves a lease that lapses, and A7's reclaim hands
the job to the next worker, which starts the recipe over. What you lose is the work: every
completed stage is recomputed, and a Lane 2 run that was two hours into `train` pays those
two hours again. You also need a root disk big enough for the largest capture plus the
image. Size the machine accordingly and treat restarts as expensive.

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

## Object storage and CORS — R2

Large assets never transit the API: the browser PUTs to presigned S3 multipart URLs and
CesiumJS reads tiles straight from the bucket. Both are cross-origin, so the bucket's CORS
configuration is part of the deployment, not an afterthought.

**One bucket, in both roles.** The API has a single `OBJECT_STORAGE_BUCKET`: uploads land
under `captures/`, published tiles under `sites/`. A bucket has exactly **one** CORS
configuration and setting it replaces what was there, so applying `upload.json` and then
`tiles.json` to that bucket would leave only the second — and browser multipart uploads
would stop being completable the moment you did it. Production gets one combined document:

| File                         | Role                                                      |
| ---------------------------- | --------------------------------------------------------- |
| `infra/cors/upload.json`     | browser PUTs to presigned URLs                            |
| `infra/cors/tiles.json`      | CesiumJS reading `tileset.json` and `.glb`                |
| `infra/cors/production.json` | **both**, for the single production bucket                |
| `infra/cors/dev-minio.xml`   | both rules in one XML document, for the single dev bucket |

`ExposeHeaders: ["ETag"]` in the upload rule is load-bearing. Without it the browser reads
`etag === null` from each part's response and a multipart upload can never be completed.
`apps/api/tests/test_cors_rules.py` fails if it is dropped, if the XML and JSON documents
drift apart, or if `production.json` stops matching the two rules it combines.

Apply it:

```bash
AWS_ACCESS_KEY_ID=…  AWS_SECRET_ACCESS_KEY=… \
  infra/cors/apply-r2.sh <account-id> twin-assets https://twin.example.com
```

The script substitutes your real origin into the committed placeholder (so the origin never
has to be committed), applies the document with `aws s3api put-bucket-cors` against
`https://<account-id>.r2.cloudflarestorage.com` with `--region auto`, and then prints what
the bucket reports back. **Read that output.** It is the first time any of this meets a real
R2 API.

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

| Name                           | Required             | What to put in it                                                         |
| ------------------------------ | -------------------- | ------------------------------------------------------------------------- |
| `NEON_API_KEY`                 | yes                  | a Neon API key (see the scopes below)                                     |
| `CLOUDFLARE_API_TOKEN`         | yes                  | a custom Cloudflare token, two permissions                                |
| `CLOUDFLARE_ACCOUNT_ID`        | yes                  | the 32-hex id in your Cloudflare dashboard URL                            |
| `R2_ACCESS_KEY_ID`             | yes                  | the R2 **S3** access key id — a different credential from the token above |
| `R2_SECRET_ACCESS_KEY`         | yes                  | its secret                                                                |
| `FLY_API_TOKEN`                | yes                  | an **org-scoped** Fly token                                               |
| `API_WRITE_TOKEN`              | yes                  | `openssl rand -hex 32`, generated by you and kept                         |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | strongly recommended | the browser ion token; without it the globe has no ion terrain or imagery |
| `CESIUM_ION_SERVER_TOKEN`      | no                   | `assets:read`, for server-side asset monitoring. Never a `VITE_` variable |
| `ANTHROPIC_API_KEY`            | no                   | without it the plan drafter is rule-based and says so                     |

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

| Name                         | Default             | What it is for                                                           |
| ---------------------------- | ------------------- | ------------------------------------------------------------------------ |
| `VITE_API_BASE_URL`          | _(passed in)_       | the API origin the bundle is built against                               |
| `VITE_OFFLINE_CATALOG_URL`   | _(passed in)_       | `catalog.json`'s public URL                                              |
| `VITE_ENABLE_PHOTOREALISTIC` | on                  | `false` unless your ion token has Google access and you accept the terms |
| `R2_BUCKET`                  | `twin-assets`       | the bucket's name                                                        |
| `R2_PUBLIC_URL`              | the `r2.dev` URL    | set it to a custom domain to take the managed URL out of the picture     |
| `WEB_BASE_URL`               | the `pages.dev` URL | set it to a custom domain on the Pages project                           |
| `NEON_PROJECT_NAME`          | `hexapod-twin`      | which Neon project to find or create                                     |
| `NEON_REGION_ID`             | `aws-us-east-1`     | Neon's name for the region `fly.toml`'s `primary_region` is in           |
| `FLY_ORG`                    | `personal`          | the Fly organization to create the app in                                |

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
  `Authorization: Bearer $API_WRITE_TOKEN`, and the worker uses the same token.

## Observability

`apps/web/src/lib/log.ts` and `lib/timing.ts` expose sinks for a vendor (Sentry,
OpenTelemetry). The API adds a `Server-Timing` header and logs slow requests to stdout;
`fly logs` is the shipper until there is a reason for another.

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
_Scope:_ **Object Read & Write**, and under "Specify bucket(s)" choose **only** the bucket
this deployment uses (`twin-assets` unless you set the `R2_BUCKET` variable). This is an
S3-compatible key pair, a different kind of credential from the token above, and there is no
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

Settings → Secrets and variables → Actions. Seven secrets are required and three are
optional; [the table above](#secrets) says which is which and the [variables](#variables)
are all optional. If you are going to use a custom domain for the tiles or for the web app,
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

The volume is the one where a duplicate is not an error but a bill — a second 50 GB volume
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

Undersold on purpose, because none of it has been deployed:

- **No deployment exists.** No Fly app, no Pages project, no R2 bucket, no Neon database, no
  account. Every command in the handover was written from the tools' documented interfaces,
  not run.
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
