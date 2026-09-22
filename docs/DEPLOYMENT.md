# Deployment

**Nothing here has been deployed.** No account exists, no `fly deploy` has run, no bucket
has been created, and no `wrangler pages deploy` has been made. What this document
describes is what the repository now _codifies_ — every row of the table below has a file
behind it — plus, at the end, a [handover](#handover) listing the things only the account
owner can do. Read the [what is unverified](#what-is-unverified) section before trusting any
provider-specific claim in here.

| Layer                           | Where                        | Codified in                                            |
| ------------------------------- | ---------------------------- | ------------------------------------------------------ |
| Web bundle                      | Cloudflare Pages             | `wrangler.toml`, `infra/pages/_headers`                |
| API (`app` process)             | Fly.io                       | `fly.toml`                                             |
| Worker (`worker` process)       | Fly.io, same image           | `fly.toml` (`[processes]`, `[[mounts]]`)               |
| Captures, tiles, `catalog.json` | Cloudflare R2                | `infra/cors/production.json`, `infra/cors/apply-r2.sh` |
| Database                        | Neon (Postgres 16 + PostGIS) | nothing: a connection string, set as a secret          |
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
workflow, where the `VITE_*` values live as repository secrets and variables.

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
fly deploy --remote-only        # from the repository root: the build context is the repo
```

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
if the role is allowed to; on a managed service, enable PostGIS in the console first. The
connection string must use the `postgresql+psycopg://` scheme (SQLAlchemy's driver
selector), which is not the scheme the provider hands you — rewrite it.

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

- **Presigning happens on the S3 API domain; public reads come from the custom domain.**
  That is why `OBJECT_STORAGE_ENDPOINT_URL` and `OBJECT_STORAGE_PUBLIC_URL` are separate
  settings. Get this wrong and uploads work while every tile 404s, or the reverse.
- **R2 wants `region="auto"`.** `fly.toml` sets `OBJECT_STORAGE_REGION=auto` for you.

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

## Deploying from GitHub Actions

`.github/workflows/deploy.yml` is **`workflow_dispatch` only**, and deliberately: it needs
credentials that do not exist in this repository, and a `push` trigger would paint every
commit on main red for want of a secret and teach everyone to ignore the badge. Run it from
the Actions tab, choosing `everything`, `api` or `web`.

Each job checks its own secrets in its first step and fails with the name of the missing one
rather than inside a CLI's auth error forty lines later. It uses no third-party actions:
flyctl comes from Fly's own installer, wrangler from npm.

| Name                           | Kind     | Used by                                        |
| ------------------------------ | -------- | ---------------------------------------------- |
| `FLY_API_TOKEN`                | secret   | `fly deploy`                                   |
| `CLOUDFLARE_API_TOKEN`         | secret   | `wrangler pages deploy`                        |
| `CLOUDFLARE_ACCOUNT_ID`        | secret   | `wrangler pages deploy`                        |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | secret   | the web build                                  |
| `VITE_API_BASE_URL`            | variable | the web build, and the API's post-deploy check |
| `VITE_OFFLINE_CATALOG_URL`     | variable | the web build                                  |
| `VITE_ENABLE_PHOTOREALISTIC`   | variable | the web build                                  |

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

Everything below needs an account, a card or a DNS record, and **no committed file can do
any of it**. In order. Each step says what it produces, because the next one usually needs
it.

### 1. Database — Neon

1. Create an account at neon.tech and a project (Postgres 16, region near Fly's `iad`).
2. In the SQL editor: `CREATE EXTENSION IF NOT EXISTS postgis;`
3. Copy the connection string and **rewrite its scheme** to `postgresql+psycopg://`. Keep
   `?sslmode=require` if it is there.

Produces: `DATABASE_URL`.

### 2. Object storage — Cloudflare R2

1. Create a Cloudflare account; note the **account ID** from the dashboard URL.
2. R2 → create a bucket, `twin-assets`. One bucket does both roles.
3. R2 → Manage API tokens → create a token with **Object Read & Write** on that bucket.
   It gives you an access key id, a secret, and the S3 endpoint
   `https://<account-id>.r2.cloudflarestorage.com`.
4. Give the bucket a **public custom domain** (R2 → Settings → Public access → Connect
   domain, e.g. `tiles.example.com`). This is a DNS change on a domain you control, and it
   is a different host from the S3 endpoint. Presigning uses the endpoint; browsers read
   from the custom domain.
5. Apply CORS once the web origin is known (after step 4 below):

   ```bash
   AWS_ACCESS_KEY_ID=…  AWS_SECRET_ACCESS_KEY=… \
     infra/cors/apply-r2.sh <account-id> twin-assets https://<your-pages-origin>
   ```

   Read what it prints back, and correct the unverified note above with what R2 actually
   accepted.

Produces: `OBJECT_STORAGE_ENDPOINT_URL`, `OBJECT_STORAGE_BUCKET`,
`OBJECT_STORAGE_ACCESS_KEY`, `OBJECT_STORAGE_SECRET_KEY`, `OBJECT_STORAGE_PUBLIC_URL`,
`CLOUDFLARE_ACCOUNT_ID`.

### 3. API and worker — Fly.io

```bash
fly auth login
fly apps create twin-api          # the name in fly.toml; change both if it is taken
fly volumes create twin_worker_data --size 50 --region iad --app twin-api

fly secrets set --app twin-api \
  DATABASE_URL='postgresql+psycopg://…' \
  API_WRITE_TOKEN="$(openssl rand -hex 32)" \
  API_HANDOFF_SECRET="$(openssl rand -hex 32)" \
  API_CORS_ORIGINS='https://<your-pages-origin>' \
  PUBLIC_API_BASE='https://twin-api.fly.dev' \
  PUBLIC_WEB_BASE='https://<your-pages-origin>' \
  OBJECT_STORAGE_ENDPOINT_URL='https://<account-id>.r2.cloudflarestorage.com' \
  OBJECT_STORAGE_BUCKET='twin-assets' \
  OBJECT_STORAGE_ACCESS_KEY='…' \
  OBJECT_STORAGE_SECRET_KEY='…' \
  OBJECT_STORAGE_PUBLIC_URL='https://tiles.example.com'

fly deploy --remote-only
```

Write down `API_WRITE_TOKEN`: you cannot read a Fly secret back, and the console and the
worker both need it.

The volume must exist before the first deploy of the `worker` group, and it must be in the
same region. `fly status` should show one `app` machine and one `worker` machine; `fly logs`
should show the worker polling.

Then, once:

```bash
fly ssh console --app twin-api -C "python -m app.seed"
```

Produces: the API's origin, and `FLY_API_TOKEN` via
`fly tokens create deploy -a twin-api`.

### 4. Web — Cloudflare Pages

```bash
pnpm dlx wrangler pages project create twin-web --production-branch main
```

That gives you the `*.pages.dev` origin. Point a custom domain at it if you want one; the
origin you end up with is what goes in `API_CORS_ORIGINS`, `PUBLIC_WEB_BASE`, the ion
token's Allowed URLs and the bucket's CORS rule — four places, all of which reject a
mismatch silently or loudly.

Create a Cloudflare API token (My Profile → API Tokens) with **Cloudflare Pages: Edit** on
your account.

Produces: `CLOUDFLARE_API_TOKEN`, the web origin.

### 5. Cesium ion

Create a token with scopes `assets:read` and `geocode`, and set its **Allowed URLs** to the
Pages origin. It ships inside the bundle; the allow-list is what stops someone else
spending it.

Produces: `VITE_CESIUM_ION_ACCESS_TOKEN`.

### 6. GitHub

Repository → Settings → Secrets and variables → Actions. Add the secrets and variables in
[the table above](#deploying-from-github-actions). Then run the **Deploy** workflow from the
Actions tab.

### 7. Publish the tiles, from a checkout that has them

```bash
cd apps/api && uv run python -m app.seed.publish
```

The repository carries only `data/tiles/synthetic-tree`; the three drone captures left git
in A9 and their bytes are in history. Publishing what you have is what makes those captures
load in the deployed app.

### 8. Check it, in this order

1. `curl https://twin-api.fly.dev/api/v1/health` — `{"status":"ok","database":true}`.
2. `curl https://twin-api.fly.dev/api/v1/recipes` — both recipes listed. A 404 here means
   the image lost `tools/pipeline`.
3. Open the web origin: the globe loads, the catalogue lists sites.
4. Open `/admin.html`, create a capture, and upload a small file. **This is the step that
   exercises the CORS rule and presigning together** — if `ExposeHeaders: ["ETag"]` is
   missing or the endpoint/public-URL pair is crossed, it fails here and nowhere earlier.
5. Run a `splat-ingest` job and watch `fly logs` for the worker claiming it.

## What is unverified

Undersold on purpose, because none of it has been deployed:

- **No deployment exists.** No Fly app, no Pages project, no R2 bucket, no Neon database, no
  account. Every command in the handover was written from the tools' documented interfaces,
  not run.
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
- **The deploy workflow has never run.** Not once, not with dummy credentials.
- What _is_ verified, on every CI run: the image builds, the API in it is healthy against a
  real Postgres, the recipe catalogue lists both recipes, the pipeline imports in the image's
  venv, the worker starts and exits cleanly, a production configuration starts, a
  misconfigured one refuses to, and every command `fly.toml` names exists in the image.
