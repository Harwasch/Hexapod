# Deployment

```text
Web       static bundle (Vercel / Netlify / S3+CloudFront / any static host)
API       container (Fly.io, Cloud Run, ECS, Railway…) built from infra/api.Dockerfile
Database  managed PostgreSQL with PostGIS (Neon, Supabase, RDS, Cloud SQL…)
Assets    Cesium ion (today); S3-compatible object storage (MinIO in dev, R2 in prod)
```

The frontend never serves large 3D assets; the browser streams them from Cesium ion or the
tileset host you registered.

## Web

```bash
pnpm install --frozen-lockfile
pnpm --filter @twin/web build          # apps/web/dist
```

Deploy `apps/web/dist` as a static site with an SPA fallback to `index.html`. Build-time
environment (all public):

| Variable                       | Production value                                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | your token, scopes `assets:read` + `geocode`, Allowed URLs = your origin            |
| `VITE_API_BASE_URL`            | `https://api.example.com` (absolute; the dev proxy is dev-only)                     |
| `VITE_ENABLE_PHOTOREALISTIC`   | on by default; `false` unless your token has Google access and you accept the terms |
| `VITE_ENABLE_DEV_TOOLS`        | leave unset/false                                                                   |
| `VITE_DEFAULT_*_ASSET_ID`      | optional                                                                            |

Vercel: framework preset "Vite", root `apps/web`, install command `pnpm install`, build
command `pnpm --filter @twin/web build`, output `apps/web/dist`. Cesium's static workers
and assets are copied into `dist/cesium/` by the build; no extra configuration.

## API

```bash
docker build -f infra/api.Dockerfile -t twin-api .
docker run -p 8000:8000 --env-file .env twin-api
```

The container runs `alembic upgrade head` then uvicorn. Seed the catalog once:
`docker run --env-file .env twin-api python -m app.seed`.

Required environment:

| Variable                   | Notes                                                                |
| -------------------------- | -------------------------------------------------------------------- |
| `DATABASE_URL`             | `postgresql+psycopg://user:pass@host:5432/db` with PostGIS enabled   |
| `API_CORS_ORIGINS`         | comma-separated browser origins (exact scheme + host)                |
| `APP_ENV=production`       | tightens URL validation (`ALLOW_PRIVATE_URLS=false` recommended)     |
| `API_WRITE_TOKEN`          | **required in production**: the shared token every write must carry  |
| `ALLOW_PRIVATE_URLS=false` | reject loopback/private dataset hosts                                |
| `OBJECT_STORAGE_*`         | endpoint, bucket, keys, region, public URL. See Object storage below |
| `CESIUM_ION_SERVER_TOKEN`  | optional; `assets:read` for job monitoring. Never a `VITE_` variable |

Put the API behind TLS (platform load balancer). It exposes `/api/v1/health` for probes and
`/api/v1/docs` for OpenAPI; disable docs at the edge if you prefer.

Reads are open so the world stays viewable; every mutating endpoint requires
`Authorization: Bearer $API_WRITE_TOKEN`, and the worker uses the same token. With no token
set writes are open, which is how a local checkout runs — so the API refuses to start when
`APP_ENV=production` and `API_WRITE_TOKEN` is empty rather than serving open writes.

## Object storage and CORS

Large assets never transit the API: the browser PUTs to presigned S3 multipart
URLs and CesiumJS reads tiles straight from the bucket. Both are cross-origin, so
the bucket's CORS configuration is part of the deployment, not an afterthought.

The rules live in `infra/cors/`:

| File            | Role                                                  |
| --------------- | ----------------------------------------------------- |
| `upload.json`   | browser PUTs to presigned URLs                        |
| `tiles.json`    | CesiumJS reading `tileset.json` and `.glb`            |
| `dev-minio.xml` | both rules in one document, for the single dev bucket |

Replace `https://twin.example.com` in `AllowedOrigins` with your real origin
before applying. A bucket has exactly **one** CORS configuration and setting it
replaces what was there, so a bucket in both roles gets a single document
containing both rules.

`ExposeHeaders: ["ETag"]` in the upload rule is load-bearing. Without it the
browser reads `etag === null` from each part's response and a multipart upload
can never be completed. `apps/api/tests/test_cors_rules.py` fails if it is
dropped, or if the XML and JSON documents drift apart.

### MinIO

`pnpm infra:up` applies `dev-minio.xml` automatically via the `minio-init`
one-shot. To apply it by hand, or to a MinIO you run elsewhere:

```bash
mc alias set local http://localhost:9000 twin twin-secret
mc cors set local/twin-assets infra/cors/dev-minio.xml   # mc takes XML, not JSON
```

If your `mc` predates the `cors` subcommand (added in 2024), use the S3 API
instead -- it is the same call the AWS CLI makes:

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

### Cloudflare R2

R2 is S3-compatible, so the JSON documents apply with the AWS CLI against the
account's S3 API endpoint:

```bash
aws s3api put-bucket-cors \
  --endpoint-url "https://<account-id>.r2.cloudflarestorage.com" \
  --bucket twin-assets \
  --cors-configuration file://infra/cors/upload.json

aws s3api put-bucket-cors \
  --endpoint-url "https://<account-id>.r2.cloudflarestorage.com" \
  --bucket twin-tiles \
  --cors-configuration file://infra/cors/tiles.json
```

Two R2 specifics to confirm at deploy time, both of which apply to `presign` as
much as to CORS:

- **Presigning happens on the S3 API domain; public reads come from the custom
  domain.** That is why `OBJECT_STORAGE_ENDPOINT_URL` and
  `OBJECT_STORAGE_PUBLIC_URL` are separate settings.
- **R2 wants `region="auto"`.** Set `OBJECT_STORAGE_REGION=auto`.

> **Unverified, and deliberately flagged as such.** It has been reported that R2
> rejects a narrow `AllowedHeaders: ["content-type"]` rule and requires `["*"]`.
> We could not confirm this: `developers.cloudflare.com` is unreachable from the
> environment these rules were written in. The rules therefore use `["*"]`, which
> works either way. What _is_ established is that the narrow form is shape-valid
> to botocore, so nothing in CI or in the test suite would catch R2 rejecting it
> -- it would first fail at deploy. **Confirm the accepted form against the real
> R2 API before trusting either shape**, and correct this note.

### Presigned URLs are SigV4, explicitly

`S3Storage` passes `signature_version="s3v4"` explicitly. This is not redundant:
with the default, botocore's `_default_s3_presign_to_sigv2` makes
`generate_presigned_url` emit **SigV2** for every region except `auto`, while
`client.meta.config.signature_version` still reports `s3v4`. MinIO in dev
(`us-east-1`) would have presigned SigV2 while R2 in production (`auto`)
presigned SigV4 -- a dev/prod split no configuration inspection can see. The
tests assert on the emitted URL string for that reason, and CI runs them against
a real MinIO.

## Database

Any PostgreSQL 16 with `CREATE EXTENSION postgis`. The first migration creates the
extension if the role is allowed to; on managed services enable PostGIS in the console
first. Back up with the provider's tooling; the schema is small and the heavy data lives in
ion/object storage.

## Security checklist

- No secrets in git; `.env` is ignored, `.env.example` is the template.
- The browser ion token has the narrowest scopes and an origin allow-list.
- The API is not a fetch proxy: the only outbound call is the documented ion asset
  metadata read, gated by a server token.
- CORS is explicit; only `GET/POST/PATCH/DELETE`.
- `pnpm audit --audit-level high` and `pip-audit` run in CI.
- Dataset URLs are validated (scheme, credentials, private hosts in production).

## Observability

`apps/web/src/lib/log.ts` and `lib/timing.ts` expose sinks for a vendor (Sentry, OpenTelemetry).
The API adds a `Server-Timing` header and logs slow requests; wire your platform's log
shipper to stdout.
