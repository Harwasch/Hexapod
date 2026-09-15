# Deployment

```text
Web       static bundle (Vercel / Netlify / S3+CloudFront / any static host)
API       container (Fly.io, Cloud Run, ECS, Railway…) built from infra/api.Dockerfile
Database  managed PostgreSQL with PostGIS (Neon, Supabase, RDS, Cloud SQL…)
Assets    Cesium ion (today); S3-compatible object storage for canonical data later
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
| `VITE_ENABLE_PHOTOREALISTIC`   | `true` only if your token has Google Photorealistic access and you accept the terms |
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
| `ALLOW_PRIVATE_URLS=false` | reject loopback/private dataset hosts                                |
| `OBJECT_STORAGE_*`         | optional; enables thumbnail uploads                                  |
| `CESIUM_ION_SERVER_TOKEN`  | optional; `assets:read` for job monitoring. Never a `VITE_` variable |

Put the API behind TLS (platform load balancer). It exposes `/api/v1/health` for probes and
`/api/v1/docs` for OpenAPI; disable docs at the edge if you prefer.

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
