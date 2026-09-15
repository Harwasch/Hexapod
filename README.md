# Hexapod monorepo

This repository holds several parts of one project. The **Living World** digital twin — a
continuously zoomable, time-aware 3D world built on CesiumJS — lives in `apps/` and
`packages/`. Hardware design, simulation and other subsystems live alongside it in their
own directories.

```text
/
├── apps/
│   ├── web/        React + Vite + CesiumJS mission control (dark glass UI)
│   └── api/        FastAPI + PostGIS catalog (sites, assets, layers, bookmarks)
├── packages/
│   ├── contracts/  OpenAPI document + generated TypeScript types
│   ├── geo/        Framework-free geospatial helpers (units, footprints, scale)
│   ├── ui/         Glass design system (tokens + accessible primitives)
│   └── config/     Shared TypeScript configuration
├── infra/          docker compose (PostGIS, MinIO), API Dockerfile
├── docs/           Architecture, Cesium notes, data model, deployment, decisions
└── .github/        CI (lint, typecheck, tests, build, audit)
```

## What it does

One continuous CesiumJS globe, one camera. Earth → region → site → a high-resolution
reality model (Gaussian splat, mesh or point cloud) → centimetre detail. The coarse world is
clipped away under each local model, so nothing z-fights. Open datasets (terrain, imagery,
land cover, hydrography, buildings) are composable layers with provenance, license and
attribution. Sites, assets and layers persist in PostGIS.

A public demo site (Cesium's Gaussian-splat sample, ion asset 4547222) works out of the box
using the evaluation token bundled with CesiumJS. On top of the world sits **mission control**
for autonomous land-management robots: project badge, Map / Plan / Fleet views, zone and
machine overlays, plans, fleet and treatment log, an agent activity stream and a command bar.
The demo fleet is simulated and labeled as such (see [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)).

## Quickstart (clean machine)

Prerequisites: Node 22+, [pnpm](https://pnpm.io) 10, Python 3.12, [uv](https://docs.astral.sh/uv/),
Docker (for PostGIS/MinIO). The lockfiles pin everything else.

```bash
git clone <this repository> && cd Hexapod
cp .env.example .env                      # fill in tokens later; nothing is required to boot
docker compose -f infra/docker-compose.yml up -d   # PostGIS on :5432, MinIO on :9000/:9001
pnpm install
cd apps/api && uv sync && uv run alembic upgrade head && uv run python -m app.seed && cd ../..
pnpm dev:api                              # http://localhost:8000/api/v1/docs  (terminal 1)
pnpm dev                                  # http://localhost:5173              (terminal 2)
```

Open <http://localhost:5173>, click **View high-resolution demo**.

Without Docker: any PostgreSQL 16 with the PostGIS extension works; point `DATABASE_URL`
and `TEST_DATABASE_URL` at it. Without the API at all, the web app still boots with the
built-in demo site and labels itself "Catalog API offline".

### Environment variables

See [`.env.example`](.env.example) for every variable with comments. The important ones:

| Variable                       | Where | Purpose                                                                                 |
| ------------------------------ | ----- | --------------------------------------------------------------------------------------- |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | web   | Browser token (`assets:read`, `geocode`). Empty → CesiumJS evaluation token (dev only). |
| `VITE_DEFAULT_*_ASSET_ID`      | web   | Your own splat / mesh / point-cloud ion assets for the built-in site (any subset).      |
| `VITE_ENABLE_PHOTOREALISTIC`   | web   | Feature flag for Google Photorealistic 3D Tiles (visual context only).                  |
| `DATABASE_URL`                 | api   | PostgreSQL + PostGIS connection.                                                        |
| `API_CORS_ORIGINS`             | api   | Allowed browser origins.                                                                |
| `OBJECT_STORAGE_*`             | api   | Optional S3/MinIO for site thumbnails.                                                  |
| `CESIUM_ION_SERVER_TOKEN`      | api   | Server-side ion token to monitor reconstruction jobs. Never exposed to the browser.     |

`VITE_` variables are public and inlined into the bundle. Server secrets never carry that prefix.

## Everyday commands

```bash
pnpm dev / pnpm dev:api        # dev servers
pnpm lint && pnpm typecheck    # ESLint (strict, type-aware) + tsc for every package
pnpm test                      # Vitest: packages/geo, packages/ui, apps/web
pnpm e2e                       # Playwright (needs Chromium: pnpm --filter @twin/web exec playwright install chromium)
pnpm build                     # production bundle in apps/web/dist
pnpm contracts:generate        # regenerate TS types after changing API schemas
cd apps/api && uv run pytest   # backend tests (needs TEST_DATABASE_URL)
cd apps/api && uv run ruff check . && uv run mypy .
```

After changing a Pydantic schema: `uv run python -m app.scripts.export_openapi ../../packages/contracts/openapi.json`
then `pnpm contracts:generate`. CI fails if the committed contract is stale.

## Keyboard

`⌘K`/`Ctrl+K` command palette · `/` search · `L` layers · `S` sites · `M` measure · `C` compare ·
`B` bookmarks · `N` reset north · `T` top-down · `H` Earth · `G` explore mode · `,` settings ·
`1`/`2`/`3` Map / Plan / Fleet · `A` agent stream · `D` developer panel (dev builds) · `Esc` closes.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why CesiumJS, 3D Tiles, PostGIS; the seams for STAC/S3/COPC/robotics
- [docs/COMPARISON.md](docs/COMPARISON.md) — mesh vs point cloud vs Gaussian splat comparison sites and how to benchmark them
- [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md) — robot mission layer: views, overlays, command bar, provider seam
- [docs/CESIUM.md](docs/CESIUM.md) — scene manager, clipping, LOD/adaptive quality, tokens, current API notes
- [docs/DATA_MODEL.md](docs/DATA_MODEL.md) — sites, assets, layers, bookmarks, provenance
- [docs/ADDING_DATA.md](docs/ADDING_DATA.md) — every supported input, validation rules, ion reconstruction
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Vercel-style static web, containerised API, managed PostGIS
- [docs/DECISIONS/](docs/DECISIONS/) — architecture decision records
- [docs/ENGINEERING_REPORT.md](docs/ENGINEERING_REPORT.md) — what was built, limitations, next steps

## Hardware subsystems

The MakeHardware workflow used by the hardware parts of this repository is documented in
[docs/HARDWARE.md](docs/HARDWARE.md).
