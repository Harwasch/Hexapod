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
│   ├── world/      Living Survey motion model (rigs, wind, deformation) — no Cesium
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

The demo is **Oak Bench Ranch**, a fictional ranch on real land: oak savanna along Figueroa
Creek on California's Central Coast, near Santa Ynez, with a simulated fleet working it (the
outline is a demo area, not a property line; the name, machines, zones and plans are simulated,
and nothing connects it with the landowner). Its geometry, sources and derived acreages live
in one file, `apps/api/app/seed/demo_ranch.json` ([docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)
says how to change it). The 3D-scan showcase is the Redmond, WA site where Cesium's
Gaussian-splat sample sits; that splat is withdrawn because this deployment's ion account can't
read it, and `VITE_DEFAULT_SPLAT_ASSET_ID` gives the offline copy one. On top of the world sits **mission control**
for autonomous land-management robots: project badge, Map / Plan / Fleet views, zone and
machine overlays, plans, fleet and treatment log, one command box (search, actions and the
agent) and a status line for the fleet, the agent and the connection.
The demo fleet is simulated and labeled as such (see [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)).

A **Living Survey** prototype makes one tree sway under a wind setting without its measurement
ever changing: the deformer recomputes every frame from an immutable copy of the canonical splat
positions, so calm restores the measured bytes exactly, and the motion is labelled Simulated
wherever it shows. Today the only tree that moves is a procedural fixture — see
[docs/LIVING_SURVEY.md](docs/LIVING_SURVEY.md) for what it proves and what it does not.

## Quickstart (clean machine)

Prerequisites: Node 22+, [pnpm](https://pnpm.io) 10, Python 3.12, [uv](https://docs.astral.sh/uv/),
Docker (for PostGIS/MinIO). The lockfiles pin everything else.

```bash
git clone <this repository> && cd Hexapod
cp .env.example .env                      # fill in tokens later; nothing is required to boot
docker compose -f infra/docker-compose.yml up -d   # PostGIS on :5432, MinIO on :9000/:9001
pnpm install
cd apps/api && uv sync && uv run alembic upgrade head && uv run python -m app.seed && cd ../..
cd apps/api && uv run python -m app.seed.demo_land --research && cd ../..   # Oak Bench Ranch as a saved Land, its overview queued
pnpm dev:api                              # http://localhost:8000/api/v1/docs  (terminal 1)
pnpm dev                                  # http://localhost:5173              (terminal 2)
pnpm dev:research                         # the Land research worker           (terminal 3)
```

Open <http://localhost:5173>, click **Open the demo ranch**. The Land workspace (saved land,
its research, the fog of knowing, plans on your own ground) is behind a flag: set
`VITE_ENABLE_LAND_EXPLORATION=true` in `.env` before `pnpm dev` to see it, with **Land** in
the tool rail (`Shift+L`). The last two commands only matter with it on: the saved demo Land and
the worker that runs its research.

Four pages are served, not one. `/` is the globe; `/view.html` is the scan gallery, each scan
rendered on its own with Spark; `/admin.html` is the data console — every capture in storage,
every run that produced it, and the form that launches more; `/upload.html` is the page a phone
opens after scanning a capture's handoff QR code. None of the last three loads CesiumJS
(`apps/web/scripts/check-bundle.mjs` checks the production build).

Without Docker: any PostgreSQL 16 with the PostGIS extension works; point `DATABASE_URL`
and `TEST_DATABASE_URL` at it. Without the API at all, the web app still boots with the
built-in demo ranch and scan showcase and labels itself "Catalog API offline". The seed is
manual (`uv run python -m app.seed`, also after a deploy).

The demo ranch as a saved Land is a separate seed:
`cd apps/api && uv run python -m app.seed.demo_land` saves Oak Bench Ranch in the pilot
workspace's Land, with three inventory features that name their sources: the perimeter fence
(simulated), Figueroa Creek (USGS NHD) and the oak woodland (ESA WorldCover 2021; oak assumed).
Select it in **Land** while the demo ranch is open and its features join the ranch's zones,
fleet and all. It is idempotent, and `--research` also queues the Land's overview for
`pnpm dev:research`. It records no inspections, findings or evidence: those come from people
and real research runs ([docs/LAND_EXPLORATION.md](docs/LAND_EXPLORATION.md)).

### Environment variables

See [`.env.example`](.env.example) for every variable with comments. The important ones:

| Variable                       | Where | Purpose                                                                                 |
| ------------------------------ | ----- | --------------------------------------------------------------------------------------- |
| `VITE_CESIUM_ION_ACCESS_TOKEN` | web   | Browser token (`assets:read`, `geocode`). Empty → CesiumJS evaluation token (dev only). |
| `VITE_DEFAULT_*_ASSET_ID`      | web   | Your own splat / mesh / point-cloud ion assets for the built-in site (any subset).      |
| `VITE_ENABLE_PHOTOREALISTIC`   | web   | Google Photorealistic 3D Tiles world, on by default; `false` switches it off.           |
| `VITE_SPLAT_GPU_MOTION`        | web   | Living Survey motion in the splat shader, on by default; `false` forces the CPU path.   |
| `VITE_ENABLE_LAND_EXPLORATION` | web   | The Land workspace, off by default; `true` shows it. Off, the build ships none of it.   |
| `TILE_PROXY_HOST`              | web   | Dev server: serve this public tile bucket under `/r2/`, as Pages does in production.    |
| `DATABASE_URL`                 | api   | PostgreSQL + PostGIS connection.                                                        |
| `API_CORS_ORIGINS`             | api   | Allowed browser origins.                                                                |
| `API_WRITE_TOKEN`              | api   | Shared token for every write (`Authorization: Bearer …`). Reads stay open.              |
| `OBJECT_STORAGE_*`             | api   | Optional S3/MinIO for site thumbnails.                                                  |
| `CESIUM_ION_SERVER_TOKEN`      | api   | Server-side ion token to monitor reconstruction jobs. Never exposed to the browser.     |

`VITE_` variables are public and inlined into the bundle. Server secrets never carry that prefix.
That is why `API_WRITE_TOKEN` has no `VITE_` twin: the console asks for it in the Captures panel the
first time a write comes back `401`, and keeps it in this browser (`twin.settings.v1`) rather than in
the bundle. With no token configured, writes are open and nothing is ever asked for.

## Everyday commands

```bash
pnpm dev / pnpm dev:api        # dev servers
pnpm dev:research              # land research worker; queued Land analyses wait for it
pnpm lint && pnpm typecheck    # ESLint (strict, type-aware) + tsc for every package
pnpm test                      # Vitest: packages/geo, packages/ui, packages/world, apps/web
pnpm e2e                       # Playwright (needs Chromium: pnpm --filter @twin/web exec playwright install chromium)
pnpm build                     # production bundle in apps/web/dist
pnpm contracts:generate        # regenerate TS types after changing API schemas
cd apps/api && uv run pytest   # backend tests (needs TEST_DATABASE_URL)
cd apps/api && uv run ruff check . && uv run mypy .
```

After changing a Pydantic schema: `uv run python -m app.scripts.export_openapi ../../packages/contracts/openapi.json`
then `pnpm contracts:generate`. CI fails if the committed contract is stale.

## Sample objects for close-range testing

`pnpm samples` (with the API and dev server running) downloads two CC0 photoscanned objects,
a 14 cm rock and a 2 m weed, and places them beside the Redmond scan showcase so millimetre-scale zoom can be
tested without capturing anything. Press `S` and pick them, or type `fly to rock` in the
command box. See [docs/COMPARISON.md](docs/COMPARISON.md).

## The hosted phone scans, locally

The phone scans whose objects carry image embeddings (the camp, the pumpkin and the spool)
exist only in the hosted catalog. `cd apps/api && uv run python -m app.seed.scans` registers
them in your local catalog from a committed snapshot (idempotent, no network), pointing at the
same public-bucket tiles. Start the dev server with
`TILE_PROXY_HOST=pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev pnpm dev` so it serves that bucket
on its own origin, as Pages does in production. Tap an object of the camp and press
**More like this**. See [docs/SCENE_OBJECTS.md](docs/SCENE_OBJECTS.md#more-like-this).

## Mouse and keyboard

Google Maps conventions: left-drag pans, wheel zooms towards the cursor, **Ctrl+drag,
right-drag or middle-drag orbit the point in the centre of the view**, and the tilt stops at
the horizon. Arrow keys pan, `Shift`+arrows orbit the view centre (left/right turn, up/down
tilt), `+`/`-` zoom towards it; held keys move continuously. Beside a hand-sized object a
plain drag orbits the point you clicked.

`⌘K`/`Ctrl+K` or `/` the command box · `?` every shortcut · `L` layers · `S` sites · `U` upload ·
`Shift+L` Land (with the flag on) · `M` measure · `C` compare · `V` saved views · `B` the brush (paint a scan's objects to select
them) · `[`/`]` cycle a selected object's candidates · `N` reset north · `T` top-down · `H`
Earth · `G` explore mode · `,` settings · `1`/`2`/`3` Map / Plan / Fleet · `A` agent activity ·
`D` developer panel (dev builds) · `Esc` steps back one thing per press. The list lives in one registry,
`apps/web/src/app/hotkeys.ts`, which the `?` sheet prints and the command box shows beside
each action. Camera readouts (altitude, scale, renderer) are under Settings › Advanced ›
Show developer readouts.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why CesiumJS, 3D Tiles, PostGIS; the seams for STAC/S3/COPC/robotics
- [docs/COMPARISON.md](docs/COMPARISON.md) — mesh vs point cloud vs Gaussian splat comparison sites and how to benchmark them
- [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md) — robot mission layer: views, overlays, command box, status line, provider seam
- [docs/CESIUM.md](docs/CESIUM.md) — scene manager, clipping, LOD/adaptive quality, splat internals, tokens, current API notes
- [docs/WEBGPU_TRIAL.md](docs/WEBGPU_TRIAL.md) — the PlayCanvas WebGPU splat renderer trial: how to switch, what to compare in person, devices, criteria for the default
- [docs/LIVING_SURVEY.md](docs/LIVING_SURVEY.md) — simulated motion over measured geometry: the mechanism, what is measured vs simulated, and the limits
- [docs/DATA_MODEL.md](docs/DATA_MODEL.md) — sites, assets, layers, bookmarks, provenance
- [docs/ADDING_DATA.md](docs/ADDING_DATA.md) — every supported input, validation rules, ion reconstruction
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — the committed deployment: Cloudflare Pages + R2, the API and worker on Fly, Neon for PostGIS; a `provision` workflow that creates all of it from CI, and the four steps left for an account owner
- [docs/DECISIONS/](docs/DECISIONS/) — architecture decision records
- [docs/ENGINEERING_REPORT.md](docs/ENGINEERING_REPORT.md) — what was built, limitations, next steps

## Hardware subsystems

The MakeHardware workflow used by the hardware parts of this repository is documented in
[docs/HARDWARE.md](docs/HARDWARE.md).
