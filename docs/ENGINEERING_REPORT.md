# Engineering report — Living World digital twin, first release

_Date: 2026-09-15 · Stack: CesiumJS 1.145, React 19.3, Vite 8, TypeScript 6, FastAPI 0.141, SQLAlchemy 2, PostGIS 3.4_

## What was built

A production-grade foundation for a geospatial digital twin: one continuous CesiumJS
world from planet to centimetre, a mission-control UI for autonomous land-management
robots that keeps the map as the hero, and a PostGIS-backed catalog of sites, reality-model assets, open-data layers and
camera bookmarks with provenance, license and attribution on every record.

**World**

- Earth renders with Cesium World Terrain and Bing Maps Aerial (via Cesium ion), with
  Natural Earth II from the CesiumJS bundle as an offline/no-token fallback.
- Pan, orbit, tilt and zoom down to 5 cm; smooth distance-scaled fly-to easing; reset
  north, top-down, Earth, double-click approach, saved views.
- Search (Cesium ion geocoder with Nominatim fallback) and a `⌘K` command palette.

**High-detail transition**

- The public Cesium Gaussian-splat LOD tileset (ion 4547222) is a seeded site. "View
  high-resolution demo" flies from orbit into an oblique view of it; the splat refines
  through 3D Tiles LOD; the terrain and imagery under it are clipped away using a polygon
  derived from the tileset's real bounding box (`clipFootprint: "tileset"`), so nothing
  fights. Zooming out returns to Earth with no modal to close; the site unloads when far.
- Representation switcher (Splat · Mesh · Points) appears near a loaded site; switching
  never moves the camera; unavailable representations are disabled, not hidden.
- Explore mode: WASD/QE free-flight with mouse look on the same camera.

**Layers and data**

- Provider architecture for ion terrain/imagery/3D Tiles, Google Photorealistic (flagged),
  3D Tiles URLs, XYZ/WMTS/WMS/ArcGIS imagery, GeoJSON/CZML, MVT (experimental) and STAC.
- Seeded catalog of ten open datasets (Cesium World Terrain, Bing, Sentinel-2, OSM raster,
  OSM Buildings, ESA WorldCover 2021 with legend, USGS NHD hydrography, USGS orthoimagery,
  USGS 3DEP shaded relief, Google Photorealistic) with source, license, attribution,
  resolution, coverage and dates.
- Layers panel as a data browser (groups, filter, opacity, fly-to-extent, "About this layer").
- Add Data workflow for six input kinds with client and server validation, persisted to
  PostGIS; new sites fly into view.
- Inspector for ground, sites, vector and 3D features (position, heights, source, date,
  attribution, properties); measurement tools (point, 2D/3D distance, area, height,
  elevation) with unit switching; swipe compare; timeline foundation for dated versions.

**Engineering**

- Strict TypeScript everywhere, type-aware ESLint with no `any`, Prettier; Ruff + mypy
  strict on the API; OpenAPI-generated frontend types with a drift check in CI.
- Tests: 34 backend (CRUD, GeoJSON validation, spatial persistence, relationships,
  invalid data, migrations round-trip), 37 unit (geo, ui, web), 11 Playwright flows
  (boot, viewer init, catalog, fly-to, representation switch, layer toggle, inspector,
  add-data validation, measurement, palette/shortcuts, keyboard access, API failure).
- Error handling: per-asset/layer `idle → loading → ready → error` states, toasts, React
  error boundaries, WebGL context-loss notice, render-loop recovery, ion token setup
  notice, labeled built-in fallback catalog when the API is down.
- Adaptive quality controller (frame rate, motion, loading, distance, DPR), developer
  panel with tile debugging, dependency audits, docker compose, CI, deployment docs.

## Architecture in one paragraph

React owns application state (Zustand) and catalog state (TanStack Query); a single
`CesiumSceneManager` owns rendering through focused managers (Camera, Layers, Sites,
Selection, Measurement, Clipping, Performance, Explore, Debug) that emit typed events; one
`SceneBridge` mirrors those events into stores and pushes settings back. FastAPI + PostGIS
hold the catalog with GeoJSON validated structurally and topologically. Cesium ion is the
current delivery provider, not the data model; canonical data will live behind the
`ObjectStorage` abstraction. See `docs/ARCHITECTURE.md` and `docs/DECISIONS/`.

## Important decisions

1. One 3D world, no 2D mode, no React-Cesium wrapper (ADR 0001, 0002).
2. Clipping polygons derived from the streamed tileset when the catalog footprint is not
   authoritative (ADR 0003) — this is what makes the public demo embed cleanly.
3. Read models in the API are fully explicit so the generated TypeScript contract is
   strict; `--default-non-nullable=false` keeps input types honest.
4. Cesium ion reconstruction: only the documented API surface is used. Job creation waits
   for ion to document the photo input `sourceType`; registration and monitoring work now.
5. Reduced-motion, reduced-transparency and high-contrast are first-class token variants.

**Mission control (second iteration)**

- UI rebuilt on the "Robot Land Management UI" design: dark olive glass, project badge,
  Map / Plan / Fleet views, layer pills, machine markers and zone chips over the world,
  selection cards, Plans and Fleet windows with a treatment log, an agent activity stream
  and a bottom command bar with local intent parsing.
- A simulated demo project (Blackrock Mesa, six machines, three zones, five plans) is
  attached to the demo site and labeled as simulated; `MissionProvider` is the seam for a
  live fleet feed. See [MISSION_CONTROL.md](MISSION_CONTROL.md).

## Known limitations

- The fleet, zones, plans, agent actions and camera feeds are simulated demo data; there
  is no robot telemetry ingestion yet.

- Headless verification used SwiftShader (software GL); frame rates there are not
  representative and splat refinement is slow. The adaptive controller correctly lowered
  resolution scale in that environment, which is the intended behaviour on weak GPUs.
- The demo asset's clip polygon is its root bounding box; where the capture is sparse at
  the box edge the terrain hole shows. Provide tight footprints for your own sites.
- Mesh and point-cloud representations are wired end to end but the seeded demo only has
  a splat; set `VITE_DEFAULT_MESH_ASSET_ID` / `VITE_DEFAULT_POINTCLOUD_ASSET_ID` or register
  assets to exercise them with real data.
- Photorealistic world mode requires an ion token with Google Photorealistic access; it is
  feature-flagged and untested here without such a token.
- STAC support resolves items to renderable assets; COG-only items need a tile server.
- Thumbnail upload requires object storage; without it the endpoint returns 503 with
  instructions.
- No authentication yet: the API is intended to sit behind your platform's auth/edge.

## Setup

See README "Quickstart". In short: `cp .env.example .env`, `docker compose -f
infra/docker-compose.yml up -d`, `pnpm install`, `cd apps/api && uv sync && uv run alembic
upgrade head && uv run python -m app.seed`, then `pnpm dev:api` and `pnpm dev`.

## Next highest-value steps

1. **Your own reality models**: register mesh + point cloud + splat for one real site and
   tune per-asset SSE and footprints; add thumbnails via MinIO.
2. **Canonical storage**: land photos/LAS/COPC in S3 behind `ObjectStorage`, record them
   as captures linked to derived assets.
3. **Temporal captures**: multiple dated assets per site make the timeline live; add a
   change-detection comparison (swipe already works).
4. **Semantic entities**: `plants`/`objects` tables with geometry; extend selection and
   the inspector to them.
5. **Self-hosted tiling**: a 3D Tiles/COPC/COG tile service so Cesium ion becomes optional.
6. **LLM assistant**: expose the manager API (fly, toggle, measure, filter) as tools behind
   the command palette.
7. **Auth and multi-tenant catalog**; per-site sharing; audit trail on catalog edits.
