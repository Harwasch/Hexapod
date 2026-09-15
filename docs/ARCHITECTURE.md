# Architecture

## The one-sentence version

React owns state and UI; a dedicated Cesium subsystem owns rendering; FastAPI + PostGIS own
the catalog; Cesium ion (today) delivers heavy 3D assets straight to the browser.

```text
┌──────────────────────────── browser ────────────────────────────┐
│  React (Zustand UI state, TanStack Query catalog state)         │
│      │ calls managers            ▲ events (camera, load, perf)  │
│  CesiumSceneManager ─ Camera · Layers · Sites · Selection ·      │
│                        Measurement · Clipping · Performance ·    │
│                        Explore · Debug                           │
│      │ streams tiles                                             │
└──────┼──────────────────────────────────────────────────────────┘
       │                       ┌─────────── FastAPI ───────────┐
       ├── Cesium ion / 3D Tiles│  /api/v1  sites · assets ·    │
       ├── imagery/WMS/XYZ/etc. │  layers · bookmarks · ion     │──── PostGIS
       └── GeoJSON/CZML/STAC    │  storage abstraction (S3/MinIO)│
                                └───────────────────────────────┘
```

## Why CesiumJS is the viewer

The product requires a single continuous 3D world from planet to centimetre. CesiumJS is
the only mature open-source engine that combines a WGS 84 globe, streamed terrain and
imagery, and the 3D Tiles standard (meshes, point clouds and, since 2025, Gaussian splats with
hierarchical LOD) in one scene graph with one camera. We never use its 2D scene mode: a
high, downward camera _is_ the map.

## Why 3D Tiles is the delivery representation

3D Tiles is an OGC community standard for streaming massive 3D content with level of
detail. Every local reality model (splat, mesh, point cloud) and every global 3D layer (OSM
buildings, Google Photorealistic) arrives as 3D Tiles, so the viewer has exactly one code
path for loading, LOD, clipping, picking and styling. Cesium ion is the current tiler and
CDN; a tileset URL from any other tiler works identically (`{ "type": "3d-tiles-url" }`).

## Why raw / canonical world data will live separately

Ion asset IDs and tileset URLs are **delivery** locators, not the data model. The catalog
records each asset's provenance (organization, URL, dates, license, CRS, resolution) and
its representation; the source captures (photos, LAS/COPC, GeoTIFF/COG, MCAP) will live in
S3-compatible object storage under the `ObjectStorage` abstraction already present in the
API. That keeps the pipeline explicit:

```text
Persistent / canonical world data  (S3: photos, COPC, COG, GeoParquet, MCAP)
            ↓  tiling / reconstruction (ion today, self-hosted later)
derived delivery assets            (3D Tiles, quantized-mesh terrain, XYZ/WMTS imagery)
            ↓
Cesium                             (one scene, one camera)
```

## Why Gaussian splats are visual, not analytical

Splats are view-dependent radiance reconstructions. They look like reality at close range
but have no surfaces to measure against. Measurements pick against terrain and mesh/point
content (`scene.pickPosition`), and the UI labels the splat representation as a visual
model. Analytical questions belong to point clouds, meshes, DEMs and semantic entities.

## Why PostGIS stores semantic and spatial metadata

Sites, assets and layers are small, relational and spatial (footprints, extents, centroids)
and must support queries like "which sites intersect this view" or "what was captured
here in 2024". PostGIS + SQLAlchemy/GeoAlchemy2 + Alembic gives typed geometry columns,
spatial indexes, validated GeoJSON in and out, and migrations. JSONB columns hold the
extensible parts (render config, resolution, license) without a migration per field.

## The web app boundary

- `apps/web/src/cesium/` — the only place that imports Cesium classes. `CesiumSceneManager`
  is constructed once by `CesiumViewport` and disposed once. Managers emit typed events
  through a tiny emitter; `SceneBridge` is the single component mirroring them into
  Zustand stores and pushing settings back. React components call managers imperatively
  (`useScene()`), never re-rendering the scene.
- `apps/web/src/state/` — Zustand stores: settings (persisted), ui, viewer telemetry,
  layer/site runtime, selection, measurements, toasts.
- `apps/web/src/api/` — `openapi-fetch` client typed from `@twin/contracts`, TanStack
  Query hooks, and a labeled built-in fallback catalog for when the API is down.
- `apps/web/src/features/` — one directory per surface (search, layers, sites, inspector,
  measure, compare, bookmarks, add-data, settings, palette, dev, status, nav, timeline,
  explore, onboarding). Panels are lazy where large.
- `packages/ui` — the glass design system; `packages/geo` — pure geospatial math;
  `packages/contracts` — the API contract.

## Layer provider architecture

`LayerManager` turns a catalog `Layer` (a discriminated `source` union) into a Cesium
object through provider adapters in `cesium/providers/`:

| Source type                            | Cesium object                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| `cesium-ion-terrain`                   | `Terrain(CesiumTerrainProvider.fromIonAssetId)`                                |
| `cesium-ion-imagery`                   | `IonImageryProvider`                                                           |
| `xyz` / `wmts` / `wms`                 | `UrlTemplate` / `WebMapTileService` / `WebMapService` imagery providers        |
| `arcgis-mapserver`                     | `ArcGisMapServerImageryProvider`                                               |
| `cesium-ion-3d-tiles` / `3d-tiles-url` | `Cesium3DTileset`                                                              |
| `google-photorealistic`                | `createGooglePhotorealistic3DTileset`                                          |
| `mvt`                                  | `MVTDataProvider` (experimental in CesiumJS)                                   |
| `geojson` / `czml`                     | `GeoJsonDataSource` / `CzmlDataSource`                                         |
| `stac`                                 | resolved client-side to one of the above, or reported as needing a tile server |

Swapping Cesium World Terrain for self-hosted Copernicus DEM or USGS 3DEP is a new row in
the `layers` table (a `3d-tiles-url`/terrain URL source), not a code change. A new source
kind is one Pydantic model, one enum value and one provider adapter.

## Seams for what comes next

| Future capability            | Where it plugs in                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| STAC catalogs                | `stac` layer source + client resolver; a backend STAC crawler can populate layers                                           |
| S3 / COG / COPC / GeoParquet | `ObjectStorage` abstraction; new `provider`/`sourceType` values; a tile server (TiTiler/COPC) as an imagery/3D Tiles source |
| Temporal captures            | `observed_at` / `valid_from` / `valid_to` on assets; `TimelineControl` already switches versions                            |
| Semantic entities / plants   | new tables keyed to `sites` with geometry; `SelectionManager` already resolves features to catalog objects                  |
| Robotics (ROS/MCAP)          | captures as canonical data; poses as CZML/time-dynamic entities via the existing data-source path                           |
| Simulation (Isaac/OpenUSD)   | consumes the same canonical store; the viewer stays a 3D Tiles client                                                       |
| LLM geospatial assistant     | the command palette is the entry point; managers expose a small imperative API to drive                                     |
| Observability vendor         | `lib/log.ts` sinks and `lib/timing.ts` spans                                                                                |
