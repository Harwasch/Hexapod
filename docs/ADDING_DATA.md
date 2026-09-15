# Adding data

Open **Add data** from the left rail (or `⌘K → Add data`). Everything you register is
validated in the browser and again by the API, then persisted in PostGIS and appears in
the Layers or Sites panel. Nothing is fetched by the server: the browser streams from the
source you register.

## Reality model (site)

A physical place with a high-resolution capture.

| Field                          | Notes                                                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Site name                      | required                                                                                                                            |
| Footprint                      | GeoJSON Polygon/MultiPolygon (paste, upload `.geojson`, or "Around camera"). Clips terrain and the global 3D world under the model. |
| Representation                 | Splat · Mesh · Points                                                                                                               |
| Asset                          | Cesium ion asset ID or a `tileset.json` URL (https, no credentials)                                                                 |
| Observation date               | optional, drives the timeline when several dated captures exist                                                                     |
| Source / license / attribution | provenance shown in the inspector and "About"                                                                                       |
| Render quality                 | `maximumScreenSpaceError` floor (lower = sharper)                                                                                   |
| Clip world                     | on by default                                                                                                                       |
| Default view                   | saves the current camera as the site's default bookmark                                                                             |

Add more representations to an existing site with `POST /api/v1/assets` (`siteId` set), or
by re-running the form for the same site with a different representation.

## Cesium ion asset (layer)

3D Tiles, imagery or terrain assets from your ion account (needs a browser token with
access). Terrain assets join the exclusive `terrain` group.

## 3D Tiles URL

Any tileset served over https with CORS. STAC items that point at 3D Tiles resolve to this.

## GeoJSON URL

Loaded with `GeoJsonDataSource`, clamped to ground, styled with the accent colour. Features
are pickable; their properties appear in the inspector. For uploads, host the file
(object storage, GitHub raw, S3) and paste its URL — the API deliberately does not accept
arbitrary file uploads for datasets.

## Imagery service

- **XYZ** — URL template with `{z}/{x}/{y}` (optionally `{s}`, `{reverseY}`).
- **WMTS** — service URL, layer, tile matrix set (default `GoogleMapsCompatible`).
- **WMS** — service URL and layer names; `transparent=true`, PNG by default.
- **ArcGIS MapServer** — service URL, optional layer ids.

## STAC

Paste an item, collection or catalog URL. The client resolves it to something Cesium can
stream: a 3D Tiles asset, an XYZ template, or a single georeferenced PNG/JPEG using the
item's bbox. Items that only expose Cloud-Optimized GeoTIFFs are reported with a clear
message: publish them through a tile server (e.g. TiTiler) and add that as an imagery
layer. That is the correct abstraction rather than decoding COGs in the browser.

## Validation summary

- Names 2–200 characters; slugs derived automatically.
- Asset IDs are positive integers; URLs are http(s) without credentials.
- Footprints: closed rings, ≥4 positions, valid topology, non-zero area.
- Dates must parse; `validTo ≥ validFrom`.
- Unknown fields are rejected by the API (`422` with field paths).

## Cesium ion photo reconstruction

Cesium ion reconstructs photos into a **mesh**, optional **point cloud** and optional
**Gaussian splats**, each becoming its own asset. As of ion's public OpenAPI document
(revision 2026-09-08) the reconstruction _outputs_ and job options are documented for
`POST /v1/assets`, but the `sourceType` that accepts photo inputs is not. We therefore do
not create jobs from this backend (no reverse-engineering of private calls). The workflow:

1. In ion, _Add data → Photos (for 3D reconstruction)_, choose the outputs.
2. When the assets complete, register their IDs here (one site, several representations).
3. With `CESIUM_ION_SERVER_TOKEN` set on the API (`assets:read`), `GET /api/v1/ion/assets/{id}`
   reports tiling status and progress so the UI can monitor jobs.

`app/services/ion.py` exposes a `ReconstructionProvider` seam; when the input source type
is documented, job creation slots in behind the same interface.
