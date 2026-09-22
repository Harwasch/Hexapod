# Data model

All geometry is GeoJSON in WGS 84 (EPSG:4326) on the wire and `geometry(…, 4326)` in
PostGIS. All timestamps are ISO 8601 with time zone. Wire field names are camelCase; the
Python models are snake_case (`CamelModel` handles both).

```text
sites 1 ──── * assets            (site_id nullable: assets can exist before a site)
sites 1 ──── * camera_bookmarks
layers                           (independent catalog entries)
```

## sites

| Column                 | Type                         | Notes                                        |
| ---------------------- | ---------------------------- | -------------------------------------------- |
| id                     | uuid                         |                                              |
| slug                   | text unique                  | derived from name when omitted               |
| name, description      | text                         |                                              |
| boundary               | MultiPolygon 4326 (GiST)     | Polygons are promoted to MultiPolygon        |
| centroid               | Point 4326 + centroid_height | defaults to the boundary centroid            |
| thumbnail_url          | text                         | set by the thumbnail upload (object storage) |
| metadata               | jsonb                        | free-form (`quality`, `origin`, …)           |
| attribution            | jsonb `Attribution[]`        |                                              |
| license                | jsonb `LicenseMetadata`      |                                              |
| created_at, updated_at | timestamptz                  |                                              |

Derived on read: `areaM2` (`ST_Area(boundary::geography)`), and for summaries the set of
`representations`, `latestObservedAt` and `quality`.

## assets

A derived, renderable delivery asset with provenance.

| Column                            | Type                               | Notes                                                                                                                                                                                                                                                                                                                             |
| --------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| site_id                           | uuid → sites (cascade)             | nullable                                                                                                                                                                                                                                                                                                                          |
| name                              | text                               |                                                                                                                                                                                                                                                                                                                                   |
| representation                    | enum                               | `gaussian-splat` · `mesh` · `point-cloud` · `terrain` · `imagery`                                                                                                                                                                                                                                                                 |
| provider                          | enum                               | `cesium-ion` · `3d-tiles-url` (delivery provider, not identity)                                                                                                                                                                                                                                                                   |
| source                            | jsonb                              | `{type:"cesium-ion", assetId}` or `{type:"3d-tiles-url", url}`                                                                                                                                                                                                                                                                    |
| footprint                         | MultiPolygon 4326 (GiST), nullable | falls back to the site boundary for clipping                                                                                                                                                                                                                                                                                      |
| observed_at, valid_from, valid_to | timestamptz                        | temporal foundation; `validTo ≥ validFrom` enforced                                                                                                                                                                                                                                                                               |
| resolution                        | jsonb                              | `groundSampleDistanceM`, `pointSpacingM`, `description`                                                                                                                                                                                                                                                                           |
| crs                               | jsonb                              | horizontal / vertical reference notes                                                                                                                                                                                                                                                                                             |
| license, attribution              | jsonb                              |                                                                                                                                                                                                                                                                                                                                   |
| render_config                     | jsonb `RenderConfig`               | `maximumScreenSpaceError`, `pointCloudShading`, `clipsWorld`, `clipFootprint` (`catalog`/`tileset`), `heightOffsetM`, `clampToGround`, `groundSamples` (the capture's own measured ground as ellipsoid heights — what the clamp rests on); provenance, including `georefMethod`/`scaleSource`/`uncertaintyM`, is stored alongside |
| default_visible                   | bool                               | the representation shown first                                                                                                                                                                                                                                                                                                    |

## layers

| Column                   | Type                        | Notes                                                                                                                                                     |
| ------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| slug, name, description  | text                        |                                                                                                                                                           |
| category                 | enum                        | `reality` · `terrain` · `imagery` · `hydrology` · `land-cover` · `ecology` · `infrastructure` · `my-data`                                                 |
| source_type              | enum                        | mirrors `source.type` for indexing                                                                                                                        |
| source                   | jsonb (discriminated union) | see `app/schemas/layer.py`: ion terrain/imagery/3D Tiles, Google Photorealistic, 3D Tiles URL, XYZ, WMTS, WMS, ArcGIS MapServer, GeoJSON, CZML, MVT, STAC |
| spatial_extent           | Polygon 4326 (GiST)         | bounding box                                                                                                                                              |
| temporal_extent          | jsonb `{start,end}`         |                                                                                                                                                           |
| observed_at              | timestamptz                 | latest/observation date                                                                                                                                   |
| resolution, coverage     | text                        | human-readable                                                                                                                                            |
| render                   | jsonb `RenderMetadata`      | `opacity`, altitude range, `maximumScreenSpaceError`, `exclusiveGroup` (`basemap`, `terrain`, `world`); provenance stored alongside                       |
| legend                   | jsonb                       | title + colour entries or image                                                                                                                           |
| attribution, license     | jsonb                       |                                                                                                                                                           |
| default_visible, builtin | bool                        | built-in catalog rows cannot be deleted                                                                                                                   |

## camera_bookmarks

`site_id`, `name`, `longitude`, `latitude`, `height` (ellipsoidal metres), `heading`,
`pitch`, `roll` (degrees), `is_default` (one per site; setting a new default clears others).

## Shared metadata objects

- **Attribution** `{ text, organization?, url? }` — must remain visible while shown.
- **LicenseMetadata** `{ name, spdxId?, url?, requiresAttribution, notes? }`.
- **Provenance** `{ sourceOrganization?, sourceUrl?, publishedAt?, notes? }`.
- **TemporalExtent** `{ start?, end? }`.

## Validation

GeoJSON input is validated structurally (closed rings, ≥4 positions, coordinate ranges)
and topologically with Shapely (`is_valid`, non-zero area). URLs must be http(s) without
embedded credentials; in production (`ALLOW_PRIVATE_URLS=false`) loopback/private hosts are
rejected. Unknown fields are rejected everywhere (`extra="forbid"`).

## Extending towards captures, sensors, plants, missions

Add new tables keyed to `sites` (and to `assets` where a capture produced an asset), keep
geometry in PostGIS and variable attributes in JSONB, and add a `provider`/`source_type`
value when a new delivery path appears. Nothing in the current tables needs to change; the
`assets.site_id` nullability and the temporal columns were chosen so captures and versions
slot in without a rewrite.

## Migrations

Alembic, hand-written (`apps/api/alembic/versions`). `alembic check` runs in the test
suite so models and migrations cannot drift. The PostGIS extension is created by the first
migration.
