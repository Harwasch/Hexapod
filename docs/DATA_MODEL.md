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

| Column                            | Type                               | Notes                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| site_id                           | uuid → sites (cascade)             | nullable                                                                                                                                                                                                                                                                                                                                                                                      |
| name                              | text                               |                                                                                                                                                                                                                                                                                                                                                                                               |
| representation                    | enum                               | `gaussian-splat` · `mesh` · `point-cloud` · `terrain` · `imagery`                                                                                                                                                                                                                                                                                                                             |
| provider                          | enum                               | `cesium-ion` · `3d-tiles-url` (delivery provider, not identity)                                                                                                                                                                                                                                                                                                                               |
| source                            | jsonb                              | `{type:"cesium-ion", assetId}` or `{type:"3d-tiles-url", url}`                                                                                                                                                                                                                                                                                                                                |
| footprint                         | MultiPolygon 4326 (GiST), nullable | falls back to the site boundary for clipping                                                                                                                                                                                                                                                                                                                                                  |
| observed_at, valid_from, valid_to | timestamptz                        | temporal foundation; `validTo ≥ validFrom` enforced                                                                                                                                                                                                                                                                                                                                           |
| resolution                        | jsonb                              | `groundSampleDistanceM`, `pointSpacingM`, `description`                                                                                                                                                                                                                                                                                                                                       |
| crs                               | jsonb                              | horizontal / vertical reference notes                                                                                                                                                                                                                                                                                                                                                         |
| license, attribution              | jsonb                              |                                                                                                                                                                                                                                                                                                                                                                                               |
| render_config                     | jsonb `RenderConfig`               | `maximumScreenSpaceError`, `pointCloudShading`, `clipsWorld`, `clipFootprint` (`catalog`/`tileset`), `heightOffsetM`, `clampToGround`, `groundSamples` (the capture's own measured ground as ellipsoid heights — what the clamp rests on), `scale`/`scaleEvidence` (a runtime size correction, below); provenance, including `georefMethod`/`scaleSource`/`uncertaintyM`, is stored alongside |
| default_visible                   | bool                               | the representation shown first                                                                                                                                                                                                                                                                                                                                                                |
| sidecar_flags                     | jsonb                              | sidecars a republish could not carry into the new generation, one `{kind, action, reason, jobId, flaggedAt}` per kind ("Objects need re-segmenting"); cleared when the kind is attached again (0009, docs/SCENE_OBJECTS.md §8)                                                                                                                                                                |

## Runtime scale

A phone video registered before the pipeline estimated scales (`scaleSource: unresolved`)
is drawn at one COLMAP unit to the metre: two to eight times too big, and its files cannot
be resized in place. `renderConfig.scale` is the factor the viewer draws such a splat at
instead, about its tileset's root transform origin (the placed coordinate), **relative to
the model as registered**: absolute, so 0.8 and then 0.5 is 0.5, and 1 is as registered.
`renderConfig.scaleEvidence` says how it was found.

`PUT /api/v1/assets/{id}/scale` (write token) sets it: `{scale, evidence?}`, or
`{reset: true}`. `scale` is 0.01–100. `evidence.method` is one of

| method                    | numbers                                                                                         | provenance `scaleSource`                             |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `measured-length`         | `measuredLengthM` (as drawn at `measuredAtScale`, default 1), `trueLengthM`; must imply `scale` | `manual`                                             |
| `camera-height-estimate`  | `uncertaintyPct` (required), `cameraHeightM`, `cameraHeightUnits`, `metresPerUnit`, `cameras`   | `camera-height-estimate`, with `scaleUncertaintyPct` |
| `direct` (or no evidence) | —                                                                                               | `manual`                                             |

The server adds `setAt` and the provenance the asset was registered with
(`registeredScaleSource`, `registeredScaleUncertaintyPct`), carried from one scale to the
next: a reset restores it — `unresolved` for these scans — and drops `scale` and the
evidence. The capture row's `scaleSource` is never changed: it describes the capture's
files (`canonical.ply`, which `fetch_capture.py` reads), and a runtime scale does not
touch them.

Everything the catalog places on the globe moves with the scale, about the same origin, by
the new scale over the old: the site's `boundary` (and so `areaM2`) and `centroid`, the
asset's `footprint` and its `groundSamples` (across and in height over the origin). So
every position the API gives already describes the scaled model, and the viewer scales only
what is drawn in the tileset's own frame — its tiles and sidecars. A boundary somebody
redrew is resized as drawn, not replaced. `PATCH /assets/{id}` keeps `scale` and
`scaleEvidence` whatever its `renderConfig` says, because only this endpoint moves the
boundary with them.

Only the splat a pipeline run registered can be resized: the site's first gaussian-splat
asset, a 3D Tiles URL, on a site whose `metadata.registration.georef` records the origin.
Anything else is a 409 with `code: not_scalable`. A re-run that registers new tiles resets
the scale (the new tiles carry the run's own, in the provenance that replaces the old) and
resizes the boundary back; the site's `metadata.registration` becomes the new run's, so the
next scale is set about the origin those tiles are placed at. Unscaled assets store neither
key, so code from before this can still read every asset nobody resized; reset scaled ones
before rolling back past it.

**Backfill.** `tools/captures/estimate_scale.py` runs the pipeline's own camera-height
estimate (`tools/pipeline/scale_estimate.py`, levelled by `sfm.camera_up` as
`exif_gps` levels a video) on the pose model a past run left — `runs/<job id>/pose/poses/`
in the private bucket — and prints the estimate, its evidence and the request it would
send. `--apply --api-url … --token …` sends it; a dry run is the default, the write is
never defaulted to production, and a scan whose scale was measured, already estimated or
set by hand is refused unless `--force`. See the script's docstring.

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
- **Provenance** `{ sourceOrganization?, sourceUrl?, publishedAt?, notes?, georefMethod?, scaleSource?, uncertaintyM?, scaleUncertaintyPct? }`.
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
