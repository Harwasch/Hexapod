# Data model

All geometry is GeoJSON in WGS 84 (EPSG:4326) on the wire and `geometry(…, 4326)` in
PostGIS. All timestamps are ISO 8601 with time zone. Wire field names are camelCase; the
Python models are snake_case (`CamelModel` handles both).

```text
sites 1 ──── * assets            (site_id nullable: assets can exist before a site)
sites 1 ──── 0..1 site_frames    (the fixed origin every epoch is placed about)
sites 1 ──── * epochs 1 ──── * assets (revisions of one epoch; epoch_id nullable: legacy)
epochs 1 ──── * epoch_maps * ──── 1 assets (one map per revision and kind: confidence, surface)
sites 1 ──── * site_check_targets (surveyed ground marks; retired, never deleted)
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
`representations`, `latestObservedAt` (the newest `observed_at` of its assets, leaving out an
epoch's superseded revisions) and `quality`.

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
| epoch_id                          | uuid → epochs (set null), nullable | the epoch this asset is a revision of (0032); null for seeded, catalog and hand-made assets                                                                                                                                                                                                                                                                                                   |
| source_kind                       | text                               | `observed` (the capture) · `inferred` (a fill, M5); one asset is one tileset of one epoch and one source kind                                                                                                                                                                                                                                                                                 |

**Superseded = `valid_to` set on an epoch asset.** A re-run of a capture inserts a new revision of
its epoch and sets the old row's `valid_to` and `default_visible = false`; nothing else of the old
row changes, its URL included. `SiteRead.assets` lists only what is not superseded; `GET /assets`
lists every row. On a legacy asset (`epoch_id` null) `valid_to` is only a validity window.

What a splat scan carries beside its tiles is declared on its tileset's root, not in these
columns: `extras.instances`, `extras.skin`, `extras.inferredLayers`, `extras.viewCones`,
`extras.collision`, `extras.nativeLod`, and `extras.variants` -- other methods' objects, fills
and skins for the same scan, for the owner's bake-offs (docs/SCENE_OBJECTS.md §4, "Root
extras" and "Variants"; the table of kinds in §8).

## site_frames

A site's fixed east/north/up frame (decision D3, ADR 0013): 1:1 with the site, never updated.

| Column       | Type                       | Notes                                                                                                                        |
| ------------ | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| site_id      | uuid → sites (cascade), PK |                                                                                                                              |
| lat, lon     | float8                     | the WGS84 origin every epoch's tiles are placed about                                                                        |
| height       | float8                     | the origin's height                                                                                                          |
| height_datum | text                       | `ellipsoid-wgs84` · `msl-unverified` (every frame today: no geoid applied)                                                   |
| epsg         | int                        | the origin's WGS84 UTM zone (32601–32660, 32701–32760)                                                                       |
| source       | text                       | `registration` (its first epoch) · `backfill` (0032) · `echo` (re-made from a run's echoed frame after its site was deleted) |
| created_at   | timestamptz                |                                                                                                                              |

Read as `SiteRead.frame` (null for a legacy site).

## epochs

One dated capture of a site, in the site's frame.

| Column                 | Type                               | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id                     | uuid                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| site_id                | uuid → sites (cascade)             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| capture_id             | uuid → captures (set null), unique | one capture per epoch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| job_id                 | uuid → jobs (set null)             | the run behind the epoch's current revision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ordinal                | int ≥ 1, unique per site           | 1 for the capture that defined the frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| observed_at            | timestamptz                        | when it was captured                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| observed_at_source     | text                               | where `observed_at` came from, first that applies: `manifest` (the run manifest's tz-aware `capture.capturedAtUtc`, the files' own clocks; read before the capture row and never copied into it) · `captured-at` (the capture row's `captured_at`) · `manifest` (the manifest's naive `capturedAt`, as UTC) · `created` (the upload time). A re-run dates a `created` epoch by the same rule, as it does an epoch a quality gate had rejected that it now publishes; any other date stays, and the epoch's current asset revision takes the epoch's date; `registration._observed_at` |
| footprint              | MultiPolygon 4326 (GiST)           | the run's measured extent in the frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| platform               | text                               | `rtk-drone` · `drone` · `phone-video` · `phone-photo` · `unknown`: what `normalize` detected from the frames (`capture_platform`, ADR 0014), written at every registration of the epoch; a run that detects nothing or `unknown` keeps a known value, and a first run with none falls back to the capture row (a phone-key upload is `phone-video`); `rtk-drone` only when the frames showed it (`platformEvidence: metadata`), a claimed one is written as `drone`                                                                                                                   |
| status                 | text                               | `published` · `rejected` (a quality gate the recipe turned on failed its run; no asset, and a later run of the capture that passes publishes it) · `superseded` (not written yet)                                                                                                                                                                                                                                                                                                                                                                                                     |
| join_method            | text                               | `explicit` (the capture named the site) · `spatial` (matched before its run) · `new-site`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| join_point_source      | text, nullable                     | a `spatial` join's point: `video` (the location its video's container carried, ADR 0013 "Joining a site by the video's GPS") · `upload` (the one sent with the upload); null for `explicit` and `new-site`                                                                                                                                                                                                                                                                                                                                                                            |
| join_point_accuracy_m  | float, nullable                    | the horizontal accuracy the phone wrote with the video's fix, metres                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| alignment              | jsonb, nullable                    | how it was registered to its site (M3b, `epoch_alignment.json` less its per-image residuals): method, reference epoch, similarity, residuals, check targets, tier, `verdict` (`survey` · `coarse` · `unaligned`), confidence, `uncertaintyM` and `label`. The web reads only `label`, `verdict` and `uncertaintyM`; `survey` sits only on an `rtk-drone` epoch. A run with no align stage (a splat upload) records `method: none`, `unaligned`, and why; a rejected epoch, `unaligned` with the gate as its reason and a `rejected` block                                             |
| quality                | jsonb, nullable                    | a `CaptureQuality` dump                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| created_at, updated_at | timestamptz                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

A rejected epoch's footprint is its run's join point (the video's GPS, else the located point)
buffered by 25 m, else the site's
boundary; one is recorded only on a site with a frame (a frameless site's first epoch defines it).

An epoch aligned visually (`alignment.method: visual`, not `unaligned`) has `georefMethod:
visual-epoch` on its capture and asset, and `scaleSource: visual-epoch` when its reference's scale
was measured (`arkit`, `exif-gps`, `visual-epoch`); otherwise it inherits the reference's word.

Read as `SiteRead.epochs` and `GET /api/v1/sites/{id}/epochs`, each with `currentAssetId`: the
epoch's newest observed asset with no `valid_to`, and `confidence`: that revision's confidence map
summarised (`{status: measured, assetId, url, cellM, level, policy, weakShare, weakCells,
scoredCells, unseenCells, reasons, alignmentCap {uncertaintyM, factor, priorEpochs, source}}`), or
null — not measured; and `surface`: that revision's height surface summarised (`{status: measured,
assetId, url, cellM, level, policy, calibration, cells, surfacedCells, medianSigmaM,
originHeightM, groundWindowM}`), or null — not measured. A capture names its site with `siteId` on
`POST /captures` (or the phone route), kept in `captures.metadata.siteId`; `captures.site_id` is
still the site a run registered it into.

### Epochs on the globe

The viewer reads a site's epochs from `SiteRead.epochs` and picks what it shows with one rule,
used by the globe, its controls and view.html alike (`apps/web/src/lib/epochs.ts`
`pickShownAsset`): an epoch is offered when it is `published` and has a current asset of the
representation shown (`currentAssetId` first, else the same rule recomputed for that
representation; never an `inferred` asset). The latest offered epoch is shown unless another
is chosen in the capture-date control (`TimelineControl`), which lists each by its UTC date,
platform and alignment. Sites without epochs keep the rule from before them: the
default-visible asset, else the newest observed.

Every date of a site is drawn with one model matrix: that of the offered epoch with the lowest
ordinal (the one that made the frame), placed and clamped to the ground as any asset is; the
others copy it exactly and are never clamped on their own, so the vertical difference between
two dates is the data's own and a new date never moves the old ones. Copying needs the two
tilesets' root transforms to be one frame (`sameRootFrame`: rotation elements within 1e-9,
translation within 1 mm, from first principles: 10 µm over 10 km, and four SPZ quanta against
~1e-9 m of ECEF rounding). A date that fails it, or whose anchor did not load, is placed on its
own and the console log says so; the Inspector's "Date shown" says it is placed on its own.
Hidden dates stay loaded (instant switching under CesiumJS) but never preload tiles.

Alignment is filled by M3. The viewer reads three keys of `alignment` only: `verdict`
(`survey` · `coarse` · `unaligned`), `uncertaintyM`, and a ready-made `label`, preferred when
present. Until then every date but the frame's own reads "Not aligned yet: placed by its own
GPS".

## site_check_targets

A site's surveyed ground marks (M3b): what an epoch's alignment is scored against. A changed mark
retires its row and gets a new one, so what an epoch was scored with stays readable.

| Column                 | Type                   | Notes                                                                          |
| ---------------------- | ---------------------- | ------------------------------------------------------------------------------ |
| id                     | uuid                   |                                                                                |
| site_id                | uuid → sites (cascade) |                                                                                |
| label                  | varchar(64)            | one active row per label (partial unique index on `retired_at IS NULL`)        |
| role                   | text                   | `check` · `control` (stored and scored like `check` until BA takes control)    |
| lat, lon, height       | float8                 | WGS84, the height from `height_datum`                                          |
| height_datum           | text                   | `ellipsoid-wgs84` · `msl-unverified`                                           |
| sigma_h_m, sigma_v_m   | float8 > 0             | the survey's own one-sigma: they set the 3-sigma tolerance an epoch is held to |
| source                 | varchar(64)            | `rtk-rover`, `total-station`, `synthetic`, ...                                 |
| surveyed_at            | timestamptz, nullable  |                                                                                |
| retired_at             | timestamptz, nullable  | set when re-surveyed or left out of a replacement; never deleted               |
| created_at, updated_at | timestamptz            |                                                                                |

`GET /api/v1/sites/{id}/check-targets` lists the active set and `PUT` replaces it, both behind the
write token (survey marks on private land are not public, unlike the site's other reads):
`{targets: [{label, role, lat, lon, height, heightDatum, sigmaHM, sigmaVM, source, surveyedAt?}]}`.
A run is given the set as it is when it starts (`site_context`), with the capture's own clicks
from an uploaded `check_observations.json` (`{"version": 1, "observations": [...]}`).

## epoch_maps

An epoch revision's per-cell maps (A4, migration 0035; A5, 0036): one row per asset and kind, recorded by
`register` from the grid the run shipped beside its tiles. A re-run is a new asset with its own
map; the superseded revision keeps the one it was measured with.

| Column     | Type                           | Notes                                                                                   |
| ---------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| id         | uuid                           |                                                                                         |
| epoch_id   | uuid → epochs (cascade), index |                                                                                         |
| asset_id   | uuid → assets (cascade)        | the revision; unique with `kind`                                                        |
| job_id     | uuid → jobs (set null)         | the run that measured it                                                                |
| kind       | varchar(20)                    | `confidence` · `surface` (M5)                                                           |
| format     | varchar(40), version int       | `hexapod.cellgrid` v1                                                                   |
| url        | text                           | the file beside the tiles, in the generation the asset shows (an attach moves it along) |
| cell_m     | float8 > 0, level int          | the finest level                                                                        |
| descriptor | jsonb                          | the tileset's root `extras.confidence` / `.surface`, verbatim, checked when recorded    |
| summary    | jsonb                          | `descriptor.summary`                                                                    |
| created_at | timestamptz                    |                                                                                         |

A descriptor the API cannot vouch for (another container, a `uri` that is not a file beside the
tiles, a cell off the lattice, missing summary keys) is logged and not recorded: the epoch reads as
not measured. A surface must also carry its five channels at their types and units, `noData`
−32768 and `vertical.originHeightM`. `confidence.bin` and `surface.bin` are sidecars of kinds
`confidence` and `surface` (bound to the tiles, not attachable).

- `POST /api/v1/sites/{id}/epochs/{epoch}/weak-areas` (write token) `{budgetM2?, focus?: [lon, lat], maxAreas
1..8 = 3}` → `{epochId, assetId, policy, cellM, level, weakCells, totalCells,
truncatedByBudget, heuristic, areas: [{footprint, areaM2, cells, meanScore, reasons, reliefM,
diameterM, pattern (nadir · oblique · orbit), targetGsdM, gimbalPitchDeg, hint}]}`. 409
  `confidence-not-measured` · `confidence-unreadable` · `no-site-frame`.
- `GET /api/v1/sites/{id}/fill-targets?epochId=` (write token) → `rescan-gate-v1`: `{siteId,
epochId, weakIn, comparedWith (at most the eight most recent earlier epochs), capped, frame: tileset, cellM, level, i0, j0, ni, nj, rowBytes, bitOrder: little, mask
(base64, rows south to north, at the epoch's own finest level), cells, satisfiedBy (a byte per
target: which of `comparedWith` it re-scanned), reason}`. 409 `confidence-not-measured` · `no-rescan-yet`.
- `GET /api/v1/sites/{id}/change?from=&to=` (write token, cell-map rate limit) → `change-v1` (ADR 0018
  § API): `{policy, derived: true, label, calibration: uncalibrated, siteId, fromEpochId,
toEpochId, fromAssetId, toAssetId, frame, level, nativeLevel, cellM, thresholdSigma: 3,
uncertainty {fromM, toM, fromVerdict, toVerdict, fromFrameM, toFrameM}, ground, vegetation,
appearance {status: not-computed, reason}}`; each layer (`ground`, the bare earth, primary;
  `vegetation`, the top surface at its own higher threshold) is `{layer, label, heights (ground ·
dsm), minDetectableM, detectable {minM, medianM, p90M, maxM, cells} (null: none compared),
detectableGrid {level, cellM, i0, j0, ni, nj, unitM: 0.01, noData: 65535, encoding:
u2le-base64-rows-south-to-north, data} (each cell's least detectable change, at areaLevel),
motionM, horizontalRadiusM, driftReachM, counts {unchanged, raised, lowered,
insufficient}, lowConfidence {unchanged, raised, lowered}, areaLevel, areaCellM, areas (raised,
lowered and insufficient only; unchanged cells are counted, not outlined): [{class, footprint
(MultiPolygon), cells, areaM2, lowConfidence, lowConfidenceCells, meanDeltaM, maxAbsDeltaM}]}`. 404
  · 422 (same epoch) · 409 `epoch-not-published` · `no-site-frame` · `surface-not-measured` ·
  `confidence-not-measured` · `epoch-unaligned` · `not-metric` · `scale-corrected` · `frames-differ`
  · `surface-unreadable` · `confidence-unreadable`.

## Confidence grid

Per epoch revision, how well each part of the capture was seen (ADR 0016), recorded per asset in
`epoch_maps`: a file beside the tiles, `confidence.bin`, declared on the tileset's root as
`extras.confidence` and copied to the top level of `registration.json` as `confidence` (null:
not measured — every `splat-ingest` epoch).

- **Lattice.** Tileset frame (the site frame), x east, y north; cell `0.25 m × 2^level`,
  `i = floor(x / cell)`, `j = floor(y / cell)`. Levels nest; epochs compare at the coarser level.
- **Container** `hexapod.cellgrid` v1 (`tools/captures/cell_grid.py`): gzip of the level planes,
  finest first, `nj × ni` row-major, channels interleaved little-endian. The descriptor carries
  `format`, `version`, `kind`, `uri`, `frame`, `lattice {baseM, level}`, `cellM`,
  `channels [{name, type, unitM?}]`, `levels [{level, cellM, i0, j0, ni, nj, offset, bytes}]` (into
  the uncompressed planes), the kind's keys, and `sha256` of the uncompressed planes.
- **Confidence** keys: `reasonBits` (`views 1, spread 2, gsd 4, accuracy 8, unseen 16,
alignment 32`), `policy: weak-v1`, `weakBelow: 128`,
  `alignmentCap {uncertaintyM, priorEpochs, factor, source}`, `roiDisc {centerM, radiusM}` (where
  `unseen` was expected), `summary {cells, scored, weak, unseen, weakShare, reasons, gaussians,
outside, medianGsdMm, metric}` (finest level; `weakShare` is weak over scored + unseen;
  `reasons` counts weak cells per bit).
- **Channels:** `score` u1 (0 no evidence, else `1 + floor(254 · keepShare · f)`), `reasons` u1,
  `relief` u1 (height range of the cell's gaussians, 0.25 m, saturating). Coarser levels are
  min-of-children (min of scored children, OR, max). Weak: `0 < score < 128` or `unseen`.
- **Per tile:** `extras.confidenceMin` (after `gaussians`): the lowest score of the finest-level
  cells the tile's box covers in plan — a scored cell at its score, an unseen cell at 0, an empty
  cell outside the pointed disc not counted; absent when none counts or no grid was shipped.
- **Pipeline artifacts:** `quality` → `support/` (`support.json` + one raw column per field,
  row for row with `trained.ply`, COLMAP frame); `confidence` → `confidence/confidence.json` and
  `confidence.bin`.
- **Fixture:** `data/tiles/synthetic-yard-epochs/` (two grids over the synthetic yard, with
  `grids.json`), the cross-language decoder check.

## Height surface

Per epoch revision, the heights change between epochs is measured in (ADR 0018), recorded per
asset in `epoch_maps` (kind `surface`, A5): `surface.bin` beside the tiles, declared on the root as
`extras.surface` (after `extras.confidence`) and copied to the top level of `registration.json` as
`surface` (null: not measured).

- **Container** `hexapod.cellgrid` v1, kind `surface`, on the epoch's confidence grid's finest
  plane exactly (same `lattice.level`, `i0`, `j0`, `ni`, `nj`). One level.
- **Channels:** `dsm` i2 and `ground` i2 (centimetres above the tileset frame's origin height,
  along its up: a tangent plane, not a datum; `-32768` = no surface, or in `ground` alone under a
  surface, ground never observed), `sigma` u1 (centimetres,
  `max(1 cm, (z1 − zm)·FEW(n)·RMS2(n)/SPAN(n))`, dsm's bias included, widened for 3–6 samples, 255 = at least 2.55 m, 0 = no
  surface), `support` u1 (bit 1 SfM points,
  bit 2 splat centres; high nibble `floor(log2 samples)`, at most 15), `groundSigma` u1
  (centimetres, ground's own: `max(1 cm, s_low·RMS2(N))` over the window's `N` samples, `s_low` the noise of the cell that set
  the minimum, 255 = at
  least 2.55 m, 0 = no ground).
- **Keys:** `policy: surface-v1`, `status: uncalibrated`, `estimators {dsm, ground, sigma,
minSamples, groundWindowM, groundWindowCells, groundUnobserved, groundMaxGrade, groundSigma,
samples}`, `noData`, `sourceBits`,
  `sigmaSaturatesAt`, `vertical {zero, originHeightM, surface}`, `alignedTo {kind: confidence,
level, sha256}`, `summary {cells, surfaced, thin, samples {sfm, splat}, outside, nonFinite,
outOfRange, sigmaSaturated, groundUnobserved, groundSigmaSaturated, medianSigmaCm, metric}`.
- **Pipeline artifact:** `surface` → `surface/surface.json` and `surface.bin`.
- **Fixture:** `data/tiles/synthetic-yard-surfaces/` (two surfaces on the planes of
  `synthetic-yard-epochs`, with `surfaces.json` and P5's demo reference change), the API decoder's
  and change map's cross-language check.

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

Only the splat a pipeline run registered can be resized. An epoch's asset is resized about its
site's frame (`site_frames`), and only when it is the current revision of the site's one published
epoch: every epoch is placed in one frame, so resizing one of two would move it against the other.
A legacy asset is the site's first gaussian-splat asset, a 3D Tiles URL, on a site whose
`metadata.registration.georef` records the origin. Anything else is a 409 with
`code: not_scalable`. A re-run that registers new tiles adds a new
revision with no scale (the new tiles carry the run's own, in the provenance that replaces the
old) and resizes the site's boundary back; the superseded revision keeps its scale as history; the next scale is set about the
origin those tiles are placed at: for an epoch asset the site's frame, which never moves; for a
legacy asset the site's `metadata.registration`, which becomes the new run's. Unscaled assets store neither
key, so code from before this can still read every asset nobody resized; reset scaled ones
before rolling back past it.

**Viewer.** `SiteManager` draws the asset at `renderConfig.scale` about its tileset's root
transform origin (the translation of the root's declared transform; a root that places nothing
on the globe is resized about its own centre instead): the model matrix is
`T(lift) · R · S(scale) · R⁻¹`, which for a uniform scale is `[scale·I | (1 − scale)·origin +
lift]` (`apps/web/src/cesium/placement.ts` `scaledModelMatrix`, `cesium/tilesetScale.ts`). At
scale 1 that is exactly the translation the clamp always set. The clamp then rests the scaled
model: on the catalog's `groundSamples` exactly as sent -- they already describe the scaled
model, and are never rescaled by the saved scale a second time -- or, without them, on the root
box's lowest point scaled about the origin (`h_origin + scale · (h − h_origin)`). Nothing else
the catalog places is rescaled: the boundary clips and outlines, the footprint clips.

Everything drawn in the tileset's own frame follows the root's computed transform, which now
carries the scale: the splats (CesiumJS re-bakes them; its slow path handles a non-rigid
transform), instance spheres, split objects, view cones, telemetry. What crosses between that
frame and the globe by hand was written for a rigid transform and now uses its true inverse
(`inverseScaledTransformation`) and converts lengths by the scale (`uniformScale`): the
collider's rays, distances, search radii and clearance (`SplatCollider`, so walking, zooming to a
surface and measuring on a scan are in metres as drawn), the dedicated renderers' camera
(`scanView/pose.ts`: near and far planes, unit direction and up) and their tile culling and
prefetch (`ScanRendererHost`), CesiumJS's pick radii (`cesiumPickSource`: the engine bakes a
splat's scales by the bake's scale; the picker works un-baked) and scene selection's comparison
of hits across scans and its nearest-label radius (`SceneSelectController`).

`SiteManager.previewScale(siteId, scale)` draws another scale without saving it, `null` puts the
catalog's back: the scan rests at once on what its last clamp sampled, the ground samples and the
clip footprint moved by the preview's ratio to the saved scale (as the API will move them), and
is clamped afresh once the scale has held for a quarter of a second. A fresh site record whose
asset was resized (`watchSiteRecords` → `updateRecord`) drops any preview and re-places the scan
at the catalog's new scale on its new ground, with the new boundary.

**Set real size** (`apps/web/src/features/inspector/RealSize.tsx`) is on the site card for the
asset the API can resize (`lib/realSize.ts` `scalableAsset`, the API's own rule; nothing else is
offered the tool). Measure on the scan: two points where the view meets the scan's solids
(`cesium/ScaleMeasure.ts`; a click or a tap, or "Mark the centre of the view" from the keyboard),
the length between them as drawn now, and the true length typed in (`1.8`, `180 cm`, `6 ft`)
make the scale `measuredAtScale × trueLengthM ÷ measuredLengthM`, sent as `measured-length`
evidence with exactly those numbers. Or the factor typed in (`direct`), or Reset to registered
(`{reset: true}`). Every change is previewed; Save sends it, a 401 asks for the write token and
sends it again, Escape or Cancel puts the scan back. The saved asset goes straight into the
site's record, so the scene and the card's Placement row ("scaled by hand from a measured
length", "scale set by hand", the estimate's "scale estimated from camera height (±N%)") follow
at once, and the record is refetched for the boundary and centroid the API moved.

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

## scan_requests

Site scans requested from the landing page's form (`POST /api/v1/scan-requests`, open; read with
the write token, `GET`, and in the admin console's Requests view). Columns: `name`, `email`,
`notes`, `boundary` (`geometry(MULTIPOLYGON, 4326)`, nullable: acres only), `parcel_ids`
(text[]), `outline_unvalidated` and `raw_outline` (jsonb; an outline kept as sent when it could
not be cleaned in time), `created_at` (indexed).

Kept indefinitely, with no automatic deletion: a request is deleted only when the person who
sent it asks (Harrison, 2026-10-20).

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
