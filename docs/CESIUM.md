# Cesium notes

Built against **CesiumJS 1.145** (September 2026) using the official
[cesiumjs-skills](https://github.com/cesiumgs/cesiumjs-skills) agent skills and the current
Sandcastle sources. Everything below was verified against that version.

## Viewer setup

`CesiumSceneManager` creates one `Viewer` with every stock widget disabled, `scene3DOnly`,
`baseLayer: false` (imagery is a catalog layer), MSAA 4×, FXAA, `depthTestAgainstTerrain`
and a WebGL2 context. The credit display is restyled into a glass chip but never hidden;
provider credits (including the ion evaluation-token notice) stay visible.

## Tokens

- `VITE_CESIUM_ION_ACCESS_TOKEN` empty → CesiumJS's built-in evaluation token is used.
  It is rate-limited and intended for development; the UI shows a notice.
- Create your own token at <https://ion.cesium.com/tokens> with only `assets:read` and
  `geocode`, and restrict _Allowed URLs_ to your origins.
- Authorization failures from ion (401/403) mark the token invalid in the UI with the
  exact variable to fix; the globe keeps running on the Natural Earth II basemap that
  ships inside CesiumJS.

## The demo asset

Ion asset **4547222** is the tileset used by the official _3D Tiles Gaussian splats with LOD_
Sandcastle. Its root oriented bounding box spans roughly lon −122.1433…−122.1205,
lat 47.6352…47.6605 (≈1.7 km × 2.8 km near Redmond, WA) at ~109 m height. The seeded
default bookmark reproduces the Sandcastle's `viewBoundingSphere(heading 100°, pitch −25°,
range 500 m)`. Loading uses the standard `Cesium3DTileset.fromIonAssetId`; CesiumJS handles
`KHR_gaussian_splatting` content automatically (WebGL2 required).

## Embedding a site: clipping

`ClippingManager` keeps one `ClippingPolygonCollection` on `scene.globe` and, when the
photorealistic world is active, one on that tileset. Each site contributes polygons (with
holes) derived from either its catalog footprint (`clipFootprint: "catalog"`) or the loaded
tileset's root bounding volume (`"tileset"`: region rectangle, OBB horizontal corners, or
sphere circle). Polygons are immutable since 1.145, so we swap them by key instead of
mutating. Clipping requires WebGL2; without it the manager logs and degrades to no
clipping.

## Representations and LOD

`SiteManager` creates one `Cesium3DTileset` per asset on demand, keeps inactive
representations loaded but hidden (`preloadWhenHidden`) so switching is instant and the
camera never moves. Sites load automatically when the camera comes within ~40 km and unload
beyond ~400 km. Point clouds get attenuation + eye-dome lighting.

`PerformanceManager` samples frame time on `postRender` and every 500 ms decides a
`maximumScreenSpaceError` within the active preset's bounds:

| Preset      | Base SSE | Adaptive range | Resolution                   |
| ----------- | -------- | -------------- | ---------------------------- |
| Performance | 24       | 12–48          | browser-recommended, no FXAA |
| Balanced    | 16       | 6–32           | browser-recommended          |
| Ultra       | 8        | 2–16           | native device pixel ratio    |

Rules: camera moving → coarser; sustained < 28 fps → coarser and, after 2 s below 22 fps,
lower `resolutionScale` (min 0.66); stationary close-up above 52 fps → refine towards the
minimum. A per-asset `maximumScreenSpaceError` acts as a quality floor. Manual SSE in
Settings › Advanced disables adaptation.

## Camera

`CameraController` clamps `minimumZoomDistance` to 5 cm so centimetre data can be inspected,
keeps terrain collision on, and derives flight durations from distance (1.2–5.5 s,
quadratic in/out). `flyToBoundingSphere` with an oblique offset is the standard arrival.
Cesium's default double-click entity tracking is removed; double-click flies halfway to the
clicked point instead.

## Explore mode

`ExploreController` pauses `ScreenSpaceCameraController.enableInputs` and moves the same
camera with WASD/QE (+Shift), drag-to-look and wheel-to-change-speed, keeping ≥0.3 m above
the globe. Escape or the HUD exits. No second renderer or camera exists.

## Picking and measuring

Selection uses `scene.pick` + `scene.pickPosition` (terrain fallback via `globe.pick`),
resolves `Cesium3DTileFeature`, entities and tilesets to catalog objects, highlights them
with the accent colour, samples terrain with `sampleTerrainMostDetailed`, and marks the
point with a small entity. Measurements are entities with `CallbackProperty` geometry;
distances use `EllipsoidGeodesic` (ground) and `Cartesian3.distance` (3D), areas use
spherical excess on lon/lat.

## Photorealistic world

Behind `VITE_ENABLE_PHOTOREALISTIC=true`, the Google Photorealistic 3D Tiles catalog layer
(via `createGooglePhotorealistic3DTileset`) replaces the globe surface, receives the same
site clipping, and switches the ion geocoder to Google as Google's terms require. It is
labeled as visual context only.

## API changes noted while building

- `ClippingPolygon` positions are frozen (1.145): rebuild instead of mutating.
- `ClippingPolygonCollection.quality` / `destroy` are deprecated (1.145); not used.
- `Cesium3DTile.boundingVolume` is not in the public typings; `footprintFromTileset` reads
  it defensively and falls back to `boundingSphere`.
- CesiumJS 1.144 added composable camera `Controller`s; the explore mode here is a
  minimal purpose-built controller and can migrate to that framework later.
