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

`ClippingManager` owns one `ClippingPolygonCollection` on the globe and one on the global
3D tileset (OSM Buildings or Google Photorealistic when enabled). Polygons are immutable in
CesiumJS ≥ 1.145, so each footprint is tracked by key and swapped, never mutated.

What gets cut depends on the representation:

- **Meshes and point clouds** (opaque) cut both the globe and the world tileset, so the two
  surfaces never z-fight. With `clipFootprint: "tileset"` the footprint is the union of the
  finest loaded tile boxes below the first branching level (`coverageFromTileset`, refreshed
  as sub-tilesets stream in), otherwise the catalog footprint.
- **Gaussian splats** cut only the world tileset and keep the terrain. A splat is blended
  over the opaque globe with a depth test, so the ground layer simply covers the imagery and
  nothing fights. Cutting the terrain was tried first and leaves a see-through hole to space
  wherever the capture is sparse: tile bounding boxes include outlier splats, so no
  tile-derived footprint is tight enough to avoid it.

Site tilesets also set `enableCollision: true` so the camera collides with the model surface
where Cesium can sample it, and `minimumZoomDistance` is 0.6 m so a scroll cannot pass
through a splat surface that has no collision geometry.

## Representations and LOD

`SiteManager` creates one `Cesium3DTileset` per asset on demand, keeps inactive
representations loaded but hidden (`preloadWhenHidden`) so switching is instant and the
camera never moves. Sites load automatically when the camera comes within ~40 km and unload
beyond ~400 km. Point clouds get attenuation + eye-dome lighting.

The viewer runs in **request-render mode** (`requestRenderMode: true`,
`maximumRenderTimeChange: ∞`): a frame is drawn only when the camera moves, tiles arrive, or a
manager calls `scene.requestRender()` after mutating the scene. An idle view costs nothing on
the GPU, and a Gaussian splat is not re-sorted every 16 ms while nobody is touching it.

`PerformanceManager` counts rendered frames on `postRender` and every 500 ms runs
`decideScreenSpaceError` (pure, unit-tested) within the active preset's bounds:

| Preset      | Base SSE | Adaptive range | Resolution                | MSAA |
| ----------- | -------- | -------------- | ------------------------- | ---- |
| Performance | 24       | 12–48          | browser-recommended       | off  |
| Balanced    | 16       | 6–32           | browser-recommended       | 4×   |
| Ultra       | 8        | 2–16           | native device pixel ratio | 4×   |

Rules, highest priority first: tileset memory above 125 % of its cache budget → coarser;
camera moving → coarser; < 28 fps → coarser; at rest within 600 m of a site (idle, or
measured > 52 fps) → refine one step per tick towards the minimum; idle elsewhere → back to
the preset base. Idle is "fewer than 6 frames in the last second", which in request-render
mode means nobody is touching the view, so it is headroom by definition. Each SSE change
calls `scene.requestRender()`; tile selection only runs inside a frame.
Two render profiles: **rest** (native device pixels, resolution scale 1, MSAA 4×) for still
frames, which in request-render mode are drawn once and can afford it; **motion**
(browser-recommended resolution, adaptive scale down to 0.5, MSAA shed first) while the
camera moves. Frame rate is only treated as evidence about motion cost while moving; slow
frames at rest are tiles arriving. Gaussian splats never refine below SSE 12 / 8 / 4
(performance / balanced / ultra): they are sorted on the CPU every camera change, so their
cost grows with splat count far faster than a mesh. Tile cache budgets come from
`navigator.deviceMemory` (256/384/512 MB + overflow). A per-asset `maximumScreenSpaceError`
acts as a quality floor. Manual SSE in Settings › Advanced disables adaptation.

While the camera moves the root element carries `data-moving`; glass panels drop their
backdrop blur for the duration (a full-screen pass per panel otherwise) and use a flat tint.

Things that are deliberately _not_ done per frame: hover picking waits until the pointer has
rested 120 ms and never runs while the camera moves (each `scene.pick` is a render pass);
overlay anchors use `globe.getHeight` (a CPU lookup) while moving and call `sampleHeight`
only at rest, once per anchor every few seconds; the camera pose is throttled to 10 Hz.

## Camera

`CameraController` clamps `minimumZoomDistance` to 5 cm so centimetre data can be inspected,
keeps terrain collision on, and derives flight durations from distance (1.2–5.5 s,
quadratic in/out). `flyToBoundingSphere` with an oblique offset is the standard arrival.
Cesium's default double-click entity tracking is removed; double-click flies halfway to the
clicked point instead.

## Explore mode

`ExploreController` pauses `ScreenSpaceCameraController.enableInputs` and moves the same
camera with WASD/QE (+Shift), drag-to-look, wheel-to-move and Shift+wheel to change speed,
keeping ≥0.3 m above the globe. Escape or the HUD exits. While it is on, ordinary
drag-to-pan and scroll-to-zoom are intentionally off, which is why the HUD stays visible.
Its tick runs on `scene.preUpdate` (raised every widget tick) rather than `preRender`, so it
keeps working in request-render mode. No second renderer or camera exists.

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
