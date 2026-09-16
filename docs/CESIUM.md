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

Tilesets never set `enableCollision`: Cesium then ray-casts every loaded tile's triangles on
the CPU every frame to find the height under the camera (measured 130 ms per frame on the
Google world and 400 ms beside the AGI drone mesh). The camera floor over meshes is
`CameraController.keepAboveDrawnSurface`: 250 ms after a gesture ends, one depth sample
(`scene.sampleHeight`) under the camera, and a 0.35 s ease back up when the camera ended up
below the surface plus the zoom floor. Wheel zoom already stops at the surface under the
cursor. Terrain collision stays on for the globe (cheap, CPU heightmap).

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
| Performance | 16       | 4–48           | browser-recommended       | off  |
| Balanced    | 8        | 2–32           | native device pixel ratio | 4×   |
| Ultra       | 4        | 1–16           | native device pixel ratio | 4×   |

Smoothness comes first, the way a maps app does it: nothing about the render settings
changes during a gesture. Tile selection is frozen while the camera moves (every change of
`maximumScreenSpaceError` pops tiles mid-drag), and slow frames at rest never coarsen
anything (they are tiles arriving, not a stall). At rest the scene uses the idle time the
way Google Maps does: once nothing is loading (sites and the world both report through
`reportLoading`) and tileset memory is under 70 % of its budget, the error walks one step
finer per 500 ms tick, at any height, down to the preset minimum (2 px on balanced: a 2 to
5 cm survey mesh only shows its detail there, measured at 176k triangles and a Google-like
softness at 8 px versus 416k and the real detail at 2 px). Nothing returns to the base
on its own; finer tiles stay until memory pressure (125 % of budget) coarsens, so the next
gesture starts from what is already loaded. Each SSE change calls `scene.requestRender()`;
tile selection only runs inside a frame.

Resolution and anti-aliasing are constant for still and moving frames alike, and adapt only
on evidence, one ladder step at a time: a frame rate under 26 fps sustained for 1.2 s _while
moving_ drops MSAA, then resolution scale (0.8, 0.65, 0.5), and only then tile detail (+3
SSE per step, never past the preset maximum). A step is undone only after 8 s of motion
above 50 fps, applied while the camera rests (a resolution switch re-allocates the
framebuffers, a visible hitch mid-gesture), and every recovery has to earn twice the smooth
motion of the last, so a borderline machine settles instead of oscillating. Cesium divides
screen-space error by the pixel ratio, so a resolution step never changes which tiles are
drawn by itself. A ladder step that raises the error floor coarsens every group at once, so
the expensive Google world feels it and not only a cheap survey mesh. Mesh tilesets (sites
and the world) use `skipLevelOfDetail`: the level a view needs loads directly instead of
every level on the way, which is what made the near, deep part of a pitched view wait
longest (measured 265k versus 145k triangles in the same time). Every consumer divides the
error by the pixel ratio the scene renders at
(`PerformanceManager.addScreenSpaceErrorSink` passes both): Cesium measures screen-space
error in CSS pixels, which on a HiDPI screen picks tiles twice as coarse as they look, while
a maps app chooses detail by the pixels you see. A resolution cut therefore also lightens
the tile load. Sites and the Google world are two tileset groups under the same rules and
bounds (parity: collected data is never allowed less detail than its surroundings; Google's
tiles are coarse at Cesium's default 16, hence a base of 8 on balanced), but each group
walks on its own with its own loading state and memory budget, so the world filling its
cache never holds a survey mesh at a coarse level. No tileset is asked for finer than 2
device pixels, except through an asset's `screenSpaceErrorScale`: a tiler assigns geometric
error from geometry alone, so a survey mesh with centimetre textures but conservative errors
looks soft next to the Google world at the same error. The Aerometrex San Francisco mesh is
seeded at 0.25 (measured 7 km up: 2 px draws 17k triangles and a grey blob, 0.5 px draws
119k and the real detail, still a quarter of what Google draws in the same view). Ladder evidence comes from every motion frame
(slow frames add up, smooth frames pay them back), so three short slow drags count as much
as one long one. The dev panel shows the profile (`full` or `reduced`) and the step taken. Gaussian
splats never refine below SSE 12 / 8 / 4 (performance / balanced / ultra): they are sorted on
the CPU every camera change, so their cost grows with splat count far faster than a mesh.
Tile cache budgets come from `navigator.deviceMemory` (256/384/512 MB + overflow). A per-asset
`maximumScreenSpaceError` acts as a quality floor. Manual SSE in Settings › Advanced disables
adaptation. Mesh coverage clips (the hole cut in the globe under a photogrammetry model) are
re-derived as tiles arrive but only swapped in at rest.

While the camera moves the root element carries `data-moving`; glass panels drop their
backdrop blur for the duration (a full-screen pass per panel otherwise) and use a flat tint.

Things that are deliberately _not_ done per frame: hover picking waits until the pointer has
rested 120 ms and never runs while the camera moves (each `scene.pick` is a render pass);
overlay anchors use `globe.getHeight` (a CPU lookup) while moving and call `sampleHeight`
only at rest, once per anchor every few seconds; the camera pose is throttled to 10 Hz.

## Camera

`CameraController` clamps `minimumZoomDistance` to 0.6 m (5 mm beside a hand-sized object)
so centimetre data can be inspected, keeps terrain collision on, and derives flight durations
from distance (1.2–5.5 s, quadratic in/out). Mouse mapping follows Google Maps: left-drag
pans (Cesium's rotate), wheel and pinch zoom towards the cursor, and Ctrl+drag, right-drag
and middle-drag orbit the point in the centre of the view. That orbit is implemented here
rather than with Cesium's tilt: the pivot is the depth-buffer hit at the view centre (a
model, a building, the ground), else the terrain, with placeholder tile heights rejected,
and the camera turns around it in its east-north-up frame with the tilt clamped between the
horizon and straight down. Gaussian splats write no depth, so under a splat the pivot is its
ground. Cesium keeps pinch tilt for touch. `KeyboardNavigator` adds arrows (pan, screen-space
speed), Shift+arrows (orbit the same pivot) and `+`/`-` (zoom towards it), ticked on
`scene.preUpdate` while held and active only when the page body or canvas has focus.

Never set `ScreenSpaceCameraController.minimumCollisionTerrainHeight` to 0: Cesium tests
terrain collision, and picks tilt pivots on the terrain instead of the ellipsoid, only while
the camera is _below_ that height (15 km by default). At 0 both switch off, and every tilt
pivots on sea level, which is 100 m under Redmond and felt like translation at close range. Flights point at the ground: the arrival pitch defaults to -45° (seeded
bookmarks use -40° to -45°), and any flight that climbs 150 m or 1.5× above its destination
passes `pitchAdjustHeight`, so the camera looks straight down at the top of the arc and eases
back to the arrival tilt on the way down instead of interpolating the pitch linearly and
spending the high part staring at the horizon. `flyToBoundingSphere` with that offset is the
standard arrival; hand-sized objects arrive at a few times their radius.
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

On by default (`VITE_ENABLE_PHOTOREALISTIC=false` switches it off), the Google Photorealistic
3D Tiles catalog layer (via `createGooglePhotorealistic3DTileset`) replaces the globe surface,
receives the same site clipping, and switches the ion geocoder to Google as Google's terms
require. It is labeled as visual context only. The globe is hidden in this world except
inside site footprints: `ClippingManager.setWorldMode` flips the globe's clipping collection
to `inverse` and fills it with every site's footprint, so terrain and imagery stay as an
opaque floor under a splat's sparse patches and in the gap between a mesh and its footprint
(a hole in the Google mesh otherwise shows sky), while the Google mesh is cut away there (its
buildings would otherwise poke through the site's). The globe is never hidden in that world,
only clipped away everywhere (a sentinel polygon at the pole keeps the inverse clip active),
so terrain and imagery keep loading under the Google mesh and are already sharp when a site
engages; a hidden globe would start from level 0 and show a dark band for seconds. The world clip under a splat uses the
authored footprint first: a splat's root box spans every outlier splat (the Redmond demo's is
1.7 × 2.8 km) and would blank the photorealistic world for blocks; the demo boundary is the
outline traced from a top-down render of the splat. Mesh sites with `clipFootprint:
"tileset"` use the tiles' coverage only from box or region volumes and only while it is no
larger than the authored footprint (`tighter`): coarse tiles' spheres reach far past the
data, and cut out of the world they showed as a ring of black circles around the San
Francisco mesh. Seeded boundaries of sphere-based meshes are measured from a render and
inset about 20 m inside the content edge, so the Google mesh overlaps the survey's ragged
edge rather than leaving a band of coarse terrain imagery between the two. Terrain, imagery
and ion-hosted meshes share one host, so its request concurrency is raised (36) to keep a
streaming mesh from crowding out the terrain under it.

A site's model only takes over from the world once the camera is close enough for its
detail to matter: below 2.5 footprint radii of altitude and within 3 radii horizontally
(`SiteManager.shouldEngage`, handing back at 3.5 and 4.5 so the threshold does not flicker; a
flight target is always engaged so the model is there on arrival). Further out the model
stays loaded but hidden and no clip is applied, so from 20 km up the world is seamless
instead of showing a 5 km patch of a differently lit capture with a hard edge. Clamped objects (the sample rock and plant) rest on the drawn surface
(`scene.sampleHeightMostDetailed`, excluding themselves) when it is within 60 m of the
terrain, so they sit on Google's ground rather than floating over or sinking into it.

## Patched engine

`patches/@cesium__engine@26.3.0.patch` (applied by pnpm on install) guards
`TerrainFillMesh.propagateEdge` against a neighbour without a mesh. With inverse clipping
polygons on the globe (the photorealistic world keeps terrain only inside site footprints),
Cesium skips a clipped-away tile before creating its fill mesh, and a neighbour's fill then
reads `undefined.westIndicesSouthToNorth` and stops rendering. The guard drops that edge;
the fill falls back to the tile's height range. Cesium's own error panel is off
(`showRenderLoopErrors: false`); render errors are logged, toasted and recovered from up to
five times.

## API changes noted while building

- `ClippingPolygon` positions are frozen (1.145): rebuild instead of mutating.
- `ClippingPolygonCollection.quality` / `destroy` are deprecated (1.145); not used.
- `Cesium3DTile.boundingVolume` is not in the public typings; `footprintFromTileset` reads
  it defensively and falls back to `boundingSphere`.
- CesiumJS 1.144 added composable camera `Controller`s; the explore mode here is a
  minimal purpose-built controller and can migrate to that framework later.
