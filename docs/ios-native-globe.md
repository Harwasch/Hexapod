# Native globe: scope only

The staged default is the bundled CesiumJS globe in WKWebView. A native replacement would
use **cesium-native for tile selection, streaming, caching and geospatial transforms**, with
our own Metal renderer. cesium-native is not a ready-made iOS globe renderer. No native
rendering code, dependency, build, or training job is introduced here.

The replacement must implement the same `GlobeRenderer` snapshot and selection semantics:
property boundaries (including holes), dated scan footprints, answer pins, native entity
cards, exact capture selection and camera flights. Keep EntitySchema and property-memory
transport unchanged. Preserve ellipsoid heights and the pipeline's ENU transforms; never
place an unlocated observation at a guessed origin.

## Milestones

1. **Streaming spike on a device.** Pin cesium-native and its transitive dependencies,
   produce a reproducible iOS arm64 build, and expose a narrow C/Swift adapter for view
   state, tile requests, cancellation and selected tiles. Use the committed four-splat
   fixture and a small terrain fixture. Measure allocations and cancellation while
   repeatedly switching properties. No production rollout at this milestone.
2. **Terrain and imagery.** Render an ellipsoid, terrain meshes, imagery tiles and
   attribution in Metal. Define double-precision ECEF on CPU and camera-relative float
   coordinates on GPU, origin rebasing, depth convention, horizon culling, globe picking
   and camera constraints. Validate poles, antimeridian and centimetre local alignment.
3. **3D Tiles.** Bridge selected glTF content into Metal resources, material decoding,
   texture uploads, refinement and eviction. Test transforms, meshes, point clouds,
   holes, stale asynchronous work, and bounded queues. Make supported glTF extensions
   explicit; do not imply parity with every CesiumJS material or tileset feature.
4. **Splats through LiveSplat.** Reuse its Metal drawing/sorting implementation through
   an adapter for immutable decoded gaussian buffers (position, scale/covariance,
   rotation, opacity, SH), tile ENU-to-ECEF transform, camera matrices, viewport and
   shared depth. The adapter needs asynchronous upload completion, tile retirement and
   memory accounting; the globe owns streaming/LOD while LiveSplat owns GPU resources,
   sorting and drawing. Validate REPLACE refinement, tile-local SPZ origins, SH frames,
   terrain occlusion and seam-free transitions against CesiumJS reference captures.
5. **Parity and release gate.** Native pins/cards and scan flights must match the bridge
   fixtures. Measure cold/warm startup, resident memory, GPU time, battery/thermal load
   and recovery from backgrounding/memory pressure on supported phones. Only then make
   a separate decision about replacing the WKWebView default.

## Known interface versus proposed adapter

This checkout contains only `LiveSplatPlaceholder`, with an empty public initializer;
there is no Metal renderer interface to reuse here. The renderer described in PR #131
is not available in local files, and the task prohibits network access. The buffer,
view/depth and lifetime interface above is therefore a **proposed integration contract**,
not a claim about PR #131's symbols. Reconcile it with that implementation before coding.

## Risks and decisions

- Native dependency size, licensing/attribution, compiler compatibility and CI build time.
- Terrain/imagery formats and authentication: determine supported providers and offline
  storage rights before implementation. Keep secrets outside public resource bundles.
- GPU/CPU memory: a tile cache does not bound decoded copies, sort buffers or textures.
  Agree a device-measured total budget and eviction/back-pressure policy.
- Gaussian extension and SPZ version churn; preserve the fixture and explicit decoder
  compatibility tests. Confirm whether LiveSplat supports SH and tiled REPLACE refinement.
- Precision, transparency ordering and depth composition are separate correctness gates.
  Do not widen tolerances based on one device or claim centimetre accuracy from phone GPS.
- Feature parity is expensive. Leave CesiumJS available until native behavior and device
  measurements justify the maintenance cost. Apple Maps remains the simple debug option.
