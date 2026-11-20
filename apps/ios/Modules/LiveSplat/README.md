# LiveSplat

Live LiDAR seeding, a point preview, a Metal Gaussian preview
and bounded photometric updates. The App feeds this package from its shared AR session; the renderer does not own it. Linux validates the pure Swift numerical reference and
state management. **Apple-framework code and Metal have not been compiled or run.**

For isolated tests, fakes, bundle replay and the standalone SwiftUI debug screen,
see [TESTING.md](TESTING.md). `LiveSplatRenderer` is the injectable protocol;
`DefaultLiveSplatRenderer` is the concrete implementation and `FakeLiveSplat`
provides deterministic simulated coverage.

## App integration

All renderer methods run on the main actor. Use the App's existing ARSession
frame feed; this module does not install a delegate, configure, run or reset the
session. Enable `.sceneDepth` in the App's `ARWorldTrackingConfiguration`.

```swift
import LiveSplat

let renderer = DefaultLiveSplatRenderer() // training on by default
renderer.start(session: session)

// Forward each frame on the main actor. Pass the actual overlay viewport and
// current interface orientation, including after rotation or resizing.
try renderer.ingest(frame, orientation: orientation, viewportSize: viewportSize)

// Above the existing RealityKit ARView in a ZStack:
LiveSplatView(renderer: renderer).allowsHitTesting(false)

let fraction = renderer.coverageFraction
renderer.stop()
let ply = try renderer.exportPLY(to: captureBundleFolder)
```

`start()` and `ingest(SplatFrame)` are also available without Apple frameworks.
`start()` clears the previous scan; `stop()` preserves the snapshot for export
and invalidates outstanding GPU work. Stop/start after an AR world reset or
relocalization into a different world basis. Tracking loss or missing depth hides
the overlay and clears recent keyframes. The App must stop on session interruption
and handle frame conversion errors. The scan toggle uses an icon and accessibility label.

`LiveSplatView` is a SwiftUI wrapper around the public `LiveSplatOverlay` UIView.
The transparent Metal view composites premultiplied output above RealityKit's
camera, without taking over its rendering callbacks. The current ARFrame's display
transform maps sensor pixels to the viewport; the shader uses its inverse.
The host must remain mounted while a scan is active. The renderer is not an
ObservableObject: poll `coverageFraction` and `status` from the App's frame handler
if separate SwiftUI state needs updates.

Training is **on by default** (Harrison's decision, 2026-11-16). Public modes are
`LiveSplatMode.training` and `.pointPreview`; statuses are `.training`,
`.pointPreview`, `.fallback(String)` and `.stopped`. `configuration.mode` defaults
to `.training`; setting it to `.pointPreview` forces a preview-only renderer.

`LiveSplatPreference.shared` persists `trainingEnabled` in UserDefaults under
`hex.liveSplat.training`, defaulting to true when absent. The renderer reads it at
start and reacts immediately to writes through the same preference instance,
without resetting the scan or the App's ARSession. For isolated state, inject
`LiveSplatPreference(store:)` using `LiveSplatPreferenceStore` or
`UserDefaultsLiveSplatStore(defaults:)`. Direct writes to UserDefaults or separate
preference instances are not live notifications; controls and renderers must share
an instance and write through `trainingEnabled`.

```swift
let preference = LiveSplatPreference.shared
let renderer = DefaultLiveSplatRenderer(preference: preference)
// Scan screen: icon only, accessibility label/value.
LiveSplatToggle(preference: preference)
// Settings: a single short Toggle label, no caption.
LiveSplatSettingsRow(preference: preference)
// Also available without SwiftUI:
preference.trainingEnabled = false // point preview; true requests training again
```

The host requires Metal family Apple7 or newer; unsupported devices, pipeline
allocation/compilation errors, GPU errors, low power mode, serious/critical thermal
state or a measured budget overrun use the point preview and report
`.fallback(reason)`. Fallback never changes the stored preference. The first
fallback reason latches until renderer restart; switching off shows `.pointPreview`
and switching back on restores that reason. `FakeLiveSplat(preference:)` follows
the same user preference, with simulated training status and no GPU work.
No capability claim is inferred from Linux tests.

## Numerical and resource boundaries

- Camera poses are ARKit camera-to-world, +Y up, optical direction −Z. Matrix
  storage is row-major; the ARFrame adapter explicitly reads SIMD columns.
  Projection applies `diag(1,-1,-1) R_cwᵀ` after subtracting camera position.
- Raw scene depth is optical-axis metres. Only high confidence, finite depths in
  `[0.1, 8]` metres seed splats. Intrinsics scale from unrotated capturedImage
  dimensions to depth resolution. Row strides and the two YCbCr planes are read
  separately. RGB retains camera gamma encoding; this is not radiometric training.
- Covariance is `R(q) diag(exp(2 logScale)) R(q)ᵀ`, with normalized Hamilton wxyz.
  Projected covariance is `J W Σ Wᵀ Jᵀ + 0.3 I` in the raster's pixel units.
  The blur term is a chosen antialiasing variance, not a fitted accuracy tolerance.
- Metal projects, bins into 16×16 sensor-space tiles, sorts deterministically by
  depth and composites front to back. Each tile keeps the nearest 64 splats;
  overflow adds the cyan coverage hint instead of writing beyond its allocation.
  Rendering truncates at three standard deviations and caps alpha at 0.99.
- Optimization is a deliberately small surrogate: two steps on up to 32 rotating
  splats, using the two newest keyframes out of a three-frame ring (at most 5 Hz
  insertion). Each step minimizes mean half-squared RGB error in a fixed 3×3 patch
  for an isolated splat over black, with continuous Gaussian support. It does
  **not** differentiate the occlusion-aware composite, masks or visibility, and can
  misfit occlusions. Position, log scales, normalized quaternion, opacity logit and
  RGB all receive forward-mode derivatives. Step size is 0.001 with component
  gradients clipped to ±1. No convergence or training-quality claim is made.
- `Math.swift` mirrors the Metal scalar formulas. Tests check hand values and
  independent central differences for all 14 parameters, including rotated camera
  covariance. CPU tile/composite references are available for future GPU parity
  tests. Linux does not execute the shader or establish float32 parity.
- Default caps: 8,192 splats, 128 seeds/frame, three keyframes, depth at most
  512×512, 320-pixel longest preview edge, one GPU command buffer in flight.
  Configuration is clamped to 32,768 splats and 256 new seeds/frame. Low-opacity,
  invalid and unobserved splats older than 600 processed frames are pruned.
  New seeds stop when full until pruning frees space. Stable IDs prevent delayed
  GPU results from updating newly inserted splats; session IDs reject old scans.
- The ≤8 ms GPU budget is a **target**, not a proven deadline: Metal command buffers
  cannot be preempted here. Timing includes training and preview kernels. One
  overrun or unavailable GPU timing latches point-preview fallback until restart.
  CPU frame conversion, allocation and draw cost are outside this GPU target.
- Coverage is current-view 32×24 grid occupancy by depth-consistent splat centres,
  not pixel opacity, calibrated confidence or full-scene completeness. Depth must
  agree within 10 cm or 5%; these are explicit heuristic gates. Nearest depth-pixel
  centres determine cells, avoiding repeated seeding at floating-point boundaries.
  Cyan cells are under-covered; unknown or low-confidence depth stays uncovered.

## PLY handoff boundary

`exportPLY(to:)` atomically writes `livesplat.ply` after `stop()`, before the
capture manifest is finalized. It refuses a folder containing `manifest.json` so
it cannot silently invalidate an existing inventory. The binary little-endian PLY
contains xyz in ARKit world metres, zero normals, SH degree-zero `f_dc_0..2`, opacity
logits, log standard deviations and Hamilton `rot_0..3`. RGB converts via
`(rgb - 0.5) / (1 / sqrt(4π))`. Point-preview mode exports its LiDAR-seeded Gaussians.

Capture inventories this optional export through `manifest.liveSplat`; the Python
bundle validator accepts the bounded PLY and the pipeline ignores it for now.
Training initialization from the preview remains unimplemented. Exporting after
finalization is still forbidden because it would invalidate the inventory.

## Verification and acceptance

Run offline, from the repository root:

```bash
bash apps/ios/Modules/LiveSplat/test-piece.sh
pnpm format:check
rm -rf ./.swift-scratch ./.tmp-test
```

Apple/device acceptance still required (training now defaults on):

1. Xcode 26 / Swift 6 strict-concurrency compile of the ARKit, UIKit, SwiftUI and
   MetalKit branches; verify the package's compiled Metal library and pipeline creation.
2. GPU/CPU projection, covariance, all 14 derivatives, tile overflow/depth ties and
   alpha-compositing readback comparisons on the same fixtures. Record float32
   errors before selecting device tolerances; none have been inferred here.
3. iPhone 15 Pro or newer capture: color range/matrix, RGB/depth alignment, camera
   pose conventions, portrait/landscape/crop changes, foreground/occlusion quality,
   coverage hints, tracking loss, world reset, stop/restart and export timing.
4. Measured GPU times at growing/max splat counts and tile overload, allocation
   peaks, CPU capture throughput and camera frame drops. Demonstrate fallback
   after an overrun or unavailable timing; ≤8 ms has not been measured.
5. Sustained device thermals/battery, low-power transitions, background/foreground,
   memory pressure and fallback rendering. Confirm the weak display-link lifecycle
   and that the App's RealityKit camera remains aligned and interactive.
6. A separate training acceptance test before initializing cloud reconstruction
   from the preview. Bundles can carry it today; training deliberately ignores it.

Open choices: whether this isolated-patch optimizer gives sufficient quality or needs a full composite
loss; and training initialization from the optional bundle PLY.
