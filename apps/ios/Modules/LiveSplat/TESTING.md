# Test LiveSplat independently

From the repository root, with the existing Swift 6 toolchain and `heavy` installed.
No package dependencies, downloads, camera, server or GPU are needed for Linux
tests. The runner uses one compiler job, serial tests and `TMPDIR=./.tmp-test`
(resolved to an absolute path). It removes `.tmp-test` and `.swift-scratch` on exit,
including failures. These directories are reserved for this runner; do not share
them with another running test. On this workstation it also sources the installed
Swift environment if Swift is absent from `PATH`.

| Piece                        | One command                                               | What it proves                                                       |
| ---------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- |
| Whole module / required gate | `bash apps/ios/Modules/LiveSplat/test-piece.sh`           | All Linux tests, including existing numerical and lifecycle tests    |
| Bundle and fake frame source | `bash apps/ios/Modules/LiveSplat/test-piece.sh replay`    | Fixture replay, EOF/reset, tracking loss, malformed records/payloads |
| LiDAR seeding                | `bash apps/ios/Modules/LiveSplat/test-piece.sh seeding`   | Confidence/depth gates and ground-truth surface/probe agreement      |
| Coverage                     | `bash apps/ios/Modules/LiveSplat/test-piece.sh coverage`  | View occupancy, stationary saturation, fixture growth, tracking loss |
| Renderer fake and protocol   | `bash apps/ios/Modules/LiveSplat/test-piece.sh fake`      | Exact growth sequence, duplicate timestamps, restart and shared API  |
| Numerical reference          | `bash apps/ios/Modules/LiveSplat/test-piece.sh math`      | Projection, covariance, derivatives, compositing and tile binning    |
| Lifecycle and budget         | `bash apps/ios/Modules/LiveSplat/test-piece.sh lifecycle` | Caps, pruning, session IDs, fallback, stop/start and tracking loss   |
| PLY export                   | `bash apps/ios/Modules/LiveSplat/test-piece.sh export`    | Encoding, stop requirement and finalized-manifest protection         |
| Repository formatting        | `bash apps/ios/Modules/LiveSplat/test-piece.sh format`    | Repository Markdown/config formatting after runner cleanup           |
| SwiftUI debug screen (Mac)   | `open -a Xcode apps/ios/Modules/LiveSplat/Package.swift`  | Opens the package for the interactive checks below                   |

The whole-module command runs `swift test --package-path apps/ios/Modules/LiveSplat`
with scratch/cache locations, `--jobs 1 --no-parallel` and sandbox-compatible
SwiftPM options. For another Mac, use its existing Xcode Swift toolchain and local
`heavy` wrapper; the runner does not install anything.

## Seams and fixture replay

`LiveSplatRenderer` is now the main-actor protocol. Construct
`DefaultLiveSplatRenderer()` for the real seeding/coverage pipeline or
`FakeLiveSplat()` for deterministic UI tests. Existing construction sites must
rename `LiveSplatRenderer()` to `DefaultLiveSplatRenderer()`; the Metal and ARFrame
adapters still take the concrete implementation. App-facing logic can accept
`any LiveSplatRenderer`.

`SplatFrameSource` supplies an asynchronous, manually pulled event stream:
`.frame`, `.trackingUnavailable`, then `nil` at EOF. Serialize `next()` and
`reset()` calls. The renderer's event overload sends loss events through the same
coverage/keyframe invalidation used by the live ARFrame adapter.

```swift
// On the main actor; fixtureURL points to yard-small/bundle, not capture.hexcap.
let source: any SplatFrameSource = try BundleReplayFrameSource(
  bundleDirectory: fixtureURL, decodeRGB: false)
let renderer: any LiveSplatRenderer = DefaultLiveSplatRenderer()
renderer.start()
while let event = try await source.next() {
  renderer.ingest(event)
}
renderer.stop()
let coverage = renderer.coverageFraction
```

Tests resolve the committed fixtures from `#filePath`; no copies or regeneration
are required. `yard-small` contributes 22 tracked frames and two skipped frames.
`yard-limited` contributes three tracked frames and 21 skipped frames. Skips do
not seed, age or prune existing points, and clear current-view coverage/keyframes.
Depth is decoded explicitly from little-endian float16; confidence bytes are
validated. The source rejects malformed matrices/buffers and paths outside the
flat bundle directory. Missing depth also emits a loss event.

Linux uses flat grey, depth and poses. With AVFoundation available, the default
`decodeRGB: true` decodes the corresponding video frame at its relative timestamp,
sampling unrotated RGB at depth pixel centres. `decodeRGB: false` gives identical
flat colours on both platforms. Apple decode errors are surfaced to the caller.
`replayAppleRGBMatchesSyntheticColorClasses` is an Apple-only test for RGB alignment
against the fixture's labelled probes; it has **not been run on Linux**.
`replayAppleRGBPreservesRowsAndIntegerPixelCentres` separately checks exact colour
channels, row ordering and sampling with a tiny generated bitmap, also Apple-only.

The geometry test checks every eligible depth seed against the bounded ground,
box shed, capped cylinder trunk and spherical crown in `ground_truth.json`. It
also checks exact per-frame surface probes and all points retained by the
renderer. The tolerance is derived, not fitted: binary16 has ten fraction bits,
so optical-depth rounding is at most `2^(floor(log2(depth)) - 10) / 2` metres.
Multiply by the length of the corresponding unit-depth camera ray, then add
`1e-10` metres for Double arithmetic. At the 8 m seeding cap, the conservative
corner-ray bound is about **4.85 mm** for these fixtures. Integer pixel centres
introduce no extra half-pixel allowance. Coverage is checked against a fresh grid
projecting all retained points; it remains current-view occupancy, not scene
completeness.

`FakeLiveSplat` adds 64 of 768 cells per accepted timestamp by default, saturates
at 100%, ignores duplicates/stopped input, and resets on start or tracking loss.
It produces no geometry; its export is an empty valid PLY with the same lifecycle
and manifest guards. Use the real renderer for geometry/export-content tests.
`FakeFrameSource(events:)` replays an explicit event list;
`try FakeFrameSource.plane()` supplies a small stationary plane without files.

## Debug screen

In Xcode, open `LiveSplatDebugView.swift` and resume its Canvas preview. Press
**Next frame**: fake coverage rises by 8.33 percentage points until 100%. Press
**Restart**: it returns to zero. The screen labels the replay **Simulated** and
shows coverage, seeded point count, frame count, skip status, EOF and errors.
It is compiled only with `DEBUG` and SwiftUI available.

To exercise the actual seeding pipeline in a debug host or local preview, provide
a fixture URL accessible to that process (copy the inner bundle folder into a
device debug host's resources if needed):

```swift
let source = try BundleReplayFrameSource(bundleDirectory: fixtureURL)
LiveSplatDebugView(source: source, renderer: DefaultLiveSplatRenderer())
```

Step all 24 `yard-small` frames: the first two show a tracking skip, subsequent
frames add points/coverage, and one more step reports completion. Restart must
reproduce the same results. With `yard-limited`, coverage clears after the three
tracked frames while seeded points remain. A bad fixture path throws at source
creation; a bad depth/video payload is displayed when stepping that frame.

The standalone screen uses replay. Live integration keeps the existing App-owned
ARSession and frame-forwarding API described in [README.md](README.md). Linux
does not compile or validate SwiftUI, AVFoundation, ARKit or Metal branches.
Apple compilation, preview interaction and RGB probe tests remain device/Mac
acceptance work; no GPU timing or visual-quality claim follows from these tests.
