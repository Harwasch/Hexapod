# Scan preview verification

From the repository root, with the installed Swift environment and no downloads:

```bash
. ~/hw-tools/swift/env.sh
mkdir -p .scratch/tmp
TMPDIR="$PWD/.scratch/tmp" heavy swift test --package-path apps/ios/Modules/LiveSplat \
  --scratch-path "$PWD/.scratch/build/LiveSplat" --cache-path "$PWD/.scratch/cache" \
  --config-path "$PWD/.scratch/config" --security-path "$PWD/.scratch/security" \
  --disable-sandbox --jobs 4
```

The Linux suite covers negative-coordinate floor quantization, near/far hash keys,
collisions, running colour averages, bounded eviction, sample-spacing radii,
reversible quality adaptation, dense seeding beyond the old 768-cell ceiling,
fixture geometry/coverage, preference persistence, replay and legacy PLY validation.
A 3,000-keyframe synthetic reference run fills a deliberately small hash and asserts
flat allocated capacity. Its measured CPU budget is for a 128-sample fixture subset;
it is **not** a GPU timing measurement or a full 12,288-sample CPU performance claim.

The same five-minute capacity/budget regression now exercises Gaussian voxel
fusion, a 128-entry bitonic sort and orbit controls (`PreviewPerformanceTests`).
It asserts fixed cell/attribute array counts and the same 8 ms mean CPU subset
budget; it does not measure process resident memory. `GaussianTests` checks
normal-oriented covariance eigenvalues, near/far footprint scales, confidence
opacity, invalid inputs and depth-edge fallback, analytic 2D projection, dedupe,
eviction, allocation caps and sorting up to the full 32,768-entry draw limit.
`ScanOrbitTests` checks viewport bounds, rotation limits and orbit periodicity.
App support tests persist the independent defaults/choices and ingest both preview
styles through the real facade. The existing capture admission performance test
continues to run unchanged.

`apps/ios/App/Tests/ScanPerformanceTests.swift` uses XCTest `measure` and a monotonic
clock over five minutes of 60 Hz synthetic input, with exactly 4,500 15 Hz admissions
and a mean admission target below 0.5 ms. The same package is already selected by
`ci_scripts/packages.py test` in iOS simulator CI; no new CI/network service is needed.
App tests also block a writer to prove the four-packet reservation limit, no packet
creation after overflow, drain ordering and fail-closed publication.

The Apple-only `ScanCodecTests` feeds synthetic colour/depth/confidence buffers
through the actual recording worker and decodes the resulting HEVC. It compares
video frame count/timestamps, JSONL indices, manifest counts, and float16 depth
bytes. Backpressure may reduce the number of complete packets, never just one stream.
`ShaderLibraryTests` constructs all compute pipelines and locates the render
functions from the compiled package library. `SurfelMetalTests` checks the 64-byte
cell/256-byte uniform ABI and executes fusion twice on an exactly representable
fixture, comparing key, position, radius and observation count without a fitted
tolerance. `HighlightLifecycleTests` verifies callback installation/removal.
`GaussianMetalTests` executes the actual GPU bitonic network and compares exact
depths/indices against the portable reference. The library smoke test constructs
all Gaussian, reduction, composite and frustum pipelines. Fusion's exactly
representable fixture now runs in both styles and checks normal/opacity attributes
and the new 48-byte orbit uniform ABI. Async Metal library/pipeline creation uses
`await` for the iOS 26 overloads. Linux does not execute these Apple-only tests.

## Still requires Apple hardware or simulator

- Xcode 26 strict-concurrency build of the App and packages, compiled Metal library,
  simulator XCTest performance run and real HEVC encode/decode test. Linux excludes
  the Apple implementations and cannot establish their compilation or execution.
- Metal validation and GPU/reference comparisons: ABI sizes, quantization/hash,
  bucket contention, eviction, colour ranges/matrices, projection and crop/orientation.
  No float32 parity tolerance has been fitted to one machine.
- iPhone 15 Pro or newer, five-minute scan: Instruments signposts/profiling for
  delegate p50/p95/p99 and maximum time (target ≤0.5 ms), displayed fps under full
  capture/fusion load (target 60), preview GPU time, encoder drops, memory plateau,
  and record/decoded count equality on the resulting real bundle.
- Phone visual acceptance: dense opaque-ish discs, edge softness, depth occlusion,
  small structures, motion, near/far transition, coverage tint, camera alignment in
  portrait/landscape, tracking loss, relocalization and new-scan clearing.
- Sustained thermals, battery/Low Power transitions, recovery of quality, background/
  foreground, memory pressure and saving while GPU or disk work is in flight.
- Settings comparison: confirm Discs and Inset off on a clean install; switch to
  Splat and enable Inset, scan near/far and slanted surfaces, inspect low-coverage
  tint and phone frustum, tap to enlarge and drag in both orientations. Compare
  both styles with inset off/on for a five-minute scan each, including quality
  degradation/recovery, relaunch persistence and memory plateau. Style changes
  reset preview geometry; neither change recording or server output. Check correct
  sort order during orbit and phone motion, EWA ellipses, alpha overlap and depth
  occlusion. All of this remains unverified here.
- Confirm preview off shows camera plus coverage only (Inset remains independent); no inactive answer
  postprocess; highlighting resumes outside scans with a cached mask texture.
- Upload a real capture, await server Ready, and verify the detail renders the
  cloud-trained scan through the globe. Local surfels must never substitute for it.

The debug replay screen remains manually stepped and labelled Simulated. It tests
fixture ingestion, not the live GPU path. There is no UIKit fallback or local trainer.

Before `pnpm format:check`, remove generated `.scratch/build`, `.scratch/cache`,
`.scratch/config`, `.scratch/security` and `.scratch/tmp`: Prettier otherwise scans
SwiftPM's generated JSON/YAML. Keep test logs and the task's commit manifest.

## Coverage hairs

Iterate with `bash apps/ios/Modules/LiveSplat/test-piece.sh coverage` or the
`CoverageHairTests` filter. The Linux suite asserts the exact 32-byte ABI/offsets,
cone boundaries, sharpness, saturation, resolution weighting, octahedral quantization
(in both hemispheres, using an analytic quantization bound), parameter resource
loading/defaults/rejection, hash dedupe/collisions/replacement, normal EMA and
per-keyframe admission. A synthetic hemisphere camera orbit fills every plane hair;
an opposing orbit leaves the excluded hair empty. The parallax test distinguishes
repeated same-view credit, insufficient separation and a valid second contributor.
The capacity run feeds four times the full grid capacity and asserts fixed storage
counts, not process RSS. Preference/App wiring tests cover default off, persistence,
independence and keeping the active renderer and scan generation.

The Apple-only library smoke test additionally constructs all coverage compute and
line render pipelines. `CoverageMetalTests` exercises the actual credit kernel with
an orbit and excluded views, comparing saturated/empty bytes exactly. All async
Metal library/pipeline creation uses the awaited overload. Linux excludes these
checks; they have not been run here.

On the phone, enable Coverage next to Preview/Inset. Scan a wall from several
angles and distances: confirm long amber hairs shorten toward green and disappear,
back/edge/poor-resolution views do not falsely fill directions, and new scans clear
all hairs. Compare Discs/Splat, preview off, inset on/off, toggle off/on, style
switches, portrait/landscape and relaunch persistence. If fitted JSON enables gates,
confirm distant views earn less credit and a repeated single viewpoint cannot remove
a gated stub. Judge normal/frame stability, clutter, line visibility, occlusion,
and the 10 cm grid. Measure FPS, GPU budget, thermals and resident-memory plateau
over a five-minute scan, including quality decimation/recovery. The UI prototype's
appearance and usefulness for training are phone acceptance work, not Linux claims.
