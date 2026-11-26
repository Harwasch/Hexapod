# Scan preview verification

From the repository root, using the installed Swift environment without downloads:

```bash
. ~/hw-tools/swift/env.sh
mkdir -p .tmp-codex
export TMPDIR="$PWD/.tmp-codex"
HEAVY_CPUS=2 heavy swift test --package-path apps/ios/Modules/LiveSplat --jobs 2 --disable-sandbox
HEAVY_CPUS=2 heavy swift test --package-path apps/ios/App --jobs 2 --disable-sandbox
```

For focused iteration, add `--filter 'CoverageColourTests|ScanGuidanceTests'` to the
LiveSplat command. Keep only these packages' two `.build` directories. Remove
`.tmp-codex` after testing; do not commit it. On restricted runners, SwiftPM's
`--cache-path`, `--config-path`, `--security-path` and compiler module caches can
also point inside that temporary directory.

## Linux evidence

- `CoverageColourTests`: monotonic count/confidence/direction mapping, maturity
  thresholds, cool translucent start, exact true-colour endpoint, smooth eased
  transitions and retarget continuity, pulse expiry, fixed grid capacity/dedupe,
  negative quantization, and all new GPU strides/offsets. Source checks guard the
  Swift/MSL structs, uniforms and both appearance call sites, and prevent retired
  renderers/settings/resources from returning.
- `ScanGuidanceTests`: half orbit → Around, full level orbit → Higher, raised loop,
  poor per-surface spread → Around, distant camera → Closer, high blur → Slower,
  range/spread hysteresis, candidate/minimum dwell, ring sector completion,
  duplicate timestamp rejection, one event per sector, empty/blurred paths and reset.
- Preference and App wiring tests: independent Preview/colour choices, default on,
  persisted off, stale keys ignored, shared active renderer and scan continuity.
- Existing tests: voxel hashing/fusion/eviction, radii, coverage/replay, Gaussian
  normals/covariance/projection/sort, lifecycle, reversible quality and PLY validation.
  Five-minute synthetic references assert flat allocated arrays and an 8 ms mean
  CPU budget for 128 samples/keyframe. They do not measure full GPU work or RSS.
- App's existing capture admission test runs five minutes of synthetic 60 Hz input
  with exactly 4,500 admissions and a mean admission target below 0.5 ms. Existing
  bounded-writer, publication and capture bundle tests are unchanged.

## Apple compile and execution checks

Linux excludes Apple implementations. `ShaderLibraryTests` constructs every
remaining compute/render pipeline. `SurfelMetalTests` runs the actual coarse-grid
and surfel fusion kernels in both styles, checks exact keys/counts/animation targets,
normal attributes, the unchanged 64-byte surfel and 256-byte preview uniforms, and
surfel offsets. Async library/pipeline creation awaits the iOS 26 overloads.
`GaussianMetalTests` compares actual GPU sorting with the portable order.
These tests need the iOS SDK and Metal device and have not run on Linux.

Before phone acceptance, compile App and LiveSplat against iOS 26, including the
postprocess callback, new buffer bindings, SwiftUI pill/ring and soft haptic.
Use Metal validation. Check YCbCr range/matrix, display crop/orientation,
foreground rejection, tracking loss/relocalization, start/stop/new-scan clearing,
style switches and stopping while a command is in flight. Existing Apple-only
codec tests must still verify the complete colour/depth/confidence recording.

## Phone acceptance

1. On a clean install, Discs and Coverage colours are on. Scan a textured object:
   new areas should look cool/translucent and subtly alive; repeated views and a
   sweep around it should smoothly reveal true colour with one soft completion
   glow. Judge visibility, edge softness, clutter, occlusion and whether this feels
   like progress in both Discs and Splat, near/far and portrait/landscape.
2. Walk half a loop, finish at one height, then add elevated views. Judge the
   icon-first Around/Higher pill, calm dwell, ring readability and subtle sector
   haptics. Move too far or quickly and confirm Closer/Slower. A stationary phone
   turning in place must not complete an orbit. Start a new scan and check reset.
3. Toggle Coverage colours and Scan preview independently, switch styles mid-scan,
   relaunch and verify persistence. The camera remains usable with preview off;
   guidance continues. Recording and server output must be unaffected.
4. Profile five minutes in each style: displayed fps (target 60), preview GPU time
   (existing target 8 ms), delegate p50/p95/p99 (target ≤0.5 ms), memory plateau,
   thermals, battery/Low Power adaptation/recovery and encoder drops. Compare with
   colours off. Flat buffers alone do not establish these performance targets.

The first observed centroid anchors the viewpoint ring. Judge framing sensitivity
on a phone, particularly broad walkthroughs, floors and a subject initially near
the edge. No Linux test can establish the visual polish, haptic feel, actual GPU
budget or reconstruction benefit. The replay screen remains manually stepped and
labelled Simulated; it exercises fixture ingestion rather than the live GPU path.
