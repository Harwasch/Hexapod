# Phone training

This package trains saved captures on an iPhone and retains the hidden measurement
workflow. Build B adds its product screen in `App/TrainingScreen.swift`. `PhoneTrainPrep` uses Foundation only; `PhoneTrain`
contains the Apple adapter and actor. Linux never resolves a remote dependency.
Apple hosts resolve Apache-2.0 msplat-ios **exactly 3.0.0**, product `Msplat`, with
its binary XCFramework and bundled metallibs. No dependency source is vendored.

## Preparation policy

`PrepConfig` is the single dataset policy. Translation/rotation default to 4 cm /
4 degrees. Normally tracked frames pass a scan-relative Laplacian-variance blur
gate (25% of the median at a common 160-pixel maximum side). Selection relaxes
to 3 cm / 3 degrees below 120 views and thins uniformly above 200. Small scans
retain fewer views; fewer than two usable views is an error. A uniformly blurry
scan cannot be diagnosed by a relative score alone.

Pose validation uses the server's positive-determinant, 1e-3 orthonormality and
1e-6 homogeneous-row bounds. The Foundation-only 3×3 Jacobi SVD polar projection
computes `U Vᵀ`; accepted positive determinant makes the server's reflection
correction +1. Translation is unchanged. The parity test records numbers actually
computed with `capture_bundle.world_to_camera` and NumPy SVD. There is no axis flip:
ARKit is already row-major OpenGL camera-to-world for `DatasetFrame`.

Intrinsics scale independently with raster width/height. Two sequential
`AVAssetReader` passes first score small RGB-derived luma rasters, then write only
selected frames as JPEG quality 0.9 in a unique `Caches/PhoneTrain` directory.
Neither pass applies movie display transforms or EXIF orientation. Every decoded
sample must match its logged index, nanosecond timestamp and dimensions, including
excluded frames. The saved manifest and frame counts must agree.

Only training views seed geometry. Use smoothed depth when present, raw depth
otherwise; missing depth is skipped and corrupt referenced depth fails preparation.
Back-project high-confidence (2) samples, reject four-neighbour depth jumps greater
than `max(5 cm, 5% of depth)`, and retain depths 0.1–20 m. Integer sample centres
follow the capture contract. Colour uses normalized coordinates in the training RGB
raster. The deterministic streaming voxel grid starts at 1.5 cm and coarsens to
stay within 300k points. The 100k minimum is advisory: sparse captures never invent
points. Benchmark initialization caps at 100k for identical seeds below every cap;
other callers are limited to half their Gaussian cap for topology reserve.

Holdout indices are **0, 8, 16, …**, matching native
`InputData::splitTrainTestIndices` (`core/src/input_data.cpp:1475`). Do not remove
these images from the descriptor: `DatasetOptions(evalMode: true, testEvery: 8)`
performs the split. Holdout images never contribute seed geometry or colour.

## Trainer integration seam

Build the dataset off MainActor with `CaptureDatasetBuilder.prepare(bundle:config:)`.
Use `PhoneTrainer.recommendedPreset(dataset:)` to choose Best (800k / 5,000 steps /
SH2) or Quick (300k / 1,500 steps / SH1) against `os_proc_available_memory()` and
`TrainingPlan.estimatedPeakMemory`, retaining 30% available-memory reserve. These
are initial budgets intended for roughly 6–8 / 2 minutes, not measured promises.
The resolution stages, iteration counts, SH degree and thermal delay live in
`PhoneTrainingConfig`. A rejected memory estimate is an explicit failed benchmark
row, never a silently reduced cap.

Create one `PhoneTrainer`, subscribe to `progress()`, then await `run()` from a
Task. Forward scene inactivity immediately with `setActive(false)`; foreground
return uses `setActive(true)`. Manual `pause()` / `resume()` use the same step
boundary. Pauses checkpoint once and wait without submitting GPU work. Serious
thermal state inserts a 100 ms delay; critical state checkpoints and returns
`.critical` with a clear progress message. Explicit `resume()` after cooling may
start another `run()` on the same actor. Benchmark critical detection is latched
for the entire sweep and never resumes automatically. In-flight native calls
finish before a checkpoint or stop; an app suspension can delay completion.

The private `MsplatSession` always runs on its declared `MsplatRuntimeActor`.
`PhoneTrainer` serializes its lifecycle, step submissions, evaluation and export;
concurrent runs and auxiliary operations are rejected. The loop polls query-only
telemetry until each submitted GPU step completes; a 60-second completion timeout
reports failure. Progress carries step, completed loss, splats, ETA and the latest
held-out PSNR. PSNR stays nil until `evaluate()` provides a real measurement; it
is never inferred from training loss. The stream keeps the newest two events;
`recordedSamples()` retains all measurement steps independently.

`cancel()` stops at a completed step. Always await the run and `close()` before
cleaning a caller-owned dataset with `cleanup()`, or use `closeAndCleanup()`.
`PhoneTrainer.restoring(from:)` reads `checkpoint.json`, rebuilds disposable
JPEGs, and returns an actor for `run(loadCheckpoint: true)`. Metadata stores the
capture path, full prep/training policy and step. Native checkpoint writes use
a temporary file and atomic rename; metadata writes are atomic. This assumes the
saved bundle remains immutable and at the recorded path. Checkpoints are a local
measurement artifact, not a portable interchange format.

Outputs live in `Application Support/PhoneTrain/<capture-id>/<run>/`, beside the
`Captures` collection. They never add undeclared files to immutable upload bundles.
`exportPLY()` preserves ARKit world metres with `keepCrs = true`. Session v3.0.0
has no public SPZ export. The deprecated `GaussianTrainer.exportSpz` belongs to a
different ownership API and is deliberately not mixed into a session.
For the product, set an orbit with `setPreviewPose(_:)` and consume `previews()`.
The loop submits a native Metal preview every 40 completed steps and evaluates
held-out PSNR every 250 steps when a pose has been requested. Preview completion
is awaited before another step; no auxiliary GPU operation races optimization.
The hidden benchmark requests no pose and gets no extra preview/evaluation work.
The independent `submitPreview(pose:referenceCamera:)` seam remains idle-only.
`TrainingSnapshotPolicy` and active-step `TrainingETA` are portable and tested.
Final evaluation/export obey the same pause, cancellation and critical-temperature
policy; inactivity also checkpoints an idle completed session.
The saved result seam is the exported PLY URL. Native sessions are process-global: no other msplat owner
may train concurrently.

## Hidden benchmark and reports

Enable Settings → Testing (`hex.testflight.debugEnabled`), then **Benchmark train**.
The screen chooses the finalized local scan with the most recent manifest start
date. It runs 200k / 400k / 800k at maximum sides 640 / 960, each for 600 steps,
SH1, half resolution for the first 300 steps and full resolution for the rest.
The sweep order is deterministic; there is no artificial cooling interval.

Reports are `Documents/phone-train-<UUID>.json`, visible through Files and the share
sheet. An initial report and every finished/failed run are written atomically.
The report contains per-step wall/GPU/completion timing, loss, splat count,
physical footprint, available process memory, thermal state, ETA, preparation /
training / evaluation / total time, plan memory estimate, actual seed/keyframe
counts, output path and held-out **PSNR and SSIM** from `evaluate()`.
A separate 250 ms timeline and thermal notifications cover prep/evaluation too.
Physical memory uses `task_vm_info.phys_footprint`; peak means the maximum sampled
footprint, not an allocation high-water guarantee. Critical temperature stops the
sweep immediately after the in-flight operation, records why, and leaves earlier
rows available. Missing or nonfinite quality metrics stay nil. Failed runs retain
their errors and available step samples. The dark result card shows time, memory,
PSNR and thermal/status icons; SSIM and the complete trace are in JSON.

## Verified msplat v3.0.0 signatures

Source paths below are relative to the read-only `/home/harrison/.cache/msplat-ios/`
checkout, under `swift/Sources/Msplat/`. They are source verification, not an Apple
SDK compilation claim.

- `DatasetCalibration.init(width:height:fx:fy:cx:cy:...)`: `DatasetDescriptor.swift:68`.
- `DatasetFrame.init(id:calibrationID:imageURL:rasterOrientation:calibration:cameraToWorld:...)`:
  `DatasetDescriptor.swift:125`; `.encodedPixels`: line 16.
- `DatasetSparsePointSet.init(xyz:rgb:...)`: `DatasetDescriptor.swift:172`;
  `xyz`, `rgb`, `count`: lines 162, 164, 170.
- `DatasetDescriptor.init(frames:points:...)`: `DatasetDescriptor.swift:316`;
  `frames`, `points`: lines 310, 311.
- `CameraPose.init(elements:)`: `MsplatSession.swift:43`.
- `DatasetOptions.init(downscaleFactor:evalMode:testEvery:...)`: `MsplatSession.swift:23`.
- `TrainingConfig.init()`: `TrainingConfig.swift:77`; used fields `iterations`,
  `shDegree`, `shDegreeInterval`, `ssimWeight`, `numDownscales`, `resolutionSchedule`,
  `warmupLength`, `keepCrs`: lines 33–38, 40, 48.
- `TrainingImageDimensions.init(width:height:)`: `TrainingPlan.swift:16`.
- `TrainingResolutionStage.init(iterations:downscaleFactor:)`: `TrainingPlan.swift:43`.
- `TrainingPlan.init(inputDimensions:inputDecodeScale:iterationBudget:stages:targetSHDegree:maximumGaussianCount:...)`:
  `TrainingPlan.swift:204`; `estimatedPeakMemory`: line 200.
- `MsplatSession.init(dataset:securityScopedResourceURLs:options:config:maximumGaussianCount:)`:
  `MsplatSession.swift:149` (global actor: line 76).
- `step() throws -> TrainingStats`: `MsplatSession.swift:192`;
  `TrainingStats.iteration`: `TrainingStats.swift:5`.
- `trainingMetrics() throws -> TrainingTelemetry`: `MsplatSession.swift:203`;
  `completed`, `lastFailedIteration`: `TrainingStats.swift:155`, `160`;
  completed `iteration`, `splatCount`, `gpuExecutionMs`, `endToEndMs`, `loss`:
  `TrainingStats.swift:64`, `65`, `72`, `75`, `77`.
- `memoryMetrics() throws -> TrainingMemorySnapshot`: `MsplatSession.swift:219`;
  `processPhysicalFootprintBytes`: `TrainingStats.swift:187`.
- `evaluate() throws -> EvalMetrics`: `MsplatSession.swift:269`;
  `psnr`, `ssim`, `numTest`: `TrainingStats.swift:250`, `251`, `253`.
- `submitPreview(pose:referenceCamera:) throws -> MetalPreviewSubmission`:
  `MsplatSession.swift:393`; future consumer `waitUntilReady()`:
  `MetalPreviewSurface.swift:22` (documented seam, not called here).
- `exportPLY(to: URL) throws`: `MsplatSession.swift:435`.
- `saveCheckpoint(to: URL) throws`: `MsplatSession.swift:444`.
- `loadCheckpoint(from: URL) throws -> Int`: `MsplatSession.swift:454`.
- `iteration: Int`: `MsplatSession.swift:491`; `close() throws`: line 504.

## Validation and phone acceptance

Linux tests cover log validation, keyframe gates/relaxation/thinning, relative
blur scoring, SVD parity/rejections, intrinsics scaling, float16 depths, plane
back-projection, confidence/edge rejection, bounded deterministic voxels, native
holdout numbering, boundary control, configuration validation, sweep composition
and partial-report/checkpoint metadata persistence.

```bash
. ~/hw-tools/swift/env.sh
mkdir -p .build/tmp
TMPDIR="$PWD/.build/tmp" swift test --package-path apps/ios/Modules/PhoneTrain \
  --scratch-path "$PWD/.build/swift/PhoneTrain" --jobs 2 --disable-sandbox
```

Apple-only files are `CaptureDataset.swift`, `PhoneTrainer.swift`,
`ProcessMemory.swift`, `BenchmarkRunner.swift` in `Sources/PhoneTrain`, and
`../../App/TrainBenchmarkScreen.swift`. Existing `App/DebugMenu.swift` adds only
its gated navigation entry. Package.swift adds the Apple-host-only dependency;
XcodeGen declares the same exact remote version and app product.

Before merging, build with Xcode 26 / Swift 6 strict concurrency, including binary
linking, metallib resources and iOS simulator slices. On a LiDAR iPhone, verify
RGB orientation/colour registration, timestamps, ARKit PLY alignment/scale,
quality on the most recent scan, actual Gaussian growth, each memory-budget gate,
repeated sweep memory cleanup, thermal throttle/critical stop, inactive/foreground
pause/resume, native checkpoint reload after terminating/relaunching, cancellation,
cache eviction/rebuild, Files visibility and sharing. Compare reports from a cool
phone and a warmed phone. Linux excludes Apple code and establishes none of the
GPU timing, duration, quality or thermal targets.

## Build B product

The post-scan card and capture detail open the dark full-screen training flow.
Sharpness, UI identifiers, durable result manifests and cloud intent live in the
portable App support target. The saved PLY viewer and in-place rendering live in
LiveSplat. See [Build B implementation and acceptance](../../BUILD_B_REPORT.md)
for the full SDK inventory, assumptions and compile self-audit.
