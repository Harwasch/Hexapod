# Build B implementation and acceptance

The capture detail and post-scan card now open phone training with Quick/Best,
native Metal snapshots, orbit/zoom, a progress ring, ETA, held-out Sharpness,
pause/resume, cancel, thermal messaging and a settling animation with haptic.
The result has focus, PLY sharing, in-place viewing and confirmed cloud refinement.
Captures have durable phone/cloud badges and share actions.

## Implementation locations

| Behaviour                                          | Source, relative to `apps/ios`                                                         |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Product flow and icon controls                     | `App/TrainingScreen.swift:10`, presets at `:140`, results at `:224`                    |
| Training lifetime and result loading               | `App/TrainingModel.swift:11`, finalization at `:115`, camera updates at `:212`         |
| Metal snapshot display                             | `App/TrainingMetalView.swift:7`                                                        |
| Serialized native previews and held-out evaluation | `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift:34`, boundary work at `:296` |
| Presets, snapshot cadence and active-step ETA      | `Modules/PhoneTrain/Sources/PhoneTrain/TrainingPolicy.swift:17`, `:96`, `:109`         |
| UI state, Sharpness and durable result records     | `App/Support/TrainingExperience.swift:4`, `:50`, `:69`, `:108`                         |
| Confirmed cloud request survives upload/relaunch   | `App/Support/CaptureUploadQueue.swift:99`                                              |
| Cloud tileset readiness and bridge                 | `App/TrainingCloudPreview.swift:12`, `:75`, `:104`                                     |
| PLY decoding, sorting and orbit math               | `Modules/LiveSplat/Sources/LiveSplat/SavedSplat.swift:12`, `:94`, `:112`               |
| World-pose compositing                             | `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift:12`, `:75`, `:104`       |

Phone results are committed atomically under `Application Support/PhoneResults/<id>`.
The immutable capture bundle is untouched. LocalCaptures exposes the result;
deleting the local scan also removes its result directory. A cloud-ready record
preserves badges when offline. Confirmed refinement intent lives beside upload
state and is removed only after the existing refinement endpoint acknowledges it.
The endpoint retains `X-Hex-Refine: 1` through the existing Upload implementation.

Sharpness has one mapping, in `TrainingPresentation.sharpness`: 15 dB → 0,
35 dB → 100, linear and clamped between. Unknown/nonfinite held-out PSNR remains
unknown. This display heuristic is not a percentage of physical accuracy.
ETA averages completed active step time, including thermal delay, excluding
pauses. Preview/evaluation overhead can make it optimistic. No measured duration
is promised. The one `TODO-tune-from-benchmark` block contains Quick/Best defaults.

## See it in place

The implementation reuses LiveSplat's RealityKit postprocess ownership pattern.
It copies the camera and renders saved Gaussian covariance/opacity into the same
ARView command buffer. This reuses more of the existing camera/compositing path
than rendering through msplat, whose pose preview uses a reference camera's saved
intrinsics. The AR overlay needs current portrait view/projection matrices instead.
`keepCrs = true` preserves the scan's ARKit world metres, so no guessed placement
transform is applied. Only value matrices cross the frame-ingress boundary.

The training screen is an overlay that keeps the original ARView mounted. A saved
capture is eligible only while that original session identity and world are retained.
An ordinary pause preserves the identity. Interruption, dismantling and explicit
tracking reset invalidate access conservatively; the
option disappears. There is no relocalization claim. Fade-in and camera/splat
switching change the renderer's opacity. The phone splat is used for AR, since the
cloud output has a different georeferenced coordinate frame.

Cloud viewing reuses the existing `globe-embed.html` renderer and public Globe
bridge in a screen-owned WKWebView. It selects the exact native capture route.
The catalog's Ready state starts loading; the embed's `data-scan-state=ready`
signal starts the crossfade. The native phone view remains mounted for fallback.
This is a crossfade between independent cameras, not a registered geometric morph.
Subsequent catalog polls do not override an explicit Phone selection. A failed
cloud view returns to Phone and offers retry. Memory warnings reach the existing
cloud renderer's memory-tier policy.

## Export and assumptions

- Phone export is PLY only, via `MsplatSession.exportPLY`. The session has no public
  SPZ export or publicly reachable GaussianTrainer owner. The separate deprecated
  `GaussianTrainer.exportSpz` (`GaussianTrainer.swift:278`) is not mixed into it.
- The original exported PLY, including higher SH coefficients, is retained for
  sharing. Saved native display uses covariance, opacity and SH DC colour; it does
  not reproduce view-dependent higher SH colour. Training snapshots use msplat's
  native renderer. No native SPZ decoder was added.
- Cloud sharing uses the existing public `view.html#<site-id>` link. That public
  page shows the site's latest published date. Exact capture selection is retained
  in the native cloud viewer; the public page currently has no dated-scan route.
- Quick is the initial product choice; Best is explicit and retains the existing
  memory-budget rejection. Presets remain provisional until phone measurements.
- Background resume means the retained training screen/actor resumes on foreground
  return. Native checkpoints are saved at safe boundaries, including finalization.
  Process-termination restoration remains the existing engine/debug seam; this
  change does not add a relaunch recovery chooser.
- The app's existing portrait-only configuration and HTTPS API/web configuration
  are retained. Cloud requires existing credentials; Settings is available from
  the refinement action when credentials are missing.
- This implementation does not run training on Linux or synthesize quality values.
  Imported synthetic captures retain their Simulated label in the training viewer.
- The requested cache directory could not be created: the sandbox returned
  `Read-only file system` for `~/.cache/tmp-train-ui`. Tests use
  `apps/ios/.build/tmp` instead of the small `/tmp` RAM disk. Build output and the
  temporary root `node_modules` symlink are removed after verification.

## Accessibility identifiers

All new product identifiers are constants in `App/Support/TrainingExperience.swift:4`.
Controls have spoken labels; orbit supports accessible focus/zoom actions, and
source buttons expose selection. IDs may repeat inside distinct capture rows.
Existing capture/navigation/settings identifiers remain outside this new contract.

```text
train.start
train.quick
train.best
train.begin
train.pause
train.resume
train.cancel
train.retry
train.progress
train.sharpness
train.thermal
result.close
result.orbit
result.focus
result.share
result.shareCloud
result.cloudRetry
result.phone
result.cloud
inplace.open
inplace.camera
inplace.splat
inplace.close
refine.start
refine.confirm
refine.cancel
refine.retry
refine.progress
captures.badge.phone
captures.badge.cloud
```

## Compile self-audit

Every new/changed Apple-only line was re-read for the brief's compile-safety rules.
The list below is the complete Apple-only file inventory, including the shader.
`Support/CaptureUploadQueue.swift` is additionally listed because its iOS
Observation expansion is excluded on Linux.

| File                                                           | Audit                                                                                                                                                                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `App/ARCameraView.swift`                                       | Locked frame-consumer handoff; only immutable matrices retained; origin invalidated on tracking reset.                                                                                                                   |
| `App/CameraActions.swift`                                      | MainActor ARView access, current camera projection, Sendable renderer callback and explicit session/world identity lifetime.                                                                                             |
| `App/CameraScreen.swift`                                       | Overlay keeps ARView mounted; explicit registry injection; post-scan hero identifier.                                                                                                                                    |
| `App/DebugMenu.swift`                                          | Existing CameraScreen initializer receives an internal registry; no cross-module private symbols.                                                                                                                        |
| `App/HomeScreen.swift`                                         | Small badge view, explicit hero label, URL ShareLinks, result cleanup and cost-confirm IDs. Existing raw backend error text removed.                                                                                     |
| `App/TrainingModel.swift`                                      | UI MainActor; prep/training actors awaited; detached decoding/sorting pass Sendable values; cancellation awaits training before cleanup; thermal finalization returns to cooling.                                        |
| `App/TrainingMetalView.swift`                                  | Synchronous Metal setup in a synchronous helper; MainActor delegate; one display command in flight; completed surfaces retained through GPU completion.                                                                  |
| `App/TrainingScreen.swift`                                     | Small typed helper views; no arithmetic map/stride builders, optional-map interpolation, implicit-colour ternaries or duplicate declarations; lifecycle forwards to trainer.                                             |
| `App/TrainingCloudPreview.swift`                               | MainActor WebKit delegates; completion-form JavaScript calls; JSON arguments; weak message proxy; same-origin checks; no Any values cross actors.                                                                        |
| `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift`     | Every session call awaits its runtime actor, including throwing iteration getters; preview completion precedes the next step; auxiliary ownership serializes checkpoints/evaluation/export.                              |
| `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift` | Immutable GPU geometry/pipelines, locked uniforms/order/live camera matrices; synchronous Metal factories; AR command buffer owns compositing. Narrow unchecked Sendable boundary matches the existing renderer pattern. |
| `Modules/LiveSplat/Sources/LiveSplat/Shaders/SavedSplat.metal` | Five float4 Gaussian fields (80 bytes), two matrices plus float4 uniforms (144 bytes), matching Swift layout; covariance projection and premultiplied Gaussian blending.                                                 |
| `App/Support/CaptureUploadQueue.swift`                         | MainActor pending requests/tasks, persisted intent before submission, actor uploader calls awaited; no credentials persisted in intent.                                                                                  |

All changed Apple Swift files receive an iOS-target frontend syntax parse. It does
not resolve Apple SDK APIs, perform SwiftUI type checking or compile Metal. The
portable PLY reader, orbit, sorting, Sharpness, state transitions, source swaps,
ETA, snapshot cadence, badges, simulation labels, session identity and durable manifests/intents
are covered by Linux tests. Existing numeric tolerances were not widened. New
float32 round-trip tests use a fixed 1e-6 bound, independent of host measurements.

## msplat source inventory

Verified against the read-only v3.0.0 checkout. Every location below is under
`/home/harrison/.cache/msplat-ios/swift/Sources/Msplat/`. This includes retained
calls in the changed trainer, not just new preview calls.

| API or accessed members                                                                                                      | Source file:line                                                           |
| ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `MsplatRuntimeActor`, session global isolation                                                                               | `MsplatSession.swift:7`, `:76`                                             |
| `CameraPose.init(elements:)`, `elements`                                                                                     | `MsplatSession.swift:43`, `:41`                                            |
| `DatasetOptions.init(evalMode:testEvery:)`                                                                                   | `MsplatSession.swift:23`                                                   |
| `DatasetDescriptor.frames`, `.points`, initializer                                                                           | `DatasetDescriptor.swift:310`, `:311`, `:316`                              |
| `DatasetFrame.cameraToWorld`                                                                                                 | `DatasetDescriptor.swift:121`                                              |
| `DatasetSparsePointSet.xyz`, `.rgb`, `.count`, initializer                                                                   | `DatasetDescriptor.swift:162`, `:164`, `:170`, `:172`                      |
| `TrainingConfig.init()`                                                                                                      | `TrainingConfig.swift:77`                                                  |
| `iterations`, `shDegree`, `shDegreeInterval`, `ssimWeight`, `numDownscales`, `resolutionSchedule`, `warmupLength`, `keepCrs` | `TrainingConfig.swift:33`, `:34`, `:35`, `:36`, `:37`, `:38`, `:40`, `:48` |
| `TrainingResolutionStage.init(iterations:downscaleFactor:)`                                                                  | `TrainingPlan.swift:43`                                                    |
| `TrainingImageDimensions.init(width:height:)`                                                                                | `TrainingPlan.swift:16`                                                    |
| `TrainingPlan.init(...)`, `.estimatedPeakMemory`                                                                             | `TrainingPlan.swift:204`, `:200`                                           |
| `MsplatSession.init(dataset:options:config:maximumGaussianCount:)`                                                           | `MsplatSession.swift:149`                                                  |
| `step()`, returned `TrainingStats.iteration`                                                                                 | `MsplatSession.swift:192`, `TrainingStats.swift:5`                         |
| `trainingMetrics()`, `.completed`, `.lastFailedIteration`                                                                    | `MsplatSession.swift:203`, `TrainingStats.swift:155`, `:160`               |
| Completed `.iteration`, `.splatCount`, `.gpuExecutionMs`, `.endToEndMs`, `.loss`                                             | `TrainingStats.swift:64`, `:65`, `:72`, `:75`, `:77`                       |
| `memoryMetrics()`, `.processPhysicalFootprintBytes`                                                                          | `MsplatSession.swift:219`, `TrainingStats.swift:187`                       |
| `evaluate()`, `.psnr`, `.ssim`                                                                                               | `MsplatSession.swift:269`, `TrainingStats.swift:250`, `:251`               |
| `submitPreview(pose:referenceCamera:)`                                                                                       | `MsplatSession.swift:393`                                                  |
| `MetalPreviewSubmission.waitUntilReady()`                                                                                    | `MetalPreviewSurface.swift:22`                                             |
| `MetalPreviewSurface.texture`, `.width`, `.height`                                                                           | `MetalPreviewSurface.swift:39`, `:40`, `:41`                               |
| `exportPLY(to:)`                                                                                                             | `MsplatSession.swift:435`                                                  |
| `saveCheckpoint(to:)`, `loadCheckpoint(from:)`                                                                               | `MsplatSession.swift:444`, `:454`                                          |
| Throwing `iteration`, `close()`                                                                                              | `MsplatSession.swift:491`, `:504`                                          |

No msplat `render`/RGBA readback or separate GaussianTrainer is used by this flow.

## Validation and real-phone acceptance

The final command tails are reported with the delivery message. Run from the
worktree root, sequentially, with the installed Swift environment and at most four
jobs. Scoped tests were used during implementation; the complete gate follows the
last edit:

```bash
. ~/hw-tools/swift/env.sh
export TMPDIR="$PWD/apps/ios/.build/tmp"
heavy swift test --package-path apps/ios/Modules/PhoneTrain --jobs 4
heavy swift test --package-path apps/ios/Modules/LiveSplat --jobs 4
heavy swift test --package-path apps/ios/Modules/Upload --jobs 4
heavy swift test --package-path apps/ios/App --jobs 4
python3 -m unittest discover -s apps/ios/ci_scripts
# Remove all Swift .build directories before Prettier scans generated manifests.
node_modules/.bin/prettier --check .
```

Still requires Xcode and a real LiDAR phone:

- iOS 26/Swift 6 SDK type checking, binary/metallib linkage and Metal compilation.
- Actual native snapshot materialization, reference-camera orientation, colour,
  retained-surface lifetime after session close and repeated-run memory release.
- Saved PLY covariance appearance, portrait AR pose/scale/alignment, colour through
  the HDR postprocess target, camera/splat fades and origin invalidation.
- Quick/Best quality, thermal slowdown/critical resume, peak memory, CPU sorting
  cost, frame rate, ETA overhead and long background/checkpoint boundaries.
- Cancellation during prep, stepping, final evaluation/export and result loading;
  foreground resume, durable cloud request after relaunch and offline retry.
- Cloud upload → processing → training → ready, tileset-ready crossfade, public
  site-date sharing limitation, Phone fallback and memory-pressure recovery.
- Haptics, gestures, double-tap focus, VoiceOver, Dynamic Type and UI automation.

No Mac CI dispatch, push, merge, deploy or GPU training was performed here.

## Commit plan

Changes are left uncommitted as requested. Suggested decision-sized commits:

1. **Keep product previews serialized with phone optimization**
   - `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift`
   - `Modules/PhoneTrain/Sources/PhoneTrain/TrainingPolicy.swift`
   - `Modules/PhoneTrain/Tests/PhoneTrainPrepTests/ProductTrainingTests.swift`
   - `Modules/PhoneTrain/README.md`
2. **Render saved phone splats in the scan's existing world**
   - `Modules/LiveSplat/Sources/LiveSplat/SavedSplat.swift`
   - `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift`
   - `Modules/LiveSplat/Sources/LiveSplat/Shaders/SavedSplat.metal`
   - `Modules/LiveSplat/Tests/LiveSplatTests/SavedSplatTests.swift`
   - `Modules/LiveSplat/README.md`
3. **Connect local training to durable cloud refinement**
   - `App/TrainingModel.swift`, `App/TrainingScreen.swift`
   - `App/TrainingMetalView.swift`, `App/TrainingCloudPreview.swift`
   - `App/Support/TrainingExperience.swift`, `App/Support/LocalCaptures.swift`
   - `App/Support/CaptureUploadQueue.swift`, `App/Tests/TrainingExperienceTests.swift`
   - `App/ARCameraView.swift`, `App/CameraActions.swift`, `App/CameraScreen.swift`
   - `App/HomeScreen.swift`, `App/DebugMenu.swift`
4. **Record the device acceptance boundary for Build B**
   - `BUILD_B_REPORT.md`, `App/APPLE_API_AUDIT.md`, `SDK_PREFLIGHT.md`
