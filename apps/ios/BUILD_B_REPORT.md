# Build B implementation and acceptance

The capture detail and post-scan card now open phone training with Quick/Best,
native Metal snapshots, orbit/zoom, a progress ring, ETA, held-out Sharpness,
pause/resume, cancel, thermal messaging and a settling animation with haptic.
The result has focus, PLY sharing, in-place viewing and confirmed cloud refinement.
Captures have durable phone/cloud badges and share actions.

## Implementation locations

| Behaviour                                          | Source, relative to `apps/ios`                                                         |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Product flow and icon controls                     | `App/TrainingScreen.swift:10` (presets and result controls)                            |
| Training lifetime and result loading               | `App/TrainingModel.swift:11` (lifetime, finalization and camera updates)               |
| Metal snapshot display                             | `App/TrainingMetalView.swift:7`                                                        |
| Serialized native previews and held-out evaluation | `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift:34`, boundary work at `:296` |
| Presets, snapshot cadence and active-step ETA      | `Modules/PhoneTrain/Sources/PhoneTrain/TrainingPolicy.swift:17`, `:96`, `:109`         |
| UI state, Sharpness and durable result records     | `App/Support/TrainingExperience.swift:4`, `:50`, `:69`, `:115`                         |
| Confirmed cloud request survives upload/relaunch   | `App/Support/CaptureUploadQueue.swift:109`                                             |
| Cloud tileset readiness and bridge                 | `App/TrainingCloudPreview.swift:12`, `:75`, `:104`                                     |
| PLY decoding, sorting and orbit math               | `Modules/LiveSplat/Sources/LiveSplat/SavedSplat.swift:12`, `SplatSortPolicy.swift`     |
| World-pose compositing                             | `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift:12`                      |

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
Catalog polling stops at Ready and does not override an explicit Phone selection. A failed
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

| File                                                               | Audit                                                                                                                                                                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `App/ARCameraView.swift`                                           | Locked frame-consumer handoff; only immutable matrices retained; origin invalidated on tracking reset.                                                                                                                   |
| `App/CameraActions.swift`                                          | MainActor ARView access, current camera projection, Sendable renderer callback and explicit session/world identity lifetime.                                                                                             |
| `App/CameraScreen.swift`                                           | Overlay keeps ARView mounted; explicit registry injection; post-scan hero identifier.                                                                                                                                    |
| `App/DebugMenu.swift`                                              | Existing CameraScreen initializer receives an internal registry; no cross-module private symbols.                                                                                                                        |
| `App/HomeScreen.swift`                                             | Small badge view, explicit hero label, URL ShareLinks, result cleanup and cost-confirm IDs. Capture-detail backend error text and video sharing retain their original contracts.                                         |
| `App/TrainingModel.swift`                                          | UI MainActor; prep/training actors awaited; detached decoding/sorting pass Sendable values; cancellation awaits training before cleanup; thermal finalization returns to cooling.                                        |
| `App/TrainingMetalView.swift`                                      | Synchronous Metal setup in a synchronous helper; MainActor delegate; one display command in flight; completed surfaces retained through GPU completion.                                                                  |
| `App/TrainingScreen.swift`                                         | Small typed helper views; no arithmetic map/stride builders, optional-map interpolation, implicit-colour ternaries or duplicate declarations; lifecycle forwards to the shared session protocol.                         |
| `App/TrainingCloudPreview.swift`                                   | MainActor WebKit delegates; completion-form JavaScript calls; JSON arguments; weak message proxy; same-origin checks; no Any values cross actors.                                                                        |
| `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift`         | Every session call awaits its runtime actor, including throwing iteration getters; preview completion precedes the next step; auxiliary ownership serializes checkpoints/evaluation/export.                              |
| `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift`     | Immutable GPU geometry/pipelines, locked uniforms/order/live camera matrices; synchronous Metal factories; AR command buffer owns compositing. Narrow unchecked Sendable boundary matches the existing renderer pattern. |
| `Modules/LiveSplat/Sources/LiveSplat/Shaders/SavedSplat.metal`     | Five float4 Gaussian fields (80 bytes), two matrices plus float4 uniforms (144 bytes), matching Swift layout; covariance projection and premultiplied Gaussian blending.                                                 |
| `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainingSession.swift` | Optional public native capability; unchanged base protocol; adapter awaits every trainer operation and returns only Sendable values.                                                                                     |
| `UITests/HexUITests.swift`                                         | MainActor XCTest helpers and synchronous predicate expectations; training/refine/relaunch flow awaits identifiers, with in-place absent without an AR session.                                                           |
| `App/Support/CaptureUploadQueue.swift`                             | MainActor pending requests/tasks, persisted intent before submission, actor uploader calls awaited; no credentials persisted in intent.                                                                                  |

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

## Original commit plan

These four steps were committed before the follow-up and rebased onto #138:

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

## Follow-up: integrate the #138 training seam

This change depends on draft PR **#138**, branch `codex/ios-uitest-mode`, merging
first. Its `PhoneTrainingSession`, launch factory, recorded PNG preview, isolated
capture storage and in-process capture API remain the integration contract. No
members of that base protocol or `PhoneTrainingResult` changed. The existing
fixture implementation is unchanged.

`TrainingModel` stores only `any PhoneTrainingSession` plus the optional
`PhoneTrainingNativePreview` capability. Every session comes from
`PhoneTrainingFactory.make(outputDirectory:real:)`; dataset preparation and native
owner construction execute only inside `real`. The Device adapter alone adopts
the optional capability for pose requests, Metal surfaces, held-out PSNR and
thermal recovery. Its `run()` retains the original export contract; the optional
quality measurement follows the completed run, before cleanup. There are no new
msplat calls: `CameraPose.init(elements:)` remains at `MsplatSession.swift:43`,
`MetalPreviewSurface` fields at `MetalPreviewSurface.swift:39–41`, and evaluation
returns the existing `TrainingStats.swift:250` PSNR through the trainer. The full
source inventory above remains applicable.

Other sessions use `previewImage()` PNG bytes and their provided progress. The
fixture's held-out Sharpness stays unknown. A portable progress function uses
elapsed/remaining time when no native step budget is available. The PNG and
simulation marker are saved atomically with the result manifest, so reopening a
fixture result never parses its ASCII PLY with the native binary reader. Older
result manifests still decode. Production binary PLY rendering is unchanged.

Catalog reads now share `CaptureStatus.fetch`, which routes through
`LaunchMode.current.captureAPI` before considering a service URL. The queue uses
that same fixture API for storage and confirmed refinement, without credentials,
HTTP or upload polling. Production credential checks and durable intent monitoring
are retained. Fixture Ready updates badges and the source toggle using the saved
PNG, without creating a Metal view or web view. UI-test updates are driven by
session/state events; there is no training display timer, orbit sorting, settling
sleep or success haptic. All 30 existing identifiers above remain unchanged.

The two hand-resolved conflicts were checked: `CameraActions.suspend()` retains
both `previewTask` cancellation and `frameFeed.setTrainedPreview(nil)`;
`CameraScreen` retains `CameraTimeline { date in ... }` and applies opacity and
hit-testing to the complete timeline while training covers it. `HexApp`,
`DebugMenu`, `TrainBenchmarkScreen` and `LaunchMode` retain #138's quiet launch
and benchmark guards. They need no additional changes. In-place eligibility now
also explicitly rejects UI-test mode, including the production-provider probe.
There are no duplicate declarations.

### Follow-up compile self-audit

Every Apple-only line changed in this follow-up was re-read. The complete changed
Apple-only list is below; the cumulative Build B list above includes the new
adapter capability and UI test as well.

- `App/TrainingModel.swift`: MainActor UIKit image decoding; only PNG Data crosses
  detached disk operations; protocol lifecycle awaits; optional native calls await;
  preparation occurs only in the factory closure; no direct trainer field.
- `App/TrainingScreen.swift`: small bitmap/native helper view; existing identifier
  constants preserved; Equatable upload-state observation replaces fixture polling;
  fixture cloud view is a still PNG; real web readiness and native rendering remain.
- `App/CameraActions.swift`: one MainActor eligibility guard; the merged suspension
  cleanup remains intact and the world-identity policy is unchanged.
- `App/HomeScreen.swift`: shared catalog route, persisted fixture cloud badge and
  explicit accessibility containment for badge descendants; existing UI-test return
  guards retained.
- `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainingSession.swift`: public
  capability types verified against pinned msplat source; only Device adopts it;
  Sendable streams/values; every underlying actor call awaits.
- `UITests/HexUITests.swift`: MainActor XCUIApplication actions, existing attachment
  helpers, predicate expectation and explicit waits; no fixed sleeps.
- `App/Support/CaptureUploadQueue.swift`: additionally audited because the iOS
  Observation expansion is excluded on Linux. Fixture injection is Sendable;
  production secrets are still checked; fixture confirmation completes synchronously
  through actor awaits and preserves the durable intent boundary.

`App/Support/CaptureStatus.swift` and `App/Support/TrainingExperience.swift` are
portable and compile under Linux, including new progress math and manifest fields.
The changed Apple files receive an iOS-target syntax parse; this remains distinct
from Apple SDK type checking, simulator execution and Metal compilation.

Portable coverage now joins factory → fixture session → PNG/PLY persistence →
queue storage → confirmed refinement → Ready → both badges, without configured
services. It also checks fallback progress, old manifest decoding and PNG cleanup.
The existing fixture pause/resume/cancel tests still run unchanged. The identifier
audit resolves `TrainingID` constants and includes the new UI test prefixes.
`testPhoneTrainingRefinesThroughTheFixtureAPIAndPersists` exercises the actual
screens, declined/confirmed refinement, Phone fallback, badges and result reopen
in XCUITest; it asserts that in-place is absent. This test has not run on Linux
and still needs the simulator gate. Native snapshot quality, AR placement and
thermal behavior still require the real-phone checks above.

### Follow-up commit

**Connect training screens to the shared session and fixture APIs**

File list, relative to `apps/ios`:

- `App/TrainingModel.swift`, `App/TrainingScreen.swift`, `App/CameraActions.swift`
- `App/HomeScreen.swift`, `App/Support/CaptureStatus.swift`
- `App/Support/CaptureUploadQueue.swift`, `App/Support/TrainingExperience.swift`
- `App/Tests/FixtureCaptureAPITests.swift`, `App/Tests/TrainingExperienceTests.swift`
- `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainingSession.swift`
- `UITests/HexUITests.swift`, `ci_scripts/test_ui_tests.py`, `BUILD_B_REPORT.md`

The follow-up remains uncommitted. No Mac CI dispatch, push, merge or deployment
was performed. Final full-gate tails are in the delivery message.

## Follow-up 2: PR #159 review fixes

All nine numbered findings and all three renderer suspicions are addressed. This
section supersedes earlier implementation details where they overlap. Changes
remain uncommitted; no Apple CI, simulator or real-device execution was requested.
The dependency on #138 remains unchanged. All 30 Build B identifier constants are
unchanged. Capture detail again uses its original `capture.refine` identifier.

1. In-place projection and renderer viewport use `arView.bounds.size`, including
   safe areas, both on attachment and on camera updates. `offer()` reads that same
   viewport. There is no standalone portable projection function: ARKit constructs
   the projection directly. Phone verification must compare alignment at screen edges.
2. The portable `TrainingOutputPolicy` permits deletion only after the session is
   closed and the result is saved or training is terminal. `TrainingModel` then
   deletes only `PhoneResults/<id>/Training`, including PLY/checkpoints. Cancel and
   failure use the same release path; cooling/paused owners retain their scratch
   files for resume. The immutable capture bundle is never a deletion target.
3. `TrainingCameraPolicy` drives perception enablement independently of scan phase.
   Busy training disables the driver, display work, tap admission and frame-feed
   consumers, so the 10 Hz tick cannot re-enable Vision. Already-running inference
   drains through the driver's existing disabled-generation handling. AR tracking
   stays alive only when the original capture world is still eligible for in-place
   viewing; otherwise the coordinator pauses it. Completion/dismissal restores the
   ordinary camera policy. Origin invalidation also removes the keep-alive reason.
4. Saved-result loading checks `.choosing` before and after every suspension that
   precedes a UI mutation. A slow PLY read cannot replace a newer training state;
   its errors also cannot change that newer state to failure.
5. Portable cloud decisions skip busy training and stop at Ready or catalog failure.
   Stage changes restart the task without fetching terminal state again. A Ready
   route can still mount when the result appears. Explicit refinement clears old
   terminal state. `saveIfChanged` compares both IDs before writing `cloud.json`,
   including list/detail catalog reads.
6. Capture detail once again displays the original full server/upload error and
   shares the capture video. Result-viewer and badge PLY actions remain. The detail
   Refine button retains `capture.refine`, matching `TESTING.md`.
7. Both capture-row queries search all descendants using the unchanged prefix,
   preserving discoverable contained badge elements.
8. The tap helper waits for existence, hittability and enabled state with an
   `XCTNSPredicateExpectation`; timeout fails without tapping. No sleeps were added.
9. `test_ui_tests.py` has a direct-execution `unittest.main()` entry point.

Renderer suspicions:

- A portable three-degree direction threshold gates CPU sorts against the last
  successfully uploaded order. Translation alone does not affect depth ordering.
  Uniforms still update every tick. Three index buffers are allocated once; locked
  reader counts and command-completion handlers prevent overwriting in-flight GPU
  data. An unavailable slot defers publication and retries instead of allocating.
- After GPU upload the model retains 16-byte positions for sorting, center and
  radius, rather than the full 80-byte Gaussian attributes. The temporary decoded
  scene is released; GPU attributes remain in one buffer. CPU positions are still
  required for CPU sorting.
- Release cancels the preview observer and clears the last Metal surface before
  closing the session, including error/cancel paths. The observer checks task
  cancellation before publishing. An already submitted display blit deliberately
  retains its surface only until its command finishes.

### Review-fix compile self-audit and Apple-only list

Every changed Apple-only line was re-read against COMPILE SAFETY. The complete
Apple-only changed file list for this follow-up is:

- `App/ARCameraView.swift`: MainActor coordinator activity; lock-protected frame
  admission flag; no frame crosses an actor boundary.
- `App/CameraActions.swift`: MainActor ARView bounds and named value tuple; policy
  helpers are internal portable symbols; every driver enable path respects busy
  training; no new ARSession reset or configuration overload.
- `App/TrainingModel.swift`: stage guards after detached reads; native lifecycle
  awaits; scratch deletion occurs after close; detached closures capture Sendable
  URLs/compact geometry; saved renderer identity checked after sorting; no retained
  native surface after release. Typed local viewport and direction values.
- `App/TrainingScreen.swift`: small policy-driven tasks and helper methods; no heavy
  ViewBuilder arithmetic; cancellation/stage checks before catalog publication;
  explicit MainActor camera restore after stop; unchanged product identifiers.
- `App/HomeScreen.swift`: restored URL ShareLink and optional error binding; existing
  `.textSelection(.enabled)`; catalog writes use the portable idempotent helper.
- `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift`: local slot array
  initialized before assigning stored properties; synchronous Metal allocations;
  immutable buffers and lock-protected reader counts/uniforms; command completion
  owns each read until GPU completion; no asynchronous Metal overload introduced.
- `UITests/HexUITests.swift`: MainActor XCTest queries; predicate waiter replaces
  immediate tap; timeout exits safely; descendant query retains the same prefix.

An iOS-target frontend syntax parse covers these seven files. It does not perform
Apple SDK type checking or Metal execution. New policy/source files and tests are
portable. No new msplat APIs are used, and the existing source inventory remains
applicable (`CameraPose` and MetalPreviewSurface declarations rechecked read-only).
No numeric tolerances were changed. Three degrees is a new documented display
cadence policy, tested away from its boundary rather than calibrated to this host.

Portable tests were written before implementation. The App and LiveSplat test
builds first failed on the missing policies/types; an additional catalog `error`
case then failed behaviorally before the terminal-state policy was corrected.
Coverage includes scratch eligibility, camera re-enablement, terminal/ busy cloud
polling, idempotent manifest writes, translation-independent sorting and compact
geometry parity. Final full gate tails are provided in the delivery message.

### Per-fix commit plan

Paths are relative to `apps/ios`. Shared integration files require staging the
corresponding hunks: viewport, cleanup, load-race and sorting changes share
`TrainingModel`; camera and polling changes share `TrainingScreen`; fixes 7 and 8
share the UI-test file. The report can be staged by section alongside each fix.

| Fix                 | Commit subject                                                  | Exact file list (excluding this shared report)                                                                                                                                                                                                                                      |
| ------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1                   | Match in-place projection to the full AR viewport               | `App/CameraActions.swift`, `App/TrainingModel.swift`, `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift`                                                                                                                                                                |
| 2                   | Remove closed training scratch files after save or termination  | `App/Support/TrainingOutputPolicy.swift`, `App/Tests/TrainingOutputPolicyTests.swift`, `App/TrainingModel.swift`                                                                                                                                                                    |
| 3                   | Suspend camera inference while phone training is busy           | `App/Support/TrainingCameraPolicy.swift`, `App/Tests/TrainingCameraPolicyTests.swift`, `App/CameraActions.swift`, `App/ARCameraView.swift`, `App/TrainingScreen.swift`                                                                                                              |
| 4                   | Keep slow saved-result reads from replacing active training     | `App/TrainingModel.swift`                                                                                                                                                                                                                                                           |
| 5                   | Stop terminal cloud polling and avoid unchanged manifest writes | `App/Support/TrainingCloudPolicy.swift`, `App/Tests/TrainingCloudPolicyTests.swift`, `App/Support/TrainingExperience.swift`, `App/TrainingScreen.swift`, `App/HomeScreen.swift`                                                                                                     |
| 6                   | Preserve capture-detail sharing errors and refinement identity  | `App/HomeScreen.swift`                                                                                                                                                                                                                                                              |
| 7                   | Find contained capture rows through descendant queries          | `UITests/HexUITests.swift`                                                                                                                                                                                                                                                          |
| 8                   | Wait for enabled controls before UI-test taps                   | `UITests/HexUITests.swift`                                                                                                                                                                                                                                                          |
| 9                   | Run UI workflow tests when invoked directly                     | `ci_scripts/test_ui_tests.py`                                                                                                                                                                                                                                                       |
| Sorting and buffers | Reuse GPU sort buffers and sort only after direction changes    | `Modules/LiveSplat/Sources/LiveSplat/SplatSortPolicy.swift`, `Modules/LiveSplat/Tests/LiveSplatTests/SplatSortPolicyTests.swift`, `Modules/LiveSplat/Sources/LiveSplat/SavedSplat.swift`, `Modules/LiveSplat/Sources/LiveSplat/SavedSplatRenderer.swift`, `App/TrainingModel.swift` |
| CPU geometry        | Retain only positions after uploading saved Gaussian attributes | `Modules/LiveSplat/Sources/LiveSplat/SplatSortPolicy.swift`, `Modules/LiveSplat/Tests/LiveSplatTests/SplatSortPolicyTests.swift`, `Modules/LiveSplat/Sources/LiveSplat/SavedSplat.swift`, `App/TrainingModel.swift`                                                                 |
| Surface lifetime    | Release the final native preview when training closes           | `App/TrainingModel.swift`                                                                                                                                                                                                                                                           |

The CPU geometry and sort-buffer changes share one portable source/test file and
can also be committed together. The surface-lifetime change shares the session
release helper with fix 2. Real-phone acceptance still includes safe-area AR
alignment, thermal/cooling resume, actual disk cleanup, sustained memory/GPU
behavior and the cloud crossfade. UI tests 7/8 need an Apple simulator run.
