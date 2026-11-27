# Rebuilt UI SDK pre-flight

Reviewed 2026-11-19, baseline `1f3f77ba3..cd498d4cf`, plus the two expression
simplifications below. Review mode: self; compile correctness only. No Git writes
or CI dispatches. This host runs Swift 6.2 on Linux and has no Apple SDK.

No additional definite initializer, argument-label, access-control or concurrency
error was established. This is a source/documentation review, **not an iOS 26
type-check or Metal compilation**. Apple's live documentation now includes newer
SDK declarations; availability and older overloads/examples were considered.

## Changes

- `App/CameraScreen.swift`: evaluate the listening waveform's phase and `CGFloat`
  height before passing the height into the nested view builder. This reduces
  inference within `TimelineView` / `ForEach` without changing the formula.
  [TimelineView initializer](<https://developer.apple.com/documentation/swiftui/timelineview/init(_:content:)-67h35>),
  [frame dimensions](<https://developer.apple.com/documentation/swiftui/view/frame(width:height:alignment:)>).
- `Modules/LiveSplat/Sources/LiveSplat/MetalPreview.swift`: separate pixel radius,
  point radius and the final `CGFloat` clamp. Multiplication/division order and
  the existing 2–14 bounds are preserved; no tolerance was changed.
  [CGContext ellipse](<https://developer.apple.com/documentation/coregraphics/cgcontext/fillellipse(in:)>).

These are preventive inference simplifications requested by the brief. The
original isolated arithmetic expressions also type-check on Linux; no Apple
compiler timeout is claimed to have been reproduced.

## File-by-file coverage

Every Swift file from `git diff --name-only 1f3f77ba3..HEAD -- '*.swift'` is listed.
Repeated framework overloads are consolidated. The earlier
[App API inventory](App/APPLE_API_AUDIT.md) supplies the common layout, gesture,
AR recorder and Foundation symbol index; its linked symbol pages were revisited.
The following table covers the rebuilt UI and the additional changed files.

| File relative to `apps/ios`                                   | Review result                                                                                                                                                                                                                                  |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `App/ARCameraView.swift`                                      | ARView initializer, representable witnesses, permission await and AR session delegate signatures match. Main queue delegate and preconcurrency conformance retained.                                                                           |
| `App/CameraActions.swift`                                     | ARView.project, frame transform, impact feedback and UIKit geometry uses checked. Camera model/tasks remain MainActor; detached operations capture immutable frame transfer or Sendable data.                                                  |
| `App/CameraScreen.swift`                                      | Both TimelineView schedules, geometry, shapes, gestures, actions, sheets, tasks, scene changes, toolbars, progress, images, VideoPlayer/AVPlayer and site-picker calls checked. Waveform expression split.                                     |
| `App/DebugMenu.swift`                                         | Navigation destinations, picker bindings, file importer and single-URL result match. URLProtocol overrides/client callbacks and stream reading match; fixture subclass retains its existing unchecked Sendable conformance.                    |
| `App/HexApp.swift`                                            | App lifecycle adaptor and background-session delegate signature match. Completion stays in a MainActor object captured by a MainActor Sendable callback.                                                                                       |
| `App/HomeScreen.swift`                                        | NavigationStack/NavigationLink, ShareLink URL/custom label, confirmation dialog, refreshable, tasks, secure field/focus/submit, menus, image generation and replay signatures checked. UI state remains MainActor; detached rows are Sendable. |
| `App/SharedSessionRecorder.swift`                             | AV writer/input/adaptor, callback finishWriting, CMTime, CoreVideo and Core Location calls checked. Only the continuation enters the Sendable writer callback. Disk finalization crosses to actors with Sendable values.                       |
| `App/Support/APICredentials.swift`                            | All four SecItem functions, CFDictionary/CFTypeRef result bridging, OSStatus and accessibility constant match. Computed query creates local dictionaries.                                                                                      |
| `App/Support/BackgroundWriter.swift`                          | NSLock.withLock and continuations match. Mutable state remains lock protected; the worker and disk writer are actors. Linux compiled/tested.                                                                                                   |
| `App/Support/CaptureFlow.swift`                               | Date and formatted-duration uses checked; enum is Equatable/Sendable. Linux compiled/tested.                                                                                                                                                   |
| `App/Support/CapturePreview.swift`                            | UIImage constructors/JPEG encoding and AVAssetImageGenerator async image tuple match. Detached reads/writes return Data/Void; UIKit stays MainActor.                                                                                           |
| `App/Support/CaptureUploadQueue.swift`                        | Detached filesystem/archive work captures Sendable values and returns an actor. Actor calls are awaited; background completion is MainActor/Sendable. Linux compiled/tested.                                                                   |
| `App/Support/FixtureCaptureRecording.swift`                   | Filesystem/JSON calls and disk actor hops checked. Detached packets contain Sendable frame/data values. Linux compiled/tested.                                                                                                                 |
| `App/Support/LocalCaptures.swift`                             | Filesystem/JSON/URL uses checked; captures and details are Sendable. Linux compiled/tested.                                                                                                                                                    |
| `App/Support/PieceProviders.swift`                            | Registry/probes/preferences are MainActor. Fixture stream captures URL/continuation and cancels via a Sendable task handle. App-facing module constructors checked. Linux compiled/tested.                                                     |
| `App/Tests/BackgroundWriterTests.swift`                       | Continuations and test actors compile; new metadata write awaits disk actor. Linux tests pass.                                                                                                                                                 |
| `App/Tests/LiveSplatWiringTests.swift`                        | MainActor preference/renderer construction and public module access compile. Linux tests pass.                                                                                                                                                 |
| `App/Tests/PieceProvidersTests.swift`                         | MainActor registration closures and protocol conformances compile. Linux tests pass.                                                                                                                                                           |
| `App/Tests/ReleaseFlowTests.swift`                            | Upload actor/UI queue boundaries, fixture recording and filesystem calls compile. Linux tests pass.                                                                                                                                            |
| `Modules/ARAnswer/Sources/ARAnswer/ARAnswer.swift`            | Public evidence-fact initializer and value properties compile; Foundation only. Linux tests pass.                                                                                                                                              |
| `Modules/ARAnswer/Sources/ARAnswer/EntityCard.swift`          | Public initializer includes optional pin/ask callbacks and required close callback. Button/Label/Text/shape APIs checked; view and callback creation remain MainActor.                                                                         |
| `Modules/ARAnswer/Tests/ARAnswerTests/ARAnswerTests.swift`    | New value-model evidence assertions compile and pass on Linux.                                                                                                                                                                                 |
| `Modules/Globe/Sources/Globe/CatalogState.swift`              | JSON decode/awaited transport matches; status/site properties and fetch are public. Linux tests pass.                                                                                                                                          |
| `Modules/Globe/Sources/Globe/CesiumGlobeView.swift`           | WKWebView JavaScript callback uses arguments:in:in:completionHandler:. WebKit delegates, representable and UIView hooks checked; session stays MainActor.                                                                                      |
| `Modules/Globe/Sources/Globe/GlobeBridge.swift`               | URL resolution and public Sendable bridge structures compile; no UI/actor crossing introduced. Linux tests pass.                                                                                                                               |
| `Modules/Globe/Sources/Globe/GlobeData.swift`                 | Foundation network/URL/JSON calls checked. Public catalog types and source protocol are Sendable. Linux tests pass.                                                                                                                            |
| `Modules/Globe/Sources/Globe/GlobeView.swift`                 | Menu/picker/List/task/sheet forms checked. MapPolygon, Annotation, MapContent styling, controls and MKPolygon pointer initializers checked. UIKit/Safari wrapper stays on representable actor.                                                 |
| `Modules/Globe/Tests/GlobeTests/BridgeTests.swift`            | Foundation round-trip/URL and bridge assertions compile and pass on Linux.                                                                                                                                                                     |
| `Modules/Globe/Tests/GlobeTests/CatalogStateTests.swift`      | Transport actor and public catalog methods compile and pass on Linux.                                                                                                                                                                          |
| `Modules/Globe/Tests/GlobeTests/WebKitBridgeTests.swift`      | MainActor WebKit calls checked; task-group children capture only a Sendable stream. Still explicitly disabled for lack of a package-test app host.                                                                                             |
| `Modules/LiveSplat/Sources/LiveSplat/LiveSplat.swift`         | Public coverage hint/renderer values compile and pass Linux tests. No Apple-only call in this file.                                                                                                                                            |
| `Modules/LiveSplat/Sources/LiveSplat/MetalPreview.swift`      | UIView drawing, CADisplayLink, MTKViewDelegate, default library/pipeline/buffer/encoder/command APIs checked. Driver stays MainActor; no Metal completion captures. Radius expression split.                                                   |
| `Modules/LiveSplat/Tests/LiveSplatTests/LiveSplatTests.swift` | New gap-hint assertions and existing numeric tests compile and pass; bounds/tolerances retained.                                                                                                                                               |
| `Modules/Perception/Sources/Perception/FixtureReplay.swift`   | Existing split Float16/transform expressions and Sendable snapshot construction compile. Linux replay/geometry tests pass.                                                                                                                     |
| `UITests/HexUITests.swift`                                    | XCUIApplication launch/terminate/launchArguments, queries/predicates, waits, taps/press/typeText, screenshot attachment/name/lifetime/add and async setup/teardown signatures checked. Class isolation supplies MainActor.                     |

`Canvas`, `Gauge`, `ContentUnavailableView` and `sensoryFeedback` do not occur in
these changed Swift files. Progress uses `ProgressView()` and a custom shape ring.

## Additional signature references

- SwiftUI presentation/navigation:
  [Boolean sheet](<https://developer.apple.com/documentation/swiftui/view/sheet(ispresented:ondismiss:content:)>),
  [item sheet](<https://developer.apple.com/documentation/swiftui/view/sheet(item:ondismiss:content:)>),
  [Boolean destination](<https://developer.apple.com/documentation/swiftui/view/navigationdestination(ispresented:destination:)>),
  [ShareLink URL example](https://developer.apple.com/documentation/swiftui/sharelink),
  [confirmation dialog](<https://developer.apple.com/documentation/swiftui/view/confirmationdialog(_:ispresented:titlevisibility:actions:message:)>).
- SwiftUI lifecycle/input:
  [task and defaulted parameters](<https://developer.apple.com/documentation/swiftui/view/task(id:name:priority:file:line:_:)>),
  [refreshable Sendable handler](<https://developer.apple.com/documentation/swiftui/view/refreshable(action:)>),
  [two-value onChange](<https://developer.apple.com/documentation/swiftui/view/onchange(of:initial:_:)-4psgg>),
  [SecureField](https://developer.apple.com/documentation/swiftui/securefield),
  [FocusState binding](<https://developer.apple.com/documentation/swiftui/view/focused(_:)>).
- Video/AR:
  [async image generation](<https://developer.apple.com/documentation/avfoundation/avassetimagegenerator/image(at:)>),
  [VideoPlayer](<https://developer.apple.com/documentation/avkit/videoplayer/init(player:)>),
  [ARView projection](<https://developer.apple.com/documentation/realitykit/arview/project(_:)>),
  [finishWriting Sendable callback](<https://developer.apple.com/documentation/avfoundation/avassetwriter/finishwriting(completionhandler:)>).
- Security:
  [copy](<https://developer.apple.com/documentation/security/secitemcopymatching(_:_:)>),
  [update](<https://developer.apple.com/documentation/security/secitemupdate(_:_:)>),
  [add](<https://developer.apple.com/documentation/security/secitemadd(_:_:)>),
  [delete](<https://developer.apple.com/documentation/security/secitemdelete(_:)>),
  [after-first-unlock accessibility](https://developer.apple.com/documentation/security/ksecattraccessibleafterfirstunlockthisdeviceonly).
- WebKit/maps:
  [callback JavaScript labels](<https://developer.apple.com/documentation/webkit/wkwebview/callasyncjavascript(_:arguments:in:in:completionhandler:)>),
  [MapPolygon](https://developer.apple.com/documentation/mapkit/mappolygon),
  [MapContent stroke](<https://developer.apple.com/documentation/mapkit/mapcontent/stroke(_:linewidth:)>).
- UI testing:
  [XCUIApplication](https://developer.apple.com/documentation/xcuiautomation/xcuiapplication),
  [predicate query](<https://developer.apple.com/documentation/xcuiautomation/xcuielementquery/matching(_:)>),
  [screenshot attachment](<https://developer.apple.com/documentation/xctest/xctattachment/init(screenshot:)>),
  [MainActor async setup/teardown](https://developer.apple.com/documentation/xctest/set-up-and-tear-down-state-in-your-tests).

## Metal, project and workflow

Both `LiveSplat.metal` and `SubjectHighlight.metal` were read. Kernel buffer and
texture attributes, thread/device/constant address spaces, scalar dual-number
operators, texture read/write/sample calls, sampler syntax, precise math and the
safe math pragma agree with the
[Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
The removed coverage decoration leaves unused parameters, not a signature mismatch.
The Swift driver still binds the same buffer indices. Shader resources use SwiftPM
processing and are loaded with
[makeDefaultLibrary(bundle:)](<https://developer.apple.com/documentation/metal/mtldevice/makedefaultlibrary(bundle:)>).
Neither shader has been compiled here.

The [XcodeGen specification](https://yonaskolb.github.io/XcodeGen/Docs/ProjectSpec.html)
confirms `bundle.ui-testing`, source-relative exclusions and scheme test targets.
`HexUITests` depends on `Hex`, sets `TEST_TARGET_NAME: Hex`, builds for test and has
a Debug test action. `App/Package.swift`, `Tests/**` and root `*.md` are excluded
from the app source entry; all current App Markdown files are at that root.
Assets remain in the App source tree. The post-build fixture is copied only for
Debug. Privacy keys, generated Info.plist, public service substitutions, iPhone
device family and portrait orientation are consistent with the sources.

`ui_tests.py` selects an available iPhone UDID using the package-test policy and
passes it to `xcodebuild test -project Hex.xcodeproj -scheme HexUITests`, with
Debug configuration, derived-data and result-bundle paths, four jobs and serial
testing. Old results are deleted before the run. Workflow artifact paths match
the script's absolute paths. Export uses the modern command:

```sh
xcrun xcresulttool export attachments --path apps/ios/build/HexUITests.xcresult \
  --output-path apps/ios/build/ui-screenshots
```

[Apple's xcresulttool release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-16_3-release-notes)
identify the modern attachments command separately from legacy object export.
Export and artifact steps run with `always()`; manually added screenshots use
`keepAlways`. No workflow or project correction was warranted. Actual XcodeGen
generation, installed simulator availability and CLI execution remain untested here.

## Validation

After the Swift edits, the five changed packages passed serially under `heavy`, using:

```sh
. ~/hw-tools/swift/env.sh
mkdir -p .tmp-test-r5
TMPDIR="$PWD/.tmp-test-r5" heavy swift test --package-path apps/ios/Modules/LiveSplat \
  --scratch-path ./.swift-scratch-r5/LiveSplat --jobs 4
```

The same command was run for `Globe`, `ARAnswer`, `Perception` and `App` (scratch
name `AppSupport`). Results: Globe 16, LiveSplat 39, ARAnswer 6, Perception 17,
AppSupport 14: **92 Swift tests passed**. The requested Python CI-script gate
passed all **18 tests**.

The frontend's iOS-target syntax parse passed on all 35 changed Swift files, as
did a Swift 6 strict-concurrency XCTest isolation probe. Isolated old/new arithmetic
matched exactly for 15 point radii and 42 waveform heights. Parsing requires no SDK
and does not resolve APIs. These are supplementary checks, not Apple-framework
type checks. The temporary probes are not added to the permanent test suite.

`pnpm format:check` must run after scratch cleanup: the requested `-r5` scratch
names are outside the existing ignore rules, so generated SwiftPM JSON/YAML is
otherwise reported as unformatted. Scratch and temporary directories are deleted
at completion.

## Remaining risks, descending order

1. **Medium: Apple type checking and shader compilation.** SwiftUI generic
   constraints, exact iOS 26 SDK actor/Sendable annotations, macros and Metal code
   generation still require Xcode. Passing Linux packages excludes Apple-only UI.
2. **Medium: simulator execution/resources.** XcodeGen output, bundled Cesium web
   resources/default Metal libraries, Keychain behavior and screenshot export need
   the macOS run. No UI flow or attachment was executed locally.
3. **Low: historical documentation gaps.** Some live symbol pages select a newer
   overload, a different platform or fail to load. Existing orientation-based
   ARFrame transforms and some legacy convenience constructors have Apple example
   support but no exact historical SDK declaration checked here.
4. **Low: existing runtime concurrency assumptions.** Main-thread AR/Metal/location
   delegate delivery and the retained immutable ARFrame transfer justify the
   existing preconcurrency/unchecked boundaries; device validation remains necessary.

## Phone training Build A (2026-11-26)

The new module's Apple-only source inventory is complete in the
[App API audit](App/APPLE_API_AUDIT.md#phone-training-build-a-2026-11-26).
This is local source verification, without an Apple SDK or network:

- `Modules/PhoneTrain/Sources/PhoneTrain/CaptureDataset.swift`: awaited AVAsset
  track discovery followed by a synchronous actor-owned AVAssetReader pull loop;
  CoreImage/CoreGraphics RGB scaling and ImageIO JPEG encoding. Verify iOS 26
  AVAssetReader annotations, track Sendability, image orientation and pixel/depth
  colour registration in Xcode and on a saved phone scan.
- `Modules/PhoneTrain/Sources/PhoneTrain/PhoneTrainer.swift`: awaited session
  creation, steps, metrics, evaluation, checkpoint, PLY, preview and close on
  msplat's global runtime actor. Config/descriptor/telemetry values are Sendable
  in the pinned library. No direct Metal factory is called; msplat owns Metal
  creation. Private actor ownership, pause boundaries and auxiliary-operation
  exclusion avoid concurrent native operations. Verify native checkpoint restore,
  foreground transitions, cancellation, thermal stop and resource release on phone.
- `Modules/PhoneTrain/Sources/PhoneTrain/ProcessMemory.swift`: task VM info pointer
  rebinding and physical footprint. Confirm Darwin symbol imports/type layout
  with the iOS 26 SDK and compare the report to Instruments.
- `Modules/PhoneTrain/Sources/PhoneTrain/BenchmarkRunner.swift`: serial sweep,
  thermal notifications and sampled timeline, `os_proc_available_memory`, `uname`,
  actor-owned tasks and atomic JSON. Confirm peak sampling and critical detection
  across native calls and suspension. No background training is scheduled.
- `App/TrainBenchmarkScreen.swift`: MainActor SwiftUI, async detached inventory,
  scene-phase forwarding, cooperative cancellation, icon rows and URL ShareLink.
  `App/DebugMenu.swift` adds only the existing-style gated NavigationLink. Confirm
  hidden access, latest-scan choice, Files visibility and share sheet on phone.

`project.yml` wires local PhoneTrain and exact-version remote Msplat 3.0.0;
`Modules/PhoneTrain/Package.swift` includes that remote only on macOS hosts.
Verify XcodeGen package unification, binary linking and metallib packaging in the
Apple build. Linux tests compile the pure prep/policy/report code, not these five
Apple-only files. API source-file/line references, preset assumptions and the
complete phone acceptance list are in [PhoneTrain](Modules/PhoneTrain/README.md).

## Build B

See [Build B compile self-audit](BUILD_B_REPORT.md#compile-self-audit) for the
complete Apple-only source inventory and SDK ownership checks. The final local
gates cover PhoneTrain, LiveSplat, Upload, App support, CI scripts and formatting.
Apple compilation and physical-device acceptance remain required.
