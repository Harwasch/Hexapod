# Test each iOS piece independently

Run from the repository root with the installed Swift toolchain (Swift 6.2 on
Linux; Xcode 26 on macOS). No downloads, services, credentials or phone are needed
for these Linux package tests. Dependencies are local on Linux; PhoneTrain adds
an Apple-host-only exception for pinned msplat-ios 3.0.0 (binary framework and
metallibs). Apple package builds need that dependency resolved. On this Linux host,
first run `. ~/hw-tools/swift/env.sh`.

Create the local temporary directories once:

```bash
mkdir -p .tmp-test .swift-scratch
export TMPDIR="$PWD/.tmp-test"
```

Run one command for the piece you are changing. Each command limits build jobs to
two; run them sequentially. `--disable-sandbox` accommodates the restricted Linux
runner. Scratch output stays in this worktree.

| Piece        | One command                                                                                                                            | What it exercises                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| EntitySchema | `swift test --package-path apps/ios/Modules/EntitySchema --scratch-path "$PWD/.swift-scratch/EntitySchema" --jobs 2 --disable-sandbox` | Entity and Ask Codable contracts, including scope/siteId                                |
| Capture      | `swift test --package-path apps/ios/Modules/Capture --scratch-path "$PWD/.swift-scratch/Capture" --jobs 2 --disable-sandbox`           | Bundle/archive/location validation and committed synthetic bundles                      |
| Upload       | `swift test --package-path apps/ios/Modules/Upload --scratch-path "$PWD/.swift-scratch/Upload" --jobs 2 --disable-sandbox`             | Upload state, injected HTTP transport, retries and cancellation                         |
| Voice        | `swift test --package-path apps/ios/Modules/Voice --scratch-path "$PWD/.swift-scratch/Voice" --jobs 2 --disable-sandbox`               | Scripted volatile/final transcript replay, release, cancel and restart                  |
| Router       | `swift test --package-path apps/ios/Modules/Router --scratch-path "$PWD/.swift-scratch/Router" --jobs 2 --disable-sandbox`             | Planning, fixture fast path, deep stream ordering/dedupe/done and question cancellation |
| Perception   | `swift test --package-path apps/ios/Modules/Perception --scratch-path "$PWD/.swift-scratch/Perception" --jobs 2 --disable-sandbox`     | Masks, lifting, tracking, scene selection and yard fixture geometry                     |
| ARAnswer     | `swift test --package-path apps/ios/Modules/ARAnswer --scratch-path "$PWD/.swift-scratch/ARAnswer" --jobs 2 --disable-sandbox`         | Presentation, highlight projection and recorded overlay commands                        |
| App support  | `swift test --package-path apps/ios/App --scratch-path "$PWD/.swift-scratch/App" --jobs 2 --disable-sandbox`                           | Independent persisted provider choices; FIFO disk writes, capacity, drain and failure   |
| Cloud        | `swift test --package-path apps/ios/Modules/Cloud --scratch-path "$PWD/.swift-scratch/Cloud" --jobs 1 --disable-sandbox`               | Recorded SSE parsing, retry/cancellation and server recording parity                    |
| LiveSplat    | `swift test --package-path apps/ios/Modules/LiveSplat --scratch-path "$PWD/.swift-scratch/LiveSplat" --jobs 1 --disable-sandbox`       | Frame replay, coverage, numerical reference, lifecycle and PLY export                   |
| Globe        | `swift test --package-path apps/ios/Modules/Globe --scratch-path "$PWD/.swift-scratch/Globe" --jobs 1 --disable-sandbox`               | Property endpoints, routing, pagination and geographic pins                             |

Repository gates, also from the root. Remove Swift scratch output before the
format gate because Prettier scans its generated JSON too:

```bash
python3 -m unittest discover -s apps/ios/ci_scripts
python3 -c 'import shutil; shutil.rmtree(".swift-scratch", ignore_errors=True)'
pnpm format:check
```

Remove temporary output after the commands finish:

```bash
python3 -c 'import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (".tmp-test", ".swift-scratch")]'
```

## Standalone screens

Open each package's Swift file in Xcode and use its preview canvas, or open the
phone's **Piece testing → Run pieces alone** links. These SwiftUI screens are
compiled only when SwiftUI is available; the app gates access as described below.
They have no microphone, camera or network dependency.

| Piece      | Screen / preview                               | Source file                                                 |
| ---------- | ---------------------------------------------- | ----------------------------------------------------------- |
| Voice      | `VoiceDebugScreen()`                           | `Modules/Voice/Sources/Voice/FakeVoice.swift`               |
| Router     | `RouterDebugScreen()`                          | `Modules/Router/Sources/Router/Fakes.swift`                 |
| Perception | `PerceptionDebugScreen(directory: fixtureURL)` | `Modules/Perception/Sources/Perception/FixtureReplay.swift` |
| ARAnswer   | `ARAnswerDebugScreen()`                        | `Modules/ARAnswer/Sources/ARAnswer/OverlaySink.swift`       |
| App        | `CameraHost(registry:)` in the app             | `App/DebugMenu.swift`                                       |

Voice displays the script's volatile replacements followed by its committed final
question. Router runs the rule planner on editable text; `FakePlanner` supplies
exact scripted plans in tests. Perception accepts a directory containing `bundle/`
and `ground_truth.json`, then reports lifted visible heights. Its Xcode preview
resolves yard-small relative to the source checkout; on a phone the menu supplies
the imported directory. A missing fixture produces a visible error.
ARAnswer records draw, mask and speech commands in `FakeOverlaySink` without audio
or RealityKit. Capture, Upload and EntitySchema use their package tests above;
their standalone device screens belong to those modules' owners and are not added
by this fast-answer job.

## Phone debug menu

In a DEBUG app, open Settings → Testing. TestFlight builds expose the same explicit
entry and store whether testing is enabled; the menu can disable it again. Ordinary
App Store builds do not expose the menu. Leaving testing restores real providers,
with an explicit unavailable state where capabilities or credentials are missing.
Saved developer choices are ignored in App Store builds.

Choose providers independently for capture, voice, planner, perception, deep
answer, live splat, globe data and answer overlay. Choices persist in UserDefaults
under `hex.piece.*`; tap **Apply camera providers** to cancel the current camera
model and construct a fresh one. Fake camera/answer providers display a
**Simulated test providers** label. Real speech may acquire its OS assets at
runtime; use fake voice for an entirely offline test.

For a complete offline camera path:

1. Copy `Fixtures/bundles/yard-small` (or `yard-limited`) into Files on the phone.
   Preserve the folder with both `bundle/` and `ground_truth.json`.
2. Select that fixture under **Capture source**, then tap **Import selected fixture
   folder**. The app copies the folder to Application Support so replay survives
   relaunches. This does not require changing the Xcode project or bundling fixtures
   in a production app.
3. Set Voice, Planner and Perception to **fake**, Deep answer to **fake** or
   **offline**, and Answer overlay to **real** to inspect pins or **fake** to record
   commands without drawing. Apply. No ARKit session or camera permission is used
   by replay; hold/release the microphone control to replay the scripted question.
4. Reopen the menu to run each standalone screen, or probe the selected LiveSplat
   and Globe providers. Fake probes report **Simulated**. LiveSplat receives the camera's shared AR frames;
   Globe uses the property endpoint and Cloud uses the authenticated Ask stream. The globe renders with the bundled Cesium page by default; the menu's "Globe renderer" picker switches to Apple Maps 3D, and "Cesium globe with fixture data" runs it with no API (build the bundle first: `bash apps/ios/ci_scripts/build_globe.sh`).

`App/Support/PieceProviders.swift` registers Cloud's recorded fake and real client,
LiveSplat's off/fake/real renderers, and Globe's fake/real sources. App support tests
exercise configuration, provider resolution, coverage and local capture status.

The camera is Home, with controls for Captures, Globe and Settings. Captures lists finalized local bundles and
status from `Application Support/Uploads/<local-bundle-id>/upload.json`; upload state
stays outside immutable capture bundles. An uploaded capture with a known site opens
that site's Globe route. Map pins and camera pins share ARAnswer's entity card.
A configured web URL enables the published scan preview from the map card.

Public host configuration is read from `hex.api.baseURL` and `hex.web.baseURL`
in UserDefaults, or `HEX_API_BASE_URL` and `HEX_WEB_BASE_URL` in Info.plist.
The write credential comes only from the runtime Keychain API. Supply credentials
in Settings; never commit a token. API URLs may be an origin, `/api`, or `/api/v1`.
With URL and token, release defaults use Cloud; otherwise deep answering is offline.
LiveSplat training defaults on. Home → Settings has the Live splat row; the scan
screen has an icon-only toggle. Both use `LiveSplatPreference.shared`, also injected
into the real renderer. Off switches immediately to the LiDAR point preview without
resetting the scan; on requests training again (device fallback still applies).
The hidden menu selects the provider first: **off** removes the renderer and **fake**
uses simulated training independent of the user preference. Only **real** follows
the user toggle. Applying providers keeps the same real camera renderer; standalone
probes have separate renderers so they cannot reset an active scan.
App support tests cover the default, live mode changes, override precedence and
shared preference identity. On a phone, verify the Settings preference persists across navigation and relaunch,
and repeat with off/fake debug overrides. Globe defaults to the real source;
missing configuration displays Server unavailable. Saved debug provider
choices are ignored outside DEBUG or an explicitly enabled TestFlight menu.

An unregistered live camera session has no catalog scan ID. Its fast path still uses
local perception; its Cloud request omits scene evidence rather than attaching it to
an unrelated saved scan. For saved-scan Cloud testing, use Cloud's standalone harness
with the catalog capture UUID. Background bundle uploads continue after Done, and capture detail offers video replay.
The globe scan picker selects the exact published scan tileset.

One command runs **every Linux integration check**, sequentially, using installed
local dependencies (Swift, CI scripts, web, contracts, API, analysis and pipeline):

```bash
bash scripts/test-ios-integration.sh
```

The command removes `.tmp-test` and `.swift-scratch` on exit. Without
`TEST_DATABASE_URL`, API tests select `--without-db`; with a dedicated PostGIS test
URL they include migrations and DB tests. The DB fixture truncates its test catalog.
See [server testing](../../docs/TESTING-SERVER.md) for the combined replay test.
Browser e2e and Apple-only framework compilation remain separate acceptance gates.

## What fixture tests establish

`FixtureReplay` reads committed float16 depth, confidence, intrinsics and row-major
poses from both bundles. It transposes poses to Perception's column-major layout
and shifts the principal point by half a depth pixel because the fixture uses
integer sample centres. Every independent ground-truth probe must unproject within
5 mm, covering float16 rounding at these distances. Limited frames stay in the log
but do not feed the perception replay.

For Linux, the replay reconstructs semantic RGB colours at depth resolution from
ground-truth bounds and sampled depth. `FakeSegmenter` classifies the fixture's
known flat palette, joining trunk and crown into one tree. It also accepts decoded
RGBA images. This is an oracle mask seam: it tests lifting and orchestration, not
HEVC decoding, segmentation quality or real-world measurement accuracy.

Every normal yard-small frame must yield exactly one shed and one tree. The shed's
centre X/Y and visible X/Y extents must be within 0.4 m of ground truth (two depth
texels plus boundary uncertainty). Its centre Z bound is half the known object
depth plus 0.15 m, and Z extent may miss the full hidden depth plus 0.15 m. Tree
centre X and width use 0.4 m; median centre Y may differ from full-tree centre by
one crown radius; height uses 0.6 m for the narrow, sparsely sampled trunk and
crown boundary. Tree Z uses the same visibility rule based on crown depth. These
are synthetic visible-surface bounds, not full-volume reconstruction claims.

The first replay exposed a 0.69–0.73 m tree-height shortfall and 0.44–0.46 m width
shortfall. The tolerances above were kept: locally supported depth survives global
median filtering, and extents now include all retained surface samples instead of
trimming narrow limbs with per-axis percentiles. Isolated depth outliers remain
rejected. Coherent bad masks/depth still need device validation.

Router's Linux integration test runs scripted Voice → scripted planner → actual
fixture lifting → device fast answer alongside a recorded `DeepAnswering` stream.
It asserts exact fast-before-deep ordering, duplicate suppression, one done,
request scope/siteId forwarding and cancellation when a new question arrives.
Controlled continuations synchronize cancellation and writer tests without sleeps.
The recorder queue counts queued and active packets, rejects overflow, drains FIFO,
and propagates disk errors. Accepted depth files precede their frame JSONL entry;
manifest publication follows drain and video finalization. Overflow fails the
capture rather than silently mismatching video and depth.

Apple-framework compilation, preview layout, permission flows, HEVC throughput,
Metal rendering, TestFlight receipt/settings behavior and real phone accuracy
remain device checks. No network/install, paid service, GPU or cloud job is used by
this test suite.

## Captures, globe and settings acceptance (uxb)

The camera is the root screen. Its callbacks push `CapturesScreen(registry:)`,
`GlobeScreen(registry:route:)` and `SettingsScreen(registry:)` on the existing
NavigationStack. Captures opens a scan detail, never another copy of the site list.
Rows read real finalized bundles, sort by manifest start date, show a video thumbnail
and the final recorded frame timestamp. Upload percentages count acknowledged bytes;
a queued job is not proof of a finished reconstruction. The catalog capture endpoint
supplies Queued / Refining / Ready / Failed and the server-assigned site. Network failure
shows Status offline. Pull to refresh or leave the foreground screen open to poll.
Detail provides video replay, a mapped scan route, video sharing and local deletion.
An upload in progress disables local deletion; the confirmation explains that server
copies remain. No API delete is issued.

Settings uses the shared Keychain credential API, `APICredentials.writeToken` and
`try APICredentials.setWriteToken(_:)`. A successful save resumes queued uploads.
There is no write token Info.plist key and no credential is stored in UserDefaults.
The public API and web origins are build settings. Testing is reachable only in DEBUG
or with a TestFlight receipt; enabling it opens the existing independent provider menu.

`HexUITests` is an XCUITest target and scheme. On macOS, build the local globe, generate
the project, then run:

```bash
python3 apps/ios/ci_scripts/ui_tests.py
xcrun xcresulttool export attachments --path apps/ios/build/HexUITests.xcresult \
  --output-path apps/ios/build/ui-screenshots
```

The workflow runs package `tests` and `ui-tests` in parallel after `app`, retaining the full
result bundle and `ios-ui-screenshots`. Every UI step attaches a screenshot with
`lifetime = .keepAlways`. Screenshots are evidence produced by the simulator run;
Linux does not produce substitutes for these attachments. The UI job has a 40-minute
limit, boots the selected simulator and waits for `bootstatus -b` before Xcode runs,
and gives each test a 300-second default and maximum execution allowance.

UI fixtures launch with `-uiTestFixture yard-small`. The Xcode build copies the
recorded bundle under `UITestFixtures/yard-small` only in Debug and removes it in
Release. `App/Support/LaunchMode.swift` reads the arguments once; Release ignores
them. It supplies the frozen clock, provider selection, capture storage and API seam.
Without this switch, production rendering, providers and timing retain their behavior.

In UI-test mode:

- The camera shows the fixture's first RGB frame, without constructing an ARView or
  starting ARKit, the live perception driver or Vision inference. The answer flow uses
  fixture perception, voice and answers.
- Recording seeds one tracked CPU splat frame after tracking pre-roll. A static Canvas
  displays its initial coverage colours at time zero, without Metal or a display link.
  Guidance resolves once from that frame and freezes, including the tip and coverage ring.
- Camera timelines use one date; UIKit/SwiftUI animations and haptics are disabled.
  Capture screens refresh once. Replay uses a still image. Standalone developer probes
  and the hardware training benchmark are disabled.
- Cesium is created and loaded when its screen opens. The test session forces bundled
  resources, bundled Natural Earth imagery and ellipsoid terrain; no online map providers
  or heap polling. Its clock stops, flights are instant and rendering is on request.
- Captures and upload state live in Application Support/UITest/{Captures,Uploads}.
  `-uiTestResetCaptures` clears only that isolated inventory before launch. The harness
  uses it by default; omit it for a relaunch/resume test. `-uiTestGlobeEmpty` selects the
  existing empty-site fixture. Both flags require `-uiTestFixture`.

### In-process fixture API

`LaunchMode.current.captureAPI` selects `InProcessFixtureCaptureAPI` behind
`FixtureCaptureServing`, only in Debug UI-test mode. This is the chosen API stub:
`CaptureUploadQueue` uses it for store/refine, and the capture catalog refresh uses
it for status. It constructs no URLSession or background upload. Site selection and
globe snapshots use `FakeGlobeDataSource`. The tests enter a fixture credential through
Settings and remove it afterward, exercising the existing Keychain UI.

Store writes the real durable `UploadState` to the isolated upload folder, with
`storedInCloud=true`, no job, and catalog status `not-started`/empty jobs. The real
status presentation therefore shows **Stored in cloud** and enables **Refine in cloud**.
Only the explicit Refine confirmation calls `refine`; the stub persists a fixed job
ID and immediately returns **Ready** on the next status read. Repeated calls and
relaunches retain that result until an explicit fixture reset. No timer advances the
server state, no cloud job is queued and no production API is contacted. Unit tests
cover the complete stored/refine/ready transition and reload from disk. Real multipart
transfers, retry budgets and cloud progress remain device/service gates.

The production-provider probe uses `-uiTestFixture production-providers`: it keeps
quiet rendering and the recorded still while resolving real camera providers. It
checks that selection through Debug's `settings.providers` accessibility value.
It starts no camera, voice or inference work, and its capture/catalog entry points
still select the in-process API. This is a selection check; the Linux default-mode
regression test separately verifies that ordinary launches select no fixture client.

The existing UI tests use `home.record`, `scan.stop`, `scan.tip`, `fixture.still`,
`splat.staticPreview`, `upload.done`, `upload.complete`, `home.captures`, `home.globe`,
`home.settings`, `ask.microphone`, `answer.pin.1` and `entity.card`. Capture screens
provide `capture.row.<local-id>`, `capture.detail`, `capture.status`, `capture.refine`,
`capture.share`, `capture.delete`, `capture.confirmDelete` and `capture.openGlobe`.
The globe provides `globe.scanPicker`, `globe.scan.<capture-id>`, `globe.empty` and
`globe.back`; Settings provides `settings.token`, `settings.signIn`, `settings.signOut`,
`settings.server`, `settings.providers` and the test-mode `settings.back`.
`ci_scripts/test_ui_tests.py` checks the identifiers used by XCUITest against app/module
accessibility declarations, including generated identifiers.

### PhoneTrain seam and adding the later flow tests

`PhoneTrain.PhoneTrainingSession` is a Sendable protocol for progress, lifecycle,
run/result, preview image and cleanup. `DevicePhoneTrainingSession` adapts the existing
`PhoneTrainer` actor; the real trainer's implementation and API are unchanged.
`PhoneTrainingFactory.make(outputDirectory:real:)` is the app selection point. Its
real-construction closure is evaluated on ordinary launches. Only a Debug UI-test
launch selects `FixturePhoneTrainingSession`, with a PNG decoded from the fixture's
still. There is no persisted setting or provider-menu choice for the fake.

The fake has three finite steps, emits `TrainProgress` with a Simulated message,
returns the same PNG on preview requests and writes a fixed one-Gaussian `trained.ply`
result marked `simulated=true`. It uses no Metal, training dataset preparation,
timer or sleeps. Pause/inactivity wait on continuations; resume, cancellation and
close wake them. The UI must label this output Simulated. The real adapter exports
after a finished run and forwards lifecycle controls. Its `previewImage()` returns
nil: native preview consumers keep using `trainer.submitPreview(...)` and await the
submission's `waitUntilReady()` before presenting the surface.

The train/result/in-place/refine UI is being built separately. Add its tests after
those screens and identifiers land:

1. Route the new training screen's construction through `PhoneTrainingFactory.make`.
   Pass a closure that prepares the real dataset and returns
   `DevicePhoneTrainingSession(trainer: ...)`. Keep output outside the immutable
   capture bundle, for example beside Captures in `PhoneTrain/<capture-id>` under the
   same storage root. Start observing `progress()` before calling `run()`; use
   `previewImage()` for the fixture and the native preview seam on a real phone.
2. Forward scene activity to `setActive`, user controls to pause/resume/cancel, and
   await the running task before `closeAndCleanup`. Present `PhoneTrainingResult.ply`
   through the result/in-place UI, preserving its Simulated label. In-place fixture
   viewing must keep the still-camera path instead of starting a second AR session.
3. Keep cloud storage/status/refinement routed through `CaptureUploadQueue` and the
   existing catalog refresh seam. Do not create a separate transport in the new UI.
   Upload alone must leave the stub Stored in cloud; the confirmed Refine action
   deterministically makes it Ready. Derive `captures.badge.*` from that same status.
4. Add the agreed `train.*`, `result.*`, `inplace.*`, `refine.*` and
   `captures.badge.*` accessibility identifiers to the actual controls. Check each
   literal identifier against source, and extend the identifier audit's prefix list.
   The current app still uses `capture.refine`; update the audit and any callers if
   the new UI replaces it.
5. Extend `HexUITests` using `launchFixture()`, `visible`, `tap` and `screenshot`:
   scan → stop → train → result → view in place → confirmed Refine → capture badge.
   Wait for identifiers/state predicates, attach a screenshot at every transition,
   and assert the Simulated label and final Ready state. Test cancel/resume separately;
   for relaunch persistence use `launch("yard-small", resetCaptures: false)`.
6. Run the relevant Linux seam tests and identifier audit, then run `ui_tests.py` on
   macOS. Inspect `ios-ui-screenshots` and the xcresult even on failure. Keep the
   300-second per-test limit and four-job TestFlight gate; do not add a bypass input.

For the bundled globe's browser acceptance on a separate port:

```bash
E2E_PORT=5287 heavy pnpm e2e --config src/globe-embed/playwright.config.mjs
```

This blocks online imagery, checks that the bundled Natural Earth globe still renders,
asserts no ion evaluation-token requests, and saves phone-sized screenshots under
`test-results/globe-embed`. SwiftUI layout, Keychain writes, file-origin WebKit workers,
real ion entitlements, private bucket CORS and LiDAR recording still need the Apple
simulator/device gates. API references were checked against Apple's
[image generator documentation](https://developer.apple.com/documentation/avfoundation/avassetimagegenerator)
and [test attachment documentation](https://developer.apple.com/documentation/xctest/xctattachment).

## Smooth scan preview and capture gate

See [scan preview verification](Modules/LiveSplat/TESTING.md) for the GPU/phone
acceptance checklist and the distinction between measured pure CPU admission and
unmeasured device GPU/frame-rate targets. The shared-session App path records at
15 Hz; the delegate only admits and hands off immutable frames. AppSupport's
`ScanPerformanceTests` runs the five-minute synthetic 60 Hz/15 Hz workload under
XCTest on Linux and in the existing simulator package-test job. `ScanCodecTests`
checks real HEVC decoded counts against frame/depth metadata on Apple platforms.

Run every module and App package sequentially through `heavy`, with `--jobs 1`,
`--scratch-path ./.swift-scratch/<Name>` and `TMPDIR="$PWD/.tmp-test"`. Then run:

```bash
(cd tools/pipeline && uv run --no-sync pytest -q tests/test_capture_bundle.py tests/test_fixture_bundles.py)
python3 -m unittest discover -s apps/ios/ci_scripts
python3 -m unittest discover -s .github/scripts
rm -rf ./.swift-scratch ./.tmp-test
pnpm format:check
```

Device-only acceptance still includes sustained 60 Hz display, ≤0.5 ms delegate
latency under load, memory plateau, thermals, quality recovery, alignment/occlusion,
visual density and real-bundle frame counts. Linux cannot substitute for these checks.

## Phone training Build A

PhoneTrain's Foundation-only dataset preparation and training/report policy tests
need no network or Apple framework. Keep scratch and temporary files in the
worktree; `.build` is not git-ignored and must never be staged. Remove scratch
before running Prettier, which otherwise scans its generated manifests.

```bash
. ~/hw-tools/swift/env.sh
mkdir -p .build/tmp
TMPDIR="$PWD/.build/tmp" swift test --package-path apps/ios/Modules/PhoneTrain \
  --scratch-path "$PWD/.build/swift/PhoneTrain" --jobs 2 --disable-sandbox
TMPDIR="$PWD/.build/tmp" swift test --package-path apps/ios/App \
  --scratch-path "$PWD/.build/swift/App" --jobs 2 --disable-sandbox
TMPDIR="$PWD/.build/tmp" python3 -m unittest discover -s apps/ios/ci_scripts
```

Settings → Testing → **Benchmark train** is present only with
`hex.testflight.debugEnabled`. It measures the latest finalized local capture,
not a replay provider. Reports go to Documents and can be shared; PLY/checkpoints
live beside Captures in Application Support/PhoneTrain. Preset time budgets,
quality, actual cap growth, memory headroom, thermal handling and checkpoint
recovery remain phone measurements. See the [module README](Modules/PhoneTrain/README.md)
for the sweep, complete Apple-only inventory, pinned source API references and
phone acceptance steps. No user-facing post-scan training UI is included.
