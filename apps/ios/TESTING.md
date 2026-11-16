# Test each iOS piece independently

Run from the repository root with the installed Swift toolchain (Swift 6.2 on
Linux; Xcode 26 on macOS). No downloads, services, credentials or phone are needed
for these package tests. All package dependencies are local. On this Linux host,
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
| Cloud        | See that module's `TESTING.md`: `Modules/Cloud/TESTING.md` after merge.                                                                | Placeholder owned by Cloud                                                              |
| LiveSplat    | See that module's `TESTING.md`: `Modules/LiveSplat/TESTING.md` after merge.                                                            | Placeholder owned by LiveSplat                                                          |
| Globe        | See that module's `TESTING.md`: `Modules/Globe/TESTING.md` after merge.                                                                | Placeholder owned by Globe                                                              |

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
| App        | `CameraHost()` in the app                      | `App/DebugMenu.swift`                                       |

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

In a DEBUG app, long-press **Current scan** at the top of the camera. In a
TestFlight app, first tap the gear and enable **Enable piece testing menu**, then
long-press **Current scan**. Release access requires both a `sandboxReceipt` and
the persisted setting; ordinary App Store builds do not expose the menu. Disabling
the TestFlight setting restores real/offline camera defaults while keeping test
choices saved for later. App Store launches also ignore saved test choices.

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
   and Globe providers. Fake probes report **Simulated** and retain a run counter
   within that provider instance. The sibling renderers/data sources are not linked
   yet; their real choices report unavailable until registered.

`App/Support/PieceProviders.swift` owns the typed registry. Its three
`TODO(merge)` lines mark registrations for Cloud's `DeepAnswering`, a LiveSplat
adapter and a Globe adapter. Register one factory per piece after merge. The latter
two use the local `DebugPieceService` seam for standalone probes; their eventual
renderer/source adapters own the sibling-specific setup. Unregistered choices
fail visibly and never silently substitute a fake or contact a service. The
registry and queue tests run separately via the App support command above.

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
