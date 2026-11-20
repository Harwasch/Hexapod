# Hex for iPhone — M0

Requires Xcode 26.2 and XcodeGen (`brew install xcodegen`). From `apps/ios`, run
`xcodegen generate`, open `Hex.xcodeproj`, select your team and run Hex on an iPhone
15 Pro or newer with LiDAR. Capture and Upload must be present from M1 before
project generation/build. All eight modules are local packages.

`HEX_BUNDLE_ID` defaults to `com.harwasch.hex`; override that build setting with
your registered identifier. The shell opens an AR camera with icon-only record
and hold-to-talk controls. Both controls call no-op integration hooks in M0.
Older devices and devices without LiDAR have disabled controls.

Dispatch the **iOS** workflow manually for simulator validation. `packages` (20-minute
limit) and `app` (30-minute limit) run in parallel; `tests` (30-minute limit) follows
`app`. Every module and AppSupport builds against the iOS Simulator SDK, so UIKit/ARKit
paths compile. Package failures are collected into a summary table with individual log
artifacts. The unsigned app build uses xcbeautify, preserves its raw log, and reports
all `error:` lines in the summary. Tests run for packages with test targets on an available
iPhone 17 Pro simulator, falling back to the newest available iPhone/runtime. SwiftPM
and app DerivedData caches reduce repeat work. Each independent build bundles the globe;
tests reuse the app job's globe artifact. New dispatches cancel older runs on the same ref.

Set `testflight=true` and `bundle_id` to archive and upload after all three jobs pass
(`testflight` has a 45-minute limit). Required repository secrets: `ASC_KEY_ID`,
`ASC_ISSUER_ID`, `ASC_KEY_P8` (the complete PEM key), and `APPLE_TEAM_ID`. Missing
secrets skip upload with a notice. The API key needs access to certificates,
identifiers and profiles and permission to upload builds; configure the app record,
bundle identifier and signing access in the Apple account first. Xcode uses
automatic provisioning and signing. No private key is committed or uploaded as an
artifact. Xcode Cloud runs `ci_scripts/ci_post_clone.sh` after checkout; configure
its workflow to use this script directory.

Linux validation (Swift 6.2, up to four build jobs):

```bash
. ~/hw-tools/swift/env.sh
export TMPDIR="$HOME/.cache/hex-ios-m0"
mkdir -p "$TMPDIR"
for module in EntitySchema LiveSplat Voice Router Perception ARAnswer; do
  swift test --package-path "apps/ios/Modules/$module" --jobs 4
done
```

Additional Linux checks (using the existing PyYAML and jsonschema environment):
`python3 -m unittest discover -s apps/ios/ci_scripts -v`. These validate the schema
fixture and exercise the workflow's actual missing-secret, key preparation and
cleanup shell steps without contacting Apple.

The manifests retain tools version 6.0 and spell the iOS minimum as `.iOS("26.0")`:
`.iOS(.v26)` requires PackageDescription 6.2 even with a Swift 6.2 compiler.
EntitySchema defines session-world positions and extents in metres (Y up), optional
WGS84 latitude/longitude/ellipsoid altitude, bundle-relative per-keyframe masks,
JSON-valued attributes and module/model provenance. Its Codable types transport
data; ingestion must validate against `Modules/EntitySchema/schema/entity.schema.json`.
The committed fixture covers every JSON value kind and an integer beyond Double's
exact range.

Unverified on Linux: XcodeGen generation and all Apple-framework compilation,
camera permission and session lifecycle, supported-device checks, touch and
VoiceOver behavior, signing/provisioning, Xcode Cloud discovery, and TestFlight
upload. These require Xcode and/or an eligible physical iPhone and Apple account;
the dispatch workflow has not been run. No capture, voice recognition or rendering
pipeline beyond the AR camera is implemented in M0.
