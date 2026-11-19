# Globe verification

Run from the repository root using existing dependencies. No API, token, GPU or network is
needed for the fixtures. Use `TMPDIR="$PWD/.tmp-test"`, create that directory first, and
limit jobs to at most four processes. Heavy commands below are sequential.

| Piece                                                 | One command                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Generate TS + Swift from the bridge IDL               | `node apps/web/scripts/generate-globe-bridge.mjs`                                                                |
| Check generated bridge against the IDL                | `node apps/web/scripts/generate-globe-bridge.mjs --check`                                                        |
| JS bridge, scan ownership and bundle regression tests | `heavy pnpm --filter @twin/web exec vitest run src/globe-embed src/__tests__/checkBundle.test.ts --maxWorkers=2` |
| Swift data, bridge round trips and native dispatch    | `heavy swift test --package-path apps/ios/Modules/Globe --scratch-path ./.swift-scratch/Globe --jobs 2`          |
| Pipeline's tiny KHR/SPZ output fixture                | `(cd tools/pipeline && heavy uv run --offline --no-sync pytest -q tests/test_globe_fixture.py)`                  |
| Regenerate the four simulated gaussians               | `(cd tools/pipeline && heavy uv run --offline --no-sync python -m scripts.make_globe_fixture)`                   |
| iOS build integration contracts                       | `python3 -m unittest discover -s apps/ios/ci_scripts`                                                            |
| Build/copy local iOS page and assets                  | `heavy bash apps/ios/ci_scripts/build_globe.sh`                                                                  |
| Production web entry budgets and Cesium isolation     | `heavy pnpm build && node apps/web/scripts/check-bundle.mjs`                                                     |
| Web fixture, no API                                   | `pnpm --filter @twin/web dev` then open `/globe-embed.html?fixture`                                              |

On this Linux workstation, source `~/hw-tools/swift/env.sh` before the Swift command.
`Bridge/schema.json` is the sole wire definition: the generator writes TS types and Swift
Codable cases. Both language tests decode and re-encode the **same** `messages.json`,
including every command and event. The native dispatcher refuses stale/unknown IDs and
JS refuses events in the native-command channel. Python checks generation and CI wiring.
Do not edit the generated languages independently. Run Prettier after regeneration.

`GlobeView` defaults to Cesium and exposes the existing data-source, route and open-scan
callback seams. The app has a Globe tab; its debug menu selects Cesium or Apple Maps 3D.
`GlobeDebugView` and its Swift preview force Cesium with `FakeGlobeDataSource` and local
splats. The web fixture labels itself Simulated and shows a pin label on tap. The native
fixture opens the existing entity card. The published property router remains an
integration prerequisite for live pins; it is not mounted by this task.

## Bundling and mobile policy

The shared `createGlobeWidget`, tile provider and `addAnswerPins` run in both web and embed.
The embed does not mount the full `SceneBridge` (which fetches the application catalog and
binds mission UI); its typed adapter supplies native snapshots to the shared scene code.
`GLOBE_EMBED=1` selects only `globe-embed.html`, a relative-path IIFE build. The copy script
removes module/CORS attributes for local-file loading and copies Cesium workers, WASM,
Natural Earth imagery and the tiny fixture. The module uses `Bundle.module` and
`loadFileURL(...allowingReadAccessTo:)`, with no HTTP server or remote app page. Only the
catalog and selected scan content use remote URLs in live mode. Navigation away from the
local document is rejected and messages are accepted only from its main frame.

The generated page and engine assets are about **11 MB** uncompressed (initial local
measurement); bundled JS is about **1.04 MB gzip**. Keep them out of git. Both jobs in
`.github/workflows/ios.yml` generate them before XcodeGen; `ci_post_clone.sh` also calls the
same script. Xcode Cloud must provide Node 22.12+ and pnpm 10 with installed lockfile
dependencies before that hook. The build script itself never installs dependencies.
The regular web build includes the entry and checks a **40 KB gzip entry** and **1.8 MB
initial JS** budget. The standalone iOS build checks **1.8 MB gzip** for its combined JS.
These are bundle limits, not startup promises.

Chosen phone defaults, conservative starting policy rather than measured optimal values:

- One selected splat tileset; **64 MiB** cache plus **16 MiB** overflow. Selection destroys
  the old tileset; a late asynchronous load is destroyed instead of replacing the new one.
  This is a Cesium cache target, not a hard total RAM cap or a maximum single tile size.
- Tile **maximumScreenSpaceError 32**; no hidden preload, flight-destination preload or
  collision mesh. This trades some fine detail for fewer requests and decoded tiles.
- Globe cache **64 terrain tiles**, globe screen-space error **4**, and canvas resolution
  capped at one render pixel per CSS pixel. Natural Earth is local; no default ion calls,
  Google imagery, geocoder, timeline, animation controls or picker widgets.
- Request-render mode, infinite maximum time change and one MSAA sample are shared with
  the web bootstrap. Pin updates and tile changes request renders.
- A retained WKWebView preloads when the app starts and survives tab/card transitions.
  Native snapshots wait for `ready`; process termination reloads local resources and
  replays the latest snapshot/selection. The **<0.5 s warm-start goal is unmeasured**.

## Device gate

Linux compiles only Foundation code. On a Mac build the guarded SwiftUI/WebKit/MapKit code,
then use the fixture on an iPhone before treating this as release-ready. Verify file-origin
XHR, Cesium workers/WASM, local imagery, SPZ decoding, WebGL2 and depth composition; the
IIFE avoids file-origin ES-module loading but cannot prove those runtime behaviors.
Check camera framing, footprint holes, pin/card/scan navigation, touch/VoiceOver and the
Apple Maps debug choice. Confirm exact-epoch asset selection and remote R2 CORS in live mode.
The WebKit round-trip test (`WebKitBridgeTests`) is disabled in package tests, which have no
host app for WebKit's content process; it needs an app-hosted test target or this device check.

Measure app launch → JS `ready`, tab activation → first rendered frame (warm target <0.5 s),
peak resident/GPU memory, battery/thermal load, background/foreground, memory warnings and
web-content-process termination/replay. Test large tiles: cache overflow is not a strict
allocation guard. Evaluate whether to discard the retained web view under memory pressure.
No live capture, browser e2e, GPU run, iOS build or device measurement is claimed here.
Delete `.tmp-test` and `.swift-scratch` after collecting local test results.
