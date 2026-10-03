# WebGPU trial: PlayCanvas on WebGPU

A fourth splat renderer, **PlayCanvas WebGPU (beta)** (`playcanvas-webgpu`), sits beside
PlayCanvas on WebGL2 (the default), Spark and CesiumJS. It is the same PlayCanvas splat
renderer drawing the same tiles or the same streamed package, under the same overlay rules
(CESIUM.md, _Dedicated splat renderers_); only the graphics device differs. It stays off by
default until it has been judged on real devices, in person, by the criteria at the end.

Nothing else moves to WebGPU. The globe, the world and every tool stay CesiumJS on WebGL2;
the splats are drawn on their own transparent canvas above the globe's and composited by the
browser (premultiplied alpha). The two canvases share no depth in any renderer, so occlusion
is exactly what it is under PlayCanvas on WebGL2.

## How to switch

- **Settings › Advanced › Splat renderer › WebGPU.** Saved on the device like any setting.
- **The page address**, for a quick A/B without opening Settings (on a phone, edit the URL):
  `?renderer=webgpu` (or `playcanvas-webgpu`), `?renderer=webgl` (or `playcanvas`),
  `?renderer=spark`, `?renderer=cesium`. It holds for that visit only and never changes the
  saved setting, so a shared link cannot leave a device on the trial; choosing a renderer in
  the app (Settings, or the objects panel's switch to CesiumJS) ends it. Settings › Advanced
  says "Chosen by the page address for this visit" while it holds.
- **The numbers**: Settings › Advanced › _Show developer readouts_. On a desktop they are in
  the bottom bar. A phone hides the bottom bar, so the same line is printed under the renderer
  switch in Settings › Advanced: move, stop, open Settings, read.

Judge on a production build (`pnpm build && pnpm preview`, or the deployed site), never the
dev server: the dev server runs PlayCanvas's debug build and unbundled Cesium (CESIUM.md,
_Judging performance_).

## What the readouts say

With a splat scan engaged:

- **`PlayCanvas · WebGPU`**: the trial is drawing on WebGPU.
- **`PlayCanvas · WebGL2 (WebGPU unavailable)`**, and a one-line notice with the reason: the
  trial is drawing with WebGL2 instead. The reasons: _this browser has no WebGPU_, _WebGPU
  needs a secure (https) page_, _no WebGPU adapter or device_ (PlayCanvas also declines
  PowerVR GPUs), _WebGPU did not start: …_, or _WebGPU device lost: …_.
- **`PlayCanvas · WebGL2`**, **`Spark · WebGL2`**: the other renderers, for the A side.
- **`58 fps · p95 21 ms · draw 2.3 ms`**: the overlay's last camera motion
  (`cesium/scanView/frameMeter.ts`). Only frames drawn because the camera moved are counted
  (the overlay draws nothing at rest, and the frames it draws when a tile arrives come at the
  network's pace); a reading is the latest gesture, its last two seconds at most, and stays
  after the camera stops, prefixed `last move`.
  - `fps`: motion frames per second. The overlay draws in the globe's own frame while the
    camera moves, so this is the rate you see, globe and scan together.
  - `p95`: the 95th percentile time between frames: where a fast turn's hitches show and an
    average hides them.
  - `draw`: the median main-thread time of the renderer's draw call. On WebGPU (and on most
    WebGL drivers) the call records and submits work; the GPU's time shows only as a lower
    `fps`. A lower `draw` on WebGPU is main-thread time handed back to the app.

The developer panel (key `d`, desktop) shows the same renderer status with tile and gaussian
counts.

## What to compare in person

Same device, same site, same bookmark, same quality preset, the battery state the same, and
in the order A, B, A (WebGL2, WebGPU, WebGL2) so a device warming up does not favour either
side. Use `?renderer=webgl` and `?renderer=webgpu` and reload between runs.

1. **Frame rate on the same view.** Orbit steadily (right-drag, or two fingers) for about five
   seconds over the densest part of the scan; read `fps` and `p95`. Repeat close to the ground
   (walking height) and from the arrival bookmark.
2. **Smoothness on fast turns.** Fling the view and pinch quickly. Watch for hitches and read
   `p95`: a WebGPU run that averages more frames but hitches more is worse. PlayCanvas sorts on
   the GPU on WebGPU (every frame, in the frame that draws) and in a worker on WebGL2 (a sort
   lands a frame or more later), so a fast turn should show fewer mis-sorted, flickering
   splats on WebGPU; look for it.
3. **Splats trailing the globe during fast moves (the cross-API risk).** The scan and the globe
   are two canvases drawn by two APIs in the same task, and the browser composites them; if a
   browser presents the WebGPU canvas a frame later than the WebGL one, the scan slides against
   the ground while the camera moves and settles back when it stops. Under WebGL2 the overlay
   is drawn from exactly the globe's camera, in step (overlayFrames.ts), and does not trail.
   Pan fast along an edge where the scan meets the world (a road, a building, the clipped
   boundary) and watch the edge; then stop sharply and look for a snap back. If unsure, film
   the screen in slow motion (a phone's 240 fps mode) and compare the two runs frame by frame.
   Any visible trailing on a device rules the trial out as that device's default.
4. **Memory.** The trial runs a second GPU API in the same tab, beside Cesium's WebGL2. Chrome:
   Task Manager (GPU process and the tab); Safari: Web Inspector › Timelines › Memory. On iOS
   the test is whether the tab reloads by itself ("This webpage was reloaded…") after a few
   site switches or a long session where WebGL2 did not.
5. **Load time.** From engaging a site to the scan sharp, by stopwatch, cold (first visit of
   the session) and warm. WebGPU's first frames compile pipelines; the second site of a session
   shows the warm cost.
6. **Phone thermals.** Five minutes of continuous orbit on each renderer: is `fps` sustained or
   does it sag as the phone warms; how warm does it get; battery used.
7. **Correctness.** A scan with objects or motion is drawn with WebGL2 under the trial (see
   below), and the readouts must say "WebGL2 for scans with objects or motion"; on one without,
   new detail fading in without black flashes; rotating the phone, resizing the window;
   switching tabs or apps and coming back (where a device loss is most likely; the trial must
   come back on WebGL2 with the notice, never blank); switching renderers back and forth.

## Devices to try

| Device and browser                                  | Expected API                                                 |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Desktop Chrome / Edge, Windows (D3D12)              | WebGPU                                                       |
| Desktop Chrome / Edge, macOS (Metal)                | WebGPU                                                       |
| Desktop Chrome, Linux                               | WebGL2 fallback unless WebGPU is enabled in `chrome://flags` |
| Safari 26, macOS                                    | WebGPU                                                       |
| Safari 26, iPhone and iPad (iOS / iPadOS 26)        | WebGPU                                                       |
| Android Chrome (Android 12+, recent Qualcomm / ARM) | WebGPU; older or blocklisted GPUs fall back                  |
| Firefox (Windows has WebGPU; elsewhere check)       | WebGPU on Windows, WebGL2 fallback otherwise                 |

A device that falls back is still worth one run: the notice must name the reason and the scan
must look exactly as under PlayCanvas on WebGL2.

## How it falls back

The trial never leaves a scan undrawn (`ScanRendererHost.start`, `createWebgpuBackend`):

- **No WebGPU, or no adapter or device.** PlayCanvas's `createGraphicsDevice` tries WebGPU and
  then WebGL2 on the same canvas (WebGPU takes the canvas only once its adapter and device
  exist), and the trial draws with that WebGL2 device, saying why.
- **PlayCanvas's Null device** (neither started on that canvas: it draws nothing, silently).
  Refused; PlayCanvas on WebGL2, the default renderer, draws on a fresh canvas.
- **WebGPU device lost** (driver reset, GPU process crash, memory pressure). PlayCanvas would
  make another WebGPU device on the same canvas; the trial instead replaces the renderer with
  PlayCanvas on WebGL2 on a fresh canvas, which streams the scan again and draws. Later scans
  this visit go straight to WebGL2; choosing the trial again in Settings tries WebGPU again.
- **The renderer's code did not arrive** (the chunk's download failed: a dropped connection, a
  new release replacing the files). That says nothing about WebGPU, so it is not held against
  it: this scan is drawn by PlayCanvas on WebGL2 with the notice _The playcanvas-webgpu
  renderer's code did not load: …_, and the next scan asks for WebGPU again.
- **A scan with objects or motion** (its root declares `instances`, split objects, a skin or
  telemetry: `declaresObjectsOrMotion`). The work-buffer modifiers that move the objects
  (docs/SCENE_OBJECTS.md §4) are GLSL only for now, so such a scan is drawn by PlayCanvas on
  WebGL2 from the start, WebGPU never tried for it and not held against; the readouts say
  _WebGL2 for scans with objects or motion_ and the renderer line reads _PlayCanvas · WebGL2
  (objects or motion)_. The WGSL port of the motion is to come.

The overlay canvas is tagged with the renderer actually drawing on it
(`data-scan-renderer="playcanvas"` for a fallback on a fresh canvas) and the API it draws with
(`data-api`).

## What changes on WebGPU, in the code

- `createWebgpuBackend` (`cesium/scanView/playcanvasBackend.ts`): the device is made first,
  asynchronously, and the app is an `AppBase` with a camera, gsplats and the gsplat and texture
  asset handlers (`Application` always makes a WebGL2 device of its own).
- A work-buffer modifier is `{ glsl, wgsl? }` (`WorkBufferModifier`); PlayCanvas takes the
  language its device speaks, and one without WGSL is not applied there. Hide and highlight have
  WGSL ports (`PLAYCANVAS_INSTANCE_WGSL`, `SCAN_INSTANCE_RULE_WGSL`; `instanceShaders.test.ts`
  holds the WGSL rule to the GLSL one statement for statement); the motion has none yet, so a
  renderer on WebGPU offers no `setMotion` and the host draws such scans with WebGL2.
- PlayCanvas sorts on the GPU in the frame that draws, so no sort result comes back to ask
  for the frame that confirms a new tile is drawn (handover.ts waits for that before taking
  its parent away). The renderer asks for that one frame itself, once per batch of tiles, and
  never at rest.

## Automated checks

Headless Chromium 141 has software WebGPU (Dawn on SwiftShader's Vulkan, which Chromium
ships) with `--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=swiftshader
--use-webgpu-adapter=swiftshader` beside the usual ANGLE/SwiftShader WebGL flags. Without
`--use-vulkan=swiftshader` the adapter request fails ("A valid external Instance reference no
longer exists"), as it does with `--use-webgpu-adapter=swiftshader` alone, with
`--use-angle=vulkan`, or with only `--enable-unsafe-webgpu`; with the ANGLE flags alone there
is no adapter. WebGPU also needs a secure context (`http://127.0.0.1` is one, `about:blank`
in Playwright is not).

The `webgpu` Playwright project (`apps/web/playwright.config.ts`) runs the tests tagged
`@webgpu`; the `chromium` project runs everything else, as before. A runner without an adapter
skips them and says so.

- `e2e/scanRenderers.spec.ts`: the trial must come up on WebGPU (not its fallback), with no
  WGSL or pipeline errors, and cover the screen where CesiumJS and PlayCanvas on WebGL2 do,
  for the streamed package and for tiles. Measured on the yard fixture: coverage 0.276 for all
  three, overlap with CesiumJS 0.997, with PlayCanvas on WebGL2 1.0 (package) and 0.999
  (tiles).
- `e2e/scanOverlayIdle.spec.ts`: nothing drawn at rest and PlayCanvas's loop paused, on
  WebGPU too: 0 draws and 0 loop ticks in five seconds at rest, before and after a turn, for
  tiles and package; and every variant's turn is read by the frame meter.
- `e2e/instances.spec.ts`: the yard has objects, so the trial draws it with WebGL2 and says
  so; hide, highlight and hide all measured there as under PlayCanvas.

Software rasterisers say nothing about speed, presentation timing, memory or heat: that is
what the in-person comparison above is for. The fallbacks are unit-tested
(`scanRendererWebgpu.test.ts`), since a headless browser cannot lose a device on demand.

## Criteria for making it the default

All of these, on every device in the table that comes up on WebGPU:

1. **Never worse to look at**: the same scan, the same objects behaviour, fades without
   flashes, and **no visible trailing** of the scan against the globe during fast moves.
2. **Faster where it matters**: motion `fps` at least WebGL2's on every device, and clearly
   better (10% or more) on the phones, where the default matters most; `p95` no worse.
3. **No new costs**: no extra iOS tab reloads in a long session; load to sharp no more than
   half a second slower cold; sustained `fps` after five minutes no worse than WebGL2's.
4. **Falls back cleanly** wherever it does not come up on WebGPU (the notice names the reason,
   the scan looks as under WebGL2), and recovers on WebGL2 from a device loss if one is seen.

If a device class fails only on itself (for example trailing on one browser), the default can
still change with that class sent to WebGL2 by detection, rather than waiting on it.

Making it the default is then `DEFAULT_SPLAT_RENDERER = "playcanvas-webgpu"`
(`state/settings.ts`), with a settings migration (version 4) that moves stored `playcanvas`
choices along, as version 3 did when PlayCanvas became the default.
