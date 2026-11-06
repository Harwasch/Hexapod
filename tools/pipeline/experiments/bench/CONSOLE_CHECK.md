# Full model versus console preview

`bench console-check --scene beach-full --run RUN [--compare-run RUN]` makes one
evaluation-only L4 call on the existing bench image. The comparison defaults to
`ce19d9f1a7a54ee6b98930e757d94d84` (upgrade-excl). It accepts training that completed
even if the subsequent bench evaluation failed, as rescore does. Both retained
`trained.ply` files are checked with the volume API before reserving/launching the GPU;
a missing file is an error. The last logged SPZ is read from `live/`, falling back to
`train/checkpoint/live/`. It is never regenerated from the final PLY.

The local cached final snapshot for `4769ee65e33c44f2aacd87a45dfa3811` was measured as
**100,000 splats, SH0, 1,654,681 bytes (16.54681 bytes/splat)**, at step 60,000. This is
about **1.38%** of its 7,247,091 trained splats. The saved rescore records the retained
PLY at **1,710,314,954 bytes**. A fresh read-only volume lookup timed out in this sandbox;
that is not evidence that the file is missing. GPU renders have not been run here.

## Exact initial orbit pose

This reproduces a **fresh load before auto-rotation or user input**, not the unknown
orbit position of a browser left running. The initial fit depends on browser viewport;
defaults are **1600×900**, overridable with `--viewport-width/--viewport-height`.

The implementation is ported from `apps/web/src/view/liveData.ts` and `live.ts`:

1. Read the actual logged `live-cameras` payload. Decode camera centres (12-byte records)
   and sparse points (9-byte records) using `origin + int16 / 32767 * scale`, rounding
   each coordinate to Float32 exactly as `decodeCameras`/`decodePoints` do.
2. `drawCameras` chooses the coordinate-wise **upper median** (`sorted[floor(n/2)]`) of
   the sparse points, falling back to camera centres, then payload origin. This is the
   orbit target; the payload origin alone is not generally the target.
3. `spreadRadius` sorts distances of **both** decoded sets from that target and uses
   `sorted[floor(n*0.9)]`, not an interpolated percentile. Below three points it uses
   `scale * 0.8`. Set `size = max(radius, 0.001)`.
4. `upOf` normalizes payload up, or falls back to `[0,-1,0]`. `orient` applies Three's
   `Quaternion.setFromUnitVectors(up, [0,1,0])`, including r186's `1e-8` antiparallel
   branch. The display transform is `display = Q * (world - target)`.
5. `frameView` uses vertical FOV 50°, initial aspect `viewportWidth/viewportHeight`,
   `halfH = atan(tan(25°)*aspect)` and
   `distance = size / tan(min(25°,halfH)) * 1.1`. In display coordinates the eye is
   `normalize([0,0.5,1]) * distance`; near/far are `size/1000` and `size*200`.
6. Invert the display transform: `eyeWorld = target + transpose(Q)*eyeDisplay`.
   Convert Three's right/up/back camera axes to gsplat's right/down/forward axes.
   The resulting world-to-camera matrix, eye, target, rotation, radius, clipping planes
   and intrinsics are written as `cameras.orbit` in `cameras.json`, with the live payload.

There is a load-order detail: `drawCameras` frames before `liveConsole.update` moves the
canvas into `#lc-scene`. The resize then changes the **projection only**, not eye/target.
`view.css` makes that panel 16:10, or 4:3 at viewport width ≤620px. Diagnostic orbit PNGs
use 800×500 (800×600 on mobile), with the same FOV and panel aspect; browser CSS rounding,
device-pixel ratio and literal panel pixel dimensions are not being replicated.
Training cameras keep native Dataset resolution and its exact undistorted K/ROI.

A CPU oracle executed the actual TypeScript framing functions and installed Three.js
against the beach-full payload and three up-vector edge cases. The Python world-view
matrix differed by at most 3.6e-15. Reproduction artifacts are under
`tools/pipeline/.tmp/console-check/`; no web files were edited.

## Renders and scores

Pick nearest 10%, 50%, 90% rank of the actual training pool after recorded exclusions
and any internal validation split; sort by capture timestamp (lexical fallback).
Use the pinned gsplat Parser/Dataset to derive the native undistorted cameras from a
temporary three-frame dataset. Do not normalize the frozen world frame. The baseline's
scene must have identical frozen test pixels, intrinsics and extrinsics, so it is rendered
from the primary run's orbit pose, irrespective of its own LiveState framing.

Decode the SPZ with `tools/captures/splat_tiles.py::unpack_spz(sh=True)`, preserving
stored coordinates and SH. The full PLY uses every stored SH band. Render each with
gsplat **classic**, a black background, no PPISP, alignment, colour correction or overlays.
This isolates the model/preview difference; it does not reproduce Spark's renderer,
its dark background, browser colour handling or auto-rotation.

Local results go to `~/.local/share/hexapod-bench/runs/console-check-<id>/`:

- `comparison-{train-10,train-50,train-90,orbit}.png`: labelled trained PLY | snapshot SPZ.
- `baseline-comparison-orbit.png`: primary trained PLY | baseline trained PLY.
- Nine individual renders, `cameras.json`, `scores.json`, and `console-check.json`.
- Counts, bytes, bytes/splat and actual SH degree for each source. PSNR compares clamped
  **float** RGB snapshot versus full-model render over valid training-camera pixels;
  these are not scores against photos. JSON null means identical images (+infinite PSNR).

One model resides on the GPU at a time; PLY host decoding uses bounded chunks. Reserve
16 GiB host RAM and two CPU cores. With L4 this is $1.021392/hour. A fixed 1,200-second
timeout plus 240 seconds of startup/restart allowances reserves **$0.4085568**. Batch
and cumulative tag caps default to **$0.49** and cannot exceed $0.50. A restart is
refused and unknown charges retain their reservation; no automatic second GPU attempt.
