# Hex capture bundle v1

M1 owns `apps/ios/Modules/Capture` and `apps/ios/Modules/Upload`. Both are local
Swift packages. They use Swift tools 6.0 and `.iOS("26.0")`: PackageDescription
makes the equivalent `.iOS(.v26)` spelling available only at tools version 6.2.
The pure code also supports macOS 13 and Linux. No app UI is included.

## Existing pipeline and integration

The existing twin-api capture routes create a `captures` row, register source
files in `capture_files`, return windowed S3 multipart PUT URLs, and complete
uploads from the returned ETags. Bytes go directly to private R2/MinIO, not
through Fly. The `/captures/{id}/process` route creates a `jobs` row. The Fly
capture worker claims its lease, downloads the source into the `upload` artifact,
and executes `photo-reconstruct`. These are distinct from the separate Worlds
worker and the robot mission `scan_requests` table.

For ordinary videos, `ffmpeg_frames` extracts native frames and applies the
recipe's selection/deduplication settings. COLMAP estimates camera poses; the
pose backbone can estimate a subset first and then register the remaining
images. The Modal CPU image runs pose processing; Modal's GPU stage trains from
the resulting `frames` and binary COLMAP `poses` artifacts. Quality, placement,
tiling, and catalog registration follow. `tools/captures` supplies packing and
capture utilities, and the worker uses `tools/pipeline` through its existing
bridge. This change adds neither another worker nor another training path.

A bundle travels as one **uncompressed USTAR archive**, named `capture.hexcap`,
with content type `application/x-hex-capture`. Its storage key uses the existing
private `captures/{captureId}/source/{fileId}/...` layout. Its capture kind is
`video`, with `metadata.bundleFormat: "hex-capture-v1"`; no enum or database
migration is needed. New processing explicitly sets `capture_bundle: true` on
normalize and pose. Ordinary video processing and its recipe defaults are
unchanged.

The bundle normalize branch validates the archive, extracts **every** RGB frame
at its native dimensions without rotation, scaling, deduplication, or selection,
and checks the decoded count and dimensions against the log. It carries the
ARKit records in `frames_meta.json`. The bundle pose branch writes per-frame
PINHOLE calibration and known poses, using COLMAP's actual database image IDs,
then matches features sequentially and triangulates at fixed poses. It refuses
an empty point cloud, missing cameras, or altered poses. It produces the same
binary COLMAP model the existing trainer consumes. ARKit gravity and metric
scale are preserved through placement; geolocation comes from the bundle's CoreLocation fixes by default. An optional
existing site supplies its coordinate frame through the existing site context.
Heading remains arbitrary unless provided through existing placement controls;
local ARKit coordinates are not GPS.

All raw observations remain in the archive and **every RGB frame** stays in
normalize's outputs, including limited/unavailable tracking. Only frames whose
`trackingState` is exactly `normal` enter COLMAP feature extraction (via
`--image_list_path`), matching, pose seeding, and fixed-pose triangulation.
Original frame indices/names are preserved across gaps. Pose requires
`min_tracked_frames` normal frames (default 10, with an absolute minimum of 2);
otherwise it reports `capture bundle has N of M frames with normal ARKit tracking;
at least K are needed`. The registration check covers only those normal frames.
`poses.json.tracking` records `total`, `normal`, `excluded`, and `excludedByState`;
stage metrics include `excluded` and JSON-encoded `excludedByState` counts, such as
`limited:initializing` and `limited:excessiveMotion`. LiDAR
points are not yet used as training seeds; their full recorded depth and
confidence remain available for a later implementation.

## Folder and manifest

All files are flat, case-sensitive ASCII names using letters, digits, `.`, `_`,
and `-`, fewer than 100 bytes. No paths, directories, symbolic links, hard links,
sparse files, duplicate archive members, or undeclared files are accepted.

```text
manifest.json
frames.jsonl
location.jsonl     (optional)
capture-info.json (optional app metadata)
rgb.mov
depth_00000000.f16
depth_00000000.u8
smooth_00000000.f16
smooth_00000000.u8
...
```

`manifest.json` is written atomically **after** AVAssetWriter successfully
finishes. An interrupted/failed recording has no finished manifest and cannot be
archived by `CaptureArchive.write`. Keep its partial directory for diagnosis or
explicit user deletion.

```json
{
  "version": 1,
  "format": "hex-capture",
  "device": "iPhone16,1",
  "startedAt": "2026-11-16T12:00:00Z",
  "frameCount": 1,
  "depthFrameCount": 0,
  "smoothedDepthFrameCount": 0,
  "video": "rgb.mov",
  "frameLog": "frames.jsonl",
  "files": [
    { "path": "frames.jsonl", "bytes": 312 },
    { "path": "rgb.mov", "bytes": 12345 }
  ]
}
```

The example byte sizes are illustrative; every actual size must match. `files`
includes every payload file, excluding the manifest itself. Counts distinguish
raw and smoothed depth availability. Limits reject an entire bundle, never
subsample it: 128 GiB total expanded archive bytes, 64 MiB manifest, one million
frames, 16 KiB per frame/location-log line, and one million location fixes. The Swift USTAR writer limits each member to
less than 8 GiB (USTAR's size field); longer recordings must finish a bundle
before reaching that bound. The archive duplicates the folder's bytes locally,
and Upload retains another private archive copy until the app deletes it.

`capture-info.json` is optional app metadata (for example `siteName`, `siteID`,
and `duration`). The app writes it before finalizing the bundle, so it must appear
in `files` with its exact byte size. Its fixed filename is its implicit reference;
it needs no additional manifest field. It must contain a UTF-8 JSON object of at
most 64 KiB. Reconstruction validates but does not interpret its fields. Older
bundles without it remain valid; other unreferenced inventory files remain invalid.

Validation reports the `frames.jsonl` line count and manifest `frameCount`, plus
sorted lists of unreferenced inventory files and references missing from the
inventory. Filename lists show at most ten names and the number omitted.

## Frames, clocks, and coordinates

`frames.jsonl` has one UTF-8 JSON object and newline per delivered ARFrame, in
video presentation order. `index` is consecutive from zero. `timestamp` is
seconds on ARKit's monotonic clock **relative to the first recorded frame**;
it starts at zero and strictly increases. It is also the timestamp submitted to
AVAssetWriter. `startedAt` is UTC wall-clock recording start, not a replacement
for the monotonic clock or a claim of submillisecond UTC synchronization.

```json
{
  "index": 0,
  "timestamp": 0,
  "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  "intrinsics": [1000, 0, 960, 0, 1000, 540, 0, 0, 1],
  "imageWidth": 1920,
  "imageHeight": 1080,
  "trackingState": "normal"
}
```

Matrices are **row-major serialized**, despite Apple's SIMD matrices being
column-major in memory. `transform` is a rigid camera-to-world 4×4 matrix in
metres. World coordinates are right-handed, gravity aligned, +Y up. Camera +X
is right, +Y up, and its optical direction is −Z. `intrinsics` is the 3×3 pinhole
matrix for the original, unrotated `capturedImage` dimensions. Pixel coordinates
have their origin at the top left, +X right, +Y down. No device-display orientation
or portrait transform is applied to the movie.

For `T_cw = [R_cw | C]`, define `D = diag(1, -1, -1)`. COLMAP's camera has +Y down
and +Z forward, so:

```text
R_wc = D R_cwᵀ
t_wc = −R_wc C
p_colmap = R_wc p_world + t_wc
u = fx x/z + cx
v = fy y/z + cy
```

The world basis stays unchanged. Flip **camera axes after inversion**, not the
world axes. COLMAP quaternions are Hamilton `(w,x,y,z)`. The inverse is
`R_cw = R_wcᵀ D`, `C = −R_wcᵀ t_wc`. Tests cover identity, translation, quarter
turns, half turns, off-axis points, pixel agreement, and quaternion round trips.
Normal-tracking poses use a bounded nearest-rotation correction. ARKit's float32
matrices are serialized as Doubles, which preserves float32 drift rather than
restoring an exact rotation. For `trackingState == "normal"`, the parser requires
finite 4×4 entries, positive determinant, `max(abs(RᵀR − I)) <= 1e-3`, and
`max(abs(lastRow − [0,0,0,1])) <= 1e-6`. Conversion projects the rotation onto
SO(3) by SVD/polar decomposition (`U diag(1,1,det(UVᵀ)) Vᵀ`), then uses that
rotation for both COLMAP seeding and fixed-pose verification. Camera centres
remain unchanged; the accepted homogeneous-row drift is treated as `[0,0,0,1]`.
Every conversion uses the correction, including poses loaded from older frame metadata.

These are input precision bounds for float32 pose drift, not reconstruction
accuracy thresholds fitted to one machine. The synthetic phone-like regression
uses prescribed rotation drift rounded through float32, measuring approximately
`1e-6`–`1e-4` maximum orthonormality error, with `2e-7` homogeneous-row drift.
Scale 1.1 has error 0.21 and shear 0.02 has error 0.02; both are rejected,
as are reflections, nonfinite values and excessive homogeneous-row error.
The 2026-11-22 production failures establish that the previous strict check
rejected phone bundles; no real-device error distribution has been measured here.
The `1e-3` envelope is an explicit conservative input policy, not a claim about
measured device accuracy. A rejected normal transform reports the frame index
and measured orthonormality, determinant and homogeneous-row diagnostics (or
shape/nonfinite error when those measurements cannot be computed).

For `notAvailable` and `limited:*` tracking, transform validation requires only
finite entries and 4×4 shape. Identity, zero and degenerate startup transforms
remain stored and never enter posing. Original transforms for every tracking
state remain unchanged in the archive and `frames_meta.json`; only downstream
conversion of normal poses applies the rotation correction. All other archive,
frame, timestamp, intrinsics, image and depth checks still apply.

Optional `depth` and `smoothedDepth` objects each contain `file`, `confidence`,
`width`, and `height`. Missing depth is omitted, never synthesized. Each `.f16`
is tightly packed row-major IEEE 754 binary16, **little endian**, in metres,
converted from ARKit's float32 depth buffer with row padding removed. Preserve
nonfinite/invalid depth samples as reported. Each `.u8` is one tightly packed
confidence byte per pixel (ARKit low/medium/high: 0/1/2). Both references must
exist and have exactly `width × height × 2` and `width × height` bytes. Raw and
smoothed measurements are separate. There is no depth image rotation or resizing.

## Phone location and default placement

CoreLocation requests **when-in-use** permission and records delivered fixes during
recording. The containing app must supply `NSLocationWhenInUseUsageDescription`.
Manager and delegate access run on the main actor; writes run on the capture's
serial queue. Denied permission or unavailable GPS leaves location absent and does
not fail recording. The Codable model and log writer have no CoreLocation dependency
on Linux. No format-version bump is needed: older bundles omit these optional fields.

When fixes are delivered, `location.jsonl` is inventoried like any other payload,
and `manifest.locationLog` is `"location.jsonl"`. Each line is one observation:

```json
{
  "timestamp": 1800000000,
  "lat": 37.25,
  "lon": -122.1,
  "altitudeM": 100,
  "horizontalAccuracyM": 12,
  "verticalAccuracyM": 20
}
```

`timestamp` is Unix UTC seconds from CoreLocation, independent of ARKit's relative
clock. Coordinates are latitude/longitude degrees; altitude and accuracies are
metres as reported by CoreLocation. Negative accuracies are retained as invalid
measurement indicators. `manifest.location` is the full observation with the lowest
**positive** horizontal accuracy (first delivered wins ties), or absent if none is
usable. All logged fixes remain stored; the parser verifies this summary against
the log. A negative vertical accuracy prevents altitude from being used for placement.

Normalize writes all fixes to `frames_meta.locations` and the chosen fix to
`source_meta.location` in the existing video-location shape: `lat`, `lon`,
`altitudeM`, `accuracyM`, and `source: "capture-bundle-corelocation"`. It does not
invent per-frame EXIF fixes or GPS/ARKit alignment. The existing `exif_gps` stage
reads this capture location and accuracy, preserves ARKit metres/gravity, and
applies an existing site's frame when supplied. Video uploads are unchanged.

## Resumable protocol

The API base passed to Upload includes `/api/v1`. The app supplies the existing
write bearer token from its credential store; neither tokens nor signed PUT URLs
are saved in `upload.json`.

1. `BundleUploader` copies a completed `.hexcap` into its private persistent
   directory and atomically saves a client UUID before making any request.
2. `PUT /captures/bundles/{clientUUID}` with the existing `CaptureCreate` schema
   creates or recovers the capture. The UUID maps to a unique `hex-{UUID}` slug;
   a transaction advisory lock serializes first requests.
3. `PUT /captures/{id}/bundle-upload` with `filename: "capture.hexcap"`, its
   content type and byte count creates or recovers the single multipart file.
   A changed size, type, or conflicting upload is rejected.
4. Existing `POST /captures/{id}/files/{fileId}/parts` returns fresh presigned
   URLs and the server's part size. Each part is materialized as an on-disk file
   using bounded reads, then submitted through background URLSession. Only one
   part is outstanding. No API Authorization header accompanies the storage PUT.
5. Each successful PUT's ETag is atomically persisted before continuing. On
   relaunch, reconnect the same background session identifier before querying
   outstanding tasks. Missing acknowledgments are safely retried at the same
   S3 part number; expired URLs are re-presigned. Errors remain retryable by
   calling `resume()` again. The archive is retained through completion.
6. Existing `/complete` receives the ordered ETags. A lost response is recovered
   by querying the idempotent bundle-upload endpoint for the completed file.
7. `POST /captures/{id}/process-bundle` queues or returns the same reconstruction
   job, including after it finishes. Explicit reruns use the existing job retry
   tools. Worker wakeup follows the existing committed-job behavior.

All three new endpoints require the existing write token, not the narrower
phone-handoff upload token. OpenAPI and TypeScript contracts include the routes.
Archive validation runs in normalize before Modal pose/training dispatch, so
acceptance of a job means queued, not validated or reconstructed.

The app must retain one uploader per transfer, recreate it with the same private
directory/API base on launch, and forward the system's background-session
completion handler through `onBackgroundEventsFinished` before `resume`.
Deliver the completion handler on the app's main actor. The uploader waits for
ETag persistence and queued delegate events before invoking it. Control-plane
requests run when the app is active or receives a background wake; iOS does not
guarantee continuous execution, and force-quitting requires a user relaunch.
Upload does not own app lifecycle, Keychain, progress UI, or cleanup policy;
those remain M0/App integration responsibilities. Capture owns the location
permission request; the app supplies its usage description and camera permission.
`BundleUploader` explicitly defaults to `placement: .bundleGPS`: it reads the
bounded manifest from the owned archive and derives a `BundlePlacement`. Capture
creation sends `metadata.lat`, `lon`, `uncertaintyM`, and `locationAccuracyM` through
the existing API fields. Optional `siteID` also sends `siteId`; GPS still supplies
the offset in that site's frame. When GPS is absent, a selected site can supply
its frame origin. Without GPS or a site, starting the upload raises
`UploadError.missingLocation`, allowing the app to request a site selection.
Explicit placement remains available with `.explicit(BundlePlacement(...))`.
Placement and site are persisted for relaunch. ARKit supplies no geographic fix.

## Verification boundary

Run `bash tools/pipeline/test_swift_bundle.sh` from any directory in the checkout
to build the real local Swift Capture package and the App's `BackgroundWriter` /
`RecordingDiskWriter`, then validate their folders and archives with Python.
The script can source `~/hw-tools/swift/env.sh` on Linux; otherwise put Swift 6
on `PATH`. Python dependencies must already be installed (`uv sync` at setup).
The ordinary pipeline pytest suite also runs this test, skipping it when Swift
is absent from `PATH`. Builds use four jobs, `heavy` when available, no remote
dependencies, and explicit SwiftPM build/cache/config/security paths under pytest's
temporary directory. Set `TMPDIR` to a disk-backed scratch directory; the script
defaults to `.scratch/tmp` in this checkout. No build output belongs in git.

The harness stages an exact copy of the current App `BackgroundWriter.swift`
alongside a small `CaptureDetails` declaration, avoiding SwiftPM's restriction
against targets outside their package without source symlinks. It exercises 1, 18,
and 48 logical seconds through the real cadence, background queue, disk writer,
bundle writer, and archive writer, with and without capture details. Both depth
streams and GPS are synthetic. Linux replaces finalized video with nonempty bytes:
this verifies bundle validation, not AVFoundation, ARKit, or video decoding.

Linux exercises Codable models, folder writing, USTAR layout, chunk planning,
chunk materialization, GPS log/summary round trips, default upload placement,
upload-state persistence, parser rejection cases, real
FFmpeg extraction on synthetic video, and pose math/database seeding. API tests
exercise the new route lifecycle using the existing PostGIS and moto fixtures.
They do not prove live R2 authorization or an iOS background upload.

**Device-only implementation is unverified:** Linux excludes ARKit and
AVFoundation code. Xcode must compile it under Swift 6, and an iPhone 15 Pro or
newer must verify camera and location permissions (including denial/revocation),
CoreLocation delivery and placement accuracy, LiDAR availability with the chosen 4K
format, color encoding, RGB/depth alignment, recording throughput, writer
backpressure, interruption, locked-device file access, and relaunch callbacks.
The App's shared-session recorder admits coherent packets at a **nominal 15 fps**
from the 60 Hz ARSession input. Admission and immutable-frame handoff run on a
dedicated serial delegate queue. A four-packet bound includes queued and active
work; capacity is checked before retaining/copying a packet. Missing raw/smoothed
depth or confidence drops the whole input frame. HEVC encoding declares
15 fps expected input and disables frame reordering. Every accepted video sample
has exactly one `frames.jsonl` row, with the same relative ARKit timestamp, pose,
raw/smoothed depth and confidence. The first successfully recorded sample is time zero.

Depth float32 → float16 uses Accelerate's
[vImage conversion](<https://developer.apple.com/documentation/accelerate/vimageconvert_planarftoplanar16f(_:_:_:)>)
on the recording actor, respecting row strides; confidence rows are copied there.
The iOS little-endian output preserves the v1 format. No depth conversion or codec
append occurs on MainActor. Saturation or codec backpressure drops an entire packet
before conversion/video append. Such gaps preserve real timestamps and may reduce
the effective video rate below 15 fps; no duplicate frames or synthetic metadata are
inserted. Partial disk/codec failures prevent manifest publication. Stop first closes
ingress, drains accepted packets, finishes HEVC, validates its finalized sample
count against the accepted packet count, then publishes the manifest.

Normalize still decodes every frame and requires **decoded video count == JSONL
count**. This invariant is unchanged; 15 fps is an acquisition policy, not a new
bundle version or a restriction on older 60 fps bundles. The standalone legacy
`ARCaptureRecorder` is not used by the App; its older lossless delegate behavior is
not the shared-session capture contract.

COLMAP triangulation at fixed ARKit poses, real-device drift, the full Modal
training path, and production cloud delivery need a real capture acceptance
run. Synthetic math alone does not verify those. Nothing is deployed by M1.

Protocol references: Apple's [scene depth](https://developer.apple.com/documentation/arkit/arconfiguration/framesemantics-swift.struct/scenedepth)
and [video formats](https://developer.apple.com/documentation/arkit/arconfiguration/videoformat-swift.class),
and COLMAP's [known-pose reconstruction](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses).

## App RGB compression budget

The shared-session `ScanRecordingWorker` builds `rgb.mov` settings through the
pure `CaptureVideoSettings` builder in the Capture package. HEVC is retained.
Its average bitrate is `min(width × height × 15 × 0.65, 55_000_000)` bits/s:
**26,956,800 bit/s at 1920×1440**, about 5.7 times the reported 4.7 Mbit/s default
in the 48-second capture. The 0.65 bits/pixel/frame budget deliberately spends
more bits on reconstruction texture than the default; it is an initial policy,
not a measured optimum. Larger formats cap at 55 Mbit/s, accepting fewer bits
per pixel: the 3840×2160 format records at 55 Mbit/s (~0.44 bits/pixel/frame). The writer retains expected input rate 15, maximum keyframe interval
15 (one second at nominal cadence), and frame reordering disabled. These existing
GOP/reordering settings are preserved; no independent quality gain is claimed for
them. Dropped packets can make the wall-clock GOP longer. `AVVideoQualityKey` is
omitted: support for quality-based control alongside HEVC average-bitrate control
is not assumed. The standalone legacy recorder remains unchanged.

At the target rate, 48 seconds of RGB is **161.7 MB** (decimal) at 1920×1440 and
**330 MB** at 3840×2160. Average bitrate is not a hard byte ceiling: actual encoder output,
scene complexity and movie overhead need device measurement. Duration remains
user-controlled. Raw plus smoothed depth/confidence at 256×192 and 720 packets
adds about **212.3 MB**, so this example's full uncompressed bundle is about
374 MB plus JSON/USTAR overhead. Budget phone space for the recording folder,
archive, uploader's retained archive copy and one upload part; RGB size alone is
not the storage requirement. Higher bitrate cannot undo exposure motion blur;
the exposure lock and slow-down cue below address that.

No upload or worker limits need raising for that example:

- `CaptureArchive` streams payloads in 1 MiB reads and requires each file to be
  less than 8 GiB (USTAR). It does not impose a smaller total archive cap.
- API bundle registration requires positive declared bytes, with no smaller
  bundle-size ceiling. `services/captures.py` uses 8 MiB multipart parts, growing
  them beyond 10,000 parts; the phone's `ChunkPlan` enforces that part-count limit
  and uploads one disk-backed part at a time. Archive bytes go directly to object
  storage, so they do not meet an API HTTP multipart-body limit.
- Pipeline validation caps total payload at 128 GiB and 1,000,000 frames. Those
  parser ceilings are not practical phone/worker storage budgets.
- The capture worker has a committed 20 GB volume and normally refuses new claims
  below 5 GiB free (`WORKER_MIN_FREE_GB`), cleaning finished work first. This is a
  claim-time guard, not a per-job reservation; archive, extracted bundle and decoded
  images still need room together. Long captures can exhaust storage.

Linux Capture tests cover bitrate arithmetic, orientation invariance and the cap.
An AVFoundation-conditional unit test checks the actual HEVC settings dictionary
without opening a device. It cannot execute on Linux. No macOS build or CI dispatch
is needed for this edit. Before release, an eligible iPhone/TestFlight build must
verify that HEVC accepts the settings, measure bitrate/file size on a textured
48-second capture, inspect codec/profile, cadence, keyframe spacing and decoding,
compare detail at matched motion/exposure, and check thermal behavior, writer
backpressure and finalized video/JSONL counts. Also complete a real bundle upload
and reconstruction; simulator or arithmetic tests cannot establish image quality.

## Sharp capture: 4K format, exposure lock and slow-down cue

The scan session asks ARKit for `recommendedVideoFormatFor4KResolution` (3840×2160,
16:9, 30 fps; frames are still admitted at 15 Hz). Its first ten frames must bring raw
and smoothed depth whose aspect matches the image; otherwise the app remembers the
device as unsupported (`hex.capture.fourKUnsupported`) and restarts in the default
1920×1440 format. The manifest's optional `videoFormat` records what was used:
`{"kind": "4k" | "default", "width", "height", "framesPerSecond"}`, the size landscape
as `capturedImage` is stored, and must match the frames.

When a scan starts, after at least a second of metering, the primary capture device
(`configurableCaptureDeviceForPrimaryCamera`) holds its exposure: a metered shutter
at or under 1/250 s is locked as is, a longer one is capped at 1/250 s with ISO raised
by the same factor (`ExposurePlan`; if the ISO ceiling stops it, frames are darker by
the recorded shortfall). White balance is locked. Both return to auto when the scan
stops. The manifest's optional `exposureControl` records `{"mode": "auto" | "locked" |
"custom", "whiteBalance": "auto" | "locked", "durationS"?, "iso"?}`.

Each frame may carry `exposureDurationS` (ARCamera.exposureDuration, in (0, 1] s),
`exposureOffsetEV`, `iso` and `whiteBalanceGains` (red, green, blue). All are
optional; older bundles omit them and stay valid.

While recording, the record button's ring shows predicted motion blur, from
consecutive ARKit poses: `blur_px ≈ shutter × fx × (ω + v / d)`, with `d` the confident
smoothed depth at the image centre (1.5 m otherwise). White below 1 px, amber to 3 px,
red and pulsing above. One pixel at 1/250 s is about 5°/s at the 4K focal length
(fx ≈ 2830 px) and 10°/s at 1440p (`MotionBlurMeter`).

The pipeline extracts frames larger than 1920×1440 as JPEG at ffmpeg `-q:v 2`, the
quality the ordinary video path trains from, instead of PNG: 720 PNGs at 3840×2160 are
~11–13 GB, beyond what the 20 GB capture worker holds beside the archive. 1440p bundles
stay PNG. A 48 s 4K bundle is ~330 MB of RGB plus ~212 MB of depth, and the worker
peaks near 4 GB with the archive, the unpacked bundle and the JPEG frames.

Training at native 3840×2160 would lower the memory-bounded splat ceiling (L40S
11.8M at 1440p, 5.1M at 4K; `gaussian_budget.memory_ceiling`) and roughly triple the
time per step. Until a bench says otherwise, train 4K bundles with
`{"train": {"train_max_side": 2560}}` (2560×1440, L40S ceiling 9.7M).

## Optional live preview export

A bundle may contain `livesplat.ply`, referenced by the optional manifest field
`"liveSplat": "livesplat.ply"` and included in `files` with its exact byte size.
The App no longer exports its live GPU preview. Its thumbnail comes from captured
video, and Ready details show the cloud-trained scan. Older/offline previews remain
accepted for bundle compatibility; any export must finish before the manifest is published.
The binary little-endian PLY stores ARKit world metres, float xyz/normals, SH DC,
opacity logits, log scales and Hamilton wxyz rotation, with at most 32,768 vertices.
Zero vertices is valid for the simulated renderer. The Python validator checks the
bounded header, vertex inventory, exact payload size and finite floats. Unknown or
unreferenced payloads remain invalid. Archives and the Swift writer preserve the file.
The reconstruction pipeline ignores this optional preview; it does not initialize
training or alter metric poses from it.
