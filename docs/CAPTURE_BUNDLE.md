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
scale are preserved through placement; heading and geolocation still need the
existing capture placement/site context. Local ARKit coordinates are not GPS.

All raw observations remain in the archive. Frames with limited/unavailable
tracking are retained, but M1 refuses reconstruction for such a bundle rather
than pretending those poses are reliable or silently discarding frames. LiDAR
points are not yet used as training seeds; their full recorded depth and
confidence remain available for a later implementation.

## Folder and manifest

All files are flat, case-sensitive ASCII names using letters, digits, `.`, `_`,
and `-`, fewer than 100 bytes. No paths, directories, symbolic links, hard links,
sparse files, duplicate archive members, or undeclared files are accepted.

```text
manifest.json
frames.jsonl
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
frames, 16 KiB per frame-log line. The Swift USTAR writer limits each member to
less than 8 GiB (USTAR's size field); longer recordings must finish a bundle
before reaching that bound. The archive duplicates the folder's bytes locally,
and Upload retains another private archive copy until the app deletes it.

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
The rigid-transform acceptance bound of `1e-5` accounts for float32 matrix
serialization; it is not a fitted reconstruction accuracy threshold.

Optional `depth` and `smoothedDepth` objects each contain `file`, `confidence`,
`width`, and `height`. Missing depth is omitted, never synthesized. Each `.f16`
is tightly packed row-major IEEE 754 binary16, **little endian**, in metres,
converted from ARKit's float32 depth buffer with row padding removed. Preserve
nonfinite/invalid depth samples as reported. Each `.u8` is one tightly packed
confidence byte per pixel (ARKit low/medium/high: 0/1/2). Both references must
exist and have exactly `width × height × 2` and `width × height` bytes. Raw and
smoothed measurements are separate. There is no depth image rotation or resizing.

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
The module does not own app lifecycle, Keychain, permissions, progress UI, or
cleanup policy; those remain M0/App integration responsibilities. Pass a
`BundlePlacement` (latitude, longitude, uncertainty in metres) or a `siteID` whose
site already has a coordinate frame when starting an upload. These use existing
capture placement fields and are persisted with the transfer. Without either,
the source still uploads and its job queues, but the existing georeference stage
will refuse to invent a world location. ARKit does not supply geographic coordinates.

## Verification boundary

Linux exercises Codable models, folder writing, USTAR layout, chunk planning,
chunk materialization, upload-state persistence, parser rejection cases, real
FFmpeg extraction on synthetic video, and pose math/database seeding. API tests
exercise the new route lifecycle using the existing PostGIS and moto fixtures.
They do not prove live R2 authorization or an iOS background upload.

**Device-only implementation is unverified:** Linux excludes ARKit and
AVFoundation code. Xcode must compile it under Swift 6, and an iPhone 15 Pro or
newer must verify camera permission, LiDAR availability with the chosen 4K
format, color encoding, RGB/depth alignment, recording throughput, writer
backpressure, interruption, locked-device file access, and relaunch callbacks.
`start` first warms up tracking; its completion fires when tracking is normal
and recording begins. No recording is claimed during initialization. Every
frame delivered after that boundary is recorded, including later limited
tracking. The recorder fails a capture on writer backpressure or session interruption;
it never marks a dropped-frame recording complete. This preserves every frame
ARKit delivers, but does not claim ARKit delivers every physical sensor exposure.
M1 does not enable any device-dependent feature in App code.

COLMAP triangulation at fixed ARKit poses, real-device drift, the full Modal
training path, and production cloud delivery need a real capture acceptance
run. Synthetic math alone does not verify those. Nothing is deployed by M1.

Protocol references: Apple's [scene depth](https://developer.apple.com/documentation/arkit/arconfiguration/framesemantics-swift.struct/scenedepth)
and [video formats](https://developer.apple.com/documentation/arkit/arconfiguration/videoformat-swift.class),
and COLMAP's [known-pose reconstruction](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses).
