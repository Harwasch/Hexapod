# Fast answer integration

The camera shares one AR session between perception, overlays and capture. Hold
the mic to transcribe, release to ask, tap a subject to select it, and tap a
numbered pin for its card. Controls and failure feedback are icons. Record writes
M1 capture bundles under Application Support/Captures; incomplete bundles have no
manifest. SharedSessionRecorder adapts M1's bundle writer because ARCaptureRecorder
owns a separate private AR session. The minimal Captures list reads finalized bundles and durable upload status.

CameraScreen accepts a CameraModel constructed from CameraProviders. PieceRegistry
resolves VoiceInput, Planning, PerceptionServing, DeepAnswering and the capture
source independently. Cloud supplies both the recorded fake and the real deep provider.
Release uses Cloud when an API URL and token are configured, otherwise offline.
LiveSplat receives shared-session frames and exports before the manifest is finalized.
Training defaults on; the scan icon and Home → Settings row share the real renderer's
persisted preference. Off shows the LiDAR point preview immediately. Hidden debug
off/fake provider overrides take precedence; the user setting applies within real.
Home owns the camera registry and reuses its real renderer; debug probes stay separate.
Home and Captures expose the Globe. Ask requests
preserve the shared scope/siteId contract; the camera currently asks in scan scope.
See [TESTING.md](../TESTING.md) for isolated screens and phone provider selection.

The deep request begins concurrently with fast planning. Deep events wait until
the fast stream finishes, then refine it. A matching label (case/outer whitespace
ignored) within 0.35 m replaces the same pin; distinct labels remain distinct.
A completed question emits done once after both paths; cancellation ends
the old stream immediately. Path failures retain the other answer and emit a
status. No placeholder geometry or subject identity is invented.

PerceptionScene contains portable selection orchestration. A tap chooses a mask;
a voice noun requires a matching on-device classification. Generic questions use
the selected subject or the image centre. Count finds visible foreground
instances, not a complete scene inventory. Depth estimates discard holes and
depth outliers, then use median centres and supported visible bounds. A surface with neighbouring
depth support survives the global depth filter, preserving narrow trunks and curved
crowns. Bounds include those retained samples rather than trimming narrow parts
with per-axis percentiles. These are
approximate visible-surface extents, not calibrated full-object dimensions.
Tracks reproject anchors and associate by distance/IoU, tolerate short gaps, and
age on processed frames. Explicit new selections replace the tracked subject set.
Camera selections lift masked LiDAR depth into world-space surfaces. The overlay
reprojects them with the latest camera and display transform at a requested 60 fps,
keeping geometry until replacement or deselection, and hiding it when the display
camera is stale using PerceptionLimits. Vision runs only for pending taps or selected
objects, off the main thread, one request at a time. Busy camera frames are dropped
before copying; the next available frame starts when the worker is idle without
an interval timer. Queued taps keep owned buffer copies, not ARFrames, with a limit
of two. The 10 Hz target is not a measured device rate. Each successful match
refreshes the selected surface.

## Optional neural segmentation

No weight file, URL or digest is supplied. Vision foreground masks work by
default. To configure a promptable export, inject
RuntimeSegmenter(configuration:layout:) into ScenePerception(segmenter:).
ModelConfiguration requires an HTTPS URL and SHA-256; CoreMLPromptLayout names
the image, point, box and output features. The adapter expects a single-file
.mlmodel with a colour image input, normalized [1,2] point / [1,4] box inputs
(-1 for the unused prompt), and [H,W] or [1,1,H,W] output logits. A split
encoder/decoder export needs its own Segmenter adapter.

Weights download to Application Support/Hex/Segmentation on first use. The first
answer continues with Vision during acquisition. Cached bytes are checked before
reuse, and new bytes are verified before atomic publication. Missing or
incompatible exports fall back to Vision. The cache/integrity boundary is
Linux-tested using injected downloads; no network request was made during this
task. Speech asset installation is similarly runtime-only.

## Device acceptance still required

Linux cannot compile or execute Speech, FoundationModels, Vision, Core ML,
ARKit, RealityKit, AVFoundation or the Metal pass. The guarded adapters are
implementation candidates, not a claimed device build. Run Xcode 26 strict
concurrency compilation, then check on an eligible iPhone:

- Microphone/camera denial, speech asset availability, volatile/final transcript
  order, rapid press/release, interruptions and background/foreground recovery.
- Tool calls with Apple Intelligence available, and rule fallback when disabled.
- An asymmetric image/depth fixture to confirm bitmap orientation, display
  transform, tap location, lifted world anchor and mask alignment.
- The selected model export's feature layout, resizing, digest failure and Vision
  fallback; generic classification does not establish fine-grained species.
- Metal texture formats/coordinates, subject-only brightness, outline glow,
  billboard pins, hit testing and spoken-answer audio routing.
- Recording while asking, HEVC backpressure, raw/smoothed depth and location
  files, finalized bundle playback and rendering/inference frame times.

Depth buffers are copied on the main actor, then submitted to a background writer
actor. The four-packet bound includes the active write; overflow stops recording
and leaves an incomplete bundle without a manifest. Depth files precede their
JSONL frame, and finalization drains accepted packets before publishing a manifest.
Directory creation, JSONL/location writes, synchronization and manifest publication
also run off the main actor. HEVC setup/append and buffer-copy throughput still
need device profiling. No GPU run, paid service, deployment, workflow, signing or
upload was performed.

Open decisions: choose a licensed single-file promptable export and its verified
URL/digest/layout; confirm acceptable visible-surface measurement accuracy and
foreground-count coverage; choose retention/sharing for locally saved bundles.
