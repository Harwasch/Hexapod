# Scan preview (LiveSplat package)

LiveSplat is the existing package/preference identity. Its live renderer is now a
**scan preview**: coloured LiDAR surfels, with no on-device training. The real splat
is reconstructed in the cloud. Ready capture details open the server scan through
the in-app globe. No live GPU geometry is read back or exported by the App.

## Integration and ownership

The App owns one ARSession and one RealityKit ARView. A dedicated serial delegate
performs capture admission and hands immutable frames to bounded mailboxes. It
never converts pixels, projects points, or calls MainActor per frame. A 10 Hz UI
timer polls coverage and state; perception is suspended during capture.

`DefaultLiveSplatRenderer.preparePreview()` creates a `ScanPreviewProcessor`.
The App forwards `offer(ARFrame)` from its delegate and installs `render(context)`
as RealityKit's `postProcess` callback while scanning. RealityKit supplies the
camera image and the sole display loop; there is no MTKView or CADisplayLink.
The callback reprojects from the latest retained camera pose on each display frame.
Camera passthrough is encoded if tracking/depth is unavailable or the previous
preview submission is still outstanding. Start resets the hash on the next GPU
submission; stop disables ingress. A generation check rejects stale summaries.

`LiveSplatPreference.previewEnabled` defaults on. The stored key remains
`hex.liveSplat.training` to preserve existing installations. Off means camera plus
coverage; fusion continues to measure coverage, with no coloured surfel drawing.
Settings calls this **Scan preview**. The simulated provider and replay screen are
labelled Simulated. The old UIKit dots, tile binning, optimizer, per-frame CPU
colour conversion, model uploads and model readbacks are removed.

## GPU resource and work bounds

- 262,144 fixed 64-byte cells (16 MiB), four cells per hash bucket, plus 256 KiB
  bucket claims and eight summary bytes. No growing geometry arrays on the live path.
- World-space cells are 2 cm below 3 m measured depth, 4 cm farther away. Resolution
  level is part of the key. Only high-confidence, finite depth in 0.1–8 m is fused.
- CVMetalTextureCache wraps Y, CbCr, float32 depth and confidence planes directly.
  Frame and texture wrappers remain alive until command completion. The mailbox
  holds one latest frame, and only one preview command may be outstanding.
- At most 12,288 samples/keyframe and 10 keyframes/s. The sampling phase rotates.
  One nonblocking claim wins each bucket per keyframe; contention is skipped and
  retried in later views. There are no spin locks or cross-thread cell publication.
- Four probes match a voxel, use an empty slot, or evict the least recently observed
  slot (lowest observation count breaks age ties). Position/colour use a running
  average capped at 32 observations. Capacity stays flat when the scan moves on.
- Disc radius is 0.85 times the larger of voxel/depth-pixel sample spacing, capped
  at 12 cm (minimum 1.7 cm near, 3.4 cm far). Opaque
  cores and soft edges overlap the sample grid. An internal depth attachment orders
  surfels; live LiDAR depth rejects geometry behind foreground surfaces. Raw camera
  range and colour matrix select the YCbCr conversion, followed by linearisation.
- Coverage is a subtle tint over high-confidence depth with fewer than two voxel
  observations, not a claim about unseen surfaces. The summary samples this same
  condition at fusion rate. The tint hash lookup runs at the internal resolution.
- Three GPU submissions above 8 ms reduce fusion from 10 to 5 to 2.5 Hz, visible
  cells from all to every second/fourth, and longest render edge from 960 to 720 to
  480 pixels. 120 inexpensive submissions recover a level. Low Power/thermal state
  sets a reversible quality floor. Timing includes RealityKit's command buffer,
  so it is conservative, and **8 ms/60 Hz are targets, not phone measurements**.

The pure Swift `SurfelReference.swift` mirrors quantization, hashing, four-way
fusion/eviction, sizing and quality policy. `Math.swift` retains Gaussian
projection/export utilities for offline fixtures; it has no optimizer. The
`ingest(SplatFrame)` API is an offline reference only. It lazily allocates its CPU
hash and is never called for live ARFrames. `FakeLiveSplat` supplies deterministic
coverage for isolated UI tests. See [TESTING.md](TESTING.md) for the evidence boundary.

Apple API references: [RealityKit postprocess callback](https://developer.apple.com/documentation/realitykit/arview/rendercallbacks-swift.struct/postprocess),
[ARSession delegate queue](https://developer.apple.com/documentation/arkit/arsession/delegatequeue),
and [Core Video Metal texture cache](<https://developer.apple.com/documentation/corevideo/cvmetaltexturecachecreatetexturefromimage(_:_:_:_:_:_:_:_:_:)>).
