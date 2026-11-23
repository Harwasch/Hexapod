# Scan preview (LiveSplat package)

LiveSplat is the existing package/preference identity. Settings → **Preview** selects
**Discs** (the existing default) or the experimental **Splat** renderer. Both fuse
LiDAR depth and camera colour into the same bounded voxel layout, with no on-device
training. Cloud reconstruction remains the published scan. No live GPU geometry is
read back or exported by the App.

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
Settings retains **Scan preview**, adds the **Preview** picker and an **Inset**
toggle (default off). They persist independently. Switching Discs/Splat clears only
the preview hash on its next submission, so medium-confidence Gaussian seeds cannot
leak into Discs; capture recording is unaffected. The simulated provider and replay
screen are labelled Simulated. The old UIKit dots, tile binning, optimizer, per-frame CPU
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

## Experimental Gaussian and inset paths

`GaussianSeed.swift` defines the portable seeding/sort reference. A Gaussian has a
world-space surface normal, equal tangent standard deviations of half the disc
radius, and a normal deviation of one fifth of that (at least 1 mm). Adjacent valid
depth samples estimate the normal; image edges and depth discontinuities fall back
to the viewing normal. Medium/high depth confidence maps to opacity 0.45/0.85;
coverage continues to count high-confidence observations only. RGB is constant
per Gaussian. Photometric refinement is omitted until a safe phone budget and a
visual benefit are measured; there is no refinement job to enable accidentally.

The GPU retains all 262,144 voxel slots and a parallel fixed 16-byte normal/opacity
attribute per slot. For low compute, Splat draws at most 32,768 evenly spaced hash
slots (16,384/8,192 at reduced quality). This deliberately trades density for work;
it is not a trained reconstruction. A descending bitonic network sorts those slots
from each current camera, with index tie-breaking and invalid depths last. Six
vertices form each EWA quad: the pinhole Jacobian and display transform project
its 3D covariance to 2D, a 0.3-pixel filter regularizes it, and a three-sigma conic
uses premultiplied back-to-front alpha blending without depth writes. LiDAR depth
still rejects foreground occlusion. Quad extent is capped at 96 pixels per axis.

The inset uses the same buffers, active renderer, command buffer, quality policy
and one-in-flight guard. A bounded GPU reduction finds the full scene centroid and
bounds; a virtual camera orbits it once per two minutes. Sparse observations tint
teal and an amber wireframe marks the phone frustum. Tap the corner to enlarge;
drag it to rotate. `ScanOrbit` shares the normalized viewport with `ScanInsetHost`,
the transparent SwiftUI gesture host. No second display loop or CPU model readback
is introduced. The inset has at most a 320-pixel longest edge. Its two textures
require at most 1,228,800 bytes. Additional fixed GPU buffers total 4,505,648 bytes
(attributes, sort order and reduction partials), independent of scan duration.
Normals and sorting are skipped in Discs; the inset reduction/draw is skipped when
the toggle is off. These are work bounds, **not measured GPU timings or a measured
memory plateau**. Metal compilation, rendering, 60 fps and appearance need the phone.

The pure Swift `SurfelReference.swift` mirrors quantization, hashing, four-way
fusion/eviction, sizing and quality policy. `Math.swift` supplies Gaussian projection/export math for offline fixtures and
analytic tests; it has no optimizer. The
`ingest(SplatFrame)` API is an offline reference only. It lazily allocates its CPU
hash and is never called for live ARFrames. `FakeLiveSplat` supplies deterministic
coverage for isolated UI tests. See [TESTING.md](TESTING.md) for the evidence boundary.

Apple API references: [RealityKit postprocess callback](https://developer.apple.com/documentation/realitykit/arview/rendercallbacks-swift.struct/postprocess),
[ARSession delegate queue](https://developer.apple.com/documentation/arkit/arsession/delegatequeue),
and [Core Video Metal texture cache](<https://developer.apple.com/documentation/corevideo/cvmetaltexturecachecreatetexturefromimage(_:_:_:_:_:_:_:_:_:)>).
