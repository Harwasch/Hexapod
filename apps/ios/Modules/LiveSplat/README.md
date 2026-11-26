# Scan preview (LiveSplat package)

LiveSplat fuses LiDAR depth and camera colour into a bounded live preview, with no
on-device training. Settings → **Preview** selects **Discs** (default) or experimental
**Splat**. Cloud reconstruction remains the published scan; the App never reads
back or exports live GPU geometry.

## Coverage that develops in the camera view

**Coverage colours** defaults on. Under-observed cells appear as translucent,
desaturated cool grey-blue, with a slow, low-amplitude shimmer. Observations and
new viewing directions ease them toward opaque, true-colour cores. A newly mature
region gets one brief soft glow. Both preview styles use the same appearance
function in their existing draw pass, with live LiDAR foreground rejection.
Soft disc edges and Gaussian falloff remain transparent.

`CoverageAppearance.swift` owns the documented prototype thresholds and packs them
into a 64-byte Metal uniform: eight observations, six consecutive high-confidence
observations, and either three 15° azimuth bins or two elevation bands. Targets are
monotonic while a surfel remains resident. Retargeting starts at its current
interpolated appearance, avoiding jumps; development takes 0.7 seconds, the pulse
1.2 seconds, and shimmer has a 3.6-second period. These are visual heuristics for
observed surfaces, not a guarantee of reconstruction quality or unseen coverage.

Settings keeps **Scan preview** and **Preview**, and adds **Coverage colours**.
Turning colours off shows ordinary preview colour; turning Scan preview off shows
the camera while direction measurement continues. Preferences persist independently.
The existing `hex.liveSplat.training` key preserves Scan preview choices. Retired
preference keys are ignored; they cannot disable the new default. Switching styles
resets only preview geometry/animation, retaining the direction grid and recording.
A new scan clears all coverage state. Simulated providers/replay remain labelled
**Simulated**.

## Live guidance

A small icon-first pill near the top shows at most one word: **Around**, **Higher**,
**Slower**, or **Closer**. Priority is motion, range, missing orbit arc, missing
height, then insufficient surface angles. Slower reuses Capture's existing
`MotionBlurMeter.Level`; it does not estimate motion a second time.

`ScanGuidance.swift` is pure Swift. It combines the bounded grid summary with camera
positions around a fixed subject anchor (the first observed coarse-grid centroid).
Its elevation bands match server quality.py: below −10°, level to 20°, raised to
45°, and above. Higher asks for an elevated pass when no above views and fewer than
20% raised views exist. Around uses the largest empty azimuth arc, entering at 60°
and exiting below 45°, rather than counting occupied bins. The within-bin extrema
retain angular gaps with at most 15° bin uncertainty. Range enters Closer beyond
3.5 m and exits below 3 m. A candidate must persist 0.6 seconds; every tip dwells at
least two seconds. Thresholds are grouped in `ScanGuidance.Thresholds`.

The record button has a 24-segment viewpoint ring. Each 15° sector needs 0.4 seconds
of nearby, steady observations; revisits retain credit. A newly completed sector
emits one subtle haptic. This measures a loop around the anchored subject, not
phone heading or a percentage of the whole world. A broad walkthrough or a badly
framed starting view may give a less useful anchor; judge it on the phone.

## Integration and fixed work

The App owns one ARSession and RealityKit ARView. Its serial delegate forwards
immutable frames to a one-frame mailbox. RealityKit's postprocess callback owns the
only display loop/command queue. One submission may be outstanding; tracking/depth
loss or a busy GPU receives camera passthrough. Texture wrappers and the ARFrame
live through GPU completion. A generation check rejects stale results. The 10 Hz
UI timer polls compact summaries and updates tips/ring without geometry readback.

- 262,144 surfels retain the exact 64-byte ABI (16 MiB). Four slots per bucket,
  nonblocking claims, capped running averages and bounded eviction keep memory flat.
- World cells are 2 cm below 3 m depth, 4 cm farther away. At most 12,288 rotating
  depth samples/keyframe and 10 keyframes/s. Discs admit high-confidence depth;
  Splat also admits medium confidence, while coverage needs high confidence.
- The 10 cm direction grid has 32,768 fixed 32-byte cells, two direction bitmasks
  per cell, 32,768 claim bytes and 129 fixed 32-byte reduction records. One coarse
  observation per bucket/keyframe; collisions evict the oldest cell. The CPU
  `CoverageHash` is a test oracle and is never allocated on the live frame path.
- A separate 16-byte animation record per surfel stores previous/target progress,
  transition start and pulse start. The grid, claims, reductions and animation use
  exactly 5,279,776 bytes, independent of scan length. Only the final 32-byte coarse
  summary and the existing eight-byte view summary return to the CPU. No new
  per-frame buffers or render passes are allocated.
- Splat retains a fixed 16-byte normal/opacity per surfel, estimates normals from
  adjacent depth with an edge fallback, and sorts at most 32,768 evenly spaced
  slots with a bounded bitonic network. EWA quads use a 0.3-pixel filter and a
  96-pixel extent cap. It is a seeded preview, not a trained reconstruction.
- Three submissions above the existing 8 ms target reduce fusion from 10 to 5 to
  2.5 Hz, visible cells to every second/fourth slot, and internal longest edge from
  960 to 720 to 480 pixels. 120 inexpensive submissions recover a level. Low Power
  and thermal state impose a reversible floor. Direction updates follow fusion
  cadence. These are work bounds; GPU timing, 60 fps and resident memory need a phone.

`SurfelReference.swift` and `GaussianSeed.swift` provide portable fusion/projection
references; `ingest(SplatFrame)` lazily allocates CPU hashes for offline fixtures
only. `FakeLiveSplat` supplies deterministic isolated UI coverage. See
[TESTING.md](TESTING.md) for commands and the evidence boundary.
