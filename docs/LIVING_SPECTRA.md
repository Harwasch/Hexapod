# Living lab: simulated voxel spectra (prototype)

Behind `?living=1`, choose **Voxel spectra** in the Living lab panel and give it a
`spectra.json` URL. Use PlayCanvas/WebGL2, with `&norig=1` for the rigged tree.
The panel says **Simulated**. As captured removes the hook; zero strength gives zero
displacement. The scan bytes remain immutable. This is P0/P1 research, not a production
sidecar, trained DynamicTree reproduction, or measured wind. No publication/API integration.

## Format v1

JSON fields:

| Field                                     | Meaning                                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------------- |
| `format`, `version`                       | `hexapod.living-spectra`, `1`                                                          |
| `encoding`, `binary`                      | `float16-le`, relative binary URL                                                      |
| `motionEvidence`                          | `simulated` (required by this prototype)                                               |
| `origin`, `size`                          | Three metre coordinates and extents in the tileset's scan frame: x east, y north, z up |
| `resolution`                              | Lattice samples per axis, 128 for this bake (player accepts 2–128)                     |
| `occupied`                                | Sparse lattice sample count, at most 50,000 for CPU playback                           |
| `frequenciesHz`                           | Sorted unique nonnegative frequencies, at most 16                                      |
| `referenceBearingDeg`                     | Wind **to**, clockwise from north, used for the bake (90°)                             |
| `coefficientOffset`, `byteLength`         | Exact binary offset and byte count                                                     |
| `durationS`, `sampleRateHz`               | Bake provenance: 10 seconds, 30 fps                                                    |
| `sourceChecksum`, `sourceNote`, `metrics` | Provenance and bake measurements; not runtime identity verification                    |

Binary begins with `occupied` little-endian uint32 lattice indices,
`x + resolution * (y + resolution * z)`. Then, with no padding, half-float
coefficients in `[occupied][frequency][xyz][real,imaginary]` order.
Total size is `occupied * (4 + K * 12)` bytes. NaN/Inf, duplicate or out-of-bounds
indices, invalid sizes and truncated/oversized payloads are rejected.

Samples lie at `origin + indexXYZ * size / (resolution - 1)`, including the
bounds; these are interpolation lattice vertices, not cell centres. The baker occupies
all eight corners surrounding every measured leaf-tile splat. Missing samples mean
zero displacement; outside the lattice is static. The cube expands until the sparse
sample count fits the CPU budget, so **128³ does not imply a fixed physical detail level**.
The baker prints the actual spacing and never silently discards occupied cells.

Playback at time t:

```text
d(t) = sum_k (real(C_k) cos(2π f_k t) - imag(C_k) sin(2π f_k t))
```

Coefficients are rFFT/N, with non-DC/non-Nyquist bins doubled. DC preserves mean lean.
The selected frequencies are shared by all voxels. DC plus the strongest energy bins
are retained, with four slots reserved above 1.5 Hz for fine motion. Every frequency
is an integer multiple of 0.1 Hz, so the ten-second playback loops exactly, including
between the last recorded sample and the first. Random access needs no integration.

## Player

Decode half floats once. Each animated frame evaluates K sine/cosine pairs, then the
linear sum once per occupied voxel, into a sparse RGBA32F texture (width 256).
A static R32F lookup texture (width 2048) maps the flattened 128³ lattice to sparse row+1;
zero is empty. The splat hook does eight index/displacement fetch pairs and trilinear
interpolation. It does no per-splat colour classification, Fourier sum, or oscillator
integration. A fixed lab clock caches the evaluated field until time/strength/bearing
changes. Baseline benchmark timing begins before hook synchronization so CPU evaluation
and texture upload are included in the read-back-fenced frame cost.

Strength is a linear amplitude multiplier. Bearing rotates displacement xy by
`referenceBearingDeg - windBearingDeg`; z and temporal phase are unchanged. It rotates
motion vectors, not lattice locations or the tree. This is an artistic approximation:
a new direction does not recompute drag, branch exposure or resonances. Gust-speed and
greenness controls belong to the older arms and do not affect spectra.

The URL is explicit and **must match the displayed scan frame**. There is no automatic
checksum binding or production registration; the prototype is for one manually selected
scan. Requests are aborted on replacement/destruction, previous textures cleared on URL
change, and failed loads leave spectra motion static with an error in the panel.
WebGPU remains unsupported, as for the round-1 hook.

## CPU baker

From the worktree, with the existing local mirror:

```bash
cd tools/captures
TMPDIR=/home/harrison/.cache/codex-living-spectra-tmp \
UV_CACHE_DIR=/home/harrison/.cache/codex-living-spectra-tmp/uv \
uv run --frozen python living_spectra.py \
  ../../.lab-data/minnetonka-tree ../../.lab-out/living-spectra/bake
```

The baker decodes leaf GLBs with their composed tile offsets, avoiding duplicate LOD
parents. It uses the scan's extracted branch hierarchy (200 nodes for Minnetonka).
Each branch has a damped angular oscillator, driven by shared slow gusts, spatial phase,
and faster branch forcing. The parent rotations and positions propagate down the
hierarchy; root pinned, segment lengths preserved. It integrates at 120 Hz for six
10-second forcing periods, discards five warmup periods, and samples the last at 30 Hz.
No biomechanical calibration is claimed; frequencies, damping, drag gains and forcing
are deterministic artistic priors. The period is set by the forcing, not a network.

The lattice is the voxelization of splat occupancy. At those locations the baker
smoothly blends the four nearest branch-segment transforms, then adds spatially coherent
2.3/3.1 Hz flutter away from the woody skeleton and fades the base to an exact pin.
It samples this branch-attached field at the lattice vertices rather than materializing
300 full 2.2-million-splat frames. The resulting sampled displacements are FFT-compressed.
This approximates splat-attached motion at the reported lattice spacing; it cannot
separate two touching leaves or crossed branches inside the same cell.

## What this does not reproduce

[DynamicTree](https://arxiv.org/html/2510.22213v2) learns sparse motion spectra and deforms
mesh-bound Gaussians. This prototype borrows the representation and uses scan-specific
simulation for the source motion. It has no learned prior, mesh binding, covariance
rotation, branch collision, force interaction, or motion-aware sort. Only centres move;
large deformation can expose gaps and stale sorting. Improving the spectrum cannot
repair bad branch attachment or missing leaf geometry. Measured spectra from a real
clip (P2) remain a separate brief.

[Generative Image Dynamics](https://generative-dynamics.github.io/) is a related 2D
spectral-volume reference, not a dependency.

## Calm breeze experiment (round 4)

**Simulated**, research only. Use the articulation sidecar with `?breeze=calm`,
`?breeze=breezy`, or `?breeze=windy`. No query retains round-3 spectra. Bake with
`living_articulation.py SOURCE OUT --breeze --ground 0`: this keeps canonical GLBs unchanged,
zeros ground/trunk-owned bindings before flutter classification, and groups small
(maximum axis <6.5 cm), green splats into branch-specific 12 cm packets. These are
approximate spatial packets, not segmented leaves. The original 3.5 cm bake is
still available for the round-3 reference.

The following publications inform the frequency bands and coherence; **none measured
this Minnetonka specimen**:

- [Campiformio (2012), measured white ash branches](https://digitalcommons.lib.uconn.edu/gs_theses/253/):
  wind-loaded frequencies 0.2–1.1 Hz; pull-and-release mean 0.2 Hz.
- [Jackson et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597771/):
  field strain from temperate and tropical trees; architecture changes broadleaf
  modes. Their sycamore model illustrates a 0.26 Hz fundamental. This is a model
  example, not a measured universal trunk frequency.
- [Kolbe et al. (2022), Scots pine field loads](https://doi.org/10.3390/f13071026):
  discusses approximately 20 s organized turbulence and a response dominated by
  frequencies below the fundamental sway mode. These canopy-scale observations
  motivate slow shared forcing, not independent oscillators at every joint.
- [Roden & Pearcy (1993), poplar photocells](https://pubmed.ncbi.nlm.nih.gov/28313608/):
  flutter-associated light variation at 3–5 Hz.
- [Tadrist et al. (2018), cherry wind tunnel](https://doi.org/10.1098/rsif.2018.0010):
  leaves dominate low-wind foliage motion, branches dominate at higher wind;
  measured leaf torsional frequency 6 ± 1 Hz. Species-dependent, not a reason to
  turn every splat into a rapidly oscillating leaf.
- [Odijk (2014), scaling analysis](https://doi.org/10.1007/s10867-014-9361-0):
  bending stiffness scales as diameter⁴/length³, bending timescale as length²/diameter;
  eddy time as length^(2/3). This is theory, not an empirical amplitude law.
  There is no defensible universal amplitude-by-branch-order statistic here.

Implementation choices are explicitly artistic inferences: a 20/31 s travelling
positive gust envelope, advection at 1.5/2.5/4 m/s (coherence lengths 30/50/80 m
for the 20 s component), branch modal frequencies bounded to 0.24–1.1 Hz, and a
0.19 Hz low-frequency component. Gust delay follows wind bearing. Local angular
caps are 0.006/0.014/0.028 radians times bounded segment-length compliance divided
by sqrt(rig stiffness) and `1 + 0.45 * (order - 1)`. The stiffness is an extracted
geometric prior, not measured EI; neither caps nor wind speeds are calibrated.
Root and trunk nodes stay pinned, satisfying the explicit immobile trunk/ground
constraint; primary branches supply the perceptual crown sway. Global motion
inherits parent rotations, so local response attenuation does not erase parent
movement. Leaf amplitude multipliers are 0.35/0.65/1 with 3–5 Hz flutter.

All classification and ownership are baked. Branch preparation happens once at
load. Each frame evaluates four slow sines per branch and one flutter oscillator
per leaf packet, then rigid quaternion composition. Leaf phases are factored once
at load; each frame shares sine/cosine pairs across the 21 baked flutter frequencies.
For leaf half-angles up to 0.15 rad, degree-five sine and degree-four cosine
polynomials have analytic remainder bounds of 0.15^7/5040 and 0.15^6/720; larger
angles use exact trig. This avoids repeated trig calls without a machine-fitted
tolerance. A packet's branch and flutter
transform are precomposed once, allowing two texture reads in the per-splat leaf
path. No colour classification, nearest-branch lookup or per-splat sine; no scale
change. New presets retain captured colour instead of modulating brightness by
an approximate normal, avoiding crown-wide brightness pulses. Sorting still uses
rest centres. Preset selection reloads the sidecar; comparison cuts are not smooth
preset transitions. Steady playback is continuous and deterministic.

Build and serve the dedicated static 1080p laptop benchmark from the repository:

```sh
TMPDIR=~/.cache/hexapod-living-breeze/tmp heavy node apps/web/scripts/living-lab/serve-bench.mjs --host 127.0.0.1 --port 5710
```

`--host` accepts a specific Tailscale address; wildcard binds are refused. Ports
5710–5719 only. Defaults read source/bake from
`~/.cache/hexapod-living-breeze/{source,bake}`; use `--source` and
`--bake` to override. Open `/`, run the benchmark, download JSON. It records three
interleaved rounds of plain/calm/plain/breezy/plain/windy, four warmups and 32
retained samples per block, median/p95, paired overhead and renderer provenance.
A one-pixel readPixels fence includes CPU evaluation, upload and completed draw,
not display refresh. Device pixel ratio is normalized to one. The +1 ms target
requires a real GPU run; SwiftShader cannot establish it.

### Round-4 software measurements (2026-10-27)

Same 512×320 cameras and depth-two LOD (687,226 splats), three interleaved
rounds, eight retained read-back-fenced frames per block after four warmups.
Final software measurements include the phase cache and bounded half-angle math.

| Arm    | Plain median ms | Animated median ms | Median paired ratio | Paired ratio / round 3 |
| ------ | --------------: | -----------------: | ------------------: | ---------------------: |
| round3 |           783.2 |             1197.3 |              1.533× |                 1.000× |
| calm   |           766.6 |             1183.3 |              1.563× |                 1.019× |
| breezy |           737.6 |             1184.3 |              1.606× |                 1.047× |
| windy  |           748.3 |             1190.5 |              1.595× |                 1.040× |

These shared-machine SwiftShader results show **no dependable total draw-speed
improvement over round 3**. Animated medians differ by about one percent while
paired ratios are slightly worse for the new presets; baseline drift limits the
comparison. The archived round-3 ratio was 2.002× under different load and is not
a direct speedup denominator.

Isolated Node CPU evaluation: round 3 6.15 ms median / 7.41 ms p95; calm
0.87 / 0.99 ms. Packets fall from 196,889 to 21,305; pose/leaf texture uploads
from 6,324,224 to 704,512 bytes/frame. These CPU/upload reductions do not establish
the full +1 ms laptop GPU target. That remains unverified pending a real-device
run of the static benchmark.

The four-second videos retain exact-trig recordings. Final fast evaluation was
compared over 45 whole-tree cases (all presets, strengths 0/1/3, times through
1000 seconds): branch poses bit-identical, maximum serialized leaf-transform
component difference 2.3841858e-7, zero strength exact. This is below visual
resolution; the approximation bounds are analytic rather than machine-fitted.
The final static page and pixel controls use the optimized evaluator.
