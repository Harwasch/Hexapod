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
