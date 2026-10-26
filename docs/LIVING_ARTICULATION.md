# Living lab: articulated branch spectra (round 3)

Choose **Articulated branch spectra** under `?living=1` in PlayCanvas/WebGL2, and supply
an `articulation.json` URL. The label is **Simulated**; As captured removes motion.
Round-2 voxel spectra, gust and clump arms remain available. No production registration,
WebGPU support, learned model, or measured wind is implied.

## Representation and playback

Nodes are parent-first. Each node represents its incoming branch segment, rotating about
its parent's rest position (its proximal joint). A root rotates about its own position and
is pinned by the baker. Local rotations are three-component rotation vectors in radians,
encoded as K≤16 complex Fourier coefficients. CPU playback evaluates these, composes
quaternions down the hierarchy, and uploads global XYZW quaternions plus translations.
Children inherit their parent's motion; this is forward kinematics, not independent
world-space displacement vectors. Strength scales local rotation angles. Bearing rotates
rotation-vector axes horizontally by `referenceBearingDeg - windBearingDeg`; it does not
re-solve drag or branch exposure.

Each splat carries two node IDs, one blend weight, and a leaf-packet ID. These immutable
bindings are loaded by exact per-tile position checksum, reordered to PlayCanvas's Morton
resource order, and copied through an extra work-buffer stream. Unknown tiles stay static.
The final-draw hook performs no branch search or colour classification. Most splats use
one rigid transform; within 6 cm of a joint, woody splats blend only the selected branch
and its parent. Unrelated nearby branches are never blended. Quaternion blending uses
hemisphere-correct nlerp; translation blends linearly. This local joint transition is an
approximation, not a dual-quaternion skin or a mesh.

The hook rotates the centre and, in `modifySplatRotationScale`, left-multiplies the splat's
**XYZW** orientation by the same quaternion. Scale eigenvalues stay unchanged. PlayCanvas
2.22.6 passes XYZW to this hook even though its packed rotation reader returns WXYZ.

Green small splats (maximum axis <6.5 cm) are grouped into 3.5 cm spatial packets per branch.
Each packet has a fixed pivot, rotation axis, phase, 2–6 Hz frequency, and 2–5.7° amplitude.
Its quaternion is evaluated once per packet on the CPU, then rotates both centres about
the pivot and covariance before branch motion. Packets never span two assigned branches.
These are **approximate leaf groups**, not segmented physical leaves; a leaf may span
packets, or a packet may contain parts of several leaves. The number of packets, not just
200 branch nodes, contributes to CPU/upload cost.

Leaf colour gets a relative two-sided Lambert factor from its old and new shortest-axis
normals, preserving the captured photograph at zero motion. This is simulated relighting
of baked colour, not recovered albedo or new visibility. Existing centre-based sorting
remains at rest; large motion can still expose gaps or sorting errors.

## Sidecar v1

`articulation.json` identifies `hexapod.living-articulation`, version 1, and
`motionEvidence: simulated`. It contains `nodes` (position/parent), shared sorted
`frequenciesHz`, `referenceBearingDeg`, `leafCount`, and `tiles`, keyed by canonical
position checksum with `{offset, count}` into the binding binary.

All numbers below are little-endian; floats are IEEE float32, quaternions XYZW.

| File           | Layout                                                                               |
| -------------- | ------------------------------------------------------------------------------------ |
| `motion.bin`   | `[node][frequency][xyz][real, imaginary]` rotation-vector amplitudes                 |
| `leaves.bin`   | Per packet: pivot xyz, unit axis xyz, frequency Hz, phase radians, amplitude radians |
| `bindings.bin` | Per splat, four uint32: node+1, parent+1, first weight 0–65535, leaf-packet+1        |

Zero node ID is static; zero packet ID has no flutter. Fourier coefficients use the same
normalization as voxel spectra: `real*cos(2πft) - imaginary*sin(2πft)`, with non-DC and
non-Nyquist rFFT bins doubled. Exact byte lengths, finite values, hierarchy order, node and
packet ranges are validated. Limits: 1024 nodes, 16 terms, 500,000 packets. A sidecar must
match the scan's frame and tile identities; checksums guard binding, not authenticity.

## CPU baking and the Spool crop

With the existing environment (no added dependencies):

```bash
cd tools/captures
UV_CACHE_DIR=$TMPDIR/uv uv run --frozen python living_articulation.py \
  ../../.lab-data/minnetonka-tree ../../.lab-out/round3/tree-bake
UV_CACHE_DIR=$TMPDIR/uv uv run --frozen python living_articulation.py \
  ../../.lab-data/spool ../../.lab-out/round3/spool-bake --spool-crop
```

Minnetonka's rig has 200 nodes and no bindingPath. The baker therefore uses nearest
branch-segment assignment. It refuses a rig with a bindingPath rather than silently
replacing existing bindings. Every tile, including parent LODs, receives checksum-matched
bindings. No original scan or rig bytes are changed.

The Spool command reads leaf tiles once and writes a derivative one-tile scene under
`spool-bake/scene/splat/`. The explicit scan-frame ROI is x 0.55–1.75, y 0.15–1.30,
z 0.23–0.72 m. It retains low ground below z 0.355 and vegetation-coloured splats above
it, excluding axes ≥12 cm. This removes distracting chair fragments; it is a heuristic
crop, not semantic segmentation. `crop.json` records the selection. The derivative is
SPZ re-encoded with DC colour only; both comparison arms use the identical crop.

Vegetation is clustered into 22 cm xy patches with at least 40 splats. Each tuft has a
pinned root at the estimated flat ground z 0.29 and three segments reaching its measured
95th-percentile vegetation height. Root/branch structure is inferred, not recovered stems.
The scene retains static soil and unclassified splats.

Damped oscillators integrate at 120 Hz for six ten-second periods, discard five warmup
periods, and sample the last at 30 Hz. Branches have different forcing frequencies and
phases. These deterministic artistic priors are not calibrated biomechanics. FFT
reconstruction error only measures compression against that simulation. P2 motion
recovered from real footage remains outside this experiment.
