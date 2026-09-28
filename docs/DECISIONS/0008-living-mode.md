# ADR 0008 — Living Mode: a stateless modal model on the rig; generative models only as offline teachers

**Status:** accepted · **Date:** 2026-09-28

## Context

The Living Survey ([LIVING_SURVEY.md](../LIVING_SURVEY.md)) moves a measured splat tree without
ever writing its measurement: every frame is `displaced = f(canonical, t, wind)`, and calm
restores the canonical bytes exactly. Until now `f` was the S8/S9 model — nine fixed sinusoids
with an `f^-1/2` amplitude falloff, per-node resonances from `radius / length²` with a fitted
constant, and per-splat flutter phases hashed from the splat's index. It moved, but nothing in it
was sourced: the frequencies came from a radius `skeleton.py` cannot measure on most of a crown
(~130 of 189 nodes of the synthetic tree fall back to the cloud's resolution limit), the forcing
was a tuning constant, and the crown shimmered splat by splat instead of in patches.

We still have no recorded motion of any tree. [LIVING_WORLD.md](../LIVING_WORLD.md) plans for
synchronized clips and an anemometer; until those exist the question is how to get believable,
honest motion from what we do have — a skeleton — and what role video and world models may play.
Two research passes on 28 September 2026 (tree animation; physics on splats, video and world
models) answered it. The findings that decided this ADR:

- **Branch frequency depends on branch length alone**: `f = 2.55·L^-0.59` Hz (Coder 2000, used by
  Habel, Kusternig & Wimmer, "Physically Guided Animation of Trees", EG 2009, eq. 17). It needs no
  radius, which is exactly what our extracted skeletons lack.
- **Whole-tree sway follows a pendulum law for open-grown broadleaves**, `f0 ∝ 1/√H` (Jackson et
  al. 2021, Biogeosciences 18, 4059, Table 2), and **deflection is linear in U²** (same paper).
- **Real trees are lightly damped**: ζ = 8.6 ± 2.2 % in leaf, 3.9 ± 1.3 % leafless (Jackson et al.
  2019, J. R. Soc. Interface 16: 20190116, pull-and-release on four broadleaves).
- **Stateless spectral animation is cheap and never repeats** (Habel et al. 2009: per-branch
  damped-oscillator response to the wind spectrum `P(f) ∝ v_m/(1 + f/v_m)^{5/3}`, synthesised
  into motion textures sampled along irrational trajectories, "previous states never accessed").
- **Damping cannot be recovered from monocular video** (Wind on Trees, arXiv 2609.17810), MPM
  physics on splats is seconds per frame on trees (PhysGaussian 1.8 s, PhysFlow 15.6 s,
  DynamicTree arXiv 2510.22213), and generic 4D-from-video fails on trees in the only comparative
  user study (DynamicTree: 1.7 and 3.7 of 100).

**Hexapod is intended for commercial use (decided 2026-09-28).** That decides which models can be
part of the product at all.

## Decision

### 1. Motion comes from a stateless modal model on the rig

Every rig joint belongs to one **branch** — a chain of joints continuing one limb — and every
branch is one damped harmonic oscillator; the trunk's branch is the whole-tree pendulum mode. A
branch's motion is read out of a precomputed spectral **motion texture** along a straight,
irrationally sloped trajectory, so it has the oscillator's stationary response spectrum, never
repeats, and needs no previous frame. Parent motion propagates to children by composition, as
before. Leaf flutter is a spatially correlated field advected with the wind. The whole frame is a
pure function of `(t, wind, seed, rig)`; `U = 0` gives the identity by value.

Phase 1 is the **allometric** rung: parameters from the skeleton alone.

| quantity           | rule                                                           | source                                                        | status                   |
| ------------------ | -------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------ |
| branch frequency   | `2.55·L^-0.59` Hz, `L` the branch's length (m); ×2.5 leafless  | Coder 2000 via Habel et al. 2009, eq. 17 and §5.4             | cited (unit assumed m)   |
| whole-tree mode    | `2.4/√H` Hz                                                    | Jackson et al. 2021 Table 2; `2.4` read off their Fig. 2a     | law cited, constant est. |
| whole-tree damping | 0.086 summer, 0.039 winter                                     | Jackson et al. 2019                                           | cited                    |
| limb damping       | 0.086 → 0.15 with √(limb's share of leaf tips), 4 levels       | Habel §5.4 citing Moore & Maguire 2004 ("near critical")      | **UNVERIFIED prior**     |
| wind spectrum      | shape `(1 + f/U)^(-5/3)`, taken at each branch's own frequency | Simiu & Scanlan 1986 via Habel eq. 14                         | cited                    |
| drag               | deflection ∝ `U²`                                              | Jackson et al. 2021                                           | cited                    |
| bend per branch    | same angle for every branch, spread over its joints by length  | elastic similarity (McMahon & Kronauer 1976); magnitudes est. | estimate                 |
| gusts              | +35 % ± 30 % every ~20 s for ~5 s (SpeedTree-style envelope)   | chosen                                                        | estimate                 |
| leaf flutter       | wavelengths 4–10 leaf sizes, advected at 0.3·U                 | Habel §7.2 (≥ 4× leaf size, advected by −W·t)                 | estimate                 |
| radius             | **not read**                                                   | —                                                             | —                        |

Two departures from Habel, both measured: the 2D texture spectrum is solved by inverse Abel
transform so a straight-line sample has the oscillator's 1D spectrum (Habel's radially symmetric
texture smears a ζ = 0.2 resonance ~20 % low); and trajectory directions are chosen so the line
does not pass within 1.5 wavelengths of its own start over an hour of travel, which "irrational"
alone does not guarantee.

The parameters travel in a **motion sidecar**, `motion.json` beside `rig.json`, which gains a
`"motion": "motion.json"` pointer (a claim written down, never probed for):

```json
{
  "format": "hexapod.motion", "version": 1, "motionEvidence": "allometric",
  "rigChecksum": "fnv1a32:12000:8b008bc0", "nodeCount": 214, "seed": 1,
  "treeHeightM": 6.46, "leafSizeM": 0.092, "referenceSpeedMps": 10,
  "wind": { "meanSpeedMps": 5, "bearingDeg": 0,
            "gust": { "strength": 0.35, "variance": 0.3, "frequencyPerMin": 3, "durationS": 5 },
            "turbulence": { "along": 0.8, "across": 0.6 }, "canopyAdvection": 0.3 },
  "seasons": { "winter": { "dampingScale": 0.4535, "branchFrequencyScale": 2.5, "flutterScale": 0 } },
  "nodes": { "branch": [...], "mode": [...], "share": [...], "frequencyHz": [...],
             "damping": [...], "gainRad": [...], "flutterM": [...] },
  "provenance": { "branchFrequency": { "rule": "...", "source": "...", "status": "cited" }, ... },
  "generator": "tools/captures/motion_params.py"
}
```

Per-node columns, ~53 bytes per node. `tools/captures/motion_params.py` writes it (called by
`synthetic_tree.py` and `skeleton.py`); `deriveMotionSidecar` in `@twin/world` is its tested twin.
The runtime is `packages/world/src/living.ts`, `spectral.ts`, `leafFlutter.ts`, `motionParams.ts`.

### 2. Every motion says how much evidence stands behind it

`motionEvidence` is a ladder, strongest first; a higher rung replaces a lower one as evidence
arrives, and the Inspector shows the rung beside "Simulated":

1. **`recorded`** — replay of the object's own filmed motion (synchronized clips, LIVING_WORLD §8).
2. **`fitted-real`** — parameters fitted to real footage of this object, with an anemometer.
3. **`fitted-generated`** — parameters fitted to video-model clips of this object (Teacher A).
4. **`allometric`** — parameters from tree size and branch lengths alone (this phase).

The legacy nine-sine model claims no rung (`motionEvidence: null`) and remains only as the
fallback for a rig without a sidecar and as the control arm of the blind comparison.

### 3. Video and world models are offline teachers only

- **Teacher A — motion statistics (weeks; next).** Render 4–6 still views near real cameras,
  generate swaying clips with image-to-video models under several seeds and wind prompts, reject
  clips whose first frame disagrees with our render, track rig nodes, and fit each branch's
  frequency, gain and spatial coherence **from spectra, not trajectories** (each seed is a wind
  realisation). The trunk stays allometric (a 5 s clip resolves only 0.2 Hz) and ζ stays from the
  literature (not recoverable monocularly). Output: the same sidecar, tagged `fitted-generated`.
  Models: **Wan 2.2 TI2V-5B** (Apache-2.0) or **Cosmos-Predict2.5** (NVIDIA Open Model License);
  tracking with **TAPIR** (Apache-2.0). Pass: fitted parameters beat a random-parameter control
  on held-out generated clips, and the blind test prefers them over phase 1.
- **Teacher B — unseen-view completion (deferred).** Repaint only low-support regions (support
  < 3 views) of rendered real camera poses and distil them into a **separate tileset tagged
  "generated"**, with measured splats checksum-frozen. Experiment first (leave-one-tier-out on the
  Minnetonka tree); it ships only if low-support pixels beat both null baselines, measured pixels
  are no worse (ΔPSNR ≥ −0.1 dB, ΔLPIPS ≤ +0.005), measured splats are bit-identical, and a planted
  canary neither moves nor vanishes. One model-agnostic interface, so the model is a config choice
  within the licence rule below.

### 4. What we will not build

| idea                                                                   | why not                                                                    |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| MPM physics on splats at runtime (PhysGaussian, PhysDreamer, PhysFlow) | seconds per frame on trees; a "plastic" global response                    |
| video or world models generating frames at runtime                     | not measured, not consistent, not anchored; breaks canonical-never-written |
| generic 4D-from-video for trees (SV4D, 4DGen, DG4D)                    | fails on complex trees in the only comparative study (DynamicTree)         |
| post-render enhancers on measured tiles                                | lowers fidelity (HAD, Table 1) and repaints what was measured              |

### 5. Licence rule

**Only models whose code _and_ weights permit commercial use may ship** — in the product, or in
any pipeline stage whose output ships (a sidecar fitted by Teacher A, a generated layer from
Teacher B). Checked per model before it enters a pipeline, and recorded in its provenance.

| model                           | role                    | licence                                    | may ship?                                    |
| ------------------------------- | ----------------------- | ------------------------------------------ | -------------------------------------------- |
| Wan 2.2 (TI2V-5B)               | Teacher A video         | Apache-2.0                                 | yes                                          |
| Cosmos (Predict2.5)             | Teacher A alternative   | NVIDIA Open Model License (commercial use) | yes                                          |
| NVIDIA Fixer                    | artifact cleanup        | Apache code, NVIDIA Open Model weights     | yes                                          |
| TAPIR                           | point tracking          | Apache-2.0                                 | yes                                          |
| VGGT                            | poses/geometry          | non-commercial default weights             | **only via the gated commercial checkpoint** |
| ArtiFixer                       | Teacher B reference     | weights non-commercial                     | **no**                                       |
| Difix3D+, Stable Virtual Camera | view repair / synthesis | non-commercial                             | **no**                                       |
| HunyuanVideo, HunyuanWorld      | video / world           | excludes EU, UK, South Korea               | **no**                                       |

A non-commercial model may still be named as a published quality reference (ArtiFixer's numbers
in the Teacher B design); running it is a separate licence question, answered before anyone does.

## Consequences

- **Measured on the synthetic tree** (`packages/world/src/living.test.ts`, the phase-1 pass
  criteria, all passing): every one of 107 branches has its local-bend spectral peak within ±10 %
  of `2.55·L^-0.59` (mean −2.3 %, worst −5.8 %); the whole-tree peak sits within 1 % of `2.4/√H`
  on a 6.5 m and a 15.9 m tree; RMS tip deflection scales as `U^2.03` over 2–10 m/s; the largest
  autocorrelation from 10 s to 1 h is at most 0.05 (tip, trunk, fastest branch) and 0.075 for leaf flutter
  at a point; flutter correlation is 0.96 at half a leaf and −0.06 at four; calm is the identity by
  value and restores the measured bytes; the same `t` gives bit-identical output.
- **Not settled by any test:** whether a person prefers it. The phase-1 criterion is a blind
  two-choice test against the nine-sine wind and a looped clip; `apps/web/e2e/livingCompare.spec.ts`
  renders anonymous A/B clips for it (`LIVING_COMPARE=1`), and the comparison has not been run with
  viewers.
- **Cost.** CPU, synthetic tree (12,000 splats, 214 nodes): ~2.6 ms/frame for transforms, advected
  flutter and positions, against ~2.0 ms for the legacy model (2.6–3.8 and 2.0–3.6 across runs on a shared, loaded 4-core machine); 18–40 ms at 144,000 splats, most of it
  per-splat flutter (three bilinear lookups a splat). A real capture needs the GPU deformer (M5):
  the textures are plain `Float32Array`s (1024², 4 MB each: one per damping level plus one for
  flutter), and the per-node state is 214 quaternions a frame. Textures take ~0.5 s each to build,
  once per session, at attach.
- **The wind control now means something physical.** `strength` is read as dynamic pressure:
  `U = 20·√strength` m/s, so deflection is linear in the control and the default 0.1 is 6.3 m/s.
  The amplitude at a given speed is still an estimate (Teacher A's job); only the scaling is cited.
- **The legacy model stays** as the fallback and the comparison's control. It is not deleted until
  the blind test has been run.
- **For the Minnetonka rig** the extractor's output is enough: nothing reads a radius. What it
  needs is a measured height (`motion_params.py --tree-height`; isolation trims the base, 8.5 % on
  the synthetic tree) and a leaf size (estimated from foliage splats, or `--leaf-size`).

## Addendum (2026-09-28): limbs, not joints; a band, not a formula; skinned splats

The first real tree (Minnetonka, 6.0 m, a 200-joint extracted rig) looked like it was
"vibrating very quickly". Measured, the cause was the parameters and the binding, not the model:

- **The rig's joints were read as branches.** The 35° continuation rule could not survive an
  extracted skeleton that zigzags (median 56° turn per joint), so 200 joints became 150
  oscillators, 121 of them one joint 0.2–0.7 m long, at 2–10 Hz, each with the full 0.05 rad and
  compounding down 22-joint chains: tip RMS 16 cm at the default wind, 63 % of tip speed above
  4 Hz, displacement spectral centroid 2.4 Hz.
- **Every splat followed one joint rigidly**, so neighbouring splats on different joints differed
  by 9 mm at the median and 186 mm at p99 (pairs < 1.7 cm apart): the crown sheared at every
  Voronoi boundary.

What changed, every rule general (read from the rig's geometry, never fitted to a tree) and
tagged in the sidecar's provenance:

| Rule                                                                                                                                              | Basis                                                                                                                                                                                                                                                                                                   | Status                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| At each joint the child carrying the most tips continues the axis; no angle limit                                                                 | da Vinci's rule                                                                                                                                                                                                                                                                                         | estimate                        |
| A limb's length `L` is its **span**: the farthest its subtree reaches from its attachment                                                         | beam scaling `f ~ D/L²`, `D ~ L^1.37–1.38` holds for whole branches (Rodriguez, de Langre & Moulia 2008)                                                                                                                                                                                                | cited                           |
| `f = 2.55·L^-0.59`, metres                                                                                                                        | Coder 2000 via Habel 2009 eq. 17; unit not in Habel, primary (UGA FOR00-24) not retrievable; only metres agrees with whole-tree data (6 m: 0.89 Hz vs 0.98 Hz from `2.4/√H`; feet: 0.44 Hz)                                                                                                             | cited, unit inferred            |
| Modes ring in `[f0, 3·f0]`: a limb eq. 17 puts above `3·f0` is a **twig** (rides the limb it hangs from, no bend of its own, leaf flutter only)   | Rodriguez 2008: a 7.9 m walnut's first 25 modes in 1.4–2.6 Hz; branch modes 2.5–3 Hz over 1–1.5 Hz fundamentals                                                                                                                                                                                         | estimate from cited data        |
| At most third-order limbs get a mode                                                                                                              | SpeedTree 1–2 branch levels, Pivot Painter 2 ≤ 4; many-part rigs are not identifiable from video (Chen & Lou 2026)                                                                                                                                                                                      | design choice                   |
| A limb's bend `∝ (f0/f)^0.305`, i.e. tip deflection `∝ 1/f²`                                                                                      | sub-resonant response `(F/m)/(2πf)²` (Habel eq. 15) with `L` from eq. 17, at equal drag per unit mass                                                                                                                                                                                                   | estimate                        |
| Leaf flutter 6 mm at 10 m/s (was 12), leaf size ≥ 5 cm                                                                                            | a splat is a piece of a leaf: 2 cm splats put the advected field at 9–22 Hz; now 4–9 Hz. Tadrist et al. 2018: flutter dominates only at low wind                                                                                                                                                        | estimate                        |
| Joint `i` hinges at its **parent's** rest position                                                                                                | a limb's first segment must bend; a one-joint limb must move                                                                                                                                                                                                                                            | design choice                   |
| Every splat blends its 4 nearest joints, modified Shepard weights (radius at the 5th), linear blend skinning in displacement form, 10-bit weights | Franke & Nielson 1980 (continuous weights); LBS over DQS because no joint twists about its limb, bends are hundredths of a radian (LBS shrink `≈ Δθ²/8`, sub-mm), splat covariances are not rotated on either path, and LBS is linear in the per-node texels (Kavan et al. 2008: 33 vs 42 instructions) | cited method, parameters chosen |

Measured on the Minnetonka rig at the default wind (6.3 m/s; `living.test.ts`, before → after):
24 modes at 0.98–2.91 Hz (150 at 0.98–10.05 Hz); tip RMS median 16.4 → 3.7 cm; share of tip speed
above 4 Hz 0.63 → 0.10; displacement spectral centroid 2.44 → 0.99 Hz; limb tip deflection
`∝ f^-1.67` across its 23 limbs; seam p99 186 → 2.8 mm (and a skinned seam shrinks with the probe
spacing, a rigid one does not). The synthetic tree: 47 modes at 0.94–2.82 Hz (was 107 at
1.67–4.97 Hz). The sidecar format is unchanged (v1); old sidecars load and move under the new
hinge and skinning.

Costs: binding is ~1.2× the nearest-node search (400,000 splats: ~0.4 s against ~0.3 s here;
a 1,500-gaussian LOD tile ~1.2 ms warm, 2.6 ms cold). The vertex shader fetches `1 + 4k` texels
for `k` weighted joints (3.4 on average on the Minnetonka tree, so ~15) where the rigid binding
fetched 5, plus the unchanged flutter lookups. The CPU fallback's per-splat blend is ~5× the rigid
transform (12,000 splats: 0.8 against 0.15 ms).

Still for a person on real hardware: whether 3–4 cm of tip sway at the default wind reads as a
tree in a moderate breeze (the magnitudes `0.02`/`0.05 rad` at 10 m/s remain estimates), whether
4–9 Hz, 2–3 mm leaf flutter reads as leaves or as noise, and the GPU cost of ~15 fetches per
vertex at a million splats.
