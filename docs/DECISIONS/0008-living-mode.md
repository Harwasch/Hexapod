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

## Addendum (2026-09-29): the wind has a spectrum, and it moves with the wind speed

After the limb fix the Minnetonka tree read "a lot better, but still somewhat animatronic or
jittery — especially at low wind, where I would expect more of a gentle swaying". Measured on the
code before this change, the cause was the forcing, not the tree:

- **The response spectrum's shape did not depend on the wind.** Each limb read a motion texture
  whose spectrum was its oscillator's response to _flat_ forcing, placed at its own frequency; the
  wind only scaled it (`(1 + f/U)^(−5/6)` at the limb's frequency, then `U²`). Every limb rang at
  its resonance at every speed: over the Minnetonka rig's 23 limbs, the median own-tip spectral
  centroid was 0.64·f_n at 2 m/s and 0.69·f_n at 16 m/s, and 40–41 % of the variance lay within
  `[0.8, 1.25]·f_n` at every speed. Real response is `|H(f)|²·S_wind(f; U)`, and the wind's
  energy sits at `f ≈ 0.15·U/L` with `L` ≈ 35 m near a 6 m tree — 0.008 Hz at 2 m/s — so at low
  wind a limb follows the gusts quasi-statically and barely rings.
- **The gusts were scripted**: a SpeedTree-style `sin²` bump, +35 % every ~20 s, the same shape
  every time, on top of a steady lean.
- **Neighbouring limbs moved independently** (one texture trajectory each), where a gust front
  crosses a crown at `U` and moves neighbours together at low frequency.

What changed (`packages/world/src/turbulence.ts`, `living.ts`), every rule general and tagged in
the sidecar's provenance:

| Rule                                                                                                                                                                                                                                                       | Basis                                                                                                                                                                                                                                          | Status                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Turbulence intensity `I = 1/ln(z/z0)` and length scale `L = 300·(z/200)^(0.67 + 0.05 ln z0)` at the tree's height `z = H`, held at `z_min`                                                                                                                 | EN 1991-1-4:2005+A1:2010 eqs. 4.7 (`k_I = c0 = 1`) and B.1; `z_s = h` for a structure outside its Fig. 6.1 (§6.3.1)                                                                                                                            | cited (read at the source)               |
| Terrain category III, `z0 = 0.3 m`, `z_min = 5 m`                                                                                                                                                                                                          | EN Table 4.1: "villages, suburban terrain, permanent forest"                                                                                                                                                                                   | estimate (the category is chosen)        |
| Wind spectrum `S_L(f_L) = 6.8·f_L/(1 + 10.2·f_L)^(5/3)`, `f_L = f·L/U`                                                                                                                                                                                     | EN eq. B.2                                                                                                                                                                                                                                     | cited                                    |
| Background `B² = 1/(1 + 0.9·((b + h)/L)^0.63)`, `b`, `h` the extent of what the oscillator carries, from the rig                                                                                                                                           | EN eq. B.3                                                                                                                                                                                                                                     | cited                                    |
| Resonance `R² = π²/(2δ)·S_L(f_n)·R_h(η_h)·R_b(η_b)`, `R(η) = 1/η − (1 − e^(−2η))/(2η²)`, `η = 4.6·(h or b)·f_L/L`                                                                                                                                          | EN eqs. B.6–B.8                                                                                                                                                                                                                                | cited                                    |
| Aerodynamic damping `ζ_a = 2π·f_n·x_s/U`, `x_s` the limb's static tip deflection at `U` (its gains and the rig)                                                                                                                                            | EN eq. F.18 with `c_f·ρ·b/m_e = 2·x_s·(2πn)²/U²`; James & Haritos 2010 saw branch damping rise with sway amplitude and put it down to drag                                                                                                     | cited formula, estimated inputs          |
| Sway RMS `2I·√(B²φ + R²)` of the mean lean along the wind, `0.75·I·(…)` across                                                                                                                                                                             | EN eq. 6.3 (`2I` is the linearised drag); `σ_v/σ_u = 0.75` kept from the previous 0.6 : 0.8 (IEC 61400-1's 0.8 could not be read at the source)                                                                                                | cited / estimate                         |
| The background is a **frozen field** of 320 random Fourier modes with the EN spectrum along the wind, carried at `U`; each oscillator reads it at the centroid of what it carries, through a `ζ = 1/√2` low-pass at `f_n` (`φ` above: the share it passes) | Taylor's frozen turbulence, as Habel 2009 §7.2 advects the leaf field; along-wind wavenumbers stratified `∝ √(k1·F(k1))` in `log k1`, the other two from the isotropic field's conditional; the low-pass is the flattest second-order response | method cited, sampling a design choice   |
| Fluctuations slower than 10 minutes are changes of the mean, high-passed out                                                                                                                                                                               | EN B.2(3): the mean wind is a 600 s average                                                                                                                                                                                                    | cited                                    |
| The resonant channel is the existing flat-forced texture at `f_n`, scaled by `R`                                                                                                                                                                           | its variance is the white-noise integral `R²` stands for                                                                                                                                                                                       | method                                   |
| Limb damping 0.045 → 0.106 with `√(tip share)` (was 0.086 → 0.15, unverified); the tree's 0.086 summer, 0.039 winter unchanged                                                                                                                             | James & Haritos 2010 pluck tests: single branches 3.5–4.5 % at small amplitude, the tree with its branches 10.6 % (sub-branches as tuned mass dampers)                                                                                         | cited endpoints, estimated interpolation |
| Scripted gust envelope off (`gust.strength: 0` in new sidecars); leaf flutter amplitude follows its branch's own gusts, `× (1 + 2I·B·u)`                                                                                                                   | the spectrum already holds the gusts; the drag on leaves follows the same `(U + u)²`                                                                                                                                                           | design choice                            |

Measured on the Minnetonka rig (`living.test.ts`; its 23 limbs' own-tip deflection, 600 s at
10 Hz, medians):

| wind   | centroid / f_n, before → after | variance within `[0.8, 1.25]·f_n` | variance below `f_n/2` |
| ------ | ------------------------------ | --------------------------------- | ---------------------- |
| 2 m/s  | 0.64 → 0.058                   | 0.40 → 0.008                      | 0.38 → 0.98            |
| 6 m/s  | 0.67 → 0.10                    | 0.41 → 0.024                      | 0.35 → 0.95            |
| 12 m/s | 0.68 → 0.18                    | 0.41 → 0.055                      | 0.34 → 0.88            |
| 16 m/s | 0.69 → 0.19                    | 0.41 → 0.060                      | 0.34 → 0.88            |

The model's own resonant share `R²/(B² + R²)` is 0.005, 0.021, 0.064, 0.098 and 0.125 at 2, 4,
8, 12 and 16 m/s. The resonance is always a peak at `f_n` (a lightly damped limb has one at any
wind); its prominence roughly doubles from 2 to 12 m/s. At 12 m/s every synthetic-tree branch's
velocity-spectrum peak is within 4.3 % of its model frequency, and the trunk's within 0.5 % of
`2.4/√H`. Two Minnetonka limbs 2.35 m apart along a 4 m/s wind move with a 0.573 s lag against a
predicted 0.581 s (`Δx/U` plus their filters' group delays), coherence 0.93 at 0.02–0.2 Hz and
0.48 at 0.8–1.6 Hz. The mean deflection goes as `U^1.97`, the fluctuation about it as `U^2.08`,
the extra because `R²` grows with the wind. No autocorrelation above 0.17 from 10 s to an hour
over three hours of motion.

Kept: calm is the identity by value; the output is a pure function of `(t, wind, seed, rig)` (the
field and a per-bearing phase table are memoised, invisibly); the displacement bound is still
proven (the background is soft-clipped at 3σ of the raw field); the GPU path is untouched — it
consumes per-node transforms and the flutter field, both still computed on the CPU. Transforms
cost 0.30 ms a frame on the Minnetonka rig against 0.22 ms before, and a whole frame 0.33 ms
either way, since its two halves now share one evaluation. The sidecar stays v1: it gains an
optional `wind.lengthScaleM`, and a sidecar without one takes `L` from EN eq. B.1 at its height;
older sidecars (sway ratios 0.8/0.6, gusts on, the old damping) load and move under the new
forcing with their own numbers until regenerated.

Not settled, and worth saying:

- **The resonant texture's bandwidth is the structural damping only.** `R²` includes the
  aerodynamic damping, but a limb's texture is built once per damping class, so its peak is
  narrower than `ζ_s + ζ_a` would make it at high wind.
- **Resonance that grows with the wind is what EN's procedure gives**, and field data are mixed:
  Jackson et al. 2021 find the tree spectrum's slope flat above 3–4 m/s and cite Schindler & Mohr
  2018, whose four Scots pines' oscillatory component _diminished_ with wind. The aerodynamic
  damping tempers the rise here; it does not reverse it.
- **The frozen field is a line spectrum.** 320 modes put 3–6 lines in each ±10 % band a limb
  sways in; they blend over a 30 s look, and a 20-minute spectrum resolves them.
- **The amplitudes are still the previous addendum's estimates** (0.02 / 0.05 rad at 10 m/s). At
  2 m/s a Minnetonka limb's own tip now moves ~1.4 mm RMS, slowly. Whether that reads as gentle
  swaying or as too little is for eyes.

## Addendum (2026-09-29): many plants in one tileset — trees, shrubs, snags, and everything else still

The next real capture is an outdoor splat of a park (Fort Clatsop National Historical Park, an
upload through `splat-ingest`): low shrubs, leafy trees, bare trunks without branches or leaves
("snags"), paths, lawn and buildings, all in one level-of-detail tileset. Every rule so far was
for one tree that is the whole tileset. What changed, none of it tuned to that park — every
threshold is derived from the capture or cited, and `tools/captures/scene_plants.py` writes each
one, with its source, into the `thresholds` of the `scene.json` beside the tiles.

**A scene step finds the plants**, from geometry and colour only (the scene-understanding plan's
"Path B"): a ground surface, a class per splat, plant instances.

| Rule                                                                                                                                                                                                                           | Basis                                                                                                          | Status                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Ground: the highest surface below every cell's lowest splat whose slope never exceeds S (a point is ground when nothing within d lies more than S·d below it); S = median + 3σ (MAD) of the capture's own adjacent-cell slopes | Vosselman 2000, slope-based filtering (IAPRS 33(B3)); 3σ clipping                                              | cited method, derived bound         |
| Ground cell: the plan cell holding 8 splats on average                                                                                                                                                                         | the pipeline's `splat_ground` `min_points`                                                                     | derived                             |
| Ground layer: median + 3σ of heights above the surface where it rests, floored at the point spacing                                                                                                                            | robust clipping                                                                                                | derived                             |
| Green: ExG = 2g − r − b on chromatic coordinates above the Otsu threshold of the capture's own histogram (VARI reported beside it)                                                                                             | Woebbecke et al. 1995; Otsu 1979; Gitelson et al. 2002                                                         | cited                               |
| Shrub 0.5–5 m, tree ≥ 5 m                                                                                                                                                                                                      | FAO FRA 2020 Terms and Definitions                                                                             | cited                               |
| Objects: splats above the ground layer, linked within 4.5× the **local** point spacing of either splat (12 neighbours); an object reaches 0.5 m with at least 8 splats                                                         | skeleton.py's link factor and cluster size, made local so a scene of many densities neither welds nor shatters | derived                             |
| Tree tops: local maxima within CW(H)/2, CW = 2.51503 + 0.00901 H² (mixed stands), at tree heights only; marker-controlled watershed of the canopy height                                                                       | Popescu & Wynne 2004, PE&RS 70(5)                                                                              | cited (search summary, not the PDF) |
| A plant stands on the ground: an object with no splat within 0.5 m of it joins the rooted object under its hull (narrower than itself), else the nearest one closer than its own size                                          | physical; guards the window's over-segmentation of wide crowns, and trunks a capture leaves sparse             | design choice                       |
| Leafy: a majority of the instance's splats green. Snag: tree height, not leafy, plan spread < CW(H)/2 (lost ≥ ¾ of the width a live crown of that height has). Anything else not leafy stays still                             | majority vote; Popescu & Wynne for the live crown                                                              | chosen rules on cited inputs        |
| A plant owns what stands under its foliage: the plan hull of its green splats. A wall a crown touches stays still                                                                                                              | —                                                                                                              | design choice                       |
| Rig: skeleton.py's banded skeleton when the instance has ≥ 464 splats per metre of height (12,000 ÷ 4 over 6.46 m, the thinnest cloud its recovery is tested on); a crown rig otherwise                                        | `tests/test_skeleton.py`, the quarter-density test                                                             | derived                             |

**Motion per class**, on the existing allometric rules (`motion_params.py`), plant by plant:

| Class      | Rig                                                                                                                 | Motion                                                                                                                                                                                                                                                                               | Status        |
| ---------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| leafy tree | skeleton (dense) or crown rig                                                                                       | the limb model above, unchanged                                                                                                                                                                                                                                                      | as above      |
| shrub      | crown rig: a stem through crown base and centroid to top, one two-joint limb per quarter of the crown (≤ 12 joints) | the same laws at its height: `f0 = 2.4/√H`, 2–3 Hz for 0.6–2 m (extrapolated below the 4.7 m of Jackson's trees; de Langre 2019 puts plant frequencies between about 1 and 10 Hz from wheat to trees); the same bend angles on a short lever, so a small swing; flutter at every tip | estimate      |
| snag       | trunk: the ground and each quarter of its height                                                                    | the trunk mode alone, its bend × `D/(D + CW(H)/2)` — its frontal area against a crowned tree's of the same height (deflection linear in drag, drag linear in frontal area; the crown over the upper half and equal drag coefficients are estimates) — and **no flutter**             | estimate      |
| static     | one anchor node                                                                                                     | identity, pinned by the deformer on both paths whatever the model says                                                                                                                                                                                                               | design choice |

**One rig, one sidecar, one binding per tileset.** The rig is a _forest rig_: node 0 a static
anchor, then each plant's joints as a contiguous run whose first node is that plant's root
(`rig.plants`: id, class, `[nodeStart, nodeEnd)`), every root an anchor. The motion sidecar
stays `hexapod.motion` **version 1**: per-node columns as before (branch indices into the whole
rig), plus an optional `plants` array — each plant's height and the EN 1991-1-4 turbulence
ratios and length scale at _its own_ height — while one frozen field, read at the tallest
plant's length scale, sways them all: one wind, gusts crossing the yard at the mean speed. Old
sidecars load unchanged. The **plant binding** (`plants.json`, `hexapod.plants` v1) is new: per
tile, keyed by the tile's checksum, run-length pairs of plant labels in the tile's gaussian
order — 0 static, `k` plant `k − 1`. It is computed by replaying the packer's own plan
(`splat_tiles.prepare`): a leaf's labels are its rows', and a merged parent takes a plant's
label only when **every** original merged into it is that plant's. This departs from the
multi-tile rule "bind from position at load" (LIVING_SURVEY.md): for one tree nothing else is in
the tileset, but in a scene which splat is a plant is a measurement that position cannot
recover, so it travels with the tiles and is proven per tile.

**Runtime.** A splat labelled static is bound with weight 1 to the anchor, which the deformer
pins (`pinStaticNodes`): the CPU path writes the engine's own bytes back for it, the GPU path's
node rows are exact zeros and the shader returns the fetched position — bit-exact, not merely
still. A plant's splat is skinned to its four nearest joints _of that plant_
(`skinSplatsToPlants`: exactly `skinSplatsToNodes` on the plant's own joints). A forest rig
without its binding is refused (`binding`), and so is a tile its binding does not list
(`checksum`). The upright check is not applied to a forest (a scene is not a standing tree); the
frame check stands.

**Measured** on the synthetic yard (`tools/captures/synthetic_yard.py`: three leafy trees of 7,
9.8 and 11.9 m — the last at a quarter of the others' density — five shrubs of 0.6–2 m, two snags
of 7 and 10 m, a building, lawn and a path; 43,337 splats in 15 tiles):

- per-class IoU over splats: tree 0.998, shrub 0.992, snag 0.984, grass/low 0.985, ground
  0.930, other-static 0.859; instances 3 of 3 trees, 5 of 5 shrubs, 2 of 2 snags; height error
  mean 0.04 m, max 0.16 m; stems within 0.07 m. At half the density trees and snags are exact,
  two shrubs 36 cm apart merge (four point spacings, inside the link) and shrub IoU is 0.88; on
  ground sloping 8 % and 5 % every instance is found.
- rigs: 406 joints (skeleton trees of 154 and 169, a crown tree and five crown shrubs of 12, two
  snags of 5); rig 63.5 KB, sidecar 24.5 KB, binding 2.4 KB for 43k gaussians.
- whole-plant frequencies: trees 0.70–0.90 Hz, shrubs 1.7–3.0 Hz, snags 0.76–0.91 Hz. Top
  deflection at 6.3 m/s, mean over ten minutes: trees 2.7–7.0 cm, shrubs 0.4–1.2 cm, snags
  0.8–1.1 cm; at 12 m/s trees 9.6–24 cm, shrubs 1.3–4.0 cm, snags 2.7–3.8 cm.
- static splats: 17,808 of the near view's 43,337; none moved, checked splat by splat on both
  paths, in unit tests and in the browser (`e2e/livingSurveyYard.spec.ts`).

**Cost.** Binding a tile: 0.75 µs a gaussian warm in Node (15 tiles, 47k gaussians, 36 ms), 1.7
µs cold in the browser (43k gaussians in 74 ms). Per frame the model is per joint and per
oscillator, never per plant: 406 joints 0.8 ms; 200 plants (the yard twenty times, forty of
them skeleton trees) 8,101 joints 11–15 ms; 203 shrubs and snags, 2,031 joints, 4.4 ms (Node, a
shared 4-core machine). Most of it is each oscillator reading the 320-mode frozen field. The GPU
path's per-frame upload is the node rows: 32 rows of 1,024 texels at 8,101 joints, 512 KB.

Not settled, and worth saying:

- **The per-frame cost of a large park.** Fifteen milliseconds of main thread for 200 plants is
  too much beside everything else a frame does. The next step is to evaluate only the plants whose
  joints the current snapshot's splats reference (a far view draws merged parents bound to few
  plants), or to move the model to a worker; neither is done.
- **Greenness is one threshold per capture.** A capture that is nearly all vegetation — dark
  conifers over bright lawn — could put Otsu's split between the two greens; the report carries
  Otsu's effectiveness and the VARI agreement so a reader can see it. A leafless deciduous tree is
  neither leafy nor narrow: it stays still (conservative, and wrong in winter).
- **The ground under a closed canopy seen only from above** is interpolated by the slope bound
  from wherever ground was seen; a bank steeper than the capture's own slope statistics reads as
  an object. Tall non-green poles pass the snag rule and sway about a centimetre.
- **Crowns wider than Popescu & Wynne's average** would over-segment without the rooting rule, and
  a canopy seen only from above has no stems for that rule to use: it keeps its tops as they are.
- **No real capture has been through it.** The workflow that runs it on an upload exists
  (`.github/workflows/living-plants.yml`); it has not been run.
