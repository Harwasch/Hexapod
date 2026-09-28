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
