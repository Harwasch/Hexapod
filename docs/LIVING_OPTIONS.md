# Best options per view

Recommendation, written 2026-09-30 on branch `living-models`. It condenses
[LIVING_MODELS.md](LIVING_MODELS.md), [LIVING_ENGINE.md](LIVING_ENGINE.md),
[WORLD_MODEL_ROUTES.md](WORLD_MODEL_ROUTES.md),
[GAUSSIAN_WORLD_MODELS.md](GAUSSIAN_WORLD_MODELS.md),
[WORLD_MODEL_CATALOG.md](WORLD_MODEL_CATALOG.md) and [WORLD_LABS.md](WORLD_LABS.md).

## The shape of the answer

- **Lock the state representation now.** Every good option shares it:
  - an enduring measured base, frozen;
  - object slots: plants on rigs, agents (these exist);
  - a latent only for the region an action affects;
  - fields (wind, water, grass height);
  - evidence labels on everything.
- **Keep the engine that steps that state open.** A bake-off decides it.
- **No world model runs at view time.** They run offline or server-side, and write into
  labelled layers.

## Per view

| View       | Use now                                                                                                                                                                                 | Build next                                                                                                                                            | Challenger (bake-off)                                                                                                                                                        | Watch                                                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Static** | our capture → gsplat → SPZ 3D Tiles; fill **on our GPUs**: render low-support views → **Cosmos-Predict2.5 + NVIDIA Fixer** → distil into a separate `inferred` tileset, measured frozen | the fill/enhance policy and its cutoffs from S1, S2, S5, S6                                                                                           | **World Labs Marble** (fast, ~$1 per world; but the data leaves us, no splat input, AMD roadmap)                                                                             | native 3D outpainting (GaussianGPT, InfiniCube); video + 3D memory (Spatia, GEN3C); **Atlas**; **ABot-Earth** for context around a scan |
| **Living** | the compressed runtime we have: skeleton rigs + modal motion + Eurocode wind, on the GPU                                                                                                | class priors → **Teacher A** (Wan 2.2 / Cosmos clips → fitted parameters); grass-field and water-flow families                                        | a learned deformation field (AniGS / PhysMani style) for things with no family                                                                                               | Atlas motion reconstruction; 4D capture (Gracia) for recorded replay                                                                    |
| **Live**   | —                                                                                                                                                                                       | **state estimation** (a Kalman filter per value) driving the same knobs; robot and drone frames → change detection → re-train only the affected tiles | —                                                                                                                                                                            | Embodied Gaussians (vision-corrected simulation), Spatia (SLAM-updated map), Niantic VPS                                                |
| **Dream**  | —                                                                                                                                                                                       | **v0**: Claude turns a prompt into a typed scenario → our models (mission, robot, grass, wind, water) → time slider; optional Cosmos-Transfer clip    | **v1**: one learned model over the structured state, starting from **PointWorld** (NVIDIA; code and weights, licence to check), trained on v0 rollouts and repeated captures | Applied Intuition Neural Sim (the commercial precedent); Atlas; action-conditioned 4D (PerpetualWonder)                                 |

## Avoid

- Genie 3 (no API).
- Lyra 2.0 (research-only weights).
- Difix3D+, Stable Virtual Camera (non-commercial).
- HunyuanWorld / NeoVerse (regional licence).
- Video generated at view time.
- Treating any generated output as measured.

## Order

1. **Static bake-off:** S6 (spool) and S1 (Minnetonka): Cosmos + Fixer against Marble. Try
   Fixer, Cosmos and NuRec on the Brev A100 at the same time.
2. **Engine core:** the state contract, a synthetic telemetry log, tests V1 and V2 (live), and
   L1 (parameter recovery for living).
3. **Dream v0:** D1 (mow) and D2 (storm) on the hand-built models, plus D4 (prompt →
   scenario).
4. **Dream v1 probe:** reproduce the PointWorld pattern on the synthetic yard using v0
   rollouts. Can one learned model match v0 inside its envelope?
5. **Decide the engine** from the bake-off, then do Teacher A and V3 (change detection).

Licence checks before any of these is used: PointWorld weights, ABot-Earth, SpAItial Echo-2,
GEN3C, the NuRec containers, and legal review of World Labs §3.3d and §3.6.
