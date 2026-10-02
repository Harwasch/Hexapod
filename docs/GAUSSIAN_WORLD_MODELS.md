# Gaussian-native world models: how they work, how they scale

Research note, written 2026-09-30 on branch `living-models`. It follows
[WORLD_MODEL_ROUTES.md](WORLD_MODEL_ROUTES.md). Claims were checked against the arXiv HTML or
code named in each row; rows marked **(mem)** were not re-checked and are **UNVERIFIED**.

## 1. Four encoding patterns

Every Gaussian-native world model does one of four things with Gaussians. None of them feeds a
transformer millions of raw Gaussians.

| Pattern                         | Input → latent                                                                                                 | Output                                                                   | Examples                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| **i · pixels → Gaussians**      | images or video → feed-forward per-pixel Gaussians; the latent is a video latent                               | Gaussians **regenerated** every frame                                    | L4GM (mem), Lyra, GaussianWorld's front end             |
| **ii · set encoder**            | a farthest-point-sampled subset of Gaussians (GWM: 2,048) → cross-attention VAE → a fixed set (GWM: 512 × D)   | the VAE decodes a whole new set                                          | GWM, Can3Tok (40k → 64×64×4)                            |
| **iii · sparse voxel / anchor** | a sparse grid, a feature per voxel or anchor (GaussianGPT: 2.5 cm → 20 cm latent voxels, 4,096-code quantiser) | K Gaussians per voxel as offsets from its centre                         | TRELLIS (mem), GaussianGPT, MRO-GWM anchors, InfiniCube |
| **iv · object-centric**         | canonical Gaussians per object stay fixed; the state is one SE(3) pose or 1–5 part twists per object           | the **same** Gaussians rigidly transformed or skinned; never regenerated | MRO-GWM, 4DGS-WAM, PhysTwin (LBS), our rig              |

The dynamics model on top is a DiT (GWM), a point transformer (MRO-GWM PTv2, 3DGSim PTv3), a
geometric GNN (4DGS-WAM), or a particle-grid velocity field (PGND). The losses are direct 3D
losses (pose, flow, Chamfer) plus a render loss, often only on what changed.

| Model                                                     | Primitives it steps                            | Scale               |
| --------------------------------------------------------- | ---------------------------------------------- | ------------------- |
| [GWM](https://arxiv.org/html/2508.17600)                  | 2,048 → 512 latents                            | tabletop            |
| [MRO-GWM](https://arxiv.org/html/2606.01950)              | object anchors, 1 cm                           | tabletop            |
| [GaussianWorld](https://github.com/zuosc19/GaussianWorld) | 25,600                                         | driving, ~50 m      |
| [GEM](https://arxiv.org/html/2605.17682)                  | 25,600 (4D, with velocity)                     | 80 × 80 m occupancy |
| [4DGS-WAM](https://arxiv.org/html/2608.25956)             | 1–5 twists per actor; static background reused | driving             |
| [GaussianGPT](https://arxiv.org/html/2603.26661)          | 16k-token context per chunk                    | rooms, outpainting  |
| a Hexapod site                                            | **10⁶–10⁷ Gaussians**                          | 20–500 m, hours     |

## 2. Compression: two strands that don't meet yet

- **Storage codecs** make files smaller while rendering them faithfully:
  - HAC++: over 100× vs 3DGS ([2501.12255](https://arxiv.org/abs/2501.12255));
  - LightGaussian: ~15× (mem);
  - SPZ: ~10× vs PLY (mem).

  None of these is a latent a dynamics model can learn on.

- **Generative latents** reduce linear resolution by 8–16×:
  - TRELLIS.2: a 1024³ object in ~9.6k tokens;
  - GaussianGPT: 2.5 → 20 cm;
  - InfiniCube: ~300 × 400 m by outpainting sparse voxels.

  They are capped at roughly 10⁴–10⁵ tokens per chunk.

- **LOD hierarchies** (Hierarchical 3DGS, Octree-GS, CityGaussian) serve rendering only. **No
  world model runs on an LOD hierarchy.**

**The compression that makes dynamics tractable is structural, not a codec.** Every model that
reaches outdoor scale does three things:

1. freezes the static background;
2. puts a latent only where something happens;
3. collapses each movable object into a few pose or part parameters.

## 3. Token budget for a site

This is an estimate for a 10⁷-Gaussian park like Fort Clatsop.

| Part                             | Share / count                            | Tokens                                            | Already in Hexapod                                                  |
| -------------------------------- | ---------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| Enduring base, frozen            | most Gaussians (41 % static in the yard) | **0** (context and loss mask only)                | yes: measured SPZ tiles, pinned bit-exact                           |
| Plants (pattern iv)              | ~200 plants, ~8,000 joints               | ~8k joint tokens (6 numbers each)                 | yes: `scene_plants.py` instances, forest rig, `plants.json` binding |
| Agents                           | robots, vehicles                         | a few, one pose each                              | mission control `Machine`                                           |
| Region of interest (pattern iii) | e.g. a 50 × 50 m mowing zone             | 62,500 at 20 cm; ~15,600 at 40 cm, or 20 m chunks | no: the new part                                                    |
| Fields                           | wind, water level, grass height per zone | tens                                              | wind yes; the rest new                                              |

About 10⁴–10⁵ tokens per step is the budget current models handle.

**Hexapod already has the hard half of a site tokenizer.** Plant instances, rigs and per-tile
bindings are exactly pattern iv's object slots, and the frozen base is how 4DGS-WAM and InfiniCube
scale. What is missing is a region-of-interest voxel latent (pattern iii) with a decoder that
writes residual Gaussians back into only the affected tiles.

## 4. Through the four views

| View       | Pattern that fits                               | What it would do                                                                                                                                                                    | Open problem                                                                                                               |
| ---------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Static** | iii, outpainting conditioned on measured voxels | fill low-support regions **natively in 3D** (GaussianGPT/InfiniCube style), with the measured neighbours as context, instead of video → views → distil (route A)                    | licences UNVERIFIED; trained on synthetic or indoor data; seams and lighting against the frozen base                       |
| **Living** | iv                                              | our runtime already _is_ the pattern-iv output (canonical Gaussians + per-joint transforms). A learned model would output the sidecar or the joint transforms                       | nothing needed now; a model predicts parameters (after a storm, a season)                                                  |
| **Live**   | GaussianWorld's three-part stream               | per update: re-align, move the dynamic objects, complete the newly seen areas. Only region-of-interest tiles are re-encoded from robot frames; objects are corrected from telemetry | outdoor photogrammetry is noisy; no benchmark above ~100 m                                                                 |
| **Dream**  | ii/iv tokens + iii for the region of interest   | a DiT or point transformer over object, agent and region-of-interest tokens, conditioned on actions (GWM/MRO-GWM), decoding pose deltas and residual Gaussians into affected tiles  | deciding the region of interest automatically; horizons of hours; training data (repeated captures and simulated rollouts) |

## 5. Open problems

- **Georeferenced scale:** the recipes assume normalised coordinates, which is Can3Tok's core
  problem. Our ENU tile frames help.
- **No LOD-aware tokenizer.**
- **Seams:** SH and lighting consistency when written-back tiles meet the frozen base.
- **Captured data is noisy:** outdoor captures differ from the synthetic or robot-scale data the
  models are trained on.
- **Compression and learnability are separate:** an entropy-coded anchor is not a
  dynamics-ready latent.
