# World model routes: what steps the world forward

Open decision, written 2026-09-30 on branch `living-models`. It follows
[LIVING_ENGINE.md](LIVING_ENGINE.md), whose world-state representation and tests are proposed
for adoption, and whose dynamics engine is **not** locked. Claims were read on that date;
anything not confirmed is marked **UNVERIFIED**.

## The question

Should dream mode (and eventually live render) run on many hand-built transition models, or on
one overarching world model that "knows all"? And is there a world model that works natively in
Gaussian-splat space, so that it takes a Gaussian scene plus actions and predicts the next
Gaussian scene?

## What exists (Gaussian- and 3D-native)

| Work                                                                | In → out                                                            | Actions?          | Scale                   | Commercial use?                                                |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------- | ----------------------- | -------------------------------------------------------------- |
| [GWM](https://github.com/Gaussian-World-Model/gaussianwm) (ICCV 25) | RGB → Gaussians → 3D-VAE latent → DiT → future Gaussians            | robot actions     | tabletop                | code MIT (WIP); depends on MASt3R, non-commercial (UNVERIFIED) |
| MRO-GWM ([2606.01950](https://arxiv.org/abs/2606.01950))            | per-object canonical Gaussians + actions → future poses             | yes               | tabletop, synthetic     | no code                                                        |
| ManiGaussian / ++                                                   | multi-view RGB-D → dynamic Gaussian field                           | yes               | tabletop                | MIT                                                            |
| [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) (ICCV 25)      | RGB-D video → spring-mass + Gaussians                               | physics           | one deformable object   | MIT                                                            |
| [PGND](https://github.com/kywind/pgnd) (RSS 25)                     | RGB-D video → learned particle-grid dynamics, 3DGS render           | yes               | object                  | MIT                                                            |
| GausSim, 3DGSim                                                     | Gaussians → learned simulator (GNN/transformer)                     | forces            | object                  | UNVERIFIED                                                     |
| [GaussianWorld](https://github.com/zuosc19/GaussianWorld) (CVPR 25) | camera → semantic Gaussians, evolved in time → occupancy            | ego motion        | driving, ~50 m          | no licence file; nuScenes data                                 |
| GEM ([2605.17682](https://arxiv.org/abs/2605.17682)), 4DGS-WAM      | 4D Gaussians, static and dynamic parts separate                     | ego and actors    | driving                 | no code                                                        |
| [L4GM](https://github.com/nv-tlabs/L4GM-official), Lyra 1.0 / 2.0   | video → feed-forward 4D or static 3DGS                              | no (reconstructs) | object / scene          | code Apache; Lyra 2.0 weights research-only                    |
| [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B)          | image → structured sparse latent → asset                            | static            | object                  | MIT; nvdiffrast dependency non-commercial                      |
| WorldGrow, GS-Voxel, GaussianGPT                                    | structured 3D latents → large static scenes                         | static            | up to ~1,800 m²; aerial | UNVERIFIED                                                     |
| World Labs Marble / Atlas (Sep 2026)                                | text/image/video → static splats; Atlas adds depth and point clouds | no dynamics shown | scene                   | API (Marble); early access (Atlas)                             |

**Answer: not yet at our scale.** Action-conditioned Gaussian world models exist, but only
for tabletops, single objects and short driving horizons. Every one keeps a static
background separate from dynamic parts. The site-scale 3D-native models generate static scenes.

The gaps are:

- **scale**: 10⁶–10⁷ Gaussians against 10⁴–10⁵;
- **domain**: no vegetation, water or weather processes;
- **horizon**: seconds against hours to seasons;
- **data**: no training pairs for mowing or storms;
- **licences**: permissive code with non-commercial weights or dependencies.

## The routes

Every route shares the same world-state representation (LIVING_ENGINE §2): an enduring
measured base, per-object Gaussian assets on skeletons, low-dimensional state per object,
agents, and evidence labels. What differs is what steps it forward.

| Route                                               | Steps the world with                                                                                                                                                   | General?                       | Grounded?                                       | Cost per scenario    | Available                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------- | -------------------- | ------------------------------------------------ |
| **A · Video world model → Gaussians**               | Cosmos-Predict2.5 renders the scenario from several camera paths; Lyra-style lifting or gsplat distillation fits Gaussians; measured Gaussians frozen                  | high (anything it can imagine) | weak: multi-view drift; quantities not readable | GPU-hours            | now (commercial parts)                           |
| **B · Hand-built transition models**                | physics and rules per domain (LIVING_ENGINE §2)                                                                                                                        | only what is modelled          | strong: deterministic, quantities exact         | ~free, real time     | now                                              |
| **C · One learned model over the structured state** | one transformer or GNN over objects and their low-dimensional state, conditioned on actions and forcing (MRO-GWM / GWM pattern); decoders to Gaussians per asset class | grows with data                | strong where trained; says where it isn't       | cheap after training | build: ~2–4 engineers × 6–12 months (UNVERIFIED) |
| **D · End-to-end Gaussian-native world model**      | raw Gaussians in → raw Gaussians out                                                                                                                                   | highest                        | unknown                                         | unknown              | research, years                                  |

**How the routes relate.** Route C is the "overarching world model" in a form that can stay
grounded. It is one learned model, and it works on the compressed representation we already
chose for living mode (objects, skeletons, motion state), so it is not a special case.
Routes A and B are how C gets built:

- B generates training rollouts and acts as the test oracle wherever physics is known;
- A supplies appearance and motion priors;
- repeated real captures of the same sites supply the ground-truth transitions.

Each route sits behind the same `step()` interface, so moving from B to C to D is a swap, not a
rewrite.

## Bake-off: decide by measurement

Run the same scenarios through each available route and score them on the same sheet.

| Scenario                                  | Why it discriminates                                            |
| ----------------------------------------- | --------------------------------------------------------------- |
| D1 · mow zone B at 3 cm (synthetic yard)  | a precise, checkable quantity (swept area) and a sharp boundary |
| D2 · storm, 20 m/s, 1 h (Minnetonka tree) | plausible motion over a long horizon; the envelope matters      |
| S6 · spool table, top filled              | the static fill case of A, on a small real capture              |

| Score                  | Measure                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| Grounding              | measured Gaussians outside the affected region bit-identical; held-out real views not degraded |
| Correctness            | quantities against truth (mowed m² ±2 %; deflection against the U² law)                        |
| Multi-view consistency | reprojection error between two rendered paths of the same result                               |
| Plausibility           | blind preference from viewers                                                                  |
| Cost and latency       | GPU-seconds and wall time per scenario-hour                                                    |

Routes A and B can be scored now. Route C enters the bake-off once a first learned transition
exists. The cheapest first version is a transformer over the synthetic yard's objects, trained on
route-B rollouts; it tests whether one learned model can match B inside B's envelope, before any
real data is spent on it.

## Living mode

Living mode is agreed: it keeps its compressed form (skeletons plus motion properties),
because running a world model forever to invent frames nobody inspects is wasteful. A route-C
model would _predict_ that state, for example a new wind response after a storm or growth over
a season. It would not replace the runtime.
